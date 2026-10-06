"""The values steps hand each other, and the audio rules that belong to them.

A leaf: numpy, torch and events only, so the result writer and the tasks both
sit above it.
"""

import logging

import numpy
import torch

from .events import emit_warning

logger = logging.getLogger("dw")


class AudioVideo:
    """A generated video together with the audio track generated alongside it.

    Pipelines like LTX-2 return audio next to their frames. Keeping the two paired lets
    the result mux them into one file instead of dropping the audio on the floor.
    """

    def __init__(self, frames, audio, sample_rate, fps=None, shots=None):
        """
        Args:
            frames: The video, as PIL images or an array of frames
            audio: Waveform for this video, shaped (channels, samples)
            sample_rate: Sample rate of the waveform, or None if the pipeline did not report one
            fps: Frame rate these frames are meant to play at, when something
                knows it - a joined video's own rate, or the rate of the file
                a task read. Carried for the same reason AudioTrack carries
                its sample rate: `result.fps` defaults to 8, and a step that
                joins 24 fps shots writing them at 8 is three times slow with
                its audio still the right length (#84). A declared
                `result.fps` still wins over this
            shots: Where each input landed, for a video a step joined from
                several - a list of shot records (dw/shots.py), or None for a
                video that is one shot. Carried into the step's manifest
                entry when the video is saved (#378)
        """
        self.frames = frames
        self.audio = audio
        self.sample_rate = sample_rate
        self.fps = fps
        self.shots = shots


class AudioTrack:
    """A generated waveform together with the rate it was generated at.

    A step that produces audio alone usually returns the waveform by itself, and
    the workflow declares the rate - which is fine where the rate is a property of
    the workflow (a slice of a file it named) rather than of the model. It is not
    fine for a generated track: every text-to-speech model has its own rate, and a
    declared 44100 against a 24 kHz model plays the speech fast without failing.

    Carrying the rate with the waveform is what lets a workflow say nothing about
    it. Everything downstream of audio already reads '.audio' and '.sample_rate'
    off whatever it is handed - slice_audio, fade_audio, pair_audio and the H3
    audio references all accept one of these - and a rate the workflow does declare
    still wins over the one carried here.
    """

    def __init__(self, audio, sample_rate, source_mean_dbfs=None):
        """
        Args:
            audio: The waveform, shaped (channels, samples)
            sample_rate: Sample rate the waveform was generated at
            source_mean_dbfs: The mean level of the material this track was
                taken from, when a task (slice_audio) measured one before
                cutting it down - lets a save skip the near-silent warning
                for a slice whose source was already this quiet (#309)
        """
        self.audio = audio
        self.sample_rate = sample_rate
        self.source_mean_dbfs = source_mean_dbfs


class Selected:
    """The winning candidate, plus the metadata that makes the choice
    replayable (position, score). Compares equal to its own value so a
    caller that only wants the winner can treat it as one."""

    def __init__(self, value, position, score):
        self.value = value
        self.position = position
        self.score = score

    def __eq__(self, other):
        if isinstance(other, Selected):
            return (
                self.value == other.value
                and self.position == other.position
                and self.score == other.score
            )
        return self.value == other

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return f"Selected(value={self.value!r}, position={self.position}, score={self.score!r})"


class JsonRecord(dict):
    """A JSON-safe dict a step hands on as data, and saves as a .json file.

    A dict artifact is otherwise saved one entry per file under the step's
    declared content type, so a record riding beside a video (a face track
    beside its crops) would be exploded key by key into files that are not
    video. Marking it says what it is: the writer saves it whole, as JSON,
    whatever the step's content type, and `previous_result:<step>.<key>`
    still reads it as the dict it is.
    """


# How far a decoded track may be off the frames' own duration and still be
# treated as codec padding rather than a track of its own length. AAC codes
# 1024 samples at a time, so a file's audio runs up to one such block long -
# a hundredth of a second, which accumulates into visible lip-sync drift once
# a dozen shots are joined end to end
AUDIO_FIT_TOLERANCE_SECONDS = 0.25


def fit_codec_padding(audio, frame_count, frame_rate, sample_rate):
    """Trim or pad a decoded track to exactly the frames' own duration.

    Only when the difference is codec padding. A track that genuinely runs to
    a different length than the picture - a song laid over a short clip - is
    left alone.

    `audio` may be a numpy array (the decode path) or a torch tensor still on
    its generating device (an in-memory pipeline output, #197) - the pad and
    trim below keep whichever type and device it arrived with rather than
    forcing a host round trip the caller may not want yet.
    """
    axis = sample_axis(audio)
    if axis is None:
        return audio

    expected = round(frame_count / frame_rate * sample_rate)
    difference = audio.shape[axis] - expected
    if difference == 0 or abs(difference) > AUDIO_FIT_TOLERANCE_SECONDS * sample_rate:
        return audio

    logger.debug(
        f"Fitting decoded audio to {frame_count} frames ({difference:+} samples)"
    )
    if difference > 0:
        trim = [slice(None)] * audio.ndim
        trim[axis] = slice(None, expected)
        return audio[tuple(trim)]
    if isinstance(audio, torch.Tensor):
        # torch.nn.functional.pad takes its pairs from the last axis backwards
        padding = [0, 0] * audio.ndim
        padding[2 * (audio.ndim - 1 - axis) + 1] = -difference
        return torch.nn.functional.pad(audio, padding)
    widths = [(0, 0)] * audio.ndim
    widths[axis] = (0, -difference)
    return numpy.pad(audio, widths)


def sample_axis(audio):
    """The axis a waveform's samples run along, or None if it has no such axis.

    Not a fixed index: a generated track arrives in any of the layouts
    _as_stereo reads - (channels, samples), (samples, channels), or a bare
    (samples,) - and a mono one written (samples,) or (samples, 1) used to
    reach shape[1] here and either raise IndexError or fit the wrong axis
    into a silent no-op. Channels are few and samples are many, so the
    longer axis is the sample axis.
    """
    if audio.ndim == 1:
        return 0
    if audio.ndim != 2:
        return None
    return 0 if audio.shape[0] > audio.shape[1] else 1


def warn_on_rate_override(command, actual_rate, given_rate):
    """Say when a given sample_rate relabels a named source's real rate.

    'sample_rate' always overrides the rate a file or video carries - that is
    what lets a raw waveform (which has none of its own) be handed in at all -
    but for a named source it is easy to mistake for a conversion: a workflow
    reused one variable as both 'the rate a mix runs at' and 'the rate this
    file is at', and the mismatch reached nobody until the deliverable played
    at the wrong speed with `warnings: []` (#180). emit_warning rather than
    logger.warning for the reason every other run-time audio warning here is
    (#82, #108): a caller reading the job over the API or MCP sees the
    warnings list and nothing else.
    """
    emit_warning(
        f"{command}: sample_rate={given_rate} was given, but the source "
        f"actually carries {actual_rate} Hz. The samples are being relabeled "
        f"at {given_rate} Hz, not resampled - this changes speed and pitch. "
        f"If you meant to convert the rate, use 'resample_audio' "
        f"(target_sample_rate={given_rate}) instead.",
        kind="rate_override_mismatch",
        command=command,
        file_rate=actual_rate,
        given_rate=given_rate,
    )
