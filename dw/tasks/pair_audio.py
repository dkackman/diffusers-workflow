"""Pair a video with an audio track so the two are saved as one file.

A pipeline that generates its own soundtrack returns the pair together, and the
result muxes them into a single mp4. Anything that works on the frames alone -
a latent upsampler, an interpolator, an upscaler - returns frames without it, so
the soundtrack has to be carried across the step that dropped it. That is what
this does: it puts the two back together for the step that saves them.
"""

import logging

from ..events import emit_warning
from ..result import AudioVideo
from ..shots import remeasured_shots
from .audio_utils import as_channels_samples

logger = logging.getLogger("dw")

# Only gates the unfitted mismatch warning (no 'fit' given): a track and a
# cut are frame-aligned by construction here, so a difference smaller than
# this is rounding rather than a decision anyone can act on - one video frame
# at 24 fps is 41 ms. An explicit 'fit': 'video' always fits and warns on any
# nonzero difference - the caller asked for exactness, not a guess at whether
# the gap matters
LENGTH_WARN_MS = 100.0


class _Loaded:
    """A waveform read from a file, shaped like the artifact pair_audio expects."""

    def __init__(self, audio, sample_rate):
        self.audio = audio
        self.sample_rate = sample_rate


def _frame_count(frames):
    """How many frames the video is, or None when that cannot be told cheaply
    (a lazily-decoded reader, an object with no length)."""
    try:
        return len(frames)
    except TypeError:
        shape = getattr(frames, "shape", None)
        return int(shape[0]) if shape else None


def _fit_to_video(waveform, rate, frames, fps, fit):
    """Answer the track that goes with these frames, and say when the two do
    not agree.

    The lengths of a video and the track laid over it are two numbers a
    workflow used to have to keep equal by hand, and nothing checked: the
    music-video template sliced a soundtrack of a fixed 496 frames while its
    cut followed a `shots` list, so a two-shot run wrote 10.3 s of picture
    into a 20.7 s container and reported `succeeded` with no warnings (#142).
    `fit: "video"` derives the length from the frames instead, and with no
    `fit` the mismatch is at least said out loud.
    """
    from .audio_utils import frames_to_samples, slice_samples

    if fit not in (None, "video"):
        # Refused rather than ignored: a misspelled 'fit' that quietly did
        # nothing is the silence this argument exists to end
        raise ValueError(
            f"pair_audio: 'fit' takes 'video' or nothing, got {fit!r}. "
            f"'video' cuts or pads the track to the length of the frames"
        )

    count = _frame_count(frames)
    if not count or not fps or not rate:
        # Nothing to compare against - a frame count or a rate this layer
        # cannot know is not a mismatch
        return waveform

    wanted = frames_to_samples(count, fps, rate)
    have = waveform.shape[1]
    video_seconds = count / float(fps)
    audio_seconds = have / float(rate)

    if fit != "video":
        if abs(have - wanted) / float(rate) * 1000.0 < LENGTH_WARN_MS:
            return waveform
        emit_warning(
            f"pair_audio: the track is {audio_seconds:.2f} s and the video it "
            f"is laid over is {video_seconds:.2f} s ({count} frames at "
            f"{fps:g} fps), so the saved file's duration and its frame count "
            f"disagree. Pass 'fit': 'video' to cut or pad the track to the "
            f"frames, or make the track the length of the cut.",
            kind="audio_video_length_mismatch",
            command="pair_audio",
            audio_seconds=audio_seconds,
            video_seconds=video_seconds,
        )
        return waveform

    if have == wanted:
        # Already exact - nothing to pad, trim or warn about
        return waveform

    fitted = slice_samples(waveform, 0, wanted)
    # A gap smaller than one video frame cannot line up with anything the cut
    # does - the two rates just don't divide evenly - so it is reported as
    # what it is (a sample count) rather than as a claim on the picture
    # ("the last part of the cut has no soundtrack") that a 1-sample pad does
    # not support. At 24 fps a frame is 41.67 ms; formatting a sub-frame gap
    # to two decimal places of a second is what produced "0.00 s of silence
    # ... has no soundtrack" (#429) - self-contradictory and, followed as
    # written, unfixable, since the gap is smaller than either 'a longer
    # track' or 'fewer frames' can address.
    frame_seconds = 1.0 / float(fps)
    if wanted > have:
        pad_samples = wanted - have
        pad_seconds = pad_samples / float(rate)
        if pad_seconds < frame_seconds:
            emit_warning(
                f"pair_audio: 'fit' padded the {audio_seconds:.2f} s track with "
                f"{pad_samples} sample{'s' if pad_samples != 1 else ''} "
                f"({pad_seconds * 1000:.2f} ms) of silence to reach the "
                f"{video_seconds:.2f} s of video it is laid over - under one "
                f"video frame ({frame_seconds * 1000:.1f} ms), most likely "
                f"ordinary rounding between the track's sample rate and the "
                f"video's frame rate rather than a real gap.",
                kind="audio_padded_to_video",
                command="pair_audio",
                audio_seconds=audio_seconds,
                video_seconds=video_seconds,
                pad_samples=pad_samples,
            )
        else:
            emit_warning(
                f"pair_audio: 'fit' padded the {audio_seconds:.2f} s track with "
                f"{pad_seconds:.2f} s of silence to reach the "
                f"{video_seconds:.2f} s of video it is laid over - the last part "
                f"of the cut has no soundtrack. A longer track, or fewer frames, "
                f"is what covers it.",
                kind="audio_padded_to_video",
                command="pair_audio",
                audio_seconds=audio_seconds,
                video_seconds=video_seconds,
                pad_samples=pad_samples,
            )
    else:
        trimmed_samples = have - wanted
        trimmed_seconds = trimmed_samples / float(rate)
        if trimmed_seconds < frame_seconds:
            emit_warning(
                f"pair_audio: 'fit' trimmed {trimmed_samples} sample"
                f"{'s' if trimmed_samples != 1 else ''} "
                f"({trimmed_seconds * 1000:.2f} ms) off the {audio_seconds:.2f} s "
                f"track to reach the {video_seconds:.2f} s of video it is laid "
                f"over - under one video frame ({frame_seconds * 1000:.1f} ms), "
                f"most likely ordinary rounding between the track's sample rate "
                f"and the video's frame rate rather than lost content.",
                kind="audio_trimmed_to_video",
                command="pair_audio",
                audio_seconds=audio_seconds,
                video_seconds=video_seconds,
                trimmed_seconds=trimmed_seconds,
                trimmed_samples=trimmed_samples,
            )
        else:
            emit_warning(
                f"pair_audio: 'fit' trimmed {trimmed_seconds:.2f} s "
                f"off the {audio_seconds:.2f} s track to reach the "
                f"{video_seconds:.2f} s of video it is laid over - that part of "
                f"the track, whatever it held, is gone from the deliverable. A "
                f"shorter track, or more frames, is what keeps it.",
                kind="audio_trimmed_to_video",
                command="pair_audio",
                audio_seconds=audio_seconds,
                video_seconds=video_seconds,
                trimmed_seconds=trimmed_seconds,
                trimmed_samples=trimmed_samples,
            )
    return fitted


def pair_audio(video, audio, sample_rate=None, fps=None, fit=None):
    """Pair a video's frames with an audio track.

    Args:
        video: The frames - a frame list, a frame array or tensor, or an
            AudioVideo whose own soundtrack is replaced by this one; the
            frames' own rate is carried through to the output, so set
            `result.fps` only to override it (a loaded file brings its rate
            along; frames that carry none are written at 8 fps)
        audio: The soundtrack - a waveform, an AudioVideo (or any object
            carrying '.audio') to take it from, or the path or URL of an
            audio or video file; the last two bring their sample rate along.
            A mono track is fine - saving as mp4 duplicates it into the two
            channels the audio stream takes, and warns that it did
        sample_rate: Sample rate of the waveform. Required unless `audio`
            carries one; given here it wins, for a track whose rate was
            reported wrong
        fps: The rate the frames play at, when the frames do not carry one -
            only used to work out how long the video is, never written
        fit: "video" cuts or pads the track with silence to the length of the
            frames, warning either way (`audio_padded_to_video` when it pads,
            `audio_trimmed_to_video` when it cuts). This is
            how a soundtrack follows a cut whose length is an argument
            rather than a constant: nothing in a workflow can multiply a
            list's length by a frame count, so a slice written to fit four
            shots stayed 496 frames long when the list held two, and the
            deliverable's audio ran twice as long as its picture with
            `succeeded` and no warnings (#142). Left unset the track is used
            as it is, and a length that disagrees with the frames' is
            warned about rather than passing in silence. The exactness
            `fit` guarantees is of the *waveform handed to the encoder*, not
            of the file a lossy mux (AAC, the only container this saves
            audio+video into) writes: encoding is downstream of this
            function and can still trim or pad the written track by a
            further handful of samples (#428 measured up to ~30, well under
            a millisecond) while decoding as `succeeded` with no warning of
            its own, because there is no threshold that separates that from
            ordinary codec rounding. `get_gallery_metadata`'s `media.shots`
            and `assess_output`'s `sync_length` are measured against the
            file as written, not this prediction, so they are the ground
            truth for exactly how long the saved track runs

    Returns:
        One AudioVideo holding the frames and the track, at the rate the
        frames carry - a file loaded for the `video` argument brings its
        own, so the saved mp4 plays at the rate that went in and
        `result.fps` is only needed to write it at a different one (#104)

    Raises:
        ValueError: If no waveform was given, or if no sample rate can be
            established for the one that was
    """
    if isinstance(audio, str):
        # A file an earlier run wrote - a score, or a cut whose track is wanted
        from .audio_utils import load_audio

        audio = _Loaded(*load_audio(audio))
    waveform = getattr(audio, "audio", audio)
    if waveform is None:
        raise ValueError(
            "pair_audio needs an audio track - the video it was given carries none"
        )

    rate = (
        sample_rate if sample_rate is not None else getattr(audio, "sample_rate", None)
    )
    if rate is None:
        raise ValueError(
            "pair_audio needs 'sample_rate' - the audio it was given does not "
            "carry one of its own"
        )

    # Frames are left in whatever shape they arrived in - the result saves a frame
    # list, an array and a tensor alike, and converting a long video here would
    # cost a copy of the whole thing for nothing
    frames = video.frames if isinstance(video, AudioVideo) else video
    frame_rate = fps if fps is not None else getattr(video, "fps", None)
    logger.debug(f"Pairing frames with audio at {rate} Hz")
    waveform = _fit_to_video(
        as_channels_samples(waveform), rate, frames, frame_rate, fit
    )
    # The picture's shots survive; their samples are re-measured on the new
    # track, which was laid under whole rather than built shot by shot
    return AudioVideo(
        frames,
        waveform,
        rate,
        fps=getattr(video, "fps", None),
        shots=remeasured_shots(
            getattr(video, "shots", None), frame_rate, rate, waveform.shape[1]
        ),
    )
