"""Waveform utilities for audio tasks and segment-chained video generation.

Waveforms are handled as (channels, samples) float32 numpy arrays throughout -
as_channels_samples normalizes the shapes pipelines and files actually produce
into that layout.
"""

import io
import logging

import numpy
import soundfile
import torch

from ..security import (
    validate_path,
    validate_url,
    validate_file_extension,
    ALLOWED_AUDIO_EXTENSIONS,
)

logger = logging.getLogger("dw")

# A few milliseconds of fade applied on each side of a butt-joined seam so the
# discontinuity does not click
DECLICK_MS = 3.0


def as_channels_samples(audio):
    """Normalize a waveform to a (channels, samples) float32 numpy array.

    Accepts torch tensors or numpy arrays shaped (samples,), (channels, samples),
    (samples, channels), or a one-item batch (1, channels, samples). Channel
    position is decided the way normalize_audio in result.py decides it: there
    are always more samples than channels.
    """
    if torch.is_tensor(audio):
        audio = audio.detach().cpu().float().numpy()
    audio = numpy.asarray(audio, dtype=numpy.float32)

    if audio.ndim == 1:
        return audio[numpy.newaxis, :]

    if audio.ndim == 3:
        if audio.shape[0] != 1:
            raise ValueError(f"Cannot normalize a waveform batch of {audio.shape[0]}")
        audio = audio[0]

    if audio.ndim != 2:
        raise ValueError(f"A waveform must have 1-3 dimensions, got {audio.ndim}")

    if audio.shape[0] > audio.shape[1]:  # (samples, channels) -> transpose
        audio = audio.T

    return numpy.ascontiguousarray(audio)


def frames_to_samples(frames, fps, sample_rate):
    """The number of audio samples spanning a run of video frames."""
    return int(round(frames / fps * sample_rate))


def slice_samples(waveform, start, length):
    """Cut length samples out of a (channels, samples) waveform from start.

    A slice reaching past the end of the waveform is zero-padded to the
    requested length, so frame-aligned slicing near the end of a track always
    yields full-size chunks.
    """
    channels, total = waveform.shape
    piece = waveform[:, start : start + length]
    if piece.shape[1] < length:
        padding = numpy.zeros((channels, length - piece.shape[1]), dtype=waveform.dtype)
        piece = numpy.concatenate([piece, padding], axis=1)
    return piece


def equal_power_crossfade_join(previous, head, following, sample_rate, crossfade_ms):
    """Join two segments' audio at a seam without changing the total duration.

    previous ends at the seam. head is the audio trimmed off the next segment's
    start - it covers the same stretch of time as the tail of previous, so the
    two are blended with an equal-power crossfade over the last
    min(crossfade_ms, len(head)) of that stretch. following is the next
    segment's on-timeline audio and is appended unchanged.

    With no head material (nothing was trimmed), the seam gets a short declick
    ramp instead - a few milliseconds of fade-out and fade-in in place.
    """
    previous, head, following = _matched_channels(previous, head, following)

    window = min(
        int(crossfade_ms / 1000.0 * sample_rate),
        head.shape[1],
        previous.shape[1],
    )

    if window == 0:
        return _declick_join(previous, following, sample_rate)

    fade_out, fade_in = _equal_power_ramps(window)
    blended = previous[:, -window:] * fade_out + head[:, -window:] * fade_in
    return numpy.concatenate([previous[:, :-window], blended, following], axis=1)


def crossfade_concat(waveforms, sample_rate, crossfade_ms):
    """Concatenate waveforms, overlapping each seam by an equal-power crossfade.

    The classic crossfade: each seam overlaps the two waveforms by the fade
    window, so the result is shorter than the plain sum by one window per seam.
    """
    waveforms = [as_channels_samples(waveform) for waveform in waveforms]
    if not waveforms:
        raise ValueError("No waveforms to concatenate")

    result = waveforms[0]
    for following in waveforms[1:]:
        result, following = _matched_channels(result, following)
        window = min(
            int(crossfade_ms / 1000.0 * sample_rate),
            result.shape[1],
            following.shape[1],
        )
        if window == 0:
            result = _declick_join(result, following, sample_rate)
            continue

        fade_out, fade_in = _equal_power_ramps(window)
        blended = result[:, -window:] * fade_out + following[:, :window] * fade_in
        result = numpy.concatenate(
            [result[:, :-window], blended, following[:, window:]], axis=1
        )

    return result


def load_audio(location, base_dir=None):
    """Load an audio file from a local path or http(s) URL.

    Returns:
        Tuple of a (channels, samples) float32 waveform and its sample rate
    """
    if location.startswith(("http://", "https://")):
        import requests

        validated_url = validate_url(location)
        logger.debug(f"Downloading audio from {validated_url}")
        response = requests.get(validated_url, timeout=60)
        response.raise_for_status()
        data, sample_rate = soundfile.read(
            io.BytesIO(response.content), dtype="float32"
        )
    else:
        validated_path = validate_path(location, base_dir=base_dir, allow_create=False)
        validate_file_extension(validated_path, ALLOWED_AUDIO_EXTENSIONS)
        logger.debug(f"Reading audio from {validated_path}")
        data, sample_rate = soundfile.read(validated_path, dtype="float32")

    # soundfile returns (samples,) or (samples, channels)
    return as_channels_samples(data), sample_rate


def slice_audio(
    audio,
    start_seconds=None,
    duration_seconds=None,
    start_frame=None,
    num_frames=None,
    fps=None,
    sample_rate=None,
):
    """Task command: cut a slice out of an audio track.

    The slice is addressed either in seconds (start_seconds + duration_seconds)
    or in video frames (start_frame + num_frames + fps). Slices reaching past
    the end of the track are zero-padded.

    Args:
        audio: Path or URL of an audio file, or a waveform (which needs
            sample_rate alongside it)
        sample_rate: Sample rate of a waveform passed directly; ignored for
            files, which carry their own

    Returns:
        The slice as a (samples, channels) float32 array - the layout audio
        results are saved in
    """
    if isinstance(audio, str):
        waveform, sample_rate = load_audio(audio)
    else:
        if sample_rate is None:
            raise ValueError("slice_audio needs 'sample_rate' with a raw waveform")
        waveform = as_channels_samples(audio)

    if start_seconds is not None or duration_seconds is not None:
        if start_seconds is None or duration_seconds is None:
            raise ValueError(
                "slice_audio needs both 'start_seconds' and 'duration_seconds'"
            )
        start = int(round(start_seconds * sample_rate))
        length = int(round(duration_seconds * sample_rate))
    elif start_frame is not None or num_frames is not None:
        if start_frame is None or num_frames is None or fps is None:
            raise ValueError(
                "slice_audio needs 'start_frame', 'num_frames' and 'fps' together"
            )
        start = frames_to_samples(start_frame, fps, sample_rate)
        length = frames_to_samples(num_frames, fps, sample_rate)
    else:
        raise ValueError(
            "slice_audio needs either 'start_seconds'/'duration_seconds' or "
            "'start_frame'/'num_frames'/'fps'"
        )

    return slice_samples(waveform, start, length).T


def crossfade_audio(audios, crossfade_ms=75, sample_rate=None):
    """Task command: join audio tracks with an equal-power crossfade.

    Each seam overlaps the two tracks by the fade window, so the result is
    shorter than the plain sum by one window per seam.

    Args:
        audios: The waveforms to join, in order
        crossfade_ms: Length of each crossfade
        sample_rate: Sample rate of the waveforms

    Returns:
        The joined track as a (samples, channels) float32 array
    """
    if sample_rate is None:
        raise ValueError("crossfade_audio needs 'sample_rate'")
    return crossfade_concat(audios, sample_rate, crossfade_ms).T


def _equal_power_ramps(window):
    """Cosine/sine fade curves that sum to constant power across the window."""
    theta = numpy.linspace(0.0, numpy.pi / 2.0, window, endpoint=False)
    return numpy.cos(theta, dtype=numpy.float32), numpy.sin(theta, dtype=numpy.float32)


def _declick_join(previous, following, sample_rate):
    """Butt-join two waveforms with a short fade on each side of the seam."""
    ramp = int(DECLICK_MS / 1000.0 * sample_rate)
    ramp = min(ramp, previous.shape[1], following.shape[1])
    if ramp > 0:
        fade_out, fade_in = _equal_power_ramps(ramp)
        previous = previous.copy()
        following = following.copy()
        previous[:, -ramp:] *= fade_out  # cos: 1 down to ~0
        following[:, :ramp] *= fade_in  # sin: ~0 up to 1
    return numpy.concatenate([previous, following], axis=1)


def _matched_channels(*waveforms):
    """Tile mono up so every waveform has the same channel count."""
    channels = max(waveform.shape[0] for waveform in waveforms)
    return tuple(
        (
            numpy.tile(waveform, (channels, 1))
            if waveform.shape[0] == 1 and channels > 1
            else waveform
        )
        for waveform in waveforms
    )
