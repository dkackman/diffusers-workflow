"""Waveform utilities for audio tasks and segment-chained video generation.

Waveforms are handled as (channels, samples) float32 numpy arrays throughout -
as_channels_samples normalizes the shapes pipelines and files actually produce
into that layout.
"""

import io
import logging
from fractions import Fraction

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


def equal_power_crossfade_join(
    previous, head, following, sample_rate, crossfade_ms, seam_fade_ms=None
):
    """Join two segments' audio at a seam without changing the total duration.

    previous ends at the seam. head is the audio trimmed off the next segment's
    start - it covers the same stretch of time as the tail of previous, so the
    two are blended with an equal-power crossfade over the last
    min(crossfade_ms, len(head)) of that stretch. following is the next
    segment's on-timeline audio and is appended unchanged.

    With no head material (nothing was trimmed), the seam gets a fade-out and
    fade-in in place instead, of seam_fade_ms - a few milliseconds by default,
    just enough not to click.
    """
    previous, head, following = _matched_channels(previous, head, following)

    window = min(
        int(crossfade_ms / 1000.0 * sample_rate),
        head.shape[1],
        previous.shape[1],
    )

    if window == 0:
        return _declick_join(previous, following, sample_rate, seam_fade_ms)

    fade_out, fade_in = _equal_power_ramps(window)
    blended = previous[:, -window:] * fade_out + head[:, -window:] * fade_in
    return numpy.concatenate([previous[:, :-window], blended, following], axis=1)


def bleed_join(previous, following, sample_rate, bleed_ms, seam_fade_ms=None):
    """Butt-join two waveforms, ringing the outgoing tail on across the seam.

    Cut-based workflows generate every shot independently, so nothing overlaps at
    a seam and there is no trimmed material to crossfade. Generated shots also
    tend to open on near-silence and end mid-sound - a laugh track still rolling,
    a room still ringing - so a plain butt-join drops a wall of sound into a hole.

    This lays a decaying copy of the outgoing tail over the head of the incoming
    waveform, the way an audience carries across a picture cut. The copy is
    time-reversed so it starts on the outgoing waveform's own last sample and the
    seam stays continuous without a declick fade; crowd noise and room tone are
    direction-agnostic, so the reversal itself is not audible.

    The tail is added to whatever the incoming waveform already carries, and
    neither side is shortened, so frames and samples stay in step.

    Args:
        previous: Waveform ending at the seam
        following: Waveform starting at the seam
        sample_rate: Sample rate of both waveforms
        bleed_ms: How long the tail rings on, clamped to the material available
        seam_fade_ms: Fade applied on each side of the seam when there is no
            material to bleed at all

    Returns:
        The two waveforms joined, of their full combined length
    """
    previous, following = _matched_channels(previous, following)

    window = min(
        int(bleed_ms / 1000.0 * sample_rate),
        previous.shape[1],
        following.shape[1],
    )
    if window <= 0:
        return _declick_join(previous, following, sample_rate, seam_fade_ms)

    decay, _ = _equal_power_ramps(window)  # cos: 1 down to ~0
    following = following.copy()
    following[:, :window] += previous[:, ::-1][:, :window] * decay

    peak = numpy.abs(following[:, :window]).max()
    if peak > 1.0:
        logger.warning(
            f"Audio bleed pushed the seam to {peak:.2f} - it is added to the "
            f"incoming track, which was not silent enough to absorb it"
        )
    return numpy.concatenate([previous, following], axis=1)


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


def _as_number(value, kind, name):
    """Coerce a numeric slice argument given as a string, leaving None alone."""
    if not isinstance(value, str):
        return value
    try:
        return kind(value)
    except ValueError as e:
        raise ValueError(
            f"slice_audio needs a number for '{name}', got {value!r}"
        ) from e


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

    Either half of a pair may be left out: with no start the slice begins at the
    head of the track, and with no duration it runs to the end of it. A workflow
    that trims only when it is told a length therefore still produces the track
    rather than failing.

    Args:
        audio: Path or URL of an audio file, a video generated with a
            soundtrack (which brings its sample rate along), or a waveform
            (which needs sample_rate alongside it)
        sample_rate: Sample rate of a waveform passed directly; given for a
            file or a video it overrides the rate they carry

    Returns:
        The slice as a (samples, channels) float32 array - the layout audio
        results are saved in
    """
    # A variable a workflow declares null carries no type, so a value given for
    # it on the command line arrives as a string - the same coercion the upscale
    # and interpolation tasks do on their numeric arguments
    start_seconds = _as_number(start_seconds, float, "start_seconds")
    duration_seconds = _as_number(duration_seconds, float, "duration_seconds")
    start_frame = _as_number(start_frame, int, "start_frame")
    num_frames = _as_number(num_frames, int, "num_frames")
    fps = _as_number(fps, Fraction, "fps")

    waveform, sample_rate = _waveform_and_rate(audio, sample_rate, "slice_audio")
    total = waveform.shape[1]

    if start_seconds is not None or duration_seconds is not None:
        start = int(round((start_seconds or 0) * sample_rate))
        length = (
            max(total - start, 0)
            if duration_seconds is None
            else int(round(duration_seconds * sample_rate))
        )
    elif start_frame is not None or num_frames is not None:
        if fps is None:
            raise ValueError("slice_audio needs 'fps' to address a slice in frames")
        start = frames_to_samples(start_frame or 0, fps, sample_rate)
        length = (
            max(total - start, 0)
            if num_frames is None
            else frames_to_samples(num_frames, fps, sample_rate)
        )
    else:
        raise ValueError(
            "slice_audio needs either 'start_seconds'/'duration_seconds' or "
            "'start_frame'/'num_frames'/'fps'"
        )

    return slice_samples(waveform, start, length).T


def resample_audio(audio, target_sample_rate, sample_rate=None):
    """Task command: resample an audio track to a different sample rate.

    MiniMax H3 conditions on audio at its audio VAE's own rate and resamples
    anything else with torchaudio, which dw does not depend on. Resampling a
    supplied recording once, up front, feeds the pipeline what it already wants
    and keeps the dependency out - PyAV, which dw needs for video anyway, does
    the conversion.

    Args:
        audio: Path or URL of an audio file, a video generated with a
            soundtrack (which brings its sample rate along), or a waveform
            (which needs sample_rate alongside it)
        target_sample_rate: Rate to convert to
        sample_rate: Sample rate of a waveform passed directly; given for a
            file or a video it overrides the rate they carry

    Returns:
        The resampled track as a (samples, channels) float32 array
    """
    waveform, sample_rate = _waveform_and_rate(audio, sample_rate, "resample_audio")

    if sample_rate == target_sample_rate:
        return waveform.T

    import av
    from av.audio.resampler import AudioResampler

    channels = waveform.shape[0]
    layout = {1: "mono", 2: "stereo"}.get(channels, f"{channels}c")
    frame = av.AudioFrame.from_ndarray(
        numpy.ascontiguousarray(waveform, dtype=numpy.float32),
        format="fltp",
        layout=layout,
    )
    frame.sample_rate = sample_rate
    frame.pts = 0
    frame.time_base = Fraction(1, sample_rate)

    resampler = AudioResampler(format="fltp", layout=layout, rate=target_sample_rate)
    converted = [f.to_ndarray() for f in resampler.resample(frame)]
    converted += [f.to_ndarray() for f in resampler.resample(None)]
    logger.debug(
        f"Resampled {waveform.shape[1]} samples at {sample_rate}Hz "
        f"to {target_sample_rate}Hz"
    )
    return numpy.concatenate(converted, axis=1).astype(numpy.float32).T


def crossfade_audio(audios, crossfade_ms=75, sample_rate=None):
    """Task command: join audio tracks with an equal-power crossfade.

    Each seam overlaps the two tracks by the fade window, so the result is
    shorter than the plain sum by one window per seam.

    Args:
        audios: The tracks to join, in order - waveforms, audio file paths, or
            videos generated with a soundtrack
        crossfade_ms: Length of each crossfade
        sample_rate: Sample rate of the waveforms. Required unless every track
            brings its own; given here it wins

    Returns:
        The joined track as a (samples, channels) float32 array
    """
    if not isinstance(audios, list) or not audios:
        raise ValueError("crossfade_audio needs a non-empty list of audio tracks")
    waveforms, rates, bare = [], set(), False
    for audio in audios:
        if isinstance(audio, str) or hasattr(audio, "audio"):
            waveform, rate = _waveform_and_rate(audio, sample_rate, "crossfade_audio")
            rates.add(rate)
        else:
            waveform, bare = as_channels_samples(audio), True
        waveforms.append(waveform)
    if sample_rate is None:
        if bare or not rates:
            raise ValueError("crossfade_audio needs 'sample_rate' with a raw waveform")
        if len(rates) > 1:
            raise ValueError(
                f"crossfade_audio needs one sample rate, got {sorted(rates)}"
            )
        sample_rate = rates.pop()
    return crossfade_concat(waveforms, sample_rate, crossfade_ms).T


def mix_audio(audios, gains=None, sample_rate=None):
    """Task command: layer audio tracks on top of one another.

    crossfade_audio puts tracks one after another; this puts them on top of
    each other. It is what a score laid under a film's own sound needs: the
    music runs unbroken while the world underneath it is replaced at every cut.

    Tracks of different lengths are padded with silence to the longest, so a
    score shorter than the picture leaves the tail dry rather than cutting the
    picture down to fit.

    Summing can push peaks past full scale. This returns the plain weighted sum
    and does not rescale it, since quietening a mix is a decision about how it
    should sound - follow it with normalize_audio to bring the peak back down.

    Args:
        audios: The tracks to layer - waveforms, audio file paths, or videos
            generated with a soundtrack
        gains: One plain multiplier per track, in the same order - not decibels.
            Defaults to unity on every track
        sample_rate: Sample rate of the waveforms. Required unless every track
            brings its own; given here it wins

    Returns:
        The mixed track as a (samples, channels) float32 array
    """
    if not isinstance(audios, list) or not audios:
        raise ValueError("mix_audio needs a non-empty list of audio tracks")
    if gains is not None and len(gains) != len(audios):
        raise ValueError(
            f"mix_audio needs one gain per track - got {len(gains)} for "
            f"{len(audios)} tracks"
        )

    waveforms, rates, bare = [], set(), False
    for audio in audios:
        if isinstance(audio, str) or hasattr(audio, "audio"):
            waveform, rate = _waveform_and_rate(audio, sample_rate, "mix_audio")
            rates.add(rate)
        else:
            waveform, bare = as_channels_samples(audio), True
        waveforms.append(waveform)
    if sample_rate is None:
        if bare or not rates:
            raise ValueError("mix_audio needs 'sample_rate' with a raw waveform")
        if len(rates) > 1:
            raise ValueError(f"mix_audio needs one sample rate, got {sorted(rates)}")
        sample_rate = rates.pop()

    waveforms = _matched_channels(*waveforms)
    channels = waveforms[0].shape[0]
    length = max(waveform.shape[1] for waveform in waveforms)

    mixed = numpy.zeros((channels, length), dtype=numpy.float32)
    for index, waveform in enumerate(waveforms):
        gain = 1.0 if gains is None else float(gains[index])
        mixed[:, : waveform.shape[1]] += waveform * gain
    return mixed.T


def _equal_power_ramps(window):
    """Cosine/sine fade curves that sum to constant power across the window."""
    theta = numpy.linspace(0.0, numpy.pi / 2.0, window, endpoint=False)
    return numpy.cos(theta, dtype=numpy.float32), numpy.sin(theta, dtype=numpy.float32)


def _declick_join(previous, following, sample_rate, fade_ms=None):
    """Butt-join two waveforms with a fade on each side of the seam.

    The default is the few milliseconds that keep a butt-join from clicking.
    A longer fade is a deliberate edit - the graceful hard cut you want when
    neither a crossfade nor a bleed applies.
    """
    ramp = int((DECLICK_MS if fade_ms is None else fade_ms) / 1000.0 * sample_rate)
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


def fade_audio(audio, fade_in_ms=0, fade_out_ms=0, sample_rate=None):
    """Task command: fade a track in from silence and out to it.

    A slice cut out of the middle of a piece ends on whatever was sounding at
    the cut; a short fade turns that into an ending. The curve is the
    equal-power cosine the seam joins use, so a fade sounds like a fade and
    not a volume knob.

    Args:
        audio: Path or URL of an audio file, a video generated with a
            soundtrack, or a waveform (which needs sample_rate alongside it)
        fade_in_ms: Length of the fade in, from the head of the track
        fade_out_ms: Length of the fade out, to the tail of the track
        sample_rate: Sample rate of a waveform passed directly

    Returns:
        The faded track as a (samples, channels) float32 array
    """
    waveform, sample_rate = _waveform_and_rate(audio, sample_rate, "fade_audio")
    if fade_in_ms < 0 or fade_out_ms < 0:
        raise ValueError("fade_audio fade lengths cannot be negative")
    faded = waveform.copy()
    length = faded.shape[1]

    fade_in = min(int(round(fade_in_ms / 1000 * sample_rate)), length)
    if fade_in:
        faded[:, :fade_in] *= _fade_curve(fade_in)[::-1]
    fade_out = min(int(round(fade_out_ms / 1000 * sample_rate)), length)
    if fade_out:
        faded[:, length - fade_out :] *= _fade_curve(fade_out)
    return faded.T


def normalize_audio(audio, peak_dbfs=-1.0, sample_rate=None):
    """Task command: scale a track so its loudest sample sits at a level.

    Generated music comes out at whatever level the model happened to land
    on - quiet takes need lifting before they sit under a picture, and a
    hot one needs headroom before the encoder. Peak normalization changes
    nothing but the gain, so the dynamics survive.

    Args:
        audio: Path or URL of an audio file, a video generated with a
            soundtrack, or a waveform (which needs sample_rate alongside it)
        peak_dbfs: The level the loudest sample is moved to, in dB below full
            scale. 0 is full scale; -1 leaves a little headroom
        sample_rate: Sample rate of a waveform passed directly

    Returns:
        The scaled track as a (samples, channels) float32 array; a silent
        track is returned unchanged
    """
    waveform, _ = _waveform_and_rate(audio, sample_rate, "normalize_audio")
    if peak_dbfs > 0:
        raise ValueError("normalize_audio 'peak_dbfs' cannot be above full scale (0)")
    peak = float(numpy.abs(waveform).max()) if waveform.size else 0.0
    if peak == 0.0:
        logger.warning("normalize_audio: the track is silent - left unchanged")
        return waveform.T
    gain = 10 ** (peak_dbfs / 20) / peak
    logger.debug(
        f"normalize_audio: peak {peak:.3f}, gain {20 * numpy.log10(gain):+.1f} dB"
    )
    return (waveform * gain).astype(numpy.float32).T


def _fade_curve(window):
    """A cosine fall from full level to exact silence, both ends included -
    unlike the seam ramps, which stop short of the endpoint so two of them
    tile a crossfade without a doubled sample."""
    theta = numpy.linspace(0.0, numpy.pi / 2.0, window, endpoint=True)
    return numpy.cos(theta, dtype=numpy.float32)


def _waveform_and_rate(audio, sample_rate, command):
    """A command's audio argument as a (channels, samples) array with its rate.

    A path loads with the file's own rate; a video generated with a soundtrack
    (an AudioVideo, or anything carrying `.audio`) contributes that track and
    its rate; a bare waveform needs the rate given. A given rate always wins.
    """
    if isinstance(audio, str):
        waveform, file_rate = load_audio(audio)
        return waveform, sample_rate if sample_rate is not None else file_rate
    if hasattr(audio, "audio"):
        if audio.audio is None:
            raise ValueError(
                f"{command} needs an audio track - the video it was given carries none"
            )
        rate = sample_rate if sample_rate is not None else audio.sample_rate
        if rate is None:
            raise ValueError(
                f"{command} needs 'sample_rate' - the video it was given does not "
                "carry one of its own"
            )
        return as_channels_samples(audio.audio), rate
    if sample_rate is None:
        raise ValueError(f"{command} needs 'sample_rate' with a raw waveform")
    return as_channels_samples(audio), sample_rate
