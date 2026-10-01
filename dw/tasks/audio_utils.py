"""Waveform utilities for audio tasks and segment-chained video generation.

Waveforms are handled as (channels, samples) float32 numpy arrays throughout -
as_channels_samples normalizes the shapes pipelines and files actually produce
into that layout.
"""

import io
import os
import logging
from fractions import Fraction

import numpy
import soundfile

from .. import dsp
from ..dsp import (
    LEVEL_MEASURES,
    LIMITER_HEAVY_DB,
    LIMITER_MAX_REDUCTION_DB,
    LIMITER_TOLERANCE_LU,
    UNITY_TOLERANCE,
    apply_biquad,
    as_channels_samples,
    biquad_coefficients,
    equal_power_ramps,
    fade_curve,
    follow_envelope,
    harmonicity,
    integrated_lufs,
    level_dbfs,
    limit_at,
    matched_channels,
    search_gain,
    slice_samples,
    spectral_balance,
    spectral_flatness,
    true_peak_envelope,
)
from ..events import emit_log, emit_warning
from ..media_types import AudioTrack, warn_on_rate_override
from ..task_domains import (
    SLICE_PAD_WARN_MS as SLICE_PAD_WARN_MS,
    as_number,
    check_arguments,
    frames_to_samples as frames_to_samples,
    slice_padding,
)
from ..security import (
    validate_file_extension,
    ALLOWED_AUDIO_EXTENSIONS,
)

logger = logging.getLogger("dw")

# A few milliseconds of fade applied on each side of a butt-joined seam so the
# discontinuity does not click
DECLICK_MS = 3.0

# SLICE_PAD_WARN_MS and frames_to_samples live in dw/task_domains.py, shared
# with validate's slice check; imported above and re-exported from here for
# the modules that already import them from this one

# A dropped tail is only the "almost reached the end" signature this warning
# exists for when it is both short next to the slice and short in absolute
# terms - a deliberate excerpt out of a long recording drops most of the
# source and should not warn
SLICE_TRIM_WARN_SECONDS = 10.0
SLICE_TRIM_WARN_FRACTION = 0.05

# A remainder shorter than this is the rounding frame-aligned slicing
# produces (or a sub-millisecond seconds-addressed remainder), not a
# noticeable dropped tail - the same floor SLICE_PAD_WARN_MS applies on the
# other side of a slice
SLICE_TRIM_WARN_MIN_MS = 10.0


def fit_audio_to_frames(audio, sample_rate, total_frames, fps, command):
    """Pad a joined track that falls short of its frame grid, and warn when
    the gap is a frame or more.

    concat_videos and dissolve_videos each build their joined track by
    measuring and concatenating/crossfading the actual input waveforms, with
    nothing reconciling a shortfall against total_frames - so an input that
    is itself short of its own frame grid (#435 traced this to an
    ltx2/keyframes clip short of its 121-frame bucket) propagates its
    shortfall into the join, and the shortfall compounds across further
    joins that each take the previous join's output as an input. The same
    remedy #428 gave pair_audio's 'fit: video' for a shortfall, applied here
    at the one place every join's audio passes through before its shot map
    is measured.

    A track *longer* than its frame grid is left alone: concat_videos has
    measured such an overrun deliberately since #378 (its own shot keeps the
    samples it actually took, not a count derived from frame/fps
    arithmetic), and trimming it here would silently reverse that contract
    for the whole joined track.
    """
    if audio is None or not total_frames or not fps or not sample_rate:
        return audio

    wanted = frames_to_samples(total_frames, fps, sample_rate)
    have = audio.shape[1]
    if have >= wanted:
        return audio

    audio_seconds = have / float(sample_rate)
    video_seconds = total_frames / float(fps)
    pad_samples = wanted - have
    audio = numpy.pad(audio, ((0, 0), (0, pad_samples)))
    if pad_samples < sample_rate / fps:
        # Less than one frame is rounding between the rate and the frame
        # grid, the gap pair_audio's own unfitted check leaves unwarned
        # (LENGTH_WARN_MS): padded, and logged, but not a warning on every
        # stock join (#454)
        emit_log(
            f"{command}: padded the joined track with {pad_samples} sample"
            f"{'s' if pad_samples != 1 else ''} of silence to the frame grid",
            command=command,
            pad_samples=pad_samples,
        )
        return audio
    emit_warning(
        f"{command}: the joined track is {audio_seconds:.3f} s and the "
        f"joined video is {video_seconds:.3f} s ({total_frames} frames at "
        f"{fps:g} fps) - padded the track with {pad_samples} sample"
        f"{'s' if pad_samples != 1 else ''} of silence to reach the frame "
        "grid, so the shortfall does not carry into a later join.",
        kind="joined_audio_padded_to_frames",
        command=command,
        audio_seconds=audio_seconds,
        video_seconds=video_seconds,
        pad_samples=pad_samples,
    )
    return audio


def equal_power_crossfade_join(
    previous, head, following, sample_rate, crossfade_ms, seam_fade_ms=None, seam=None
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
    previous, head, following = matched_channels(previous, head, following)

    window = min(
        int(crossfade_ms / 1000.0 * sample_rate),
        head.shape[1],
        previous.shape[1],
    )

    if window == 0:
        return _declick_join(previous, following, sample_rate, seam_fade_ms, seam)

    fade_out, fade_in = equal_power_ramps(window)
    blended = previous[:, -window:] * fade_out + head[:, -window:] * fade_in
    return numpy.concatenate([previous[:, :-window], blended, following], axis=1)


# Below this, the reversed tail's spectral energy is concentrated in a few
# bins rather than spread across the band - speech or a pitched/tonal element
# rather than room tone or crowd noise, the direction-agnostic material a
# bleed is meant for
TONAL_FLATNESS_THRESHOLD = 0.3

# Above this, the tail's waveform repeats closely enough within a plausible
# pitch period to be voiced speech or a pitched note rather than noise -
# a bandwidth-insensitive companion to flatness, since a resample's own
# band limiting does not touch how periodic the waveform is (#198)
HARMONICITY_THRESHOLD = 0.45


def bleed_join(
    previous,
    following,
    sample_rate,
    bleed_ms,
    seam_fade_ms=None,
    gain_db=0.0,
    native_sample_rate=None,
    seam=None,
    between=None,
):
    """Butt-join two waveforms, ringing the outgoing tail on across the seam.

    Cut-based workflows generate every shot independently, so nothing overlaps at
    a seam and there is no trimmed material to crossfade. Generated shots also
    tend to open on near-silence and end mid-sound - a laugh track still rolling,
    a room still ringing - so a plain butt-join drops a wall of sound into a hole.

    This lays a decaying copy of the outgoing tail over the head of the incoming
    waveform, the way an audience carries across a picture cut. The copy is
    time-reversed so it starts on the outgoing waveform's own last sample and the
    seam stays continuous without a declick fade; crowd noise and room tone are
    direction-agnostic, so the reversal itself is not audible - speech or a
    tonal/musical tail is not, which is what gets a warning below rather than a
    refusal, since a caller who has already listened to the material may still
    want the bleed.

    The tail is added to whatever the incoming waveform already carries, and
    neither side is shortened, so frames and samples stay in step.

    Args:
        previous: Waveform ending at the seam
        following: Waveform starting at the seam
        sample_rate: Sample rate of both waveforms
        bleed_ms: How long the tail rings on, clamped to the material available
        seam_fade_ms: Fade applied on each side of the seam when there is no
            material to bleed at all
        gain_db: Gain applied to the bled copy before it is added, in dB -
            negative ducks a tail that would otherwise push the seam over
            0 dBFS; 0 (the default) is unchanged, full-scale, the prior
            behavior
        native_sample_rate: The rate the outgoing tail was actually recorded
            or generated at, when that differs from sample_rate because the
            caller upsampled it to join. Band-limits the flatness check to
            below the tail's own Nyquist, so upsampling's near-silent high
            band cannot itself read as tonal (#198). Omit when the tail is
            already at its native rate
        seam: The seam's index, named in the log line and the tonal warning
        between: What the seam joins ("a -> b"), named beside the index

    Returns:
        The two waveforms joined, of their full combined length
    """
    previous, following = matched_channels(previous, following)
    where = "a seam" if seam is None else f"seam {seam}"
    if between:
        where = f"{where} ({between})"

    window = min(
        int(bleed_ms / 1000.0 * sample_rate),
        previous.shape[1],
        following.shape[1],
    )
    if window <= 0:
        return _declick_join(previous, following, sample_rate, seam_fade_ms, seam)

    tail = previous[:, ::-1][:, :window]

    tail_source = previous[:, -window:]
    flatness = spectral_flatness(tail_source, sample_rate, native_sample_rate)
    # sample_rate, not native_sample_rate: harmonicity turns a rate into lag
    # bounds in samples of the waveform it is handed, and that waveform is at
    # sample_rate however it got there. Passing the native rate of an upsampled
    # tail searched the wrong lag range (16k against a 48k track: 180-1500 Hz
    # rather than 60-500) and could miss the voiced speech #198 added it for.
    # Only spectral_flatness wants the native rate, to band-limit its window.
    periodicity = harmonicity(tail_source, sample_rate)
    if flatness < TONAL_FLATNESS_THRESHOLD or periodicity > HARMONICITY_THRESHOLD:
        emit_warning(
            f"bleed_join: the tail being reversed onto {where} looks tonal or "
            f"speech-like (spectral flatness {flatness:.2f}, harmonicity "
            f"{periodicity:.2f}) rather than the room tone or crowd noise a "
            f"bleed is meant for - the reversal is likely to be audible as a "
            f"stutter or a note running backwards. Pass 'audio_bleed_ms': 0 "
            f"for a hard cut on this material instead - seam_fade_ms has no "
            f"effect while audio_bleed_ms is non-zero.",
            kind="bleed_tonal_material",
            command="bleed_join",
            seam=seam,
            flatness=round(flatness, 3),
            harmonicity=round(periodicity, 3),
        )

    decay, _ = equal_power_ramps(window)  # cos: 1 down to ~0
    gain = 10.0 ** (gain_db / 20.0) if gain_db else 1.0
    following = following.copy()
    following[:, :window] += tail * decay * gain
    emit_log(
        f"audio_bleed: {window / sample_rate * 1000:.0f} ms of tail over {where}"
        f" at {gain_db:g} dB (asked {bleed_ms} ms)",
        command="audio_bleed",
        seam=seam,
        bleed_ms=round(window / sample_rate * 1000, 1),
        requested_ms=bleed_ms,
        gain_db=gain_db,
    )

    peak = numpy.abs(following[:, :window]).max()
    if peak > 1.0:
        logger.warning(
            f"Audio bleed pushed the seam to {peak:.2f} - it is added to the "
            f"incoming track, which was not silent enough to absorb it. Pass "
            f"a negative gain_db to duck the bled copy."
        )
    return numpy.concatenate([previous, following], axis=1)


def crossfade_concat(waveforms, sample_rate, crossfade_ms, starts=None):
    """Concatenate waveforms, overlapping each seam by an equal-power crossfade.

    The classic crossfade: each seam overlaps the two waveforms by the fade
    window, so the result is shorter than the plain sum by one window per seam.

    `starts`, when given a list, is filled with the sample each waveform
    begins at in the result - where its crossfade opens - measured as the
    result grows rather than worked out from the lengths (#378).
    """
    waveforms = [as_channels_samples(waveform) for waveform in waveforms]
    if not waveforms:
        raise ValueError("No waveforms to concatenate")

    result = waveforms[0]
    if starts is not None:
        starts.append(0)
    for following in waveforms[1:]:
        result, following = matched_channels(result, following)
        window = min(
            int(round(crossfade_ms / 1000.0 * sample_rate)),
            result.shape[1],
            following.shape[1],
        )
        if starts is not None:
            starts.append(result.shape[1] - window)
        if window == 0:
            result = _declick_join(result, following, sample_rate)
            continue

        fade_out, fade_in = equal_power_ramps(window)
        blended = result[:, -window:] * fade_out + following[:, :window] * fade_in
        result = numpy.concatenate(
            [result[:, :-window], blended, following[:, window:]], axis=1
        )

    return result


def load_audio(location, base_dir=None):
    """Load an audio file from a local path or http(s) URL.

    A video file loads too, and contributes the track muxed into it: the cut
    an earlier run wrote is exactly what a scoring pass wants to mix under,
    and refusing its extension made an agent extract the audio by hand
    (2026-09-08). A video without an audio stream is an error, not silence.

    Returns:
        Tuple of a (channels, samples) float32 waveform and its sample rate
    """
    from ..security import ALLOWED_VIDEO_EXTENSIONS

    extension = os.path.splitext(location.split("?", 1)[0])[1].lower()
    if extension in ALLOWED_VIDEO_EXTENSIONS:
        from .video_utils import load_audio_video

        video = load_audio_video(location, base_dir=base_dir)
        if video.audio is None:
            raise ValueError(
                f"{location} carries no audio track - a video's soundtrack is "
                "what an audio task takes from it"
            )
        return as_channels_samples(video.audio), video.sample_rate

    if location.startswith(("http://", "https://")):
        from ..locations import safe_get

        logger.debug(f"Downloading audio from {location}")
        response = safe_get(location, "an audio argument", timeout=60)
        data, sample_rate = soundfile.read(
            io.BytesIO(response.content), dtype="float32"
        )
    else:
        from ..locations import validate_media_path

        validated_path = validate_media_path(location, base_dir, "an audio argument")
        validate_file_extension(validated_path, ALLOWED_AUDIO_EXTENSIONS)
        logger.debug(f"Reading audio from {validated_path}")
        data, sample_rate = soundfile.read(validated_path, dtype="float32")

    # soundfile returns (samples,) or (samples, channels)
    return as_channels_samples(data), sample_rate


def _as_track(waveform, sample_rate, command="an audio task", source_mean_dbfs=None):
    """An audio task's return value: the waveform with the rate it is at.

    Every one of these commands already knows the rate - it was given, or it
    came off the file or the video the track was taken from - and dropping it
    on the way out made the next command in the chain ask for it again. A
    resample fed straight from a slice failed for want of a number the slice
    had read and thrown away (2026-09-11). An AudioTrack carries it, and
    everything downstream of audio reads '.audio'/'.sample_rate' already; a
    'sample_rate' the workflow declares on the result still wins at save.

    A rate that is not a rate stops here. Saving falls back to 44100 Hz for a
    track that carries none (DEFAULT_AUDIO_SAMPLE_RATE in result.py), so a
    zero handed through would have been written as a 44100 Hz header over
    samples at some other rate - the same audio at the wrong speed and pitch,
    reported as a success (#140). There is no waveform whose rate is zero, so
    the only thing to do with one is refuse it.
    """
    rate = int(sample_rate) if sample_rate is not None else 0
    if rate <= 0:
        raise ValueError(
            f"{command} ended up with a sample rate of {sample_rate!r}, which "
            f"is not a rate. Labelling a waveform with a rate it is not at "
            f"changes its speed and pitch, so it is refused rather than "
            f"written"
        )
    return AudioTrack(
        numpy.ascontiguousarray(waveform), rate, source_mean_dbfs=source_mean_dbfs
    )


def _as_number(value, kind, name, command="slice_audio"):
    """Coerce a numeric task argument given as a string, leaving None alone."""
    if not isinstance(value, str):
        return value
    try:
        return kind(value)
    except ValueError as e:
        raise ValueError(f"{command} needs a number for '{name}', got {value!r}") from e


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
    or in video frames (start_frame + num_frames + fps).

    A slice reaching past the end of the track is zero-padded to the length
    asked for - it does not fail and it is not shortened - and the padding is
    digital silence, so asking for more than the source holds returns a track
    that is partly empty. Anything past a few milliseconds of that is
    reported as a 'slice_past_end' warning on the job. To fill a cut longer
    than the recording, make a bed with the 'loop_audio' task first
    ('target_frames' + 'fps' matches one exactly) and slice that.

    Either half of a pair may be left out: with no start the slice begins at the
    head of the track, and with no duration it runs to the end of it. A workflow
    that trims only when it is told a length therefore still produces the track
    rather than failing.

    Args:
        audio: Path or URL of an audio file (or of a video file, whose
            soundtrack is taken), a video generated with a
            soundtrack (which brings its sample rate along), or a waveform
            (which needs sample_rate alongside it)
        sample_rate: Sample rate of a waveform passed directly; given for a
            file or a video it overrides the rate they carry

    Returns:
        An AudioTrack holding the slice and the rate it is at, so the next
        audio command in the chain does not have to be told the rate again
    """
    # A variable a workflow declares null carries no type, so a value given for
    # it on the command line arrives as a string - the same coercion the upscale
    # and interpolation tasks do on their numeric arguments
    start_seconds = _as_number(start_seconds, float, "start_seconds")
    duration_seconds = _as_number(duration_seconds, float, "duration_seconds")
    start_frame = _as_number(start_frame, int, "start_frame")
    num_frames = _as_number(num_frames, int, "num_frames")
    fps = _as_number(fps, Fraction, "fps")

    # A count or an offset outside its domain is refused rather than handed to
    # Python's slice semantics, which answered a negative 'num_frames' with
    # the track minus its last N frames and called it a success (#139).
    # validate_workflow refuses a literal one for free; this is the same
    # refusal for a value that arrived from a variable or an earlier step
    check_arguments(
        "slice_audio",
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        start_frame=start_frame,
        num_frames=num_frames,
        fps=fps,
        sample_rate=sample_rate,
    )

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
        if num_frames is None:
            length = max(total - start, 0)
        else:
            # Round the end frame directly rather than adding two separately
            # rounded halves - start's and num_frames' - which can each round
            # down half a sample and together land one sample short of a
            # slice meant to reach the source's exact end (#557)
            end = frames_to_samples((start_frame or 0) + num_frames, fps, sample_rate)
            length = end - start
    else:
        raise ValueError(
            "slice_audio needs either 'start_seconds'/'duration_seconds' or "
            "'start_frame'/'num_frames'/'fps'"
        )

    _warn_on_slice_past_end(total, start, length, sample_rate)
    _warn_on_slice_trims_tail(waveform, total, start, length, sample_rate)
    if start:
        emit_log(
            f"slice_audio: starts {start / sample_rate:.2f} s in, "
            f"{length / sample_rate:.2f} s long",
            command="slice_audio",
            start_seconds=round(start / sample_rate, 3),
            seconds=round(length / sample_rate, 3),
            start_frame=start_frame,
        )
    # #309: a cut out of a source that was already near-silent (room tone,
    # a deliberate quiet bed) is not a defect the slice introduced - measure
    # the source before cutting it down, so save can tell the two apart from
    # a track that arrived at a normal level and something upstream lost
    source_mean_dbfs = level_dbfs(waveform, "rms")
    return _as_track(
        slice_samples(waveform, start, length),
        sample_rate,
        "slice_audio",
        source_mean_dbfs=source_mean_dbfs,
    )


def gain_audio(
    audio,
    gain_db,
    start_seconds=None,
    duration_seconds=None,
    start_frame=None,
    num_frames=None,
    fps=None,
    sample_rate=None,
):
    """Task command: apply a gain to a region of an audio track.

    The region is addressed the same way slice_audio's is - either in
    seconds (start_seconds + duration_seconds) or in video frames
    (start_frame + num_frames + fps). Everything outside the region is
    passed through unchanged, so ducking a scene under another is one step
    rather than the slice/gain/mix/rejoin/pair_audio chain that was
    previously the only way to apply a gain to part of a track rather than
    all of it (#187). With no region given at all, the gain applies to the
    whole track - the same "no region means everything" reading mix_audio's
    gains use, and the obvious meaning of "duck this clip by 8 dB" (#395).
    To gain everything from some point on, give just start_seconds=0 (or
    start_frame=0 + fps) and leave duration_seconds/num_frames unset, which
    runs to the end of the track without the caller needing to already know
    how long that is.

    Unlike slice_audio, a region reaching past the end of the track is
    clipped to it rather than zero-padded: there is no silence there to
    gain, only the end of the real material.

    A file's or video's own sample rate is read automatically; sample_rate
    is for a waveform passed directly, or to override what a file carries -
    which relabels the waveform at that rate rather than resampling it, the
    same caveat slice_audio's sample_rate carries (#180).

    Args:
        audio: Path or URL of an audio file (or of a video file, whose
            soundtrack is taken), a generated video carrying its own
            soundtrack, or a waveform (which needs sample_rate alongside it)
        gain_db: Gain to apply within the region, in decibels - negative
            ducks it, positive boosts it
        start_seconds: Start of the region, in seconds. Omitted along with
            every other region argument, the gain applies to the whole track
        duration_seconds: Length of the region, in seconds
        start_frame: Start of the region, in video frames
        num_frames: Length of the region, in video frames
        fps: Frame rate used to convert start_frame/num_frames to samples
        sample_rate: Sample rate of a waveform passed directly; given for a
            file or a video it overrides the rate they carry

    Returns:
        An AudioTrack holding the whole track with the region's gain
        applied, and the rate it is at
    """
    start_seconds = _as_number(
        start_seconds, float, "start_seconds", command="gain_audio"
    )
    duration_seconds = _as_number(
        duration_seconds, float, "duration_seconds", command="gain_audio"
    )
    start_frame = _as_number(start_frame, int, "start_frame", command="gain_audio")
    num_frames = _as_number(num_frames, int, "num_frames", command="gain_audio")
    fps = _as_number(fps, Fraction, "fps", command="gain_audio")
    gain_db = _as_number(gain_db, float, "gain_db", command="gain_audio")

    check_arguments(
        "gain_audio",
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        start_frame=start_frame,
        num_frames=num_frames,
        fps=fps,
        sample_rate=sample_rate,
    )

    waveform, sample_rate = _waveform_and_rate(audio, sample_rate, "gain_audio")
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
            raise ValueError("gain_audio needs 'fps' to address a region in frames")
        start = frames_to_samples(start_frame or 0, fps, sample_rate)
        length = (
            max(total - start, 0)
            if num_frames is None
            else frames_to_samples(num_frames, fps, sample_rate)
        )
    else:
        start = 0
        length = total

    region_start = max(0, min(start, total))
    region_end = max(region_start, min(start + max(length, 0), total))

    gained = waveform.copy()
    if region_end > region_start:
        gain = 10 ** (gain_db / 20)
        gained[:, region_start:region_end] = (
            gained[:, region_start:region_end] * gain
        ).astype(waveform.dtype)

    region_start_seconds = region_start / float(sample_rate)
    region_end_seconds = region_end / float(sample_rate)
    emit_log(
        f"gain_audio: {gain_db:.1f} dB over "
        f"{region_start_seconds:.3f}-{region_end_seconds:.3f} s "
        f"(samples {region_start}-{region_end} @ {sample_rate} Hz)",
        command="gain_audio",
        gain_db=gain_db,
        start_seconds=region_start_seconds,
        duration_seconds=region_end_seconds - region_start_seconds,
        start_sample=region_start,
        end_sample=region_end,
        sample_rate=sample_rate,
    )

    return _as_track(gained, sample_rate, "gain_audio")


def _warn_on_slice_past_end(total, start, length, sample_rate):
    """Say when a slice asked for more material than its source holds.

    slice_samples zero-pads the shortfall, which is what makes frame-aligned
    chunking near the end of a track work at all - but the same padding is
    how a score shorter than the film it is laid under leaves the film
    unscored for the rest of its length, with nothing anywhere saying so
    (#126). emit_warning rather than logger.warning for the reason the
    concat_videos resample warning is emitted: silently substituting silence
    for four fifths of a track is an audio decision made on the caller's
    behalf, and a caller reading the job over the API or MCP sees the
    warnings list and nothing else (#82, #108).
    """
    # Frame-aligned slicing lands a sample or two past the end routinely;
    # that is rounding, not a decision anyone can act on, and slice_padding
    # answers None for it - the same rule validate's slice check applies
    padded_seconds = slice_padding(total, start, length, sample_rate)
    if padded_seconds is None:
        return
    emit_warning(
        f"slice_audio: the requested slice runs "
        f"{padded_seconds:.2f} s past the end of a "
        f"{total / float(sample_rate):.2f} s source, so that much of the "
        f"{length / float(sample_rate):.2f} s returned is digital silence. "
        f"If you meant to fill a cut of this length, make a bed with the "
        f"'loop_audio' task ('target_frames' + 'fps' matches one exactly) "
        f"and slice that; if you meant the tail pad, nothing is wrong.",
        kind="slice_past_end",
        command="slice_audio",
        source_seconds=round(total / float(sample_rate), 3),
        requested_seconds=round(length / float(sample_rate), 3),
        padded_seconds=round(padded_seconds, 3),
        sample_rate=sample_rate,
    )


def _warn_on_slice_trims_tail(waveform, total, start, length, sample_rate):
    """Say when a slice left material behind that the caller likely wanted.

    slice_audio is a slice, so most unused remainders are deliberate excerpts
    and warning on every one would be noise. What #342 found is a narrower
    signature: a cut landing a few seconds short of a source's natural end
    (a frame-lattice total that cannot land exactly on the score's length)
    silently drops the source's tail, including whatever is loudest there.
    Only fires when the dropped remainder is both short in absolute terms
    and small next to the slice itself, and only when that remainder is not
    already silence - a track that legitimately ends in a fade should not
    warn just because its last seconds are quiet.
    """
    if not sample_rate or length <= 0:
        return
    slice_end = start + length
    remainder = total - slice_end
    if remainder <= 0:
        return
    remainder_seconds = remainder / float(sample_rate)
    if remainder_seconds * 1000.0 < SLICE_TRIM_WARN_MIN_MS:
        return
    if remainder_seconds >= SLICE_TRIM_WARN_SECONDS:
        return
    if remainder_seconds / (length / float(sample_rate)) >= SLICE_TRIM_WARN_FRACTION:
        return
    dropped = waveform[:, slice_end:total]
    peak_dbfs = level_dbfs(dropped, "peak")
    if peak_dbfs is None:
        # No level at all is silence - nothing was lost
        return
    emit_warning(
        f"slice_audio: the slice ends {remainder_seconds:.2f} s before the "
        f"{total / float(sample_rate):.2f} s source does, dropping its tail "
        f"(peak {peak_dbfs:.1f} dBFS in the dropped {remainder_seconds:.2f} s) "
        f"- if the slice was meant to reach the source's end, adjust "
        f"start/length to land there, or fade the source's own tail first",
        kind="slice_trimmed_tail",
        command="slice_audio",
        source_seconds=round(total / float(sample_rate), 3),
        dropped_seconds=round(remainder_seconds, 3),
        dropped_peak_dbfs=round(peak_dbfs, 1),
        sample_rate=sample_rate,
    )


def resample_waveform(waveform, sample_rate, target_sample_rate):
    """A waveform at a different rate, as a plain (channels, samples) array.

    The conversion resample_audio performs, without the task's argument
    handling or its AudioTrack return, so a task that has waveforms in hand
    already can reach the rate conversion directly.
    """
    # PyAV's resampler accepts a zero rate and answers with the samples
    # unchanged, which is indistinguishable from a conversion that happened
    # (#140) - so neither rate is allowed to be one that cannot be a rate
    for name, rate in (("sample_rate", sample_rate), ("target", target_sample_rate)):
        if as_number(rate) is None or as_number(rate) <= 0:
            raise ValueError(
                f"resample_waveform needs a {name} above zero, got {rate!r}"
            )
    return dsp.resample_samples(waveform, sample_rate, target_sample_rate)


def resample_audio(audio, target_sample_rate, sample_rate=None):
    """Task command: resample an audio track to a different sample rate.

    MiniMax H3 conditions on audio at its audio VAE's own rate and resamples
    anything else with torchaudio, which dw does not depend on. Resampling a
    supplied recording once, up front, feeds the pipeline what it already wants
    and keeps the dependency out - PyAV, which dw needs for video anyway, does
    the conversion.

    Args:
        audio: Path or URL of an audio file (or of a video file, whose
            soundtrack is taken), a video generated with a
            soundtrack (which brings its sample rate along), or a waveform
            (which needs sample_rate alongside it)
        target_sample_rate: Rate to convert to
        sample_rate: Sample rate of a waveform passed directly; given for a
            file or a video it overrides the rate they carry

    Returns:
        An AudioTrack holding the resampled waveform and its new rate
    """
    target_sample_rate = _as_number(
        target_sample_rate, int, "target_sample_rate", "resample_audio"
    )
    # A zero or negative rate is not a rate. It used to reach PyAV's resampler,
    # which left the samples alone, and then the save, which fell back to
    # 44100 Hz - the original audio under a header 38% off, reported as a
    # success (#140)
    check_arguments(
        "resample_audio",
        target_sample_rate=target_sample_rate,
        sample_rate=sample_rate,
    )
    waveform, sample_rate = _waveform_and_rate(audio, sample_rate, "resample_audio")
    resampled = resample_waveform(waveform, sample_rate, target_sample_rate)
    noop = sample_rate == target_sample_rate
    emit_log(
        f"resample_audio: already {target_sample_rate} Hz, unchanged, "
        f"{resampled.shape[-1] / target_sample_rate:.2f} s"
        if noop
        else f"resample_audio: {sample_rate} → {target_sample_rate} Hz, "
        f"{resampled.shape[-1] / target_sample_rate:.2f} s",
        command="resample_audio",
        source_sample_rate=sample_rate,
        target_sample_rate=target_sample_rate,
        seconds=round(resampled.shape[-1] / target_sample_rate, 2),
    )
    return _as_track(
        resampled,
        target_sample_rate,
        "resample_audio",
    )


def _track_names(audios):
    """A name per audio track, for a warning that has to say which one.

    Mirrors concat_videos' video_names: a caller passes a path, or a
    previous step's result; only the path says anything by itself, so the
    rest are named by position.
    """
    return [
        original if isinstance(original, str) else f"track {index + 1}"
        for index, original in enumerate(audios)
    ]


def _load_tracks_matching_rate(audios, sample_rate, command):
    """Load a list of audio tracks, resampling any that disagree on rate.

    A mismatch among tracks that bring their own rate has no editorial
    meaning - the same reasoning concat_videos (#108) and dissolve_videos
    (#287) apply to shots - so when the caller has not pinned a rate the
    highest one found is chosen as the target and the rest are converted,
    with a warning naming each track's rate. When the caller *does* pin
    `sample_rate`, _waveform_and_rate's own per-track relabel warning
    (#180) still applies and this function changes nothing about it.
    """
    names = _track_names(audios)
    waveforms, native_rates, bare = [], [], False
    for audio in audios:
        if isinstance(audio, str) or hasattr(audio, "audio"):
            waveform, rate = _waveform_and_rate(audio, sample_rate, command)
            native_rates.append(rate)
        else:
            waveform, bare = as_channels_samples(audio), True
            native_rates.append(None)
        waveforms.append(waveform)

    if sample_rate is not None:
        return waveforms, sample_rate

    resolved = [rate for rate in native_rates if rate is not None]
    if bare or not resolved:
        raise ValueError(f"{command} needs 'sample_rate' with a raw waveform")
    target_rate = max(resolved)
    if len(set(resolved)) > 1:
        per_track = {
            name: rate for name, rate in zip(names, native_rates) if rate is not None
        }
        emit_warning(
            f"{command}: tracks carry audio at different sample rates ("
            + ", ".join(f"{name}: {rate} Hz" for name, rate in per_track.items())
            + f") - resampling them all to {target_rate} Hz. Pass "
            "'sample_rate' to pin a different target, or resample ahead of "
            "this step with the 'resample_audio' task.",
            kind="sample_rate_mismatch",
            command=command,
            sample_rate=target_rate,
            sample_rates=per_track,
        )
        waveforms = [
            (
                waveform
                if rate is None or rate == target_rate
                else resample_waveform(waveform, rate, target_rate)
            )
            for waveform, rate in zip(waveforms, native_rates)
        ]
    return waveforms, target_rate


def crossfade_audio(audios, crossfade_ms=75, sample_rate=None):
    """Task command: join audio tracks with an equal-power crossfade.

    Each seam overlaps the two tracks by the fade window, so the result is
    shorter than the plain sum by one window per seam.

    Args:
        audios: The tracks to join, in order - waveforms, audio or video file
            paths, or videos generated with a soundtrack
        crossfade_ms: Length of each crossfade
        sample_rate: Sample rate of the joined track. Required unless every
            track brings its own. Left unset, tracks at different rates are
            not a constraint - the highest rate found is used and the rest
            are resampled up to it, with a warning naming which (#108, #287,
            #293). Given here instead, it *relabels* rather than resamples
            any track whose real rate disagrees - changing its speed and
            pitch, not just its rate - which warns separately (#180); use
            resample_audio ahead of this step if conversion is what is
            wanted at a pinned rate

    Returns:
        An AudioTrack holding the joined waveform and its rate
    """
    if not isinstance(audios, list) or not audios:
        raise ValueError("crossfade_audio needs a non-empty list of audio tracks")
    waveforms, sample_rate = _load_tracks_matching_rate(
        audios, sample_rate, "crossfade_audio"
    )
    return _as_track(
        crossfade_concat(waveforms, sample_rate, crossfade_ms), sample_rate
    )


# #306: templates/assemble-and-score and templates/dissolve-between-shots
# both ship a stock world_gain of 1.8 - a deliberate multiplier, not a dB
# figure typed into the wrong unit - so the not-dB heuristic below has to sit
# above it
GAIN_LOOKS_LIKE_DB_ABOVE = 3.0


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
        audios: The tracks to layer - waveforms, audio or video file paths, or videos
            generated with a soundtrack
        gains: One plain multiplier per track, in the same order - not decibels.
            Defaults to unity on every track
        sample_rate: Sample rate of the joined mix. Required unless every
            track brings its own. Left unset, tracks at different rates are
            not a constraint - the highest rate found is used and the rest
            are resampled up to it, with a warning naming which (#108, #287,
            #293). Given here instead, it *relabels* rather than resamples
            any track whose real rate disagrees - changing its speed and
            pitch, not just its rate - which warns separately (#180); use
            resample_audio ahead of this step if conversion is what is
            wanted at a pinned rate

    Returns:
        An AudioTrack holding the mixed waveform and its rate
    """
    if not isinstance(audios, list) or not audios:
        raise ValueError("mix_audio needs a non-empty list of audio tracks")
    if gains is not None and len(gains) != len(audios):
        raise ValueError(
            f"mix_audio needs one gain per track - got {len(gains)} for "
            f"{len(audios)} tracks"
        )
    check_arguments("mix_audio", gains=gains, sample_rate=sample_rate)
    if gains is not None:
        # #306: a modest boost (a stock template's world_gain: 1.8 among them)
        # is a legitimate multiplier a caller chose on purpose, not a typo -
        # only a gain loud enough that a caller almost certainly meant it as
        # dB (12, 6, 20, ...) is worth flagging. GAIN_LOOKS_LIKE_DB_ABOVE sits
        # above any observed catalog default and below the smallest figure a
        # dB-as-multiplier typo would produce (a "6 dB" or "12 dB" boost).
        # #555: a hand-typed dB figure is a round number; a computed
        # multiplier (find_loop_bed's "gain", meant for this argument) almost
        # never lands on an exact integer, so only an integer value above the
        # threshold is flagged.
        loud = [
            g
            for g in gains
            if as_number(g) is not None
            and as_number(g) > GAIN_LOOKS_LIKE_DB_ABOVE
            and float(as_number(g)).is_integer()
        ]
        if loud:
            emit_warning(
                f"mix_audio: gain(s) {loud} are a multiplier, not decibels - "
                f"a value like 12, 6 or -3 is almost always a dB figure typed "
                f"into the wrong unit. A multiplier above 1 boosts the track; "
                f"convert a dB figure with 10 ** (db / 20) if that was intended.",
                kind="mix_audio_gain_not_db",
                command="mix_audio",
                gains=gains,
            )

    waveforms, sample_rate = _load_tracks_matching_rate(
        audios, sample_rate, "mix_audio"
    )

    waveforms = matched_channels(*waveforms)
    channels = waveforms[0].shape[0]
    length = max(waveform.shape[1] for waveform in waveforms)

    mixed = numpy.zeros((channels, length), dtype=numpy.float32)
    applied = []
    for index, waveform in enumerate(waveforms):
        gain = 1.0 if gains is None else float(gains[index])
        applied.append(gain)
        mixed[:, : waveform.shape[1]] += waveform * gain
    emit_log(
        f"mix_audio: {len(waveforms)} tracks, gains {applied}",
        command="mix_audio",
        gains=applied,
    )
    return _as_track(mixed, sample_rate, "mix_audio")


def loop_audio(
    audio,
    duration_seconds=None,
    target_frames=None,
    fps=None,
    crossfade_ms=250,
    sample_rate=None,
):
    """Task command: make a bed of a given length out of a short recording.

    A cut between two independently generated shots has a hole in it: each
    shot carries its own room, and nothing runs underneath the seam. A
    continuous bed laid under the whole cut is what fills it - the way a
    location's room tone is laid under a dialogue scene so the edits stop
    being audible - and a bed is made by looping a few seconds of tone to
    the length of the picture.

    Laps are joined with an equal-power crossfade rather than butted
    together, so the loop point itself is not a click. That only smooths the
    seam, though: a transient in the source (a hit, a swell) still recurs
    once per lap at full strength, so the loop still reads as a level pulse
    at the lap rate - measured at 9.3 dB on a source with one such transient.
    Picking a source with even internal level avoids the pulse; the
    crossfade does not. The source is used whole
    every lap; only the last one is trimmed, to land exactly on the
    requested length. A source longer than the request is trimmed to it.

    Args:
        audio: Path or URL of an audio file (or of a video file, whose
            soundtrack is taken), a video generated with a soundtrack, or a
            waveform (which needs sample_rate alongside it)
        duration_seconds: How long the bed should be, in seconds
        target_frames: How long the bed should be, in video frames - needs
            'fps', and is how a bed is matched to a cut exactly
        fps: Frame rate 'target_frames' is counted at
        crossfade_ms: Length of the crossfade at each loop point, clamped to
            the material available
        sample_rate: Sample rate of a waveform passed directly; given for a
            file or a video it overrides the rate they carry

    Returns:
        An AudioTrack holding the bed and the rate it is at
    """
    duration_seconds = _as_number(duration_seconds, float, "duration_seconds")
    target_frames = _as_number(target_frames, int, "target_frames")
    fps = _as_number(fps, Fraction, "fps")
    crossfade_ms = _as_number(crossfade_ms, float, "crossfade_ms")

    waveform, sample_rate = _waveform_and_rate(audio, sample_rate, "loop_audio")
    if waveform.size == 0:
        raise ValueError("loop_audio needs a source with samples in it")

    if duration_seconds is not None:
        length = int(round(duration_seconds * sample_rate))
    elif target_frames is not None:
        if fps is None:
            raise ValueError("loop_audio needs 'fps' to count a length in frames")
        length = frames_to_samples(target_frames, fps, sample_rate)
    else:
        raise ValueError(
            "loop_audio needs either 'duration_seconds' or 'target_frames'/'fps' "
            "to know how long a bed to make"
        )
    if length <= 0:
        raise ValueError(f"loop_audio needs a length above zero, got {length} samples")
    if crossfade_ms < 0:
        raise ValueError("loop_audio 'crossfade_ms' cannot be negative")

    window = min(int(crossfade_ms / 1000.0 * sample_rate), waveform.shape[1] // 2)
    bed = waveform
    laps = 1
    # Each lap after the first overlaps the one before it by the crossfade, so
    # a lap adds (source - window) samples rather than a whole source
    while bed.shape[1] < length:
        bed = crossfade_concat(
            [bed, waveform], sample_rate, window / sample_rate * 1000.0
        )
        laps += 1
    emit_log(
        f"loop_audio: {waveform.shape[1]} samples at {sample_rate}Hz looped "
        f"{laps}x to {length} samples ({length / sample_rate:.2f} s)",
        command="loop_audio",
        laps=laps,
        output_samples=length,
        output_seconds=round(length / sample_rate, 2),
    )
    return _as_track(bed[:, :length], sample_rate, "loop_audio")


# Shots generated independently land at whatever level the model chose, and
# joining two of them butts one loudness against another - the one seam
# artifact no fade can hide, because it is not at the seam, it is either side
# of it. These are the levels a matched join targets, and the spread at which
# an unmatched one is worth warning about
DEFAULT_MATCH_DBFS = {"peak": -1.0, "rms": -20.0}
# Matching to an rms target can ask for a gain that would clip; the peak is
# held here instead, which keeps a loud shot's relative level honest rather
# than squaring off its transients
MATCH_CEILING_DBFS = -0.5
LEVEL_SPREAD_WARN_DB = 6.0
# Mirrors result.py's NEAR_SILENT_WARN_DBFS: the same mean/rms level a job's
# own near-silent check treats as having no real content. Gaining an input
# already this quiet up to the target raises a noise floor rather than
# leveling a performance, and #434 found a +29.9 dB case that only reached
# the log, never job.warnings
MATCH_NEAR_SILENT_DBFS = -40.0
MATCH_LARGE_GAIN_WARN_DB = 20.0


def match_levels(waveforms, measure, target_dbfs=None, command="concat_videos"):
    """Scale each waveform so its level sits at one shared target.

    Returns a new list in the same order and shape; a None entry (a video
    with no soundtrack) and a silent track pass through untouched, since
    neither has a level to move. A gain that would push the peak past
    MATCH_CEILING_DBFS is held there and said so in the log - the shot is
    then quieter than the target rather than clipped.
    """
    if measure not in LEVEL_MEASURES:
        raise ValueError(
            f"{command} 'match_levels' must be one of {LEVEL_MEASURES}, got '{measure}'"
        )
    if target_dbfs is None:
        target_dbfs = DEFAULT_MATCH_DBFS[measure]
    if target_dbfs > 0:
        raise ValueError(
            f"{command} 'match_levels_dbfs' cannot be above full scale (0)"
        )

    matched = []
    for index, waveform in enumerate(waveforms):
        level = level_dbfs(waveform, measure)
        if level is None:
            matched.append(waveform)
            continue
        target_gain_db = target_dbfs - level
        gain_db = target_gain_db
        peak = level_dbfs(waveform, "peak")
        held = False
        if peak is not None and peak + gain_db > MATCH_CEILING_DBFS:
            gain_db = MATCH_CEILING_DBFS - peak
            held = True
            shortfall_db = target_gain_db - gain_db
            # emit_warning rather than logger.warning: a clip-held shot stays
            # off the target and the residual spread is exactly the level
            # jump match_levels exists to remove (#214) - a caller reading
            # the job's warnings list is the one who can act on it (#82)
            emit_warning(
                f"{command}: video {index + 1} would clip at the {measure} target "
                f"({peak + target_gain_db:+.1f} dBFS peak) - held to "
                f"{MATCH_CEILING_DBFS} dBFS, {shortfall_db:.1f} dB short of target",
                kind="match_levels_held",
                command=command,
                index=index,
                measure_dbfs=round(level, 1),
                target_dbfs=target_dbfs,
                gain_db=round(gain_db, 1),
                shortfall_db=round(shortfall_db, 1),
                ceiling_dbfs=MATCH_CEILING_DBFS,
            )
        elif level <= MATCH_NEAR_SILENT_DBFS or gain_db >= MATCH_LARGE_GAIN_WARN_DB:
            # The other end of the range `held` covers (#434): an input this
            # quiet is noise floor, not a performance at a lower level, and
            # matching it up to the target passes that noise off as content -
            # a consumer reading job.warnings sees nothing was wrong
            emit_warning(
                f"{command}: video {index + 1} {measure} {level:.1f} dBFS is "
                f"near-silent - matched up to the target with a {gain_db:+.1f} dB "
                "gain, raising its noise floor rather than leveling content",
                kind="match_levels_near_silent",
                command=command,
                index=index,
                measure_dbfs=round(level, 1),
                target_dbfs=target_dbfs,
                gain_db=round(gain_db, 1),
            )
        emit_log(
            f"{command}: video {index + 1} {measure} {level:.1f} dBFS, "
            f"gain {gain_db:+.1f} dB{' (held)' if held else ''}",
            index=index,
            measure_dbfs=round(level, 1),
            gain_db=round(gain_db, 1),
            held=held,
        )
        matched.append((waveform * (10 ** (gain_db / 20.0))).astype(numpy.float32))
    return matched


def warn_on_level_spread(waveforms, command="concat_videos", measure="rms"):
    """Say something when shots about to be joined are levels apart.

    Independently generated shots drift by 10 dB and more, and each one reads
    as fine on its own - it is only wrong relative to what it is cut against,
    and nothing else compares them.
    """
    levels = [level for level in (level_dbfs(w, measure) for w in waveforms) if level]
    if len(levels) < 2:
        return None
    spread = max(levels) - min(levels)
    if spread >= LEVEL_SPREAD_WARN_DB:
        # emit_warning rather than logger.warning: this is a property of the
        # file the run is about to write, and the caller reading the job is
        # the one who can act on it (#82)
        emit_warning(
            f"{command}: the tracks being joined span {spread:.1f} dB "
            f"({measure} {min(levels):.1f} to {max(levels):.1f} dBFS) - "
            "audible as a level jump unless the difference is intended (a "
            "shot written silent against the score). If it is not, pass "
            "match_levels to even them out",
            kind="level_spread",
            command=command,
            spread_db=round(spread, 1),
            measure=measure,
        )
    return spread


def _declick_join(previous, following, sample_rate, fade_ms=None, seam=None):
    """Butt-join two waveforms with a fade on each side of the seam.

    The default is the few milliseconds that keep a butt-join from clicking.
    A longer fade is a deliberate edit - the graceful hard cut you want when
    neither a crossfade nor a bleed applies.
    """
    ramp = int((DECLICK_MS if fade_ms is None else fade_ms) / 1000.0 * sample_rate)
    ramp = min(ramp, previous.shape[1], following.shape[1])
    if ramp > 0:
        fade_out, fade_in = equal_power_ramps(ramp)
        previous = previous.copy()
        following = following.copy()
        previous[:, -ramp:] *= fade_out  # cos: 1 down to ~0
        following[:, :ramp] *= fade_in  # sin: ~0 up to 1
    if fade_ms is not None:
        # A deliberate fade leaves a trace; the default declick is not asked for
        where = "a seam" if seam is None else f"seam {seam}"
        emit_log(
            f"seam_fade: {ramp / sample_rate * 1000:.0f} ms each side of {where}"
            f" (asked {fade_ms} ms)",
            command="seam_fade",
            seam=seam,
            fade_ms=round(ramp / sample_rate * 1000, 1),
            requested_ms=fade_ms,
        )
    return numpy.concatenate([previous, following], axis=1)


def fade_audio(audio, fade_in_ms=0, fade_out_ms=0, sample_rate=None):
    """Task command: fade a track in from silence and out to it.

    A slice cut out of the middle of a piece ends on whatever was sounding at
    the cut; a short fade turns that into an ending. The curve is the
    equal-power cosine the seam joins use, so a fade sounds like a fade and
    not a volume knob.

    Args:
        audio: Path or URL of an audio file (or of a video file, whose
            soundtrack is taken), a video generated with a
            soundtrack, or a waveform (which needs sample_rate alongside it)
        fade_in_ms: Length of the fade in, from the head of the track
        fade_out_ms: Length of the fade out, to the tail of the track
        sample_rate: Sample rate of a waveform passed directly

    Returns:
        An AudioTrack holding the faded waveform and its rate
    """
    waveform, sample_rate = _waveform_and_rate(audio, sample_rate, "fade_audio")
    if fade_in_ms < 0 or fade_out_ms < 0:
        raise ValueError("fade_audio fade lengths cannot be negative")
    faded = waveform.copy()
    length = faded.shape[1]

    fade_in = min(int(round(fade_in_ms / 1000 * sample_rate)), length)
    if fade_in:
        faded[:, :fade_in] *= fade_curve(fade_in)[::-1]
    fade_out = min(int(round(fade_out_ms / 1000 * sample_rate)), length)
    if fade_out:
        faded[:, length - fade_out :] *= fade_curve(fade_out)
    if fade_in or fade_out:
        emit_log(
            f"fade_audio: in {fade_in / sample_rate * 1000:.0f} ms, "
            f"out {fade_out / sample_rate * 1000:.0f} ms",
            command="fade_audio",
            fade_in_ms=round(fade_in / sample_rate * 1000, 1),
            fade_out_ms=round(fade_out / sample_rate * 1000, 1),
        )
    return _as_track(faded, sample_rate, "fade_audio")


def normalize_audio(
    audio, peak_dbfs=-1.0, target_lufs=None, limit=False, sample_rate=None
):
    """Task command: scale a track so its loudest sample sits at a level.

    Generated music comes out at whatever level the model happened to land
    on - quiet takes need lifting before they sit under a picture, and a
    hot one needs headroom before the encoder. Peak normalization changes
    nothing but the gain, so the dynamics survive.

    Args:
        audio: Path or URL of an audio file (or of a video file, whose
            soundtrack is taken), a video generated with a
            soundtrack, or a waveform (which needs sample_rate alongside it)
        peak_dbfs: The level the loudest sample is moved to, in dB below full
            scale. 0 is full scale; -1 leaves a little headroom. Still
            applies as a ceiling when target_lufs is also given
        target_lufs: Integrated loudness (BS.1770) to gain the track to, in
            LUFS. Peak alone says nothing about how loud a track sounds - a
            sparse voice-over and a dense score can share a peak and still
            sit tens of dB apart to the ear (#361). When given, the gain
            targets this loudness first; peak_dbfs still holds as a ceiling,
            and if reaching target_lufs would cross it the gain stops at the
            ceiling and a warning names the shortfall in LU. None (the
            default) leaves behavior exactly as peak-only
        limit: Hold peak_dbfs with a look-ahead limiter instead of capping the
            gain, so target_lufs can be reached past a transient that sets
            the peak. peak_dbfs becomes a true-peak (4x oversampled, BS.1770)
            ceiling; the limiter itself never adds gain, but the static gain
            applied before it is searched for - not read off target_lufs
            directly - since limiting takes back some of the loudness a
            plain gain would have reached, more on dense material, so the
            reported gain_db can run past target_lufs's own gain. Limiting
            stops at 12 dB of reduction, past which target_lufs is warned as
            capped. False (the default) leaves behavior exactly as without it
        sample_rate: Sample rate of a waveform passed directly

    Returns:
        An AudioTrack holding the scaled waveform and its rate; a silent
        track is returned unchanged
    """
    check_arguments("normalize_audio", sample_rate=sample_rate, target_lufs=target_lufs)
    if not isinstance(limit, (bool, numpy.bool_)):
        raise ValueError(
            f"normalize_audio 'limit' must be true or false, got {limit!r}"
        )
    waveform, sample_rate = _waveform_and_rate(audio, sample_rate, "normalize_audio")
    if peak_dbfs > 0:
        raise ValueError("normalize_audio 'peak_dbfs' cannot be above full scale (0)")
    peak = float(numpy.abs(waveform).max()) if waveform.size else 0.0
    if peak == 0.0:
        logger.warning("normalize_audio: the track is silent - left unchanged")
        return _as_track(waveform, sample_rate, "normalize_audio")
    if limit:
        return _as_track(
            _normalize_limited(waveform, sample_rate, peak_dbfs, target_lufs),
            sample_rate,
            "normalize_audio",
        )

    peak_db = 20 * numpy.log10(peak)
    measured_lufs = None
    if target_lufs is None:
        constraint = "peak_dbfs"
        gain_db = peak_dbfs - peak_db
    else:
        ceiling_gain_db = peak_dbfs - peak_db
        current_lufs = integrated_lufs(waveform.T, sample_rate)
        measured_lufs = current_lufs
        if current_lufs is None:
            emit_warning(
                f"normalize_audio: target_lufs={target_lufs} was given, but the "
                "track's loudness could not be measured (shorter than the 400 ms "
                "gating block, or silent throughout) - falling back to peak_dbfs "
                "alone.",
                kind="target_lufs_unmeasurable",
                command="normalize_audio",
                target_lufs=target_lufs,
            )
            gain_db = ceiling_gain_db
            constraint = "peak_ceiling"
        else:
            target_gain_db = target_lufs - current_lufs
            gain_db = min(target_gain_db, ceiling_gain_db)
            constraint = "peak_ceiling" if gain_db < target_gain_db else "target_lufs"
            if gain_db < target_gain_db:
                emit_warning(
                    f"normalize_audio: target_lufs={target_lufs} would need "
                    f"{target_gain_db:+.1f} dB of gain, but peak_dbfs={peak_dbfs} "
                    f"caps the gain at {gain_db:+.1f} dB - lands "
                    f"{target_gain_db - gain_db:.1f} LU below the target.",
                    kind="target_lufs_capped",
                    command="normalize_audio",
                    target_lufs=target_lufs,
                    peak_dbfs=peak_dbfs,
                    shortfall_lu=target_gain_db - gain_db,
                )
    gain = 10 ** (gain_db / 20)
    emit_log(
        f"normalize_audio: measured {peak_db:.1f} dBFS peak"
        + ("" if measured_lufs is None else f", {measured_lufs:.1f} LUFS")
        + f" -> gain {gain_db:+.1f} dB, set by {constraint}",
        command="normalize_audio",
        measured_peak_dbfs=round(peak_db, 1),
        measured_lufs=round(measured_lufs, 1) if measured_lufs is not None else None,
        gain_db=round(gain_db, 1),
        constraint=constraint,
    )
    return _as_track(
        (waveform * gain).astype(numpy.float32), sample_rate, "normalize_audio"
    )


def _normalize_limited(waveform, sample_rate, peak_dbfs, target_lufs):
    """normalize_audio(limit=True): a static gain, applied uncapped, with a
    true-peak limiter holding peak_dbfs. The limiter is never a gain stage -
    it only tames what the static gain sends it - but that static gain is
    searched for so the *limited* output lands on target_lufs, since
    limiting takes back some of the loudness a plain target_lufs gain would
    have reached; the search stops at LIMITER_MAX_REDUCTION_DB, past which
    the gain is what stops instead."""
    ceiling = 10 ** (peak_dbfs / 20)
    envelope = true_peak_envelope(waveform)
    input_peak_db = 20 * numpy.log10(float(envelope.max()))
    peak_gain_db = peak_dbfs - input_peak_db
    most_gain_db = peak_gain_db + LIMITER_MAX_REDUCTION_DB

    measured_lufs = None
    target_gain_db = None
    if target_lufs is not None:
        measured_lufs = integrated_lufs(waveform.T, sample_rate)
        if measured_lufs is None:
            emit_warning(
                f"normalize_audio: target_lufs={target_lufs} was given, but the "
                "track's loudness could not be measured (shorter than the 400 ms "
                "gating block, or silent throughout) - falling back to peak_dbfs "
                "alone.",
                kind="target_lufs_unmeasurable",
                command="normalize_audio",
                target_lufs=target_lufs,
            )
        else:
            target_gain_db = target_lufs - measured_lufs

    if target_gain_db is None:
        gain_db = peak_gain_db
        output, curve, trim, output_peak = limit_at(
            waveform, envelope, gain_db, ceiling, sample_rate
        )
        output_lufs = integrated_lufs(output.T, sample_rate)
    else:
        gain_db, (output, curve, trim, output_peak), output_lufs = search_gain(
            waveform,
            envelope,
            ceiling,
            sample_rate,
            target_lufs,
            min(target_gain_db, most_gain_db),
            most_gain_db,
        )

    lowest = (1.0 if curve is None else float(curve.min())) * trim
    max_reduction_db = max(0.0, -20 * numpy.log10(lowest))
    if curve is None:
        limited_fraction = 1.0 if trim < 1.0 else 0.0
    else:
        limited_fraction = float(
            numpy.count_nonzero(curve * trim < 1.0 - UNITY_TOLERANCE) / curve.size
        )
    output_peak_db = 20 * numpy.log10(output_peak)

    shortfall = None
    if target_gain_db is not None:
        if output_lufs is not None:
            shortfall = target_lufs - output_lufs
        elif gain_db < target_gain_db:
            shortfall = target_gain_db - gain_db
    if shortfall is not None and shortfall > LIMITER_TOLERANCE_LU:
        reason = (
            f"would need more than {LIMITER_MAX_REDUCTION_DB:.0f} dB of limiting "
            f"under peak_dbfs={peak_dbfs}, so the gain stops at {gain_db:+.1f} dB"
            if gain_db >= most_gain_db - 1e-9
            else f"was not reached under peak_dbfs={peak_dbfs} at "
            f"{gain_db:+.1f} dB of gain"
        )
        emit_warning(
            f"normalize_audio: target_lufs={target_lufs} {reason} - lands "
            f"{shortfall:.1f} LU below the target.",
            kind="target_lufs_capped",
            command="normalize_audio",
            target_lufs=target_lufs,
            peak_dbfs=peak_dbfs,
            shortfall_lu=round(shortfall, 2),
            limited=True,
        )
    if max_reduction_db > LIMITER_HEAVY_DB:
        emit_warning(
            f"normalize_audio: the limiter reduced the loudest moments by "
            f"{max_reduction_db:.1f} dB to hold peak_dbfs={peak_dbfs} - past "
            f"{LIMITER_HEAVY_DB:.0f} dB, pumping can be audible. A lower "
            "target_lufs asks less of it.",
            kind="limiter_heavy",
            command="normalize_audio",
            max_gain_reduction_db=round(max_reduction_db, 1),
            peak_dbfs=peak_dbfs,
            target_lufs=target_lufs,
        )
    emit_log(
        f"normalize_audio: measured {input_peak_db:.1f} dBTP"
        + ("" if measured_lufs is None else f", {measured_lufs:.1f} LUFS")
        + f" -> gain {gain_db:+.1f} dB, limiter up to {max_reduction_db:.1f} dB "
        f"on {limited_fraction:.1%} of samples -> {output_peak_db:.1f} dBTP"
        + ("" if output_lufs is None else f", {output_lufs:.1f} LUFS"),
        command="normalize_audio",
        measured_true_peak_dbfs=round(input_peak_db, 1),
        measured_lufs=round(measured_lufs, 1) if measured_lufs is not None else None,
        gain_db=round(gain_db, 1),
        constraint="limiter",
        max_gain_reduction_db=round(max_reduction_db, 2),
        limited_fraction=round(limited_fraction, 4),
        output_true_peak_dbfs=round(output_peak_db, 2),
        output_lufs=round(output_lufs, 1) if output_lufs is not None else None,
    )
    return output


def _waveform_and_rate(audio, sample_rate, command):
    """A command's audio argument as a (channels, samples) array with its rate.

    A path loads with the file's own rate; a video generated with a soundtrack
    (an AudioVideo, or anything carrying `.audio`) contributes that track and
    its rate; a bare waveform needs the rate given. A given rate always wins -
    correct for a raw waveform, which carries none of its own, but for a named
    source (a file or a video) a rate that disagrees with the one it actually
    carries relabels the samples rather than converting them, changing speed
    and pitch with nothing saying so (#180) - so that case warns.
    """
    if isinstance(audio, str):
        waveform, file_rate = load_audio(audio)
        if (
            sample_rate is not None
            and file_rate is not None
            and sample_rate != file_rate
        ):
            warn_on_rate_override(command, file_rate, sample_rate)
        return waveform, sample_rate if sample_rate is not None else file_rate
    if hasattr(audio, "audio"):
        if audio.audio is None:
            raise ValueError(
                f"{command} needs an audio track - the video it was given carries none"
            )
        if (
            sample_rate is not None
            and audio.sample_rate is not None
            and sample_rate != audio.sample_rate
        ):
            warn_on_rate_override(command, audio.sample_rate, sample_rate)
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


COMPRESS_MODES = ("compress", "limit", "gate")

# A floor below which an envelope is treated as digital silence, so its dBFS
# reading is a large negative number rather than -inf
_ENVELOPE_FLOOR_DBFS = -120.0
_ENVELOPE_FLOOR_LINEAR = 10.0 ** (_ENVELOPE_FLOOR_DBFS / 20.0)


def compress_audio(
    audio,
    threshold_dbfs,
    ratio=4.0,
    attack_ms=10.0,
    release_ms=100.0,
    mode="compress",
    sample_rate=None,
):
    """Task command: shape a track's dynamics with an envelope-follower.

    A compressor, a limiter and a gate are the same envelope-follower
    algorithm with different knob settings: a limiter is a ratio pushed
    toward infinity with a fast attack, and a gate is downward expansion
    below the threshold rather than compression above it - so one command
    covers all three through 'mode' rather than three near-duplicate ones.

    Args:
        audio: Path or URL of an audio file (or of a video file, whose
            soundtrack is taken), a video generated with a
            soundtrack, or a waveform (which needs sample_rate alongside it)
        threshold_dbfs: The level, in dB below full scale, above which
            'compress'/'limit' reduce gain, or below which 'gate' does
        ratio: How strongly gain is reduced past the threshold. Unused by
            'limit', which reduces enough to hold the signal at the
            threshold regardless
        attack_ms: How fast the envelope follows a rise in level
        release_ms: How fast the envelope follows a fall in level
        mode: 'compress' (downward compression above threshold), 'limit'
            (holds the signal at threshold), or 'gate' (downward expansion
            below threshold)
        sample_rate: Sample rate of a waveform passed directly

    Returns:
        An AudioTrack holding the processed waveform and its rate
    """
    waveform, sample_rate = _waveform_and_rate(audio, sample_rate, "compress_audio")
    check_arguments(
        "compress_audio",
        ratio=ratio,
        attack_ms=attack_ms,
        release_ms=release_ms,
        sample_rate=sample_rate,
    )
    if mode not in COMPRESS_MODES:
        raise ValueError(
            f"compress_audio mode must be one of {COMPRESS_MODES}, got {mode!r}"
        )
    if threshold_dbfs > 0:
        raise ValueError(
            "compress_audio 'threshold_dbfs' cannot be above full scale (0)"
        )
    if waveform.size == 0:
        return _as_track(waveform, sample_rate, "compress_audio")

    envelope = follow_envelope(waveform, sample_rate, attack_ms, release_ms)
    envelope_dbfs = 20.0 * numpy.log10(numpy.maximum(envelope, _ENVELOPE_FLOOR_LINEAR))

    if mode == "gate":
        past_threshold = numpy.maximum(0.0, threshold_dbfs - envelope_dbfs)
    else:
        past_threshold = numpy.maximum(0.0, envelope_dbfs - threshold_dbfs)

    if mode == "limit":
        reduction_db = past_threshold
    else:
        reduction_db = past_threshold * (1.0 - 1.0 / ratio)

    gain = (10.0 ** (-reduction_db / 20.0)).astype(numpy.float32)
    processed = (waveform * gain[numpy.newaxis, :]).astype(numpy.float32)
    return _as_track(processed, sample_rate, "compress_audio")


FILTER_KINDS = ("lowpass", "highpass", "bandpass", "notch")


def filter_audio(audio, cutoff_hz, kind="lowpass", q=0.707, sample_rate=None):
    """Task command: run a track through a single biquad filter stage.

    Args:
        audio: Path or URL of an audio file (or of a video file, whose
            soundtrack is taken), a video generated with a
            soundtrack, or a waveform (which needs sample_rate alongside it)
        cutoff_hz: The filter's corner (lowpass/highpass) or center
            (bandpass/notch) frequency
        kind: 'lowpass', 'highpass', 'bandpass', or 'notch'
        q: Resonance/bandwidth of the filter. Higher narrows a bandpass or
            notch, and peaks the corner of a lowpass or highpass
        sample_rate: Sample rate of a waveform passed directly

    Returns:
        An AudioTrack holding the filtered waveform and its rate
    """
    waveform, sample_rate = _waveform_and_rate(audio, sample_rate, "filter_audio")
    check_arguments("filter_audio", cutoff_hz=cutoff_hz, sample_rate=sample_rate)
    if kind not in FILTER_KINDS:
        raise ValueError(
            f"filter_audio kind must be one of {FILTER_KINDS}, got {kind!r}"
        )
    if q <= 0:
        raise ValueError("filter_audio 'q' must be above zero")
    nyquist = sample_rate / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(
            f"filter_audio 'cutoff_hz' ({cutoff_hz}) must be below the "
            f"Nyquist frequency ({nyquist}) for sample_rate {sample_rate}"
        )
    if waveform.size == 0:
        return _as_track(waveform, sample_rate, "filter_audio")

    b, a = biquad_coefficients(kind, cutoff_hz, q, sample_rate)
    filtered = numpy.stack(
        [apply_biquad(channel, b, a) for channel in waveform]
    ).astype(numpy.float32)
    return _as_track(filtered, sample_rate, "filter_audio")


def analyze_audio(audio, sample_rate=None):
    """Task command: measure a track without changing it.

    Read-only: the waveform passes through unmodified, and what comes back
    is diagnostics rather than an AudioTrack, since there is no processed
    track to hand a later step. Meant to feed a decision earlier in a
    workflow (whether 'compress_audio' or 'filter_audio' is needed, and
    with what settings) rather than to sit in the middle of a chain.

    Args:
        audio: Path or URL of an audio file (or of a video file, whose
            soundtrack is taken), a video generated with a
            soundtrack, or a waveform (which needs sample_rate alongside it)
        sample_rate: Sample rate of a waveform passed directly

    Returns:
        A dict: peak_dbfs, rms_dbfs, crest_factor_db (peak minus rms), and
        a rough low_dbfs/mid_dbfs/high_dbfs spectral-balance reading whose
        three bands are shares of the same power that gives rms_dbfs, so
        they sit on that scale rather than tens of dB under it. Any value
        is None where a silent track leaves it undefined.
    """
    waveform, sample_rate = _waveform_and_rate(audio, sample_rate, "analyze_audio")
    check_arguments("analyze_audio", sample_rate=sample_rate)

    peak_dbfs = level_dbfs(waveform, measure="peak")
    rms_dbfs = level_dbfs(waveform, measure="rms")
    crest_factor_db = (
        peak_dbfs - rms_dbfs if peak_dbfs is not None and rms_dbfs is not None else None
    )
    bands = spectral_balance(waveform, sample_rate)
    return {
        "peak_dbfs": peak_dbfs,
        "rms_dbfs": rms_dbfs,
        "crest_factor_db": crest_factor_db,
        **bands,
    }
