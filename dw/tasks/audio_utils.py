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

from ..dsp import (
    as_channels_samples,
    fade_curve,
    level_dbfs,
    matched_channels,
    resample_waveform,
    slice_samples,
)
from ..events import emit_log, emit_warning
from ..media_types import AudioTrack, warn_on_rate_override
from ..task_domains import (
    as_number,
    check_arguments,
    frames_to_samples,
    slice_padding,
    slice_region,
)
from .joins import crossfade_concat
from ..security import (
    validate_file_extension,
    ALLOWED_AUDIO_EXTENSIONS,
)

logger = logging.getLogger("dw")


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


def as_track(waveform, sample_rate, command="an audio task", source_mean_dbfs=None):
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


def coerce_number(value, kind, name, command="slice_audio"):
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
    start_seconds = coerce_number(start_seconds, float, "start_seconds")
    duration_seconds = coerce_number(duration_seconds, float, "duration_seconds")
    start_frame = coerce_number(start_frame, int, "start_frame")
    num_frames = coerce_number(num_frames, int, "num_frames")
    fps = coerce_number(fps, Fraction, "fps")

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

    waveform, sample_rate = waveform_and_rate(audio, sample_rate, "slice_audio")
    total = waveform.shape[1]

    region = slice_region(
        sample_rate,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        start_frame=start_frame,
        num_frames=num_frames,
        fps=fps,
        total=total,
    )
    if region is None:
        in_seconds = start_seconds is not None or duration_seconds is not None
        if not in_seconds and (start_frame is not None or num_frames is not None):
            raise ValueError("slice_audio needs 'fps' to address a slice in frames")
        raise ValueError(
            "slice_audio needs either 'start_seconds'/'duration_seconds' or "
            "'start_frame'/'num_frames'/'fps'"
        )
    start, length = region

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
    return as_track(
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
    start_seconds = coerce_number(
        start_seconds, float, "start_seconds", command="gain_audio"
    )
    duration_seconds = coerce_number(
        duration_seconds, float, "duration_seconds", command="gain_audio"
    )
    start_frame = coerce_number(start_frame, int, "start_frame", command="gain_audio")
    num_frames = coerce_number(num_frames, int, "num_frames", command="gain_audio")
    fps = coerce_number(fps, Fraction, "fps", command="gain_audio")
    gain_db = coerce_number(gain_db, float, "gain_db", command="gain_audio")

    check_arguments(
        "gain_audio",
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        start_frame=start_frame,
        num_frames=num_frames,
        fps=fps,
        sample_rate=sample_rate,
    )

    waveform, sample_rate = waveform_and_rate(audio, sample_rate, "gain_audio")
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
        start, length = slice_region(
            sample_rate,
            start_frame=start_frame,
            num_frames=num_frames,
            fps=fps,
            total=total,
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

    return as_track(gained, sample_rate, "gain_audio")


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
    target_sample_rate = coerce_number(
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
    waveform, sample_rate = waveform_and_rate(audio, sample_rate, "resample_audio")
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
    return as_track(
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
    `sample_rate`, waveform_and_rate's own per-track relabel warning
    (#180) still applies and this function changes nothing about it.
    """
    names = _track_names(audios)
    waveforms, native_rates, bare = [], [], False
    for audio in audios:
        if isinstance(audio, str) or hasattr(audio, "audio"):
            waveform, rate = waveform_and_rate(audio, sample_rate, command)
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
    return as_track(crossfade_concat(waveforms, sample_rate, crossfade_ms), sample_rate)


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
    return as_track(mixed, sample_rate, "mix_audio")


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
    duration_seconds = coerce_number(duration_seconds, float, "duration_seconds")
    target_frames = coerce_number(target_frames, int, "target_frames")
    fps = coerce_number(fps, Fraction, "fps")
    crossfade_ms = coerce_number(crossfade_ms, float, "crossfade_ms")

    waveform, sample_rate = waveform_and_rate(audio, sample_rate, "loop_audio")
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
    return as_track(bed[:, :length], sample_rate, "loop_audio")


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
    waveform, sample_rate = waveform_and_rate(audio, sample_rate, "fade_audio")
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
    return as_track(faded, sample_rate, "fade_audio")


def waveform_and_rate(audio, sample_rate, command):
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
