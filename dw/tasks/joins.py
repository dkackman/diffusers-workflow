"""What the joining commands share: the shots' track plumbing and the seams.

concat_videos, dissolve_videos, join_into_song, crossfade_audio, loop_audio
and the segment chain all butt, bleed or crossfade waveforms and reconcile the
result with a frame grid; the pieces that do are here so none of them
carries its own copy. Waveforms are (channels, samples) float32 arrays, as in
audio_utils.
"""

import logging
import os

import numpy

from ..dsp import (
    LEVEL_MEASURES,
    as_channels_samples,
    equal_power_ramps,
    harmonicity,
    level_dbfs,
    matched_channels,
    spectral_flatness,
)
from ..events import emit_log, emit_warning
from ..task_domains import frames_to_samples

logger = logging.getLogger("dw")


def video_names(videos):
    """A name per video, for an error or a warning that has to say which one.

    A caller passes a path, or a previous step's result; only the path says
    anything by itself, so the rest are named by position - which is what a
    six-entry `shots` list needs to be actionable ("24000 then 32000" does
    not say which entry to fix). By the time this runs, an `asset:`/`output:`
    reference has already been resolved to its absolute path on this server
    (#390) - naming a shot by that path leaked server layout onto a consumer
    surface, so a path is trimmed to its file name, the one part that means
    anything off this box.
    """
    return [
        os.path.basename(original)
        if isinstance(original, str)
        else f"video {index + 1}"
        for index, original in enumerate(videos)
    ]


# A few milliseconds of fade applied on each side of a butt-joined seam so the
# discontinuity does not click
DECLICK_MS = 3.0


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
