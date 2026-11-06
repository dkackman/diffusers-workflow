"""The slice_audio task: cut a slice out of an audio track, by seconds or
by video frames, with the warnings a slice owes its caller - a slice past
the end of its source, a dropped tail that was likely wanted, and a
section's tail cut while still playing.

Split out of `audio_utils.py`, whose `as_track` and `waveform_and_rate`
this module reads; `audio_utils` never imports it.
"""

import numpy

from ..dsp import level_dbfs, slice_samples
from ..events import emit_log, emit_warning
from ..task_domains import (
    SLICE_PAST_END_CHOICES,
    check_arguments,
    real_number,
    slice_lead_problem,
    slice_padding,
    slice_region,
    whole_number,
)
from .audio_utils import as_track, waveform_and_rate


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

# A section's dropped tail (`past_end: "stop"`) is still playing when its
# RMS is within this many dB of the kept part's. A decay or a fade to
# silence sits well below it; a band that keeps going sits at about 0 dB
# (#1199 Task 22: Music 3 sections cut 40 s into an overrun measured there)
SLICE_LOUD_TAIL_DB = 10.0

# The short window `end_snap_seconds` measures RMS over: long enough that
# the gap between two drum hits does not read as a pause, short enough to
# find the 0.27-0.33 s dips between sung phrases the S6 bench round heard
SLICE_SNAP_WINDOW_SECONDS = 0.2


def slice_audio(
    audio,
    start_seconds=None,
    duration_seconds=None,
    start_frame=None,
    num_frames=None,
    fps=None,
    sample_rate=None,
    lead_frames=0,
    past_end="pad",
    end_snap_seconds=None,
):
    """Task command: cut a slice out of an audio track.

    The slice is addressed either in seconds (start_seconds + duration_seconds)
    or in video frames (start_frame + num_frames + fps).

    In the frame form 'lead_frames' (default 0) adds audio before the cut: the
    slice starts at start_frame - lead_frames and runs num_frames frames, so a
    later trim of the head can land exactly on start_frame. A lead-in reaching
    before the head of the track (start_frame - lead_frames below zero) is
    refused, as is 'lead_frames' together with the seconds form.

    A slice reaching past the end of the track is zero-padded to the length
    asked for - it does not fail and it is not shortened - and the padding is
    digital silence, so asking for more than the source holds returns a track
    that is partly empty. Anything past a few milliseconds of that is
    reported as a 'slice_past_end' warning on the job. To fill a cut longer
    than the recording, make a bed with the 'loop_audio' task first
    ('target_frames' + 'fps' matches one exactly) and slice that.

    'past_end' "stop" asks for the other answer: a slice reaching past the end
    is cut at the end, so a source shorter than the slice is kept at its own
    length with no padding (logged, not warned - it is what was asked for). A
    joined piece built from sections that may come back short wants this:
    padding would put silence at every join. The returned track carries the
    length asked for as `asked_seconds`, which fade_audio passes on and
    crossfade_audio reads to warn when the joined piece is shorter than its
    sections asked for. A slice that starts at or past the end has nothing
    to keep and is refused. A `"stop"` slice declares its source a section
    to be kept whole up to the length, not an excerpt, so one that drops a
    tail still carrying real signal (RMS within SLICE_LOUD_TAIL_DB of the
    kept part's) warns 'slice_dropped_loud_tail' whatever the tail's length:
    the section was cut while it was still playing.

    'end_snap_seconds' (off by default; 0 is off too) moves an end that cuts
    into the source back to the quietest point - the lowest RMS over a
    SLICE_SNAP_WINDOW_SECONDS window - within that many seconds before it,
    which in a song is a phrase gap or a bar break. The slice comes out that
    much shorter; the track carries the length asked for and how far the cut
    moved, so crossfade_audio does not report it as a section that ended
    early. An end the source does not reach has nothing to snap.

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
    command = "slice_audio"
    start_seconds = real_number(start_seconds, "start_seconds", command)
    duration_seconds = real_number(duration_seconds, "duration_seconds", command)
    start_frame = whole_number(start_frame, "start_frame", command)
    num_frames = whole_number(num_frames, "num_frames", command)
    fps = real_number(fps, "fps", command, exact=True)
    lead_frames = whole_number(lead_frames, "lead_frames", command)
    end_snap_seconds = real_number(end_snap_seconds, "end_snap_seconds", command)

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
        lead_frames=lead_frames,
        end_snap_seconds=end_snap_seconds,
    )
    _refuse_slice_shape(
        past_end, start_frame, lead_frames, start_seconds, duration_seconds
    )

    waveform, sample_rate = waveform_and_rate(audio, sample_rate, "slice_audio")
    total = waveform.shape[1]

    start, length = _region_or_refuse(
        sample_rate,
        total,
        start_seconds,
        duration_seconds,
        start_frame,
        num_frames,
        fps,
        lead_frames,
    )
    length, asked_seconds, snapped_seconds = _end_of_slice(
        waveform, total, start, length, sample_rate, past_end, end_snap_seconds
    )
    _warn_on_slice_past_end(total, start, length, sample_rate)
    _warn_on_slice_trims_tail(
        waveform,
        total,
        start,
        length,
        sample_rate,
        section=past_end == "stop",
        snapped_seconds=snapped_seconds,
    )
    if start:
        _log_slice_start(start, length, sample_rate, start_frame)
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
        asked_seconds=asked_seconds,
        snapped_seconds=snapped_seconds,
    )


def _refuse_slice_shape(
    past_end, start_frame, lead_frames, start_seconds, duration_seconds
):
    """Refuse an unknown `past_end` and a lead-in the slice cannot take."""
    if past_end not in SLICE_PAST_END_CHOICES:
        raise ValueError(
            f"slice_audio needs 'past_end' as one of "
            f"{list(SLICE_PAST_END_CHOICES)}, got {past_end!r}"
        )
    problem = slice_lead_problem(
        start_frame, lead_frames, start_seconds, duration_seconds
    )
    if problem:
        raise ValueError(problem)


def _log_slice_start(start, length, sample_rate, start_frame):
    """Say in the job log where a slice that does not start at 0 begins."""
    emit_log(
        f"slice_audio: starts {start / sample_rate:.2f} s in, "
        f"{length / sample_rate:.2f} s long",
        command="slice_audio",
        start_seconds=round(start / sample_rate, 3),
        seconds=round(length / sample_rate, 3),
        start_frame=start_frame,
    )


def _region_or_refuse(
    sample_rate,
    total,
    start_seconds,
    duration_seconds,
    start_frame,
    num_frames,
    fps,
    lead_frames,
):
    """(start, length) in samples of the slice asked for, refused when it is
    addressed neither in seconds nor in frames with a frame rate."""
    region = slice_region(
        sample_rate,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        start_frame=start_frame,
        num_frames=num_frames,
        fps=fps,
        total=total,
        lead_frames=lead_frames or 0,
    )
    if region is None:
        in_seconds = start_seconds is not None or duration_seconds is not None
        if not in_seconds and (start_frame is not None or num_frames is not None):
            raise ValueError("slice_audio needs 'fps' to address a slice in frames")
        raise ValueError(
            "slice_audio needs either 'start_seconds'/'duration_seconds' or "
            "'start_frame'/'num_frames'/'fps'"
        )
    return region


def _end_of_slice(
    waveform, total, start, length, sample_rate, past_end, end_snap_seconds
):
    """(length, asked_seconds, snapped_seconds): the slice's length once a
    `past_end: "stop"` cut and an `end_snap_seconds` move have been made,
    the length asked for when either changed it, and how far the snap moved
    the end (None when it did not)."""
    asked_seconds = None
    if past_end == "stop":
        asked_seconds = length / float(sample_rate)
        length = _stop_at_the_end(total, start, length, sample_rate)
    snapped_seconds = None
    if end_snap_seconds:
        snapped = _snap_end_to_quiet(
            waveform, total, start, length, sample_rate, end_snap_seconds
        )
        if snapped < length:
            if asked_seconds is None:
                asked_seconds = length / float(sample_rate)
            snapped_seconds = (length - snapped) / float(sample_rate)
            length = snapped
    return length, asked_seconds, snapped_seconds


def _snap_end_to_quiet(waveform, total, start, length, sample_rate, window_seconds):
    """The slice length with its end moved back to the quietest point in the
    last `window_seconds` before it - the centre of the SLICE_SNAP_WINDOW
    with the lowest RMS - or `length` unchanged when the end does not cut
    into the source or nothing in the window is quieter than the end itself.
    The latest of equally quiet points wins, so an even level keeps the
    exact cut."""
    end = start + length
    half = max(int(round(SLICE_SNAP_WINDOW_SECONDS * sample_rate / 2)), 1)
    if end >= total:
        return length
    lo = max(start + half, end - int(round(window_seconds * sample_rate)))
    hi = min(end, total - half)
    if hi <= lo:
        return length
    energy = numpy.mean(
        numpy.square(waveform[:, lo - half : hi + half], dtype=numpy.float64), axis=0
    )
    sums = numpy.concatenate(([0.0], numpy.cumsum(energy)))
    # sums[i + 2 * half] - sums[i] is the window centred at lo + i
    windows = sums[2 * half :] - sums[: -2 * half]
    quietest = windows.min()
    tolerance = max(abs(quietest) * 1e-6, 1e-12)
    cut = lo + int(numpy.flatnonzero(windows <= quietest + tolerance)[-1])
    if cut >= end:
        return length
    rms = numpy.sqrt(max(quietest, 0.0) / (2 * half))
    emit_log(
        f"slice_audio: the cut moved {(end - cut) / float(sample_rate):.2f} s "
        f"back, to the quietest point in the last {window_seconds:g} s before "
        f"{end / float(sample_rate):.2f} s (RMS "
        + (f"{20 * numpy.log10(rms):.1f} dBFS" if rms > 0 else "silence")
        + ")",
        command="slice_audio",
        seconds=round((cut - start) / float(sample_rate), 3),
        asked_seconds=round(length / float(sample_rate), 3),
        snapped_seconds=round((end - cut) / float(sample_rate), 3),
    )
    return cut - start


def _stop_at_the_end(total, start, length, sample_rate):
    """A `past_end: "stop"` slice's length: cut at the end of the source
    rather than padded past it, and logged when that keeps it short."""
    if start >= total:
        raise ValueError(
            f"slice_audio starts {start / float(sample_rate):.2f} s in, at or "
            f"past the end of a {total / float(sample_rate):.2f} s source, so "
            f"with 'past_end' \"stop\" there is nothing to keep"
        )
    kept = min(length, total - start)
    if kept < length:
        emit_log(
            f"slice_audio: the source ends {kept / float(sample_rate):.2f} s "
            f"in, so it is kept at that length of the "
            f"{length / float(sample_rate):.2f} s asked ('past_end' \"stop\")",
            command="slice_audio",
            seconds=round(kept / float(sample_rate), 3),
            asked_seconds=round(length / float(sample_rate), 3),
        )
    return kept


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


def _warn_on_slice_trims_tail(
    waveform, total, start, length, sample_rate, section=False, snapped_seconds=None
):
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
    if (
        remainder_seconds >= SLICE_TRIM_WARN_SECONDS
        or remainder_seconds / (length / float(sample_rate)) >= SLICE_TRIM_WARN_FRACTION
    ):
        if section:
            _warn_on_loud_tail(
                waveform, total, start, slice_end, sample_rate, snapped_seconds
            )
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


def _warn_on_loud_tail(
    waveform, total, start, slice_end, sample_rate, snapped_seconds=None
):
    """Say when a section's slice dropped a tail that is still playing.

    A `past_end: "stop"` slice keeps a section whole up to its length, so a
    long remainder is not a deliberate excerpt: it is a section that ran
    past the length and was cut while still playing - a Music 3 section
    40 s into its overrun, its last lines never sung (#1199 Task 22). The
    small-trim warning above only reached remainders under 5% and under
    10 s. Measured as RMS against the kept part's, so a decay or a fade to
    silence does not warn.

    What was playing is the dropped tail, not the cut: when `end_snap_seconds`
    moved the cut (`snapped_seconds` back from the asked length) it sits in
    the quietest window there, so the wording names that move and the quiet
    point rather than calling the cut mid-signal (#1199, the snap bench).
    """
    dropped_rms = level_dbfs(waveform[:, slice_end:total], "rms")
    kept_rms = level_dbfs(waveform[:, start:slice_end], "rms")
    if dropped_rms is None or kept_rms is None:
        return
    if dropped_rms < kept_rms - SLICE_LOUD_TAIL_DB:
        return
    dropped_seconds = (total - slice_end) / float(sample_rate)
    kept_seconds = (slice_end - start) / float(sample_rate)
    levels = f"(RMS {dropped_rms:.1f} dBFS dropped against {kept_rms:.1f} dBFS kept)"
    if snapped_seconds:
        cut = (
            f"'end_snap_seconds' moved the cut {snapped_seconds:.2f} s back from "
            f"{kept_seconds + snapped_seconds:.2f} s to a quiet point at "
            f"{kept_seconds:.2f} s, but the source runs {dropped_seconds:.2f} s "
            f"past that cut and is still playing there {levels}, so whatever "
            f"those {dropped_seconds:.2f} s held is lost. The snap keeps "
            f"nothing more"
        )
    else:
        cut = (
            f"the source runs {dropped_seconds:.2f} s past the slice and is "
            f"still playing there {levels}: the slice cut at "
            f"{kept_seconds:.2f} s while it was still playing, and whatever "
            f"the last {dropped_seconds:.2f} s held is lost. "
            f"'end_snap_seconds' moves the cut to a quiet point but keeps "
            f"nothing more"
        )
    emit_warning(
        f"slice_audio: {cut}. If the source was meant to end inside the "
        f"slice, give it less material (a shorter lyric) or a longer slice",
        kind="slice_dropped_loud_tail",
        command="slice_audio",
        source_seconds=round(total / float(sample_rate), 3),
        seconds=round(kept_seconds, 3),
        dropped_seconds=round(dropped_seconds, 3),
        dropped_rms_dbfs=round(dropped_rms, 1),
        kept_rms_dbfs=round(kept_rms, 1),
        snapped_seconds=round(snapped_seconds, 3) if snapped_seconds else None,
        sample_rate=sample_rate,
    )
