"""find_loop_bed: rank the windows of a recording worth looping into a bed (#218).

A dialogue cut needs a bed of room tone under it, and the bed is made by
looping a short stretch of the programme's own quiet (`slice_audio` ->
`loop_audio` -> `mix_audio`). Picking that stretch has three traps a level
check misses, each found on a real episode:

1. near-programme material - faint speech attenuated ~30 dB reads as quiet,
   and is audible once it repeats every lap;
2. lap-rate modulation - the loop's repeat rate beats against the source's
   own level movement, which one pass through the source cannot show;
3. ticks - a 1 ms transient is invisible to 50 ms RMS and recurs once per lap.

This searches a time range for windows that pass all three and measures each
survivor *looped*, the way `loop_audio` will make it, so the ranking is of the
bed rather than of the source. It picks a window and builds nothing: the
answer's `start_seconds`/`duration_seconds` are `slice_audio`'s arguments and
`gain` is `mix_audio`'s multiplier, so the remedy is a copy.

Like the assessment probes it answers JSON and decides nothing, but it is a
search rather than a check of a finished cut, so it is not listed among them.
"""

import logging
import math
import os
import types

import numpy

from .. import dsp
from ..task_domains import check_arguments
from .audio_utils import coerce_number, waveform_and_rate
from .joins import (
    HARMONICITY_THRESHOLD,
    TONAL_FLATNESS_THRESHOLD,
    crossfade_concat,
)

logger = logging.getLogger("dw")

COMMAND = "find_loop_bed"

# The envelope's resolution - the field method's, and the bin every level
# reading here is taken over
BIN_SECONDS = 0.05
# A tick's resolution: a 1 ms transient is what 50 ms RMS cannot see
PEAKS_PER_BIN = 50
# A looped candidate whose strongest envelope component sits above this
# (relative to the envelope's mean) carries `lap_modulation`. The two field
# winners read -21 and -19 dB
LAP_MODULATION_WARN_DB = -15.0
# How many non-overlapping survivors are looped and measured. Fixed rather
# than scaled by max_candidates, so asking for fewer candidates returns the
# head of the same ranking rather than a ranking of a smaller pool (#544)
LOOPED_POOL = 200
# The ripple spread, as percentiles of the 50 ms bins
RIPPLE_PERCENTILES = (5.0, 95.0)
# The band flatness is measured over: the energy an upsampled source's empty
# tail holds (band-limited interpolation leaves it ~60 dB and more down), and
# how much of the band must be occupied before the whole of it is used
OCCUPIED_BAND_FLOOR = 1e-6
OCCUPIED_BAND_FULL = 0.95
OCCUPIED_FRAME = 4096
# Tonality is judged per block, sliding on the 50 ms grid, and a window is
# tonal when any block inside it is: faint speech comes and goes, and a
# whole-window measurement averages a syllable into the pauses around it
# (#544). Two block lengths, because neither catches both: 0.1 s sits inside
# one syllable, where a longer block averages the pause beside it back under
# the threshold, and 0.2 s holds enough periods of a low hum that the
# autocorrelation's shrinking overlap does not read it below the threshold
TONAL_BLOCK_SECONDS = (0.1, 0.2)
# A tick this close outside a window still counts against it: a window
# ending on a click would put the click's edge at the loop's seam
TICK_GUARD_PEAKS = 5

REJECTION_RULES = ("too_loud", "silent", "spike", "tonal")


def _round(value, places=2):
    return None if value is None else round(float(value), places)


def _read_source(audio):
    """(waveform, rate, fps, carried shots, file path) for the `audio`
    argument. A local video file is read audio-only through media -
    load_audio would decode every frame of a cut to reach its soundtrack -
    and its frame rate comes off the container's headers. `fps` and the
    carried shots are an in-memory video's own; `path` is the validated file
    whose run manifest may record shots, None for anything in memory or
    remote."""
    from .video_utils import VideoFileReference

    if isinstance(audio, VideoFileReference):
        audio = audio.path
    if isinstance(audio, str) and not audio.startswith(("http://", "https://")):
        from ..locations import validate_media_path
        from ..security import ALLOWED_VIDEO_EXTENSIONS

        path = validate_media_path(audio, None, "an audio argument")
        extension = os.path.splitext(audio)[1].lower()
        if extension in ALLOWED_VIDEO_EXTENSIONS:
            from ..media import NoSoundtrack, container_fps, decode_soundtrack

            try:
                waveform, rate = decode_soundtrack(path)
            except NoSoundtrack:
                raise ValueError(
                    f"{COMMAND}: {audio} carries no audio track - a bed is "
                    "found in a soundtrack"
                ) from None
            return waveform, rate, container_fps(path), None, path
        waveform, rate = waveform_and_rate(audio, None, COMMAND)
        return waveform, rate, None, None, path
    waveform, rate = waveform_and_rate(audio, None, COMMAND)
    return (
        waveform,
        rate,
        getattr(audio, "fps", None),
        getattr(audio, "shots", None),
        None,
    )


def _resolve_shots(shots, carried, path):
    """The shots to keep candidates inside, and where they came from - the
    assessment probes' order (`resolve_shots`, dw/tasks/assess.py): the
    argument, then the shots the video carries, then the run manifest beside
    the file, else (None, None)."""
    if shots:
        return [dict(shot) for shot in shots], "argument"
    if carried:
        return [dict(shot) for shot in carried], "artifact"
    if path is not None:
        from ..runs import shots_beside

        recorded = shots_beside(path)
        if recorded:
            return [dict(shot) for shot in recorded], "manifest"
    return None, None


def _check_shots(shots):
    """Refuse a `shots` argument that is not a list of shot records."""
    if not isinstance(shots, (list, tuple)):
        raise ValueError(
            f"{COMMAND}: 'shots' is a list of {{name, start_frame, num_frames}} "
            f"records, not {type(shots).__name__}"
        )
    for index, shot in enumerate(shots):
        if not isinstance(shot, dict) or not shot.get("name"):
            raise ValueError(
                f"{COMMAND}: shots[{index}] needs a 'name' - a candidate names "
                "the shot it lies in"
            )
        for field in ("start_frame", "num_frames", "start_sample", "num_samples"):
            value = shot.get(field)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"{COMMAND}: shots[{index}].{field} must be a number, not {value!r}"
                )
            if value < 0 or (field.startswith("num_") and value <= 0):
                raise ValueError(
                    f"{COMMAND}: shots[{index}].{field} must be "
                    f"{'above' if field.startswith('num_') else 'at or above'} "
                    f"zero, not {value!r}"
                )


def _shot_spans(shots, fps, sample_rate):
    """Each shot's (name, first sample, end sample) in the source's samples.

    A shot is placed by its frames (at `fps`) and by its recorded samples,
    and where it has both it is the overlap of the two: the picture's cut
    and the soundtrack's can sit a few samples apart (a shot's audio
    overrun, dw/shots.py), and a window inside both crosses neither.
    """
    spans = []
    for index, shot in enumerate(shots):
        found = []
        start_frame, num_frames = shot.get("start_frame"), shot.get("num_frames")
        if fps and start_frame is not None and num_frames is not None:
            found.append(
                (
                    start_frame / fps * sample_rate,
                    (start_frame + num_frames) / fps * sample_rate,
                )
            )
        start_sample, num_samples = shot.get("start_sample"), shot.get("num_samples")
        if start_sample is not None and num_samples is not None:
            found.append((start_sample, start_sample + num_samples))
        if not found:
            raise ValueError(
                f"{COMMAND}: shot '{shot.get('name')}' (shots[{index}]) is placed "
                "by its frames and the source has no frame rate - pass 'fps', "
                "or give the shot start_sample/num_samples"
            )
        first = max(span[0] for span in found)
        end = min(span[1] for span in found)
        spans.append((str(shot["name"]), int(math.ceil(first)), int(end)))
    return spans


def _bin_shots(spans, offset, bin_length, count):
    """Which shot each 50 ms bin of the searched range lies wholly inside, as
    an index into `spans`; -1 for a bin a boundary cuts through or no shot
    covers."""
    owner = numpy.full(count, -1, dtype=numpy.int64)
    starts = offset + numpy.arange(count) * bin_length
    ends = starts + bin_length
    for index, (_, first, end) in enumerate(spans):
        inside = (starts >= first) & (ends <= end) & (owner < 0)
        owner[inside] = index
    return owner


def _sliding(array, length):
    """Every run of `length` consecutive rows of `array`, one per start."""
    return numpy.lib.stride_tricks.sliding_window_view(array, length, axis=0)


def _envelope(mono, sample_rate):
    """50 ms mean-square bins and each bin's 1 ms peaks, over whole bins.

    A bin is split into PEAKS_PER_BIN sub-blocks, the last padded with zeros
    when the bin does not divide evenly, so no sample escapes the peak search
    - a tick in a bin's last few samples is still a tick.
    """
    bin_length = max(int(round(sample_rate * BIN_SECONDS)), PEAKS_PER_BIN)
    count = mono.shape[0] // bin_length
    bins = mono[: count * bin_length].reshape(count, bin_length)
    power = numpy.mean(numpy.square(bins), axis=1)
    block = -(-bin_length // PEAKS_PER_BIN)
    padded = numpy.zeros((count, block * PEAKS_PER_BIN))
    padded[:, :bin_length] = numpy.abs(bins)
    peaks = padded.reshape(count, PEAKS_PER_BIN, block).max(axis=2)
    return bin_length, power, peaks


def _bin_db(power):
    return 10.0 * numpy.log10(numpy.maximum(power, 1e-20))


def _occupied_rate(mono, sample_rate):
    """The rate the material actually fills, read off its long-term spectrum:
    twice the frequency below which all but OCCUPIED_BAND_FLOOR of its energy
    sits, or None when that is the whole band.

    Resampling up leaves the band above the original Nyquist empty, and an
    empty band reads as tonal to spectral flatness whatever the material is
    (#198) - a 16 kHz bed mixed at 24 kHz read tonal in every window, and a
    voice under it filled the band and read as noise (#544). The source
    carries no record of its native rate once mixed, so it is measured.
    """
    frame = min(OCCUPIED_FRAME, mono.shape[0])
    frames = mono.shape[0] // frame
    if frames < 1:
        return None
    spectrum = numpy.mean(
        numpy.square(
            numpy.abs(
                numpy.fft.rfft(
                    mono[: frames * frame].reshape(frames, frame)
                    * numpy.hanning(frame),
                    axis=1,
                )
            )
        ),
        axis=0,
    )
    total = spectrum.sum()
    if total <= 0.0:
        return None
    above = numpy.cumsum(spectrum[::-1])[::-1] / total
    occupied = int(numpy.flatnonzero(above > OCCUPIED_BAND_FLOOR)[-1]) + 1
    if occupied >= OCCUPIED_BAND_FULL * spectrum.shape[0]:
        return None
    return sample_rate * occupied / (spectrum.shape[0] - 1)


def _tonal_blocks(
    searched, bin_length, sample_rate, block_bins, needed, native_rate=None
):
    """bleed_join's flatness/harmonicity test over every block of
    `block_bins` bins, one per starting bin: (flatness, harmonicity, tonal)
    arrays. Only blocks marked `needed` are measured - a block holding a loud
    or silent bin lies only inside windows already thrown out - and the rest
    read as noise (flatness 1, harmonicity 0), never as tonal."""
    count = needed.shape[0]
    flatness = numpy.ones(count)
    harmonicity = numpy.zeros(count)
    for block in numpy.flatnonzero(needed):
        segment = searched[block * bin_length : (block + block_bins) * bin_length][
            numpy.newaxis, :
        ]
        flatness[block] = dsp.spectral_flatness(segment, sample_rate, native_rate)
        harmonicity[block] = dsp.harmonicity(segment, sample_rate)
    tonal = (flatness < TONAL_FLATNESS_THRESHOLD) | (
        harmonicity >= HARMONICITY_THRESHOLD
    )
    return flatness, harmonicity, tonal


def _looped(segment, sample_rate, crossfade_ms, loop_seconds):
    """`segment` looped to `loop_seconds` exactly as loop_audio loops it, and
    the readings of the result: its 50 ms ripple and its envelope spectrum."""
    samples = segment.shape[0]
    window = min(int(crossfade_ms / 1000.0 * sample_rate), samples // 2)
    length = int(round(loop_seconds * sample_rate))
    lap = samples - window
    laps = 1 + max(0, -(-(length - samples) // lap))
    bed = crossfade_concat(
        [segment[numpy.newaxis, :]] * laps,
        sample_rate,
        window / sample_rate * 1000.0,
    )[0, :length]

    _, power, _ = _envelope(bed, sample_rate)
    low, high = numpy.percentile(_bin_db(power), RIPPLE_PERCENTILES)
    envelope = numpy.sqrt(power)
    mean = float(envelope.mean())
    lap_hz = sample_rate / lap
    if mean <= 0 or envelope.shape[0] < 4:
        return {
            "ripple_db": _round(high - low),
            "envelope_peak_db": None,
            "envelope_peak_hz": None,
            "lap_hz": _round(lap_hz, 3),
            "lap_component_db": None,
        }

    # Each component as the amplitude of the sinusoid it stands for, over
    # the envelope's mean: a level wobbling by a fraction m reads 20*log10(m)
    varying = envelope - mean
    count = varying.shape[0]
    spectrum = numpy.abs(numpy.fft.rfft(varying)) * 2.0 / count
    frequencies = numpy.fft.rfftfreq(count, BIN_SECONDS)
    strongest = 1 + int(numpy.argmax(spectrum[1:]))
    # The lap rate rarely lands on an FFT bin, so it is measured directly
    times = numpy.arange(count) * BIN_SECONDS
    at_lap = abs(numpy.sum(varying * numpy.exp(-2j * numpy.pi * lap_hz * times)))
    at_lap = at_lap * 2.0 / count
    return {
        "ripple_db": _round(high - low),
        "envelope_peak_db": _round(dsp.dbfs(spectrum[strongest] / mean)),
        "envelope_peak_hz": _round(frequencies[strongest], 3),
        "lap_hz": _round(lap_hz, 3),
        "lap_component_db": _round(dsp.dbfs(at_lap / mean)),
    }


def _empty_finding(rejected, criteria):
    """The one finding for a search that found nothing, naming the rule that
    threw the most out and the argument that relaxes it."""
    if not any(rejected.values()):
        message = "no window fit in the range; widen it or lower min_seconds"
        rule = None
    else:
        rule = max(rejected, key=lambda name: rejected[name])
        message = {
            "too_loud": (
                f"nothing below {criteria['max_bin_dbfs']:g} dBFS per 50 ms "
                f"bin and {criteria['max_mean_dbfs']:g} dBFS mean in the "
                "range; widen it or raise max_bin_dbfs/max_mean_dbfs"
            ),
            "silent": (
                "the quiet stretches in the range are digital silence, which "
                "fills nothing; widen the range to reach real room tone"
            ),
            "spike": (
                f"every quiet window holds a 1 ms peak more than "
                f"{criteria['max_spike_db']:g} dB over its median - a tick "
                "that would recur once per lap; widen the range or raise "
                "max_spike_db"
            ),
            "shot_boundary": (
                "every window that fits crosses a shot boundary or lies "
                "outside every shot - the shots in the range are shorter "
                "than min_seconds; lower it or search another shot"
            ),
            "tonal": (
                "every quiet window is tonal or speech-like - faint "
                "programme that would be audible once it repeats; widen "
                "the range"
            ),
        }[rule]
    return {"rule": "no_loop_bed", "rejected_by": rule, "message": message}


def _coerce_arguments(shots, **raw):
    """Coerce and check the arguments; returns them as a namespace, less `shots`.

    `raw` is the command's numeric arguments in signature order (the order
    their errors are reported in); `max_candidates` is the one integer.
    """
    values = {
        name: coerce_number(
            value, int if name == "max_candidates" else float, name, COMMAND
        )
        for name, value in raw.items()
    }
    if shots is not None:
        _check_shots(shots)
    check_arguments(COMMAND, **values)
    if values["min_seconds"] > values["max_seconds"]:
        raise ValueError(
            f"{COMMAND} needs 'min_seconds' ({values['min_seconds']:g}) at or "
            f"below 'max_seconds' ({values['max_seconds']:g})"
        )
    return types.SimpleNamespace(**values)


def _search_window(waveform, sample_rate, args):
    """The mono source, its duration and the (first, last) seconds searched.

    Refused when the source or the range holds no window.
    """
    start_seconds, end_seconds = args.start_seconds, args.end_seconds
    min_seconds = args.min_seconds
    mono = numpy.asarray(waveform, dtype=numpy.float64)
    mono = mono.mean(axis=0) if mono.ndim == 2 else mono
    duration = mono.shape[0] / sample_rate
    if duration < min_seconds:
        raise ValueError(
            f"{COMMAND}: the source is {duration:.2f} s long, shorter than "
            f"'min_seconds' ({min_seconds:g}) - there is no window to find"
        )

    first = 0.0 if start_seconds is None else start_seconds
    last = duration if end_seconds is None else end_seconds
    # Half a sample of slack: an end_seconds copied from a rounded duration
    # is the end of the file, not past it
    slack = 0.5 / sample_rate
    if first >= duration:
        raise ValueError(
            f"{COMMAND}: 'start_seconds' ({first:g}) is past the end of the "
            f"{duration:.2f} s source"
        )
    if last > duration + slack:
        raise ValueError(
            f"{COMMAND}: 'end_seconds' ({last:g}) is past the end of the "
            f"{duration:.2f} s source"
        )
    if last <= first:
        raise ValueError(
            f"{COMMAND} needs 'end_seconds' ({last:g}) after 'start_seconds' "
            f"({first:g})"
        )
    if last - first < min_seconds:
        raise ValueError(
            f"{COMMAND}: the range {first:g}-{last:g} s is shorter than "
            f"'min_seconds' ({min_seconds:g}) - there is no window to find"
        )
    return mono, duration, first, last


def _tonal_scales(
    searched, bin_length, sample_rate, bin_seconds, count, shortest, usable
):
    """The tonal test's blocks, measured once over the range at each block length.

    A block is only worth measuring when every bin in it is quiet and not
    silent, since any window holding one of the others is thrown out before
    tonality. Each entry is (block length in bins, flatness, harmonicity, tonal).
    """
    native_rate = _occupied_rate(searched, sample_rate)
    scales = []  # (block length in bins, flatness, harmonicity, tonal)
    for block_seconds in TONAL_BLOCK_SECONDS:
        block_bins = max(1, min(shortest, int(round(block_seconds / bin_seconds))))
        if any(block_bins == known[0] for known in scales):
            continue
        needed = numpy.zeros(count, dtype=bool)
        if count >= block_bins:
            needed[: count - block_bins + 1] = _sliding(usable, block_bins).all(axis=1)
        scales.append(
            (
                block_bins,
                *_tonal_blocks(
                    searched, bin_length, sample_rate, block_bins, needed, native_rate
                ),
            )
        )
    return scales


def _edge_ticks(power, peaks, bin_threshold):
    """The loudest 1 ms peak just outside each window start and end: (before, after).

    The loudest few 1 ms peaks at the tail of the bin before each start and
    the head of the bin after each end. A neighbour loud enough to be thrown
    out on its own is louder material, not a tick, and the window stops short
    of it.
    """
    edge_quiet = power <= bin_threshold
    tails = numpy.where(edge_quiet, peaks[:, -TICK_GUARD_PEAKS:].max(axis=1), 0.0)
    heads = numpy.where(edge_quiet, peaks[:, :TICK_GUARD_PEAKS].max(axis=1), 0.0)
    before = numpy.concatenate(([0.0], tails[:-1]))
    after = numpy.concatenate((heads[1:], [0.0]))
    return before, after


def _scan_windows(
    power,
    peaks,
    bin_shot,
    scales,
    edges,
    lengths,
    args,
):
    """Judge every (start, length) window; returns (survivors, rejected).

    Vectorised per window length: every pair is judged at once and counted
    under the first rule it fails - loudness, silence, ticks, tonality - so
    `rejected` is a tally of every window on the grid, and each survivor
    carries its source ripple for the pre-rank.
    """
    before, after = edges
    shortest, longest = lengths
    bin_threshold = 10.0 ** (args.max_bin_dbfs / 10.0)
    mean_threshold = 10.0 ** (args.max_mean_dbfs / 10.0)
    spike_ratio = 10.0 ** (args.max_spike_db / 20.0)
    rejected = dict.fromkeys(REJECTION_RULES, 0)
    if bin_shot is not None:
        rejected["shot_boundary"] = 0
    survivors = []  # (source ripple, start bin, length in bins, readings)
    for length in range(shortest, longest + 1):
        windows = _sliding(power, length)  # (starts, length)
        loudest = windows.max(axis=1)
        mean = windows.mean(axis=1)
        if bin_shot is None:
            crossing = numpy.zeros(windows.shape[0], dtype=bool)
        else:
            # A window is inside one shot when every bin it holds is: the
            # same shot throughout, and none a boundary cuts or no shot covers
            owners = _sliding(bin_shot, length)
            crossing = (owners.min(axis=1) != owners.max(axis=1)) | (
                owners.min(axis=1) < 0
            )
            rejected["shot_boundary"] += int(crossing.sum())
        loud = ~crossing & ((loudest > bin_threshold) | (mean > mean_threshold))
        peak_windows = _sliding(peaks, length).reshape(windows.shape[0], -1)
        median = numpy.median(peak_windows, axis=1)
        # More than half the window's 1 ms peaks at zero is digital silence
        # with something in it, not room tone - and would read every
        # sample of that something as a tick
        silent = ~crossing & ~loud & ((mean <= 0.0) | (median <= 0.0))
        largest = numpy.maximum(
            peak_windows.max(axis=1),
            numpy.maximum(before[: windows.shape[0]], after[length - 1 :]),
        )
        ticked = ~crossing & ~loud & ~silent & (largest > spike_ratio * median)
        any_tonal = numpy.zeros(windows.shape[0], dtype=bool)
        for block_bins, _, _, block_tonal in scales:
            inside = _sliding(block_tonal, length - block_bins + 1)
            any_tonal |= inside[: windows.shape[0]].any(axis=1)
        tonal = ~crossing & ~loud & ~silent & ~ticked & any_tonal
        rejected["too_loud"] += int(loud.sum())
        rejected["silent"] += int(silent.sum())
        rejected["spike"] += int(ticked.sum())
        rejected["tonal"] += int(tonal.sum())
        kept = numpy.flatnonzero(~crossing & ~loud & ~silent & ~ticked & ~tonal)
        if kept.size == 0:
            continue

        in_db = _bin_db(windows[kept])
        low, high = numpy.percentile(in_db, RIPPLE_PERCENTILES, axis=1)
        for position, start in enumerate(kept):
            # The block nearest to failing, at either length, which is what
            # the test judged
            flatness = min(
                float(flat[start : start + length - bins + 1].min())
                for bins, flat, _, _ in scales
            )
            harmonicity = max(
                float(harm[start : start + length - bins + 1].max())
                for bins, _, harm, _ in scales
            )
            survivors.append(
                (
                    float(high[position] - low[position]),
                    int(start),
                    length,
                    {
                        "mean_dbfs": 10.0 * math.log10(float(mean[start])),
                        "max_bin_dbfs": 10.0 * math.log10(float(loudest[start])),
                        "spike_db": 20.0
                        * math.log10(float(largest[start] / median[start])),
                        "flatness": flatness,
                        "harmonicity": harmonicity,
                    },
                )
            )
    return survivors, rejected


def _rank_and_loop(
    survivors,
    searched,
    offset,
    bin_length,
    sample_rate,
    bin_shot,
    spans,
    args,
):
    """Thin, loop and rank the survivors; returns (kept, ranked candidates)."""
    # Step 5: steadiest first, thinned so no two overlap, up to a pool that
    # does not depend on how many candidates were asked for
    survivors.sort(key=lambda entry: (entry[0], -entry[2], entry[1]))
    kept = []
    pool = max(LOOPED_POOL, args.max_candidates)
    for ripple, start, length, readings in survivors:
        if len(kept) >= pool:
            break
        end = start + length
        if any(start < k_end and k_start < end for k_start, k_end, _ in kept):
            continue
        kept.append((start, end, readings))

    # Steps 6 and 7: loop each, measure the bed, rank by its ripple
    candidates = []
    for start, end, readings in kept:
        segment = searched[start * bin_length : end * bin_length]
        looped = _looped(segment, sample_rate, args.crossfade_ms, args.loop_seconds)
        start_at = (offset + start * bin_length) / sample_rate
        seconds = (end - start) * bin_length / sample_rate
        gain_db = args.target_bed_dbfs - readings["mean_dbfs"]
        warnings = []
        peak = looped["envelope_peak_db"]
        if peak is not None and peak > LAP_MODULATION_WARN_DB:
            warnings.append("lap_modulation")
        candidates.append(
            {
                "start_seconds": round(start_at, 3),
                "duration_seconds": round(seconds, 3),
                "end_seconds": round(start_at + seconds, 3),
                "shot": None if bin_shot is None else spans[bin_shot[start]][0],
                "mean_dbfs": _round(readings["mean_dbfs"]),
                "max_bin_dbfs": _round(readings["max_bin_dbfs"]),
                "spike_db": _round(readings["spike_db"]),
                "flatness": _round(readings["flatness"], 3),
                "harmonicity": _round(readings["harmonicity"], 3),
                "looped": looped,
                "gain_db": _round(gain_db),
                "gain": _round(10.0 ** (gain_db / 20.0), 3),
                "warnings": warnings,
            }
        )
    candidates.sort(key=lambda c: (c["looped"]["ripple_db"], c["start_seconds"]))
    candidates = [
        {"rank": rank, **candidate}
        for rank, candidate in enumerate(candidates[: args.max_candidates], start=1)
    ]
    return kept, candidates


def _answer(
    candidates,
    rejected,
    args,
    survivor_count,
    kept_count,
    duration,
    sample_rate,
    first,
    last,
    shots_source,
    spans,
):
    """The command's JSON answer, with the finding when nothing was found."""
    criteria = {
        "max_bin_dbfs": args.max_bin_dbfs,
        "max_mean_dbfs": args.max_mean_dbfs,
        "max_spike_db": args.max_spike_db,
        "tonal_flatness": TONAL_FLATNESS_THRESHOLD,
        "harmonicity": HARMONICITY_THRESHOLD,
        "crossfade_ms": args.crossfade_ms,
        "loop_seconds": args.loop_seconds,
        "min_seconds": args.min_seconds,
        "max_seconds": args.max_seconds,
        "target_bed_dbfs": args.target_bed_dbfs,
    }
    findings = [] if candidates else [_empty_finding(rejected, criteria)]
    logger.info(
        f"{COMMAND}: {survivor_count} windows passed level and tick tests, "
        f"{kept_count} looped, {len(candidates)} returned; rejected {rejected}"
    )
    return {
        "source": {
            "duration_seconds": round(duration, 3),
            "sample_rate": int(sample_rate),
            "searched": [round(first, 3), round(last, 3)],
            "shots_source": shots_source,
            **(
                {
                    "shots": [
                        {
                            "name": name,
                            "start_seconds": round(first / sample_rate, 3),
                            "end_seconds": round(end / sample_rate, 3),
                        }
                        for name, first, end in spans
                    ]
                }
                if spans is not None
                else {}
            ),
        },
        "criteria": criteria,
        "candidates": candidates,
        "rejected": rejected,
        "findings": findings,
    }


def find_loop_bed(
    audio,
    start_seconds=None,
    end_seconds=None,
    min_seconds=0.5,
    max_seconds=2.0,
    max_bin_dbfs=-55.0,
    max_mean_dbfs=-60.0,
    max_spike_db=12.0,
    crossfade_ms=250,
    loop_seconds=10.0,
    target_bed_dbfs=-60.0,
    max_candidates=5,
    shots=None,
    fps=None,
):
    """Task command: rank the windows of a recording worth looping into a
    room-tone bed, measured as they will sound looped.

    Every window on a 50 ms grid, from min_seconds to max_seconds long, is
    kept only if it is quiet (every 50 ms bin at or below max_bin_dbfs, its
    mean at or below max_mean_dbfs), not digital silence, free of ticks (its
    largest 1 ms peak at most max_spike_db over its median 1 ms peak) and not
    tonal or speech-like (bleed_join's flatness and harmonicity test, over
    every 0.1 s and 0.2 s block inside it); `rejected` counts each window under the
    first of those it fails. Overlapping survivors are thinned to the
    steadiest, up to a pool that max_candidates does not shrink, and each
    remaining one is looped with loop_audio's own crossfade to loop_seconds and
    measured: `ripple_db` (the 5-95 % spread of the looped 50 ms bins),
    `envelope_peak_db`/`envelope_peak_hz` (the strongest level wobble) and
    `lap_component_db` (the wobble at the lap rate). Candidates are ranked by
    looped ripple, lowest first; one whose envelope peak is above -15 dB
    carries the `lap_modulation` warning.

    A candidate's start_seconds/duration_seconds are slice_audio's arguments
    and its gain is mix_audio's multiplier for reaching target_bed_dbfs.
    With shot boundaries - the `shots` argument, else the shots a video from
    an earlier step carries, else the ones the run manifest beside the file
    records - no candidate crosses a boundary, each names the `shot` it lies
    in, `source.shots_source` says which of the three the shots came from
    (`argument`, `artifact`, `manifest`; null with none) and `rejected`
    counts the windows a boundary cut through under `shot_boundary`. A
    stretch no shot covers is not searched.

    Finding nothing is an answer: `candidates` is empty, `rejected` counts
    the windows each rule threw out, and one finding says which to relax.
    The result must be saved as application/json.

    Args:
        audio: Path of an audio or video file (a video is read audio-only,
            never frame-decoded), or a track or video from an earlier step
        start_seconds: Where the search starts; the head of the file if omitted
        end_seconds: Where the search ends; the end of the file if omitted
        min_seconds: Shortest window tried
        max_seconds: Longest window tried
        max_bin_dbfs: Every 50 ms bin of a window must be at or below this
        max_mean_dbfs: A window's mean (RMS) level must be at or below this
        max_spike_db: Most a window's largest 1 ms peak may sit above its
            median 1 ms peak, in dB
        crossfade_ms: The crossfade candidates are looped with - loop_audio's
            default, so what is measured is what loop_audio will make
        loop_seconds: Length of the looped result that is measured
        target_bed_dbfs: The level each candidate's gain is computed to reach
        max_candidates: How many ranked candidates to return
        shots: Shot boundaries, the assessment probes' shape: a list of
            {name, start_frame, num_frames}, optionally with start_sample and
            num_samples; overrides the shots a video carries or its run records
        fps: Frame rate the shots' frames are counted at, for a source that
            has none of its own (an audio file); a video's own rate otherwise

    Returns:
        A JSON-safe dict: source, criteria, candidates, rejected, findings
    """
    args = _coerce_arguments(
        shots,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        max_bin_dbfs=max_bin_dbfs,
        max_mean_dbfs=max_mean_dbfs,
        max_spike_db=max_spike_db,
        crossfade_ms=crossfade_ms,
        loop_seconds=loop_seconds,
        target_bed_dbfs=target_bed_dbfs,
        max_candidates=max_candidates,
        fps=fps,
    )

    waveform, sample_rate, source_fps, carried, path = _read_source(audio)
    shots, shots_source = _resolve_shots(shots, carried, path)
    mono, duration, first, last = _search_window(waveform, sample_rate, args)

    offset = int(round(first * sample_rate))
    stop = min(int(round(last * sample_rate)), mono.shape[0])
    searched = mono[offset:stop]
    bin_length, power, peaks = _envelope(searched, sample_rate)
    bin_seconds = bin_length / sample_rate
    count = power.shape[0]
    fps = args.fps or source_fps
    spans = _shot_spans(shots, fps, sample_rate) if shots else None
    bin_shot = (
        _bin_shots(spans, offset, bin_length, count) if spans is not None else None
    )

    shortest = max(1, int(round(args.min_seconds / bin_seconds)))
    longest = min(count, int(round(args.max_seconds / bin_seconds)))
    bin_threshold = 10.0 ** (args.max_bin_dbfs / 10.0)
    usable = (power <= bin_threshold) & (power > 0.0)
    scales = _tonal_scales(
        searched, bin_length, sample_rate, bin_seconds, count, shortest, usable
    )
    edges = _edge_ticks(power, peaks, bin_threshold)
    survivors, rejected = _scan_windows(
        power, peaks, bin_shot, scales, edges, (shortest, longest), args
    )
    kept, candidates = _rank_and_loop(
        survivors, searched, offset, bin_length, sample_rate, bin_shot, spans, args
    )
    return _answer(
        candidates,
        rejected,
        args,
        len(survivors),
        len(kept),
        duration,
        sample_rate,
        first,
        last,
        shots_source,
        spans,
    )
