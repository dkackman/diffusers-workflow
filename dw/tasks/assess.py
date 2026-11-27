"""Assessment probes: measure a finished cut and say where to look (#387).

Four read-only task commands - `analyze_shots`, `analyze_seams`,
`analyze_sync_drift` and `analyze_held_frames` - each take a video (a path, or the AudioVideo an
earlier step returned) and answer a JSON-safe dict: every measurement it
took, the `findings` its rules raised (`dw/assessment_rules.py`), the names
of the rules it read the measurements against and where the shot
boundaries came from. Nothing acts on a finding; they are places to look.

The reader streams. A cut is minutes of full-resolution picture, and every
one of these measurements needs only a 64x36 grey thumbnail of each frame
and the soundtrack, so `read_media` decodes the file once and keeps exactly
that - never a frame list, and never `load_audio`'s 0.25 s fit of the track
to the picture, which would move the very sample count `analyze_sync_drift`
is measuring. The soundtrack is trimmed to the audio stream's own duration,
which the container already reports net of the encoder's priming, so a
lossy codec's padding is not read as drift.

Shot boundaries come, in order, from the `shots` argument, the AudioVideo's
own `shots`, the run manifest beside a file (`dw.runs.shots_beside`), and
otherwise the whole file is one shot. `shots_source` says which. A shot
record may carry `hard_cut: true` - a cut meant as a cut - and the frame
jump rule does not fire at the seam that shot opens.
"""

import logging
import math

import numpy

from ..assessment_rules import (
    HOLE_VOICED_DBFS,
    crosses,
    finding,
    rules_for,
    sort_findings,
)
from .. import dsp
from ..media import iter_rgb_frames, read_thumbnails_and_track
from ..events import emit_warning
from .audio_utils import (  # noqa: F401 - re-exported: the level probes' names live here too
    DEAD_AIR_WINDOW,
    SILENCE_PEAK,
    to_dbfs,
    measure_dead_air,
    measure_noise_floor,
    window_levels,
)
from .image_ops import (  # noqa: F401 - re-exported: the look probes' names live here too
    LOOK_MEASURES,
    TYPICAL_DELTA_PERCENTILE,
    measure_seam_look,
    measure_segment_looks,
    as_image,
    look_stats,
    thumbs_and_looks,
)

logger = logging.getLogger("dw")

THUMB_WIDTH = 64
THUMB_HEIGHT = 36

# The colour frame each look stat is taken from (#1031): big enough that a
# 3x3 high-pass still sees detail, small enough to reduce at decode speed
LOOK_WIDTH = 128
LOOK_HEIGHT = 72


# Audio windows around a seam, in seconds. The edge windows say whether the
# join itself is voiced (the hole guard) and what the balance does across it;
# they are not the level step, which is shot against shot (`_shot_rms`): a
# shot's own last and first quarter-second differ by whatever the take does
# there - 20 dB on a line that trails off and opens on a breath - and read
# as a step at every seam of a cut made of one clip (#387's bounce).
LEVEL_WINDOW = 0.25
FLOOR_WINDOW = 0.02
CLICK_WINDOW = 0.002
CLICK_NEIGHBOUR_WINDOW = 0.01

# The inter-frame difference a shot is expected to show, below which its
# own motion is not a baseline: a static shot's 90th percentile is ~0, and
# dividing a seam's change by it would call any change at all a jump.
# Grey levels on the 0-255 scale.
TYPICAL_DELTA_FLOOR = 2.0

# The held-frame measure (#1527): a frame is held when its thumbnail differs
# from its predecessor's by less than HELD_EPSILON (0-255 grey) while a
# neighbouring frame moves by TYPICAL_DELTA_FLOOR or more. Frames within
# HELD_CUT_GUARD of a hard cut never count. Holds whose gaps are at most
# HELD_CLUSTER_GAP frames form a cluster when there are HELD_CLUSTER_MIN or
# more of them. The d series `detail` returns is cut to HELD_SERIES_POINTS.
HELD_EPSILON = 1.0
HELD_CUT_GUARD = 1
HELD_CLUSTER_GAP = 4
HELD_CLUSTER_MIN = 2
HELD_SERIES_POINTS = 240


# A ratio against a zero neighbour has no size; report it capped
CLICK_CAP_DB = 120.0


class Media:
    """What the probes read from a video: thumbnails, soundtrack, timing.

    thumbs: (frames, THUMB_HEIGHT, THUMB_WIDTH) uint8 grey, or None
    audio: (channels, samples) float32, or None
    video_seconds / audio_seconds: each stream's own duration, or None
    looks: (frames, 4) float32 `look_stats` per frame (LOOK_MEASURES
        order), or None
    """

    def __init__(
        self,
        thumbs,
        audio,
        sample_rate,
        fps,
        video_seconds=None,
        audio_seconds=None,
        shots=None,
        looks=None,
        rgb_frames=None,
    ):
        self.thumbs = thumbs
        self.looks = looks
        # A zero-argument callable yielding the picture as (H, W, 3) uint8
        # frames, for the cut detector analyze_held_frames runs; None when
        # the picture is not to be had again
        self.rgb_frames = rgb_frames
        self.audio = audio
        self.sample_rate = sample_rate
        self.fps = fps
        self.video_seconds = video_seconds
        self.audio_seconds = audio_seconds
        self.shots = shots

    @property
    def frame_count(self):
        return 0 if self.thumbs is None else int(self.thumbs.shape[0])


def read_media(path):
    """Stream a video file once into a Media.

    Each decoded frame is reduced to a grey thumbnail as it arrives and then
    dropped, so memory holds thumbnails, not pictures. The soundtrack is
    kept whole (the probes cut windows out of it anywhere) and trimmed to
    the audio stream's reported duration. Each frame's look stats are taken
    from a LOOK_WIDTH x LOOK_HEIGHT colour copy as it arrives.
    """
    thumbs, chunks, fps, video_seconds, audio_seconds, sample_rate, looks = (
        read_thumbnails_and_track(
            path,
            THUMB_WIDTH,
            THUMB_HEIGHT,
            look=look_stats,
            look_width=LOOK_WIDTH,
            look_height=LOOK_HEIGHT,
        )
    )

    waveform = None
    if chunks:
        waveform = numpy.ascontiguousarray(numpy.concatenate(chunks, axis=0).T)
        if audio_seconds is not None:
            waveform = waveform[:, : int(round(audio_seconds * sample_rate))]
    if video_seconds is None and fps and thumbs:
        video_seconds = len(thumbs) / fps
    return Media(
        numpy.stack(thumbs) if thumbs else None,
        waveform,
        sample_rate,
        fps,
        video_seconds,
        audio_seconds,
        looks=numpy.stack(looks) if looks else None,
        rgb_frames=(lambda: iter_rgb_frames(path)) if thumbs else None,
    )


def _in_memory_thumbs(frames):
    """(thumbnails, looks) of an AudioVideo's frames at THUMB_ and LOOK_
    size; `image_ops.thumbs_and_looks` is the reduction."""
    return thumbs_and_looks(
        frames, (THUMB_WIDTH, THUMB_HEIGHT), (LOOK_WIDTH, LOOK_HEIGHT)
    )


def frame_measures(frames):
    """(thumbnails, looks) of in-memory frames - the public name the chain's
    per-segment check (dw/pipeline_processors/chain_check.py) reads them by;
    `_in_memory_thumbs` is the one reduction."""
    return _in_memory_thumbs(frames)


def seam_measures(thumbs, looks, previous, shot, fade=0):
    """The picture and look measurements `analyze_seams` takes at the seam
    `shot` opens, over thumbnails and looks already reduced - the public name
    the chain's per-segment check measures a seam by, before the segment is
    on any timeline. `_seam_video` and `measure_seam_look` are the arithmetic."""
    media = Media(thumbs, None, None, None, looks=looks)
    return {
        **_seam_video(media, previous, shot, fade),
        **measure_seam_look(media, previous, shot, fade),
    }


def _file_path(video):
    """The validated file a probe's 'video' names, or None for an in-memory
    clip: a path, or the VideoFileReference an asset:/output: reference or a
    literal path is realized to (dw/arguments.py, #387)."""
    from ..locations import validate_media_path
    from .video_utils import VideoFileReference

    if isinstance(video, VideoFileReference):
        video = video.path
    if isinstance(video, str):
        return validate_media_path(video, None, "a video to assess")
    return None


def media_from(video):
    """A Media from a path, a VideoFileReference or an in-memory AudioVideo."""
    path = _file_path(video)
    if path is not None:
        return read_media(path)
    if isinstance(video, numpy.ndarray) and video.ndim == 4:
        # video_frames / fit_to_model hand on the frames as one array (a
        # FittedVideo carries its rate): frames only, no soundtrack
        frames, audio, sample_rate, shots = video, None, None, None
    elif hasattr(video, "frames") or hasattr(video, "audio"):
        frames = getattr(video, "frames", None)
        audio = getattr(video, "audio", None)
        sample_rate = getattr(video, "sample_rate", None)
        shots = getattr(video, "shots", None)
    else:
        raise ValueError(
            "a probe takes a stored video (asset:, output: or a path) or the "
            "video an earlier step returned (a video, or the frames video_frames "
            f"or fit_to_model returned), not {type(video).__name__}"
        )
    waveform = None if audio is None else dsp.as_channels_samples(audio)
    thumbs, looks = (None, None) if frames is None else _in_memory_thumbs(frames)
    fps = getattr(video, "fps", None)
    return Media(
        thumbs,
        waveform,
        sample_rate,
        fps,
        video_seconds=thumbs.shape[0] / fps if thumbs is not None and fps else None,
        audio_seconds=(
            waveform.shape[1] / sample_rate
            if waveform is not None and sample_rate
            else None
        ),
        shots=shots,
        looks=looks,
        rgb_frames=None if frames is None else (lambda: _in_memory_rgb(frames)),
    )


def _in_memory_rgb(frames):
    """In-memory frames as (H, W, 3) uint8 arrays, one at a time."""
    from ..pipeline_processors.chain_spill import SegmentedFrames

    if isinstance(frames, SegmentedFrames):
        frames = (frame for chunk in frames for frame in chunk)
    for frame in frames:
        yield numpy.asarray(as_image(frame).convert("RGB"), dtype=numpy.uint8)


def resolve_shots(video, media, shots=None):
    """The shot records to measure against, and where they came from."""
    if shots:
        return [dict(shot) for shot in shots], "argument"
    if media.shots:
        return [dict(shot) for shot in media.shots], "artifact"
    path = _file_path(video)
    if path is not None:
        from ..runs import shots_beside

        recorded = shots_beside(path)
        if recorded:
            return [dict(shot) for shot in recorded], "manifest"
    return None, "none"


def _whole_file_shot(media):
    samples = media.audio.shape[1] if media.audio is not None else None
    return {
        "name": "whole",
        "start_frame": 0,
        "num_frames": media.frame_count,
        "start_sample": 0 if samples is not None else None,
        "num_samples": samples,
    }


def sample_span(shot, media):
    """A shot's (start, count, source) on the soundtrack: recorded, else
    derived from its frames. `media` needs only `fps` and `sample_rate`;
    `check_script` places its shots with this too."""
    start = shot.get("start_sample")
    count = shot.get("num_samples")
    if start is not None and count is not None:
        return int(start), int(count), "recorded"
    if media.fps and media.sample_rate:
        scale = media.sample_rate / media.fps
        return (
            int(round(shot.get("start_frame", 0) * scale)),
            int(round(shot.get("num_frames", 0) * scale)),
            "derived",
        )
    return None, None, None


def _clip(media, start, end):
    total = media.audio.shape[1]
    start = max(0, min(total, int(start)))
    end = max(start, min(total, int(end)))
    return media.audio[:, start:end]


def _findings(probe, record, at, skip=()):
    found = []
    for rule in rules_for(probe):
        if rule["name"] in skip or rule["field"] not in record:
            continue
        if crosses(rule, record[rule["field"]]):
            found.append(finding(rule, record[rule["field"]], at))
    return found


def _span_overrun(media, shot):
    """How far past the file `shot`'s frames or samples reach, or None when
    it fits. `_clip` silently clamps an out-of-range window to the file
    (#425): a shot record with `start_frame + num_frames` past the video's
    real length, or explicit `start_sample + num_samples` past the
    soundtrack's, is measured over a window shorter than the caller asked
    for and nothing says so unless this is checked first."""
    over_frames = None
    if media.thumbs is not None:
        start_frame = shot.get("start_frame", 0)
        num_frames = shot.get("num_frames")
        if isinstance(start_frame, (int, float)) and not isinstance(start_frame, bool):
            if isinstance(num_frames, (int, float)) and not isinstance(
                num_frames, bool
            ):
                end = int(start_frame) + int(num_frames)
                if end > media.frame_count:
                    over_frames = end - media.frame_count
    over_samples = None
    if media.audio is not None:
        start_sample, num_samples = shot.get("start_sample"), shot.get("num_samples")
        if isinstance(start_sample, (int, float)) and not isinstance(
            start_sample, bool
        ):
            if isinstance(num_samples, (int, float)) and not isinstance(
                num_samples, bool
            ):
                total = media.audio.shape[1]
                end = int(start_sample) + int(num_samples)
                if end > total:
                    over_samples = end - total
    if over_frames is None and over_samples is None:
        return None
    return {"frames": over_frames, "samples": over_samples}


def _shot_span_findings(probe, records, media):
    """Findings (and a run warning) for every shot record whose declared
    span reaches past the file - clipped silently otherwise (#425)."""
    findings = []
    overrun_names = []
    for index, shot in enumerate(records or []):
        overrun = _span_overrun(media, shot)
        if overrun is None:
            continue
        detail = (
            f"{overrun['frames']} frame(s)"
            if overrun["frames"] is not None
            else f"{overrun['samples']} sample(s)"
        )
        findings.append(
            {
                "rule": "shot_span_overrun",
                "severity": "warn",
                "at": {
                    "shot": shot.get("name"),
                    "index": index,
                    "start_frame": shot.get("start_frame"),
                },
                "value": overrun,
                "threshold": 0,
                "says": f"shot {shot.get('name')!r} reaches {detail} past the file's end",
            }
        )
        overrun_names.append(shot.get("name"))
    if overrun_names:
        emit_warning(
            f"{probe}: shot record(s) {', '.join(str(name) for name in overrun_names)} "
            "reach past the file's end and were silently clipped to it",
            kind="shot_span_overrun",
            probe=probe,
            shots=overrun_names,
        )
    return findings


def _answer(probe, measurements, findings, shots_source, shot_dependent=()):
    """A probe's answer, with `rules_applied` cut down to the rules that
    actually ran. `shot_dependent` names (or `True` for all of the probe's
    rules) the ones that only mean anything measured shot against shot; with
    no shot boundaries at all (`shots_source == "none"`) those are reported
    as `rules_skipped` instead of `rules_applied`, and a run warning says why
    - without this a shotless file (an asset kept before #393, an upload, a
    cut joined outside dw) read as a clean pass with nothing measured (#394).
    """
    names = [rule["name"] for rule in rules_for(probe)]
    dependent = set(names) if shot_dependent is True else set(shot_dependent)
    applied, skipped = names, []
    if shots_source == "none" and dependent:
        applied = [name for name in names if name not in dependent]
        skipped = [
            {"rule": name, "reason": "no shot boundaries"}
            for name in names
            if name in dependent
        ]
        emit_warning(
            f"{probe} found no shot boundaries for this file, so "
            f"{', '.join(sorted(dependent))} could not be measured - pass "
            "shots= to supply them",
            kind="no_shot_boundaries",
            probe=probe,
        )
    return {
        **measurements,
        "findings": sort_findings(findings),
        "rules_applied": applied,
        "rules_skipped": skipped,
        "shots_source": shots_source,
    }


def _floors_below_neighbours(measured):
    """Fill each shot's `floor_below_neighbours_db`: the louder adjacent
    shot's `noise_floor_dbfs` minus this shot's own, unclamped, so a shot
    whose floor sits above its neighbours' reads negative (#1620).

    In a set with a loud bed the shots without dialogue keep a floor near
    the bed, while a voiced shot whose gaps between lines fall through it
    reads a floor far below them - a hole level alone cannot tell from a
    dry room's pause, since in a quiet room every shot's floor is low. Null
    for a cut of one shot, and wherever this shot's floor or both
    neighbours' floors are null.
    """
    for index, shot in enumerate(measured):
        own = shot["noise_floor_dbfs"]
        neighbours = [
            measured[at]["noise_floor_dbfs"]
            for at in (index - 1, index + 1)
            if 0 <= at < len(measured) and measured[at]["noise_floor_dbfs"] is not None
        ]
        if own is None or not neighbours:
            continue
        shot["floor_below_neighbours_db"] = dsp.rounded(max(neighbours) - own)


def analyze_shots(video, shots=None):
    """Task command: each shot's level and spectral balance, and how far
    apart the shots sit.

    Args:
        video: A video file's path, or the video an earlier step returned
        shots: Shot records to measure by, overriding any the video carries

    Returns:
        {shots: [{name, start_frame, num_frames, peak_dbfs, rms_dbfs, crest_db, low_dbfs, mid_dbfs,
        high_dbfs, high_share, noise_floor_dbfs, floor_below_neighbours_db,
        dead_air_seconds, dead_air_at,
        dead_air_floor_dbfs, samples}], rms_range_db, findings, rules_applied, rules_skipped,
        shots_source}
    """
    media = media_from(video)
    return shots_answer(media, *resolve_shots(video, media, shots))


def shots_answer(media, records, source):
    """`analyze_shots` over an already-read Media and resolved shots."""
    if media.audio is None:
        return _answer(
            "analyze_shots",
            {"shots": [], "rms_range_db": None, "has_audio": False},
            _shot_span_findings("analyze_shots", records, media),
            source,
            shot_dependent={"shot_level_spread"},
        )
    records = records or [_whole_file_shot(media)]
    overrun_findings = _shot_span_findings("analyze_shots", records, media)

    measured = []
    for shot in records:
        start, count, samples_source = sample_span(shot, media)
        window = _clip(media, start, start + count) if start is not None else None
        peak = to_dbfs(dsp.peak(window))
        rms = to_dbfs(dsp.rms(window))
        balance = (
            dsp.spectral_balance(window, media.sample_rate)
            if window is not None and window.size
            else {"low_dbfs": None, "mid_dbfs": None, "high_dbfs": None}
        )
        shares = dsp.band_shares(balance)
        dead_air_seconds, dead_air_at, dead_air_floor_dbfs = (
            measure_dead_air(media, start, count)
            if start is not None
            else (None, None, None)
        )
        measured.append(
            {
                "name": shot.get("name"),
                "start_frame": shot.get("start_frame"),
                "num_frames": shot.get("num_frames"),
                "peak_dbfs": dsp.rounded(peak),
                "rms_dbfs": dsp.rounded(rms),
                "crest_db": dsp.rounded(
                    None if peak is None or rms is None else peak - rms
                ),
                **{key: dsp.rounded(value) for key, value in balance.items()},
                "high_share": dsp.rounded(
                    shares["high_dbfs"] if shares is not None else None, 4
                ),
                "noise_floor_dbfs": dsp.rounded(
                    measure_noise_floor(media, start, count, rms)
                ),
                "floor_below_neighbours_db": None,
                "dead_air_seconds": dead_air_seconds,
                "dead_air_at": dead_air_at,
                "dead_air_floor_dbfs": dead_air_floor_dbfs,
                "samples": samples_source,
            }
        )

    _floors_below_neighbours(measured)

    voiced = [shot for shot in measured if shot["rms_dbfs"] is not None]
    rms_range = None
    at = None
    if len(voiced) > 1:
        loudest = max(voiced, key=lambda shot: shot["rms_dbfs"])
        quietest = min(voiced, key=lambda shot: shot["rms_dbfs"])
        rms_range = dsp.rounded(loudest["rms_dbfs"] - quietest["rms_dbfs"])
        # Names repeat once a cut re-joins earlier episodes, so the shots'
        # positions in `shots` and their start frames say which is meant
        at = {
            "between": [loudest["name"], quietest["name"]],
            "indices": [measured.index(loudest), measured.index(quietest)],
            "start_frames": [loudest["start_frame"], quietest["start_frame"]],
        }
    elif voiced:
        rms_range = 0.0
    answer = {"shots": measured, "rms_range_db": rms_range, "has_audio": True}
    dead_air_findings = []
    for index, (shot, record) in enumerate(zip(measured, records)):
        # A shot that is quiet throughout, on purpose, holds no gap to find -
        # the same guard seam_hole applies to a seam's two sides (#465); a
        # shot recorded silent_by_design is silent on purpose (#1162). A
        # quiet shot's low floor is its own level, not a hole in a bed, so
        # shot_bed_drop takes the same guard (#1620)
        quiet = (
            shot["rms_dbfs"] is not None and shot["rms_dbfs"] <= HOLE_VOICED_DBFS
        ) or (isinstance(record, dict) and record.get("silent_by_design") is True)
        dead_air_findings.extend(
            _findings(
                "analyze_shots",
                shot,
                {
                    "shot": shot["name"],
                    "index": index,
                    "start_frame": shot["start_frame"],
                    "seconds": shot["dead_air_at"],
                },
                skip={"shot_dead_air", "shot_bed_drop"} if quiet else (),
            )
        )
    return _answer(
        "analyze_shots",
        answer,
        overrun_findings
        + (_findings("analyze_shots", answer, at) if at else [])
        + dead_air_findings,
        source,
        shot_dependent={"shot_level_spread"},
    )


def _band_shares(window, sample_rate):
    if window is None or window.size == 0:
        return None
    return dsp.band_shares(dsp.spectral_balance(window, sample_rate))


def _shot_rms(media, shot):
    """A shot's RMS level over its whole sample span, in dBFS, or None."""
    start, count, _source = sample_span(shot, media)
    if start is None:
        return None
    return to_dbfs(dsp.rms(_clip(media, start, start + count)))


def _seam_audio(media, before_end, after_start, previous_rms, next_rms):
    """Audio measurements at a seam: the edge windows end at `before_end`
    and open at `after_start` (the same sample at a cut, either side of the
    fade at a dissolve), and the join is what lies between them - or the
    FLOOR_WINDOW centred on the cut. The level step is between the two
    shots' own levels, `previous_rms` and `next_rms`."""
    rate = media.sample_rate
    level = int(round(LEVEL_WINDOW * rate))
    before = _clip(media, before_end - level, before_end)
    after = _clip(media, after_start, after_start + level)
    before_rms = to_dbfs(dsp.rms(before))
    after_rms = to_dbfs(dsp.rms(after))

    centre = (before_end + after_start) // 2
    half_floor = max(1, int(round(FLOOR_WINDOW * rate / 2)))
    join = (
        _clip(media, before_end, after_start)
        if after_start - before_end > 2 * half_floor
        else _clip(media, centre - half_floor, centre + half_floor)
    )
    floor = to_dbfs(dsp.rms(join))

    half_click = max(1, int(round(CLICK_WINDOW * rate / 2)))
    neighbour = int(round(CLICK_NEIGHBOUR_WINDOW * rate))
    click_peak = dsp.peak(_clip(media, centre - half_click, centre + half_click))
    neighbour_peak = max(
        dsp.peak(_clip(media, centre - half_click - neighbour, centre - half_click))
        or 0.0,
        dsp.peak(_clip(media, centre + half_click, centre + half_click + neighbour))
        or 0.0,
    )
    click = None
    if click_peak is not None and click_peak > SILENCE_PEAK:
        click = (
            CLICK_CAP_DB
            if neighbour_peak <= SILENCE_PEAK
            else min(CLICK_CAP_DB, 20.0 * math.log10(click_peak / neighbour_peak))
        )

    shares_before = _band_shares(before, rate)
    shares_after = _band_shares(after, rate)
    spectral_shift = (
        sum(abs(shares_after[key] - shares_before[key]) for key in shares_before) / 2.0
        if shares_before and shares_after
        else None
    )
    return {
        "before_rms_dbfs": dsp.rounded(before_rms),
        "after_rms_dbfs": dsp.rounded(after_rms),
        "before_shot_rms_dbfs": dsp.rounded(previous_rms),
        "after_shot_rms_dbfs": dsp.rounded(next_rms),
        "level_step_db": dsp.rounded(
            None
            if previous_rms is None or next_rms is None
            else abs(next_rms - previous_rms)
        ),
        "floor_dbfs": dsp.rounded(floor),
        "click_db": dsp.rounded(click),
        "spectral_shift": dsp.rounded(spectral_shift, 3),
    }


def _typical_delta(thumbs, start, end):
    """The TYPICAL_DELTA_PERCENTILE of frame-to-frame change inside
    thumbs[start:end], or None for fewer than two frames."""
    span = thumbs[max(0, start) : max(0, end)]
    if span.shape[0] < 2:
        return None
    deltas = numpy.abs(numpy.diff(span.astype(numpy.int16), axis=0)).mean(axis=(1, 2))
    return float(numpy.percentile(deltas, TYPICAL_DELTA_PERCENTILE))


def _seam_video(media, previous, shot, fade):
    """Picture measurements at the seam `shot` opens: the largest single-frame
    change across it (at a dissolve, across the whole fade - each step of a
    fade is small, which is what a dissolve is), against the larger of the
    two shots' own typical change, floored."""
    thumbs = media.thumbs
    start = int(shot.get("start_frame", 0))
    first = max(0, start - 1)
    last = min(thumbs.shape[0] - 1, start + max(0, fade - 1) if fade else start)
    if last <= first:
        return {"frame_delta": None, "typical_delta": None, "jump_ratio": None}
    across = numpy.abs(
        numpy.diff(thumbs[first : last + 1].astype(numpy.int16), axis=0)
    ).mean(axis=(1, 2))
    frame_delta = float(across.max())
    typical = [
        _typical_delta(
            thumbs,
            int(previous.get("start_frame", 0))
            + int(previous.get("overlap_frames") or 0),
            start,
        ),
        _typical_delta(thumbs, start + fade, start + int(shot.get("num_frames", 0))),
    ]
    typical = max(
        [TYPICAL_DELTA_FLOOR] + [value for value in typical if value is not None]
    )
    return {
        "frame_delta": dsp.rounded(frame_delta),
        "typical_delta": dsp.rounded(typical),
        "jump_ratio": dsp.rounded(frame_delta / typical),
    }


def analyze_seams(video, shots=None):
    """Task command: measure every seam between shots, audio and picture.

    Args:
        video: A video file's path, or the video an earlier step returned
        shots: Shot records to measure by, overriding any the video carries.
            A shot marked `hard_cut: true` opens a seam meant as a cut;
            one with `audio_bleed_ms` a cut with the tail bled on;
            one with `seam_fade_ms` or `crossfade_ms` opens a seam faded on request

    Returns:
        {seams: [{seam, between, seconds, kind, level_step_db,
        before_shot_rms_dbfs, after_shot_rms_dbfs, floor_dbfs, click_db,
        spectral_shift, before_rms_dbfs, after_rms_dbfs,
        frame_delta, typical_delta, jump_ratio, saturation_step,
        lightness_step, contrast_step, look_ratio, look_ratio_measure,
        detail_ratio}], segment_looks: [{shot, between_frames: [first,
        last], first: {lightness, contrast, saturation, detail}, last:
        {...}}], findings, rules_applied, rules_skipped, shots_source}. The
        look fields are measures only - no rule reads them - and null when
        the media has no picture. `segment_looks` is each shot's mean look
        over the first and last second of its own new frames (after any
        overlap), so seam differences plus in-segment changes sum to the
        take's change; `[]` with no picture or fewer than two shots.
    """
    media = media_from(video)
    return seams_answer(media, *resolve_shots(video, media, shots))


def seams_answer(media, records, source):
    """`analyze_seams` over an already-read Media and resolved shots."""
    if not records or len(records) < 2:
        return _answer(
            "analyze_seams",
            {"seams": [], "segment_looks": []},
            _shot_span_findings("analyze_seams", records, media),
            source,
            shot_dependent=True,
        )

    seams = []
    blended = set()
    findings = _shot_span_findings("analyze_seams", records, media)
    for index in range(1, len(records)):
        previous, shot = records[index - 1], records[index]
        fade = int(shot.get("overlap_frames") or 0)
        start_frame = int(shot.get("start_frame", 0))
        seam_frame = start_frame + fade / 2.0
        seconds = seam_frame / media.fps if media.fps else None
        record = {
            "seam": index,
            "between": [previous.get("name"), shot.get("name")],
            "seconds": dsp.rounded(seconds, 3),
            "kind": "dissolve" if fade else "cut",
            "hard_cut": bool(shot.get("hard_cut")),
        }
        if shot.get("seam_fade_ms") is not None:
            record["seam_fade_ms"] = shot["seam_fade_ms"]
        if shot.get("audio_bleed_ms") is not None:
            record["audio_bleed_ms"] = shot["audio_bleed_ms"]
        if shot.get("crossfade_ms") is not None:
            record["crossfade_ms"] = shot["crossfade_ms"]
        skip = set()
        if media.audio is not None and media.sample_rate:
            start, _count, _source = sample_span(shot, media)
            if start is not None:
                fade_samples = (
                    int(round(fade / media.fps * media.sample_rate))
                    if fade and media.fps
                    else 0
                )
                record.update(
                    _seam_audio(
                        media,
                        start,
                        start + fade_samples,
                        _shot_rms(media, previous),
                        _shot_rms(media, shot),
                    )
                )
                if (
                    record["before_rms_dbfs"] is None
                    or record["after_rms_dbfs"] is None
                    or record["before_rms_dbfs"] <= HOLE_VOICED_DBFS
                    or record["after_rms_dbfs"] <= HOLE_VOICED_DBFS
                ):
                    skip.add("seam_hole")
        else:
            skip.add("seam_hole")
        if media.thumbs is not None:
            record.update(_seam_video(media, previous, shot, fade))
        record.update(measure_seam_look(media, previous, shot, fade))
        if record["hard_cut"] or "audio_bleed_ms" in record:
            # A bled seam is still a picture cut (#783)
            skip.add("seam_frame_jump")
        if "seam_fade_ms" in record or "crossfade_ms" in record:
            # A fade or chain crossfade dips the join by design (#659, #660)
            skip.add("seam_hole")
            blended.add(index)
        seams.append(record)
        findings.extend(
            _findings(
                "analyze_seams",
                record,
                {
                    "seam": index,
                    "between": record["between"],
                    "seconds": record["seconds"],
                },
                skip,
            )
        )
    answer = _answer(
        "analyze_seams",
        {"seams": seams, "segment_looks": measure_segment_looks(media, records)},
        findings,
        source,
        shot_dependent=True,
    )
    if blended and "seam_hole" in answer["rules_applied"]:
        # Skipped at those seams only, but say so (#660)
        answer["rules_skipped"].append(
            {
                "rule": "seam_hole",
                "reason": "seam carries its own fade/crossfade",
                "seams": sorted(blended),
            }
        )
        if len(blended) == len(seams):
            # Measured nowhere: not "applied" (#394 contract, #1179)
            answer["rules_applied"].remove("seam_hole")
    return answer


def analyze_sync_drift(video, shots=None):
    """Task command: how far the soundtrack sits from the picture, shot by
    shot and over the whole file.

    Args:
        video: A video file's path, or the video an earlier step returned
        shots: Shot records to measure by, overriding any the video carries

    Returns:
        {shots: [{name, start_offset_ms, end_offset_ms}], max_offset_ms,
        video_seconds, audio_seconds, length_delta_ms, findings,
        rules_applied, rules_skipped, shots_source}
    """
    media = media_from(video)
    return sync_drift_answer(media, *resolve_shots(video, media, shots))


def sync_drift_answer(media, records, source):
    """`analyze_sync_drift` over an already-read Media and resolved shots."""
    measured = []
    findings = _shot_span_findings("analyze_sync_drift", records, media)
    rate = media.sample_rate
    fps = media.fps
    for shot in records or []:
        start, count = shot.get("start_sample"), shot.get("num_samples")
        if start is None or count is None or not rate or not fps:
            continue
        start_frame = int(shot.get("start_frame", 0))
        end_frame = start_frame + int(shot.get("num_frames", 0))
        record = {
            "name": shot.get("name"),
            "start_offset_ms": dsp.rounded(
                (int(start) / rate - start_frame / fps) * 1000.0
            ),
            "end_offset_ms": dsp.rounded(
                ((int(start) + int(count)) / rate - end_frame / fps) * 1000.0
            ),
        }
        measured.append(record)
        findings.extend(
            _findings(
                "analyze_sync_drift",
                record,
                {
                    "shot": record["name"],
                    "seconds": dsp.rounded(end_frame / fps, 3),
                },
            )
        )

    length_delta = None
    if media.audio_seconds is not None and media.video_seconds is not None:
        length_delta = dsp.rounded((media.audio_seconds - media.video_seconds) * 1000.0)
    answer = {
        "shots": measured,
        "max_offset_ms": (
            max((record["end_offset_ms"] for record in measured), key=abs)
            if measured
            else None
        ),
        "video_seconds": dsp.rounded(media.video_seconds, 4),
        "audio_seconds": dsp.rounded(media.audio_seconds, 4),
        "length_delta_ms": length_delta,
    }
    findings.extend(
        _findings(
            "analyze_sync_drift",
            {"length_delta_ms": length_delta},
            {"file": True},
        )
    )
    return _answer("analyze_sync_drift", answer, findings, source)


def _frame_deltas(thumbs):
    """d[i] = mean |thumbs[i] - thumbs[i-1]| (0-255 grey) for i >= 1; d[0]
    is nan - frame 0 has no predecessor."""
    thumbs = numpy.asarray(thumbs, dtype=numpy.float32)
    d = numpy.full(len(thumbs), numpy.nan, dtype=numpy.float64)
    if len(thumbs) > 1:
        d[1:] = numpy.mean(numpy.abs(numpy.diff(thumbs, axis=0)), axis=(1, 2))
    return d


def _held_frames(d, skip, epsilon=HELD_EPSILON, moving=TYPICAL_DELTA_FLOOR):
    """The held frame indices: d[i] < epsilon with a neighbouring d at least
    `moving`. A missing neighbour (either end) is not moving, so a still
    shot holds nothing."""
    out = []
    n = len(d)
    for i in range(1, n):
        if i in skip or not d[i] < epsilon:
            continue
        before = d[i - 1] if i - 1 >= 1 else numpy.nan
        after = d[i + 1] if i + 1 < n else numpy.nan
        if numpy.nanmax([before, after, -1.0]) >= moving:
            out.append(i)
    return out


def _held_clusters(frames, fps):
    """Runs of held frames whose gaps are at most HELD_CLUSTER_GAP frames, of
    at least HELD_CLUSTER_MIN holds: [{first_frame, last_frame, held}], with
    {start, end} in seconds when the video carries a frame rate - a time at an
    assumed rate would be a wrong one."""
    runs, current = [], []
    for frame in sorted(frames):
        if current and frame - current[-1] - 1 > HELD_CLUSTER_GAP:
            runs.append(current)
            current = []
        current.append(frame)
    if current:
        runs.append(current)
    return [
        {
            "first_frame": run[0],
            "last_frame": run[-1],
            **(
                {"start": round(run[0] / fps, 3), "end": round(run[-1] / fps, 3)}
                if fps
                else {}
            ),
            "held": len(run),
        }
        for run in runs
        if len(run) >= HELD_CLUSTER_MIN
    ]


def _cut_frames(media):
    """The frames a hard cut lands on (`scene_cuts.content_cuts`), or [] when
    the picture can't be read again."""
    if media.rgb_frames is None:
        return []
    from .scene_cuts import content_cuts

    return [frame for frame, _ in content_cuts(media.rgb_frames())]


def _downsampled(d, points=HELD_SERIES_POINTS):
    """The d series as [{frame, d}], at most `points` of them: each kept
    point is the smallest d of its stretch, so a hold still shows."""
    n = len(d)
    if n <= 1:
        return []
    indices = numpy.arange(1, n)
    step = max(1, math.ceil(len(indices) / points))
    series = []
    for start in range(0, len(indices), step):
        chunk = indices[start : start + step]
        values = d[chunk]
        pick = int(chunk[int(numpy.argmin(values))]) if step > 1 else int(chunk[0])
        series.append({"frame": pick, "d": dsp.rounded(float(d[pick]), 3)})
    return series


def analyze_held_frames(video, shots=None, detail=False):
    """Task command: the frames a video repeats while its picture moves - a
    held frame shown twice, the stutter of a take rendered at half its rate.

    Args:
        video: A video file's path, or the video an earlier step returned
        shots: Shot records, accepted for the probes' common signature; the
            measure reads the picture's own hard cuts, not shot boundaries
        detail: Also return the frame-to-frame difference series, downsampled

    Returns:
        {held_frames, held_ratio, frames, clusters: [{first_frame,
        last_frame, held, and start, end in seconds when the video has a
        frame rate}], findings, rules_applied, rules_skipped, shots_source} - with detail,
        also series: [{frame, d}]
    """
    media = media_from(video)
    return held_frames_answer(media, *resolve_shots(video, media, shots), detail=detail)


def held_measure(media):
    """(d series, held frame indices, hard-cut frames) by analyze_held_frames'
    measure. A frame held next to a hard cut is the cut's own, so frames
    within HELD_CUT_GUARD of one (and frame 0, which has no predecessor)
    never count. The cut detector decodes the picture again, so it runs only
    when there is a hold for a cut to explain away: with no hold the cuts
    come back [] unmeasured."""
    frames = media.frame_count
    d = _frame_deltas(media.thumbs) if frames else numpy.full(0, numpy.nan)
    skip = {0}
    held = _held_frames(d, skip)
    cuts = []
    if held:
        cuts = [cut for cut in _cut_frames(media) if _cut_among_holds(d, cut, held)]
        for cut in cuts:
            skip.update(
                range(
                    max(0, cut - HELD_CUT_GUARD),
                    min(frames, cut + HELD_CUT_GUARD + 1),
                )
            )
        held = _held_frames(d, skip)
    return d, held, cuts


def _cut_among_holds(d, cut, held):
    """False for a "cut" that is only a new picture between holds. The cut
    detector calls a change a spike against the median of its neighbours;
    where most of those are held that median is 0 and every new picture of a
    clip on twos or threes spikes. There the change is held against the
    neighbours that did move instead."""
    from .scene_cuts import CUT_SPIKE, CUT_WINDOW

    around = [
        i
        for i in range(max(1, cut - CUT_WINDOW), min(len(d), cut + CUT_WINDOW + 1))
        if i != cut
    ]
    if not around or numpy.nanmedian(d[around]) >= HELD_EPSILON:
        return True
    moved = [d[i] for i in around if i not in set(held) and d[i] >= HELD_EPSILON]
    return bool(moved) and d[cut] >= CUT_SPIKE * float(numpy.median(moved))


def frames_media(frames):
    """A picture-only Media over in-memory frames (a PIL list, an array or
    SegmentedFrames), for measuring frames a task holds rather than a file."""
    thumbs, looks = _in_memory_thumbs(frames)
    return Media(
        thumbs,
        None,
        None,
        None,
        looks=looks,
        rgb_frames=None if thumbs is None else (lambda: _in_memory_rgb(frames)),
    )


def held_frames_answer(media, records, source, detail=False):
    """`analyze_held_frames` over an already-read Media (`held_measure` is
    the measure)."""
    frames = media.frame_count
    d, held, _ = held_measure(media)
    clusters = _held_clusters(held, media.fps)
    answer = {
        "held_frames": len(held),
        "held_ratio": dsp.rounded(len(held) / frames, 4) if frames else 0.0,
        "frames": frames,
        "clusters": clusters,
    }
    if detail:
        answer["series"] = _downsampled(d)
    findings = _findings(
        "analyze_held_frames", {"held_frames": len(held)}, {"file": True}
    )
    if findings and media.fps:  # a finding is placed in seconds
        for cluster in clusters:
            findings.append(
                {
                    "rule": "held_frames",
                    "severity": findings[0]["severity"],
                    "at": cluster["start"],
                    "range": [cluster["start"], cluster["end"]],
                    "value": cluster["held"],
                    "threshold": findings[0]["threshold"],
                    "says": (
                        f"{cluster['held']} held frames between "
                        f"{cluster['start']:g}s and {cluster['end']:g}s"
                    ),
                }
            )
    return _answer("analyze_held_frames", answer, findings, source)


__all__ = [
    "Media",
    "analyze_held_frames",
    "analyze_seams",
    "analyze_shots",
    "analyze_sync_drift",
    "frames_media",
    "held_frames_answer",
    "held_measure",
    "media_from",
    "read_media",
    "resolve_shots",
    "seams_answer",
    "shots_answer",
    "sync_drift_answer",
]
