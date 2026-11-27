"""The held-frames probe: the frames a video repeats while its picture moves (#1527).

`analyze_held_frames` is the fourth assessment probe (`dw/tasks/assess.py`
holds the other three, the `Media` reader and the answer shape they all
share). This module owns the measure itself - `held_measure`, which
`interpolate_frames`' `refill: "held"` reads to find the frames to replace -
the `HELD_*` constants it is tuned by, the clusters and the downsampled d
series `detail` returns, and `held_frames_answer`, the probe over an
already-read Media that `dw/server/assess.py` runs in the server process.

It imports `assess`, never the reverse: a Media comes from `assess.media_from`
or `assess.frames_media`, and the measure only reads one.
"""

import math

import numpy

from .. import dsp
from .assess import (
    TYPICAL_DELTA_FLOOR,
    probe_answer,
    probe_findings,
    media_from,
    resolve_shots,
)

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
    # No neighbour moved at all (two still shots): nothing to hold the cut
    # against, so it stands
    return not moved or d[cut] >= CUT_SPIKE * float(numpy.median(moved))


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
    findings = probe_findings(
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
    return probe_answer("analyze_held_frames", answer, findings, source)


__all__ = [
    "HELD_CLUSTER_GAP",
    "HELD_CLUSTER_MIN",
    "HELD_CUT_GUARD",
    "HELD_EPSILON",
    "HELD_SERIES_POINTS",
    "analyze_held_frames",
    "held_frames_answer",
    "held_measure",
]
