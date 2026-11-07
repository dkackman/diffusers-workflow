"""The per-segment check a chain runs on each segment before keeping it, and
the policy of its bounded re-roll.

After segment k renders (k > 1), its kept frames are measured against
segment k - 1's - the seam's picture jump and look step, by the arithmetic
`analyze_seams` uses (`dw/tasks/assess.py` `seam_measures`) - and against
segment 1's, for fine-band texture drift (`dw/texture_bands.py`). A segment
over any ceiling fails, naming the measures; `run_chain` re-renders it from
the same carry with `reroll_seed`'s seed up to the chain's `retries`, and
keeps `best_attempt` of what it rendered. Segment 1 has no seam before it
and is never checked. `check_segment` can skip a seam marked `hard_cut`, but
`run_chain` never passes it: a chain's own seams are never marked cuts.

Measurement and choice only: nothing here renders, and nothing here touches
the timeline. The chain keys are validated by `chain_rules.chain_check_problem`.
"""

import math
from dataclasses import dataclass, field

import numpy

# The default ceilings, set from the 99th percentile of each seam measure
# across the noise-floor runs of S1-S3 (bench/records/noise-floor.md, "Chain
# check thresholds"; 28 seams of unchecked chains, measured with this
# module's own arithmetic). The jump ceiling is about twice its P99 (2.02),
# so 4.0; the look ceiling is doubled and rounded up to half a point:
# - jump_ratio: P99 2.02 -> 4.0. Well under `seam_frame_jump`'s own threshold
#   (25, dw/assessment_rules.py) and the 9.9-20.3 the stage D runs measured
#   for an intended cut, above a dissolve's 1.8 - a chain seam over it is a
#   cut nobody asked for
# - look_ratio: P99 2.67 -> 5.5. The calibration
#   (docs/proposals/complete/seam-look-measures-complete.md) read 0.64 for a
#   clip joined to itself and 19.2 for a saturation-1.6 grade, so 5.5 is
#   well under a visible grade
# - texture_ratio: still PROVISIONAL, 2.0, the segment's fine-band luma
#   against segment 1's, either way - halved (blur) or doubled (crust). The
#   noise runs cannot set it: the two same-seed runs of S1 and S2 are
#   identical (no spread), and the unchecked chains' own ratio is the drift
#   the check exists to catch (P99 3.0x, up to 2.4x in S1 and 0.32x in S2
#   by segment 10), so a percentile of it would raise the ceiling to the
#   very defect. It is set from a checked chain's runs, not these
JUMP_RATIO_MAX = 4.0
LOOK_RATIO_MAX = 5.5
TEXTURE_RATIO_MAX = 2.0
# - cut_ratio: the largest frame step inside the segment over its own typical
#   step - a seam nobody marked. The seam measures above miss a segment that
#   cuts away inside itself (bench/records/check.md: an S3 re-roll cut to
#   another scene 26 frames past its seam and passed). 282 segments of 47 bench
#   takes read P50 1.22, P99 2.62, max 2.98 (bench/cuts.py); the two cuts in
#   that take read 4.16 and 17.2. The seam's own ceiling, 4.0, sits between
CUT_RATIO_MAX = 4.0

# The measures a check can fail, in the order a failure names them
MEASURES = ("jump_ratio", "look_ratio", "texture_ratio", "cut_ratio")

# Measures that warn and record but never re-roll a segment (ruling R38):
# texture drift is in the carry, so a fresh seed from the same carry inherits
# it (bench/records/check.md). The others a new seed can fix
WARN_ONLY = ("texture_ratio",)

# Frames of a segment, evenly spaced, its fine-band texture is the mean of
TEXTURE_SAMPLES = 4

# Attempt n of a segment (0 is the first render) runs at the step's seed
# plus n times this prime, so a re-roll is reproducible from the step's seed
REROLL_PRIME = 1000003


@dataclass(frozen=True)
class SegmentCheck:
    """The ceilings one segment is held to. `texture_ratio_max` bounds the
    fine-band ratio both ways (None skips it); `face` is reserved."""

    jump_ratio_max: float = JUMP_RATIO_MAX
    look_ratio_max: float = LOOK_RATIO_MAX
    texture_ratio_max: float | None = TEXTURE_RATIO_MAX
    cut_ratio_max: float | None = CUT_RATIO_MAX
    face: None = None


@dataclass
class CheckResult:
    passed: bool
    measures: dict
    failed: list = field(default_factory=list)

    @property
    def rerolls(self):
        """Whether a re-roll could help: a failed measure that is not warn-only."""
        return any(name not in WARN_ONLY for name in self.failed)


def segment_check(value):
    """A chain block's `check` as a SegmentCheck, or None when off: true is
    the defaults, an object overrides the fields it names. The value has
    passed `chain_rules.chain_check_problem`."""
    if value is None or value is False:
        return None
    if value is True:
        return SegmentCheck()
    fields = {
        name: float(value[name])
        for name in ("jump_ratio_max", "look_ratio_max")
        if name in value
    }
    for name in ("texture_ratio_max", "cut_ratio_max"):
        if name in value:
            fields[name] = None if value[name] is None else float(value[name])
    return SegmentCheck(**fields)


def texture_samples(frames):
    """The TEXTURE_SAMPLES evenly spaced frames the texture measure reads -
    all a chain needs to keep of segment 1."""
    if len(frames) <= TEXTURE_SAMPLES:
        return list(frames)
    picks = numpy.linspace(0, len(frames) - 1, TEXTURE_SAMPLES).round().astype(int)
    return [frames[i] for i in picks]


def check_segment(previous_frames, frames, segment_1_frames, check, hard_cut=False):
    """Measure a segment's kept frames against the segment before it and
    segment 1, as a CheckResult. `hard_cut` marks a seam meant as a cut,
    which is never checked."""
    if hard_cut:
        return CheckResult(True, {"skipped": "hard_cut"}, [])
    from ..tasks.assess import frame_measures, seam_measures

    before_thumbs, before_looks = frame_measures(previous_frames)
    after_thumbs, after_looks = frame_measures(frames)
    if before_thumbs is None or after_thumbs is None:
        return CheckResult(True, {"skipped": "no frames"}, [])
    seam = seam_measures(
        numpy.concatenate([before_thumbs, after_thumbs]),
        numpy.concatenate([before_looks, after_looks]),
        {"start_frame": 0, "num_frames": len(before_thumbs)},
        {"start_frame": len(before_thumbs), "num_frames": len(after_thumbs)},
    )
    measures = {
        "jump_ratio": seam["jump_ratio"],
        "frame_delta": seam["frame_delta"],
        "typical_delta": seam["typical_delta"],
        "look_ratio": seam["look_ratio"],
        "look_ratio_measure": seam["look_ratio_measure"],
        "texture_ratio": _texture_ratio(frames, segment_1_frames),
        "cut_ratio": _cut_ratio(after_thumbs),
    }
    failed = []
    if _over(measures["jump_ratio"], check.jump_ratio_max):
        failed.append("jump_ratio")
    if _over(measures["look_ratio"], check.look_ratio_max):
        failed.append("look_ratio")
    texture = measures["texture_ratio"]
    if (
        check.texture_ratio_max is not None
        and texture is not None
        and max(texture, 1 / texture if texture else math.inf) > check.texture_ratio_max
    ):
        failed.append("texture_ratio")
    if check.cut_ratio_max is not None and _over(
        measures["cut_ratio"], check.cut_ratio_max
    ):
        failed.append("cut_ratio")
    return CheckResult(not failed, measures, failed)


def best_attempt(results):
    """The index of the attempt to keep: fewest failed measures, then the
    lowest jump ratio, then the earliest."""

    def rank(index):
        result = results[index]
        jump = result.measures.get("jump_ratio")
        return (len(result.failed), math.inf if jump is None else jump, index)

    return min(range(len(results)), key=rank)


def reroll_seed(seed, attempt):
    """The seed attempt `attempt` of a segment runs at (0 is the first)."""
    return seed + attempt * REROLL_PRIME


def _cut_ratio(thumbs):
    """The largest frame step inside a segment over its own typical step,
    floored as a seam's is - `_seam_video`'s arithmetic on a seam inside the
    segment - or None for fewer than three frames. The typical step is the
    90th percentile, so at 3-5 frames (2-4 steps) it sits within a few percent
    of the largest step and the ratio cannot reach the 4.0 ceiling: a short
    segment cannot fail this measure."""
    from ..tasks.assess import TYPICAL_DELTA_FLOOR, TYPICAL_DELTA_PERCENTILE

    if thumbs.shape[0] < 3:
        return None
    deltas = numpy.abs(numpy.diff(thumbs.astype(numpy.int16), axis=0)).mean(axis=(1, 2))
    typical = max(
        TYPICAL_DELTA_FLOOR, float(numpy.percentile(deltas, TYPICAL_DELTA_PERCENTILE))
    )
    return round(float(deltas.max()) / typical, 2)


def _over(value, ceiling):
    return value is not None and value > ceiling


def _fine_luma(frames):
    """The mean fine-band luma of `frames`' texture samples, or None when a
    frame is too small to hold the band."""
    from ..texture_bands import MIN_SIDE, texture_bands

    values = []
    for frame in texture_samples(frames):
        # A chain's frames are PIL images; an array is taken as RGB already
        rgb = numpy.asarray(
            frame.convert("RGB") if hasattr(frame, "convert") else frame
        )
        if min(rgb.shape[:2]) < MIN_SIDE:
            return None
        values.append(texture_bands(rgb)["luma"]["fine"])
    return float(numpy.mean(values)) if values else None


def _texture_ratio(frames, segment_1_frames):
    """The segment's fine-band luma over segment 1's, or None where either
    has none to measure."""
    base = _fine_luma(segment_1_frames)
    value = _fine_luma(frames)
    if not base or value is None:
        return None
    return round(value / base, 4)
