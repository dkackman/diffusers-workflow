"""The rules the assessment probes read their measurements against (#387).

A probe (`dw/tasks/assess.py`) measures a finished file and reports every
number it took; a rule here names one of those numbers and the threshold
past which it is worth a look, and a crossing becomes a `finding` in the
probe's answer. Findings are places to look, not verdicts: nothing in the
engine acts on one, no run fails for one, and a finding a person has looked
at and accepted is simply left alone. That is the authority rule, and it is
why the thresholds sit in one table - a number an agent is told to trust
has to be one somebody can find, argue with and change in one place.

Each entry names the probe that reports the field, the field (a key of the
probe's per-shot or per-seam record, or of its answer itself -
`tests/test_assessment_rules.py` pins every one to a real probe field so a
rename cannot leave a rule reading nothing), how the value is compared,
the threshold, the severity and what a crossing says. `magnitude` compares
the value's absolute size, for a signed measurement whose direction is not
the problem. `shot_level_spread` shares its threshold with the run-time
`LEVEL_SPREAD_WARN_DB`, and `seam_level_step` with `INNER_LEVEL_STEP_WARN_DB`
(concat_videos' warning for a multi-shot input), so the join-time warning
and the after-the-fact probe cannot disagree about the same cut.

Four rules carry a guard the table alone cannot express, applied by the
probe and named here in `unless` so it is written down beside the number:
`seam_hole` holds only while both sides of the seam are voiced,
`seam_frame_jump` does not fire at a seam whose incoming shot is marked
`hard_cut: true` - a cut meant as a cut - and `shot_dead_air` and
`shot_bed_drop` do not fire inside a shot whose own level is already at or
below `HOLE_VOICED_DBFS` - a shot that is quiet throughout, on purpose,
rather than one holding a gap.

The thresholds were settled against real runs on the server in stage D
(#386), and `held_frames`'s against stage A of #1527; `docs/WORKFLOW_GUIDE.md` and `docs/TASKS.md` quote every one, and
`tests/test_assessment_rules.py` pins those quotes to this table.
"""

from .tasks.joins import INNER_LEVEL_STEP_WARN_DB, LEVEL_SPREAD_WARN_DB

SEVERITIES = ("info", "warn")

# How loud both sides of a seam have to be for a quiet join to be a hole
# rather than a pause the shots themselves hold
HOLE_VOICED_DBFS = -30.0

# Below this, a 50ms window inside a shot counts toward a dead-air run - low
# enough that ordinary room tone under a line (well above this) never
# qualifies, high enough to catch the -70 to -84 dBFS gaps an H3 dialogue
# take leaves between lines (#465)
DEAD_AIR_FLOOR_DBFS = -65.0

COMPARATORS = {
    ">": lambda value, threshold: value > threshold,
    ">=": lambda value, threshold: value >= threshold,
    "<": lambda value, threshold: value < threshold,
}

RULES = (
    {
        "name": "shot_level_spread",
        "probe": "analyze_shots",
        "field": "rms_range_db",
        "comparator": ">=",
        "threshold": LEVEL_SPREAD_WARN_DB,
        "severity": "warn",
        "says": "the shots sit this far apart in level - a jump a listener hears at the cut",
    },
    {
        "name": "seam_level_step",
        "probe": "analyze_seams",
        "field": "level_step_db",
        "comparator": ">",
        "threshold": INNER_LEVEL_STEP_WARN_DB,
        "severity": "warn",
        "says": "the shots either side of the seam sit this far apart in level",
    },
    {
        "name": "seam_click",
        "probe": "analyze_seams",
        "field": "click_db",
        "comparator": ">",
        "threshold": 12.0,
        "severity": "warn",
        "says": "the join peaks this far above the audio either side of it - an audible click",
    },
    {
        "name": "seam_hole",
        "probe": "analyze_seams",
        "field": "floor_dbfs",
        "comparator": "<",
        "threshold": -50.0,
        "severity": "warn",
        "says": "the track drops out at the join while both sides are voiced",
        "unless": (
            f"either side's rms is at or below {HOLE_VOICED_DBFS} dBFS, or the"
            " incoming shot is marked seam_fade_ms - a fade the caller asked for"
        ),
    },
    {
        "name": "seam_frame_jump",
        "probe": "analyze_seams",
        "field": "jump_ratio",
        "comparator": ">",
        # Above every intended cut the stage D field runs measured (#386):
        # seven ordinary cuts between generated shots read 9.9-20.3, dissolves
        # about 1.8. At the original 8 the rule fired on every cut
        "threshold": 25.0,
        "severity": "info",
        "says": "the picture changes this many times more across the seam than inside either shot",
        "unless": "the incoming shot is marked hard_cut: true",
    },
    {
        "name": "shot_dead_air",
        "probe": "analyze_shots",
        "field": "dead_air_seconds",
        "comparator": ">",
        "threshold": 0.4,
        "severity": "warn",
        "says": (
            "the shot holds a gap this long, well below the floor a line's own room"
            " tone sits at - find_loop_bed on the cut names a quiet stretch inside"
            " one shot and the gain to lay it at; cut it with slice_audio, loop it"
            " to the gap's length with loop_audio, and mix it under the line with"
            " mix_audio rather than leaving the drop silent"
        ),
        "unless": f"the shot's own rms is at or below {HOLE_VOICED_DBFS} dBFS",
    },
    {
        # One field case (#1620): a storm set's bed near -25 dBFS in the
        # shots either side, and a voiced shot's gaps between lines at -52 -
        # above DEAD_AIR_FLOOR_DBFS, so shot_dead_air never saw it. Against
        # the louder neighbour, so one quiet neighbour does not hide the drop
        "name": "shot_bed_drop",
        "probe": "analyze_shots",
        "field": "floor_below_neighbours_db",
        "comparator": ">",
        "threshold": 20.0,
        "severity": "warn",
        "says": (
            "the shot's floor sits this far below the louder neighbouring shot's -"
            " the bed the shots around it carry drops out between its lines. Lay"
            " the bed under it: run find_loop_bed on the cut with max_bin_dbfs,"
            " max_mean_dbfs and target_bed_dbfs raised to find a loud bed, then"
            " rerun assemble-and-score with its pick's start_seconds,"
            " duration_seconds and gain as bed_start_seconds, bed_seconds and"
            " bed_gain - the template cuts that stretch with slice_audio, loops"
            " it with loop_audio and mixes it under the cut with mix_audio (a"
            " hand-built cut lays the same chain itself). Leave it at a"
            " deliberate scene change, where the set itself goes quiet"
        ),
        "unless": f"the shot's own rms is at or below {HOLE_VOICED_DBFS} dBFS",
    },
    {
        "name": "sync_drift",
        "probe": "analyze_sync_drift",
        "field": "end_offset_ms",
        "comparator": ">",
        "threshold": 40.0,
        "severity": "warn",
        "magnitude": True,
        "says": "by this shot's end the audio sits this far off the picture",
    },
    {
        "name": "sync_length",
        "probe": "analyze_sync_drift",
        "field": "length_delta_ms",
        "comparator": ">",
        "threshold": 40.0,
        "severity": "warn",
        "magnitude": True,
        "says": "the soundtrack and the picture differ in length by this much",
    },
    {
        # The lowest held count among the takes #1527 stage A's record
        # (bench/records/held-frames.md) flags with a cluster, at the
        # probe's HELD_EPSILON of 1.0
        "name": "held_frames",
        "probe": "analyze_held_frames",
        "field": "held_frames",
        "comparator": ">=",
        "threshold": 13,
        "severity": "warn",
        "says": (
            "the take repeats this many frames while its picture moves - it"
            " stutters; the clusters say where. Reroll the take or re-plan"
            " the shot"
        ),
    },
)

RULES_BY_NAME = {rule["name"]: rule for rule in RULES}


def rules_for(probe):
    """The rules one probe's measurements are read against, in table order."""
    return [rule for rule in RULES if rule["probe"] == probe]


def crosses(rule, value):
    """Whether a measured value crosses a rule's threshold. None never does -
    a measurement that could not be taken (a silent window, no soundtrack)
    is not a finding."""
    if value is None:
        return False
    if rule.get("magnitude"):
        value = abs(value)
    return COMPARATORS[rule["comparator"]](value, rule["threshold"])


def finding(rule, value, at):
    """A crossing as the probe reports it."""
    return {
        "rule": rule["name"],
        "severity": rule["severity"],
        "at": at,
        "value": value,
        "threshold": rule["threshold"],
        "says": rule["says"],
    }


def sort_findings(findings):
    """Findings with every `info` moved after the `warn`s, order otherwise
    unchanged - a real problem (`seam_hole`, `shot_dead_air`, ...) never
    sits behind an informational one (`seam_frame_jump`) a reader skims
    past first (#466)."""
    return sorted(findings, key=lambda found: found.get("severity") == "info")


__all__ = [
    "DEAD_AIR_FLOOR_DBFS",
    "HOLE_VOICED_DBFS",
    "RULES",
    "RULES_BY_NAME",
    "SEVERITIES",
    "crosses",
    "finding",
    "rules_for",
    "sort_findings",
]
