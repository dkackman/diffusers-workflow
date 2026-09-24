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
`LEVEL_SPREAD_WARN_DB`, so the join-time warning and the after-the-fact
probe cannot disagree about the same cut.

Two rules carry a guard the table alone cannot express, applied by the
probe and named here in `unless` so it is written down beside the number:
`seam_hole` holds only while both sides of the seam are voiced, and
`seam_frame_jump` does not fire at a seam whose incoming shot is marked
`hard_cut: true` - a cut meant as a cut.
"""

from .tasks.audio_utils import LEVEL_SPREAD_WARN_DB

SEVERITIES = ("info", "warn")

# How loud both sides of a seam have to be for a quiet join to be a hole
# rather than a pause the shots themselves hold
HOLE_VOICED_DBFS = -30.0

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
        "threshold": 3.0,
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
        "unless": f"either side's rms is at or below {HOLE_VOICED_DBFS} dBFS",
    },
    {
        "name": "seam_frame_jump",
        "probe": "analyze_seams",
        "field": "jump_ratio",
        "comparator": ">",
        "threshold": 8.0,
        "severity": "info",
        "says": "the picture changes this many times more across the seam than inside either shot",
        "unless": "the incoming shot is marked hard_cut: true",
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


__all__ = [
    "HOLE_VOICED_DBFS",
    "RULES",
    "RULES_BY_NAME",
    "SEVERITIES",
    "crosses",
    "finding",
    "rules_for",
]
