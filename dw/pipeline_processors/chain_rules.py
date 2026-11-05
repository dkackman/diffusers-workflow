"""The chain rules that are not MiniMax-H3's: the look hold (#1191) and the
per-segment check with its bounded re-roll (dw/pipeline_processors/chain_check.py).

Torch-free, so validation (dw/guides.py) reads them without loading a model,
and the run (`ChainConfig` in dw/pipeline_processors/chain.py) asks the same
rules. Moved out of dw/pipeline_processors/h3_rules.py, which holds H3's own
rules, when that module reached the size ceiling (#1192).
"""

import math

from .. import references

# The look hold (#1191): pixel corrections applied to each finished segment
# before its carry is taken. Family-independent
LOOK_HOLD_PARTS = ("pickup", "creep", "texture", "precompensate")
# Parts that act on a single carried keyframe, so need `last_frame`
LOOK_HOLD_KEYFRAME_PARTS = ("pickup", "precompensate")


def chain_look_hold_flags(chain):
    """The four look-hold switches as {part: bool}, or None when off. `true`
    turns on all four; an object names the ones it wants."""
    hold = chain.get("look_hold")
    if hold is True:
        return dict.fromkeys(LOOK_HOLD_PARTS, True)
    if isinstance(hold, dict) and any(hold.get(part) is True for part in hold):
        return {part: hold.get(part) is True for part in LOOK_HOLD_PARTS}
    return None


def chain_look_hold_problem(chain):
    """The reason a chain block's `look_hold` is refused, or None: a value that
    is neither a boolean nor an object of the four boolean switches, or an
    explicit `pickup` / `precompensate` on a continuity other than `last_frame`
    (a guide or last_segment carry is a whole clip, not one keyframe - `true`
    alone quietly skips those two there). Unresolved references are left to the
    run."""
    if "look_hold" not in chain:
        return None
    hold = chain["look_hold"]
    if isinstance(hold, str) and references.is_ref(references.UNRESOLVED, hold):
        return None
    expected = (
        "expected true, false or an object of "
        + ", ".join(LOOK_HOLD_PARTS)
        + " booleans"
    )
    if isinstance(hold, bool):
        return None
    if not isinstance(hold, dict):
        return f"invalid look_hold {hold!r} - {expected}"
    unknown = sorted(set(hold) - set(LOOK_HOLD_PARTS))
    if unknown:
        return f"look_hold has unknown key(s) {unknown} - {expected}"
    for part, value in hold.items():
        if not isinstance(value, bool):
            return f"look_hold {part} must be true or false, got {value!r}"
    continuity = chain.get("continuity", "last_frame")
    if isinstance(continuity, str) and references.is_ref(
        references.UNRESOLVED, continuity
    ):
        return None
    if continuity != "last_frame":
        wrong = [p for p in LOOK_HOLD_KEYFRAME_PARTS if hold.get(p)]
        if wrong:
            return (
                f"look_hold {' and '.join(wrong)} need continuity 'last_frame' - "
                f"the {continuity!r} carry is a whole clip, not one keyframe; "
                "creep and texture work on any continuity"
            )
    return None


# The per-segment check and its bounded re-roll (chain_check.py): the fields a
# `check` object may set (`face` is reserved, null only) and the most re-rolls
# a checked segment may take. Family-independent
CHAIN_CHECK_FIELDS = (
    "jump_ratio_max",
    "look_ratio_max",
    "texture_ratio_max",
    "cut_ratio_max",
    "face",
)
CHAIN_RETRIES_MAX = 3


def _number_over(value, floor):
    """Whether `value` is a finite number above `floor` (never a bool), or an
    unresolved reference, which is left to the run."""
    if isinstance(value, str):
        return references.is_ref(references.UNRESOLVED, value)
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > floor
    )


def chain_check_problem(chain):
    """The reason a chain block's `check` or `retries` is refused, or None.
    `check` is true (the default ceilings), false, or an object of
    `CHAIN_CHECK_FIELDS`: the ratio ceilings above 0, `texture_ratio_max`
    above 1 (it bounds the ratio both ways) or null, `face` null. `retries`
    is a whole number from 0 to `CHAIN_RETRIES_MAX`; above 0 it needs the
    check on, as with no check nothing fails. Unresolved references are left
    to the run, whose `ChainConfig` asks this same rule."""
    check = chain.get("check")
    expected = "expected true, false or an object of " + ", ".join(CHAIN_CHECK_FIELDS)
    if isinstance(check, dict):
        unknown = sorted(set(check) - set(CHAIN_CHECK_FIELDS))
        if unknown:
            return f"check has unknown key(s) {unknown} - {expected}"
        for name in ("jump_ratio_max", "look_ratio_max"):
            if name in check and not _number_over(check[name], 0):
                return f"check {name} must be a number above 0, got {check[name]!r}"
        texture = check.get("texture_ratio_max")
        if texture is not None and not _number_over(texture, 1):
            return (
                "check texture_ratio_max bounds the fine-band ratio both ways, "
                f"so must be a number above 1 or null, got {texture!r}"
            )
        cut = check.get("cut_ratio_max")
        if cut is not None and not _number_over(cut, 0):
            return f"check cut_ratio_max must be a number above 0 or null, got {cut!r}"
        if check.get("face") is not None:
            return f"check face is reserved and must be null, got {check['face']!r}"
    elif not (
        check is None
        or isinstance(check, bool)
        or (isinstance(check, str) and _number_over(check, 0))
    ):
        return f"invalid check {check!r} - {expected}"
    retries = chain.get("retries", 0)
    if isinstance(retries, str) and _number_over(retries, 0):
        return None
    whole = isinstance(retries, int) and not isinstance(retries, bool)
    if not whole or not 0 <= retries <= CHAIN_RETRIES_MAX:
        return (
            f"retries must be a whole number between 0 and {CHAIN_RETRIES_MAX}, "
            f"got {retries!r}"
        )
    if retries and (check is None or check is False):
        return (
            f"retries {retries} needs check on - a re-roll replaces a segment "
            "its check failed, and with no check no segment fails"
        )
    return None


# The experimental late capstone (#1187, #1199 Task 14 Step 4): segment 1's
# frame `from_end` frames before its end, put one latent past every later
# segment's end as a guide that enters the forward below `enter_below_sigma`.
# A scratch spike behind the windowed refine's flag (h3_rules), MiniMax-H3 only
CHAIN_CAPSTONE = "experimental_capstone"
CHAIN_CAPSTONE_FIELDS = ("enter_below_sigma", "from_end")
CHAIN_CAPSTONE_FROM_END = 5


def chain_capstone_problem(chain):
    """The reason a chain block's `experimental_capstone` is refused, or None:
    the flag off, unknown keys, a gate not in (0, 1], a `from_end` not a whole
    number from 1. Unresolved references are left to the run."""
    capstone = chain.get(CHAIN_CAPSTONE)
    if capstone is None:
        return None
    from .h3_rules import REFINE_WINDOWS_ENV_VAR, refine_windows_enabled

    if not refine_windows_enabled():
        return (
            f"{CHAIN_CAPSTONE} is a scratch spike, off unless "
            f"{REFINE_WINDOWS_ENV_VAR}=1 is set where the pipeline loads"
        )
    if not isinstance(capstone, dict) or "enter_below_sigma" not in capstone:
        return f"{CHAIN_CAPSTONE} is {{enter_below_sigma, from_end?}}, got {capstone!r}"
    unknown = sorted(set(capstone) - set(CHAIN_CAPSTONE_FIELDS))
    if unknown:
        return f"{CHAIN_CAPSTONE} has unknown key(s) {unknown}"
    gate = capstone["enter_below_sigma"]
    if not _number_over(gate, 0) or (not isinstance(gate, str) and gate > 1):
        return f"{CHAIN_CAPSTONE} enter_below_sigma is a sigma in (0, 1], got {gate!r}"
    back = capstone.get("from_end", CHAIN_CAPSTONE_FROM_END)
    if isinstance(back, str) and _number_over(back, 0):
        return None
    if isinstance(back, bool) or not isinstance(back, int) or back < 1:
        return f"{CHAIN_CAPSTONE} from_end is a whole number from 1, got {back!r}"
    return None
