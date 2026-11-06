"""The chain rules that are not MiniMax-H3's: the look hold (#1191) and the
per-segment check with its bounded re-roll (dw/pipeline_processors/chain_check.py).

Torch-free, so validation (dw/guides.py) reads them without loading a model,
and the run (`ChainConfig` in dw/pipeline_processors/chain_config.py) asks the same
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


def chain_capstone_problem(chain, num_frames=None):
    """The reason a chain block's `experimental_capstone` is refused, or None:
    the flag off, unknown keys, a gate not in (0, 1], a `from_end` not a whole
    number from 1, or past segment 1's `num_frames` when that is known.
    Unresolved references are left to the run."""
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
    if isinstance(num_frames, int) and back > num_frames:
        return (
            f"{CHAIN_CAPSTONE} from_end {back} is past the segment's "
            f"{num_frames} frames"
        )
    return None


# The experimental latent carry (#1150 stage A, the latent-carry arm of #1044;
# #1199 Task 17 Step 2): a guide chain hands each segment's own final latents,
# the tail its guide would have decoded and re-encoded, to the next segment as
# that guide's condition rows (h3_guides.latent_guide_tail). A scratch spike
# behind the windowed refine's flag (h3_rules), MiniMax-H3 guide chains only
CHAIN_LATENT_CARRY = "experimental_latent_carry"


def chain_latent_carry_problem(chain, guide_noise=None, hold_audio=None):
    """The reason a chain block's `experimental_latent_carry` is refused, or
    None: the flag off; a value that is not true or false; a continuity other
    than `guide` (the latents replace a guide's encode, and only a guide chain
    has one); a `look_hold`, which corrects the decoded frames the carry would
    no longer read; `match_audio`, whose held track is what the guide's audio
    would be (the spike carries generated audio latents only); or the step's
    nonzero `guide_noise`, which noises pixels that are never encoded; or a
    step `hold_audio`, the track the guide's audio would be. False,
    null and absent are off. Unresolved references are left to the run."""
    carry = chain.get(CHAIN_LATENT_CARRY)
    if carry is None or carry is False:
        return None
    from .h3_rules import (
        GUIDE_CONTINUITY,
        REFINE_WINDOWS_ENV_VAR,
        refine_windows_enabled,
    )

    if isinstance(carry, str) and references.is_ref(references.UNRESOLVED, carry):
        return None
    if not refine_windows_enabled():
        return (
            f"{CHAIN_LATENT_CARRY} is a scratch spike, off unless "
            f"{REFINE_WINDOWS_ENV_VAR}=1 is set where the pipeline loads"
        )
    if carry is not True:
        return f"{CHAIN_LATENT_CARRY} is true or false, got {carry!r}"
    continuity = chain.get("continuity", "last_frame")
    unresolved = isinstance(continuity, str) and references.is_ref(
        references.UNRESOLVED, continuity
    )
    if not unresolved and continuity != GUIDE_CONTINUITY:
        return (
            f"{CHAIN_LATENT_CARRY} carries a guide's latents, so it needs "
            f"continuity 'guide', got {continuity!r}"
        )
    if chain_look_hold_flags(chain):
        return (
            f"{CHAIN_LATENT_CARRY} does not combine with look_hold: the hold "
            "corrects the decoded frames, and the carry hands on the latents "
            "the correction never reaches"
        )
    if chain.get("match_audio") is True:
        return (
            f"{CHAIN_LATENT_CARRY} does not combine with match_audio: the "
            "guide's audio would be the held track, and the spike carries "
            "generated audio latents only"
        )
    if hold_audio is not None and not (
        isinstance(hold_audio, str)
        and references.is_ref(references.UNRESOLVED, hold_audio)
    ):
        return (
            f"{CHAIN_LATENT_CARRY} does not combine with hold_audio: the "
            "guide's audio would be the held track, and the spike carries "
            "generated audio latents only"
        )
    if _number_over(guide_noise, 0) and not isinstance(guide_noise, str):
        return (
            f"{CHAIN_LATENT_CARRY} does not combine with guide_noise "
            f"{guide_noise}: the noise goes on a guide's pixels before their "
            "encode, and a latent guide is never encoded"
        )
    return None


# The experimental keystone slot (#1187/#1188 follow-on, #1199 Task 17): where
# a guide chain's `keystone: "segment_1"` frame sits in each later segment.
# "first" (today's, the default) is the fl2va keyframe, at rotary time 0 - the
# slot of the guide's own first latent, so the two condition rows contradict
# each other. "last" makes it the fl2va `last_image`, the last-latent anchor
# the model is trained on; {"frame": 17j, "enter_below_sigma"?} a one-frame
# guide on a chunk boundary past the carried tail, optionally gated like the
# capstone. Off either way, the keyframe is the guide's first frame again. A
# scratch spike behind the windowed refine's flag (h3_rules), MiniMax-H3 only
CHAIN_KEYSTONE_SLOT = "experimental_keystone_slot"
KEYSTONE_SLOT_FIRST = "first"
KEYSTONE_SLOT_LAST = "last"
KEYSTONE_SLOT_FIELDS = ("frame", "enter_below_sigma")


def _unresolved(value):
    return isinstance(value, str) and references.is_ref(references.UNRESOLVED, value)


def chain_keystone_slot_problem(chain, num_frames=None, arguments=None):
    """The reason a chain block's `experimental_keystone_slot` is refused, or
    None. Absent, null and "first" are today's keystone. Otherwise refused:
    the flag off; a keystone other than `segment_1`; a value other than
    "first", "last" or {frame, enter_below_sigma?}; a frame off the 17-frame
    grid, inside the carried guide (it would contradict a guide row again) or
    past the segment's `num_frames` when known; a gate not in (0, 1]. With
    the step's `arguments` (the run), "last" also needs the fl2va keyframe
    and refuses a `last_image` of the step's own. Unresolved references are
    left to the run."""
    slot = chain.get(CHAIN_KEYSTONE_SLOT)
    if slot is None or slot == KEYSTONE_SLOT_FIRST or _unresolved(slot):
        return None
    from .h3_rules import (
        GUIDE_CHAIN_DEFAULT,
        KEYSTONE_SEGMENT_1,
        REFINE_WINDOWS_ENV_VAR,
        guide_frame_problem,
        refine_windows_enabled,
    )

    if not refine_windows_enabled():
        return (
            f"{CHAIN_KEYSTONE_SLOT} is a scratch spike, off unless "
            f"{REFINE_WINDOWS_ENV_VAR}=1 is set where the pipeline loads"
        )
    keystone = chain.get("keystone")
    if keystone != KEYSTONE_SEGMENT_1 and not _unresolved(keystone):
        return (
            f"{CHAIN_KEYSTONE_SLOT} places the keystone, so it needs keystone "
            f"'segment_1', got {keystone!r}"
        )
    if slot == KEYSTONE_SLOT_LAST:
        return _last_slot_problem(chain, arguments)
    if not isinstance(slot, dict):
        return (
            f"{CHAIN_KEYSTONE_SLOT} is one of {KEYSTONE_SLOT_FIRST}, "
            f"{KEYSTONE_SLOT_LAST} or {{frame, enter_below_sigma?}}, got {slot!r}"
        )
    unknown = sorted(set(slot) - set(KEYSTONE_SLOT_FIELDS))
    if unknown:
        return f"{CHAIN_KEYSTONE_SLOT} has unknown key(s) {unknown}"
    if "frame" not in slot:
        return f"{CHAIN_KEYSTONE_SLOT} {{frame, enter_below_sigma?}} needs a frame"
    gate = slot.get("enter_below_sigma")
    if gate is not None and (
        not _number_over(gate, 0) or (not isinstance(gate, str) and gate > 1)
    ):
        return (
            f"{CHAIN_KEYSTONE_SLOT} enter_below_sigma is a sigma in (0, 1], "
            f"got {gate!r}"
        )
    frame = slot["frame"]
    if _unresolved(frame):
        return None
    problem = guide_frame_problem(frame)
    if problem:
        return f"{CHAIN_KEYSTONE_SLOT} {problem}"
    guide_frames = chain.get("guide_frames", GUIDE_CHAIN_DEFAULT)
    if isinstance(guide_frames, int) and frame < guide_frames:
        return (
            f"{CHAIN_KEYSTONE_SLOT} frame {frame} must be past the guide's "
            f"{guide_frames} frames, or it lands on a guide row again"
        )
    if isinstance(num_frames, int) and frame >= num_frames:
        return (
            f"{CHAIN_KEYSTONE_SLOT} frame {frame} is past the segment's "
            f"{num_frames} frames"
        )
    return None


def keystone_slot_plan_problem(slot, plan):
    """The reason a checked frame slot (`chain_keystone_slot`) does not fit a
    planned later segment, or None. A match_audio chain renders its last
    segment only as long as the track has left (snapped up to a length the
    pipeline takes), which `chain_keystone_slot_problem` cannot know from
    the step's `num_frames` - so the run asks again of the plan before
    segment 1 renders, rather than failing on the last segment after every
    earlier one has."""
    if not isinstance(slot, dict):
        return None
    frame = slot["frame"]
    for segment in plan[1:]:
        if segment.num_frames is not None and frame >= segment.num_frames:
            return (
                f"{CHAIN_KEYSTONE_SLOT} frame {frame} is past segment "
                f"{segment.index + 1}'s {segment.num_frames} frames - the "
                f"matched audio leaves the last segment that short; move the "
                f"slot below {segment.num_frames} or use 'last'"
            )
    return None


def _last_slot_problem(chain, arguments):
    if arguments is None:
        return None
    if arguments.get("last_image") is not None:
        return (
            f"{CHAIN_KEYSTONE_SLOT} 'last' sets every later segment's "
            "last_image, and the step passes one of its own - drop one"
        )
    if arguments.get(chain.get("segment_argument", "image")) is None:
        return (
            f"{CHAIN_KEYSTONE_SLOT} 'last' is an fl2va last-frame anchor, and "
            "the step has no keyframe - it needs fl2va with an image"
        )
    return None


def chain_keystone_slot(chain):
    """A checked `experimental_keystone_slot` as the run uses it: None for
    today's keystone, "last", or {"frame": int, "enter_below_sigma": float or
    None}."""
    slot = chain.get(CHAIN_KEYSTONE_SLOT)
    if slot is None or slot == KEYSTONE_SLOT_FIRST:
        return None
    if slot == KEYSTONE_SLOT_LAST:
        return slot
    gate = slot.get("enter_below_sigma")
    return {
        "frame": int(slot["frame"]),
        "enter_below_sigma": None if gate is None else float(gate),
    }


def place_keystone(arguments, keystone, slot):
    """Put segment 1's keystone frame into a later segment's `arguments` at
    `slot` (`chain_keystone_slot`, not None): as its `last_image`, or as a
    one-frame guide after the carried one."""
    if slot == KEYSTONE_SLOT_LAST:
        arguments["last_image"] = keystone
        return
    guide = {"video": [keystone], "frame": slot["frame"]}
    if slot["enter_below_sigma"] is not None:
        guide["enter_below_sigma"] = slot["enter_below_sigma"]
    # A copy: iteration arguments share nested values
    arguments["guides"] = list(arguments.get("guides") or []) + [guide]
