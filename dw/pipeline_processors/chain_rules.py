"""The chain rules that are not MiniMax-H3's: the look hold (#1191) and the
per-segment check with its bounded re-roll (dw/pipeline_processors/chain_check.py).

Torch-free, so validation (dw/guides.py) reads them without loading a model,
and the run (`ChainConfig` in dw/pipeline_processors/chain_config.py) asks the same
rules. Moved out of dw/pipeline_processors/h3_rules.py, which holds H3's own
rules, when that module reached the size ceiling (#1192); the carry pick
(#1032), family-independent, followed it there.
"""

import math

from .. import references

# The look hold (#1191): pixel corrections applied to each finished segment
# before its carry is taken. Family-independent
LOOK_HOLD_PARTS = ("pickup", "creep", "texture", "precompensate")
# Parts that act on a single carried keyframe, so need `last_frame`
LOOK_HOLD_KEYFRAME_PARTS = ("pickup", "precompensate")


# The carry pick (#1032): the sharpest of the last N frames as the next
# keyframe, the segment trimmed after it. Family-independent, `last_frame` only
CARRY_PICK_MIN = 1
CARRY_PICK_MAX = 8


def chain_carry_pick_problem(chain, num_frames=None):
    """The reason a chain block's `carry_pick` is refused, or None: a shape
    other than {"sharpest": N}, N outside 1..8, a continuity other than
    `last_frame`, `match_audio` (the plan is fixed from the track's length and
    the pick drops frames per seam), or - when the step's `num_frames` is known
    - an N that could empty a segment. Unresolved references are left to the
    run."""
    if "carry_pick" not in chain:
        return None
    from ..variable_constraints import aligned
    from .h3_rules import _set_for_plan

    pick = chain["carry_pick"]
    if isinstance(pick, str) and references.is_ref(references.UNRESOLVED, pick):
        return None
    expected = (
        f"expected {{'sharpest': N}} with N from {CARRY_PICK_MIN} to {CARRY_PICK_MAX}"
    )
    if not isinstance(pick, dict) or set(pick) != {"sharpest"}:
        return f"invalid carry_pick {pick!r} - {expected}"
    window = pick["sharpest"]
    if isinstance(window, str) and references.is_ref(references.UNRESOLVED, window):
        return None
    if (
        isinstance(window, bool)
        or not isinstance(window, int)
        or not CARRY_PICK_MIN <= window <= CARRY_PICK_MAX
    ):
        return (
            f"carry_pick sharpest must be a whole number between {CARRY_PICK_MIN} "
            f"and {CARRY_PICK_MAX}, got {window!r}"
        )
    continuity = chain.get("continuity", "last_frame")
    if isinstance(continuity, str) and references.is_ref(
        references.UNRESOLVED, continuity
    ):
        return None
    if continuity != "last_frame":
        return (
            f"carry_pick needs continuity 'last_frame' - the other modes carry "
            f"more than one frame; got {continuity!r}"
        )
    if _set_for_plan(chain, "match_audio", "duration_seconds"):
        return (
            "carry_pick does not combine with match_audio or duration_seconds: "
            "the plan covers the length up front and the pick drops up to "
            f"{CARRY_PICK_MAX - 1} more frames per seam, leaving the video "
            "shorter than asked - use 'segments'"
        )
    trim = chain.get("trim_frames", 1)
    if isinstance(num_frames, int) and isinstance(chain.get("frame_snap"), dict):
        num_frames = aligned(num_frames, chain["frame_snap"]) or num_frames
    if (
        isinstance(num_frames, int)
        and not isinstance(num_frames, bool)
        and isinstance(trim, int)
        and not isinstance(trim, bool)
        and window >= num_frames - trim
    ):
        return (
            f"carry_pick sharpest {window} must be less than num_frames "
            f"{num_frames} minus trim_frames {trim} ({num_frames - trim}), or "
            "the pick could empty a segment"
        )
    return None


CHAIN_LENGTH_RULE = (
    "A chain needs exactly one of 'segments', 'match_audio' or 'duration_seconds'"
)


MATCH_AUDIO_SOURCE_RULE = (
    "A match_audio chain needs a 'hold_audio' track, or a 'references' "
    "argument holding the audio reference to match"
)


def chain_needs_frame_snap(chain):
    """The key a chain is planned by length under - `duration_seconds` or
    `match_audio` - when its block declares no `frame_snap`, else None. Such
    a plan shortens its last segment to whatever remains, a count the
    pipeline may floor or round in silence (LTX-2.5 floors 37 to 33), so the
    run derives the snap from the step's `num_frames` constraint
    (`derive_frame_snaps` in dw/pipeline_processors/chain_config.py) and
    validate warns where it cannot (`frame_snap_warnings` in dw/guides.py).
    Asked of the block as written - a `variable:` length counts - and of the
    substituted one."""
    if chain.get("frame_snap") is not None:
        return None
    if chain.get("duration_seconds") is not None:
        return "duration_seconds"
    if chain.get("match_audio"):
        return "match_audio"
    return None


def chain_length_problem(chain):
    """Why a chain block does not ask for its length exactly one way, or None.
    A resolved duration wins over a segment count, so `segments` alongside
    `duration_seconds` is fine. The run (`_chain_length`) and validation ask
    this one rule."""
    match_audio = chain.get("match_audio", False)
    if isinstance(match_audio, str) and references.is_ref(
        references.UNRESOLVED, match_audio
    ):
        return None
    counted = (
        chain.get("segments") is not None or chain.get("duration_seconds") is not None
    )
    if counted == bool(match_audio):
        return CHAIN_LENGTH_RULE
    return None


def match_audio_source_problem(arguments):
    """Why a match_audio chain step names no track to match, or None: no
    `hold_audio` and no `references` list to hold an audio reference. The run
    also counts the audio references in the list (`_find_audio_reference`)."""
    arguments = arguments if isinstance(arguments, dict) else {}
    held = arguments.get("hold_audio")
    if held is not None:
        return None
    refs = arguments.get("references")
    if isinstance(refs, str) and references.is_ref(references.UNRESOLVED, refs):
        return None
    if isinstance(refs, list):
        return None
    return MATCH_AUDIO_SOURCE_RULE


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
# `check` object may set and the most re-rolls a checked segment may take.
# Family-independent
CHAIN_CHECK_FIELDS = (
    "jump_ratio_max",
    "look_ratio_max",
    "texture_ratio_max",
    "cut_ratio_max",
    "audio_dbfs_min",
    "words_similarity_min",
    "face_drop_max",
    "lip_sync_min",
)
CHAIN_RETRIES_MAX = 3
# Each segment's own line, which the words check holds its heard audio to
# (#1262): one string per segment, never repeated past its end as `prompts` is
CHAIN_LINES = "lines"


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


def _number_under(value, ceiling):
    """Whether `value` is a finite number below `ceiling` (never a bool), or
    an unresolved reference, which is left to the run."""
    if isinstance(value, str):
        return references.is_ref(references.UNRESOLVED, value)
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value < ceiling
    )


def chain_check_finding(chain):
    """The reason a chain block's `check` or `retries` is refused, or None.
    `check` is true (the default ceilings), false, or an object of
    `CHAIN_CHECK_FIELDS`: the ratio ceilings above 0, `texture_ratio_max`
    above 1 (it bounds the ratio both ways) or null, `audio_dbfs_min` a
    floor below 0 or null, `words_similarity_min` a similarity above 0 and at
    most 1, or null (the default, the words measure off), `face_drop_max`
    a number above 0 or null and `lip_sync_min` a correlation above -1 and at
    most 1 or null (both null by default: measured, never failed). `retries`
    is a whole number from 0 to `CHAIN_RETRIES_MAX`; above 0 it needs the
    check on, as with no check nothing fails. Unresolved references are left
    to the run, whose `ChainConfig` asks this same rule.

    Returns (key, message) - the key, `check` or `retries`, the message is
    about - or None."""
    check = chain.get("check")
    expected = "expected true, false or an object of " + ", ".join(CHAIN_CHECK_FIELDS)
    if isinstance(check, dict):
        unknown = sorted(set(check) - set(CHAIN_CHECK_FIELDS))
        if unknown:
            return "check", f"check has unknown key(s) {unknown} - {expected}"
        for name in ("jump_ratio_max", "look_ratio_max"):
            if name in check and not _number_over(check[name], 0):
                return (
                    "check",
                    f"check {name} must be a number above 0, got {check[name]!r}",
                )
        texture = check.get("texture_ratio_max")
        if texture is not None and not _number_over(texture, 1):
            return "check", (
                "check texture_ratio_max bounds the fine-band ratio both ways, "
                f"so must be a number above 1 or null, got {texture!r}"
            )
        cut = check.get("cut_ratio_max")
        if cut is not None and not _number_over(cut, 0):
            return (
                "check",
                f"check cut_ratio_max must be a number above 0 or null, got {cut!r}",
            )
        floor = check.get("audio_dbfs_min")
        if floor is not None and not _number_under(floor, 0):
            return "check", (
                "check audio_dbfs_min is a floor in dBFS, so must be a number "
                f"below 0 or null, got {floor!r}"
            )
        words = check.get("words_similarity_min")
        if words is not None and not (
            _number_over(words, 0) and (isinstance(words, str) or words <= 1)
        ):
            return "check", (
                "check words_similarity_min is the least similarity of a "
                "segment's heard words to its line, so must be a number above 0 "
                f"and at most 1, or null, got {words!r}"
            )
        face_sync = _face_sync_finding(check)
        if face_sync is not None:
            return face_sync
    elif not (
        check is None
        or isinstance(check, bool)
        or (isinstance(check, str) and _number_over(check, 0))
    ):
        return "check", f"invalid check {check!r} - {expected}"
    retries = chain.get("retries", 0)
    if isinstance(retries, str) and _number_over(retries, 0):
        return None
    whole = isinstance(retries, int) and not isinstance(retries, bool)
    if not whole or not 0 <= retries <= CHAIN_RETRIES_MAX:
        return "retries", (
            f"retries must be a whole number between 0 and {CHAIN_RETRIES_MAX}, "
            f"got {retries!r}"
        )
    if retries and (check is None or check is False):
        return "retries", (
            f"retries {retries} needs check on - a re-roll replaces a segment "
            "its check failed, and with no check no segment fails"
        )
    return None


def _face_sync_finding(check):
    """chain_check_finding's rule for a check object's identity and sync
    ceilings, or None."""
    drop = check.get("face_drop_max")
    if drop is not None and not _number_over(drop, 0):
        return "check", (
            "check face_drop_max is the most a segment's face similarity may "
            "fall below segment 1's, so must be a number above 0 or null, "
            f"got {drop!r}"
        )
    sync = check.get("lip_sync_min")
    if sync is not None and not (
        _number_over(sync, -1) and (isinstance(sync, str) or sync <= 1)
    ):
        return "check", (
            "check lip_sync_min is the least correlation of a segment's mouth "
            "motion with its audio, so must be a number above -1 and at most "
            f"1, or null, got {sync!r}"
        )
    return None


def chain_check_problem(chain):
    """`chain_check_finding`'s message alone, or None."""
    finding = chain_check_finding(chain)
    return finding[1] if finding else None


def chain_lines_problem(lines, segment_count):
    """The reason a chain block's `lines` is refused, or None (#1262): a list
    of strings, one per segment - `segment_count` of them when the count can
    be told (None leaves the length to the run, which asks this same rule
    against its plan). An unresolved reference, as the whole list or one
    entry, is left to the run."""
    if lines is None or (
        isinstance(lines, str) and references.is_ref(references.UNRESOLVED, lines)
    ):
        return None
    # An empty list with a known count is a count mismatch, and says so with
    # both numbers like any other; only with no count is it a shape error
    if not isinstance(lines, list) or (not lines and segment_count is None):
        return (
            "lines must be a list of strings, one per segment (the line each "
            f"segment speaks), got {lines!r}"
        )
    wrong = [entry for entry in lines if not isinstance(entry, str)]
    if wrong:
        return (
            "lines must hold only strings, one per segment (the line each "
            f"segment speaks), got {wrong[0]!r}"
        )
    if segment_count is not None and len(lines) != segment_count:
        return (
            f"lines holds {len(lines)} line(s) for {segment_count} segment(s) - "
            "one per segment; unlike prompts, the last is not repeated"
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


# The keystone slot (#1187/#1188 follow-on, #1199 Task 17): where a guide
# chain's `keystone: "segment_1"` frame sits in each later segment. "first"
# (plain keystone, the default) is the fl2va keyframe, at rotary time 0 - the
# slot of the guide's own first latent, so the two condition rows contradict
# each other. "last" makes it the fl2va `last_image`, the last-latent anchor
# the model is trained on; {"frame": 17j, "enter_below_sigma"?} a one-frame
# guide on a chunk boundary past the carried tail, optionally gated like the
# capstone. Off "first", the keyframe is the guide's first frame again.
# MiniMax-H3 only. A stable key since Don's 2026-11-06 call (structural.md,
# *Keystone slot*); the default stays "first" here, since "last" refuses a
# t2va step and a step's own last_image - the static long-take default is the
# template's (chained-segments `shot: static_long`). Only the sigma gate,
# which the windowed denoiser runs, stays behind that spike's flag (h3_rules)
CHAIN_KEYSTONE_SLOT = "keystone_slot"
# The spike's name, refused naming the stable one rather than read as an alias
CHAIN_KEYSTONE_SLOT_SPIKE = "experimental_keystone_slot"
KEYSTONE_SLOT_FIRST = "first"
KEYSTONE_SLOT_LAST = "last"
KEYSTONE_SLOT_FIELDS = ("frame", "enter_below_sigma")


def _unresolved(value):
    return isinstance(value, str) and references.is_ref(references.UNRESOLVED, value)


def chain_keystone_slot_problem(chain, num_frames=None, arguments=None):
    """The reason a chain block's `keystone_slot` is refused, or None. Absent,
    null and "first" are plain keystone. Otherwise refused: the spike's old
    name; a keystone other than `segment_1`; a value other than "first",
    "last" or {frame, enter_below_sigma?}; a frame off the 17-frame grid,
    inside the carried guide (it would contradict a guide row again) or past
    the segment's `num_frames` when known; a gate not in (0, 1], or with the
    windowed refine's flag off. With the step's `arguments` (the run), "last"
    also needs the fl2va keyframe and refuses a `last_image` of the step's
    own. Unresolved references are left to the run."""
    if CHAIN_KEYSTONE_SLOT_SPIKE in chain:
        return (
            f"{CHAIN_KEYSTONE_SLOT_SPIKE} was renamed {CHAIN_KEYSTONE_SLOT} "
            "when the spike became a stable key - use that name"
        )
    slot = chain.get(CHAIN_KEYSTONE_SLOT)
    if slot is None or slot == KEYSTONE_SLOT_FIRST or _unresolved(slot):
        return None
    from .h3_rules import KEYSTONE_SEGMENT_1

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
    return _gate_problem(slot.get("enter_below_sigma")) or _frame_slot_problem(
        chain, slot["frame"], num_frames
    )


def _gate_problem(gate):
    """Why a frame slot's `enter_below_sigma` is refused, or None: a value
    not in (0, 1], or - since the windowed denoiser runs the gate, and only
    its scratch flag puts that in - the flag off."""
    if gate is None:
        return None
    if not _number_over(gate, 0) or (not isinstance(gate, str) and gate > 1):
        return (
            f"{CHAIN_KEYSTONE_SLOT} enter_below_sigma is a sigma in (0, 1], "
            f"got {gate!r}"
        )
    from .h3_rules import REFINE_WINDOWS_ENV_VAR, refine_windows_enabled

    if refine_windows_enabled():
        return None
    return (
        f"{CHAIN_KEYSTONE_SLOT} enter_below_sigma is a scratch spike (the "
        f"windowed denoiser runs the gate), off unless "
        f"{REFINE_WINDOWS_ENV_VAR}=1 is set where the pipeline loads"
    )


def _frame_slot_problem(chain, frame, num_frames):
    """Why a frame slot's `frame` is refused, or None: off the 17-frame grid,
    inside the carried guide, or past the segment's `num_frames` when known."""
    from .h3_rules import GUIDE_CHAIN_DEFAULT, guide_frame_problem

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
    planned later segment, or None. A match_audio or duration_seconds chain
    renders its last segment only as long as the track or the asked length
    has left (snapped up to a length the pipeline takes), which `chain_keystone_slot_problem` cannot know from
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
                f"planned length leaves the last segment that short; move the "
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
    """A checked `keystone_slot` as the run uses it: None for plain
    keystone, "last", or {"frame": int, "enter_below_sigma": float or
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


# A chain resumed from an earlier run's kept segments (`resume_from`,
# dw/pipeline_processors/chain_resume.py): one segment file of that run,
# named by reference, and the run renders only the segments after its set
CHAIN_RESUME_FROM = "resume_from"
# The continuities a resume can carry out of a file on disk
RESUMABLE_CONTINUITIES = ("last_frame", "guide")


def resume_variables(steps, variables):
    """The variables a `resume_from` key in the authored steps names by
    'variable:' reference, at any depth, whose value is an `output:`
    reference - a chain block's own, or a composed step's argument passed
    down to its child workflow. Those keep the reference through the
    variables pass whatever they are called (#1304)."""
    held = set()

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == CHAIN_RESUME_FROM and isinstance(item, str):
                    name = references.ref_name(references.VARIABLE, item)
                    if name is not None and references.is_ref(
                        references.OUTPUT, variables.get(name)
                    ):
                        held.add(name)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(steps)
    return held


def resume_reference_problem(value):
    """Why a `resume_from` value is refused as a location, or None: anything
    but an output reference (or one still unresolved) - a bare path names a
    file outside the reference rules every earlier run's file is read by, so
    it is refused in the media location policy's words (dw/locations.py)."""
    if value is None or _unresolved(value):
        return None
    if isinstance(value, str) and references.is_ref(references.OUTPUT, value):
        return None
    example = references.make_ref(
        references.OUTPUT, "<workflow>/<run id>/<prefix>.0.segment-000.mp4"
    )
    return (
        f"Refusing to read chain {CHAIN_RESUME_FROM} at {value!r}: a resume "
        f"reads one segment file an earlier run wrote, named by its output "
        f"reference like '{example}' (the chain's failure warning quotes it), "
        f"never by a path"
    )


def chain_resume_problem(chain):
    """The reason a chain block's `resume_from` is refused, or None: absent,
    null or unresolved is no resume. Refused: without `save_segments` true
    (the kept files are copied into the new run's spill); a continuity other
    than `last_frame` or `guide`; and `experimental_latent_carry`, whose
    carried latents are not on disk. The reference itself is
    `resume_reference_problem`'s."""
    value = chain.get(CHAIN_RESUME_FROM)
    if value is None or _unresolved(value):
        return None
    if chain.get(CHAIN_LATENT_CARRY) is True:
        return (
            f"{CHAIN_RESUME_FROM} does not combine with {CHAIN_LATENT_CARRY}: "
            "the carried latents are not on disk, only the kept segments' "
            "decoded frames and audio"
        )
    continuity = chain.get("continuity", "last_frame")
    if not _unresolved(continuity) and continuity not in RESUMABLE_CONTINUITIES:
        return (
            f"{CHAIN_RESUME_FROM} carries a last frame or a guide out of a kept "
            f"file, so it needs continuity 'last_frame' or 'guide', got "
            f"{continuity!r}"
        )
    save = chain.get("save_segments", False)
    if save is not True and not _unresolved(save):
        return (
            f"{CHAIN_RESUME_FROM} copies the kept segment files into this run's "
            "own spill, so it needs save_segments true"
        )
    return None


# A probe of the take (`stop_after`): the plan is the full take's, only its
# first k segments render, and a resume_from naming the last of them continues
CHAIN_STOP_AFTER = "stop_after"


def stop_after_value(value):
    """`stop_after` as the run reads it: a digit string ("12", as
    `python -m dw.run ... stop_after=12` passes it) as its whole number, as
    `segments` is read; anything else as it is."""
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    return value


def chain_stop_after_problem(chain):
    """The reason a chain block's `stop_after` is refused, or None: absent,
    null or unresolved is a full take. Refused: anything but a whole number
    of at least 1 (or its digit string, `stop_after_value`), and a probe without `save_segments` and `keep_segments`
    true, which would leave no segment files to continue from."""
    value = chain.get(CHAIN_STOP_AFTER)
    if value is None or _unresolved(value):
        return None
    value = stop_after_value(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return (
            f"{CHAIN_STOP_AFTER} must be a whole number of at least 1 "
            f"(the segments the probe renders), got {value!r}"
        )
    for key in ("save_segments", "keep_segments"):
        setting = chain.get(key, False)
        if setting is not True and not _unresolved(setting):
            return (
                f"{CHAIN_STOP_AFTER} renders a probe that a {CHAIN_RESUME_FROM} "
                f"naming its last segment file continues, so it needs {key} true"
            )
    return None
