"""MiniMax-H3's pure rules for a chain block: the names and checks of
`continuity: "guide"` (`guide_frames`, `guide_backoff`), `anchor` and
`keystone` - what a chain may ask of H3 segments, read by validation
(dw/guides.py) and by the chain (dw/pipeline_processors/chain.py,
chain_config.py, chain_resume.py, chain_rules.py, chain_continuity.py).

The rules of one H3 call - the hold, the refine, a guide's own frame and
length, the latent anchor's strength - are dw/pipeline_processors/h3_rules.py;
this module needs none of them, and that one needs none of these.

Torch- and diffusers-free, as h3_rules.py is, so validation reads it without
the block factories.
"""

from .. import references

# A chain's `continuity: "guide"` (dw/pipeline_processors/chain.py) lays the
# previous segment's last `guide_frames` frames in at frame 0 of the next. Only
# these lengths are whole-latent guides (17m + 5) short enough to leave a
# segment most of its frames
GUIDE_CONTINUITY = "guide"
GUIDE_CHAIN_FRAMES = (22, 39)
GUIDE_CHAIN_DEFAULT = 22
# guide_backoff (#1189) moves the carry back whole 17-frame blocks
GUIDE_BACKOFF_BLOCK = 17
# Where a guide chain runs is where `guides` runs (dw/guides.py); the run and
# validate lead their refusal with this rule
GUIDE_CHAIN_RULE = (
    "continuity 'guide' runs on MiniMax-H3 t2va or fl2va only - ref2va takes one "
    "whole-render guide, never a guide chain"
)

# Every chain continuity mode, in the order chain_continuity.py registers its classes
# (`CONTINUITY_MODES`, zipped strictly against these). Kept here so validation
# (dw/guides.py) reads the same names without importing chain_continuity.py
CHAIN_CONTINUITY_MODES = ("last_frame", "last_segment", GUIDE_CONTINUITY)

# A chain's `anchor` (#1166): `segment_1` sets `latent_anchor` on every
# segment after the first to segment 1's decoded frames (the step is
# dw/pipeline_processors/h3_anchor.py, its block names in h3_rules.py)
ANCHOR_NONE = "none"
ANCHOR_SEGMENT_1 = "segment_1"
CHAIN_ANCHOR_MODES = (ANCHOR_NONE, ANCHOR_SEGMENT_1)
# The continuities whose segments are fresh t2va/fl2va generations the anchor
# block runs in; `last_segment` carries a ref2va reference, which has no block
ANCHOR_CONTINUITIES = ("last_frame", GUIDE_CONTINUITY)


# The keystone (#1187): segment 1's frame 0 as every later segment's fl2va
# keyframe, instead of the guide's first frame. Guide continuity only
KEYSTONE_NONE = "none"
KEYSTONE_SEGMENT_1 = "segment_1"
CHAIN_KEYSTONE_MODES = (KEYSTONE_NONE, KEYSTONE_SEGMENT_1)


def chain_keystone_problem(chain):
    """The reason a chain block's `keystone` is refused, or None: an unknown
    value (naming the values), or `segment_1` without `continuity: "guide"`.
    Unresolved references are left to the run."""
    keystone = chain.get("keystone", KEYSTONE_NONE)
    if isinstance(keystone, str) and references.is_ref(references.UNRESOLVED, keystone):
        return None
    if keystone not in CHAIN_KEYSTONE_MODES:
        return (
            f"unknown chain keystone {keystone!r} - expected one of "
            f"{', '.join(CHAIN_KEYSTONE_MODES)}"
        )
    if keystone == KEYSTONE_NONE:
        return None
    continuity = chain.get("continuity", "last_frame")
    if isinstance(continuity, str) and references.is_ref(
        references.UNRESOLVED, continuity
    ):
        return None
    if continuity != GUIDE_CONTINUITY:
        return (
            f"keystone 'segment_1' needs continuity 'guide' - the keystone "
            f"replaces the keyframe, and on {continuity!r} the keyframe is the "
            f"only carry; got {continuity!r}"
        )
    return None


def _set_for_plan(chain, *keys):
    """Whether a key is set; an unsubstituted reference is not (#1232)."""
    values = [chain.get(k) for k in keys]
    values = [v for v in values if not references.is_ref(references.UNRESOLVED, v)]
    return any(v is not None and v is not False for v in values)


def chain_anchor_problem(chain):
    """The reason a chain block's `anchor` is refused for its own settings, or
    None: an unknown value (naming the values), or `segment_1` with a continuity
    other than guide or last_frame, or carrying into `references`. Unresolved
    references are left to the run;
    whether the step is H3 is the caller's to ask (`not_h3`)."""
    anchor = chain.get("anchor", ANCHOR_NONE)
    if isinstance(anchor, str) and references.is_ref(references.UNRESOLVED, anchor):
        return None
    if anchor not in CHAIN_ANCHOR_MODES:
        return (
            f"unknown chain anchor {anchor!r} - expected one of "
            f"{', '.join(CHAIN_ANCHOR_MODES)}"
        )
    if anchor == ANCHOR_NONE:
        if chain.get("anchor_strength") is not None:
            return (
                "anchor_strength needs anchor 'segment_1' - it sets how far "
                "the anchor's latent match goes, and anchor is 'none'"
            )
        return None
    continuity = chain.get("continuity", "last_frame")
    if isinstance(continuity, str) and references.is_ref(
        references.UNRESOLVED, continuity
    ):
        return None
    if continuity not in ANCHOR_CONTINUITIES:
        return (
            f"anchor 'segment_1' needs continuity "
            f"{' or '.join(repr(c) for c in ANCHOR_CONTINUITIES)} - its segments "
            f"are MiniMax-H3 t2va/fl2va generations; got {continuity!r}"
        )
    # The carry lands in `segment_argument` on every later segment, and a call
    # that passes references runs ref2va, which has no anchor step - the
    # anchor would be dropped without a word
    if chain.get("segment_argument") == "references":
        return (
            "anchor 'segment_1' cannot carry into 'references' - a segment that "
            "passes references runs ref2va, which has no anchor step; use "
            "segment_argument 'image' (t2va/fl2va)"
        )
    return None


def guide_chain_problems(chain):
    """[(key, message)] for a guide-continuity chain block's own settings: a
    `guide_frames` other than 22 or 39 (refused, not snapped), and a
    `carry_frames`, which a guide chain does not read. An unresolved reference
    is left to the run."""
    problems = []
    lengths = " or ".join(str(length) for length in GUIDE_CHAIN_FRAMES)
    frames = chain.get("guide_frames", GUIDE_CHAIN_DEFAULT)
    unresolved = isinstance(frames, str) and references.is_ref(
        references.UNRESOLVED, frames
    )
    if not unresolved and (
        isinstance(frames, bool)
        or not isinstance(frames, int)
        or frames not in GUIDE_CHAIN_FRAMES
    ):
        problems.append(
            (
                "guide_frames",
                f"guide_frames must be {lengths} (the whole-latent guide lengths "
                f"a chain carries), got {frames!r}",
            )
        )
    if chain.get("carry_frames") is not None:
        problems.append(
            (
                "carry_frames",
                "carry_frames is a last_segment setting - a guide chain carries "
                "the last guide_frames frames; drop carry_frames",
            )
        )
    return problems


def guide_backoff_problems(chain):
    """[(key, message)] for a chain block's `guide_backoff` (#1189): a whole
    number of 17-frame blocks, on a guide-continuity chain, planned from a
    segment count - a `match_audio` or `duration_seconds` plan counts every segment's new frames from
    its head trim alone. An unresolved reference is left to the run."""
    if "guide_backoff" not in chain:
        return []
    backoff = chain["guide_backoff"]
    if isinstance(backoff, str) and references.is_ref(references.UNRESOLVED, backoff):
        return []
    block = GUIDE_BACKOFF_BLOCK
    if (
        isinstance(backoff, bool)
        or not isinstance(backoff, int)
        or backoff < 0
        or backoff % block
    ):
        return [
            (
                "guide_backoff",
                f"guide_backoff must be a multiple of {block} frames (whole "
                f"latent blocks), got {backoff!r}",
            )
        ]
    if backoff == 0:
        return []
    if chain.get("continuity", "last_frame") != GUIDE_CONTINUITY:
        return [
            (
                "guide_backoff",
                "guide_backoff is a guide continuity setting - set continuity "
                "'guide' or drop guide_backoff",
            )
        ]
    if _set_for_plan(chain, "match_audio", "duration_seconds"):
        return [
            (
                "guide_backoff",
                "guide_backoff does not combine with match_audio or "
                "duration_seconds: the plan covers the length from each "
                "segment's head trim alone, and backing off drops more frames "
                "per segment - use 'segments'",
            )
        ]
    return []
