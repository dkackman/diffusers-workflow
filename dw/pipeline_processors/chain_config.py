"""A chain block's settings and its segment plan.

`ChainConfig` reads and refuses the block's settings for one run and plans
its segments: by count (`segments`), by length (`duration_seconds`) or by a
matched track (`match_audio`). `plan_segments`, `snap_frames` and
`validate_frame_snap` are the plan's arithmetic, and `chain_segment_count`
is the pure count the cost estimate asks too. Split out of `chain.py`, which
imports this module and never the reverse.
"""

import logging
import math
from dataclasses import dataclass

from ..dsp import as_channels_samples
from ..events import emit_log
from .chain_check import segment_check
from .chain_rules import (
    CHAIN_CAPSTONE,
    CHAIN_CAPSTONE_FROM_END,
    CHAIN_LATENT_CARRY,
    chain_capstone_problem,
    chain_check_problem,
    chain_keystone_slot,
    chain_keystone_slot_problem,
    chain_latent_carry_problem,
    chain_look_hold_flags,
    chain_look_hold_problem,
    keystone_slot_plan_problem,
)
from .h3_hold import hold_audio_reference
from .h3_rules import (
    ANCHOR_NONE,
    CHAIN_CONTINUITY_MODES,
    GUIDE_CHAIN_DEFAULT,
    GUIDE_CONTINUITY,
    GUIDE_NOISE_INPUT,
    HOLD_AUDIO_INPUT,
    KEYSTONE_NONE,
    anchor_strength_problem,
    chain_anchor_problem,
    chain_carry_pick_problem,
    chain_keystone_problem,
    guide_backoff_problems,
    guide_chain_problems,
)

logger = logging.getLogger("dw")

# A runaway segment count is a configuration error - kept in the spirit of
# previous_results.MAX_ITERATIONS
MAX_SEGMENTS = 1000


@dataclass
class Segment:
    """One planned pipeline invocation of a chain."""

    index: int
    num_frames: int  # frames this segment generates; None - use the step's own
    audio_start_frame: int  # generation-timeline frame its audio slice starts at
    head_trim: int  # frames dropped from its head on the output timeline


def _argument(arguments, name):
    """`arguments[name]` when the step's arguments are a dict, else None."""
    return arguments.get(name) if isinstance(arguments, dict) else None


def _raise_chain_problem(problem):
    """Refuse a chain setting on its one problem, if it has one."""
    if problem:
        raise ValueError(f"Chain {problem}")


def _raise_chain_problems(problems):
    """Refuse a chain setting on all its (field, message) problems at once."""
    if problems:
        raise ValueError("; ".join(f"Chain {message}" for _, message in problems))


def _chain_length(chain_definition):
    """(segments, match_audio, duration_seconds), refused unless the chain
    asks for its length exactly one way."""
    segments = chain_definition.get("segments", None)
    match_audio = bool(chain_definition.get("match_audio", False))
    duration = chain_definition.get("duration_seconds", None)
    # A resolved duration wins over a segment count: a template carries a
    # default `segments`, and asking for seconds must not need it nulled
    counted = segments is not None or duration is not None
    if counted == match_audio:
        raise ValueError(
            "A chain needs exactly one of 'segments', 'match_audio' or "
            "'duration_seconds'"
        )
    if segments is not None and duration is not None:
        emit_log(
            f"Chain 'segments' ({segments}) ignored: 'duration_seconds' "
            f"({float(duration):g}) sets the segment count"
        )
    return segments, match_audio, duration


class ChainConfig:
    """Validated chain settings plus the planned segments for one run."""

    def __init__(
        self, chain_definition, arguments, resolved_prompts=None, base_dir=None
    ):
        segments, match_audio, duration = _chain_length(chain_definition)
        self._read_continuity(chain_definition)
        self._read_experiments(chain_definition, arguments)
        self._read_carry(chain_definition)
        head_trim = self._read_guide(chain_definition)
        self.head_trim = head_trim
        self.prompts = resolved_prompts or chain_definition.get("prompts", None)
        self.fps = _resolve_fps(chain_definition, arguments)
        self.frame_snap = chain_definition.get("frame_snap", None)
        self.save_segments = bool(chain_definition.get("save_segments", False))
        self.keep_segments = bool(chain_definition.get("keep_segments", False))

        self.source_audio = None
        self.source_rate = None
        self.audio_reference = None
        self.holds_audio = False
        self.total_frames = None
        self.duration_seconds = float(duration) if duration is not None else None

        if match_audio:
            self._plan_from_audio(arguments, base_dir)
            # Only a matched chain renders its shortened last segment at its
            # planned length (chain._segment_arguments), so only its plan can
            # leave a frame slot past a segment's end
            _raise_chain_problem(
                keystone_slot_plan_problem(self.keystone_slot, self.plan)
            )
        elif duration is not None:
            self.total_frames, self.plan = _duration_plan(chain_definition, arguments)
            logger.info(
                f"Chaining to {float(duration):g}s: {self.total_frames} frames "
                f"across {len(self.plan)} segments"
            )
        else:
            self._plan_from_segments(segments, arguments, head_trim)

    def _read_continuity(self, chain_definition):
        self.continuity = chain_definition.get("continuity", "last_frame")
        if self.continuity not in CHAIN_CONTINUITY_MODES:
            known = ", ".join(sorted(CHAIN_CONTINUITY_MODES))
            raise ValueError(
                f"Unknown chain continuity '{self.continuity}' - expected one of {known}"
            )

    def _read_experiments(self, chain_definition, arguments):
        """The anchor, keystone, capstone, latent carry, carry pick, look hold
        and per-segment check, each refused on its first problem, in order."""
        # Segment 1 as every later segment's latent statistics anchor (#1166)
        _raise_chain_problem(chain_anchor_problem(chain_definition))
        self.anchor = chain_definition.get("anchor", ANCHOR_NONE)
        # How far each later segment's match goes (#1190); None is the full one
        self.anchor_strength = chain_definition.get("anchor_strength")
        if self.anchor_strength is not None:
            _raise_chain_problem(anchor_strength_problem(self.anchor_strength))
            self.anchor_strength = float(self.anchor_strength)

        # Segment 1's frame 0 as every later keyframe (#1187)
        _raise_chain_problem(chain_keystone_problem(chain_definition))
        self.keystone = chain_definition.get("keystone", KEYSTONE_NONE)
        # Where the keystone sits in each later segment, a spike
        _raise_chain_problem(
            chain_keystone_slot_problem(
                chain_definition,
                _argument(arguments, "num_frames"),
                arguments if isinstance(arguments, dict) else None,
            )
        )
        self.keystone_slot = chain_keystone_slot(chain_definition)

        # Segment 1's late frame past every later segment's end (#1187), a spike
        _raise_chain_problem(chain_capstone_problem(chain_definition))
        capstone = chain_definition.get(CHAIN_CAPSTONE)
        self.capstone = capstone and {
            "enter_below_sigma": float(capstone["enter_below_sigma"]),
            "from_end": int(capstone.get("from_end", CHAIN_CAPSTONE_FROM_END)),
        }

        # The previous segment's own latents as its guide (#1150), a spike
        _raise_chain_problem(
            chain_latent_carry_problem(
                chain_definition,
                _argument(arguments, GUIDE_NOISE_INPUT),
                _argument(arguments, HOLD_AUDIO_INPUT),
            )
        )
        self.latent_carry = chain_definition.get(CHAIN_LATENT_CARRY) is True

        # The sharpest of the last N frames as the keyframe (#1032)
        _raise_chain_problem(
            chain_carry_pick_problem(
                chain_definition, _argument(arguments, "num_frames")
            )
        )
        pick = chain_definition.get("carry_pick")
        self.carry_pick = int(pick["sharpest"]) if pick else 0

        # Pixel corrections on each segment before its carry (#1191)
        _raise_chain_problem(chain_look_hold_problem(chain_definition))
        self.look_hold = chain_look_hold_flags(chain_definition)

        # The per-segment check and its re-roll budget; absent is off and 0
        _raise_chain_problem(chain_check_problem(chain_definition))
        self.check = segment_check(chain_definition.get("check"))
        self.retries = int(chain_definition.get("retries", 0)) if self.check else 0

    def _read_carry(self, chain_definition):
        self.segment_argument = chain_definition.get("segment_argument", "image")
        self.carry_frames = chain_definition.get("carry_frames", None)
        if self.carry_frames is not None:
            self.carry_frames = int(self.carry_frames)
            if self.carry_frames < 1:
                raise ValueError(
                    f"Chain 'carry_frames' must be at least 1, got {self.carry_frames}"
                )
        self.carry_audio = bool(chain_definition.get("carry_audio", True))
        self.trim_frames = int(chain_definition.get("trim_frames", 1))
        self.crossfade_ms = float(chain_definition.get("crossfade_ms", 75))

    def _read_guide(self, chain_definition):
        """The guide chain's settings; returns the head trim every later
        segment drops - the guide it laid in, or trim_frames off a guide."""
        # A guide chain trims the guide it laid in, not trim_frames, and a
        # held soundtrack has nothing to crossfade
        self.guide_frames = None
        self.guide_holds_audio = False
        # Frames dropped from the tail of every segment but the last (#1189)
        self.guide_backoff = 0
        self.tail_trim = 0
        _raise_chain_problems(guide_backoff_problems(chain_definition))
        if self.continuity != GUIDE_CONTINUITY:
            return self.trim_frames
        _raise_chain_problems(guide_chain_problems(chain_definition))
        self.guide_frames = chain_definition.get("guide_frames", GUIDE_CHAIN_DEFAULT)
        self.guide_backoff = int(chain_definition.get("guide_backoff", 0))
        self.tail_trim = self.guide_backoff
        if self.carry_audio:
            self.guide_holds_audio = True
            self.crossfade_ms = 0.0
        return self.guide_frames

    def _plan_from_segments(self, segments, arguments, head_trim):
        """The plan for a chain asked for by its segment count."""
        segments = int(segments)
        if not 1 <= segments <= MAX_SEGMENTS:
            raise ValueError(
                f"Chain 'segments' must be between 1 and {MAX_SEGMENTS}, got {segments}"
            )
        num_frames = arguments.get("num_frames", None)
        if num_frames is not None:
            validate_frame_snap(int(num_frames), self.frame_snap)
            if (
                self.guide_frames is not None
                and int(num_frames) <= head_trim + self.tail_trim
            ):
                raise ValueError(
                    f"Segments of {num_frames} frames cannot progress past "
                    f"a {head_trim}-frame guide"
                    + (
                        f" and a {self.tail_trim}-frame back-off"
                        if self.tail_trim
                        else ""
                    )
                )
        self.plan = [
            Segment(
                index,
                int(num_frames) if num_frames is not None else None,
                0,
                head_trim if index > 0 else 0,
            )
            for index in range(segments)
        ]

    def _plan_from_audio(self, arguments, base_dir=None):
        """Derive the segment plan from the matched track's duration - the
        step's `hold_audio` when it names one, else its audio reference."""
        if self.fps is None:
            raise ValueError(
                "A match_audio chain needs the frame rate - set 'fps' on the "
                "chain or a 'frame_rate' pipeline argument"
            )

        num_frames = arguments.get("num_frames", None)
        if num_frames is None:
            raise ValueError(
                "A match_audio chain needs 'num_frames' in the step's arguments "
                "as the per-segment length"
            )

        held = arguments.get(HOLD_AUDIO_INPUT)
        if held is not None:
            # Read once here, sliced per segment - each slice is a reference
            # already built, which the segment's own hold takes as it is
            reference = hold_audio_reference(held, base_dir)
            self.holds_audio = True
        else:
            reference = _find_audio_reference(arguments)
        self.audio_reference = reference
        self.source_audio = as_channels_samples(reference.audio)
        self.source_rate = reference.sample_rate
        if self.source_rate is None:
            raise ValueError("The chain's matched audio has no sample rate")

        total_samples = self.source_audio.shape[1]
        self.total_frames = max(1, round(total_samples / self.source_rate * self.fps))
        self.plan = plan_segments(
            self.total_frames, int(num_frames), self.head_trim, self.frame_snap
        )
        duration = total_samples / self.source_rate
        logger.info(
            f"Chaining to match {duration:.2f}s of audio: {self.total_frames} "
            f"frames across {len(self.plan)} segments"
        )


def _head_trim(chain_definition):
    """Head frames each segment after the first drops: the guide a `guide`
    chain laid in, else `trim_frames`."""
    if chain_definition.get("continuity", "last_frame") == GUIDE_CONTINUITY:
        return int(chain_definition.get("guide_frames", GUIDE_CHAIN_DEFAULT))
    return int(chain_definition.get("trim_frames", 1))


def _duration_plan(chain_definition, arguments):
    """(total_frames, plan) for a `duration_seconds` chain - the same plan a
    `match_audio` chain makes from a track, from the asked length instead."""
    duration = float(chain_definition["duration_seconds"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(
            f"Chain 'duration_seconds' must be greater than 0, got {duration:g}"
        )
    fps = _resolve_fps(chain_definition, arguments)
    if fps is None:
        raise ValueError(
            "A duration_seconds chain needs the frame rate - set 'fps' on the "
            "chain or a 'frame_rate' pipeline argument"
        )
    num_frames = arguments.get("num_frames", None)
    if num_frames is None:
        raise ValueError(
            "A duration_seconds chain needs 'num_frames' in the step's arguments "
            "as the per-segment length"
        )
    total_frames = max(1, round(duration * fps))
    plan = plan_segments(
        total_frames,
        int(num_frames),
        _head_trim(chain_definition),
        chain_definition.get("frame_snap", None),
    )
    return total_frames, plan


def chain_segment_count(chain_definition, arguments):
    """How many segments a chain block runs, or None when it cannot be told
    from the block and the step's arguments alone (`match_audio`, an
    unresolved reference, or a missing frame rate or `num_frames`). A
    `duration_seconds` wins over `segments`. Pure: the run's `ChainConfig`
    and the cost estimate both ask it."""
    segments = chain_definition.get("segments", None)
    duration = chain_definition.get("duration_seconds", None)
    if chain_definition.get("match_audio") or (segments is None and duration is None):
        return None
    try:
        if duration is None:
            return int(segments)
        frame_snap = chain_definition.get("frame_snap", None)
        definition = {**chain_definition}
        if not isinstance(frame_snap, dict):
            definition.pop("frame_snap", None)
        return len(_duration_plan(definition, arguments)[1])
    except (TypeError, ValueError):
        return None


def plan_segments(total_frames, segment_frames, trim_frames, frame_snap=None):
    """Plan the segments that cover a total frame count.

    Every segment generates segment_frames frames except possibly the last,
    which shrinks to what remains - snapped up to a count the pipeline accepts.
    Each segment after the first has trim_frames dropped from its head, so it
    contributes segment_frames - trim_frames new frames to the output.

    Args:
        total_frames: Frames the stitched output must cover
        segment_frames: Frames a full segment generates
        trim_frames: Head frames dropped from every segment after the first
        frame_snap: Optional dict with modulus/remainder and min/max_frames
            describing the counts the pipeline accepts

    Returns:
        List of Segment
    """
    if segment_frames <= trim_frames:
        raise ValueError(
            f"Segments of {segment_frames} frames cannot progress past a head "
            f"trim of {trim_frames} frames"
        )
    validate_frame_snap(segment_frames, frame_snap)

    plan = []
    covered = 0
    while covered < total_frames:
        if len(plan) >= MAX_SEGMENTS:
            raise ValueError(f"Chain would exceed {MAX_SEGMENTS} segments")

        head_trim = trim_frames if plan else 0
        needed = (total_frames - covered) + head_trim
        if needed >= segment_frames:
            num_frames = segment_frames
        else:
            # The last segment generates only what remains, snapped up to a
            # count the pipeline accepts; the overshoot is trimmed at the end
            num_frames = snap_frames(needed, frame_snap)

        plan.append(Segment(len(plan), num_frames, covered - head_trim, head_trim))
        covered += num_frames - head_trim

    return plan


def snap_frames(count, frame_snap):
    """The smallest frame count the pipeline accepts that covers count."""
    if not frame_snap:
        return count

    modulus = frame_snap["modulus"]
    remainder = frame_snap["remainder"]
    target = max(count, frame_snap.get("min_frames", 1))

    steps = max(0, math.ceil((target - remainder) / modulus))
    snapped = steps * modulus + remainder
    while snapped < target:
        snapped += modulus

    max_frames = frame_snap.get("max_frames", None)
    if max_frames is not None and snapped > max_frames:
        raise ValueError(
            f"Cannot snap {count} frames into the pipeline's accepted range - "
            f"the next valid count {snapped} exceeds max_frames {max_frames}"
        )
    return snapped


def validate_frame_snap(num_frames, frame_snap):
    """Check a configured num_frames against the pipeline's constraint."""
    if not frame_snap:
        return

    modulus = frame_snap["modulus"]
    remainder = frame_snap["remainder"]
    problems = []
    if (num_frames - remainder) % modulus != 0:
        problems.append(f"counts must be {modulus}*n+{remainder}")
    min_frames = frame_snap.get("min_frames", None)
    if min_frames is not None and num_frames < min_frames:
        problems.append(f"at least {min_frames}")
    max_frames = frame_snap.get("max_frames", None)
    if max_frames is not None and num_frames > max_frames:
        problems.append(f"at most {max_frames}")

    if problems:
        raise ValueError(
            f"num_frames {num_frames} does not satisfy the pipeline's frame "
            f"constraint: {'; '.join(problems)}"
        )


def _resolve_fps(chain_definition, arguments):
    """The frame rate used for audio math - explicit, or the pipeline's own."""
    fps = chain_definition.get("fps", arguments.get("frame_rate", None))
    return float(fps) if fps is not None else None


def _find_audio_reference(arguments):
    """The single audio reference a match_audio chain with no `hold_audio`
    slices per segment."""
    references = arguments.get("references", None)
    if not isinstance(references, list):
        raise ValueError(
            "A match_audio chain needs a 'hold_audio' track, or a 'references' "
            "argument holding the audio reference to match"
        )

    audio_references = [
        reference
        for reference in references
        if getattr(reference, "kind", None) == "audio"
    ]
    if len(audio_references) != 1:
        raise ValueError(
            f"A match_audio chain needs exactly one audio reference, "
            f"found {len(audio_references)}"
        )
    return audio_references[0]
