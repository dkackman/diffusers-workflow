"""A chain block's settings and its segment plan.

`ChainConfig` reads and refuses the block's settings for one run and plans
its segments: by count (`segments`), by length (`duration_seconds`) or by a
matched track (`match_audio`). `plan_segments`, `snap_frames` and
`validate_frame_snap` are the plan's arithmetic, `derive_frame_snaps` gives a
length-planned block with no `frame_snap` the one its `num_frames` variable's
constraint declares, and `chain_segment_count`
is the pure count the cost estimate asks too, as `chain_holds_audio`,
`chain_audio_floor` and `chain_checks_segment_1` are the pure audio-check
rules (#1221), and `chain_checks_words` and `segment_line` the words
check's (#1262). `take_frames`, `take_track` and `trimmed_take_shots` cut
the stitched take to its plan, or to a probe's rendered segments
(`stop_after`). Split out of `chain.py`, which
imports this module and never the reverse.
"""

import logging
import math
from dataclasses import dataclass

import torch

from ..dsp import as_channels_samples, slice_samples
from ..shots import without_samples
from ..task_domains import frames_to_samples
from ..script_lines import dialogue_text
from ..events import emit_log
from .chain_check import AUDIO_DBFS_MIN, segment_check
from .. import references
from ..variable_constraints import declared_constraints
from .chain_rules import (
    CHAIN_ANCHOR_EVERY,
    CHAIN_CAPSTONE,
    CHAIN_CAPSTONE_FROM_END,
    CHAIN_KEYSTONE_IMAGES,
    CHAIN_LATENT_CARRY,
    CHAIN_LINES,
    CHAIN_RESUME_FROM,
    CHAIN_STOP_AFTER,
    KEYSTONE_SLOT_LATENT,
    anchors_segment,
    chain_anchor_every_problem,
    chain_capstone_problem,
    chain_carry_pick_problem,
    chain_check_problem,
    chain_keystone_images_problem,
    chain_keystone_slot,
    chain_keystone_slot_problem,
    CHAIN_LATENT_CARRY_LEVELLED,
    chain_latent_carry_problem,
    chain_length_problem,
    chain_lines_problem,
    chain_look_hold_flags,
    chain_look_hold_problem,
    chain_needs_frame_snap,
    chain_resume_problem,
    chain_stop_after_problem,
    keystone_images_count_problem,
    stop_after_value,
    keystone_slot_plan_problem,
    match_audio_source_problem,
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


def _loaded_images(images, base_dir):
    """`keystone_images`' entries as loaded images, each a path, URL or
    {location} loaded under the media location policy (dw/locations.py), a
    loaded image kept; None when the chain has none."""
    if not images:
        return None
    from ..argument_media import fetch_image

    loaded = [fetch_image(entry, base_dir) for entry in images]
    for index, image in enumerate(loaded):
        if not (hasattr(image, "mode") and hasattr(image, "size")):
            raise ValueError(
                f"Chain {CHAIN_KEYSTONE_IMAGES} entry {index} is not an image, "
                f"got {type(image).__name__}"
            )
    return loaded


def _chain_length(chain_definition):
    """(segments, match_audio, duration_seconds), refused unless the chain
    asks for its length exactly one way."""
    segments = chain_definition.get("segments", None)
    match_audio = bool(chain_definition.get("match_audio", False))
    duration = chain_definition.get("duration_seconds", None)
    # A resolved duration wins over a segment count: a template carries a
    # default `segments`, and asking for seconds must not need it nulled
    problem = chain_length_problem(chain_definition)
    if problem:
        raise ValueError(problem)
    if segments is not None and duration is not None:
        emit_log(
            f"Chain 'segments' ({segments}) ignored: 'duration_seconds' "
            f"({float(duration):g}) sets the segment count"
        )
    return segments, match_audio, duration


class ChainConfig:
    """Validated chain settings plus the planned segments for one run."""

    def __init__(
        self,
        chain_definition,
        arguments,
        resolved_prompts=None,
        base_dir=None,
        resolved_lines=None,
        resolved_keystone_images=None,
    ):
        segments, match_audio, duration = _chain_length(chain_definition)
        self._read_continuity(chain_definition)
        # A per-segment last image (#1308 Amendment 1), resolved by Step.run
        # when it named a previous result; checked before the keystone slot
        if resolved_keystone_images is not None:
            chain_definition = {
                **chain_definition,
                CHAIN_KEYSTONE_IMAGES: resolved_keystone_images,
            }
        self._read_experiments(chain_definition, arguments)
        self._read_carry(chain_definition)
        head_trim = self._read_guide(chain_definition, arguments)
        self.head_trim = head_trim
        self.prompts = resolved_prompts or chain_definition.get("prompts", None)
        self.fps = _resolve_fps(chain_definition, arguments)
        self.frame_snap = chain_definition.get("frame_snap", None)
        self.save_segments = bool(chain_definition.get("save_segments", False))
        self.keep_segments = bool(chain_definition.get("keep_segments", False))
        # An earlier run's kept segments to start from (chain_resume.py); the
        # reference itself is refused there, as a location
        _raise_chain_problem(chain_resume_problem(chain_definition))
        self.resume_from = chain_definition.get(CHAIN_RESUME_FROM)
        # A probe renders the plan's first k segments only (chain_rules)
        _raise_chain_problem(chain_stop_after_problem(chain_definition))
        self.stop_after = stop_after_value(chain_definition.get(CHAIN_STOP_AFTER))

        self.source_audio = None
        self.source_rate = None
        self.audio_reference = None
        self.holds_audio = False
        self.total_frames = None
        self.duration_seconds = float(duration) if duration is not None else None

        if match_audio:
            self._plan_from_audio(arguments, base_dir)
        elif duration is not None:
            self.total_frames, self.plan = _duration_plan(chain_definition, arguments)
            logger.info(
                f"Chaining to {float(duration):g}s: {self.total_frames} frames "
                f"across {len(self.plan)} segments"
            )
        else:
            self._plan_from_segments(segments, arguments, head_trim)
        # Each segment's own line for the words check (#1262), one per
        # segment: refused against the plan, which match_audio sets only now
        self.lines = resolved_lines or chain_definition.get(CHAIN_LINES)
        _raise_chain_problem(chain_lines_problem(self.lines, len(self.plan)))
        # The full plan's count, a probe's too: the probe is continued
        _raise_chain_problem(
            keystone_images_count_problem(self.keystone_images, len(self.plan))
        )
        self.keystone_images = _loaded_images(self.keystone_images, base_dir)
        self.step_prompt = _argument(arguments, "prompt")
        if self.total_frames is not None:
            # A length-planned chain renders its shortened last segment at
            # its planned length (chain._segment_arguments), so its plan can
            # leave a frame slot past a segment's end
            _raise_chain_problem(
                keystone_slot_plan_problem(self.keystone_slot, self.plan)
            )

    @property
    def stops_early(self):
        """Whether `stop_after` ends this run before the plan's last segment."""
        return self.stop_after is not None and self.stop_after < len(self.plan)

    @property
    def end_index(self):
        """One past the last segment this run renders."""
        return self.stop_after if self.stops_early else len(self.plan)

    def take_frames(self, shots):
        """(frames, shots) of the stitched take: the plan's `total_frames`
        (None for a count-planned chain), or a probe's rendered frames when
        `stop_after` ended the run early, which it logs; each shot is marked
        with the plan's segment count, which a probe's take falls short of."""
        planned = len(self.plan)
        shots = [{**shot, "planned_segments": planned} for shot in shots]
        if not self.stops_early:
            return self.total_frames, shots
        rendered = sum(shot["num_frames"] for shot in shots)
        emit_log(
            f"Chain stopped after {self.stop_after} of {planned} segments "
            f"({rendered} frames): resume_from its last segment file renders "
            f"the other {planned - self.stop_after}",
            kind="chain_stopped_after",
            stop_after=self.stop_after,
            segments=planned,
            frames=rendered,
        )
        return rendered, shots

    def take_track(self, total_frames):
        """The matched track under the take: whole, unsliced, or a probe's
        stretch of it, cut to its rendered frames."""
        if not self.stops_early:
            return self.source_audio
        length = frames_to_samples(total_frames, self.fps, self.source_rate)
        return self.source_audio[:, :length]

    def audio_slice(self, segment):
        """The segment's piece of the matched track, as a new reference of the
        matched one's type - the original is never touched."""
        start = frames_to_samples(segment.audio_start_frame, self.fps, self.source_rate)
        length = frames_to_samples(segment.num_frames, self.fps, self.source_rate)
        piece = slice_samples(self.source_audio, start, length)
        return type(self.audio_reference)(
            audio=torch.from_numpy(piece), sample_rate=self.source_rate
        )

    def sliced_references(self, segment, segment_references):
        """A copy of the references list with the segment's audio slice
        swapped in. The original list and reference objects are never
        touched - iteration arguments share nested values, so they must not
        be mutated in place."""
        sliced = self.audio_slice(segment)
        return [
            sliced if reference is self.audio_reference else reference
            for reference in segment_references
        ]

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
        # ...or each segment's own last image (#1308 Amendment 1)
        _raise_chain_problem(
            chain_keystone_images_problem(
                chain_definition,
                arguments=arguments if isinstance(arguments, dict) else None,
                run=True,
            )
        )
        self.keystone_images = chain_definition.get(CHAIN_KEYSTONE_IMAGES)
        self.keystone = chain_definition.get("keystone", KEYSTONE_NONE)
        # Where the keystone sits in each later segment (#1187)
        _raise_chain_problem(
            chain_keystone_slot_problem(
                chain_definition,
                _argument(arguments, "num_frames"),
                arguments if isinstance(arguments, dict) else None,
            )
        )
        self.keystone_slot = chain_keystone_slot(chain_definition)

        # Segment 1's late frame past every later segment's end (#1187), a spike
        _raise_chain_problem(
            chain_capstone_problem(chain_definition, _argument(arguments, "num_frames"))
        )
        capstone = chain_definition.get(CHAIN_CAPSTONE)
        self.capstone = capstone and {
            "enter_below_sigma": float(capstone["enter_below_sigma"]),
            "from_end": int(capstone.get("from_end", CHAIN_CAPSTONE_FROM_END)),
        }

        # The previous segment's own latents as its guide (#1150), a spike
        _raise_chain_problem(
            chain_latent_carry_problem(
                chain_definition, _argument(arguments, GUIDE_NOISE_INPUT)
            )
        )
        carried = chain_definition.get(CHAIN_LATENT_CARRY)
        self.latent_level = carried == CHAIN_LATENT_CARRY_LEVELLED
        self.latent_carry = carried is True or self.latent_level
        # Which later segments take the keystone and capstone (Task 9), a spike
        _raise_chain_problem(chain_anchor_every_problem(chain_definition))
        every = stop_after_value(chain_definition.get(CHAIN_ANCHOR_EVERY))
        self.anchor_every = 1 if every is None else every

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
        # Held or generated audio, the floor and segment 1's check (#1221):
        # the pure rules below decide them, which the cost estimate asks too
        self.audio_held = chain_holds_audio(chain_definition, arguments)
        self.audio_floor = chain_audio_floor(chain_definition, arguments)
        self.checks_segment_1 = chain_checks_segment_1(chain_definition, arguments)
        self.checks_words = chain_checks_words(chain_definition, arguments)

    def anchors(self, index):
        """Whether segment `index` takes the keystone and the capstone."""
        return anchors_segment(index, self.anchor_every)

    def keystone_for(self, index, keystone_frame):
        """The keystone segment `index` takes, or None: its own
        `keystone_images` entry (segment 1's, as its last_image, too), else
        segment 1's `keystone_frame` - each on the anchor_every cadence."""
        if self.keystone_images:
            on = index == 0 or self.anchors(index)
            return self.keystone_images[index] if on else None
        return keystone_frame if self.anchors(index) else None

    @property
    def latent_keystone(self):
        """Whether segment 1's frame-0 latent is the keystone (Task 9)."""
        return self.keystone_slot == KEYSTONE_SLOT_LATENT

    def segment_line(self, index):
        """The line segment `index` is meant to speak, or None (#1262):
        `segment_line` over this chain's `lines`, `prompts` and step prompt."""
        return segment_line(self.lines, self.prompts, self.step_prompt, index)

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

    def _read_guide(self, chain_definition, arguments=None):
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
            return _head_trim(chain_definition)
        _raise_chain_problems(guide_chain_problems(chain_definition))
        # The one rule the duration plan and the cost estimate's segment
        # count derive the head trim by too (_head_trim)
        self.guide_frames = _head_trim(chain_definition)
        self.guide_backoff = int(chain_definition.get("guide_backoff", 0))
        self.tail_trim = self.guide_backoff
        if self.carry_audio and _guide_carry_held(chain_definition, arguments):
            # Each segment's held slice already starts at the guide span, so a
            # guide carrying audio would hold the overlap twice - as guide
            # audio rows and as the held rows written after them
            self.carry_audio = False
            emit_log(
                "Guide chain: carry_audio is off - the chain holds its track, "
                "and each segment's held slice already covers the guide span"
            )
        if self.carry_audio and _guide_holds_audio(chain_definition):
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

        problem = match_audio_source_problem(arguments)
        if problem:
            raise ValueError(problem)
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


def trimmed_take_shots(shots, total_frames, total_samples=None):
    """A length-planned chain's shots, cut where its overshooting picture is.

    A match_audio soundtrack is the caller's own, laid under whole rather than
    built segment by segment, so no shot has a stretch of it to measure: the
    sample side is cleared, not derived. A duration_seconds chain's generated
    audio is cut to `total_samples`, and each shot's stretch of it with it.
    """
    trimmed = []
    for shot in shots if total_samples is not None else without_samples(shots):
        if shot["start_frame"] >= total_frames:
            break
        end = min(shot["start_frame"] + shot["num_frames"], total_frames)
        cut = {**shot, "num_frames": end - shot["start_frame"]}
        if total_samples is not None:
            start = min(shot["start_sample"], total_samples)
            cut["num_samples"] = min(shot["num_samples"], total_samples - start)
        trimmed.append(cut)
    return trimmed


def derive_frame_snaps(definition):
    """Give every chain block planned by length (`duration_seconds`,
    `match_audio`) that declares no `frame_snap` the constraint its step's
    `num_frames` variable declares, in place, as `"constraint:<name>"` for
    `resolve_constraint_references` to resolve. Run before variables
    substitute, while the `variable:` reference is still visible. Returns
    [(step name, variable name)] for each block changed, for the run's log.

    Without it a 6.5 s LTX chain at 24 fps with 121-frame segments plans a
    37-frame last segment, which the pipeline floors to 33 - a short take. A
    block whose `num_frames` is a literal, or a variable with no modulus
    constraint, is left as written; validate warns on it
    (`frame_snap_warnings` in dw/guides.py)."""
    constraints = declared_constraints(definition)
    derived = []
    for _, step, pipeline, _, _ in references.iter_steps(
        definition.get("steps"), None, "pipeline"
    ):
        chain = pipeline.get("chain")
        if not isinstance(chain, dict) or chain_needs_frame_snap(chain) is None:
            continue
        arguments = pipeline.get("arguments")
        num_frames = (
            arguments.get("num_frames") if isinstance(arguments, dict) else None
        )
        name = references.ref_name(references.VARIABLE, num_frames)
        constraint = constraints.get(name) if name is not None else None
        if not isinstance(constraint, dict) or "modulus" not in constraint:
            continue
        chain["frame_snap"] = references.make_ref(references.CONSTRAINT, name)
        derived.append((step.get("name"), name))
    return derived


def _head_trim(chain_definition):
    """Head frames each segment after the first drops: the guide a `guide`
    chain laid in, else `trim_frames`. The one owner of that rule:
    `ChainConfig` reads its head trim here, as `_duration_plan` (and so
    `chain_segment_count`, the cost estimate's) does."""
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


def _guide_holds_audio(chain_definition):
    """Whether a `guide` chain carries its audio across each seam
    (`carry_audio`, default true), so the seam needs no crossfade. The guide
    holds only its own span, which the head trim drops: what each segment
    keeps is still generated (`chain_holds_audio` does not count it)."""
    return chain_definition.get(
        "continuity", "last_frame"
    ) == GUIDE_CONTINUITY and bool(chain_definition.get("carry_audio", True))


def _guide_carry_held(chain_definition, arguments):
    """Whether a `guide` chain holds a matched track (`match_audio` with a
    step `hold_audio`), which the guide's audio carry would hold again."""
    return (
        chain_definition.get("continuity", "last_frame") == GUIDE_CONTINUITY
        and bool(chain_definition.get("match_audio"))
        and _argument(arguments, HOLD_AUDIO_INPUT) is not None
    )


def chain_holds_audio(chain_definition, arguments):
    """Whether the chain's audio is held rather than generated by each
    segment: a `match_audio` track (and its `hold_audio`), or a step
    `hold_audio`. A guide carrying its audio is not: the held guide span is
    the head trim, so every kept sample is generated and can drop out.
    Pure: the run's `ChainConfig` and the cost estimate both ask it."""
    return (
        bool(chain_definition.get("match_audio"))
        or _argument(arguments, HOLD_AUDIO_INPUT) is not None
    )


def _floor_setting(chain_definition):
    """The check's `audio_dbfs_min` as written, or the default when the
    check is off or does not name it."""
    check = chain_definition.get("check")
    if isinstance(check, dict) and "audio_dbfs_min" in check:
        return check["audio_dbfs_min"]
    return AUDIO_DBFS_MIN


def chain_audio_floor(chain_definition, arguments):
    """The dBFS a kept segment's generated audio is warned
    `chain_segment_silent` under (#1221), whether or not the check is on:
    the check's `audio_dbfs_min` (null turns it off), the default with no
    check, and None when the chain holds its audio. Pure."""
    if chain_holds_audio(chain_definition, arguments):
        return None
    floor = _floor_setting(chain_definition)
    return None if floor is None else float(floor)


def chain_checks_segment_1(chain_definition, arguments):
    """Whether segment 1 is checked: it has no seam, so only for its audio
    level (#1221 Q3) or its words (#1262) - with the check on, the audio
    generated, and the floor or the words measure on. Pure: the run's
    `ChainConfig` and the cost estimate both ask it; it never reads the
    floor's or the minimum's value, so an unresolved reference counts as
    on."""
    check = chain_definition.get("check")
    return (
        (check is True or isinstance(check, dict))
        and not chain_holds_audio(chain_definition, arguments)
        and (
            _floor_setting(chain_definition) is not None
            or chain_checks_words(chain_definition, arguments)
        )
    )


def chain_checks_words(chain_definition, arguments):
    """Whether a chain holds each segment's kept audio to its line (#1262):
    a `check` object whose `words_similarity_min` is not null - the default
    is null, the measure off, since bench/records/words.md found no clean
    separation of good takes from bad - and the audio generated: a held
    track is not the segment's own speech. Pure, as
    `chain_checks_segment_1`; an unresolved reference counts as on."""
    check = chain_definition.get("check")
    return (
        isinstance(check, dict)
        and check.get("words_similarity_min") is not None
        and not chain_holds_audio(chain_definition, arguments)
    )


def segment_line(lines, prompts, prompt, index):
    """The line segment `index` (0-based) is meant to speak, or None when it
    has none (#1262): `lines[index]` when the chain has `lines` (one per
    segment, never repeated past its end), else the `<d>` bodies
    (`script_lines.dialogue_text`) of its prompt - `prompts[min(index,
    len - 1)]`, as the segment renders with, else the step's `prompt`. A
    prompt with no `<d>` has no line, and is not held to silence."""
    if isinstance(lines, list) and lines:
        line = lines[index] if index < len(lines) else None
        return line if isinstance(line, str) and line.strip() else None
    source = prompts[min(index, len(prompts) - 1)] if prompts else prompt
    return dialogue_text(source) or None


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
    problem = match_audio_source_problem(arguments)
    if problem:
        raise ValueError(problem)
    references = arguments.get("references") or []

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
