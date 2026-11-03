"""The MiniMax-H3 argument wrappers `Pipeline` runs before a call
(`_call_pipeline`, dw/pipeline_processors/pipeline.py): each takes the call's
arguments, checks the H3 argument it owns against the loaded pipeline and hands
back the arguments the block graph takes, or raises the step's refusal. They are
methods of `Pipeline` through the `H3Arguments` mixin, so `Pipeline._with_guides`
and the rest are still named from there; the rules they apply live in
`h3_rules.py` and the owners of each block (`h3_hold`, `h3_guides`, `h3_anchor`,
`h3_mask`), never here.
"""

from .h3_anchor import anchor_refusal, prepared_anchor
from .h3_guides import guide_frames_array, guides_refusal, prepared_guide
from .h3_hold import hold_audio_reference, holds_audio, refines
from .h3_mask import binarized, edits_regions
from .h3_rules import (
    GUIDES_INPUT,
    GUIDE_LIMIT,
    GUIDE_NOISE_INPUT,
    LATENT_ANCHOR_INPUT,
    HELD_AUDIO_OUTPUT,
    HELD_AUDIO_RATE_OUTPUT,
    HOLD_AUDIO_INPUT,
    MASK_IMAGE_INPUT,
    MASK_INPUTS,
    REFINE_SIGMAS_INPUT,
    REFINE_STRENGTH_INPUT,
    REFINE_TILES_INPUT,
    REGENERATE_MASK,
    SOURCE_VIDEO_INPUT,
    edits_region,
    guide_noise_problem,
    mask_problems,
    media_problems,
    refine_problems,
)
from ..events import emit_log


def latent_anchor_source(chain_source, value):
    """What a `latent_anchor` names, for the job log (#1181): the chain
    segment that supplied it, the reference given, or frames passed in."""
    if chain_source:
        return chain_source
    if isinstance(value, str):
        return f"'{value}'"
    return "the frames given in the step's arguments"


class H3Arguments:
    """The H3 argument wrappers, mixed into `Pipeline` (which supplies `name`,
    `pipeline`, `base_dir`, `segment_label` and `latent_anchor_source`)."""

    def _with_held_audio(self, arguments):
        """`hold_audio` as the reference the H3 hold block takes, and the
        held track asked for alongside `audio` (dw/pipeline_processors/h3_hold.py).

        Raises:
            ValueError: If this pipeline cannot hold audio, or the value is not audio
        """
        held = arguments.get(HOLD_AUDIO_INPUT)
        if held is None:
            if HOLD_AUDIO_INPUT not in arguments:
                return arguments
            # A template's null `hold_audio` variable is no hold at all, the
            # same call as one that never named the argument
            return {k: v for k, v in arguments.items() if k != HOLD_AUDIO_INPUT}
        if not holds_audio(self.pipeline):
            raise ValueError(
                f"Step '{self.name}': hold_audio is a MiniMax-H3 argument "
                f"(t2va, fl2va or ref2va), and {type(self.pipeline).__name__} "
                f"cannot hold a soundtrack"
            )
        arguments = dict(arguments)
        arguments[HOLD_AUDIO_INPUT] = hold_audio_reference(held, self.base_dir)
        output = arguments.get("output")
        # A single output is written as a string; the held keys ride along
        # only when the track itself is asked for
        names = [output] if isinstance(output, str) else list(output or [])
        if "audio" in names:
            arguments["output"] = names + [HELD_AUDIO_OUTPUT, HELD_AUDIO_RATE_OUTPUT]
        return arguments

    def _with_guides(self, arguments):
        """`guides` as the H3 guide layout takes them - each clip as uint8
        frames cut to a whole-latent length (dw/pipeline_processors/h3_guides.py).
        An empty list is no guides.

        A guide with `"audio": true` also carries its video's soundtrack, as
        `audio` and `sample_rate`; with `audio` false or absent it carries none.

        Raises:
            ValueError: If this pipeline cannot take guides, the step also passes
                `references`, a guide is not `{video, frame, audio?}` with a video,
                or `audio` is true on a video with no audio
        """
        guides = arguments.get(GUIDES_INPUT)
        if guides is None:
            return arguments
        arguments = dict(arguments)
        if isinstance(guides, (list, tuple)) and not guides:
            del arguments[GUIDES_INPUT]
            return arguments
        where = f"Step '{self.name}': guides"
        refusal = guides_refusal(self.pipeline)
        if refusal:
            raise ValueError(f"{where}: {refusal}")
        if arguments.get("references") is not None:
            raise ValueError(
                f"{where} cannot be combined with references - ref2va lays out "
                f"its own conditioning; use guides on t2va or fl2va"
            )
        if not isinstance(guides, (list, tuple)):
            raise ValueError(
                f"{where} must be a list of {{video, frame}}, got {type(guides).__name__}"
            )
        if len(guides) > GUIDE_LIMIT:
            raise ValueError(
                f"{where} takes at most {GUIDE_LIMIT} clips, got {len(guides)}"
            )
        prepared = [
            prepared_guide(where, index, guide, self.base_dir)
            for index, guide in enumerate(guides)
        ]
        arguments[GUIDES_INPUT] = prepared
        return arguments

    def _with_guide_noise(self, arguments):
        """`guide_noise` checked before the call: a number in [0, 1) on a
        pipeline with the guide layout (dw/pipeline_processors/h3_guide_steps.py).
        None or 0 is dropped, so the call is today's.

        Raises:
            ValueError: If the value is not a number in [0, 1), or this
                pipeline cannot take guides
        """
        if GUIDE_NOISE_INPUT not in arguments:
            return arguments
        value = arguments[GUIDE_NOISE_INPUT]
        problem = None if value is None else guide_noise_problem(value)
        if value is None or (value == 0 and problem is None):
            arguments = dict(arguments)
            del arguments[GUIDE_NOISE_INPUT]
            return arguments
        where = f"Step '{self.name}'"
        if problem:
            raise ValueError(f"{where}: {problem}")
        refusal = guides_refusal(self.pipeline)
        if refusal:
            raise ValueError(f"{where}: {GUIDE_NOISE_INPUT} noises guides - {refusal}")
        if arguments.get("references") is not None:
            raise ValueError(
                f"{where}: {GUIDE_NOISE_INPUT} cannot be combined with references - "
                f"ref2va lays out its own conditioning; use it on t2va or fl2va"
            )
        arguments = dict(arguments)
        arguments[GUIDE_NOISE_INPUT] = float(value)
        return arguments

    def _with_latent_anchor(self, arguments):
        """`latent_anchor` as the H3 anchor step takes it - uint8 frames cut to
        a whole-latent length (dw/pipeline_processors/h3_anchor.py). None is no
        anchor.

        Raises:
            ValueError: If this pipeline has no anchor step (not MiniMax-H3
                t2va/fl2va), the step also passes `references` (the call would
                run ref2va, which has no anchor step), or the value is not
                frames of a video
        """
        value = arguments.get(LATENT_ANCHOR_INPUT)
        if value is None:
            return arguments
        where = f"Step '{self.name}': {LATENT_ANCHOR_INPUT}"
        refusal = anchor_refusal(self.pipeline)
        if refusal:
            raise ValueError(f"{where}: {refusal}")
        if arguments.get("references") is not None:
            raise ValueError(
                f"{where} cannot be combined with references - the call would run "
                f"ref2va, which has no anchor step; use latent_anchor on t2va or fl2va"
            )
        arguments = dict(arguments)
        arguments[LATENT_ANCHOR_INPUT] = prepared_anchor(where, value, self.base_dir)
        # The one line that says the anchor applies to this call - without it
        # the job log shows only a block name (#1181)
        label = getattr(self, "segment_label", None) or self.name
        source = latent_anchor_source(
            getattr(self, "latent_anchor_source", None), value
        )
        emit_log(f"{label}: latent anchor applied - latent statistics from {source}")
        return arguments

    def _with_mask(self, arguments):
        """A regional edit's `source_video` as uint8 frames and its mask as the
        bool frames the H3 source block maps to rows, under `REGENERATE_MASK`
        (dw/pipeline_processors/h3_mask.py).

        A step without `width`, `height` or `num_frames` renders the source's.

        Raises:
            ValueError: If this pipeline cannot take a regional edit, the
                arguments break `mask_problems`, the mask is all black, or the
                source or mask does not fit the canvas
        """
        takes = edits_regions(self.pipeline)
        # An inpaint pipeline's own mask_image is no regional edit
        if not edits_region(arguments, takes):
            return arguments
        where = f"Step '{self.name}'"
        if not takes:
            raise ValueError(
                f"{where}: {SOURCE_VIDEO_INPUT} is a MiniMax-H3 argument (t2va, "
                f"fl2va or ref2va), and {type(self.pipeline).__name__} cannot "
                f"edit a region"
            )
        problems = mask_problems(arguments, chained=self.segment_label is not None)
        if problems:
            raise ValueError(f"{where}: {problems[0]}")
        arguments = dict(arguments)
        try:
            source = guide_frames_array(arguments[SOURCE_VIDEO_INPUT])
        except ValueError as e:
            raise ValueError(f"{where}: {SOURCE_VIDEO_INPUT} is not a video: {e}")
        arguments[SOURCE_VIDEO_INPUT] = source
        frames, height, width = source.shape[:3]
        for key, value in (
            ("width", width),
            ("height", height),
            ("num_frames", frames),
        ):
            if arguments.get(key) is None:
                arguments[key] = value
        problems = media_problems(
            SOURCE_VIDEO_INPUT,
            source.shape[:3],
            arguments["width"],
            arguments["height"],
            arguments["num_frames"],
        )
        name = next((n for n in MASK_INPUTS if arguments.get(n) is not None), None)
        if name is not None:
            value = arguments.pop(name)
            try:
                if name == MASK_IMAGE_INPUT:
                    mask = binarized(value)
                else:
                    mask = binarized(guide_frames_array(value))
            except (AttributeError, ValueError) as e:
                kind = "an image" if name == MASK_IMAGE_INPUT else "a video"
                raise ValueError(f"{where}: {name} is not {kind}: {e}")
            if not mask.any():
                raise ValueError(
                    f"{where}: {name} is all black, so nothing would regenerate "
                    f"- white marks the region to regenerate"
                )
            problems += media_problems(
                name,
                mask.shape,
                arguments["width"],
                arguments["height"],
                arguments["num_frames"],
            )
            arguments[REGENERATE_MASK] = mask
        if problems:
            raise ValueError(f"{where}: {problems[0]}")
        return arguments

    def _check_refine(self, arguments):
        """Refuse a `refine_strength` the H3 refine block cannot run, before the
        call (dw/pipeline_processors/h3_hold.py).

        Raises:
            ValueError: If this pipeline cannot refine, the strength, the
                `refine_sigmas` schedule or the step count is out of range, or the
                step passes no `latents` or `hold_audio`
        """
        given = [
            name
            for name in (REFINE_STRENGTH_INPUT, REFINE_SIGMAS_INPUT, REFINE_TILES_INPUT)
            if arguments.get(name) is not None
        ]
        if not given:
            return
        where = f"Step '{self.name}': {given[0]}"
        if not refines(self.pipeline):
            raise ValueError(
                f"{where} is a MiniMax-H3 argument (t2va, fl2va or ref2va), and "
                f"{type(self.pipeline).__name__} cannot refine"
            )
        problems = refine_problems(arguments)
        if problems:
            raise ValueError(f"Step '{self.name}': {problems[0]}")
