"""How many references of each kind a pipeline's request may carry.

A pipeline that conditions on reference media - MiniMax-H3's `ref2va` is the
one in the catalog - bounds what it accepts: so many images, so many videos,
so many audio clips, so many in total, and for H3 an audio reference may never
be the only one. Those bounds are enforced by the pipeline itself, which means
they are enforced *after* the checkpoint is loaded: a caller who ran the free
`validate_workflow`, was quoted eight minutes and acknowledged the cost found
out minutes in, from a failed job, what a millisecond of arithmetic could have
told them (#136).

The numbers are not written here. They live on the diffusers block that
enforces them, as its constructor's defaults, and this module reads them off
that block - so a diffusers release that raises a limit raises it here too,
and the engine holds no model knowledge but the name of the class that
declares the limits (see REFERENCE_LIMIT_BLOCKS). A family diffusers has no
such block for is simply not checked: this pass only ever refuses a request
the pipeline itself would refuse.
"""

import importlib
import inspect
import logging

from . import references
from .for_each import render_path

logger = logging.getLogger("dw")

# The module a family's reference classes live in -> the block whose
# __init__ defaults declare that family's limits. A pointer, not a number:
# what a limit *is* stays diffusers', which is the only place it can stay
# correct across a release
H3_FAMILY = "diffusers.modular_pipelines.minimax_h3"

REFERENCE_LIMIT_BLOCKS = {
    H3_FAMILY: (
        "diffusers.modular_pipelines.minimax_h3.before_encoder",
        "MiniMaxH3Ref2VASetupStep",
    ),
}

# Families where an audio reference may not stand alone - H3 conditions a
# soundtrack on a picture, so audio by itself has nothing to speak over
AUDIO_NEEDS_A_PICTURE = frozenset(REFERENCE_LIMIT_BLOCKS)

REFERENCE_TYPE_KEY = "reference_type"


def _family(module_name):
    """The REFERENCE_LIMIT_BLOCKS key a class's module belongs to, or None."""
    for family in REFERENCE_LIMIT_BLOCKS:
        if module_name == family or module_name.startswith(family + "."):
            return family
    return None


def _limits(family):
    """The (model name, per-kind limits, total) a family declares.

    Read from the block's constructor signature rather than from an instance:
    constructing one is cheap but not free, and a default is exactly what the
    signature holds. The model name is the block's own, so even the family's
    name in the message is diffusers'.
    """
    module_path, class_name = REFERENCE_LIMIT_BLOCKS[family]
    try:
        block = getattr(importlib.import_module(module_path), class_name)
        parameters = inspect.signature(block.__init__).parameters
    except Exception:
        # A diffusers that renamed or dropped the block - checking nothing is
        # the right failure here, since the pipeline still enforces its own
        logger.debug(f"No reference limits available from {family}", exc_info=True)
        return None, None, None

    per_kind = {}
    total = None
    for name, parameter in parameters.items():
        if parameter.default is inspect.Parameter.empty:
            continue
        if name == "max_references":
            total = parameter.default
        elif name.startswith("max_") and name.endswith("s"):
            per_kind[name[len("max_") : -1]] = parameter.default
    model_name = getattr(block, "model_name", family.rsplit(".", 1)[-1])
    return model_name, (per_kind or None), total


def _reference_class(value):
    """The class a reference entry's '*_type' names, or None.

    Anything that does not resolve, or that the untrusted gate refuses, is
    left alone: realize_args reports a type it cannot load, and the type
    reference check a refused one, with a better message than this pass
    could give.
    """
    if not isinstance(value, dict):
        return None
    name = value.get(REFERENCE_TYPE_KEY)
    # A value substitution resolves before this pass, still spelled out, is
    # another pass's complaint
    if not isinstance(name, str) or references.is_ref(references.UNRESOLVED, name):
        return None
    if "." not in name:
        return None
    # Through the run's own resolver, so an untrusted workflow's name meets
    # the same trust gate here, at validate time, before anything imports
    from .type_helpers import load_type_from_full_name

    try:
        return load_type_from_full_name(name, REFERENCE_TYPE_KEY)
    except Exception:
        return None


def _kinds(entries):
    """(family, [kind, ...]) for a list of reference entries, or None.

    A family is the one whose limits this pass knows how to read; an entry
    whose class carries no 'kind', or a list mixing two families, is not a
    reference set this pass understands.
    """
    if not isinstance(entries, list) or not entries:
        return None
    found = None
    kinds = []
    for entry in entries:
        reference = _reference_class(entry)
        kind = getattr(reference, "kind", None)
        if not isinstance(kind, str):
            return None
        family = _family(getattr(reference, "__module__", ""))
        if family is None:
            return None
        if found is None:
            found = family
        elif found != family:
            return None
        kinds.append(kind)
    return (found, kinds) if found else None


def _errors_for(kinds, family):
    """Every limit a set of reference kinds breaks, as messages."""
    model_name, per_kind, total = _limits(family)
    if per_kind is None and total is None:
        return []

    messages = []
    for kind, limit in sorted((per_kind or {}).items()):
        count = kinds.count(kind)
        if count > limit:
            messages.append(
                f"{model_name} accepts at most {limit} "
                f"{kind} reference{'s' if limit != 1 else ''}, got {count}."
            )
    if total is not None and len(kinds) > total:
        messages.append(
            f"{model_name} accepts at most {total} references in total, "
            f"got {len(kinds)}."
        )
    if family in AUDIO_NEEDS_A_PICTURE and set(kinds) == {"audio"}:
        messages.append(
            "An audio reference has to be paired with at least one image or "
            "video reference and cannot be used on its own."
        )
    return messages


def _untyped_entries(entries):
    """The indices of a reference list's entries that are bare strings.

    A family in REFERENCE_LIMIT_BLOCKS takes only reference objects - H3's
    setup block refuses anything else, after the load, because it "never
    opens media files" - and nothing in dw turns a bare `asset:`, `output:`,
    `previous_result:` or path string into one: only a {reference_type,
    from_file} description is built. Left uncounted, such a list also slips
    past the limits (#1374). A `variable:` or `item:` still spelled out is
    another pass's complaint.
    """
    if not isinstance(entries, list):
        return []
    return [
        index
        for index, entry in enumerate(entries)
        if isinstance(entry, str)
        and not references.is_ref((references.VARIABLE, references.ITEM), entry)
    ]


def _untyped_message(index, entry):
    return (
        f"references[{index}] is the bare string {entry!r}, and MiniMax-H3 "
        f"takes only reference objects - write it as "
        f'{{"{REFERENCE_TYPE_KEY}": "{H3_FAMILY}.MiniMaxH3ImageReference", '
        f'"from_file": "{entry}"}} (VideoReference or AudioReference for '
        f"other media)."
    )


def reference_limit_errors(workflow_definition, source_indices=None):
    """Every reference list a pipeline would refuse, as [{path, message}].

    The definition handed here has already been substituted and expanded, so a
    `for_each` member's own references are checked as they will run;
    `source_indices` maps each expanded step back to the step the author
    wrote, and the member is named in the message - the same convention
    subfolder_errors uses.
    """
    from .pipeline_processors.h3_rules import is_h3_step

    errors = []
    for _, _, pipeline, source, where in references.iter_steps(
        workflow_definition.get("steps"), source_indices, "pipeline"
    ):
        arguments = pipeline.get("arguments")
        if not isinstance(arguments, dict):
            continue
        if is_h3_step(pipeline):
            entries = arguments.get("references")
            for index in _untyped_entries(entries):
                errors.append(
                    {
                        "path": render_path(
                            (
                                "steps",
                                source,
                                "pipeline",
                                "arguments",
                                "references",
                                index,
                            )
                        ),
                        "message": (
                            f"{_untyped_message(index, entries[index]).rstrip('.')}"
                            f"{where}."
                        ),
                    }
                )
        for key, value in arguments.items():
            found = _kinds(value)
            if found is None:
                continue
            family, kinds = found
            for message in _errors_for(kinds, family):
                errors.append(
                    {
                        "path": render_path(
                            ("steps", source, "pipeline", "arguments", key)
                        ),
                        "message": f"{message.rstrip('.')}{where}.",
                    }
                )
    return errors


# The block whose constructor default is the rate a video reference is read at
VIDEO_SAMPLE_BLOCK = (
    "diffusers.modular_pipelines.minimax_h3.encoders",
    "MiniMaxH3Ref2VATextEncoderStep",
)
VIDEO_KIND = "video"


def _minimum_reference_frames():
    """(sample_fps, fps, minimum frames at fps, merge group), or None.

    Read the way diffusers' `_sample_video_condition_frames` reads it: the
    conditioner samples every `fps / sample_fps`-th frame and Qwen3-VL merges
    the samples in groups of `temporal_patch_size`, so a group needs
    `round((group - 1) * stride) + 1` frames. `sample_fps` is the encoder
    block's constructor default, `fps` the pipeline's own property and the
    group the video processor's - none written here. Anything unreadable
    means no check, and the pipeline still refuses at run time.
    """
    from .pipeline_limits import _limits as pipeline_limits

    try:
        module_path, class_name = VIDEO_SAMPLE_BLOCK
        block = getattr(importlib.import_module(module_path), class_name)
        sample_fps = inspect.signature(block.__init__).parameters["video_sample_fps"]
        sample_fps = sample_fps.default
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
            Qwen3VLVideoProcessor,
        )

        group = Qwen3VLVideoProcessor.temporal_patch_size
        limits = pipeline_limits()
        fps = limits["fps"] if limits else None
        if not (
            isinstance(sample_fps, (int, float))
            and sample_fps > 0
            and isinstance(group, int)
            and group > 1
            and fps
        ):
            return None
    except Exception:
        logger.debug(
            "No reference video length available from diffusers", exc_info=True
        )
        return None
    return sample_fps, fps, round((group - 1) * fps / sample_fps) + 1, group


def _reference_clip(entry, steps, base_dir, probe):
    """(frames, fps) of a video reference entry, each None when it cannot be
    told before the run: a `from_previous_result` naming a `trim_video` with a
    literal `num_frames`, or a `from_file` that resolves to a probed file."""
    from .hold_audio import _trim_arguments, _trimmed_frames
    from .probe_paths import resolve_probe_path

    previous = entry.get("from_previous_result")
    if isinstance(previous, str):
        spelled = references.make_ref(references.PREVIOUS_RESULT, previous)
        frames = _trimmed_frames(steps, spelled)
        trim = _trim_arguments(steps, spelled) or {}
        fps = trim.get("fps")
        source = trim.get("video")
        info = None
        if probe is not None and (path := resolve_probe_path(source, base_dir)):
            info = probe(path)
        rate = info.get("fps") if isinstance(info, dict) else None
        rate = rate if isinstance(rate, (int, float)) and rate > 0 else fps
        return frames, rate if isinstance(rate, (int, float)) and rate > 0 else None
    if probe is None:
        return None, None
    path = resolve_probe_path(entry.get("from_file"), base_dir)
    info = probe(path) if path else None
    if not isinstance(info, dict) or info.get("kind") != "video":
        return None, None
    frames, rate = info.get("frame_count"), info.get("fps")
    frames = frames if isinstance(frames, int) and frames > 0 else None
    return frames, rate if isinstance(rate, (int, float)) and rate > 0 else None


def reference_length_errors(
    workflow_definition, source_indices=None, base_dir=None, probe=None
):
    """Every H3 video reference too short for the conditioner, as [{path, message}].

    A video reference is resampled to the pipeline's rate, sampled and merged
    in groups; one with too few frames is refused by the encoder after the
    checkpoint loads, and in a chain after the earlier steps have run (#1422).
    Checked only where the length is knowable now: a `trim_video` result with
    a literal `num_frames`, or an `asset:`/`output:`/path file the probe
    reads. A length held in a variable stays the run's to refuse. A clip
    refused only if its length at the pipeline's rate is certainly short -
    never on a rounding borderline.
    """
    from .pipeline_processors.h3_rules import is_h3_step

    steps = workflow_definition.get("steps")
    errors, minimum = [], None
    for _, _, pipeline, source, where in references.iter_steps(
        steps, source_indices, "pipeline"
    ):
        arguments = pipeline.get("arguments")
        entries = arguments.get("references") if isinstance(arguments, dict) else None
        if not (is_h3_step(pipeline) and isinstance(entries, list)):
            continue
        for index, entry in enumerate(entries):
            reference = _reference_class(entry)
            if getattr(reference, "kind", None) != VIDEO_KIND:
                continue
            if minimum is None:
                minimum = _minimum_reference_frames() or False
            if not minimum:
                return errors
            sample_fps, fps, needed, group = minimum
            frames, rate = _reference_clip(entry, steps, base_dir, probe)
            if not frames or not rate:
                continue
            # Rounded up: a count that might just make the minimum is left
            at_pipeline_rate = -(-frames * fps // rate)
            if at_pipeline_rate >= needed:
                continue
            errors.append(
                {
                    "path": render_path(
                        ("steps", source, "pipeline", "arguments", "references", index)
                    ),
                    "message": (
                        f"A reference video is read at {sample_fps:g} fps and its "
                        f"sampled frames are merged in groups of {group}, so it "
                        f"must run at least {needed} frames at {fps:g} fps "
                        f"({needed / fps:.2g} seconds), got {frames} frames at "
                        f"{rate:g} fps{where}."
                    ),
                }
            )
    return errors
