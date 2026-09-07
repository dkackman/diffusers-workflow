"""What a workflow makes, read off its definition.

A catalog entry is chosen by shape before anything else - one still, a set,
one shot, a cut sequence - and the definition already says which, in the
steps it has and what they feed each other. Deriving it here means the
vocabulary cannot go stale against the file and needs no backfill; a
declared `shape`/`traits`/`summary` overrides for the odd workflow the
rules misread.

Reads the raw JSON only: no variable substitution, no type loading, so it
costs a dict walk and behaves the same on a repo template and a file an
agent saved a second ago. Nothing here names a model family - the rules
read structure (a concat step, a `references` argument), never checkpoints.
"""

import re

SHAPES = ("image", "image-set", "image-edit", "shot", "sequence", "audio", "text", "utility")
TRAITS = (
    "has-audio",
    "chained",
    "image-conditioned",
    "identity-referenced",
    "needs-input-media",
    "composes-workflows",
)
# Tasks that create content rather than process it. A workflow made only
# of processing tasks is a utility.
GENERATIVE_TASKS = frozenset(
    {"generate_speech", "text_generation", "image_to_text", "diffusion_upscale", "interpolate_frames"}
)
SUMMARY_LIMIT = 120

_KIND_PRECEDENCE = ("video", "audio", "image", "text")
_EDIT_PIPELINE = re.compile(r"inpaint|img2img|edit|upscale|outpaint", re.I)
_CHAIN_ARGUMENTS = frozenset({"last_frame", "last_segment", "last_image", "match_audio"})
_MEDIA_ARGUMENTS = frozenset({"image", "video", "audio", "mask_image", "urls"})
_CUT_TASKS = frozenset({"concat_videos", "dissolve_videos"})
# Components that exist only to synthesise a waveform. A video pipeline
# carrying one emits an audio track whether or not it says so in `output`.
_AUDIO_COMPONENTS = frozenset({"vocoder", "audio_vae"})
_SENTENCE_END = re.compile(r"(?<=[.!?])\s|\n")


def _steps(definition):
    steps = definition.get("steps") if isinstance(definition, dict) else None
    return [step for step in steps if isinstance(step, dict)] if isinstance(steps, list) else []


def _block(step):
    """The step's one body: pipeline, pipeline_reference, task or workflow."""
    for key in ("pipeline", "pipeline_reference", "task", "workflow"):
        body = step.get(key)
        if isinstance(body, dict):
            return key, body
    return None, {}


def _arguments(step):
    _kind, body = _block(step)
    arguments = body.get("arguments")
    return arguments if isinstance(arguments, dict) else {}


def _kind(step):
    result = step.get("result")
    if isinstance(result, dict) and isinstance(result.get("content_type"), str):
        return result["content_type"].split("/")[0]
    return None


def _generates(step):
    """Whether the step creates content: a pipeline, a reference to one, a
    sub-workflow, or a generative task."""
    key, body = _block(step)
    if key in ("pipeline", "pipeline_reference", "workflow"):
        return True
    return key == "task" and body.get("command") in GENERATIVE_TASKS


def _component_type(step):
    key, body = _block(step)
    if key != "pipeline":
        return ""
    configuration = body.get("configuration")
    if not isinstance(configuration, dict):
        return ""
    return str(configuration.get("component_type", ""))


def _fed_by(value):
    """Distinct step names a value's previous_result references name."""
    found = set()
    if isinstance(value, str) and value.startswith("previous_result:"):
        found.add(value.split(":", 1)[1].split(".", 1)[0])
    elif isinstance(value, list):
        for item in value:
            found |= _fed_by(item)
    elif isinstance(value, dict):
        for item in value.values():
            found |= _fed_by(item)
    return found


def _walk(value):
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _needs_input_media(steps):
    for step in steps:
        arguments = _arguments(step)
        for name, value in arguments.items():
            if name not in _MEDIA_ARGUMENTS:
                continue
            # A list argument (gather_images' `urls`) carries the same fact
            # one level in.
            candidates = value if isinstance(value, list) else [value]
            if any(isinstance(item, str) and item.startswith("variable:") for item in candidates):
                return True
        for value in _walk(arguments):
            if isinstance(value, str) and value.startswith("asset:"):
                return True
            if isinstance(value, dict) and "location" in value:
                return True
    return False


def _cuts_together(steps):
    """A concat or dissolve fed by two or more distinct steps."""
    for step in steps:
        key, body = _block(step)
        if key == "task" and body.get("command") in _CUT_TASKS:
            if len(_fed_by(_arguments(step).get("videos"))) >= 2:
                return True
    return False


def _audio_components(step):
    """Audio-only components the step's pipeline configures."""
    key, body = _block(step)
    if key != "pipeline":
        return set()
    configuration = body.get("configuration")
    if not isinstance(configuration, dict) or not isinstance(configuration.get("components"), dict):
        return set()
    return set(configuration["components"]) & _AUDIO_COMPONENTS


def _derive_shape(steps, kind):
    # Cutting shots together makes a sequence even when every shot was
    # supplied rather than generated - an edit is what comes out, and the
    # utility fallthrough would otherwise hide the assembly templates.
    if kind == "video" and _cuts_together(steps):
        return "sequence"
    if kind is None or not any(_generates(step) for step in steps):
        return "utility"
    if kind == "text":
        return "text"
    if kind == "audio":
        return "audio"
    if kind == "video":
        return "shot"
    image_steps = [step for step in steps if _generates(step) and _kind(step) == "image"]
    for step in image_steps:
        if _EDIT_PIPELINE.search(_component_type(step)):
            return "image-edit"
        if {"image", "mask_image"} & set(_arguments(step)):
            return "image-edit"
    if len(image_steps) >= 2 or any(_block(step)[0] == "workflow" for step in image_steps):
        return "image-set"
    return "image"


def _derive_traits(steps):
    """The independent facts about how the output is made or what it needs.

    `has-audio` says the workflow emits a generated audio track - a speech
    task, a video pipeline asked for audio, or one carrying a component that
    exists only to synthesise a waveform. Not specifically dialogue.
    """
    traits = set()
    for step in steps:
        key, body = _block(step)
        arguments = _arguments(step)
        if key == "task" and body.get("command") == "generate_speech":
            traits.add("has-audio")
        output = arguments.get("output")
        if _kind(step) == "video" and isinstance(output, list) and "audio" in output:
            traits.add("has-audio")
        if _kind(step) == "video" and _audio_components(step):
            traits.add("has-audio")
        if key == "pipeline" and "chain" in body:
            traits.add("chained")
        if _CHAIN_ARGUMENTS & set(arguments):
            traits.add("chained")
        if _kind(step) == "video" and ("image" in arguments or "ImageToVideo" in _component_type(step)):
            traits.add("image-conditioned")
        if "references" in arguments:
            traits.add("identity-referenced")
        if key == "workflow":
            traits.add("composes-workflows")
    if _needs_input_media(steps):
        traits.add("needs-input-media")
    return sorted(traits)


def _truncate(text):
    """One line, cut at a word boundary and elided if it runs past the limit.

    A declared summary goes through this too: `workflow_details` reads a file
    without validating it, so the schema's maxLength never runs on that path
    and a long declaration would otherwise reach a listing unclipped.
    """
    if len(text) <= SUMMARY_LIMIT:
        return text, False
    cut = text[: SUMMARY_LIMIT - 1]
    cut = cut[: cut.rfind(" ")] if " " in cut else cut
    return cut.rstrip() + "…", True


def _derive_summary(description):
    text = str(description or "").strip()
    if not text:
        return "", False
    return _truncate(_SENTENCE_END.split(text, maxsplit=1)[0].strip())


def derive_catalog_metadata(definition):
    """shape, traits and summary for one definition, declarations honoured.

    Returns {shape, traits, summary, summary_truncated, declared}; `declared`
    names which of the three came from the file rather than the rules, so a
    test can refuse a declaration that merely repeats the derivation.
    """
    if not isinstance(definition, dict):
        definition = {}
    steps = _steps(definition)
    kinds = {_kind(step) for step in steps} - {None}
    kind = next((k for k in _KIND_PRECEDENCE if k in kinds), None)

    declared = set()
    shape = _derive_shape(steps, kind)
    if definition.get("shape") in SHAPES:
        shape = definition["shape"]
        declared.add("shape")
    traits = _derive_traits(steps)
    if isinstance(definition.get("traits"), list):
        traits = sorted(t for t in definition["traits"] if t in TRAITS)
        declared.add("traits")
    summary, truncated = _derive_summary(definition.get("description"))
    if isinstance(definition.get("summary"), str) and definition["summary"].strip():
        summary, truncated = _truncate(definition["summary"].strip())
        declared.add("summary")
    return {
        "shape": shape,
        "traits": traits,
        "summary": summary,
        "summary_truncated": truncated,
        "declared": declared,
    }


COMPACT_FIELDS = (
    "summary",
    "shape",
    "traits",
    "cost",
    "kinds",
    "variable_names",
    "configures",
)


def project_listing(details, *, shape=None, traits=None, configures=None, include_models=False, view=None):
    """The listing an agent asked for: filtered by shape and traits, and in
    the compact view stripped to what choosing a template needs.

    Compact is templates-only unless `include_models` or `configures` says
    otherwise - nine checkpoint variants of text-to-image are the noise the
    two-tree split removed. A workflow with no `configures` is a template
    for this purpose, whichever directory it sits in: a user wrote it to be
    found. The full view never drops entries or fields.
    """
    if shape is not None and shape not in SHAPES:
        raise ValueError(f"Unknown shape {shape!r}. The shapes are: {', '.join(SHAPES)}.")
    traits = list(traits or [])
    unknown = [t for t in traits if t not in TRAITS]
    if unknown:
        raise ValueError(f"Unknown trait(s) {', '.join(unknown)}. The traits are: {', '.join(TRAITS)}.")
    if view not in (None, "compact"):
        raise ValueError("view must be 'compact' or omitted")

    compact = view == "compact"
    keep_models = include_models or configures is not None or not compact
    projected = {}
    for name, detail in details.items():
        is_model = bool(detail.get("configures") or detail.get("configures_missing"))
        if shape is not None and detail.get("shape") != shape:
            continue
        if traits and not set(traits) <= set(detail.get("traits", [])):
            continue
        if configures is not None and detail.get("configures") != configures:
            continue
        if is_model and not keep_models:
            continue
        if compact:
            slim = {key: detail.get(key) for key in COMPACT_FIELDS if key != "configures" or detail.get(key)}
            if detail.get("configures_missing"):
                slim["configures_missing"] = detail["configures_missing"]
            projected[name] = slim
        else:
            projected[name] = detail
    return projected
