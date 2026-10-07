"""A `guides` argument, checked before the run when it can be.

`guides` (`[{"video": ..., "frame": ..., "audio": true?}]`) lays clips of an existing video into a
MiniMax-H3 generation at chosen frames (dw/pipeline_processors/h3_blocks.py).
The rules that need no model are refused here rather than after a checkpoint
load, with the helpers the run-time check calls (`guide_frame_problem`,
`snap_guide_length`, `guide_end_problem`) so the two cannot drift:

- a step that is not an H3 pipeline, or that loads `ref2va` or also passes
  `references` - ref2va lays out its own conditioning;
- a `guides` that is not a list, more than GUIDE_LIMIT entries, an entry that is
  not `{video, frame}` (no other key but an optional `audio`, both present);
- a `frame` that is not a whole, non-negative multiple of 17, and an `audio` that
  is not true or false;
- with a probe: a video that is not a video, `audio: true` on a video with no audio
  stream, and a clip that, cut to a whole-latent length, runs past the end of a
  render whose `num_frames` is a literal.

A `previous_result:` or other unresolved value, or a video that cannot be located
or probed, names nothing yet and is left to the run-time check. An empty list is
no guides.

A chain's `continuity: "guide"` lays guides in itself (dw/pipeline_processors/
chain.py), so `guide_chain_errors` refuses it where those guides would be: off
an H3 t2va/fl2va step, with a `guide_frames` other than 22 or 39, or with a
`carry_frames` it would not read (`guide_chain_problems`, shared with the run).
"""

from . import references
from .adapter_compatibility import (
    FROM_PRETRAINED_KEY,
    REFERENCE_WORKFLOWS,
    WORKFLOW_KEY,
)
from .for_each import MEMBER_SEPARATOR, render_path
from .hold_audio import MODULAR_PIPELINE, _not_h3
from .pipeline_processors.h3_blocks import (
    GUIDE_CHAIN_WORKFLOWS,
    GUIDE_CONTINUITY,
    GUIDE_LIMIT,
    GUIDES_INPUT,
    RENDER_GRID,
    default_num_frames,
    guide_chain_problems,
    guide_end_problem,
    guide_frame_problem,
    snap_guide_length,
)
from .probe_paths import resolve_probe_path
from .variable_constraints import aligned

GUIDE_KEYS = ("video", "frame", "audio")
CONTINUITY_MODES = ("last_frame", "last_segment", GUIDE_CONTINUITY)


def _render_length(arguments):
    """The frames the step renders when `num_frames` is static, else None:
    `num_frames` (or H3's own default, `default_num_frames`) rounded up onto the
    17n + 5 grid through `variable_constraints.aligned`, as diffusers'
    `align_num_frames` rounds it at run time."""
    value = arguments.get("num_frames")
    if value is None:
        value = default_num_frames()
        if value is None:
            return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return aligned(value, RENDER_GRID)


def _clip_frames(video, base_dir, probe, with_audio=False):
    """(problem, frame_count) for one guide video: the reason it is no video, or
    has no audio stream for `with_audio`, and its frame count when known; (None,
    None) when it cannot be told."""
    if probe is None:
        return None, None
    path = resolve_probe_path(video, base_dir, "a guide video")
    if path is None:
        return None, None
    info = probe(path)
    if not isinstance(info, dict):
        return None, None
    kind = info.get("kind")
    if kind != "video":
        return f"'video' must be a video, and this is {kind or 'unreadable'}", None
    if with_audio and not info.get("sample_rate"):
        return (
            "'audio' is true, but this guide's video has no audio stream - pass a "
            "video with a soundtrack, or drop 'audio'"
        ), None
    count = info.get("frame_count")
    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
        return None, count
    return None, None


def _step_problems(pipeline, arguments, base_dir, probe):
    """[(index or None, message)] for one step's `guides`."""
    guides = arguments[GUIDES_INPUT]
    if isinstance(guides, (list, tuple)) and not guides:
        return []
    not_h3 = _not_h3(pipeline, GUIDES_INPUT)
    if not_h3:
        return [(None, not_h3)]
    from_pretrained = pipeline.get(FROM_PRETRAINED_KEY)
    workflow = (
        from_pretrained.get(WORKFLOW_KEY) if isinstance(from_pretrained, dict) else None
    )
    if workflow in REFERENCE_WORKFLOWS or arguments.get("references") is not None:
        return [
            (
                None,
                "guides cannot be combined with ref2va / references - use t2va "
                "or fl2va",
            )
        ]
    if not isinstance(guides, (list, tuple)):
        return [
            (
                None,
                f"guides must be a list of {{video, frame}}, got "
                f"{type(guides).__name__}",
            )
        ]
    problems = []
    if len(guides) > GUIDE_LIMIT:
        problems.append(
            (None, f"guides takes at most {GUIDE_LIMIT} clips, got {len(guides)}")
        )
    render = _render_length(arguments)
    for index, guide in enumerate(guides):
        if not isinstance(guide, dict):
            problems.append(
                (index, f"a guide must be {{video, frame}}, got {type(guide).__name__}")
            )
            continue
        unknown = sorted(set(guide) - set(GUIDE_KEYS))
        if unknown:
            problems.append(
                (
                    index,
                    f"unknown key(s) {unknown} - a guide is {{video, frame}}, with "
                    f"an optional 'audio'",
                )
            )
        if "video" not in guide or "frame" not in guide:
            problems.append((index, "a guide needs both 'video' and 'frame'"))
            continue
        frame = guide["frame"]
        frame_known = not (
            isinstance(frame, str) and references.is_ref(references.UNRESOLVED, frame)
        )
        frame_ok = False
        if frame_known:
            problem = guide_frame_problem(frame)
            if problem:
                problems.append((index, problem))
            else:
                frame_ok = True
        with_audio = guide.get("audio", False)
        if isinstance(with_audio, str) and references.is_ref(
            references.UNRESOLVED, with_audio
        ):
            with_audio = False
        elif not isinstance(with_audio, bool):
            problems.append(
                (
                    index,
                    f"'audio' must be true or false, got {type(with_audio).__name__}",
                )
            )
            with_audio = False
        problem, count = _clip_frames(guide["video"], base_dir, probe, with_audio)
        if problem:
            problems.append((index, problem))
        elif frame_ok and count is not None and render is not None:
            problem = guide_end_problem(frame, snap_guide_length(count), render)
            if problem:
                problems.append((index, problem))
    return problems


def guides_errors(workflow_definition, source_indices=None, base_dir=None, probe=None):
    """Every `guides` argument refused before the run, as [{path, message}].

    Walks the substituted, expanded definition the way `hold_audio_errors`
    does. `probe` is the metadata-only probe (or the validation's memoizing
    wrapper); without one the clips themselves are not looked at.
    """
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    errors = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        pipeline = step.get("pipeline")
        if not isinstance(pipeline, dict):
            continue
        arguments = pipeline.get("arguments")
        if not isinstance(arguments, dict) or arguments.get(GUIDES_INPUT) is None:
            continue
        source = references.author_index(source_indices, index)
        name = step.get("name")
        where = (
            f" in member '{name}'"
            if isinstance(name, str) and MEMBER_SEPARATOR in name
            else ""
        )
        base = ("steps", source, "pipeline", "arguments", GUIDES_INPUT)
        for entry, message in _step_problems(pipeline, arguments, base_dir, probe):
            path = base if entry is None else base + (entry,)
            errors.append({"path": render_path(path), "message": f"{message}{where}"})
    return errors


def _guide_chain_step_problem(pipeline):
    """Why this step cannot run a guide chain, or None - it is not MiniMax-H3,
    loads ref2va, or passes `references`. A step whose pipeline or workflow is
    not literal is left to the run."""
    rule = (
        "continuity 'guide' runs on MiniMax-H3 t2va or fl2va only - guides stay "
        "off ref2va"
    )
    configuration = pipeline.get("configuration")
    component_type = (
        configuration.get("component_type") if isinstance(configuration, dict) else None
    )
    if (
        isinstance(component_type, str)
        and not references.is_ref(references.UNRESOLVED, component_type)
        and component_type.rsplit(".", 1)[-1] != MODULAR_PIPELINE
    ):
        return f"{rule}, and this step loads {component_type}"
    from_pretrained = pipeline.get(FROM_PRETRAINED_KEY)
    workflow = (
        from_pretrained.get(WORKFLOW_KEY) if isinstance(from_pretrained, dict) else None
    )
    if (
        isinstance(workflow, str)
        and not references.is_ref(references.UNRESOLVED, workflow)
        and workflow not in GUIDE_CHAIN_WORKFLOWS
    ):
        return f"{rule}, and this step loads the '{workflow}' workflow"
    arguments = pipeline.get("arguments")
    if isinstance(arguments, dict) and arguments.get("references") is not None:
        return f"{rule}, and this step passes references"
    return None


def guide_chain_errors(workflow_definition, source_indices=None):
    """Every chain block refused for its continuity before the run, as
    [{path, message}]: an unknown mode (the schema lets a `variable:` through),
    and a `guide` chain its step or its own settings cannot run."""
    steps = workflow_definition.get("steps")
    if not isinstance(steps, list):
        return []

    errors = []
    for index, step in enumerate(steps):
        pipeline = step.get("pipeline") if isinstance(step, dict) else None
        chain = pipeline.get("chain") if isinstance(pipeline, dict) else None
        if not isinstance(chain, dict):
            continue
        continuity = chain.get("continuity", "last_frame")
        if isinstance(continuity, str) and references.is_ref(
            references.UNRESOLVED, continuity
        ):
            continue
        base = ("steps", references.author_index(source_indices, index), "pipeline")
        if continuity not in CONTINUITY_MODES:
            errors.append(
                {
                    "path": render_path(base + ("chain", "continuity")),
                    "message": (
                        f"unknown chain continuity {continuity!r} - expected one "
                        f"of {', '.join(CONTINUITY_MODES)}"
                    ),
                }
            )
            continue
        if continuity != GUIDE_CONTINUITY:
            continue
        problem = _guide_chain_step_problem(pipeline)
        if problem:
            errors.append(
                {
                    "path": render_path(base + ("chain", "continuity")),
                    "message": problem,
                }
            )
        for key, message in guide_chain_problems(chain):
            errors.append(
                {"path": render_path(base + ("chain", key)), "message": message}
            )
    return errors


__all__ = ["guide_chain_errors", "guides_errors"]
