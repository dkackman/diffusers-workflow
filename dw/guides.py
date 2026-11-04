"""A `guides` argument, checked before the run when it can be.

`guides` (`[{"video": ..., "frame": ..., "audio": true?}]`) lays clips of an existing video into a
MiniMax-H3 generation at chosen frames (dw/pipeline_processors/h3_guides.py).
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

The latent statistics anchor (#1166) is refused here too, by `anchor_errors`:
a `latent_anchor` argument, or a chain's `anchor: "segment_1"`, on a step the
guides step rule refuses (`_takes_guides_problem`: off H3, ref2va, or passing
`references`), and a chain `anchor` that is unknown, paired with a continuity
other than guide or last_frame, or carried into `references`
(`chain_anchor_problem`, shared with the run).

Fixed guide noise (#1166) is refused by `guide_noise_errors`: a `guide_noise`
that is not a number in [0, 1) (`guide_noise_problem`, shared with the run), or
one on a step the guides step rule refuses. `guide_noise_warnings` says when it
would do nothing: a step with no `guides` and no guide chain.
"""

from . import references
from .adapter_compatibility import (
    FROM_PRETRAINED_KEY,
    KEYFRAME_WORKFLOWS,
    REFERENCE_WORKFLOWS,
    WORKFLOW_KEY,
)
from .for_each import render_path
from .pipeline_processors.h3_guides import default_num_frames
from .pipeline_processors.h3_rules import (
    ANCHOR_NONE,
    CHAIN_CONTINUITY_MODES,
    GUIDE_CHAIN_DEFAULT,
    GUIDE_CHAIN_FRAMES,
    GUIDES_INPUT,
    GUIDE_CHAIN_RULE,
    GUIDE_CONTINUITY,
    GUIDE_LIMIT,
    GUIDE_NOISE_INPUT,
    LATENT_ANCHOR_INPUT,
    LATENT_ANCHOR_STRENGTH_INPUT,
    anchor_strength_problem,
    RENDER_GRID,
    chain_anchor_problem,
    chain_keystone_problem,
    guide_backoff_problems,
    guide_chain_problems,
    guide_end_problem,
    guide_frame_problem,
    guide_noise_problem,
    not_h3,
    snap_guide_length,
)
from .probe_paths import resolve_probe_path
from .variable_constraints import aligned

GUIDE_KEYS = ("video", "frame", "audio")


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


def _takes_guides_problem(pipeline, arguments, argument=GUIDES_INPUT):
    """Why this step cannot take `argument`, or None: it is not MiniMax-H3
    (`h3_rules.not_h3`), or it loads ref2va or passes `references` (a call
    with references routes to ref2va). The one step rule for a `guides`
    argument, a guide chain, a `latent_anchor` argument and a chain's
    `anchor: "segment_1"` - every H3 t2va/fl2va-only input. A pipeline or
    workflow that is not literal is left to the run."""
    # Named as taken by t2va/fl2va only: ref2va refuses it below
    h3_problem = not_h3(pipeline, argument, taken_by=KEYFRAME_WORKFLOWS)
    if h3_problem:
        return h3_problem
    from_pretrained = pipeline.get(FROM_PRETRAINED_KEY)
    workflow = (
        from_pretrained.get(WORKFLOW_KEY) if isinstance(from_pretrained, dict) else None
    )
    passes_references = (
        isinstance(arguments, dict) and arguments.get("references") is not None
    )
    if workflow in REFERENCE_WORKFLOWS or passes_references:
        return f"{argument} cannot be combined with ref2va / references - use t2va or fl2va"
    return None


def _guide_problems(guide, render, base_dir, probe):
    """[message] for one entry of a step's `guides`, in the order
    `_step_problems` reports them; `render` is the step's literal render
    length, or None."""
    if not isinstance(guide, dict):
        return [f"a guide must be {{video, frame}}, got {type(guide).__name__}"]
    problems = []
    unknown = sorted(set(guide) - set(GUIDE_KEYS))
    if unknown:
        problems.append(
            f"unknown key(s) {unknown} - a guide is {{video, frame}}, with "
            f"an optional 'audio'"
        )
    if "video" not in guide or "frame" not in guide:
        problems.append("a guide needs both 'video' and 'frame'")
        return problems
    frame = guide["frame"]
    frame_known = not (
        isinstance(frame, str) and references.is_ref(references.UNRESOLVED, frame)
    )
    frame_ok = False
    if frame_known:
        problem = guide_frame_problem(frame)
        if problem:
            problems.append(problem)
        else:
            frame_ok = True
    with_audio = guide.get("audio", False)
    if isinstance(with_audio, str) and references.is_ref(
        references.UNRESOLVED, with_audio
    ):
        with_audio = False
    elif not isinstance(with_audio, bool):
        problems.append(
            f"'audio' must be true or false, got {type(with_audio).__name__}"
        )
        with_audio = False
    problem, count = _clip_frames(guide["video"], base_dir, probe, with_audio)
    if problem:
        problems.append(problem)
    elif frame_ok and count is not None and render is not None:
        problem = guide_end_problem(frame, snap_guide_length(count), render)
        if problem:
            problems.append(problem)
    return problems


def _step_problems(pipeline, arguments, base_dir, probe):
    """[(index or None, message)] for one step's `guides`."""
    guides = arguments[GUIDES_INPUT]
    if isinstance(guides, (list, tuple)) and not guides:
        return []
    problem = _takes_guides_problem(pipeline, arguments)
    if problem:
        return [(None, problem)]
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
        problems.extend(
            (index, message)
            for message in _guide_problems(guide, render, base_dir, probe)
        )
    return problems


def guides_errors(workflow_definition, source_indices=None, base_dir=None, probe=None):
    """Every `guides` argument refused before the run, as [{path, message}].

    Walks the substituted, expanded definition the way `hold_audio_errors`
    does. `probe` is the metadata-only probe (or the validation's memoizing
    wrapper); without one the clips themselves are not looked at.
    """
    errors = []
    for _, _, pipeline, source, where in references.iter_steps(
        workflow_definition.get("steps"), source_indices, "pipeline"
    ):
        arguments = pipeline.get("arguments")
        if not isinstance(arguments, dict) or arguments.get(GUIDES_INPUT) is None:
            continue
        base = ("steps", source, "pipeline", "arguments", GUIDES_INPUT)
        for entry, message in _step_problems(pipeline, arguments, base_dir, probe):
            path = base if entry is None else base + (entry,)
            errors.append({"path": render_path(path), "message": f"{message}{where}"})
    return errors


def guide_chain_errors(workflow_definition, source_indices=None):
    """Every chain block refused for its continuity before the run, as
    [{path, message}]: an unknown mode (the schema lets a `variable:` through),
    and a `guide` chain its step or its own settings cannot run."""
    errors = []
    for _, _, pipeline, source, _ in references.iter_steps(
        workflow_definition.get("steps"), source_indices, "pipeline"
    ):
        chain = pipeline.get("chain")
        if not isinstance(chain, dict):
            continue
        continuity = chain.get("continuity", "last_frame")
        if isinstance(continuity, str) and references.is_ref(
            references.UNRESOLVED, continuity
        ):
            continue
        base = ("steps", source, "pipeline")
        if continuity not in CHAIN_CONTINUITY_MODES:
            errors.append(
                {
                    "path": render_path(base + ("chain", "continuity")),
                    "message": (
                        f"unknown chain continuity {continuity!r} - expected one "
                        f"of {', '.join(CHAIN_CONTINUITY_MODES)}"
                    ),
                }
            )
            continue
        for key, message in guide_backoff_problems(chain):
            errors.append(
                {
                    "path": render_path(base + ("chain", key)),
                    "message": message,
                }
            )
        if continuity != GUIDE_CONTINUITY:
            continue
        problem = _takes_guides_problem(pipeline, pipeline.get("arguments"))
        if problem:
            errors.append(
                {
                    "path": render_path(base + ("chain", "continuity")),
                    "message": f"{GUIDE_CHAIN_RULE}: {problem}",
                }
            )
        # The chain adds a guide of its own at frame 0 for every segment
        # after the first (chain.py GuideContinuity), and the run's cap
        # counts it - so the step's written guides get one slot fewer. An
        # unresolved value is the run's to count. A one-segment chain has no
        # segment after the first, so it adds none. An own guide at frame 0 is
        # not refused: the layout appends every guide as its own condition rows
        # (h3_guides DwH3GuideLayoutStep), so it sits beside the chain's.
        arguments = pipeline.get("arguments")
        own = arguments.get(GUIDES_INPUT) if isinstance(arguments, dict) else None
        segments = chain.get("segments")
        added = 0 if segments == 1 and not isinstance(segments, bool) else 1
        # With no chain guide, an overflow is the guide count's own error
        # (guides_errors); the chain's reason would not apply.
        if added and isinstance(own, (list, tuple)) and len(own) + added > GUIDE_LIMIT:
            errors.append(
                {
                    "path": render_path(base + ("chain", "continuity")),
                    "message": (
                        f"guides takes at most {GUIDE_LIMIT} "
                        f"clips and the chain adds one of its own, so a guide "
                        f"chain leaves room for {GUIDE_LIMIT - 1} - got {len(own)}"
                    ),
                }
            )
        for key, message in guide_chain_problems(chain):
            errors.append(
                {"path": render_path(base + ("chain", key)), "message": message}
            )
    return errors


def guide_lengths(pipeline, base_dir=None, probe=None):
    """The snapped frame count of every guide a step lays in, for the VRAM
    projection (dw/vram_estimate.py, #694): its own `guides`, an entry whose
    `video` is null dropped as the step's null references are, then the guide a
    `continuity: "guide"` chain appends to every segment after the first, at its
    `guide_frames`. A clip `_clip_frames` cannot count - a `previous_result:`,
    an unreadable header, no probe at all - is None, for the caller to charge
    at the most a guide can hold."""
    if not isinstance(pipeline, dict):
        return []
    arguments = pipeline.get("arguments")
    own = arguments.get(GUIDES_INPUT) if isinstance(arguments, dict) else None
    lengths = []
    for guide in own if isinstance(own, (list, tuple)) else ():
        if not isinstance(guide, dict) or guide.get("video") is None:
            continue
        _, count = _clip_frames(guide["video"], base_dir, probe)
        lengths.append(None if count is None else snap_guide_length(count))
    chain = pipeline.get("chain")
    if isinstance(chain, dict) and chain.get("continuity") == GUIDE_CONTINUITY:
        segments = chain.get("segments")
        if not (segments == 1 and not isinstance(segments, bool)):
            frames = chain.get("guide_frames", GUIDE_CHAIN_DEFAULT)
            lengths.append(frames if frames in GUIDE_CHAIN_FRAMES else None)
    return lengths


def _strength_argument_errors(arguments, base, where):
    """A `latent_anchor_strength` argument out of range, or with no
    `latent_anchor` for it to set (#1190), as [{path, message}]."""
    if not isinstance(arguments, dict):
        return []
    strength = arguments.get(LATENT_ANCHOR_STRENGTH_INPUT)
    if strength is None:
        return []
    path = render_path(base + ("arguments", LATENT_ANCHOR_STRENGTH_INPUT))
    problem = anchor_strength_problem(strength)
    if problem is None and arguments.get(LATENT_ANCHOR_INPUT) is None:
        problem = (
            f"{LATENT_ANCHOR_STRENGTH_INPUT} sets how far {LATENT_ANCHOR_INPUT} "
            f"goes, and the step has no {LATENT_ANCHOR_INPUT}"
        )
    return [{"path": path, "message": f"{problem}{where}"}] if problem else []


def anchor_errors(workflow_definition, source_indices=None):
    """Every latent statistics anchor refused before the run, as
    [{path, message}]: a `latent_anchor` argument on a step that cannot take
    it, and a chain `anchor` that is unknown, paired with a continuity it
    cannot run under, or set on a step that cannot take it."""
    errors = []
    for _, _, pipeline, source, where in references.iter_steps(
        workflow_definition.get("steps"), source_indices, "pipeline"
    ):
        base = ("steps", source, "pipeline")
        arguments = pipeline.get("arguments")
        if (
            isinstance(arguments, dict)
            and arguments.get(LATENT_ANCHOR_INPUT) is not None
        ):
            problem = _takes_guides_problem(pipeline, arguments, LATENT_ANCHOR_INPUT)
            if problem:
                errors.append(
                    {
                        "path": render_path(base + ("arguments", LATENT_ANCHOR_INPUT)),
                        "message": f"{problem}{where}",
                    }
                )
        errors += _strength_argument_errors(arguments, base, where)
        chain = pipeline.get("chain")
        if not isinstance(chain, dict):
            continue
        strength = chain.get("anchor_strength")
        problem = None if strength is None else anchor_strength_problem(strength)
        if problem:
            errors.append(
                {
                    "path": render_path(base + ("chain", "anchor_strength")),
                    "message": f"{problem}{where}",
                }
            )
        problem = chain_keystone_problem(chain)
        if problem:
            errors.append(
                {
                    "path": render_path(base + ("chain", "keystone")),
                    "message": f"{problem}{where}",
                }
            )
        path = render_path(base + ("chain", "anchor"))
        problem = chain_anchor_problem(chain)
        if problem:
            errors.append({"path": path, "message": f"{problem}{where}"})
            continue
        anchor = chain.get("anchor", ANCHOR_NONE)
        if anchor == ANCHOR_NONE or references.is_ref(references.UNRESOLVED, anchor):
            continue
        problem = _takes_guides_problem(pipeline, arguments, "chain anchor 'segment_1'")
        if problem:
            errors.append({"path": path, "message": f"{problem}{where}"})
    return errors


def _guide_noise_steps(workflow_definition, source_indices):
    """(pipeline, arguments, path, where) for every step passing a non-null
    `guide_noise`."""
    for _, _, pipeline, source, where in references.iter_steps(
        workflow_definition.get("steps"), source_indices, "pipeline"
    ):
        arguments = pipeline.get("arguments")
        if isinstance(arguments, dict) and arguments.get(GUIDE_NOISE_INPUT) is not None:
            path = render_path(
                ("steps", source, "pipeline", "arguments", GUIDE_NOISE_INPUT)
            )
            yield pipeline, arguments, path, where


def guide_noise_errors(workflow_definition, source_indices=None):
    """Every `guide_noise` refused before the run, as [{path, message}]: on a
    step that cannot take guides (off H3, ref2va, references), or not a
    number in [0, 1)."""
    errors = []
    for pipeline, arguments, path, where in _guide_noise_steps(
        workflow_definition, source_indices
    ):
        problem = _takes_guides_problem(
            pipeline, arguments, GUIDE_NOISE_INPUT
        ) or guide_noise_problem(arguments[GUIDE_NOISE_INPUT])
        if problem:
            errors.append({"path": path, "message": f"{problem}{where}"})
    return errors


def guide_noise_warnings(workflow_definition, source_indices=None):
    """["path: message"] for a nonzero `guide_noise` on a step with no
    `guides` and no guide chain: it noises guides, and there are none."""
    warnings = []
    for pipeline, arguments, path, where in _guide_noise_steps(
        workflow_definition, source_indices
    ):
        if arguments[GUIDE_NOISE_INPUT] == 0:
            continue
        guides = arguments.get(GUIDES_INPUT)
        chain = pipeline.get("chain")
        continuity = chain.get("continuity") if isinstance(chain, dict) else None
        if guides or continuity == GUIDE_CONTINUITY:
            continue
        if isinstance(continuity, str) and references.is_ref(
            references.UNRESOLVED, continuity
        ):
            continue
        warnings.append(
            f"{path}: guide_noise does nothing here - it noises the step's "
            f"guides, and this step has no 'guides' and no chain with "
            f"continuity 'guide'{where}"
        )
    return warnings


__all__ = [
    "anchor_errors",
    "guide_chain_errors",
    "guide_lengths",
    "guide_noise_errors",
    "guide_noise_warnings",
    "guides_errors",
]
