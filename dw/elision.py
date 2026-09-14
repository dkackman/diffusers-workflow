"""A step nothing references, and which saves nothing, does not run.

`templates/minimax/dialogue-short` draws its two characters with Z-Image and
references those portraits from every shot. An episode can just as well be
cast from portraits that already exist - a shot entry's subject reference
takes `from_file: "asset:cast/priya.jpg"` exactly as its voice references do -
and then the two draw steps still ran and their output was discarded: roughly
55 seconds and two model loads on portraits nothing in the run looked at
(#109, #122). The recurring cast is the headline use of that template, so
paying for it every episode was the wrong default, and no argument the caller
could pass avoided it.

So: before the first step executes, drop any step whose result no later step
reads and which writes no file. It is static - it depends only on what the
realized workflow references, never on a value produced during the run, which
is what keeps it a different thing from conditional execution (#118 closed the
step object specifically so an invented `when` is a hard error).

Four guardrails, all of them load-bearing:

- **A step that saves is kept.** `save` defaults to true, so a step declaring
  a `result` is a deliverable unless it says otherwise - a workflow whose
  whole point is writing three images references nothing.
- **The last step is kept**, whatever it declares: it is the run's answer.
- **A release moves rather than disappearing.** `release_pipeline` frees the
  memory the step it sits on took; eliding it silently would leak that for
  the rest of the run. It moves onto the last surviving step before it when
  that step loaded the same pipeline, and is dropped when nothing was loaded
  to release.
- **Every elision is a warning.** Without one a misspelled reference makes
  the step feeding it vanish and the failure moves from "previous result not
  found" to "the picture is wrong". The static reference check refuses an
  unresolvable literal reference, which contains most of it - the warning is
  what covers the rest.

Elision is transitive: dropping a step can leave the step it read
unreferenced in turn, so it runs to a fixed point.
"""

import logging

from .step_cache import reference_resolves_to, referenced_result_names

logger = logging.getLogger("dw")


def _saves(step):
    """Whether the step writes a file.

    Exactly what `Result.save` asks: a `content_type` to write, and `save`
    not turned off. `save` defaults to true, so declaring a `result` with a
    content type is declaring a deliverable - and a step with no `result` at
    all writes nothing, whatever it generates, which is what makes the
    portrait steps droppable in the first place.
    """
    result = step.get("result")
    if not isinstance(result, dict):
        return False
    if result.get("content_type") is None:
        return False
    return result.get("save", True) is not False


def _reused_component_names(steps):
    """Every component name a later step asks an earlier one to have shared.

    Component sharing is keyed on the component's name rather than on the
    step's, so the step that shares is referenced without ever being named.
    """
    names = set()
    for step in steps:
        pipeline = step.get("pipeline")
        if isinstance(pipeline, dict):
            for name in pipeline.get("reused_components") or []:
                names.add(name)
    return names


def _referenced_pipeline_steps(steps):
    """Every step name a later `pipeline_reference` addresses."""
    names = set()
    for step in steps:
        reference = step.get("pipeline_reference")
        if isinstance(reference, dict):
            name = reference.get("reference_name")
            if isinstance(name, str):
                names.add(name)
    return names


def _needed_by(step, later):
    """Whether any of `later` reads this step - by result, by pipeline, or by
    a component it shares."""
    name = step.get("name")
    if not isinstance(name, str):
        return True

    if any(reference_resolves_to(ref, name) for ref in referenced_result_names(later)):
        return True
    if name in _referenced_pipeline_steps(later):
        return True

    pipeline = step.get("pipeline")
    shared = set((pipeline or {}).get("shared_components") or [])
    return bool(shared & _reused_component_names(later))


def _carry_release(elided, kept):
    """Move an elided step's release onto the last surviving step before it.

    `release_pipeline` names the pipeline the elided step itself loaded, so
    it is only meaningful on an earlier step that loaded the same one -
    moving it anywhere else would unload something the workflow did not ask
    to unload. `release_models` frees the process-wide task-model cache, so
    the last step that ran before this one is exactly where it belongs. With
    nothing before it, nothing was loaded and the flag is dropped.
    """
    from .workflow import pipeline_cache_key

    if not kept:
        return False
    predecessor = kept[-1]
    carried = False
    if elided.get("release_models"):
        predecessor["release_models"] = True
        carried = True
    if not elided.get("release_pipeline"):
        return carried
    elided_pipeline = elided.get("pipeline")
    kept_pipeline = predecessor.get("pipeline")
    if not elided_pipeline or not kept_pipeline:
        return carried
    if pipeline_cache_key(elided_pipeline) == pipeline_cache_key(kept_pipeline):
        predecessor["release_pipeline"] = True
        carried = True
    return carried


def elide_unreferenced_steps(steps):
    """The steps that will run, and what was dropped.

    Returns (kept, elided) where `elided` is [{'step': name, 'reason': str}],
    in the order the steps were written. `steps` is the expanded, substituted
    list - a `for_each` member is a step like any other by then, and `gather:`
    has already become the `previous_result:` list it stands for.

    The list is not copied: a kept step is the same object that went in, and
    a release carried onto a survivor is written into it.
    """
    if not isinstance(steps, list) or len(steps) < 2:
        return steps, []

    kept = list(steps)
    elided = []
    changed = True
    while changed:
        changed = False
        for index, step in enumerate(kept):
            if index == len(kept) - 1:
                # The run's answer, whatever it declares
                continue
            if not isinstance(step, dict) or _saves(step):
                continue
            if _needed_by(step, kept[index + 1 :]):
                continue
            reason = "nothing after it reads its result and it saves no file"
            if _carry_release(step, kept[:index]):
                reason += "; its release was carried onto the step before it"
            elided.append({"step": step.get("name"), "reason": reason})
            del kept[index]
            changed = True
            break

    # Written order, not the order the fixed point happened to reach them in
    order = {
        step.get("name"): index
        for index, step in enumerate(steps)
        if isinstance(step, dict)
    }
    elided.sort(key=lambda entry: order.get(entry["step"], 0))
    return kept, elided


def elide_definition(workflow_def):
    """elide_unreferenced_steps over a whole definition, in place.

    Returns the elided-step records; the definition's `steps` is replaced
    when anything was dropped, so a caller that wants the untouched list must
    keep its own copy.
    """
    steps = workflow_def.get("steps")
    kept, elided = elide_unreferenced_steps(steps)
    if elided:
        workflow_def["steps"] = kept
    return elided


def warn_elided(elided):
    """Say what did not run, where whoever asked for the run can read it.

    A warning rather than a log line for the reason every run-time warning is
    one: a consumer over the API or MCP sees the job's `warnings` list and
    nothing else (#82), and "the portrait step was skipped" is precisely the
    thing that explains an otherwise inexplicable result.
    """
    from .events import emit_warning

    for entry in elided:
        emit_warning(
            f"Step '{entry['step']}' did not run: {entry['reason']}. If it "
            f"was meant to, either a later step's reference to it is "
            f"misspelled or it needs a 'result' to save.",
            kind="step_elided",
            step=entry["step"],
        )
