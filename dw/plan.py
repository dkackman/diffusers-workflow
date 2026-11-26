"""The plan a validate call answers with: what a run of a workflow with a
caller's arguments will actually execute, what it will have to download
first, and what the workflow's own cost block says it will take - with a
fingerprint over the work, so an acknowledgement can be bound to it and a
run whose shape changed after consent refused (#85, stage 2).

Everything here is derived from the same resolvers the run uses -
`Workflow.expanded_definition` folds the arguments, substitutes and expands
`for_each` through the stages the run prepares with, and `realize_workflow`
records those folded variables and inlines the prompts - so the plan
describes the run and not an approximation of it. Nothing here knows a
model: every minute comes from a `cost` block and every repo name from a
`from_pretrained_arguments`.

What a `cost` block itself prices the run at, own and per composed child, is
`dw/plan_pricing.py`; `estimate` here sets this box's history against it.

A step's denoising passes are counted by `dw/step_passes.py`: a pipeline
whose config fixes a `denoising_step_list` runs `len - 1` of them, and a
`denoising_step_list` cost driver scales the figure by that count
(`cost_scaling._driver_scaling`, #1316).
"""

import copy
import hashlib
import json
import logging
import math
import os

from .cost_scaling import (
    CATALOG,
    DERIVED,
    INHERITED,
    OBSERVED,
    OTHER_DEVICE,
    UNKNOWN,
    _driver_scaling,
    fixed_share,
    _entry_cost_field_shifted,
    words_check_term,
    observed_render_scaling,
    _chain_blocks,
    _chain_retries,
)
from .elision import elide_definition
from . import references
from .plan_downloads import _sub_workflow_paths, downloads_required
from .plan_pricing import (
    ENTRY_FRAMES_KEY,
    list_entry_frames,
    _child_catalog_price,
    _child_observed,
    _ChildTotals,
    _own_price,
    _read_child,
    _unresolved_list_note,
    task_rate_price,
    undeclared_task_commands,
    list_entries,
)
from .presets import effective_variables
from .presets import selectors as preset_selectors

# Re-exported: dw/server/routes/jobs.py and the tests read the plan's gate
# warnings from here, beside build_plan
from .plan_downloads import gate_warnings as gate_warnings
from .plan_downloads import missing_repo_errors as missing_repo_errors
from .realize import read_sub_workflow, realize_workflow
from .step_cache import _is_seeded
from .variables import same_number

logger = logging.getLogger("dw")

FINGERPRINT_PREFIX = "sha256:"
# Top-level keys that document, classify or price a workflow rather than
# shape its output - the fingerprint is keyed only by what can change the
# generated output, so editing one of these never busts an acknowledgement
# (#905). Pinned by tests/test_plan.py::TestFingerprintKeys
DOCUMENTATION_KEYS = (
    "configures",
    "cost",
    "cost_drivers",
    "description",
    "host_ram_gb",
    "shape",
    "summary",
    "traits",
    "vram_estimate",
)
# A variable constraint's keys that explain the rule rather than state it
CONSTRAINT_DOCUMENTATION_KEYS = ("reason",)
# A step result's keys that name, place or label its files without changing
# their bytes. The step cache keeps file_base_name and subfolder: a hit
# republishes the cached files verbatim and writes none (dw/step_cache.py)
PLACEMENT_RESULT_KEYS = ("file_base_name", "role", "subfolder")


def build_plan(
    candidate,
    arguments,
    *,
    device,
    prompt_dir=None,
    cache_dir=None,
    lookup_sizes=True,
    cache_probe=None,
    observed=None,
    observed_for_child=None,
):
    """What a run of `candidate` with `arguments` will execute and cost.

    Args:
        candidate: The Workflow the route built - it carries the file spec
            (so base_dir), the output root and the confinement a run has.
        arguments: The caller's arguments, already past `argument_errors`;
            like validation, bad ones would be left unfolded (the defaults
            planned) rather than raised here.
        device: The backend that is serving - 'cuda', 'mps' or 'cpu'.
        prompt_dir: The prompt library, for inlining.
        cache_dir: The hub cache to check downloads against; None for the
            default.
        lookup_sizes: Whether to ask the hub how large a missing repo is.
        cache_probe: A callable taking the run's arguments and answering the
            step names the worker's cache would serve, or None when it
            cannot say; without one `cached_steps` is None.
        observed: This box's own history for this workflow, as an `observed`
            block (dw/server/observed_cost.py) or a callable taking the
            run's arguments and answering one - what the estimate quotes in
            preference to a curated figure (#154). None on a caller that has
            no history to offer, which is every caller but the server.
        observed_for_child: A callable taking a composed child's local path,
            its parsed definition, and the composing step's own `arguments`,
            answering that child's own `observed` block or None - so a
            composing workflow's estimate can roll up a child's history
            instead of resetting to `unknown` when the parent has no figure
            of its own (#268), bucketed against the value the composing step
            actually passes rather than always the child's stored defaults
            (#341). None on a caller that cannot resolve a child's catalog
            name to look history up by.
    """
    definition = candidate.workflow_definition
    base_dir = (
        os.path.dirname(os.path.abspath(candidate.file_spec))
        if candidate.file_spec
        else None
    )
    # The expansion validation took, which folds and expands exactly as
    # the run does (Workflow._fold, Workflow._expand) - and resolves each
    # 'preference:' step, so it comes before the realization that digests
    # what they resolved to
    expanded = candidate.expanded_definition(arguments)
    realized, annotations = realize_workflow(
        definition,
        candidate.folded_variables(arguments),
        seed=0,
        base_dir=base_dir,
        prompt_dir=prompt_dir,
        output_root=candidate.output_dir,
        workflow_dir=candidate.workflow_dir,
        pin_outputs=False,
        preference_targets=candidate.preference_targets(),
    )
    # The plan is what the run does, and a run does not execute a step
    # nothing reads (dw/elision.py, #122) - so the step count, the downloads
    # and the fingerprint are all taken after elision, and the acknowledged
    # cost is the cost of the work that happens
    elided = elide_definition(expanded, definition)
    entries = list_entries(definition, realized)
    # The defaults a curated cost was measured at - the default preset
    # bundle among them
    measured_entries = list_entries(
        definition, {"variables": effective_variables(definition)}
    )
    step_count = len(expanded.get("steps") or [])
    cache_hits = cached_steps(definition, realized, arguments, cache_probe)
    # The work includes the text a stored prompt holds now, not only its
    # name - so the fingerprint is taken over the expansion with each
    # 'prompt:' inlined, as realization inlines it into the record
    work, _ = realize_workflow(
        expanded,
        None,
        seed=0,
        base_dir=base_dir,
        prompt_dir=prompt_dir,
        output_root=candidate.output_dir,
        workflow_dir=candidate.workflow_dir,
        pin_outputs=False,
    )
    return {
        "fingerprint": fingerprint(
            work,
            definition,
            {
                **annotations,
                "sub_workflows": {
                    path: child_work_digest(path, base_dir, candidate.workflow_dir)
                    for path in annotations.get("sub_workflows") or {}
                },
            },
        ),
        "steps": step_count,
        "elided_steps": elided,
        # One line per 'preference:' step: what it resolved to and what it
        # skipped (dw/preferences.py)
        **({"preferences": lines} if (lines := candidate.preference_lines()) else {}),
        "list_entries": entries,
        "cached_steps": cache_hits,
        "downloads_required": downloads_required(
            expanded, base_dir, candidate.workflow_dir, cache_dir, lookup_sizes
        ),
        "estimate": _hybrid_unpriced(
            expanded,
            _checked_estimate(
                candidate,
                arguments,
                definition,
                expanded,
                entries,
                device,
                base_dir,
                measured_entries=measured_entries,
                observed=observed,
                cached_steps=cache_hits,
                total_steps=step_count,
                observed_for_child=observed_for_child,
            ),
        ),
    }


def _hybrid_unpriced(expanded, estimate):
    """`estimate` with its minutes withdrawn when a step generates under
    `experimental_hybrid_windows` (#1188, EXPERIMENTAL): its time is windows
    times steps of forwards plus host transfers, which no template's history
    or scaling measures, so any figure would be a guess. A real cost model
    waits on its promotion."""
    from .pipeline_processors.h3_rules import HYBRID_WINDOWS_INPUT

    hybrid = any(
        isinstance(step.get("arguments"), dict)
        and step["arguments"].get(HYBRID_WINDOWS_INPUT) is not None
        for _, _, step, _, _ in references.iter_steps(
            expanded.get("steps"), key="pipeline"
        )
        if isinstance(step, dict)
    )
    if not hybrid or not isinstance(estimate, dict):
        return estimate
    unpriced = {
        key: None for key in estimate if key == "minutes" or key.endswith("_minutes")
    }
    return {
        **estimate,
        **unpriced,
        "basis": UNKNOWN,
        "unpriced_reason": (
            f"{HYBRID_WINDOWS_INPUT} is experimental: its time is one forward "
            f"per window per step plus the host-RAM transfers, which no "
            f"template's history or scaling measures"
        ),
    }


def _checked_estimate(
    candidate,
    arguments,
    definition,
    expanded,
    entries,
    device,
    base_dir,
    *,
    measured_entries,
    observed,
    cached_steps,
    total_steps,
    observed_for_child,
):
    """`estimate` for the run, floored, for a checked chain, at what the
    same chain unchecked quotes plus what checking adds (#1334).

    A checked chain's history and the same chain's unchecked history are
    separate buckets (`check` and `retries` are cost drivers). A thin checked
    bucket from a fast box tempers below the curated figure, while the
    unchecked shape with no history quotes the curated one at full
    authority, so a checked chain could quote no more than the same chain
    unchecked. The floor is the unchecked twin's own quote - its own
    expansion and its own history, exactly what validating it returns -
    plus the worst-case extra the check carries (the re-rolls and the words
    passes: the curated checked figure less the curated unchecked one).
    That extra is positive whenever the chain is checked, so a checked chain
    always quotes strictly more than its unchecked twin. The higher of the
    checked quote and the floor is quoted, with `scaled` still naming the
    checked worst case and saying the floor was taken."""

    def quote(shape, block):
        return estimate(
            definition,
            shape,
            entries,
            device,
            base_dir,
            candidate.workflow_dir,
            measured_entries=measured_entries,
            observed=block,
            cached_steps=cached_steps,
            total_steps=total_steps,
            observed_for_child=observed_for_child,
        )

    result = quote(
        expanded,
        _observed_block(observed, arguments, definition, expanded, entries, device),
    )
    if result.get("minutes") is None or not _chain_is_checked(definition, expanded):
        return result
    unchecked = _unchecked_arguments(definition, arguments)
    if unchecked is None:
        return result
    try:
        twin_expanded = candidate.expanded_definition(unchecked)
        if _chain_is_checked(definition, twin_expanded):
            return result
        # A fixed block is this run's own history, so the twin has none
        twin_block = (
            _observed_block(
                observed, unchecked, definition, twin_expanded, entries, device
            )
            if callable(observed)
            else None
        )
        twin = quote(twin_expanded, twin_block)
        extra = (quote(expanded, None).get("minutes") or 0) - (
            quote(twin_expanded, None).get("minutes") or 0
        )
    except Exception:
        logger.warning("No unchecked floor for the checked chain", exc_info=True)
        return result
    if twin.get("minutes") is None or extra <= 0:
        return result
    floor = round(twin["minutes"] + extra, 1)
    if floor <= result["minutes"]:
        return result
    note = (
        f"floored at the same chain unchecked ({_shown_minutes(twin['minutes'])} "
        f"min) plus its worst case ({_shown_minutes(round(extra, 1))} min); this "
        f"box's checked history alone quoted {_shown_minutes(result['minutes'])} min"
    )
    return {
        **result,
        "minutes": floor,
        "cached_minutes": _cached_minutes(floor, cached_steps, total_steps),
        "scaled": "; ".join(part for part in (result.get("scaled"), note) if part),
    }


def _chain_is_checked(definition, expanded):
    """Whether the workflow's one chain may re-roll or words-check at
    `expanded`'s variables - what the worst case adds to its quote."""
    variables = (expanded or {}).get("variables") or {}
    words_minutes, _ = words_check_term(definition, variables)
    return _chain_retries(definition, variables) > 0 or bool(words_minutes)


def _unchecked_arguments(definition, arguments):
    """`arguments` with the one chain's `check` off and its `retries` 0, or
    None when the chain's `check` is not a variable an argument can turn
    off (a literal `check` has no unchecked twin to price)."""
    blocks = list(_chain_blocks(definition.get("steps")))
    if len(blocks) != 1:
        return None
    chain = blocks[0][0]
    check = references.ref_name(references.VARIABLE, chain.get("check"))
    if check is None:
        return None
    twin = {**(arguments or {}), check: False}
    retries = references.ref_name(references.VARIABLE, chain.get("retries"))
    if retries is not None:
        twin[retries] = 0
    return twin


def _observed_block(
    observed, arguments, definition=None, expanded=None, entries=None, device=None
):
    """The `observed` block for this run, from a block or from a callable
    the caller passed - best effort, since a figure is a nicety and a plan
    that raised because history could not be read would cost the caller the
    whole pre-flight.

    A run whose drivers this box has never run (a preset tier, say) has no
    bucket of its own. Rather than fall to the curated row, which can sit far
    from this box's speed (#1236), the history of the same run under the
    default preset is scaled by the ratio the drivers moved by. A tier whose
    own bucket is too thin to quote alone carries that scaled figure as
    `baseline`, the one `estimate` tempers it toward in place of the
    curated row."""
    if not callable(observed):
        return observed if isinstance(observed, dict) else None
    try:
        args = arguments or {}
        block = observed(args)
        if definition is not None:
            if not isinstance(block, dict):
                block = _scaled_observed(
                    observed, args, definition, expanded, entries, device
                )
            elif _thin(block):
                baseline = _scaled_observed(
                    observed, args, definition, expanded, entries, device
                )
                if baseline is not None:
                    block = {**block, "baseline": baseline}
    except Exception:
        logger.debug("No observed history for the plan", exc_info=True)
        return None
    return block if isinstance(block, dict) else None


def _thin(block):
    """Whether an observed block's cold history is below the run count
    `_tempered` quotes at full authority."""
    runs = block.get("cold_runs")
    return isinstance(runs, int) and runs < SMALL_N_THRESHOLD


def _scaled_observed(observed, arguments, definition, expanded, entries, device=None):
    """History of the run with the preset selectors left at their defaults,
    scaled to `arguments`' drivers, or None when there is none or nothing
    but the selector itself moved.

    A driver the preset moved that no rule scales (a tier's weights dtype)
    keeps the figure, named in `differs`, as the curated path keeps it
    (#866): the observed default is still far nearer this box's speed than
    the curated row (#1236)."""
    selectors = preset_selectors(definition)
    base_args = {k: v for k, v in arguments.items() if k not in selectors}
    if len(base_args) == len(arguments):
        return None
    block = observed(base_args)
    if not isinstance(block, dict) or not isinstance(
        block.get("cold_minutes"), (int, float)
    ):
        return None
    if block.get("inherited_from"):
        # Another template's history, an approximation of its own (#797)
        return None
    baseline = effective_variables(definition, base_args)
    scaling = _driver_scaling(definition, expanded, entries, baseline=baseline)
    if scaling is None:
        return None
    factor, axes, unscaled = scaling
    # The selector itself moved by construction; what it moved is scaled
    differs = [name for name in unscaled if name not in selectors]
    if not axes and not differs:
        return None
    fixed = block["cold_minutes"] * fixed_share(definition, device)
    scaled = {
        **block,  # the fixed share held, as _scaled holds it (#1224)
        "cold_minutes": fixed + (block["cold_minutes"] - fixed) * factor,
        "scaled": f"scaled from the observed {round(block['cold_minutes'], 1)} min"
        + (f" ({_shown_minutes(fixed)} min of it fixed)" if fixed else "")
        + (": " + ", ".join(axes) if axes else ""),
    }
    if differs:
        scaled["differs"] = differs
    return scaled


def cached_steps(definition, realized, arguments, cache_probe):
    """How many steps the step cache would answer for this run: 0 without
    asking when the workflow is unseeded (the cache is off then), None
    when there is no probe or the probe cannot answer, else the count."""
    if not _is_seeded(definition, arguments):
        return 0
    if cache_probe is None:
        return None
    answer = cache_probe(arguments or {})
    return len(answer) if isinstance(answer, list) else None


def output_only(doc):
    """Remove from a workflow definition, in place, every key that cannot
    change the generated output (#905): the documentation keys, each
    variable constraint's reason, and each step result's placement keys.
    Returns `doc`."""
    for key in DOCUMENTATION_KEYS:
        doc.pop(key, None)
    constraints = doc.get("variable_constraints")
    if isinstance(constraints, dict):
        for rule in constraints.values():
            if isinstance(rule, dict):
                for key in CONSTRAINT_DOCUMENTATION_KEYS:
                    rule.pop(key, None)
    for step in doc.get("steps") or []:
        result = step.get("result") if isinstance(step, dict) else None
        if isinstance(result, dict):
            for key in PLACEMENT_RESULT_KEYS:
                result.pop(key, None)
    return doc


def _canonical_digest(doc):
    # default=repr: a realized 'constant:' can be any Python value, and the
    # fingerprint only needs it to be stable, not round-trippable
    serialized = json.dumps(
        same_number(doc),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=repr,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def child_work_digest(path, base_dir, workflow_dir):
    """The digest of a composed child's work: its file with `output_only`
    applied, so editing a child's cost or documentation leaves the parent's
    fingerprint alone. None when the file cannot be read; a file that is not
    a JSON object is digested as bytes."""
    raw = read_sub_workflow(path, base_dir, workflow_dir)
    if raw is None:
        return None
    try:
        child = json.loads(raw)
    except ValueError:
        child = None
    if not isinstance(child, dict):
        return hashlib.sha256(raw).hexdigest()
    return _canonical_digest(output_only(child))


def fingerprint(expanded, definition, annotations=None):
    """SHA-256 over the expanded definition with everything that is not
    work removed: the seed wherever it sits, and whatever `output_only`
    removes.

    `definition` is the workflow as written, consulted for whether the
    top-level seed named a variable - if it did, that variable's folded
    value is the seed too and is blanked at its source. `annotations` is
    what realization recorded beside the copy; its sub-workflow digests go
    into the hash, since a composed child edited between the quote and the
    call is different work the parent's text cannot show - build_plan
    passes them as `child_work_digest`s.
    """
    doc = copy.deepcopy(expanded)
    if annotations and annotations.get("sub_workflows"):
        doc["__sub_workflows__"] = dict(annotations["sub_workflows"])
    doc.pop("seed", None)
    output_only(doc)
    written_seed = definition.get("seed")
    name = references.ref_name(references.VARIABLE, written_seed)
    if name is not None:
        variables = doc.get("variables")
        if isinstance(variables, dict) and name in variables:
            variables[name] = None
    for step in doc.get("steps") or []:
        if isinstance(step, dict):
            step.pop("seed", None)
            pipeline = step.get("pipeline")
            if isinstance(pipeline, dict):
                pipeline.pop("seed", None)
    return FINGERPRINT_PREFIX + _canonical_digest(doc)


# #301: a single run is not the same statistical basis as a dozen. Below
# this many observed runs, an "observed" figure is tempered rather than
# quoted at full authority - see `_tempered`.
SMALL_N_THRESHOLD = 3


def _tempered(block, curated_minutes):
    """An `observed` estimate below `SMALL_N_THRESHOLD` runs, corrected
    toward the curated figure it might be papering over rather than
    presented as if it carried the same authority as a dozen runs (#301).

    `curated_minutes` is the figure to blend toward: the curated row scaled
    to the run, or this box's default-tier history scaled to it when a
    preset tier's own bucket is the thin one (#1236). With one, the point
    estimate is pulled toward it in proportion to how thin the history is -
    one run
    counts for a third of the blend, two for two thirds, three or more not
    at all. The blend is marked `tempered: true` with `observed_minutes`
    (the raw point figure, the same number `list_workflows`' own
    `observed_minutes` reports) and `curated_minutes` (what it blended
    toward) alongside it, so a caller can reconcile the returned `minutes`
    against either without the two disagreeing silently under the same
    `basis: "observed"` label (#319). With none to blend toward there is
    nothing to correct with, so the number is left alone and a
    `low_confidence` flag is added instead: machine-checkable without
    requiring a caller to know to inspect `runs` itself. No new
    range/uncertainty math (rejected as more surface than the problem
    needs) - just these two, approved shapes.
    """
    runs = block.get("runs")
    if (
        not isinstance(runs, int)
        or runs >= SMALL_N_THRESHOLD
        or block["minutes"] is None
    ):
        return block
    if curated_minutes is None:
        return {**block, "low_confidence": True}
    observed_minutes = block["minutes"]
    weight = runs / SMALL_N_THRESHOLD
    blended = curated_minutes * (1 - weight) + observed_minutes * weight
    return {
        **block,
        "minutes": round(blended, 1),
        "tempered": True,
        "observed_minutes": observed_minutes,
        "curated_minutes": curated_minutes,
    }


def _cached_minutes(minutes, cached_steps, total_steps):
    """The share of `minutes` a caller would actually wait for, once the
    steps already in the step cache are subtracted (#255).

    None when there is nothing to subtract from (`minutes`), no probe
    (`cached_steps` is None) or no step count to take a share of; equal to
    `minutes` itself when the probe found nothing cached, and 0.0 when it
    found the whole run cached.
    """
    if minutes is None or cached_steps is None or not total_steps:
        return None
    remaining = max(0, total_steps - cached_steps)
    return round(minutes * remaining / total_steps, 1)


def _unique(names):
    return list(dict.fromkeys(names))


def _rolled_up_estimate(own, totals, device, cached_steps, total_steps):
    """The final estimate from the parent's own price and its children's."""
    minutes = totals.minutes
    partial = totals.partial
    unpriced = totals.unpriced
    if minutes is None:
        # No number to be partial about - but a composed child that went
        # unpriced is the reason there is none, and an empty list gave the
        # caller nothing to act on (#655). The parent's own id is dropped:
        # it is only ever a contributor to a total that exists.
        partial = False
        unpriced = list(totals.unpriced_children)
    top_basis = own["basis"]
    top_measured_on = own["measured_on"]
    top_runs = None
    if (
        own["minutes"] is None
        and totals.had_child
        and totals.all_observed
        and not partial
    ):
        # Every composing child's share of the total was this box's own
        # history rather than a static figure, and the parent contributed
        # nothing of its own to disagree with that - so the whole total is
        # as good as observed rather than "unknown" (#268), mirroring the
        # existing rule that a child's catalog cost is skipped once the
        # *parent* has an observed figure, to avoid double-counting.
        # The children's own `runs`/`measured_on` come along with the
        # inherited basis (#275) - an "observed" estimate with `runs: null`
        # says it was measured but not how many times, which is the number
        # a caller uses to decide how much to trust the figure. `runs` is
        # the weakest history across children (the min), and `measured_on`
        # is named only when every child agrees on the device.
        top_basis = OBSERVED
        top_runs = min(totals.runs) if totals.runs else None
        top_measured_on = (
            next(iter(totals.measured_on)) if len(totals.measured_on) == 1 else None
        )
    if top_basis == UNKNOWN and minutes is not None and totals.bases:
        # A number must not carry the basis a caller reads as "no number":
        # the parent has no cost of its own, but its priced children do, so
        # the sum takes their basis (#593)
        kinds = {basis for basis, _ in totals.bases}
        if OTHER_DEVICE in kinds:
            top_basis = OTHER_DEVICE
        elif kinds == {CATALOG}:
            top_basis = CATALOG
        else:
            top_basis = DERIVED
        devices = {on for _, on in totals.bases}
        if len(devices) == 1:
            top_measured_on = next(iter(devices))
    if top_basis == CATALOG and any(basis == DERIVED for basis, _ in totals.bases):
        # A child scaled for the arguments its step passes is an estimate,
        # so the sum is one too (#866)
        top_basis = DERIVED
    rounded = round(minutes, 1) if minutes is not None else None
    result = {
        "minutes": rounded,
        "basis": top_basis,
        "device": device,
        "measured_on": top_measured_on,
        "partial": partial,
        "unpriced": unpriced,
        "runs": top_runs,
        "cached_minutes": _cached_minutes(rounded, cached_steps, total_steps),
    }
    scaled = ([own["scaled"]] if own.get("scaled") else []) + totals.scaled
    if scaled and minutes is not None and top_basis != OBSERVED:
        result["scaled"] = "; ".join(scaled)
    differs = _unique((own.get("differs") or []) + totals.differs)
    if differs and minutes is not None and top_basis != OBSERVED:
        # A figure, own or a child's, measured at another setting of a driver
        # nothing scales (#866)
        result["differs"] = differs
        result["low_confidence"] = True
    if top_basis == OBSERVED:
        # The rolled-up-from-children case (#268): no curated figure of the
        # parent's own exists to blend toward (that is this branch's own
        # precondition, above), so a thin roll-up gets the low_confidence
        # flag rather than a blend. Never changes `minutes`, so
        # `cached_minutes` above already reflects it.
        result = _tempered(result, None)
    return result


def estimate(
    definition,
    expanded,
    list_entries,
    device,
    base_dir,
    workflow_dir,
    measured_entries=None,
    observed=None,
    cached_steps=None,
    total_steps=None,
    observed_for_child=None,
):
    """Minutes from this box's own history when it has one, else from the
    workflow's own cost block, re-priced for the caller's list, plus each
    composed child's.

    `basis` names where the figure came from - the honesty is in the field,
    not in a fabricated number: 'unknown' is no cost block at all (or a
    list this figure cannot honestly be re-priced for), 'other_device' a
    figure measured on a backend other than the one serving (reported so
    the agent has something to scale, flagged so it is not quoted as a
    measurement), 'catalog' the stored total for a run whose lists are the
    ones it was measured with, 'per_entry' that total re-priced from a
    measured per-entry rate, and 'derived' re-priced by extrapolating the
    stored total linearly over a list whose length the caller changed
    (#85), or scaled along the steps, canvas or frames moved off the
    defaults it was measured at (#866, `scale_curated`) - an estimate, not
    a measurement, and the only alternative to quoting a 5-shot figure for
    a 10-shot run.

    `measured_entries` is what the stored defaults carry, which is the
    list the catalog figure was measured against; without it a catalog
    figure is taken at face value.

    `observed` is this box's own history for the run being planned
    (dw/server/observed_cost.py), and it wins: `list_workflows` already
    tells a consumer to "prefer [observed] when quoting a price for this
    machine, fall back to `cost`", and `basis: "observed"` says which one
    was used, so the two claims stay distinguishable. It is quoted only
    from the *cold* median - the one figure comparable to a curated `cost`,
    wall clock including the model load - and only for the bucket the
    caller's own arguments fall in, so a resized list falls back to the
    curated figure rather than quoting the default list's minutes (#154).
    A composed child is not added to it: an observed run is the whole run,
    children included, already measured.
    """
    measured = _with_worst_case(
        _observed(observed, device), definition, expanded, device
    )
    own = _own_price(
        definition, expanded, list_entries, device, measured_entries, base_dir
    )
    if own["minutes"] is None and not definition.get("cost"):
        # A task-only workflow with no cost block of its own is priced from
        # its commands' declared rates, probed against its input media - the
        # own slot only, so this box's history above still wins (#1619)
        own = task_rate_price(expanded, base_dir) or own
    baseline = observed.get("baseline") if isinstance(observed, dict) else None
    if measured is not None and _entry_cost_field_shifted(definition, expanded):
        # The history's drivers match on the list's length, not on what each
        # entry renders, so a shot lengthened past anything it covered would
        # be quoted at the shorter run's minutes (#796): rescaled by the
        # frames its runs rendered when the bucket says, else discarded
        # (#1696)
        measured = _with_worst_case(
            _observed(_frames_rescaled(observed, definition, expanded, device), device),
            definition,
            expanded,
            device,
        )
        baseline = _frames_rescaled(baseline, definition, expanded, device)
    if measured is not None:
        # A thin bucket blends toward this box's default-tier history scaled
        # to its drivers when there is one, not the curated row (#1236)
        partner = _with_worst_case(
            _observed(baseline, device),
            definition,
            expanded,
            device,
        )
        if partner is None:
            partner = own
        measured = _tempered(measured, partner["minutes"])
        if measured.get("tempered") and partner.get("scaled"):
            # What it blended toward was scaled for this run's drivers, and
            # `curated_minutes` alone does not say so (#866)
            measured["scaled"] = "; ".join(
                part for part in (measured.get("scaled"), partner["scaled"]) if part
            )
        if measured.get("tempered") and partner.get("differs"):
            # It blended toward a figure measured at another setting of a
            # driver nothing scales (#866)
            measured["differs"] = _unique(
                (measured.get("differs") or []) + partner["differs"]
            )
            measured["low_confidence"] = True
        measured["cached_minutes"] = _cached_minutes(
            measured["minutes"], cached_steps, total_steps
        )
        return measured
    # An unpriced parent (own["minutes"] is None) whose total ends up coming
    # only from a priced child is not a complete figure - the parent's own
    # steps (a for_each step with no cost block, say) contributed nothing to
    # it, so the sum understates the run rather than merely omitting a piece
    # of it (#242). `unpriced` names each contributor that landed here, so a
    # caller can tell a trivial utility step from an unpriced 12-shot loop
    # apart rather than just seeing `partial: true` (#252)
    partial = own["minutes"] is None and not _only_composes_children(definition)
    unpriced = [definition.get("id", "workflow")] if partial else []
    totals = _ChildTotals(own["minutes"], partial, unpriced)
    for path, step_arguments in _sub_workflow_paths(expanded):
        child_definition, child_cost = _read_child(path, base_dir, workflow_dir)
        child_observed = _child_observed(
            observed_for_child, path, child_definition, step_arguments, own["minutes"]
        )
        child = _observed(child_observed, device)
        is_observed = child is not None
        if not is_observed:
            child = _child_catalog_price(
                child_definition, child_cost, step_arguments, device
            )
        totals.add(path, child, is_observed)
        note = (
            None
            if is_observed
            else _unresolved_list_note(path, child_cost, step_arguments, device)
        )
        if note and child["minutes"] is not None:
            totals.differs.append(note)
    result = _rolled_up_estimate(own, totals, device, cached_steps, total_steps)
    if own.get("task_rates") and result["minutes"] is not None:
        # A declared rate is a bound, never a measurement (#1619)
        result["low_confidence"] = True
    withheld = observed.get("withheld") if isinstance(observed, dict) else None
    if withheld and result["basis"] == UNKNOWN and result["minutes"] is None:
        # History exists but could not be re-priced for this workflow's size:
        # say which template and why, so it reads differently from no history
        result["inherited_from"] = observed.get("inherited_from")
        result["withheld"] = withheld
    elif result["basis"] == UNKNOWN and result["minutes"] is None:
        reason = _unpriced_reason(definition)
        if reason:
            result["unpriced_reason"] = reason
    return result


def _unpriced_reason(definition):
    """Why a workflow with no price at all has none, or None when the cause
    is not a shape this can name (#948). #797 prices an inline workflow from
    the catalog template with the same pipeline identity, so a workflow of
    task steps alone has nothing to match and its cost is simply unmeasured -
    which says nothing about whether the run is seconds or an hour. One
    whose every command declares a rate is priced from those (#1619), so
    what is left unpriced names the commands that declare none."""
    undeclared = undeclared_task_commands(definition)
    if not undeclared:
        return None
    return (
        "no declared rate: " + ", ".join(undeclared) + " - a task-only "
        "workflow with no cost block and no run history of its own, and a "
        "command with no declared task rate to price it from"
    )


def _with_worst_case(measured, definition, expanded, device=None):
    """An observed or inherited estimate with a checked chain's worst-case
    re-rolls quoted on it, as the curated figure quotes them (#1246): the
    history is read as N segment renders (a matched chain's as one), and
    `observed_render_scaling` says how many a run at `expanded` may pay. Unchanged when nothing can
    re-roll. A history that ran checked already holds what re-rolls it drew,
    so this is an upper bound, which is what the worst case promises. This
    box's own history, so re-priced, is `derived`; an inherited one stays
    `inherited`."""
    if measured is None or measured.get("minutes") is None:
        return measured
    variables = (expanded or {}).get("variables") or {}
    factor, axes, unscaled = observed_render_scaling(definition, variables, measured)
    if unscaled:
        measured = {
            **measured,
            "differs": _unique((measured.get("differs") or []) + unscaled),
            "low_confidence": True,
        }
    # A words-checked chain's Whisper passes, on top, as the curated figure
    # quotes them (#1262): additive, so they are not scaled by `factor`
    words_minutes, words_axis = words_check_term(definition, variables)
    if not axes and not words_minutes:
        return measured
    # A re-roll renders the segment again; it does not load the model again
    fixed = measured["minutes"] * fixed_share(definition, device)
    terms = axes + ([words_axis] if words_minutes else [])
    worst = {
        **measured,
        "minutes": round(
            fixed + (measured["minutes"] - fixed) * factor + words_minutes, 1
        ),
        "scaled": "; ".join(
            part
            for part in (
                measured.get("scaled"),
                f"from the observed {_shown_minutes(measured['minutes'])} min: "
                + ", ".join(terms),
            )
            if part
        ),
    }
    if worst["basis"] == OBSERVED:
        worst["basis"] = DERIVED
    return worst


# How far a frames rescale reaches outside the runs it is read from before
# its figure is flagged `low_confidence`: the rescale is linear, and
# attention cost grows faster than linear in tokens (#1696)
FRAMES_CONFIDENT_RATIO = (0.5, 2.0)
FRAMES_SCALED_KEY = "frames_scaled"


def _frames_rescaled(observed, definition, expanded, device):
    """A bucket's `observed` block rescaled to the frames this run's list
    entries render, from the frames its cold runs rendered
    (`observed_cost.ENTRY_FRAMES_KEY`), or None when it carries none to
    rescale from - an older block, entries with no `num_frames`, or another
    template's history - which keeps #796's discard (#1696).

    Cold and warm minutes both scale by run frames / bucket frames, the
    workflow's fixed share of each held as `_scaled_observed` holds it
    (#1224). The block says how under `FRAMES_SCALED_KEY`, which `_observed`
    reads into `scaled` and the basis."""
    if not isinstance(observed, dict) or observed.get("inherited_from"):
        return None
    was = observed.get(ENTRY_FRAMES_KEY)
    if isinstance(was, bool) or not isinstance(was, (int, float)) or was <= 0:
        return None
    now = list_entry_frames(definition, (expanded or {}).get("variables") or {})
    cold = observed.get("cold_minutes")
    if now is None or not isinstance(cold, (int, float)):
        return None
    ratio = now / was
    if math.isclose(ratio, 1.0):
        # The bucket's runs rendered these frames: the measurement stands
        return observed
    share = fixed_share(definition, device)
    block = dict(observed)
    for key in ("cold_minutes", "warm_minutes"):
        minutes = observed.get(key)
        if isinstance(minutes, (int, float)) and not isinstance(minutes, bool):
            fixed = minutes * share
            block[key] = fixed + (minutes - fixed) * ratio
    fixed = cold * share
    block[FRAMES_SCALED_KEY] = {
        "ratio": ratio,
        "note": f"scaled from the observed {_shown_minutes(round(cold, 1))} min "
        f"at {_shown_minutes(was)} frames"
        + (f" ({_shown_minutes(round(fixed, 1))} min of it fixed)" if fixed else "")
        + f": num_frames {_shown_minutes(now)} vs {_shown_minutes(was)}",
    }
    return block


def _shown_minutes(minutes):
    return int(minutes) if float(minutes).is_integer() else round(minutes, 1)


def _observed(observed, device):
    """This box's history as an estimate, or None when it has nothing to
    quote from.

    Withheld rather than quoted when the history is a different backend's
    (a card swapped under the same jobs table), or when every comparable run
    was warm: a warm run had the weights already resident and quoting it as
    the cost of a run that has to load them would under-quote by the minutes
    this estimate exists to name.
    """
    if not isinstance(observed, dict):
        return None
    minutes = observed.get("cold_minutes")
    runs = observed.get("cold_runs")
    if not isinstance(minutes, (int, float)) or not runs:
        return None
    if observed.get("device") not in (None, device):
        return None
    estimate = {
        "minutes": round(float(minutes), 1),
        "basis": OBSERVED,
        "device": device,
        "measured_on": observed.get("name"),
        "partial": False,
        "unpriced": [],
        "runs": runs,
    }
    if observed.get("per_segment"):
        estimate["per_segment"] = True  # a matched chain's (observed_cost)
    inherited_from = observed.get("inherited_from")
    if observed.get("scaled") and not inherited_from:
        # This box's history of the default tier, re-priced for the drivers
        # that moved (#1236): derived, not measured
        estimate["basis"] = DERIVED
        estimate["scaled"] = observed["scaled"]
        estimate["low_confidence"] = True
        if observed.get("differs"):
            # Moved a driver no rule scales: measured at another setting
            estimate["differs"] = list(observed["differs"])
    frames = observed.get(FRAMES_SCALED_KEY)
    if frames and not inherited_from:
        # The bucket's history rescaled to this run's frames (#1696): derived,
        # and flagged only when it reaches far outside the runs it is from
        estimate["basis"] = DERIVED
        estimate["scaled"] = "; ".join(
            part for part in (estimate.get("scaled"), frames["note"]) if part
        )
        low, high = FRAMES_CONFIDENT_RATIO
        if not low <= frames["ratio"] <= high:
            estimate["low_confidence"] = True
    if inherited_from:
        # History of the catalog template with this workflow's pipeline,
        # not of this workflow - an approximation (#797)
        estimate["basis"] = INHERITED
        estimate["inherited_from"] = inherited_from
        estimate["differs"] = list(observed.get("differs") or [])
        if estimate["differs"]:
            # Measured at another attention backend, cache, LoRA, offload or
            # quantization than this run's: kept, but flagged (#866)
            estimate["low_confidence"] = True
        if observed.get("scaled"):
            # Re-priced linearly from the template's size (#851): an
            # approximation, not a measurement
            estimate["scaled"] = observed["scaled"]
            estimate["low_confidence"] = True
    return estimate


# Task commands that only cut or join text in memory: no model, no media, no
# loop over one, so a parent whose own steps are these plus child workflows
# has no uncosted work of its own (#1222: narration-long's `split`).
TEXT_GLUE_TASKS = frozenset({"split_takes"})


def _is_text_glue(step):
    """Whether `step` is a TEXT_GLUE_TASKS task run once (no for_each)."""
    task = step.get("task") if isinstance(step, dict) else None
    return (
        isinstance(task, dict)
        and task.get("command") in TEXT_GLUE_TASKS
        and not step.get("for_each")
    )


def _only_composes_children(definition):
    """Whether every step the workflow *declares* is a `workflow` step -
    i.e. the parent does no work of its own beyond assembling children.

    Distinguishes an uncosted parent that is pure composition (#268: its
    children's figures are the whole story) from one with real uncosted work
    of its own, such as a `for_each` task step with no `cost` block (#242:
    a priced child there still leaves a genuine gap). Reads the *written*
    definition rather than `expanded`: a step that saves nothing and that
    nothing reads is elided from the run (#122), and an elided own step is
    still work the author declared - it must not be mistaken for pure
    composition just because it would not execute.
    """
    steps = definition.get("steps") or []
    if not steps:
        return False
    for step in steps:
        if _is_text_glue(step):
            continue
        reference = step.get("workflow") if isinstance(step, dict) else None
        if not isinstance(reference, dict) or not isinstance(
            reference.get("path"), str
        ):
            return False
    return True
