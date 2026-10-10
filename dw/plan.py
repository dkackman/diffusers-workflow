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
"""

import copy
import hashlib
import json
import logging
import math
import os

from huggingface_hub import model_info
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import GatedRepoError, HFValidationError, validate_repo_id

from .elision import elide_definition
from .hub_cache import repo_download_incomplete, scan_models
from . import references
from .realize import read_sub_workflow, realize_workflow
from .security import validate_url
from .validation import _is_seeded

logger = logging.getLogger("dw")

FINGERPRINT_PREFIX = "sha256:"
# Top-level keys that document a workflow rather than shape its work
DOCUMENTATION_KEYS = ("cost", "description", "summary", "configures")
FOR_EACH_KEY = "for_each"
SIZE_LOOKUP_TIMEOUT = 5.0
GIB = 1024**3


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
    realized, annotations = realize_workflow(
        definition,
        candidate.folded_variables(arguments),
        seed=0,
        base_dir=base_dir,
        prompt_dir=prompt_dir,
        output_root=candidate.output_dir,
        workflow_dir=candidate.workflow_dir,
        pin_outputs=False,
    )
    # The expansion validation took, which folds and expands exactly as
    # the run does (Workflow._fold, Workflow._expand)
    expanded = candidate.expanded_definition(arguments)
    # The plan is what the run does, and a run does not execute a step
    # nothing reads (dw/elision.py, #122) - so the step count, the downloads
    # and the fingerprint are all taken after elision, and the acknowledged
    # cost is the cost of the work that happens
    elided = elide_definition(expanded, definition)
    entries = list_entries(definition, realized)
    measured_entries = list_entries(definition, definition)
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
        "fingerprint": fingerprint(work, definition, annotations),
        "steps": step_count,
        "elided_steps": elided,
        "list_entries": entries,
        "cached_steps": cache_hits,
        "downloads_required": downloads_required(
            expanded, base_dir, candidate.workflow_dir, cache_dir, lookup_sizes
        ),
        "estimate": estimate(
            definition,
            expanded,
            entries,
            device,
            base_dir,
            candidate.workflow_dir,
            measured_entries=measured_entries,
            observed=_observed_block(observed, arguments),
            cached_steps=cache_hits,
            total_steps=step_count,
            observed_for_child=observed_for_child,
        ),
    }


def _observed_block(observed, arguments):
    """The `observed` block for this run, from a block or from a callable
    the caller passed - best effort, since a figure is a nicety and a plan
    that raised because history could not be read would cost the caller the
    whole pre-flight."""
    if not callable(observed):
        return observed if isinstance(observed, dict) else None
    try:
        block = observed(arguments or {})
    except Exception:
        logger.debug("No observed history for the plan", exc_info=True)
        return None
    return block if isinstance(block, dict) else None


def list_entries(definition, realized):
    """{variable: length} for every `for_each` that names a list variable,
    read from `realized`'s folded variables - a literal list is not an
    argument and is not listed.

    Passing `definition` as both arguments answers the lengths the
    workflow's *stored defaults* carry, which is the list a catalog `cost`
    figure was measured against (`estimate`).
    """
    variables = realized.get("variables") or {}
    entries = {}
    for step in definition.get("steps") or []:
        if not isinstance(step, dict):
            continue
        reference = step.get(FOR_EACH_KEY)
        name = references.ref_name(references.VARIABLE, reference)
        if name is not None:
            value = variables.get(name)
            if isinstance(value, list):
                entries[name] = len(value)
    return entries


# `estimate`'s own `list_entries` parameter (the parent's) shadows this
# function's name in its scope - a child's entries are computed under this
# alias instead (#341)
_list_entries = list_entries


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


def fingerprint(expanded, definition, annotations=None):
    """SHA-256 over the expanded definition with everything that is not
    work removed: the seed wherever it sits, and the documentation keys.

    `definition` is the workflow as written, consulted for whether the
    top-level seed named a variable - if it did, that variable's folded
    value is the seed too and is blanked at its source. `annotations` is
    what realization recorded beside the copy; its sub-workflow digests go
    into the hash, since a composed child edited between the quote and the
    call is different work the parent's text cannot show.
    """
    doc = copy.deepcopy(expanded)
    if annotations and annotations.get("sub_workflows"):
        doc["__sub_workflows__"] = dict(annotations["sub_workflows"])
    doc.pop("seed", None)
    for key in DOCUMENTATION_KEYS:
        doc.pop(key, None)
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
    # default=repr: a realized 'constant:' can be any Python value, and the
    # fingerprint only needs it to be stable, not round-trippable
    serialized = json.dumps(
        doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=repr
    )
    return FINGERPRINT_PREFIX + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


UNKNOWN = "unknown"
CATALOG = "catalog"
PER_ENTRY = "per_entry"
DERIVED = "derived"
OTHER_DEVICE = "other_device"
OBSERVED = "observed"
INHERITED = "inherited"

# #301: a single run is not the same statistical basis as a dozen. Below
# this many observed runs, an "observed" figure is tempered rather than
# quoted at full authority - see `_tempered`.
SMALL_N_THRESHOLD = 3


def _tempered(block, curated_minutes):
    """An `observed` estimate below `SMALL_N_THRESHOLD` runs, corrected
    toward the curated figure it might be papering over rather than
    presented as if it carried the same authority as a dozen runs (#301).

    With a curated minutes figure to blend toward, the point estimate is
    pulled toward it in proportion to how thin the history is - one run
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


COST_DRIVERS_KEY = "cost_drivers"


def _declared_drivers(definition):
    """The variables the author says move this workflow's cost - the same
    rule `dw/server/observed_cost.py`'s `declared_drivers` applies, kept
    local so this module does not import the server package (#267)."""
    drivers = definition.get(COST_DRIVERS_KEY)
    if not isinstance(drivers, list):
        return []
    variables = definition.get("variables") or {}
    return [name for name in drivers if isinstance(name, str) and name in variables]


def _driver_comparable(value):
    """A driver value as something hashable and stable across JSON round
    trips, mirroring `dw/server/observed_cost.py`'s `_comparable`."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return json.dumps(value, sort_keys=True, default=str)


def _numeric_fields(entries):
    """{field: {values}} over the numeric fields of a list's dict entries. A
    numeric string ("243") counts as its number, as `_driver_comparable` does
    for a scalar driver."""
    fields = {}
    if not isinstance(entries, list):
        return fields
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            number = _driver_comparable(value)
            if isinstance(number, float) and math.isfinite(number):
                fields.setdefault(key, set()).add(number)
    return fields


def _list_entry_field_shifted(default_entries, effective_entries):
    """Whether a list driver's entries carry a numeric field (a per-shot
    `num_frames`, say) with a value none of the default entries had. The
    curated figure was measured over the default entries' values, so a value
    outside them has no matching bucket - the #267 rule applied inside a
    list driver (#593)."""
    measured = _numeric_fields(default_entries)
    for field, values in _numeric_fields(effective_entries).items():
        if field in measured and not values <= measured[field]:
            return True
    return False


def _per_entry_variable(cost, device):
    """The list variable the cost entry priced for `device` (else the first,
    as `_price` chooses) re-prices by a measured per-entry rate, or None."""
    entries = [entry for entry in (cost or []) if isinstance(entry, dict)]
    chosen = next((e for e in entries if e.get("device") == device), None)
    if chosen is None and entries:
        chosen = entries[0]
    per = chosen.get("per_entry") if chosen else None
    return per.get("variable") if isinstance(per, dict) else None


def _scalar_driver_shifted(definition, expanded, list_entries, per_entry=None):
    """Whether a declared, non-list `cost_driver` was overridden away from
    the default value the curated `cost` was measured against (#267).

    A list driver is `_repriced`'s to catch - it changes the plan's list
    count, not a bare variable's value. This only asks about a scalar one,
    which `_repriced` never looks at, so a `catalog` figure measured for one
    `num_frames` silently priced a run at a different one.
    """
    defaults = definition.get("variables") or {}
    effective = expanded.get("variables") or {}
    for name in _declared_drivers(definition):
        default_value = defaults.get(name)
        if isinstance(default_value, list):
            # A list the cost prices per entry is already extrapolated in
            # either direction; its entries' index/name fields are labels,
            # not cost fields (#772)
            if name != per_entry and _list_entry_field_shifted(
                default_value, effective.get(name)
            ):
                return True
            continue
        if name in list_entries:
            continue
        if _driver_comparable(effective.get(name)) != _driver_comparable(default_value):
            return True
    return False


def _own_price(definition, expanded, list_entries, device, measured_entries):
    """The workflow's own price, reset to unknown when a scalar driver moved."""
    own = _price(definition.get("cost"), device, list_entries, measured_entries or {})
    # `derived` and `per_entry` are re-priced by entry count alone, so a
    # per-entry num_frames outside the measured values needs the same check (#796)
    if own["basis"] in (CATALOG, OTHER_DEVICE, DERIVED, PER_ENTRY) and (
        _scalar_driver_shifted(
            definition,
            expanded,
            list_entries,
            _per_entry_variable(definition.get("cost"), device),
        )
    ):
        # A scalar cost_driver (H3's num_frames, say) moved away from the
        # value the curated cost was measured against, and _repriced only
        # re-prices a for_each list's length - so the catalog figure would
        # otherwise be quoted for a run it was never measured for (#267)
        own = {"minutes": None, "basis": UNKNOWN, "measured_on": None}
    return own


def _read_child(path, base_dir, workflow_dir):
    """A composed child's definition and its cost block, or (None, None)
    when it is a builtin (the parent's to price) or unreadable."""
    # A builtin is the parent's to price; a local child prices itself
    raw = read_sub_workflow(path, base_dir, workflow_dir)
    if raw is None:
        return None, None
    try:
        child_definition = json.loads(raw)
        return child_definition, child_definition.get("cost")
    except (ValueError, AttributeError):
        return None, None


def _child_observed(
    observed_for_child, path, child_definition, step_arguments, own_minutes
):
    """A child's observed figure, only when the total can honestly end up
    basis: observed."""
    # A child's observed figure only ever feeds a total that can
    # honestly end up basis: observed (own["minutes"] is None, below) -
    # a priced parent's own basis is 'catalog', and summing an observed
    # child into it produced a total that did not match either figure
    # while still claiming 'catalog' (#315)
    if (
        own_minutes is not None
        or observed_for_child is None
        or child_definition is None
    ):
        return None
    try:
        # `step_arguments` are the composing step's own overrides -
        # passed through so a child observed lookup buckets against
        # the value this step actually runs with rather than always
        # the child's stored defaults (#341)
        return observed_for_child(path, child_definition, step_arguments)
    except Exception:
        return None


def _child_catalog_price(child_definition, child_cost, step_arguments, device):
    """A child priced from its own cost block, re-priced for the composing
    step's arguments."""
    child_list_entries = {}
    child_measured_entries = {}
    child_expanded = {"variables": {}}
    if child_definition is not None:
        child_measured_entries = _list_entries(child_definition, child_definition)
        # The composing step's own `arguments` are what the child
        # actually runs with - folded over its declared defaults the
        # same way a caller's arguments are, since `expanded` has
        # already substituted them to concrete values (#341)
        effective_variables = dict(child_definition.get("variables") or {})
        effective_variables.update(step_arguments)
        child_expanded = {"variables": effective_variables}
        child_list_entries = _list_entries(child_definition, child_expanded)
    child = _price(child_cost, device, child_list_entries, child_measured_entries)
    if (
        child["basis"] in (CATALOG, OTHER_DEVICE)
        and child_definition is not None
        and _scalar_driver_shifted(
            child_definition,
            child_expanded,
            child_list_entries,
            _per_entry_variable(child_cost, device),
        )
    ):
        # A scalar cost_driver the composing step overrode (H3's
        # num_frames at 345 against a default of 124, say) is the
        # same #267 failure one level down - the child's own
        # catalog figure was never measured for the value this
        # step actually passes it (#341)
        child = {"minutes": None, "basis": UNKNOWN, "measured_on": None}
    return child


class _ChildTotals:
    """What the composed children add up to, folded in one child at a time."""

    def __init__(self, minutes, partial, unpriced):
        self.minutes = minutes
        self.partial = partial
        self.unpriced = unpriced
        self.unpriced_children = []
        self.had_child = False
        self.all_observed = True
        self.runs = []
        self.measured_on = set()
        self.bases = []

    def add(self, path, child, observed):
        self.had_child = True
        if observed:
            self.runs.append(child["runs"])
            self.measured_on.add(child["measured_on"])
        else:
            self.all_observed = False
        if child["minutes"] is None:
            self.partial = True
            self.unpriced.append(path)
            self.unpriced_children.append(path)
            return
        self.bases.append((child["basis"], child.get("measured_on")))
        if self.minutes is not None:
            self.minutes += child["minutes"]
        else:
            self.minutes = child["minutes"]


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
    (#85) - an estimate, not a measurement, and the only alternative to
    quoting a 5-shot figure for a 10-shot run.

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
    measured = _observed(observed, device)
    own = _own_price(definition, expanded, list_entries, device, measured_entries)
    if measured is not None:
        measured = _tempered(measured, own["minutes"])
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
    return _rolled_up_estimate(own, totals, device, cached_steps, total_steps)


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
    inherited_from = observed.get("inherited_from")
    if inherited_from:
        # History of the catalog template with this workflow's pipeline,
        # not of this workflow - an approximation (#797)
        estimate["basis"] = INHERITED
        estimate["inherited_from"] = inherited_from
        estimate["differs"] = list(observed.get("differs") or [])
    return estimate


def _sub_workflow_paths(expanded):
    """The local (non-builtin) sub-workflow path and composing arguments of
    every composing step, as (path, arguments) - `expanded` has already
    substituted and expanded `for_each`, so each occurrence carries the
    concrete arguments that step actually passes the child (#341)."""
    for step in expanded.get("steps") or []:
        reference = step.get("workflow") if isinstance(step, dict) else None
        path = reference.get("path") if isinstance(reference, dict) else None
        if isinstance(path, str) and not references.is_ref(references.BUILTIN, path):
            arguments = reference.get("arguments")
            yield path, arguments if isinstance(arguments, dict) else {}


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
        reference = step.get("workflow") if isinstance(step, dict) else None
        if not isinstance(reference, dict) or not isinstance(
            reference.get("path"), str
        ):
            return False
    return True


def _price(cost, device, list_entries, measured_entries):
    """One cost list priced for `device` and `list_entries`, as
    {minutes, basis, measured_on}.

    `measured_entries` is the list length the figure was measured with -
    the workflow's stored defaults. When the caller's list differs and the
    entry carries no measured `per_entry` rate, the total is extrapolated
    linearly over it and reported as 'derived'; when more than one list
    changed there is nothing honest to extrapolate along, so the figure is
    withheld rather than quoted for the wrong list.
    """
    entries = [entry for entry in (cost or []) if isinstance(entry, dict)]
    if not entries:
        return {"minutes": None, "basis": UNKNOWN, "measured_on": None}
    chosen = next((entry for entry in entries if entry.get("device") == device), None)
    basis = CATALOG
    if chosen is None:
        chosen = entries[0]
        basis = OTHER_DEVICE
    minutes = float(chosen.get("minutes", 0))
    per = chosen.get("per_entry")
    # Another device's figure is re-priced for the list length too (#589): it
    # stays `other_device` (still not a measurement here), but a 32-entry
    # for_each must not be quoted at the one-clip figure
    if isinstance(per, dict) and per.get("variable") in list_entries:
        count = list_entries[per["variable"]]
        each = float(per.get("minutes", 0))
        measured_with = int(per.get("entries", 0))
        minutes = max(0.0, (minutes - each * measured_with) + each * count)
        if basis == CATALOG:
            basis = PER_ENTRY
    else:
        minutes, repriced_basis = _repriced(minutes, list_entries, measured_entries)
        if repriced_basis == UNKNOWN:
            return {"minutes": None, "basis": UNKNOWN, "measured_on": None}
        if basis == CATALOG:
            basis = repriced_basis
    return {"minutes": minutes, "basis": basis, "measured_on": chosen.get("name")}


def _repriced(minutes, list_entries, measured_entries):
    """A catalog total re-priced for a list the caller lengthened or
    shortened, as (minutes, basis).

    Linear in the entry count: the figure was measured over
    `measured_entries` entries and the run does `list_entries` of them, so
    a 42-minute 5-shot figure quotes 84 for 10 shots rather than 42. It
    over-counts fixed setup at the long end and under-counts it at the
    short end - which is why it is labelled `derived` rather than
    `catalog`, and why a template that measures a real `per_entry` rate
    beats it.
    """
    changed = [
        (count, measured_entries[name])
        for name, count in list_entries.items()
        if name in measured_entries and count != measured_entries[name]
    ]
    if not changed:
        return minutes, CATALOG
    if len(changed) > 1:
        return None, UNKNOWN
    count, measured_with = changed[0]
    if measured_with <= 0:
        return None, UNKNOWN
    return minutes * count / measured_with, DERIVED


FROM_PRETRAINED_KEY = "from_pretrained_arguments"
MODEL_NAME_KEY = "model_name"
# An adapter names its repo directly, not through from_pretrained_arguments
LORAS_KEY = "loras"
SINGLE_FILE_KEY = "from_single_file"
VARIANT_KEY = "variant"


def downloads_required(expanded, base_dir, workflow_dir, cache_dir, lookup_sizes):
    """The hub repos and checkpoint URLs the run would fetch before its
    first step: every `model_name` in the expanded definition (and in each
    composed child) that `scan_models` does not find, plus every one it does
    find but whose cached copy is left over from an interrupted pull
    (#382) - a component the load needs is missing, or a blob is still
    `.incomplete` - plus every `from_single_file` that is a URL. Sizes come
    from the hub when asked and are None whenever it does not answer - an
    offline box is a state, not an error, so nothing here raises or logs
    above debug.
    """
    names = []
    variants = {}
    urls = []
    _collect_sources(expanded, names, variants, urls)
    for path, _arguments in _sub_workflow_paths(expanded):
        raw = read_sub_workflow(path, base_dir, workflow_dir)
        if raw is None:
            continue
        try:
            _collect_sources(json.loads(raw), names, variants, urls)
        except ValueError:
            continue
    present = {repo.get("repo_id") for repo in scan_models(cache_dir).get("repos", [])}
    required = []
    for name in names:
        # A name not shaped like a hub id is a local checkout - decided by
        # shape, never by touching the disk: the name came from the request
        # body, and a free pre-flight must not be a directory-existence oracle
        if not _is_repo_id(name):
            continue
        if name in present and not repo_download_incomplete(
            name, cache_dir, variant=variants.get(name)
        ):
            continue
        entry = {"repo": name, "gb": None, "gated": None, "access_blocked": None}
        if lookup_sizes:
            entry["gb"], entry["gated"], entry["access_blocked"] = _model_info(name)
        required.append(entry)
    for url in urls:
        required.append(
            {
                "repo": None,
                "url": url,
                "gb": None,
                "gated": None,
                "access_blocked": None,
            }
        )
    return required


def _collect_sources(tree, names, variants, urls):
    """Every from_pretrained source in a tree, first-seen order, deduplicated.
    `variants` collects each name's requested `variant` (first-seen), used to
    tell a repo that is merely missing its fp16 files from one that is fully
    cached without them.

    A `loras` entry counts too. It carries its repo under `model_name`
    directly rather than inside a `from_pretrained_arguments` block, so the
    walk missed it: `templates/ltx2/reference-sheet` on a box that had
    every base weight but not the IC-LoRA answered `downloads_required:
    []` and then pulled it mid-run, which is the one question the field
    exists to answer (found verifying #151).
    """
    if isinstance(tree, dict):
        for lora in tree.get(LORAS_KEY) or []:
            name = lora.get(MODEL_NAME_KEY) if isinstance(lora, dict) else None
            if isinstance(name, str) and name not in names:
                names.append(name)
        source = tree.get(FROM_PRETRAINED_KEY)
        if isinstance(source, dict):
            name = source.get(MODEL_NAME_KEY)
            if isinstance(name, str) and name not in names:
                names.append(name)
                variant = source.get(VARIANT_KEY)
                if isinstance(variant, str):
                    variants[name] = variant
            single = source.get(SINGLE_FILE_KEY)
            if isinstance(single, str) and _is_url(single) and single not in urls:
                urls.append(single)
        for value in tree.values():
            _collect_sources(value, names, variants, urls)
    elif isinstance(tree, list):
        for value in tree:
            _collect_sources(value, names, variants, urls)


def gate_warnings(downloads_required):
    """One line per required download this box's token is blocked from -
    the pre-flight signal #186 asked for, so a gated repo is a
    `validate_workflow`-visible condition rather than a 403 the run only
    discovers after it has loaded everything ahead of that step."""
    return [
        f"{entry['repo']} is gated and not accessible with this box's "
        "Hugging Face token - accept its license at "
        f"https://huggingface.co/{entry['repo']} before running this workflow"
        for entry in downloads_required
        if entry.get("access_blocked")
    ]


def _is_repo_id(name):
    try:
        validate_repo_id(name)
        return True
    except HFValidationError:
        return False


def _is_url(value):
    if not value.startswith(("http://", "https://")):
        return False
    try:
        validate_url(value)
        return True
    except Exception:
        return False


def _model_info(name):
    """A repo's size in GiB, gate status, and whether this box's token is
    blocked from it - a `(gb, gated, access_blocked)` triple.

    `gated` is `model_info`'s own field (`False` / `"auto"` / `"manual"`),
    readable only when the call succeeds - and the hub answers `model_info`
    for a gated repo regardless of whether this token has been granted
    access, since that call serves metadata rather than file bytes (found
    verifying #186: `access_blocked` came back `false` for repos this box's
    token was actually refused on). A `GatedRepoError` straight out of
    `model_info` is still a real signal - some other endpoint behind it
    checked and refused - but its absence proves nothing, so a `gated` repo
    that got this far is checked for real with a HEAD request against one of
    its own files (`_probe_gate_blocked`), the same request class a run's
    actual load would make and the one place the hub's 403 for "gated, token
    not accepted" actually appears.
    """
    try:
        info = model_info(name, files_metadata=True, timeout=SIZE_LOOKUP_TIMEOUT)
    except GatedRepoError as e:
        logger.debug(f"Gate not accepted for {name}: {e}")
        return None, True, True
    except Exception as e:
        logger.debug(f"No size for {name}: {e}")
        return None, None, None
    total = sum(s.size for s in (info.siblings or []) if getattr(s, "size", None))
    gb = round(total / GIB, 1) if total else None
    gated = getattr(info, "gated", False) or False
    access_blocked = False
    if gated:
        access_blocked = _probe_gate_blocked(name, info.siblings or [])
    return gb, gated, access_blocked


def _probe_gate_blocked(name, siblings):
    """HEAD one real file of a gated repo to see whether this box's token is
    actually accepted - `model_info` succeeding says nothing either way
    (#186). `None` when there is no file to probe or the probe fails for a
    reason other than the gate, since that is "unknown", not "not blocked".
    """
    filename = next(
        (s.rfilename for s in siblings if getattr(s, "rfilename", None)), None
    )
    if filename is None:
        return None
    try:
        get_hf_file_metadata(hf_hub_url(name, filename), timeout=SIZE_LOOKUP_TIMEOUT)
        return False
    except GatedRepoError as e:
        logger.debug(f"Gate not accepted for {name}: {e}")
        return True
    except Exception as e:
        logger.debug(f"Gate probe inconclusive for {name}: {e}")
        return None
