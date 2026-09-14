"""The plan a validate call answers with: what a run of a workflow with a
caller's arguments will actually execute, what it will have to download
first, and what the workflow's own cost block says it will take - with a
fingerprint over the work, so an acknowledgement can be bound to it and a
run whose shape changed after consent refused (#85, stage 2).

Everything here is derived from the same resolvers the run uses -
`realize_workflow` folds the arguments and inlines the prompts, and
`Workflow.expanded_definition` substitutes and expands `for_each` - so the
plan describes the run and not an approximation of it. Nothing here knows a
model: every minute comes from a `cost` block and every repo name from a
`from_pretrained_arguments`.
"""

import copy
import hashlib
import json
import logging
import os

from huggingface_hub import model_info
from huggingface_hub.utils import HFValidationError, validate_repo_id

from .elision import elide_definition
from .hub_cache import scan_models
from .realize import (
    BUILTIN_PREFIX,
    VARIABLE_PREFIX,
    read_sub_workflow,
    realize_workflow,
)
from .security import validate_url
from .workflow import Workflow

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
):
    """What a run of `candidate` with `arguments` will execute and cost.

    Args:
        candidate: The Workflow the route built - it carries the file spec
            (so base_dir), the output root and the confinement a run has.
        arguments: The caller's arguments, already past `argument_errors`;
            an undeclared name or an uncoercible value raises here.
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
    """
    definition = candidate.workflow_definition
    base_dir = (
        os.path.dirname(os.path.abspath(candidate.file_spec))
        if candidate.file_spec
        else None
    )
    realized, annotations = realize_workflow(
        definition,
        arguments,
        seed=0,
        base_dir=base_dir,
        prompt_dir=prompt_dir,
        output_root=candidate.output_dir,
        workflow_dir=candidate.workflow_dir,
        pin_outputs=False,
    )
    # Arguments are already folded into the realized variables, so the
    # expansion takes none; it substitutes and expands exactly as the run
    expanded = Workflow(
        realized, candidate.output_dir, candidate.file_spec, candidate.workflow_dir
    ).expanded_definition()
    # The plan is what the run does, and a run does not execute a step
    # nothing reads (dw/elision.py, #122) - so the step count, the downloads
    # and the fingerprint are all taken after elision, and the acknowledged
    # cost is the cost of the work that happens
    elided = elide_definition(expanded, definition)
    entries = list_entries(definition, realized)
    measured_entries = list_entries(definition, definition)
    return {
        "fingerprint": fingerprint(expanded, definition, annotations),
        "steps": len(expanded.get("steps") or []),
        "elided_steps": elided,
        "list_entries": entries,
        "cached_steps": cached_steps(definition, realized, arguments, cache_probe),
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
        if isinstance(reference, str) and reference.startswith(VARIABLE_PREFIX):
            name = reference.removeprefix(VARIABLE_PREFIX)
            value = variables.get(name)
            if isinstance(value, list):
                entries[name] = len(value)
    return entries


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


def unseeded_cache_warnings(definition, arguments=None):
    """Say once, where a caller is already looking, that an unseeded workflow
    gets no step cache at all.

    `cached_steps: 0` is indistinguishable from 'probed, nothing hit' out
    there, and the difference is the one that matters: without a `seed` the
    cache is off, so nothing is ever reused however many times the same
    workflow runs (#107).
    """
    if _is_seeded(definition, arguments):
        return []
    return [
        "This workflow sets no 'seed', so the step cache is disabled and "
        "'cached_steps' is 0 without being probed - every step regenerates "
        "on every run. Set a top-level 'seed' to make a repeat run reuse "
        "what it already produced"
    ]


def _is_seeded(definition, arguments):
    """Whether a run of this workflow has a seed before it draws one - read
    from the definition as written and the caller's arguments, since
    realization pins a seed of its own into the copy."""
    seed = definition.get("seed")
    if isinstance(seed, str) and seed.startswith(VARIABLE_PREFIX):
        name = seed.removeprefix(VARIABLE_PREFIX)
        if name in (arguments or {}):
            return arguments[name] is not None
        return (definition.get("variables") or {}).get(name) is not None
    return seed is not None


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
    if isinstance(written_seed, str) and written_seed.startswith(VARIABLE_PREFIX):
        name = written_seed.removeprefix(VARIABLE_PREFIX)
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


def estimate(
    definition,
    expanded,
    list_entries,
    device,
    base_dir,
    workflow_dir,
    measured_entries=None,
    observed=None,
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
    if measured is not None:
        return measured
    own = _price(definition.get("cost"), device, list_entries, measured_entries or {})
    minutes = own["minutes"]
    partial = False
    for path in _sub_workflow_paths(expanded):
        # A builtin is the parent's to price; a local child prices itself
        raw = read_sub_workflow(path, base_dir, workflow_dir)
        child_cost = None
        if raw is not None:
            try:
                child_cost = json.loads(raw).get("cost")
            except (ValueError, AttributeError):
                child_cost = None
        child = _price(child_cost, device, {}, {})
        if child["minutes"] is None:
            partial = True
        elif minutes is not None:
            minutes += child["minutes"]
        else:
            minutes = child["minutes"]
    return {
        "minutes": round(minutes, 1) if minutes is not None else None,
        "basis": own["basis"],
        "device": device,
        "measured_on": own["measured_on"],
        "partial": partial,
        "runs": None,
    }


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
    return {
        "minutes": round(float(minutes), 1),
        "basis": OBSERVED,
        "device": device,
        "measured_on": observed.get("name"),
        "partial": False,
        "runs": runs,
    }


def _sub_workflow_paths(expanded):
    """The local (non-builtin) sub-workflow path of every composing step."""
    for step in expanded.get("steps") or []:
        reference = step.get("workflow") if isinstance(step, dict) else None
        path = reference.get("path") if isinstance(reference, dict) else None
        if isinstance(path, str) and not path.startswith(BUILTIN_PREFIX):
            yield path


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
    if (
        basis == CATALOG
        and isinstance(per, dict)
        and per.get("variable") in list_entries
    ):
        count = list_entries[per["variable"]]
        each = float(per.get("minutes", 0))
        measured_with = int(per.get("entries", 0))
        minutes = max(0.0, (minutes - each * measured_with) + each * count)
        basis = PER_ENTRY
    elif basis == CATALOG:
        minutes, basis = _repriced(minutes, list_entries, measured_entries)
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


def downloads_required(expanded, base_dir, workflow_dir, cache_dir, lookup_sizes):
    """The hub repos and checkpoint URLs the run would fetch before its
    first step: every `model_name` in the expanded definition (and in each
    composed child) that `scan_models` does not find, plus every
    `from_single_file` that is a URL. Sizes come from the hub when asked
    and are None whenever it does not answer - an offline box is a state,
    not an error, so nothing here raises or logs above debug.
    """
    names = []
    urls = []
    _collect_sources(expanded, names, urls)
    for path in _sub_workflow_paths(expanded):
        raw = read_sub_workflow(path, base_dir, workflow_dir)
        if raw is None:
            continue
        try:
            _collect_sources(json.loads(raw), names, urls)
        except ValueError:
            continue
    present = {repo.get("repo_id") for repo in scan_models(cache_dir).get("repos", [])}
    required = []
    for name in names:
        # A name not shaped like a hub id is a local checkout - decided by
        # shape, never by touching the disk: the name came from the request
        # body, and a free pre-flight must not be a directory-existence oracle
        if name in present or not _is_repo_id(name):
            continue
        required.append({"repo": name, "gb": _size_gb(name) if lookup_sizes else None})
    for url in urls:
        required.append({"repo": None, "url": url, "gb": None})
    return required


def _collect_sources(tree, names, urls):
    """Every from_pretrained source in a tree, first-seen order, deduplicated.

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
            single = source.get(SINGLE_FILE_KEY)
            if isinstance(single, str) and _is_url(single) and single not in urls:
                urls.append(single)
        for value in tree.values():
            _collect_sources(value, names, urls)
    elif isinstance(tree, list):
        for value in tree:
            _collect_sources(value, names, urls)


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


def _size_gb(name):
    """A repo's size in GiB to one decimal, or None when the hub does not
    say - unreachable, gated without a token, or a file with no size."""
    try:
        info = model_info(name, files_metadata=True, timeout=SIZE_LOOKUP_TIMEOUT)
        total = sum(s.size for s in (info.siblings or []) if getattr(s, "size", None))
    except Exception as e:
        logger.debug(f"No size for {name}: {e}")
        return None
    return round(total / GIB, 1) if total else None
