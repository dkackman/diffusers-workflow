"""What a `cost` block prices a run at: the workflow's own, and each composed
child's - the curated half of the plan's estimate (`dw/plan.py` `estimate`,
which sets this box's observed history, read here too, against it and rolls
the two up).

`_price` reads one cost list for the serving device and re-prices it for the
caller's lists (`list_entries`, `_repriced`); `_own_price` and
`_child_catalog_price` then scale it along the cost drivers that moved
(`dw/cost_scaling.py`). `_ChildTotals` is what the children add up to, folded
in one child at a time by `_add_children`.

`_observed` reads this box's history as an estimate, `_frames_rescaled`
rescales a bucket to the frames a run renders, and `_with_worst_case` quotes a
checked chain's re-rolls on it. Nothing here knows a model, and this module
never imports `dw.plan` back.
"""

import json
import logging
import math

from . import references
from .cost_scaling import (
    CATALOG,
    DERIVED,
    INHERITED,
    OBSERVED,
    OTHER_DEVICE,
    PER_ENTRY,
    UNKNOWN,
    LIST_ENTRY_COST_FIELDS,
    _choose_entry,
    _declared_drivers,
    _driver_comparable,
    _driver_scaling,
    _entry_frames,
    _scaled,
    _shown,
    fixed_minutes,
    fixed_share,
    observed_render_scaling,
    words_check_term,
    with_words_check,
)
from .for_each import MEMBER_SEPARATOR
from .presets import effective_variables
from .realize import read_sub_workflow

logger = logging.getLogger("dw")


# An observed block's median of its cold runs' summed per-entry `num_frames`:
# the frames its cold figure was measured at (`dw/server/observed_cost.py`)
ENTRY_FRAMES_KEY = "entry_frames"


def list_entry_frames(definition, variables):
    """The frames a run's declared list drivers render: each entry's
    `num_frames` summed over every list driver whose entries all carry one,
    or None when none does. What an observed bucket's runs are rescaled by
    when a shot's frames moved off the defaults (#1696): the bucket matches
    on the list's length, not on what each entry renders."""
    total, found = 0.0, False
    for name in _declared_drivers(definition):
        frames = _entry_frames((variables or {}).get(name))
        if frames is not None:
            total += sum(frames)
            found = True
    return total if found else None


FOR_EACH_KEY = "for_each"


def list_entries(definition, realized):
    """{variable: length} for every `for_each` that names a list variable,
    read from `realized`'s folded variables - a literal list is not an
    argument and is not listed.

    Passing `{"variables": effective_variables(definition)}` as `realized`
    answers the lengths the workflow's *stored defaults* carry (its default
    preset bundle included), which is the list a catalog `cost`
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


def _own_price(
    definition, expanded, list_entries, device, measured_entries, base_dir=None
):
    """The workflow's own price, scaled along the cost drivers that moved."""
    own = _price(definition.get("cost"), device, list_entries, measured_entries or {})
    # `derived` and `per_entry` are re-priced by entry count alone, so a
    # per-entry num_frames is scaled as well (#796)
    if own["basis"] in (CATALOG, OTHER_DEVICE, DERIVED, PER_ENTRY):
        # _repriced only re-prices a for_each list's length, so a moved scalar
        # cost_driver (H3's num_inference_steps, say) is scaled here rather
        # than quoted at the figure measured for the defaults (#267, #866)
        own = _scaled(
            own,
            _driver_scaling(definition, expanded, list_entries, base_dir=base_dir),
            fixed_minutes(_choose_entry(definition.get("cost"), device)[0]),
            _repricing(
                definition.get("cost"), device, list_entries, measured_entries or {}
            ),
        )
        # A words-checked chain's Whisper passes, on top (#1262)
        own = with_words_check(own, definition, expanded)
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
        child_measured_entries = list_entries(
            child_definition, {"variables": effective_variables(child_definition)}
        )
        # The composing step's own `arguments` are what the child
        # actually runs with - folded over its declared defaults (and the
        # preset bundle they choose) the same way a caller's arguments are,
        # since `expanded` has already substituted them to concrete values
        # (#341)
        child_expanded = {
            "variables": effective_variables(child_definition, step_arguments)
        }
        child_list_entries = list_entries(child_definition, child_expanded)
    child = _price(child_cost, device, child_list_entries, child_measured_entries)
    if (
        child["basis"] in (CATALOG, OTHER_DEVICE, DERIVED, PER_ENTRY)
        and child_definition is not None
    ):
        # A cost_driver the composing step overrode (H3's num_frames at 345
        # against a default of 124, say) is the same #267 case one level
        # down - the child's figure was measured at its defaults, so it is
        # scaled to the value this step actually passes it (#341, #866)
        child = _scaled(
            child,
            _driver_scaling(child_definition, child_expanded, child_list_entries),
            fixed_minutes(_choose_entry(child_cost, device)[0]),
            _repricing(child_cost, device, child_list_entries, child_measured_entries),
        )
        child = with_words_check(child, child_definition, child_expanded)
    return child


class _IncludedSteps:
    """The composed steps whose time the quoted row's `minutes` already
    holds (`cost[].includes`: {measured child: [step names]}, #1773). Empty
    when the parent's own price was withheld: a figure that is not quoted
    holds nothing."""

    def __init__(self, definition, device, own_minutes, base_dir, workflow_dir):
        entry = None
        if own_minutes is not None:
            entry = _choose_entry(definition.get("cost"), device)[0]
        includes = entry.get("includes") if entry else None
        if not isinstance(includes, dict):
            includes = {}
        self._measured = {
            step: child
            for child, steps in includes.items()
            if isinstance(steps, list)
            for step in steps
            if isinstance(step, str)
        }
        self._dirs = (base_dir, workflow_dir)

    def measured_child(self, step):
        """The child the row was measured with at `step` (a `for_each`
        member answers to the step as written), or None."""
        if not isinstance(step, str):
            return None
        return self._measured.get(step.split(MEMBER_SEPARATOR)[0])

    def holds(self, path, measured):
        """Whether `path` resolves to the measured child, however either is
        spelled - the step then adds nothing to the row's own minutes."""
        found = read_sub_workflow(path, *self._dirs)
        return found is not None and found == read_sub_workflow(measured, *self._dirs)


def _replaced_child_note(step, path, measured):
    """The `scaled` clause for a covered step running another child: it is
    priced on top, and the row does not say what share to take off."""
    return (
        f"{step}: {path} added, but the minutes already hold {measured} at this "
        f"step - the quote comes out high by that covered share"
    )


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
        self.scaled = []
        self.differs = []

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
        if child.get("scaled"):
            self.scaled.append(f"{path}: {child['scaled']}")
        if not observed:
            self.differs.extend(child.get("differs") or [])
        if self.minutes is not None:
            self.minutes += child["minutes"]
        else:
            self.minutes = child["minutes"]


def _add_children(
    totals,
    composing_steps,
    included,
    own_minutes,
    device,
    base_dir,
    workflow_dir,
    observed_for_child,
    price_child,
):
    """Fold each composed child into `totals`, from `composing_steps`'
    (step name, path, arguments). A child this box has run is quoted from
    that history, else from its cost block, else by `price_child` from what
    it composes; one the parent's row already holds (`included`) is skipped."""
    for step, path, step_arguments in composing_steps:
        measured_child = included.measured_child(step)
        if measured_child and included.holds(path, measured_child):
            continue
        child_definition, child_cost = _read_child(path, base_dir, workflow_dir)
        child_observed = _child_observed(
            observed_for_child, path, child_definition, step_arguments, own_minutes
        )
        child = _observed(child_observed, device)
        is_observed = child is not None
        if not is_observed:
            child = _child_catalog_price(
                child_definition, child_cost, step_arguments, device
            )
            if child["minutes"] is None and price_child is not None:
                # A child with no figure of its own that composes others -
                # a saved workflow around a 'preference:' step, say - is
                # priced from what it composes, opened at the tier the chain
                # hands it (#1721)
                composed = price_child(path, step_arguments)
                if composed is not None and composed.get("minutes") is not None:
                    child = composed
                    if composed.get("partial"):
                        totals.partial = True
                        totals.unpriced.extend(composed.get("unpriced") or [])
        totals.add(path, child, is_observed)
        note = (
            None
            if is_observed
            else _unresolved_list_note(path, child_cost, step_arguments, device)
        )
        if note and child["minutes"] is not None:
            totals.differs.append(note)
        if measured_child and child["minutes"] is not None:
            totals.scaled.append(_replaced_child_note(step, path, measured_child))
    return totals


def _unresolved_list_note(path, child_cost, step_arguments, device):
    """Why a child's figure may under-quote, or None: its per_entry list is
    handed by an earlier step's output (`previous_result:` / `gather:`), so
    its length is unknown until the run and the child was priced at its own
    default entry count (#1222)."""
    chosen = None
    for entry in child_cost or []:
        if isinstance(entry, dict) and (
            chosen is None or entry.get("device") == device
        ):
            chosen = entry
            if entry.get("device") == device:
                break
    per = chosen.get("per_entry") if isinstance(chosen, dict) else None
    if not isinstance(per, dict):
        return None
    value = (step_arguments or {}).get(per.get("variable"))
    if not isinstance(value, str) or not (
        references.is_ref(references.PREVIOUS_RESULT, value)
        or references.is_ref(references.GATHER, value)
    ):
        return None
    return (
        f"{path}: '{per['variable']}' comes from an earlier step "
        f"({value}), so its length is unknown until the run; priced at the "
        f"child's default {per.get('entries')} entries - add about "
        f"{per.get('minutes')} min for each entry beyond that"
    )


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
    chosen, basis = _choose_entry(cost, device)
    if chosen is None:
        return {"minutes": None, "basis": UNKNOWN, "measured_on": None}
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


def _repricing(cost, device, list_entries, measured_entries):
    """How `_price` re-priced the chosen row for a resized list, for
    `_scaled` to name (#1773), or None when it did not: {minutes (the row's
    own), name, count, measured_with} plus `each` for a `per_entry` rate.
    Reads the same choice and the same rule `_price` applies."""
    chosen = _choose_entry(cost, device)[0]
    if chosen is None:
        return None
    minutes = float(chosen.get("minutes", 0))
    per = chosen.get("per_entry")
    if isinstance(per, dict) and per.get("variable") in list_entries:
        name = per["variable"]
        count, measured_with = list_entries[name], int(per.get("entries", 0))
        if count == measured_with:
            return None
        each = float(per.get("minutes", 0))
        return {
            "minutes": minutes,
            "name": name,
            "count": count,
            "measured_with": measured_with,
            "each": each,
        }
    changed = [
        (name, count, measured_entries[name])
        for name, count in list_entries.items()
        if name in measured_entries and count != measured_entries[name]
    ]
    if len(changed) != 1 or changed[0][2] <= 0:
        return None
    name, count, measured_with = changed[0]
    return {
        "minutes": minutes,
        "name": name,
        "count": count,
        "measured_with": measured_with,
    }


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


# The registrations' declared task rates (`register_command(cost=...)`,
# dw/tasks/registry.py): {"fixed_seconds", "per_input_second"} per command
def _task_rates():
    from .tasks.registry import RegistryTable

    return RegistryTable("cost")


def _task_steps(definition):
    """The steps when every one is a task step, else None - the shape
    `observed_cost._is_task_only` reads, here over the steps alone."""
    steps = (definition or {}).get("steps") or []
    if not steps or not all(
        isinstance(step, dict)
        and isinstance(step.get("task"), dict)
        and not any(
            key in step for key in ("pipeline", "pipeline_reference", "workflow")
        )
        for step in steps
    ):
        return None
    return steps


def undeclared_task_commands(definition):
    """The commands of a task-only workflow that declare no rate, sorted, or
    None when the workflow is not task-only (#1619)."""
    steps = _task_steps(definition)
    if steps is None:
        return None
    rates = _task_rates()
    return sorted(
        {
            str(step["task"].get("command", "unknown"))
            for step in steps
            if step["task"].get("command") not in rates
        }
    )


def _media_seconds(path, probes):
    """A probed file's length in seconds, or None, memoised in `probes`."""
    if path not in probes:
        from .media import probe_metadata

        try:
            info = probe_metadata(path) or {}
        except Exception:
            info = {}
        seconds = info.get("duration_seconds")
        if not seconds and info.get("frame_count") and info.get("fps"):
            seconds = info["frame_count"] / info["fps"]
        probes[path] = float(seconds) if seconds else None
    return probes[path]


def _step_input_seconds(command, arguments, base_dir, probes):
    """(seconds, deferred) for one task step's media inputs: the longest
    argument's length, a list argument summed over its entries (a join's
    `videos`), or None when none resolved. `deferred` is whether an input
    is an earlier step's output, unknown until the run."""
    from .locations import is_media_argument
    from .probe_paths import resolve_probe_path

    longest, deferred = None, False
    for name, value in (arguments or {}).items():
        if not is_media_argument(name, command):
            continue
        values = value if isinstance(value, list) else [value]
        total, found = 0.0, False
        for entry in values:
            if isinstance(entry, str) and references.is_ref(
                references.UNRESOLVED, entry
            ):
                deferred = True
                continue
            path = resolve_probe_path(entry, base_dir, f"{command} {name}")
            seconds = _media_seconds(path, probes) if path else None
            if seconds is not None:
                total += seconds
                found = True
        if found and (longest is None or total > longest):
            longest = total
    return longest, deferred


def _seconds_text(value):
    return f"{value:g}"


def task_rate_price(expanded, base_dir=None, probes=None):
    """A task-only workflow priced from its commands' declared rates (#1619),
    or None when it is not task-only or any command declares none.

    Each step costs its command's `fixed_seconds` plus `per_input_second`
    for every second of input media, probed at validate time. A step whose
    input is an earlier step's output (`previous_result:`) takes the longest
    length any step's inputs resolved to; one with no input that resolves is
    priced at its fixed seconds alone. Either is named in `differs`. The
    whole is an estimate - `derived` and always `low_confidence` - rounded
    up to a tenth of a minute.
    """
    steps = _task_steps(expanded)
    if steps is None:
        return None
    rates = _task_rates()
    if any(step["task"].get("command") not in rates for step in steps):
        return None
    probes = {} if probes is None else probes
    measured = []
    for step in steps:
        command = step["task"]["command"]
        arguments = step["task"].get("arguments")
        seconds, deferred = _step_input_seconds(
            command, arguments if isinstance(arguments, dict) else {}, base_dir, probes
        )
        measured.append((step, command, seconds, deferred))
    resolved = [seconds for _, _, seconds, _ in measured if seconds is not None]
    longest = max(resolved) if resolved else None
    total, parts, differs = 0.0, [], []
    for index, (step, command, seconds, deferred) in enumerate(measured):
        rate = rates[command]
        fixed, per = rate["fixed_seconds"], rate["per_input_second"]
        name = step.get("name") or f"step {index}"
        if seconds is None and deferred and longest is not None:
            seconds = longest
            differs.append(
                f"{name}: its input is an earlier step's output, so its length "
                f"is unknown until the run; priced at the longest input "
                f"resolved ({_seconds_text(round(longest, 1))} s)"
            )
        elif seconds is None:
            differs.append(
                f"{name}: no input media resolved to a length, so it is "
                f"priced at {command}'s fixed {_seconds_text(fixed)} s alone"
            )
        total += fixed + per * (seconds or 0.0)
        part = f"{command} {_seconds_text(fixed)} s"
        if seconds is not None:
            part += (
                f" + {_seconds_text(per)} s/input-s over "
                f"{_seconds_text(round(seconds, 1))} s"
            )
        parts.append(part)
    minutes = math.ceil(round(total / 60.0 * 10, 6)) / 10.0
    return {
        "minutes": minutes,
        "basis": DERIVED,
        "measured_on": None,
        "scaled": "task rates: " + "; ".join(parts),
        "differs": differs,
        "task_rates": True,
    }


# The first pass at a new shape (#1615): a cost row's one-time term, beside
# the curated price it is added to, but never part of it or of its scaling
FIRST_AT_SHAPE_KEY = "first_at_shape"
# How many unseen shapes the `scaled` clause names before "and N more"
SHAPES_SHOWN = 3


def _first_at_shape_minutes(value):
    """`first_at_shape.minutes` as a finite non-negative float, or None -
    the refusals `fixed_minutes` applies (a bool is not a number here)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def first_at_shape(entry):
    """One `cost` entry's first-pass-at-shape term (#1615) as (minutes, keys),
    or None when it declares none or declares one this cannot read. The
    minutes a run pays the first time this card runs a shape it has not run
    before (a kernel autotune); `keys` the scalar variables a shape is the
    tuple of. Read only by the plan's term: it is not part of `minutes`, so
    it never feeds `fixed_share` or the observed rescale (#1696)."""
    block = entry.get(FIRST_AT_SHAPE_KEY) if isinstance(entry, dict) else None
    if not isinstance(block, dict):
        return None
    minutes = _first_at_shape_minutes(block.get("minutes"))
    keys = block.get("keys")
    if (
        minutes is None
        or not isinstance(keys, list)
        or not keys
        or not all(isinstance(key, str) for key in keys)
    ):
        return None
    return minutes, list(keys)


def first_at_shape_errors(definition):
    """Every `cost[i].first_at_shape` the schema lets through that the term
    cannot use: a non-finite `minutes` (Python's JSON reader takes NaN and
    Infinity), or a `keys` name the workflow does not declare - which would
    otherwise read as an absent value and quietly join every shape."""
    errors = []
    declared = definition.get("variables") or {}
    for index, entry in enumerate(definition.get("cost") or []):
        block = entry.get(FIRST_AT_SHAPE_KEY) if isinstance(entry, dict) else None
        if not isinstance(block, dict):
            continue
        path = f"cost[{index}].{FIRST_AT_SHAPE_KEY}"
        if _first_at_shape_minutes(block.get("minutes")) is None:
            errors.append(
                {
                    "path": f"{path}.minutes",
                    "message": f"{FIRST_AT_SHAPE_KEY}.minutes must be a finite, "
                    f"non-negative number, not {block.get('minutes')!r}",
                }
            )
        keys = block.get("keys")
        for key in keys if isinstance(keys, list) else ():
            if isinstance(key, str) and key not in declared:
                errors.append(
                    {
                        "path": f"{path}.keys",
                        "message": f"{FIRST_AT_SHAPE_KEY}.keys names '{key}', "
                        f"which the workflow does not declare; declared: "
                        f"{', '.join(sorted(declared)) or '<none>'}",
                    }
                )
    return errors


def run_shapes(definition, variables, keys):
    """The shapes a run at `variables` (effective: presets resolved) renders,
    as a set of tuples: the `keys` variables' values, then - for each
    declared list driver whose entries carry a `num_frames`
    (`LIST_ENTRY_COST_FIELDS`) - one shape per distinct entry frame count.
    A workflow with no such list has one shape."""
    variables = variables or {}
    base = tuple(_driver_comparable(variables.get(key)) for key in keys)
    frames = set()
    for name in _declared_drivers(definition):
        for field in LIST_ENTRY_COST_FIELDS:
            counts = _entry_frames(variables.get(name), field)
            if counts is not None:
                frames.update(counts)
    if not frames:
        return {base}
    return {base + (count,) for count in frames}


def _shape_label(shape, keys):
    """One shape for a person: '226 frames, 1344x768, int8, sparse'."""
    values = dict(zip(keys, shape))
    parts = []
    if len(shape) > len(keys):
        parts.append(f"{_shown(shape[-1])} frames")
    shown = []
    for key in keys:
        if key == "height" and "width" in values:
            continue
        if key == "width" and "height" in values:
            shown.append(
                f"{_label_value(values['width'])}x{_label_value(values['height'])}"
            )
            continue
        shown.append(_label_value(values[key]))
    return ", ".join(parts + shown)


def _label_value(value):
    if isinstance(value, float):
        return _shown(value)
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        return value[1:-1]
    return str(value)


def _shape_order(shape):
    return [
        (0, value, "") if isinstance(value, float) else (1, 0.0, str(value))
        for value in shape
    ]


def first_at_shape_term(entry, definition, variables, seen):
    """(minutes, clause) the run adds for the shapes this card has not run,
    or (0.0, None): `first_at_shape.minutes` times the run's distinct shapes
    not in `seen`, and the `scaled` clause naming up to SHAPES_SHOWN of them.
    `seen` None (no history to judge by) adds nothing."""
    declared = first_at_shape(entry)
    if declared is None or seen is None:
        return 0.0, None
    minutes, keys = declared
    unseen = sorted(
        run_shapes(definition, variables, keys) - set(seen), key=_shape_order
    )
    if not unseen or minutes <= 0:
        return 0.0, None
    term = minutes * len(unseen)
    labels = [_shape_label(shape, keys) for shape in unseen[:SHAPES_SHOWN]]
    more = len(unseen) - SHAPES_SHOWN
    named = "; ".join(labels) + (f"; and {more} more" if more > 0 else "")
    noun = "shape" if len(unseen) == 1 else "shapes"
    clause = (
        f"plus {_shown(term)} min first pass at {len(unseen)} {noun} this card "
        f"hasn't run ({named})"
    )
    return term, clause


def with_first_at_shape(result, definition, arguments, device, seen_shapes):
    """`result` (the floored estimate, `dw/plan.py` `_checked_estimate`) plus
    the first pass at each shape this card has not run (`first_at_shape`,
    #1615): added once, after the floor, so the floor cannot swallow it and
    neither of `estimate`'s exits (observed or curated) adds it twice. It
    moves `minutes` and names itself in `scaled`; `warm_minutes` and
    `cached_minutes` keep their figures - a run kept warm or served by the
    cache has, by construction, run its shape - and `basis` and
    `low_confidence` are unchanged. `seen_shapes` None (no history to judge
    by), or a row without the key, adds nothing and the callable is never
    called."""
    if seen_shapes is None or result.get("minutes") is None:
        return result
    entry = _choose_entry(definition.get("cost"), device)[0]
    declared = first_at_shape(entry)
    if declared is None:
        return result
    try:
        seen = seen_shapes(declared[1])
        term, clause = first_at_shape_term(
            entry, definition, effective_variables(definition, arguments), seen
        )
    except Exception:
        logger.warning("No first-pass-at-shape term", exc_info=True)
        return result
    if not term:
        return result
    return {
        **result,
        "minutes": round(result["minutes"] + term, 1),
        "scaled": "; ".join(part for part in (result.get("scaled"), clause) if part),
    }


def _unique(names):
    return list(dict.fromkeys(names))


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
