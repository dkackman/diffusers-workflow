"""What a `cost` block prices a run at: the workflow's own, and each composed
child's - the curated half of the plan's estimate (`dw/plan.py` `estimate`,
which sets this box's observed history against it and rolls the two up).

`_price` reads one cost list for the serving device and re-prices it for the
caller's lists (`list_entries`, `_repriced`); `_own_price` and
`_child_catalog_price` then scale it along the cost drivers that moved
(`dw/cost_scaling.py`). `_ChildTotals` is what the children add up to, folded
in one child at a time. Nothing here knows a model, and this module never
imports `dw.plan` back.
"""

import json

from . import references
from .cost_scaling import (
    CATALOG,
    DERIVED,
    OTHER_DEVICE,
    PER_ENTRY,
    UNKNOWN,
    _choose_entry,
    _declared_drivers,
    _driver_scaling,
    _entry_frames,
    _scaled,
    fixed_minutes,
    with_words_check,
)
from .presets import effective_variables
from .realize import read_sub_workflow


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
        )
        child = with_words_check(child, child_definition, child_expanded)
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
