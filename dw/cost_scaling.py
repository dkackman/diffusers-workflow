"""How far a run's arguments moved it from the defaults a curated `cost`
figure was measured at, and that figure scaled to match (#267, #593, #866).

A workflow's `cost` entry is measured at its variable defaults. The
variables its `cost_drivers` names are the ones the author says move that
figure: a generic diffusers driver (`num_inference_steps`, `width` and
`height`, `num_frames`, per list entry summed) scales it in proportion
(`scale_curated`), and so does any driver a `cost` entry names in its
`scales_with` (a Music 3 `audio_duration`, or that field summed over a
list driver's entries - #1199); any other declared driver that moved has no
rule to scale by, so the figure is kept and flagged. `dw/plan.py` prices the run
and applies the answer (`_scaled`); `dw/server/deps.py` scales an inherited
figure with the same `scale_curated`, so an inline copy and its template
quote one request alike. The basis labels an estimate carries live here
because the scaled figure is the first place one is chosen.

Nothing here knows a model: the drivers are the workflow's own
declaration, and the defaults its own `variables`.
"""

import json
import math

from . import references
from .presets import effective_variables

# The basis an estimate quotes its minutes on
UNKNOWN = "unknown"
CATALOG = "catalog"
PER_ENTRY = "per_entry"
DERIVED = "derived"
OTHER_DEVICE = "other_device"
OBSERVED = "observed"
INHERITED = "inherited"

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


LIST_ENTRY_COST_FIELDS = ("num_frames",)


def _list_entry_field_shifted(default_entries, effective_entries):
    """Whether a list driver's entries carry a numeric field (a per-shot
    `num_frames`, say) with a value none of the default entries had. The
    curated figure was measured over the default entries' values, so a value
    outside them has no matching bucket - the #267 rule applied inside a
    list driver (#593)."""
    measured = _numeric_fields(default_entries)
    for field, values in _numeric_fields(effective_entries).items():
        # Only a field that sets how long a shot renders moves its cost; a
        # start_frame or cut_frames is a position that moves with the count
        if (
            field in LIST_ENTRY_COST_FIELDS
            and field in measured
            and not values <= measured[field]
        ):
            return True
    return False


# The generic diffusers argument names a curated figure scales along when a
# run moves one off the default it was measured at (#866). Any other declared
# scalar driver (H3's weights_dtype, a music audio_duration) has no rule to
# scale by, so moving it keeps the figure and flags it (`_scaled`).
STEPS_DRIVER = "num_inference_steps"
FRAMES_DRIVER = "num_frames"
SCALABLE_DRIVERS = (STEPS_DRIVER, "width", "height", FRAMES_DRIVER)


def _positive(value):
    """A driver value as a positive finite number, or None."""
    number = _driver_comparable(value)
    if isinstance(number, float) and math.isfinite(number) and number > 0:
        return number
    return None


def _shown(number):
    return f"{round(number, 1):g}"


def scale_curated(minutes, defaults, effective):
    """`minutes` measured at `defaults`, scaled to a run at `effective`, as
    (minutes, axes), or None when a moved value is not a number (#866).

    Both are {driver: value} over `SCALABLE_DRIVERS`. Plain proportional
    scaling - steps/steps0 x (w.h)/(w0.h0) x frames/frames0 - with no fixed
    term for the model load and decode, so it over-quotes a little when a
    value rises and under-quotes when one falls. `axes` names each axis that
    moved, for a caller to read ("steps 30 vs 9"); empty when none did.
    """
    moved = {
        name
        for name in SCALABLE_DRIVERS
        if name in effective
        and _driver_comparable(effective[name])
        != _driver_comparable(defaults.get(name))
    }
    if not moved:
        return minutes, []
    values = {}
    for name in SCALABLE_DRIVERS:
        if name not in effective:
            continue
        pair = (_positive(defaults.get(name)), _positive(effective[name]))
        if None in pair:
            if name in moved:
                return None
            continue
        values[name] = pair
    factor = 1.0
    axes = []
    if STEPS_DRIVER in moved:
        was, now = values[STEPS_DRIVER]
        factor *= now / was
        axes.append(f"steps {_shown(now)} vs {_shown(was)}")
    canvas = [name for name in ("width", "height") if name in moved]
    for name in canvas:
        was, now = values[name]
        factor *= now / was
    if canvas and "width" in values and "height" in values:
        (w0, w), (h0, h) = values["width"], values["height"]
        axes.append(f"canvas {_shown(w)}x{_shown(h)} vs {_shown(w0)}x{_shown(h0)}")
    else:
        axes.extend(
            f"{name} {_shown(values[name][1])} vs {_shown(values[name][0])}"
            for name in canvas
        )
    if FRAMES_DRIVER in moved:
        was, now = values[FRAMES_DRIVER]
        factor *= now / was
        axes.append(f"frames {_shown(now)} vs {_shown(was)}")
    return minutes * factor, axes


SCALES_WITH_KEY = "scales_with"


def _declared_proportional(definition):
    """The names the workflow's `cost` entries say their figure is
    proportional to (`scales_with`), beyond the generic diffusers ones. The
    model's rule, declared by the workflow: the engine knows no music
    argument of its own."""
    names = []
    for entry in definition.get("cost") or []:
        declared = entry.get(SCALES_WITH_KEY) if isinstance(entry, dict) else None
        for name in declared if isinstance(declared, list) else ():
            if isinstance(name, str) and name not in names:
                names.append(name)
    return names


def _entry_frames(entries, field=FRAMES_DRIVER):
    """The per-entry `field` (`num_frames` by default) of a list's entries
    as numbers, or None when an entry carries none (or a non-numeric one)."""
    if not isinstance(entries, list) or not entries:
        return None
    frames = []
    for entry in entries:
        number = _positive(entry.get(field)) if isinstance(entry, dict) else None
        if number is None:
            return None
        frames.append(number)
    return frames


def _list_frames_scaling(
    name, default_entries, effective_entries, list_entries, field=FRAMES_DRIVER
):
    """A list driver's per-entry frames as (factor, axis) - (1.0, None) when
    they did not move - or None when they moved in a way that cannot be
    scaled (#593, #866).

    The frames a list renders are the sum over its entries. When the list's
    length is already priced (`_repriced`, a `per_entry` rate), only the
    mean per entry is left to scale, so the two multiply to the summed ratio
    rather than counting the length twice.
    """
    measured = _entry_frames(default_entries, field)
    running = _entry_frames(effective_entries, field)
    if measured is None or running is None:
        # Nothing to sum on one side: only a frame count outside the values
        # the default entries carried leaves no figure, as before (#593)
        if field == FRAMES_DRIVER and _list_entry_field_shifted(
            default_entries, effective_entries
        ):
            return None
        return 1.0, None
    was, now = sum(measured), sum(running)
    if was == now:
        return 1.0, None
    factor = now / was
    if name in list_entries:
        factor = (now / len(running)) / (was / len(measured))
        if math.isclose(factor, 1.0):
            return 1.0, None
    label = "frames" if field == FRAMES_DRIVER else field
    return factor, f"{label} {_shown(now)} vs {_shown(was)} over {name}"


# The declared drivers that set how many segments a chain runs; the count
# itself is the chain's own to tell (`chain_segment_count`), so a figure
# measured per segment scales by it
CHAIN_LENGTH_DRIVERS = ("segments", "duration_seconds")


def _resolved(value, variables):
    """`value` with a `variable:` reference read from `variables`."""
    name = references.ref_name(references.VARIABLE, value)
    return variables.get(name) if name is not None else value


def _chain_blocks(node):
    """Every (chain block, arguments) pair under `node`: a dict holding a
    `chain` beside its `arguments`."""
    if isinstance(node, dict):
        if isinstance(node.get("chain"), dict):
            yield node["chain"], node.get("arguments") or {}
        for value in node.values():
            yield from _chain_blocks(value)
    elif isinstance(node, list):
        for value in node:
            yield from _chain_blocks(value)


def _chain_segments(definition, variables):
    """The segments the workflow's chain runs at `variables`, or None when
    there is no single chain or its count cannot be told."""
    from .pipeline_processors.chain_config import chain_segment_count

    blocks = list(_chain_blocks(definition.get("steps")))
    if len(blocks) != 1:
        return None
    chain, arguments = blocks[0]
    chain = {key: _resolved(value, variables) for key, value in chain.items()}
    arguments = {key: _resolved(value, variables) for key, value in arguments.items()}
    return chain_segment_count(chain, arguments)


def _chain_retries(definition, variables):
    """The re-rolls a checked segment of the workflow's one chain may take
    at `variables`: its `retries` with `check` on, else 0."""
    blocks = list(_chain_blocks(definition.get("steps")))
    if len(blocks) != 1:
        return 0
    chain = blocks[0][0]
    check = _resolved(chain.get("check"), variables)
    retries = _resolved(chain.get("retries", 0), variables)
    if not (check is True or isinstance(check, dict)):
        return 0
    if isinstance(retries, bool) or not isinstance(retries, int):
        return 0
    return max(0, retries)


def _worst_case_retries(definition, variables):
    """(factor, axes, unscaled) for the worst case of a checked chain's
    re-rolls, as `chain_check` runs them: segment 1 is never checked, and
    each later segment may render 1 + retries times, so N segments cost up
    to 1 + (N - 1)(1 + retries) renders where the curated figure, measured
    without re-rolls, paid N. `retries` is flagged unscaled when the segment
    count cannot be told; with no re-rolls nothing moves. The quote cannot
    tell a texture failure (warn-only, never re-rolled) from a jump, look or
    cut one, so it still assumes the worst: every segment re-rolls."""
    retries = _chain_retries(definition, variables)
    segments = _chain_segments(definition, variables) if retries else None
    if not retries:
        return 1.0, [], []
    if not segments:
        return 1.0, [], ["retries"]
    renders = 1 + (segments - 1) * (1 + retries)
    return (
        renders / segments,
        [
            f"worst case {retries} re-roll(s) of each of {segments - 1} checked "
            f"segment(s): {renders} renders for {segments}"
        ],
        [],
    )


def _declared_axis_scaling(name, default_value, value, list_entries, proportional):
    """A list driver's summed fields, or a `scales_with` scalar, as
    (factor, axes), or None when one moved in a way nothing can scale.

    A list sums its entries' `num_frames` and every `scales_with` field
    (`_list_frames_scaling`); a scalar `scales_with` driver scales by its
    own ratio to the default (#1199)."""
    if not isinstance(default_value, list):
        if _driver_comparable(value) == _driver_comparable(default_value):
            return 1.0, []
        was, now = _positive(default_value), _positive(value)
        if was is None or now is None:
            return None
        return now / was, [f"{name} {_shown(now)} vs {_shown(was)}"]
    factor = 1.0
    axes = []
    fields = [FRAMES_DRIVER] + [f for f in proportional if f != FRAMES_DRIVER]
    for field in fields:
        summed = _list_frames_scaling(name, default_value, value, list_entries, field)
        if summed is None:
            return None
        factor *= summed[0]
        if summed[1]:
            axes.append(summed[1])
    return factor, axes


def _driver_scaling(definition, expanded, list_entries, baseline=None):
    """How far the declared `cost_drivers` moved the run away from the
    defaults the curated `cost` was measured at, as (factor, axes, unscaled),
    or None when a scalable one moved to a value that is not a number, or a
    list's frames moved in a way nothing can scale (#267, #866).

    A list driver's length is `_repriced`'s to price; here only its entries'
    `num_frames` are. A scalar one in `SCALABLE_DRIVERS` scales in
    proportion. Any other scalar driver that moved has no rule to scale by,
    for every template: its name goes in `unscaled`, and the figure is kept
    but flagged (#866, Q4).

    `baseline` replaces the curated defaults as the point the movement is
    measured from (an observed run's own variables, #1236). In that mode
    `_worst_case_retries` is skipped: the re-rolls are in both runs.
    """
    # the defaults the curated cost was measured at, the default preset
    # bundle among them
    defaults = baseline if baseline is not None else effective_variables(definition)
    effective = expanded.get("variables") or {}
    factor = 1.0
    axes = []
    at_default = {}
    running = {}
    unscaled = []
    chain_drivers = []
    proportional = _declared_proportional(definition)
    for name in _declared_drivers(definition):
        default_value = defaults.get(name)
        if name in CHAIN_LENGTH_DRIVERS:
            chain_drivers.append(name)
            continue
        if isinstance(default_value, list) or name in proportional:
            moved = _declared_axis_scaling(
                name, default_value, effective.get(name), list_entries, proportional
            )
            if moved is None:
                return None
            factor *= moved[0]
            axes.extend(moved[1])
            continue
        if name in list_entries:
            continue
        if name in SCALABLE_DRIVERS:
            at_default[name] = default_value
            running[name] = effective.get(name)
        elif _driver_comparable(effective.get(name)) != _driver_comparable(
            default_value
        ):
            unscaled.append(name)
    scaled = scale_curated(factor, at_default, running)
    if scaled is None:
        return None
    factor, scaled_axes = scaled
    scaled_axes = scaled_axes + axes
    if chain_drivers:
        was = _chain_segments(definition, defaults)
        now = _chain_segments(definition, effective)
        if was and now:
            if was != now:
                factor *= now / was
                scaled_axes.append(f"segments {now} vs {was}")
        elif any(
            _driver_comparable(effective.get(name))
            != _driver_comparable(defaults.get(name))
            for name in chain_drivers
        ):
            unscaled.extend(chain_drivers)
    if baseline is not None:
        # A ratio between two runs: the re-rolls are in both
        return factor, scaled_axes, unscaled
    worst, worst_axes, worst_unscaled = _worst_case_retries(definition, effective)
    return factor * worst, scaled_axes + worst_axes, unscaled + worst_unscaled


FIXED_MINUTES_KEY = "fixed_minutes"


def fixed_minutes(cost, device):
    """The part of the `cost` entry a run is quoted from that does not scale
    (`fixed_minutes`: model load, encode, decode - #1224), 0.0 when the entry
    declares none. The entry is chosen as `dw/plan.py`'s `_price` chooses:
    the device's own, else the first."""
    entries = [entry for entry in (cost or []) if isinstance(entry, dict)]
    if not entries:
        return 0.0
    chosen = next((e for e in entries if e.get("device") == device), entries[0])
    value = chosen.get(FIXED_MINUTES_KEY)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value) if math.isfinite(value) and value > 0 else 0.0


def _scaled(price, scaling, fixed=0.0):
    """`price` scaled by a `_driver_scaling` answer (`fixed` minutes of it,
    the entry's `fixed_minutes`, stay as they are): `derived` once any axis
    moved (another device's figure stays `other_device`, as `_price` keeps it
    for a resized list - #589), with `scaled` saying from what. A driver with
    no rule to scale by that moved leaves the basis alone, and is named in
    `differs` with `low_confidence` set: the figure was measured at another
    setting of it (#866)."""
    if scaling is None:
        return {"minutes": None, "basis": UNKNOWN, "measured_on": None}
    factor, axes, unscaled = scaling
    if price["minutes"] is None:
        return price
    if unscaled:
        price = {**price, "differs": list(unscaled), "low_confidence": True}
    if not axes:
        return price
    basis = OTHER_DEVICE if price["basis"] == OTHER_DEVICE else DERIVED
    # The fixed share is paid once however far the drivers moved (#1224)
    fixed = min(fixed, price["minutes"])
    note = f" ({_shown(fixed)} min of it fixed)" if fixed else ""
    return {
        **price,
        "minutes": fixed + (price["minutes"] - fixed) * factor,
        "basis": basis,
        "scaled": f"scaled from the curated {_shown(price['minutes'])} min{note}: "
        + ", ".join(axes),
    }


def _entry_cost_field_shifted(definition, expanded):
    """Whether any declared list driver carries a per-entry cost field outside
    the values the default entries had."""
    # the defaults the curated cost was measured at, the default preset
    # bundle among them
    defaults = effective_variables(definition)
    effective = expanded.get("variables") or {}
    return any(
        isinstance(defaults.get(name), list)
        and _list_entry_field_shifted(defaults[name], effective.get(name))
        for name in _declared_drivers(definition)
    )
