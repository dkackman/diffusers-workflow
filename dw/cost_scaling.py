"""How far a run's arguments moved it from the defaults a curated `cost`
figure was measured at, and that figure scaled to match (#267, #593, #866).

A workflow's `cost` entry is measured at its variable defaults. The
variables its `cost_drivers` names are the ones the author says move that
figure: a generic diffusers driver (`num_inference_steps`, `width` and
`height`, `num_frames`, per list entry summed) scales it in proportion
(`scale_curated`); any other declared driver that moved has no rule to
scale by, so the figure is kept and flagged. `dw/plan.py` prices the run
and applies the answer (`_scaled`); `dw/server/deps.py` scales an inherited
figure with the same `scale_curated`, so an inline copy and its template
quote one request alike. The basis labels an estimate carries live here
because the scaled figure is the first place one is chosen.

Nothing here knows a model: the drivers are the workflow's own
declaration, and the defaults its own `variables`.
"""

import json
import math

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


def _entry_frames(entries):
    """The per-entry `num_frames` of a list's entries as numbers, or None
    when an entry carries none (or a non-numeric one)."""
    if not isinstance(entries, list) or not entries:
        return None
    frames = []
    for entry in entries:
        number = (
            _positive(entry.get(FRAMES_DRIVER)) if isinstance(entry, dict) else None
        )
        if number is None:
            return None
        frames.append(number)
    return frames


def _list_frames_scaling(name, default_entries, effective_entries, list_entries):
    """A list driver's per-entry frames as (factor, axis) - (1.0, None) when
    they did not move - or None when they moved in a way that cannot be
    scaled (#593, #866).

    The frames a list renders are the sum over its entries. When the list's
    length is already priced (`_repriced`, a `per_entry` rate), only the
    mean per entry is left to scale, so the two multiply to the summed ratio
    rather than counting the length twice.
    """
    measured = _entry_frames(default_entries)
    running = _entry_frames(effective_entries)
    if measured is None or running is None:
        # Nothing to sum on one side: only a frame count outside the values
        # the default entries carried leaves no figure, as before (#593)
        if _list_entry_field_shifted(default_entries, effective_entries):
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
    return factor, f"frames {_shown(now)} vs {_shown(was)} over {name}"


def _driver_scaling(definition, expanded, list_entries):
    """How far the declared `cost_drivers` moved the run away from the
    defaults the curated `cost` was measured at, as (factor, axes, unscaled),
    or None when a scalable one moved to a value that is not a number, or a
    list's frames moved in a way nothing can scale (#267, #866).

    A list driver's length is `_repriced`'s to price; here only its entries'
    `num_frames` are. A scalar one in `SCALABLE_DRIVERS` scales in
    proportion. Any other scalar driver that moved has no rule to scale by,
    for every template: its name goes in `unscaled`, and the figure is kept
    but flagged (#866, Q4).
    """
    defaults = definition.get("variables") or {}
    effective = expanded.get("variables") or {}
    factor = 1.0
    axes = []
    at_default = {}
    running = {}
    unscaled = []
    for name in _declared_drivers(definition):
        default_value = defaults.get(name)
        if isinstance(default_value, list):
            frames = _list_frames_scaling(
                name, default_value, effective.get(name), list_entries
            )
            if frames is None:
                return None
            factor *= frames[0]
            if frames[1]:
                axes.append(frames[1])
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
    return scaled[0], scaled[1] + axes, unscaled


def _scaled(price, scaling):
    """`price` scaled by a `_driver_scaling` answer: `derived` once any axis
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
    return {
        **price,
        "minutes": price["minutes"] * factor,
        "basis": basis,
        "scaled": f"scaled from the curated {_shown(price['minutes'])} min: "
        + ", ".join(axes),
    }


def _entry_cost_field_shifted(definition, expanded):
    """Whether any declared list driver carries a per-entry cost field outside
    the values the default entries had."""
    defaults = definition.get("variables") or {}
    effective = expanded.get("variables") or {}
    return any(
        isinstance(defaults.get(name), list)
        and _list_entry_field_shifted(defaults[name], effective.get(name))
        for name in _declared_drivers(definition)
    )
