"""`presets`: one variable selects a bundle of other variables' values.

A template exposes one quality dial - `tier`, say, with `draft`, `standard`
and `best` - that moves canvas, weights and steps together, while every
variable stays individually settable. The shape is
`{"<selector variable>": {"<value>": {"<variable>": <value>, ...}, ...}}`.

The bundle sets defaults and the caller's arguments land after it
(`apply_presets`, then `set_variables`), so an explicit argument always wins.
A bundle value goes through `set_variables` exactly as a caller's would -
name and string validation, type coercion - so a bundle is no way around
those checks. Both places a run's arguments are folded (`Workflow._fold`,
which the estimate and the realized copy read too, and the argument fold in
`dw/validation.py`) call `apply_presets` first.
"""

from .variables import set_variables

PRESETS_KEY = "presets"


def _presets(definition):
    presets = definition.get(PRESETS_KEY) if isinstance(definition, dict) else None
    return presets if isinstance(presets, dict) else {}


def _chosen(selector, arguments, variables):
    """The bundle name a run selects: the caller's value, else the default."""
    if arguments and selector in arguments:
        return arguments[selector]
    return variables.get(selector)


def apply_presets(definition, arguments, variables):
    """Set, on `variables`, the chosen bundle's values for every variable the
    caller did not pass. A selector naming no bundle sets nothing here;
    `preset_errors` is what refuses it."""
    arguments = arguments if isinstance(arguments, dict) else {}
    if not isinstance(variables, dict):
        return
    for selector, bundles in _presets(definition).items():
        if not isinstance(bundles, dict):
            continue
        chosen = _chosen(selector, arguments, variables)
        bundle = bundles.get(chosen) if isinstance(chosen, str) else None
        if not isinstance(bundle, dict):
            continue
        values = {
            name: value
            for name, value in bundle.items()
            if name not in arguments and name != selector
        }
        set_variables(values, variables)


def preset_errors(definition, arguments=None):
    """What is wrong with the `presets` block, as [{path, message}]: a bundle
    key that is no declared variable, a selector that is not a declared
    variable or whose default is no bundle name, and a caller value for the
    selector outside the bundle names (at `arguments.<selector>`)."""
    presets = _presets(definition)
    if not presets:
        return []
    declared = definition.get("variables")
    declared = declared if isinstance(declared, dict) else {}
    names = ", ".join(sorted(declared)) or "<none>"
    errors = []
    for selector, bundles in presets.items():
        if not isinstance(bundles, dict):
            continue
        allowed = sorted(bundles)
        listed = ", ".join(allowed)
        if selector not in declared:
            errors.append(
                {
                    "path": f"presets.{selector}",
                    "message": f"The selector '{selector}' is not a declared "
                    f"variable; declared: {names}",
                }
            )
        elif declared[selector] not in bundles:
            errors.append(
                {
                    "path": f"variables.{selector}",
                    "message": f"The default of the preset selector "
                    f"'{selector}' must be one of its bundles: {listed}",
                }
            )
        if isinstance(arguments, dict) and selector in arguments:
            if arguments[selector] not in bundles:
                errors.append(
                    {
                        "path": f"arguments.{selector}",
                        "message": f"'{arguments[selector]}' is not a "
                        f"'{selector}' value; allowed: {listed}",
                    }
                )
        for value, bundle in bundles.items():
            if not isinstance(bundle, dict):
                continue
            for name in bundle:
                if name not in declared:
                    errors.append(
                        {
                            "path": f"presets.{selector}.{value}.{name}",
                            "message": f"'{name}' is not a declared "
                            f"variable; declared: {names}",
                        }
                    )
    return errors
