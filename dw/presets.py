"""`presets`: one variable selects a bundle of other variables' values.

A template exposes one quality dial - `tier`, say, with `draft`, `standard`
and `best` - that moves canvas, weights and steps together, while every
variable stays individually settable. The shape is
`{"<selector variable>": {"<value>": {"<variable>": <value>, ...}, ...}}`.

The bundle sets defaults and the caller's arguments land after it
(`apply_presets`, then `set_variables`), so an explicit argument always wins.
A bundle value goes through `set_variables` exactly as a caller's would -
name and string validation, type coercion - so a bundle is no way around
those checks.

Validate checks what the run will use (`dw/workflow_run.py`), so every
reader of a run's variables sees the bundle, the default one included when
no arguments are passed (or when a bad argument leaves them all unfolded):

- the fold (`Workflow._fold`), which the run, validation, the estimate, the
  fingerprint and the VRAM projection all read - the projection reads the
  expanded definition's already-folded variables, so it never re-applies;
- `effective_variables`, the one owner of "a written definition's variables
  for these arguments", for the readers that take the definition as written
  rather than folded: the constraint checks, the argument warnings, the
  host-RAM memory key, the VRAM-inheritance comparisons and the observed-cost
  buckets;
- `preset_errors`, the static check over *every* bundle, chosen or not, and
  the constraint checks (`dw/variable_constraints.py`), which report a bundle
  value outside a declared rule at `presets.<selector>.<value>.<variable>`.
"""

import copy

from .security import SecurityError
from .variables import set_variables

PRESETS_KEY = "presets"


def _presets(definition):
    presets = definition.get(PRESETS_KEY) if isinstance(definition, dict) else None
    return presets if isinstance(presets, dict) else {}


def selectors(definition):
    """The names of the preset selectors `definition` declares."""
    return set(_presets(definition))


def _chosen(selector, arguments, variables):
    """The bundle name a run selects: the caller's value, else the default."""
    if arguments and selector in arguments:
        return arguments[selector]
    return variables.get(selector)


def apply_presets(definition, arguments, variables, *, strict=True):
    """Set, on `variables`, the chosen bundle's values for every variable the
    caller did not pass. A selector naming no bundle sets nothing here;
    `preset_errors` is what refuses it.

    `strict` (the run) raises what `set_variables` raises for a bundle value
    that will not coerce; otherwise (validation, the readers of
    `effective_variables`) that one value is left at its declared default,
    since `preset_errors` reports it at its own path."""
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
        if strict:
            set_variables(values, variables)
            continue
        for name, value in values.items():
            try:
                set_variables({name: value}, variables)
            except (ValueError, TypeError, SecurityError):
                continue


def effective_variables(definition, arguments=None):
    """The variables a run of the written `definition` uses for `arguments`,
    as a new dict: the declared defaults, the chosen bundle's values over
    them, and the caller's `arguments` over those.

    Never raises and never mutates `definition`. The arguments are laid over
    as given, uncoerced - the shallow merge every reader made before presets
    existed, so a bad argument is still there for the reader that reports
    it. For a definition that is already folded (an expanded one), read its
    `variables` instead: applying the bundle again would undo a rule's
    rounding."""
    declared = definition.get("variables") if isinstance(definition, dict) else None
    variables = dict(declared) if isinstance(declared, dict) else {}
    arguments = arguments if isinstance(arguments, dict) else {}
    apply_presets(definition, arguments, variables, strict=False)
    variables.update(arguments)
    return variables


def bundle_sources(definition, arguments=None):
    """{variable: "presets.<selector>.<value>"} for each variable the chosen
    bundle sets on a run with `arguments` - where a finding about that value
    belongs, since the author wrote it there."""
    arguments = arguments if isinstance(arguments, dict) else {}
    declared = definition.get("variables") if isinstance(definition, dict) else None
    declared = declared if isinstance(declared, dict) else {}
    sources = {}
    for selector, bundles in _presets(definition).items():
        if not isinstance(bundles, dict):
            continue
        chosen = _chosen(selector, arguments, declared)
        bundle = bundles.get(chosen) if isinstance(chosen, str) else None
        if not isinstance(bundle, dict):
            continue
        for name in bundle:
            if name not in arguments and name != selector and name in declared:
                sources[name] = f"{PRESETS_KEY}.{selector}.{chosen}"
    return sources


def coerced_bundles(definition):
    """(selector, value, {variable: coerced value}) for every bundle, each
    value coerced against the declared default as the run coerces it. A value
    that will not coerce, or names no declared variable, is left out -
    `preset_errors` reports those."""
    declared = definition.get("variables") if isinstance(definition, dict) else None
    declared = declared if isinstance(declared, dict) else {}
    for selector, bundles in _presets(definition).items():
        if not isinstance(bundles, dict):
            continue
        for value, bundle in bundles.items():
            if not isinstance(bundle, dict):
                continue
            coerced = {}
            for name, item in bundle.items():
                if name not in declared or name == selector:
                    continue
                scratch = {name: declared[name]}
                try:
                    set_variables({name: item}, scratch)
                except (ValueError, TypeError, SecurityError):
                    continue
                coerced[name] = scratch[name]
            yield selector, value, coerced


def preset_errors(definition, arguments=None):
    """What is wrong with the `presets` block, as [{path, message}]: a bundle
    key that is no declared variable, a bundle value that will not coerce to
    its declared default's type or fails string validation, a selector that
    is not a declared variable or whose default is not a string naming one of
    its bundles, and a caller value for the selector outside the bundle names
    (at `arguments.<selector>`). Every bundle is checked, not only the chosen
    one; a value outside a declared rule is `constraint_errors`' to report."""
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
        elif not isinstance(declared[selector], str):
            default = declared[selector]
            errors.append(
                {
                    "path": f"variables.{selector}",
                    "message": f"The preset selector '{selector}' defaults to "
                    f"{type(default).__name__} {default!r}, but bundle names "
                    "are JSON object keys and so always strings - it can never "
                    f"select one. Make its default a string, one of: {listed}",
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
            for name, item in bundle.items():
                path = f"{PRESETS_KEY}.{selector}.{value}.{name}"
                if name not in declared:
                    errors.append(
                        {
                            "path": path,
                            "message": f"'{name}' is not a declared "
                            f"variable; declared: {names}",
                        }
                    )
                    continue
                # Coerced against a copy, one value at a time, the way
                # argument_errors checks a caller's - so every bundle is
                # checked, chosen or not, and a bad one is a finding rather
                # than the ValueError the run's set_variables would raise
                try:
                    set_variables({name: item}, copy.deepcopy(declared))
                except (ValueError, TypeError, SecurityError) as e:
                    errors.append({"path": path, "message": str(e)})
    return errors
