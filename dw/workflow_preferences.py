"""The `preference:` side of a Workflow: each step whose `workflow.path` is a
'preference:' rewritten to the catalog workflow it resolves to, and what the
plan, the manifest and the realized workflow read back about those
resolutions.

`PreferenceResolution` is mixed into `dw.workflow.Workflow`, which owns the
state it works on - `_preferences`, `_preference_errors`, `_inherited_tier`,
`_effective_tier` - and `_open_preference_target`, the one piece that opens a
workflow file. The ladders, the fit and the pins are `dw/preferences.py`.

This module does not import `dw.workflow`.
"""

from . import references


class PreferenceResolution:
    """`Workflow`'s 'preference:' methods (see the module docstring)."""

    def resolve_preferences(self, definition, variables):
        """Rewrite every step whose `workflow.path` is a 'preference:' to
        the target it resolves to (dw/preferences.py), after substitution
        and for_each expansion, so every reader of the path - pricing,
        downloads, ownership, warnings, realization, validation and the run -
        follows the resolved child. Returns `definition`, edited in place.

        The step's arguments go over the target's own; a `tier` argument
        picks the rung and is dropped unless the target declares a `tier`
        variable itself. A refusal leaves the path as written and is kept
        for resolve_sub_workflow_path to raise. Only `workflow.path` is
        read: a 'preference:' string anywhere else stays text."""
        from . import preferences

        # Any folded value is the tier, an empty or non-string one included,
        # so it is refused below rather than quietly falling through to the
        # inherited or default tier (#1721)
        folded = (variables or {}).get(preferences.TIER_ARGUMENT)
        self._effective_tier = folded if folded is not None else self._inherited_tier
        steps = definition.get("steps") if isinstance(definition, dict) else None
        if not isinstance(steps, list):
            return definition
        session = None
        self._preference_errors = {}
        for step in steps:
            reference = step.get("workflow") if isinstance(step, dict) else None
            if not isinstance(reference, dict):
                continue
            path = reference.get("path")
            if not references.is_ref(references.PREFERENCE, path):
                self._drop_undeclared_tier(reference)
                continue
            if session is None:
                session = preferences.current_session()
            arguments = reference.get("arguments")
            arguments = dict(arguments) if isinstance(arguments, dict) else {}
            try:
                tier = arguments.get(preferences.TIER_ARGUMENT)
                if tier is None:
                    tier = self._effective_tier
                if tier is None:
                    tier = preferences.pinned_tier(session.pinned, path)
                if tier is None:
                    tier = session.preferences()["default_tier"]
                if not isinstance(tier, str):
                    raise preferences.PreferenceError(
                        f"Preference tier must be a name, not {tier!r}"
                    )
                preferences.check_tier(tier)
                pin = preferences.pin_key(path, tier)
                pinned = session.pinned.get(pin)
                resolution = preferences.resolve(
                    path,
                    tier,
                    session.facts(),
                    self._open_preference_target,
                    None if pinned else session.preferences(),
                    pinned=pinned,
                )
            except preferences.PreferenceError as e:
                # The first refusal for a path: a later step that resolves
                # the same path at another tier must not erase it
                self._preference_errors.setdefault(path, e)
                continue
            target_definition, _ = self._open_preference_target(resolution.target)
            merged = {**resolution.arguments, **arguments}
            declared = target_definition.get("variables") or {}
            if preferences.TIER_ARGUMENT not in declared:
                merged.pop(preferences.TIER_ARGUMENT, None)
            reference["path"] = resolution.target
            reference["arguments"] = merged
            self._preferences[pin] = resolution
            session.resolved[pin] = resolution.pin()
        return definition

    def _drop_undeclared_tier(self, reference):
        """A step's `tier` argument is the resolver's (design 6 of #1626):
        handed on only to a child that declares a `tier` variable. A
        `variable:` path that held a 'preference:' and was overridden with a
        workflow name reaches here with the step's pinned tier still in its
        arguments; a child with no `tier` would refuse it at run as an
        unknown variable (#1722). A child that cannot be opened is left
        alone, for sub_workflow_errors to report."""
        from . import preferences

        arguments = reference.get("arguments")
        if not isinstance(arguments, dict):
            return
        if preferences.TIER_ARGUMENT not in arguments:
            return
        path = reference.get("path")
        if not isinstance(path, str) or not path:
            return
        try:
            definition, _ = self._open_preference_target(path)
        except Exception:  # noqa: BLE001 - reported by validation, not here
            return
        if preferences.TIER_ARGUMENT not in (definition.get("variables") or {}):
            arguments = dict(arguments)
            arguments.pop(preferences.TIER_ARGUMENT)
            reference["arguments"] = arguments

    def preference_lines(self):
        """The plan's line for each resolution this workflow made."""
        return [resolution.line() for resolution in self._preferences.values()]

    def preference_pins(self):
        """Each resolution as the manifest pins it, by pin key."""
        return {key: resolution.pin() for key, resolution in self._preferences.items()}

    def preference_targets(self):
        """The catalog workflows this workflow's 'preference:' steps
        resolved to, each once."""
        return list(dict.fromkeys(r.target for r in self._preferences.values()))
