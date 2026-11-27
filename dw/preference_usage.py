"""Which catalog workflows reach each `preference:` key (#1762).

A workflow uses a key when one of its steps composes `preference:<key>` -
the step's `workflow.path` itself, or the default of the variable a
`variable:<name>` path names (what the UI's `preferenceKeys` reads,
`ui/src/lib/preferences.ts`) - or when it composes, through any chain, a
catalog workflow that does. Read off the definitions as written: a path a
run passes as an argument is not seen.

A composed child's path becomes a catalog name the way resolution looks
for it (`dw/library.py`, `resolve_sub_workflow`): a `builtin:` prefix and a
`.json` suffix are dropped, and a relative path is tried against the
composing workflow's own folder first, then as a name from the catalog
root. A path that is absolute, climbs out of the catalog, or names nothing
in the catalog is no edge.
"""

import json
import logging
import posixpath

from . import references

logger = logging.getLogger("dw")


def _definitions(library):
    """{name: definition} for every readable catalog entry; a file that is
    not a JSON object is skipped."""
    winners, _shadowed = library.entries()
    definitions = {}
    for name, root in winners.items():
        # The library's validator: contained in its root, a .json file
        path = library.path_in(root, name)
        if path is None:
            continue
        try:
            with open(path, "r") as file:
                definition = json.load(file)
        except (OSError, ValueError) as e:
            logger.debug(f"Preference usage skips unreadable workflow {name}: {e}")
            continue
        if isinstance(definition, dict):
            definitions[name] = definition
    return definitions


def _step_paths(definition):
    """Every composed path the definition's steps name, a `variable:` path
    read as its variable's default."""
    variables = definition.get("variables")
    variables = variables if isinstance(variables, dict) else {}
    for _index, _step, block, _source, _where in references.iter_steps(
        definition.get("steps"), key="workflow"
    ):
        path = block.get("path")
        name = references.ref_name(references.VARIABLE, path)
        if name is not None:
            path = variables.get(name)
        if isinstance(path, str) and path:
            yield path


def _catalog_name(path, parent, names):
    """The catalog name a composed path opens, or None."""
    builtin = references.ref_name(references.BUILTIN, path)
    if builtin is not None:
        path = builtin
    if path.endswith(".json"):
        path = path[: -len(".json")]
    if not path or path.startswith("/") or "\\" in path:
        return None
    folder = posixpath.dirname(parent)
    for candidate in (posixpath.join(folder, path), path):
        candidate = posixpath.normpath(candidate)
        if candidate.startswith("..") or candidate.startswith("/"):
            continue
        if candidate in names:
            return candidate
    return None


def preference_users(library):
    """{key: sorted workflow names} - every catalog workflow of `library`
    (a `LibraryPath` of workflows, as `dw.server.deps.sources_for` answers)
    that reaches `preference:<key>` directly or through composed catalog
    workflows. A cycle of compositions terminates."""
    definitions = _definitions(library)
    keys = {}
    children = {}
    for name, definition in definitions.items():
        keys[name] = set()
        children[name] = set()
        for path in _step_paths(definition):
            key = references.ref_name(references.PREFERENCE, path)
            if key is not None:
                if key:
                    keys[name].add(key)
                continue
            child = _catalog_name(path, name, definitions)
            if child is not None and child != name:
                children[name].add(child)

    users = {}
    for name in definitions:
        reached = set()
        seen = {name}
        stack = [name]
        while stack:
            current = stack.pop()
            reached |= keys[current]
            for child in children[current]:
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        for key in reached:
            users.setdefault(key, []).append(name)
    return {key: sorted(names) for key, names in sorted(users.items())}
