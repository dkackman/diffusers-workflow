"""Which catalog workflows reach each `preference:` key
(dw/preference_usage.py): directly, through a variable default, or through
any chain of composed catalog workflows."""

import json
import os

from dw.library import LibraryPath, LibraryRoot, WORKFLOWS_KIND
from dw.preference_usage import preference_users


def _library(tmp_path, workflows):
    root = tmp_path / "workflows"
    for name, definition in workflows.items():
        path = root / f"{name}.json"
        os.makedirs(path.parent, exist_ok=True)
        if isinstance(definition, str):
            path.write_text(definition)
        else:
            path.write_text(json.dumps(definition))
    return LibraryPath(WORKFLOWS_KIND, [LibraryRoot(root, "workspace", True)])


def _composing(path, variables=None):
    definition = {"id": "x", "steps": [{"name": "s", "workflow": {"path": path}}]}
    if variables:
        definition["variables"] = variables
    return definition


def test_direct_path_counts(tmp_path):
    library = _library(tmp_path, {"a": _composing("preference:k")})
    assert preference_users(library) == {"k": ["a"]}


def test_variable_default_counts(tmp_path):
    library = _library(
        tmp_path, {"a": _composing("variable:model", {"model": "preference:k"})}
    )
    assert preference_users(library) == {"k": ["a"]}


def test_composed_parent_counts(tmp_path):
    library = _library(
        tmp_path, {"a": _composing("preference:k"), "b": _composing("a")}
    )
    assert preference_users(library) == {"k": ["a", "b"]}


def test_child_path_forms_are_catalog_names(tmp_path):
    library = _library(
        tmp_path,
        {
            "models/a": _composing("preference:k"),
            "json": _composing("models/a.json"),
            "builtin": _composing("builtin:c.json"),
            "c": _composing("preference:j"),
            "models/sibling": _composing("./a.json"),
            "models/bare": _composing("a"),
            "outside": _composing("../a.json"),
        },
    )
    assert preference_users(library) == {
        "j": ["builtin", "c"],
        "k": ["json", "models/a", "models/bare", "models/sibling"],
    }


def test_cycle_terminates(tmp_path):
    library = _library(
        tmp_path,
        {
            "a": _composing("b"),
            "b": {
                "id": "b",
                "steps": [
                    {"name": "s", "workflow": {"path": "a"}},
                    {"name": "t", "workflow": {"path": "preference:k"}},
                ],
            },
            "self": _composing("self"),
        },
    )
    assert preference_users(library) == {"k": ["a", "b"]}


def test_unreadable_json_is_skipped(tmp_path):
    library = _library(
        tmp_path,
        {"a": _composing("preference:k"), "broken": "{not json", "list": "[1]"},
    )
    assert preference_users(library) == {"k": ["a"]}
