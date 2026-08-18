"""Every example workflow loads, validates, and points at files that exist.

The examples live in subfolders - flux/, tasks/, archive/, projects/ - so
discovery walks the tree rather than listing one directory.

The reference check guards what subfolders put at risk: a workflow's path to
another workflow resolves against the referencing file's own directory, so
moving one file without the other silently breaks it. Schema validation does
not follow those references, and nothing else does either until the workflow
is actually run.
"""

import json
import os

import pytest

from dw.arguments import (
    NON_TYPE_KEYS,
    fetch_constant,
    is_constant_reference,
    is_escaped,
    is_media_reference,
)
from dw.type_helpers import load_type_from_name
from dw.workflow import workflow_from_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")
BUILTIN_DIR = os.path.join(REPO_ROOT, "dw", "workflows")


def is_workflow_file(path):
    """Whether a JSON file under examples/ is a workflow rather than data.

    A project keeps its own data beside its workflows - a list of prompts its
    script feeds in, for instance - and those are JSON arrays. A file that does
    not parse counts as a workflow, so a broken one fails a test rather than
    quietly dropping out of the run.
    """
    try:
        with open(path, encoding="utf-8") as file:
            return isinstance(json.load(file), dict)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return True


def get_example_files():
    """Every workflow JSON under the examples tree, subfolders included.

    Paths are relative to the repository root - they are the test ids, and an
    absolute path makes for an unreadable one.
    """
    return sorted(
        os.path.relpath(path, REPO_ROOT)
        for path in (
            os.path.join(root, name)
            for root, _, files in os.walk(EXAMPLES_DIR)
            for name in files
            if name.endswith(".json")
        )
        if is_workflow_file(path)
    )


def workflow_references(definition):
    """The path of every sub-workflow a definition references, at any depth."""
    if isinstance(definition, dict):
        reference = definition.get("workflow")
        if isinstance(reference, dict) and isinstance(reference.get("path"), str):
            yield reference["path"]
        for value in definition.values():
            yield from workflow_references(value)
    elif isinstance(definition, list):
        for value in definition:
            yield from workflow_references(value)


def resolve_reference(path, base_dir):
    """Where a sub-workflow path points, or None when only a run can tell."""
    if path.startswith("variable:"):
        return None  # supplied at run time
    if path.startswith("builtin:"):
        return os.path.join(BUILTIN_DIR, path.removeprefix("builtin:"))
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


@pytest.mark.parametrize("example_file", get_example_files())
def test_example_workflow(example_file):
    """Test that each example workflow file can be loaded and validates"""
    path = os.path.join(REPO_ROOT, example_file)
    try:
        workflow = workflow_from_file(path, ".")
        workflow.validate()
    except Exception as e:
        pytest.fail(f"Example {example_file} failed validation: {str(e)}")


@pytest.mark.parametrize("example_file", get_example_files())
def test_example_workflow_references_resolve(example_file):
    """Test that every sub-workflow an example references exists"""
    path = os.path.join(REPO_ROOT, example_file)
    with open(path, encoding="utf-8") as file:
        definition = json.load(file)

    for reference in workflow_references(definition):
        target = resolve_reference(reference, os.path.dirname(path))
        if target is None:
            continue
        assert os.path.isfile(target), (
            f"{example_file} references '{reference}', which does not resolve "
            f"to a file ({os.path.normpath(target)})"
        )


def type_references(definition):
    """Every '*_type' value in a definition that names a type to load.

    Skips the keys that name a category rather than a type, the values escaped
    with {} for the same reason, the ones a run supplies, and the media
    references whose 'media_type' says what a file holds.
    """
    if isinstance(definition, dict):
        if is_media_reference(definition):
            return
        for key, value in definition.items():
            if (
                isinstance(value, str)
                and key.endswith("_type")
                and key not in NON_TYPE_KEYS
                and not is_escaped(value)
                and not value.startswith("variable:")
            ):
                yield key, value
            yield from type_references(value)
    elif isinstance(definition, list):
        for value in definition:
            yield from type_references(value)


@pytest.mark.parametrize("example_file", get_example_files())
def test_example_type_references_resolve(example_file):
    """Every type an example names is one the installed diffusers actually has.

    Schema validation cannot see this: a '*_type' is a string until a run loads
    it, so a class that moved, was renamed, or is only reachable by its full
    dotted path (LTX2LatentUpsamplerModel is not exported at the top level of
    diffusers) fails after the first model has already loaded.
    """
    path = os.path.join(REPO_ROOT, example_file)
    with open(path, encoding="utf-8") as file:
        definition = json.load(file)

    for key, name in type_references(definition):
        try:
            load_type_from_name(name)
        except Exception as error:
            pytest.fail(
                f"{example_file} names '{name}' as its '{key}', which does not "
                f"load: {type(error).__name__}: {error}"
            )


def constant_references(definition):
    """Every 'constant:' value in a definition, wherever it appears."""
    if isinstance(definition, dict):
        for value in definition.values():
            yield from constant_references(value)
    elif isinstance(definition, list):
        for value in definition:
            yield from constant_references(value)
    elif is_constant_reference(definition):
        yield definition


@pytest.mark.parametrize("example_file", get_example_files())
def test_example_constant_references_resolve(example_file):
    """Every constant an example names is one the installed library declares.

    The point of referencing a constant instead of copying it is that the
    library stays the source of truth, which only holds if a constant that
    moved or was renamed fails here rather than mid-run.
    """
    path = os.path.join(REPO_ROOT, example_file)
    with open(path, encoding="utf-8") as file:
        definition = json.load(file)

    for reference in constant_references(definition):
        try:
            fetch_constant(reference)
        except Exception as error:
            pytest.fail(
                f"{example_file} names '{reference}', which does not resolve: "
                f"{type(error).__name__}: {error}"
            )
