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
