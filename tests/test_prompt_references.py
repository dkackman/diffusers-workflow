"""Every prompt: reference in the catalog names a prompt that exists.

The engine resolves a prompt: reference when the workflow runs, so a
reference to a prompt that was renamed or removed fails on the GPU rather
than in CI. This is the one reference class in the tree that nothing else
checks - sub-workflow paths, type names and constants all have their own
test in test_examples.py.
"""

import json
import os

import pytest

from dw.prompts import PROMPT_PREFIX, resolve_prompt_reference
from dw.server.app import collect_prompt_references
from tests.test_examples import REPO_ROOT, get_example_files

PROMPT_DIR = os.path.join(REPO_ROOT, "prompts")


def prompt_references(definition):
    """Every 'prompt:' reference a workflow makes, as written."""
    return [f"{PROMPT_PREFIX}{name}" for name in sorted(collect_prompt_references(definition))]


@pytest.mark.parametrize("example_file", get_example_files())
def test_every_prompt_reference_resolves(example_file):
    path = os.path.join(REPO_ROOT, example_file)
    with open(path, encoding="utf-8") as file:
        definition = json.load(file)

    for reference in prompt_references(definition):
        # resolve_prompt_reference raises rather than returning a missing path,
        # and a bare ValueError would not say which workflow carried the
        # reference - the only thing that makes the failure actionable
        try:
            resolve_prompt_reference(reference, prompt_dir=PROMPT_DIR)
        except ValueError as e:
            pytest.fail(f"{example_file} references '{reference}': {e}")


def test_the_catalog_actually_uses_prompt_references():
    """Guards the guard: if the walk stops finding references, the test above
    passes vacuously and would keep passing through the restructure."""
    found = []
    for example_file in get_example_files():
        with open(os.path.join(REPO_ROOT, example_file), encoding="utf-8") as file:
            found.extend(prompt_references(json.load(file)))

    assert len(set(found)) >= 10
