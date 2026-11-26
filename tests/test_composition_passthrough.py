"""A composing step passes its child's values through, not as literals (#1642).

A catalog template that runs another workflow as a step hands that child its
`arguments`. A literal there fixes the child's value, so a run's `name=value`
can't reach it, and that breaks the catalog's rule that a run-affecting value
is a variable the caller can override. The convention is in WORKFLOW_GUIDE's
*Composing a stored workflow*: pass the value through a parent variable whose
default equals the child's, and keep a literal for a value the composition
fixes on purpose.

This test walks every `workflow` step in `workflows/` and `dw/workflows/`,
resolves its child, and holds every literal argument for a variable the child
declares to the DELIBERATE table below, which gives each one its reason. An
argument the child declares no variable for is validate's warning
(`sub_workflow_argument_warnings`), not this test's. Inline workflows the
server builds (`dw/server/enhancers.py`) are out of scope.
"""

import json
import os

import pytest

from dw import references
from tests.test_examples import (
    BUILTIN_DIR,
    REPO_ROOT,
    WORKFLOWS_DIR,
    is_workflow_file,
    resolve_reference,
)
from tests.test_preference_defaults import DEFAULTS, catalog, ladder_names

# Each literal a catalog composition passes on purpose, keyed by
# (parent, step name, argument), with the reason it is fixed
DELIBERATE = {
    ("workflows/templates/sub-workflow.json", "logo", "lora"): (
        "a minimal composition demo: the LoRA it applies is the point of the example"
    ),
    (
        "workflows/templates/preferred/text-to-image.json",
        "main",
        "num_images_per_prompt",
    ): (
        "the template makes one image per run, whichever ladder target the server picks"
    ),
}

# The MiniMax portraits and boards draw through preference:text_to_image at
# the template's own size, one image each (#1722): a ladder has no single
# child default to pass through, and each portrait feeds one H3 reference
PORTRAIT_STEPS = {
    "workflows/templates/minimax/dialogue-short.json": (
        "draw_character_a",
        "draw_character_b",
    ),
    "workflows/templates/minimax/generated-subject-reference.json": ("draw_subject",),
    "workflows/templates/minimax/music-video.json": ("draw_singer",),
    "workflows/templates/minimax/storyboard.json": (
        "board_1_launch",
        "board_2_gutter",
        "board_3_shore",
    ),
}
PORTRAIT_REASONS = {
    "width": "#1722: the portrait is drawn at the template's own size, whichever target draws it",
    "height": "#1722: the portrait is drawn at the template's own size, whichever target draws it",
    "num_images_per_prompt": "#1722: one portrait per step feeds one H3 reference",
}
DELIBERATE.update(
    {
        (parent, step, name): reason
        for parent, steps in PORTRAIT_STEPS.items()
        for step in steps
        for name, reason in PORTRAIT_REASONS.items()
    }
)

# Strings that are references rather than values: the caller's variables, an
# earlier step's output, or media and prompts chosen by name. A `constant:` is
# a value read from a file the caller can't change, so it counts as a literal
REFERENCE_PREFIXES = (
    references.VARIABLE,
    references.PREVIOUS_RESULT,
    references.PROMPT,
    references.ASSET,
    references.OUTPUT,
    references.GATHER,
    references.ITEM,
)


def is_reference(value):
    """Whether an argument is a reference: a reference string, or a list or
    dict whose every leaf is one. An empty list or dict holds no reference."""
    if isinstance(value, str):
        return value.startswith(REFERENCE_PREFIXES)
    if isinstance(value, list):
        return bool(value) and all(is_reference(v) for v in value)
    if isinstance(value, dict):
        return bool(value) and all(is_reference(v) for v in value.values())
    return False


def catalog_files():
    """Every workflow JSON in the catalog and the packaged builtins, relative
    to the repository root."""
    found = []
    for top in (WORKFLOWS_DIR, BUILTIN_DIR):
        for root, _, files in os.walk(top):
            for name in files:
                path = os.path.join(root, name)
                if name.endswith(".json") and is_workflow_file(path):
                    found.append(os.path.relpath(path, REPO_ROOT))
    return sorted(found)


def composing_steps(definition):
    """(step name, workflow block) for every `workflow` step at any depth."""
    if isinstance(definition, dict):
        block = definition.get("workflow")
        if isinstance(block, dict) and isinstance(block.get("path"), str):
            yield definition.get("name"), block
        for value in definition.values():
            yield from composing_steps(value)
    elif isinstance(definition, list):
        for value in definition:
            yield from composing_steps(value)


def child_files(path, base_dir, variables):
    """The files a composing step's path names. A `variable:` path follows the
    parent's default; a `preference:` key is every target in its ladder,
    since any of them may be the child; a catalog name is looked up the way
    the engine does after the referencing file's own directory."""
    if path.startswith(references.VARIABLE):
        default = (variables or {}).get(path.removeprefix(references.VARIABLE))
        assert isinstance(default, str), f"'{path}' has no default to resolve"
        path = default
    if path.startswith(references.PREFERENCE):
        key = path.removeprefix(references.PREFERENCE)
        names = sorted(ladder_names(DEFAULTS, key))
        assert names, f"'{path}' names no ladder"
        return [catalog_path(name) for name in names]
    target = resolve_reference(path, base_dir)
    if not os.path.isfile(target):
        target = catalog_path(path)
    return [target]


def catalog_path(name):
    found = catalog().find(name)
    assert found, f"{name} does not resolve to a catalog workflow"
    return found[0]


def load(relative):
    with open(os.path.join(REPO_ROOT, relative), encoding="utf-8") as file:
        return json.load(file)


def literal_arguments(parent):
    """(step, argument, child) for every literal a parent passes for a
    variable its child declares."""
    path = os.path.join(REPO_ROOT, parent)
    definition = load(parent)
    for step, block in composing_steps(definition):
        children = child_files(
            block["path"], os.path.dirname(path), definition.get("variables")
        )
        for child in children:
            with open(child, encoding="utf-8") as file:
                declared = json.load(file).get("variables") or {}
            for name, value in (block.get("arguments") or {}).items():
                if name in declared and not is_reference(value):
                    yield step, name, os.path.relpath(child, REPO_ROOT)


COMPOSING = [parent for parent in catalog_files() if any(composing_steps(load(parent)))]


def test_the_catalog_composes():
    """Guard the guard: a walk that finds no composition checks nothing."""
    assert "workflows/templates/compose-workflows.json" in COMPOSING
    assert "workflows/templates/describe-and-regenerate.json" in COMPOSING


@pytest.mark.parametrize("parent", COMPOSING)
def test_literal_arguments_are_deliberate(parent):
    undeclared = sorted(
        {
            f"{step}.{name} (child {child})"
            for step, name, child in literal_arguments(parent)
            if (parent, step, name) not in DELIBERATE
        }
    )
    assert not undeclared, (
        f"{parent} passes literals for variables its child declares: "
        f"{', '.join(undeclared)}. Pass each through a parent variable whose "
        "default equals the child's, or add it to DELIBERATE with the reason "
        "the composition fixes it (WORKFLOW_GUIDE, Composing a stored workflow)"
    )


def test_every_deliberate_literal_is_still_passed():
    """A stale entry would excuse a literal nobody passes any more."""
    passed = {
        (parent, step, name)
        for parent in COMPOSING
        for step, name, _child in literal_arguments(parent)
    }
    assert not set(DELIBERATE) - passed, sorted(set(DELIBERATE) - passed)


@pytest.mark.parametrize(
    "value",
    [
        "variable:prompt",
        "previous_result:main",
        "prompt:hero",
        "asset:face.png",
        "output:abc/0.png",
        "gather:takes",
        "item:caption",
        ["variable:image"],
        {"image": "previous_result:still", "mask": "asset:mask.png"},
    ],
)
def test_references_are_not_literals(value):
    assert is_reference(value)


@pytest.mark.parametrize(
    "value",
    [1, 0.5, True, None, "cuda", "constant:x.json", [], {}, ["variable:a", 2]],
)
def test_values_are_literals(value):
    assert not is_reference(value)
