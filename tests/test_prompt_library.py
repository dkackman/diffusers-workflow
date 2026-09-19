"""Conventions the whole prompt library is held to.

`intended_model` is documented as informational, and it is - the engine
ignores it. What it is not is free: `list_prompts(intended_model=...)`
narrows by an exact match, so `minimax-music` beside `minimax-music3` is a
filter that returns half a family and looks like it returned all of it. The
sweep is the same shape `tests/test_observed_cost.py` uses for
`cost_drivers`: a value that buckets on nothing looks exactly like a value
that works.

An empty `intended_model` is allowed. Some stored text is not written for a
model at all - an enhancer's system prompt, a generic landscape - and
inventing a family for it would be worse than saying nothing.
"""

import glob
import json
import os

import pytest

from tests.test_examples import REPO_ROOT

PROMPT_DIR = os.path.join(REPO_ROOT, "prompts")

# One spelling per family. Every value here is either a key the enhancer
# presets preselect for (`intended_models` in dw/server/enhancers.py) or a
# model family the catalog ships templates for.
INTENDED_MODELS = frozenset(
    {
        "minimax-h3",
        "minimax-music3",
        "ltx-2.5",
        "z-image",
        "flux",
    }
)


def prompt_files():
    return sorted(glob.glob(os.path.join(PROMPT_DIR, "**", "*.json"), recursive=True))


@pytest.mark.parametrize(
    "path", prompt_files(), ids=lambda p: os.path.relpath(p, PROMPT_DIR)
)
def test_intended_model_is_one_of_the_known_families(path):
    with open(path) as file:
        definition = json.load(file)
    declared = definition.get("intended_model")
    if not declared:
        return
    assert declared in INTENDED_MODELS, (
        f"{os.path.relpath(path, REPO_ROOT)} declares intended_model "
        f"{declared!r}; list_prompts filters on an exact match, so a variant "
        f"spelling hides the prompt from the family it belongs to. "
        f"Known: {sorted(INTENDED_MODELS)}"
    )


def test_every_enhancer_preset_targets_a_known_family():
    """The presets and the library share one vocabulary, or the editor
    preselects a preset no prompt will ever match."""
    from dw.server.enhancers import PRESETS

    for key, preset in PRESETS.items():
        for model in preset["intended_models"]:
            assert model in INTENDED_MODELS, (
                f"enhancer preset {key!r} targets {model!r}, which no prompt "
                f"may declare"
            )
