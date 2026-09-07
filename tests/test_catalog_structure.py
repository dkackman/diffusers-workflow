"""The catalog's two trees, and the invariant each carries.

templates/ teaches a pattern and models/ records a hardware fact. The
distinction is only useful if it is legible from outside the file, so a
template must describe itself and a model config must name the template it
configures.
"""

import json
import os

import pytest

from tests.test_examples import REPO_ROOT, get_example_files

TEMPLATES = [f for f in get_example_files() if f.startswith("workflows/templates/")]
MODEL_CONFIGS = [f for f in get_example_files() if f.startswith("workflows/models/")]


def test_there_are_templates():
    assert TEMPLATES


@pytest.mark.parametrize("path", TEMPLATES)
def test_every_template_describes_itself(path):
    """A template is read before it is run - by a person choosing one and by an
    agent matching a request's shape. An undescribed template is invisible to
    both, whatever its filename says."""
    definition = json.load(open(os.path.join(REPO_ROOT, path), encoding="utf-8"))

    assert definition.get("description", "").strip(), f"{path} has no description"


def test_there_are_model_configs():
    assert MODEL_CONFIGS


@pytest.mark.parametrize("path", MODEL_CONFIGS)
def test_every_model_config_names_the_template_it_configures(path):
    """A model config is a tuned instance of a pattern. Without the pointer it
    is just another entry in the list, which is the problem this restructure
    exists to fix."""
    definition = json.load(open(os.path.join(REPO_ROOT, path), encoding="utf-8"))
    configures = definition.get("configures", "")

    assert configures, f"{path} has no 'configures'"
    target = os.path.join(REPO_ROOT, "workflows", f"{configures}.json")
    assert os.path.isfile(
        target
    ), f"{path} configures '{configures}', which is not a workflow ({target})"
