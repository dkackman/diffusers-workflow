"""The catalog's two trees, and the invariant each carries.

templates/ teaches a pattern and models/ records a hardware fact. The
distinction is only useful if it is legible from outside the file, so a
template must describe itself and a model config must name the template it
configures.
"""

import json
import os

import pytest

from tests.test_examples import REPO_ROOT

TEMPLATES_DIR = os.path.join(REPO_ROOT, "workflows", "templates")


def files_under(directory):
    if not os.path.isdir(directory):
        return []
    found = []
    for root, _, names in os.walk(directory):
        found.extend(
            os.path.join(root, name) for name in names if name.endswith(".json")
        )
    return sorted(found)


def test_there_are_templates():
    assert files_under(TEMPLATES_DIR)


@pytest.mark.parametrize("path", files_under(TEMPLATES_DIR))
def test_every_template_describes_itself(path):
    """A template is read before it is run - by a person choosing one and by an
    agent matching a request's shape. An undescribed template is invisible to
    both, whatever its filename says."""
    definition = json.load(open(path, encoding="utf-8"))

    assert definition.get("description", "").strip(), (
        f"{os.path.relpath(path, REPO_ROOT)} has no description"
    )


MODELS_DIR = os.path.join(REPO_ROOT, "workflows", "models")


def test_there_are_model_configs():
    assert files_under(MODELS_DIR)


@pytest.mark.parametrize("path", files_under(MODELS_DIR))
def test_every_model_config_names_the_template_it_configures(path):
    """A model config is a tuned instance of a pattern. Without the pointer it
    is just another entry in the list, which is the problem this restructure
    exists to fix."""
    definition = json.load(open(path, encoding="utf-8"))
    configures = definition.get("configures", "")

    assert configures, f"{os.path.relpath(path, REPO_ROOT)} has no 'configures'"
    target = os.path.join(REPO_ROOT, "workflows", f"{configures}.json")
    assert os.path.isfile(target), (
        f"{os.path.relpath(path, REPO_ROOT)} configures '{configures}', "
        f"which is not a workflow ({target})"
    )
