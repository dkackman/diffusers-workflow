"""The seed embedded in an image is the one it was rendered under (#1765)."""

import json
from unittest.mock import patch

import pytest
import torch

from dw.pipeline_processors.pipeline import Pipeline
from dw.step_cache import step_cache
from dw.workflow import Workflow
from dw.writers import read_embedded_metadata
from tests.test_events import FakePipeline
from dw import pipeline_ownership

EMBED = {"content_type": "image/png", "embed_metadata": True}


def generate(**extra):
    return {
        "name": "gen",
        "pipeline": {
            "configuration": {"component_type": "{Fake}"},
            "from_pretrained_arguments": {"model_name": "x/y"},
            "arguments": {"prompt": "a lighthouse"},
        },
        "result": {"content_type": "image/png"},
    } | extra


def embedded(tmp_path, definition, resident=None):
    def mock_load(self, shared_components):
        self.pipeline = FakePipeline()
        # As the real load does
        self.argument_template["generator"] = torch.Generator("cpu").manual_seed(
            self.pipeline_definition.get("seed", self.default_seed)
        )

    workflow = Workflow(
        definition, str(tmp_path / "out"), str(tmp_path / "parent.json")
    )
    with patch.object(Pipeline, "load", mock_load):
        with patch.object(pipeline_ownership, "empty_device_cache"):
            workflow.run({}, previous_pipelines={} if resident is None else resident)
    return read_embedded_metadata(workflow.manifest[-1]["files"][0])


@pytest.fixture(autouse=True)
def fresh_cache():
    step_cache.clear()
    yield
    step_cache.clear()


def composing(tmp_path, child_seed_argument):
    child = {
        "id": "child",
        "seed": "variable:seed",
        "variables": {"seed": None},
        "steps": [generate()],
    }
    # The engine only runs a sub-workflow it can resolve beside the parent
    (tmp_path / "child.json").write_text(json.dumps(child))
    return {
        "id": "parent",
        "seed": 9,
        "steps": [
            {
                "name": "lit",
                "workflow": {
                    "path": "child.json",
                    "arguments": {"seed": child_seed_argument},
                },
                "result": EMBED,
            }
        ],
    }


def test_a_childs_literal_seed_is_the_one_embedded(tmp_path):
    metadata = embedded(tmp_path, composing(tmp_path, 5))

    assert metadata["seed"] == 5
    # what reopening the parent definition has to pin
    assert metadata["workflow_seed"] == 9


def test_a_child_under_the_parents_seed_embeds_it_once(tmp_path):
    metadata = embedded(tmp_path, composing(tmp_path, 9))

    assert metadata["seed"] == 9
    assert "workflow_seed" not in metadata


def test_a_pipeline_seed_is_the_one_embedded(tmp_path):
    step = generate(result=EMBED)
    step["pipeline"]["seed"] = 7
    metadata = embedded(tmp_path, {"id": "flat", "seed": 9, "steps": [step]})

    assert metadata["seed"] == 7
    assert metadata["workflow_seed"] == 9


def test_a_resident_pipelines_seed_is_the_one_embedded(tmp_path):
    """The second job on a warm worker: the real generator, built for a
    pipeline already loaded rather than by the load mocked above."""
    step = generate(result=EMBED)
    step["pipeline"]["seed"] = 7
    definition = {"id": "warm", "seed": 9, "steps": [step]}
    resident = {}
    embedded(tmp_path / "first", definition, resident)
    step_cache.clear()

    with patch.object(Pipeline, "load", side_effect=AssertionError("reloaded")):
        workflow = Workflow(
            definition, str(tmp_path / "second"), str(tmp_path / "parent.json")
        )
        workflow.run({}, previous_pipelines=resident)
    metadata = read_embedded_metadata(workflow.manifest[-1]["files"][0])

    assert metadata["seed"] == 7
    assert metadata["workflow_seed"] == 9
