"""A cached pipeline reuses its loaded model but takes each run's own
arguments and seed. Guards the bug where changing arguments between runs of
a cached pipeline did nothing."""

import copy
from unittest.mock import MagicMock, patch

from dw.pipeline_processors.pipeline import Pipeline
from dw.workflow import Workflow


def pipeline_definition(**overrides):
    definition = {
        "configuration": {"component_type": "MockPipeline"},
        "from_pretrained_arguments": {"model_name": "test-model"},
        "arguments": {"prompt": "a cat", "num_inference_steps": 20},
    }
    definition.update(overrides)
    return definition


def fake_load(self, shared_components):
    self.pipeline = MagicMock()


def create_both(tmp_path, first, second):
    """Create the step twice against one pipeline cache, as two runs do."""
    workflow = Workflow({"id": "test_args", "steps": []}, str(tmp_path), "test.json")
    cache = {}
    with patch.object(Pipeline, "load", autospec=True, side_effect=fake_load) as load:
        action1 = workflow.create_step_action(
            {"name": "generate", "pipeline": first}, {}, cache, 42, "cpu"
        )
        action2 = workflow.create_step_action(
            {"name": "generate", "pipeline": second}, {}, cache, 42, "cpu"
        )
    return load, action1, action2


def test_cached_pipeline_uses_new_arguments(tmp_path):
    first = pipeline_definition()
    second = copy.deepcopy(first)
    second["arguments"] = {"prompt": "a dog", "num_inference_steps": 30}

    load, action1, action2 = create_both(tmp_path, first, second)

    # One load; the second run reuses the model under a fresh wrapper
    assert load.call_count == 1
    assert action2 is not action1
    assert action2.pipeline is action1.pipeline
    assert action2.argument_template["prompt"] == "a dog"
    assert action2.argument_template["num_inference_steps"] == 30


def test_generator_seed_updates(tmp_path):
    first = pipeline_definition(seed=100)
    second = copy.deepcopy(first)
    second["seed"] = 200

    load, action1, action2 = create_both(tmp_path, first, second)

    assert load.call_count == 1
    assert action2.pipeline is action1.pipeline
    assert action2.argument_template["generator"].initial_seed() == 200
