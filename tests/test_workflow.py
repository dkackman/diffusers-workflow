import pytest
import torch
from unittest.mock import MagicMock
from dw.workflow import (
    Workflow,
    workflow_from_file,
    referenced_result_names,
    release_unreferenced_results,
)
from dw.pipeline_processors.pipeline import Pipeline
import os


def test_workflow_validation_valid(valid_workflow_json):
    workflow = Workflow(valid_workflow_json, "./output", "")
    workflow.validate()  # Should not raise exception


def test_workflow_validation_invalid(invalid_workflow_json):
    workflow = Workflow(invalid_workflow_json, "./output", "")
    with pytest.raises(Exception) as exc_info:
        workflow.validate()
    assert "Validation error" in str(exc_info.value)


def test_workflow_name(valid_workflow_json):
    workflow = Workflow(valid_workflow_json, "./output", "")
    assert workflow.name == "test_workflow"


def test_workflow_from_file(test_data_dir):
    workflow_path = os.path.join(test_data_dir, "workflows", "valid_workflow.json")
    workflow = workflow_from_file(workflow_path, "./output")
    assert isinstance(workflow, Workflow)


def test_workflow_variables_property(valid_workflow_json):
    workflow = Workflow(valid_workflow_json, "./output", "")
    assert "prompt" in workflow.variables
    assert workflow.variables["prompt"] == "test prompt"


def test_workflow_argument_template(valid_workflow_json):
    workflow = Workflow(valid_workflow_json, "./output", "")
    # Should return empty dict if no argument_template
    assert workflow.argument_template == {}


def test_workflow_security_validation():
    from dw.security import SecurityError

    # Test path traversal protection
    with pytest.raises(SecurityError):
        workflow_from_file("../../../etc/passwd", "./output")


class TestResultEviction:
    """Results no later step references are released after saving"""

    def test_references_are_found_wherever_they_nest(self):
        steps = [
            {
                "name": "later",
                "pipeline": {
                    "arguments": {
                        "image": "previous_result:gen",
                        "masks": [{"mask": "previous_result:segment.mask"}],
                    }
                },
            }
        ]
        assert referenced_result_names(steps) == {"gen", "segment.mask"}

    def test_unreferenced_results_are_released(self):
        results = {"gen": object(), "old": object()}
        release_unreferenced_results(results, {"gen"})
        assert list(results) == ["gen"]

    def test_property_references_keep_their_result(self):
        # 'segment.mask' must keep the result named 'segment' - and a step
        # literally named 'segment.mask' as well
        results = {"segment": object(), "segment.mask": object(), "old": object()}
        release_unreferenced_results(results, {"segment.mask"})
        assert sorted(results) == ["segment", "segment.mask"]

    def test_no_references_releases_everything(self):
        results = {"a": object(), "b": object()}
        release_unreferenced_results(results, set())
        assert results == {}


class TestSeedResolution:
    """Seeds resolve most-specific-first: pipeline > step > workflow"""

    def step_definition(self, configuration=None, pipeline_seed=None):
        pipeline = {"configuration": configuration or {}, "arguments": {}}
        if pipeline_seed is not None:
            pipeline["seed"] = pipeline_seed
        return {"name": "gen", "pipeline": pipeline}

    def cached_step_action(self, step_definition, seed, device="cpu"):
        """Run create_step_action down the cached-pipeline path"""
        workflow = Workflow({"id": "seeds", "steps": []}, "./output", "")
        cached = Pipeline(step_definition["pipeline"], seed, device, MagicMock())
        return workflow.create_step_action(
            step_definition, {}, {step_definition["name"]: cached}, seed, device
        )

    def test_generator_is_seeded_with_the_step_seed(self):
        action = self.cached_step_action(self.step_definition(), seed=222)
        assert action.argument_template["generator"].initial_seed() == 222

    def test_pipeline_seed_overrides_the_step_seed(self):
        action = self.cached_step_action(
            self.step_definition(pipeline_seed=333), seed=222
        )
        assert action.argument_template["generator"].initial_seed() == 333

    def test_no_generator_false_still_creates_a_generator(self):
        # no_generator is a boolean - an explicit false requests a generator
        action = self.cached_step_action(
            self.step_definition(configuration={"no_generator": False}), seed=1
        )
        assert "generator" in action.argument_template

    def test_no_generator_true_disables_the_generator(self):
        action = self.cached_step_action(
            self.step_definition(configuration={"no_generator": True}), seed=1
        )
        assert "generator" not in action.argument_template

    def test_cached_generator_lives_on_the_pipeline_device(self):
        # The pipeline's own device override wins over the workflow default,
        # matching the fresh-load path
        action = self.cached_step_action(
            self.step_definition(configuration={"device": "cpu"}),
            seed=1,
            device="cuda",
        )
        assert action.argument_template["generator"].device.type == "cpu"


class TestGlobalRngIsolation:
    """Workflow.run must not reseed the RNG the process may rely on"""

    def run_empty_workflow(self, definition):
        Workflow(definition, "./output", "").run({})

    def test_run_with_an_explicit_seed_leaves_global_rng_alone(self):
        torch.manual_seed(1234)
        state = torch.get_rng_state()
        self.run_empty_workflow({"id": "w", "seed": 42, "steps": []})
        assert torch.equal(state, torch.get_rng_state())

    def test_run_without_a_seed_leaves_global_rng_alone(self):
        torch.manual_seed(1234)
        state = torch.get_rng_state()
        self.run_empty_workflow({"id": "w", "steps": []})
        assert torch.equal(state, torch.get_rng_state())
