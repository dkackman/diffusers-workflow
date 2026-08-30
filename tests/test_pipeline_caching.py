#!/usr/bin/env python3
"""
Test to verify that pipelines are properly cached and reused across multiple runs.
This test demonstrates GPU model persistence in the worker process.
"""

import os
import sys
import logging
import pytest
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dw.workflow import Workflow
from dw.pipeline_processors.pipeline import Pipeline
from dw.step import Step
from dw.tasks.model_cache import _cache as _model_cache, cached_model, clear_model_cache
from dw import get_device

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_pipeline_caching():
    """Test that pipelines are reused from cache instead of being reloaded."""

    # Create a minimal workflow definition
    workflow_def = {
        "id": "test_cache",
        "steps": [
            {
                "name": "generate",
                "pipeline": {
                    "configuration": {
                        "component_type": "MockPipeline",
                        "offload": "sequential",
                    },
                    "from_pretrained_arguments": {"model_name": "test-model"},
                    "arguments": {"prompt": "test prompt", "num_inference_steps": 1},
                },
            }
        ],
    }

    workflow = Workflow(workflow_def, "/tmp/test_output", "test.json")

    # Create a pipeline cache (simulating worker's loaded_pipelines)
    pipeline_cache = {}

    # Mock the Pipeline class to track load() calls
    load_call_count = 0
    loaded_models = {}  # Track loaded models by step name

    original_pipeline_init = Pipeline.__init__
    original_pipeline_load = Pipeline.load

    def mock_pipeline_init(self, *args, **kwargs):
        # Extract the pipeline argument before calling original init
        # In test: args = (pipeline_definition, default_seed, device_identifier, [pipeline])
        # pipeline is the 4th positional arg (index 3) if provided
        pipeline_arg = kwargs.get("pipeline", args[3] if len(args) > 3 else None)
        original_pipeline_init(self, *args, **kwargs)
        # Only create a mock if no pipeline was provided
        if pipeline_arg is None:
            self.pipeline = MagicMock()
            self.pipeline.to = MagicMock(return_value=self.pipeline)

    def mock_pipeline_load(self, *args, **kwargs):
        nonlocal load_call_count
        load_call_count += 1
        logger.info(f"🔴 Pipeline.load() called (count: {load_call_count})")
        # Create a unique mock model for this step
        self.pipeline = MagicMock(name=f"model_{load_call_count}")
        loaded_models[
            self.pipeline_definition.get("from_pretrained_arguments", {}).get(
                "model_name", "unknown"
            )
        ] = self.pipeline

    with patch.object(Pipeline, "__init__", mock_pipeline_init):
        with patch.object(Pipeline, "load", mock_pipeline_load):
            # First call - should create and load pipeline
            logger.info("\n" + "=" * 60)
            logger.info("FIRST RUN - Should load pipeline fresh")
            logger.info("=" * 60)

            action1 = workflow.create_step_action(
                workflow_def["steps"][0], {}, pipeline_cache, 42, get_device()
            )

            first_load_count = load_call_count
            logger.info(
                f"✅ First run: Pipeline loaded (load_call_count={first_load_count})"
            )
            logger.info(f"✅ Cache now has {len(pipeline_cache)} pipeline(s)")

            # Second call - should reuse cached pipeline
            logger.info("\n" + "=" * 60)
            logger.info("SECOND RUN - Should reuse cached pipeline")
            logger.info("=" * 60)

            action2 = workflow.create_step_action(
                workflow_def["steps"][0], {}, pipeline_cache, 42, get_device()
            )

            second_load_count = load_call_count
            logger.info(f"✅ Second run: load_call_count={second_load_count}")

            # Verify results
            logger.info("\n" + "=" * 60)
            logger.info("VERIFICATION")
            logger.info("=" * 60)

            assert (
                first_load_count == 1
            ), f"Expected 1 load on first run, got {first_load_count}"
            logger.info(f"✅ First run loaded exactly once")

            assert (
                second_load_count == 1
            ), f"Expected no additional loads on second run, got {second_load_count}"
            logger.info(f"✅ Second run reused cached pipeline (no reload)")

            # Note: We now create a new wrapper but reuse the underlying model
            assert (
                action1.pipeline is action2.pipeline
            ), "Expected same underlying pipeline model to be reused"
            logger.info(
                f"✅ Both runs reused the same underlying model (pipeline.pipeline)"
            )

            logger.info("\n" + "=" * 60)
            logger.info("🎉 TEST PASSED - Pipeline caching works correctly!")
            logger.info("=" * 60)


def test_pipeline_caching_different_steps():
    """Test that different steps create different cached pipelines."""

    workflow_def = {
        "id": "test_cache_multi",
        "steps": [
            {
                "name": "step1",
                "pipeline": {
                    "configuration": {"component_type": "MockPipeline"},
                    "from_pretrained_arguments": {"model_name": "model1"},
                    "arguments": {"prompt": "test"},
                },
            },
            {
                "name": "step2",
                "pipeline": {
                    "configuration": {"component_type": "MockPipeline"},
                    "from_pretrained_arguments": {"model_name": "model2"},
                    "arguments": {"prompt": "test"},
                },
            },
        ],
    }

    workflow = Workflow(workflow_def, "/tmp/test_output", "test.json")
    pipeline_cache = {}

    load_call_count = 0

    original_pipeline_init = Pipeline.__init__
    original_pipeline_load = Pipeline.load

    def mock_pipeline_init(self, *args, **kwargs):
        # Check if pipeline is being reused
        pipeline_arg = kwargs.get("pipeline", args[3] if len(args) > 3 else None)
        original_pipeline_init(self, *args, **kwargs)
        # Only create new mock if no pipeline was provided
        if pipeline_arg is None:
            self.pipeline = MagicMock()

    def mock_pipeline_load(self, *args, **kwargs):
        nonlocal load_call_count
        load_call_count += 1
        logger.info(f"🔴 Pipeline.load() called (count: {load_call_count})")
        self.pipeline = MagicMock(name=f"model_{load_call_count}")

    with patch.object(Pipeline, "__init__", mock_pipeline_init):
        with patch.object(Pipeline, "load", mock_pipeline_load):
            logger.info("\n" + "=" * 60)
            logger.info("MULTI-STEP TEST")
            logger.info("=" * 60)

            # Create step1 pipeline
            action1 = workflow.create_step_action(
                workflow_def["steps"][0], {}, pipeline_cache, 42, get_device()
            )
            logger.info(f"✅ Step1 created: load_count={load_call_count}")

            # Create step2 pipeline (should load fresh)
            action2 = workflow.create_step_action(
                workflow_def["steps"][1], {}, pipeline_cache, 42, get_device()
            )
            logger.info(f"✅ Step2 created: load_count={load_call_count}")

            # Reuse step1 pipeline (should NOT reload)
            action1_reuse = workflow.create_step_action(
                workflow_def["steps"][0], {}, pipeline_cache, 42, get_device()
            )
            logger.info(f"✅ Step1 reused: load_count={load_call_count}")

            assert (
                load_call_count == 2
            ), f"Expected 2 loads (one per step), got {load_call_count}"
            assert (
                action1.pipeline is action1_reuse.pipeline
            ), "Step1 underlying model should be reused from cache"
            assert (
                action1.pipeline is not action2.pipeline
            ), "Step1 and step2 should have different underlying models"

            logger.info("\n" + "=" * 60)
            logger.info("🎉 MULTI-STEP TEST PASSED!")
            logger.info("=" * 60)


def _release_workflow_def():
    def step(name, **extra):
        return {
            "name": name,
            **extra,
            "pipeline": {
                # Escaped with {} so realize_args leaves it a string - load is
                # mocked, the type is never used
                "configuration": {"component_type": "{MockPipeline}"},
                "from_pretrained_arguments": {"model_name": f"model-{name}"},
                "arguments": {"prompt": "test"},
            },
        }

    return {
        "id": "test_release",
        "steps": [step("generate", release_pipeline=True), step("keep")],
    }


def test_release_pipeline_evicts_after_step():
    """A step with release_pipeline drops its pipeline from the cache; others stay."""

    workflow = Workflow(_release_workflow_def(), "/tmp/test_output", "test.json")
    pipeline_cache = {}

    def mock_pipeline_load(self, shared_components):
        self.pipeline = MagicMock()

    with patch.object(Pipeline, "load", mock_pipeline_load):
        with patch.object(
            Step, "run", lambda self, *args, **kwargs: MagicMock(result_list=[])
        ):
            with patch("dw.workflow.empty_device_cache") as empty_cache:
                workflow.run({}, previous_pipelines=pipeline_cache)

    # The cache is keyed by pipeline identity, not step name - the run's
    # step->key map says which entry belongs to which step
    released_key = workflow._pipeline_keys_by_step["generate"]
    kept_key = workflow._pipeline_keys_by_step["keep"]
    assert released_key not in pipeline_cache, "released pipeline should be evicted"
    assert kept_key in pipeline_cache, "other pipelines stay cached"
    # the between-step cleanup returns cached blocks to the device
    assert empty_cache.call_count == len(_release_workflow_def()["steps"])


def _release_models_workflow_def(release):
    """A task step ahead of a pipeline step - the shape release_models exists for."""
    return {
        "id": "test_release_models",
        "steps": [
            {
                "name": "expand_prompt",
                **({"release_models": True} if release else {}),
                "task": {"command": "text_generation", "arguments": {"prompt": "hi"}},
            },
            {
                "name": "generate",
                "pipeline": {
                    "configuration": {"component_type": "{MockPipeline}"},
                    "from_pretrained_arguments": {"model_name": "model-generate"},
                    "arguments": {"prompt": "test"},
                },
            },
        ],
    }


@pytest.mark.parametrize("release,expect_cached", [(True, False), (False, True)])
def test_release_models_evicts_task_models_after_step(release, expect_cached):
    """release_models drops cached task models; without it they stay for the run."""

    workflow = Workflow(
        _release_models_workflow_def(release), "/tmp/test_output", "test.json"
    )

    def mock_pipeline_load(self, shared_components):
        self.pipeline = MagicMock()

    # Stand in for the model a task handler would have loaded on its device
    cached_model(("text_generation", "some-model", "cuda"), MagicMock)
    try:
        with patch.object(Pipeline, "load", mock_pipeline_load):
            with patch.object(
                Step, "run", lambda self, *args, **kwargs: MagicMock(result_list=[])
            ):
                with patch("dw.workflow.empty_device_cache"):
                    workflow.run({}, previous_pipelines={})

        assert bool(_model_cache) is expect_cached
    finally:
        clear_model_cache()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing Pipeline Caching Implementation")
    print("=" * 60 + "\n")

    try:
        test_pipeline_caching()
        test_pipeline_caching_different_steps()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nModels will now persist in GPU memory across workflow runs!")
        print(
            "This significantly improves performance by avoiding repeated model loading."
        )

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def test_cache_hit_republishes_shared_components():
    """A warm sharing step must refill the fresh shared_components dict, or a
    later reusing step that missed the cache finds nothing."""
    workflow = Workflow({"id": "share", "steps": []}, "/tmp/test_output", "t.json")

    sharing_def = {
        "name": "loader",
        "pipeline": {
            "configuration": {"component_type": "{Mock}"},
            "from_pretrained_arguments": {"model_name": "m"},
            "shared_components": ["transformer"],
            "arguments": {},
        },
    }
    from dw.workflow import pipeline_cache_key

    cached = Pipeline(sharing_def["pipeline"], 1, "cpu", MagicMock())
    cache = {pipeline_cache_key(sharing_def["pipeline"]): cached}

    shared = {}
    workflow.create_step_action(sharing_def, shared, cache, 1, "cpu")
    assert "transformer" in shared, "cache hit must republish shared components"
    assert shared["transformer"] is cached.pipeline.transformer


def test_redefined_step_evicts_prior_pipeline_before_loading():
    """The swap must never hold the old and new model stacks at once."""
    from dw.workflow import pipeline_cache_key

    old_def = {
        "configuration": {"component_type": "{Mock}"},
        "from_pretrained_arguments": {"model_name": "old-model"},
        "arguments": {},
    }
    new_step = {
        "name": "gen",
        "pipeline": {
            "configuration": {"component_type": "{Mock}"},
            "from_pretrained_arguments": {"model_name": "new-model"},
            "arguments": {},
        },
    }
    old_key = pipeline_cache_key(old_def)
    cache = {old_key: MagicMock()}

    workflow = Workflow({"id": "swap", "steps": []}, "/tmp/test_output", "t.json")
    workflow._prior_step_keys = {"gen": old_key}

    seen_at_load = {}

    def mock_load(self, shared_components):
        seen_at_load["old_still_cached"] = old_key in cache
        self.pipeline = MagicMock()

    with patch.object(Pipeline, "load", mock_load):
        workflow.create_step_action(new_step, {}, cache, 1, "cpu")

    assert (
        seen_at_load["old_still_cached"] is False
    ), "the redefined step's previous model must be evicted before load"
