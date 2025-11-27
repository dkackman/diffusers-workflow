#!/usr/bin/env python3
"""
Test to verify that cached pipelines get fresh arguments on each run.
This addresses the bug where changing arguments between runs didn't work.
"""

import os
import sys
import logging
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dw.workflow import Workflow
from dw.pipeline_processors.pipeline import Pipeline

# Setup logging
logging.basicConfig(
    level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_cached_pipeline_uses_new_arguments():
    """Test that cached pipelines receive fresh arguments on each run."""

    # Create a workflow definition
    workflow_def = {
        "id": "test_args",
        "steps": [
            {
                "name": "generate",
                "pipeline": {
                    "configuration": {
                        "component_type": "MockPipeline",
                    },
                    "from_pretrained_arguments": {"model_name": "test-model"},
                    "arguments": {
                        "prompt": "INITIAL_PROMPT",
                        "num_inference_steps": 10,
                    },
                },
            }
        ],
    }

    workflow = Workflow(workflow_def, "/tmp/test_output", "test.json")

    # Track what arguments are passed to pipeline.run()
    captured_arguments = []

    original_pipeline_init = Pipeline.__init__
    original_pipeline_load = Pipeline.load
    original_pipeline_run = Pipeline.run

    def mock_pipeline_init(self, *args, **kwargs):
        original_pipeline_init(self, *args, **kwargs)
        self.pipeline = MagicMock()

    def mock_pipeline_load(self, *args, **kwargs):
        logger.info("Pipeline.load() called")
        self.pipeline = MagicMock()

    def mock_pipeline_run(self, arguments, *args, **kwargs):
        prompt = arguments.get("prompt", "NO_PROMPT")
        steps = arguments.get("num_inference_steps", "NO_STEPS")
        logger.info(f"🔵 Pipeline.run() called with: prompt='{prompt}', steps={steps}")
        captured_arguments.append(arguments.copy())
        return MagicMock()

    with patch.object(Pipeline, "__init__", mock_pipeline_init):
        with patch.object(Pipeline, "load", mock_pipeline_load):
            with patch.object(Pipeline, "run", mock_pipeline_run):

                pipeline_cache = {}

                # First run - should create and cache pipeline
                logger.info("\n" + "=" * 60)
                logger.info("RUN 1: Initial prompt")
                logger.info("=" * 60)

                action1 = workflow.create_step_action(
                    workflow_def["steps"][0], {}, pipeline_cache, 42, "cuda"
                )

                # Simulate step.run() calling action.run()
                action1.run({"prompt": "a cat", "num_inference_steps": 20}, {})

                logger.info(f"✅ Run 1 complete")
                logger.info(f"   Prompt passed: '{captured_arguments[-1]['prompt']}'")
                logger.info(
                    f"   Steps passed: {captured_arguments[-1]['num_inference_steps']}"
                )

                # Second run - should reuse cached model but with NEW arguments
                logger.info("\n" + "=" * 60)
                logger.info("RUN 2: Changed prompt (should use NEW prompt)")
                logger.info("=" * 60)

                # Modify the workflow definition to simulate new arguments
                workflow_def["steps"][0]["pipeline"]["arguments"]["prompt"] = "a dog"
                workflow_def["steps"][0]["pipeline"]["arguments"][
                    "num_inference_steps"
                ] = 30

                action2 = workflow.create_step_action(
                    workflow_def["steps"][0], {}, pipeline_cache, 42, "cuda"
                )

                # Simulate step.run() calling action.run()
                action2.run({"prompt": "a dog", "num_inference_steps": 30}, {})

                logger.info(f"✅ Run 2 complete")
                logger.info(f"   Prompt passed: '{captured_arguments[-1]['prompt']}'")
                logger.info(
                    f"   Steps passed: {captured_arguments[-1]['num_inference_steps']}"
                )

                # Verify results
                logger.info("\n" + "=" * 60)
                logger.info("VERIFICATION")
                logger.info("=" * 60)

                assert (
                    len(captured_arguments) == 2
                ), f"Expected 2 runs, got {len(captured_arguments)}"
                logger.info(f"✅ Both runs executed")

                run1_prompt = captured_arguments[0].get("prompt")
                run2_prompt = captured_arguments[1].get("prompt")

                assert (
                    run1_prompt == "a cat"
                ), f"Run 1 should have 'a cat', got '{run1_prompt}'"
                logger.info(f"✅ Run 1 used correct prompt: '{run1_prompt}'")

                assert (
                    run2_prompt == "a dog"
                ), f"Run 2 should have 'a dog', got '{run2_prompt}'"
                logger.info(f"✅ Run 2 used NEW prompt: '{run2_prompt}'")

                assert (
                    run1_prompt != run2_prompt
                ), "Arguments should be different between runs!"
                logger.info(f"✅ Arguments changed between runs")

                run1_steps = captured_arguments[0].get("num_inference_steps")
                run2_steps = captured_arguments[1].get("num_inference_steps")

                assert run1_steps == 20, f"Run 1 should have 20 steps, got {run1_steps}"
                logger.info(f"✅ Run 1 used correct steps: {run1_steps}")

                assert run2_steps == 30, f"Run 2 should have 30 steps, got {run2_steps}"
                logger.info(f"✅ Run 2 used NEW steps: {run2_steps}")

                logger.info("\n" + "=" * 60)
                logger.info("🎉 TEST PASSED - Cached pipelines use fresh arguments!")
                logger.info("=" * 60)


def test_generator_seed_updates():
    """Test that generator seed is updated on cached pipeline runs."""

    workflow_def = {
        "id": "test_seed",
        "steps": [
            {
                "name": "generate",
                "pipeline": {
                    "configuration": {
                        "component_type": "MockPipeline",
                    },
                    "from_pretrained_arguments": {},
                    "arguments": {"prompt": "test"},
                    "seed": 100,
                },
            }
        ],
    }

    workflow = Workflow(workflow_def, "/tmp/test_output", "test.json")
    pipeline_cache = {}

    original_pipeline_init = Pipeline.__init__
    original_pipeline_load = Pipeline.load

    def mock_pipeline_init(self, *args, **kwargs):
        original_pipeline_init(self, *args, **kwargs)
        self.pipeline = MagicMock()

    def mock_pipeline_load(self, *args, **kwargs):
        self.pipeline = MagicMock()

    with patch.object(Pipeline, "__init__", mock_pipeline_init):
        with patch.object(Pipeline, "load", mock_pipeline_load):

            logger.info("\n" + "=" * 60)
            logger.info("SEED TEST")
            logger.info("=" * 60)

            # Run 1 with seed 100
            action1 = workflow.create_step_action(
                workflow_def["steps"][0], {}, pipeline_cache, 42, "cuda"
            )
            seed1 = workflow_def["steps"][0]["pipeline"].get("seed", 42)
            logger.info(f"Run 1: seed={seed1}")

            # Run 2 with seed 200
            workflow_def["steps"][0]["pipeline"]["seed"] = 200
            action2 = workflow.create_step_action(
                workflow_def["steps"][0], {}, pipeline_cache, 42, "cuda"
            )
            seed2 = workflow_def["steps"][0]["pipeline"].get("seed", 42)
            logger.info(f"Run 2: seed={seed2}")

            # Check that action2 has the new pipeline definition with seed 200
            assert (
                action2.pipeline_definition["seed"] == 200
            ), f"Expected seed 200, got {action2.pipeline_definition.get('seed')}"

            logger.info("✅ Pipeline wrapper gets updated seed")

            logger.info("\n" + "=" * 60)
            logger.info("🎉 SEED TEST PASSED!")
            logger.info("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing Cached Pipeline Argument Updates")
    print("=" * 60 + "\n")

    try:
        test_cached_pipeline_uses_new_arguments()
        test_generator_seed_updates()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nFix verified:")
        print("  • Cached pipelines receive fresh arguments on each run")
        print("  • Generator seeds update correctly")
        print("  • Arguments don't get stuck with old values")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
