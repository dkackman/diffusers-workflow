#!/usr/bin/env python3
"""
Test to verify that pipelines are properly cached and reused across multiple runs.
This test demonstrates GPU model persistence in the worker process.
"""

import os
import sys
import logging
import pytest
from unittest.mock import patch, MagicMock

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

            assert first_load_count == 1, (
                f"Expected 1 load on first run, got {first_load_count}"
            )
            logger.info("✅ First run loaded exactly once")

            assert second_load_count == 1, (
                f"Expected no additional loads on second run, got {second_load_count}"
            )
            logger.info("✅ Second run reused cached pipeline (no reload)")

            # Note: We now create a new wrapper but reuse the underlying model
            assert action1.pipeline is action2.pipeline, (
                "Expected same underlying pipeline model to be reused"
            )
            logger.info(
                "✅ Both runs reused the same underlying model (pipeline.pipeline)"
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

            assert load_call_count == 2, (
                f"Expected 2 loads (one per step), got {load_call_count}"
            )
            assert action1.pipeline is action1_reuse.pipeline, (
                "Step1 underlying model should be reused from cache"
            )
            assert action1.pipeline is not action2.pipeline, (
                "Step1 and step2 should have different underlying models"
            )

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
            # Both steps save, so both run: a step that saves nothing and
            # which nothing reads is elided before the run (#122), and these
            # are release tests rather than elision ones
            "result": {"content_type": "image/png"},
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
    # the between-step cleanup returns cached blocks to the device - once per
    # step, plus once for the release itself, which reclaims where it drops
    # rather than waiting for the end of the step
    assert empty_cache.call_count == len(_release_workflow_def()["steps"]) + 1


def test_release_pipeline_happens_before_the_result_is_written():
    """A released pipeline is gone by the time the step writes its files.

    The write does not touch the pipeline - the result is already in host
    memory - and writing a long video is the longest phase of the step. A
    release that waits for the write holds ~10 GB of weights on the device
    for minutes after the last thing that needed them.
    """
    workflow = Workflow(_release_workflow_def(), "/tmp/test_output", "test.json")
    pipeline_cache = {}
    cached_at_save = {}

    def mock_pipeline_load(self, shared_components):
        self.pipeline = MagicMock()

    def mock_run(self, *args, **kwargs):
        result = MagicMock(result_list=[])
        step_name = self.name

        def save(*save_args, **save_kwargs):
            key = workflow._pipeline_keys_by_step.get(step_name)
            cached_at_save[step_name] = key in pipeline_cache
            return []

        result.save.side_effect = save
        return result

    with patch.object(Pipeline, "load", mock_pipeline_load):
        with patch.object(Step, "run", mock_run):
            with patch("dw.workflow.empty_device_cache"):
                workflow.run({}, previous_pipelines=pipeline_cache)

    assert cached_at_save == {"generate": False, "keep": True}


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
                    # Reads the expanded prompt, which is the shape this
                    # exists for - and keeps the task step in the run, since
                    # a step nothing reads and which saves nothing is elided
                    # before the first step executes (#122)
                    "arguments": {"prompt": "previous_result:expand_prompt"},
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

    assert seen_at_load["old_still_cached"] is False, (
        "the redefined step's previous model must be evicted before load"
    )


def _model_step(name, model_name):
    return {
        "name": name,
        "pipeline": {
            "configuration": {"component_type": "{Mock}"},
            "from_pretrained_arguments": {"model_name": model_name},
            "arguments": {},
        },
    }


def _load_with(workflow, step, cache, key):
    """Run the redefined step's load and report whether `key` was still in
    the cache at the moment load began."""
    seen_at_load = {}

    def mock_load(self, shared_components):
        seen_at_load["still_cached"] = key in cache
        self.pipeline = MagicMock()

    with patch.object(Pipeline, "load", mock_load):
        workflow.create_step_action(step, {}, cache, 1, "cpu")
    return seen_at_load["still_cached"]


def test_a_pipeline_another_running_step_currently_maps_to_is_not_released():
    """#150 carries step->key across jobs. Two steps that resolved to the
    same pipeline last time, and a rerun that changes only the first: the
    old model is still the second step's cache hit THIS run - its current
    key is the old key - so releasing it here forces a cold reload of a
    resident model, and holds both stacks while the first step's
    replacement loads, the exact transition #150 avoids. The end-of-run
    sweep (_evict_untouched_pipelines) drops it if nothing touched it."""
    from dw.workflow import pipeline_cache_key

    shared_key = pipeline_cache_key(_model_step("x", "shared-model")["pipeline"])
    changed_step = _model_step("gen", "new-model")
    cache = {shared_key: MagicMock()}

    workflow = Workflow({"id": "shared", "steps": []}, "/tmp/test_output", "t.json")
    workflow._prior_step_keys = {"gen": shared_key, "gen_again": shared_key}
    # gen_again still loads shared-model this run: its current key is the
    # one gen is about to leave behind
    workflow._running_pipeline_keys = {
        "gen": pipeline_cache_key(changed_step["pipeline"]),
        "gen_again": shared_key,
    }

    assert _load_with(workflow, changed_step, cache, shared_key) is True, (
        "a key another running step currently maps to must survive the "
        "redefined step's load"
    )


def test_a_prior_key_no_running_step_currently_maps_to_is_released():
    """_prior_step_keys is merged across every job the worker has ever run
    and never pruned (Worker._record_step_keys), so it can carry a step
    name from an earlier, unrelated workflow that happened to resolve to
    the same pipeline. That name is not a step this run executes, so no
    running step's current key is the old key - it must not save the key
    from release."""
    from dw.workflow import pipeline_cache_key

    shared_key = pipeline_cache_key(_model_step("x", "shared-model")["pipeline"])
    changed_step = _model_step("gen", "new-model")
    cache = {shared_key: MagicMock()}

    workflow = Workflow({"id": "shared", "steps": []}, "/tmp/test_output", "t.json")
    # stale_step mapped to shared_key in some earlier, different workflow's
    # run - it is not a step of the workflow this run executes
    workflow._prior_step_keys = {"gen": shared_key, "stale_step": shared_key}
    workflow._running_pipeline_keys = {
        "gen": pipeline_cache_key(changed_step["pipeline"])
    }

    assert _load_with(workflow, changed_step, cache, shared_key) is False, (
        "a name that is not a step of the running workflow must not save "
        "the key from release"
    )


def test_a_prior_key_every_sharing_step_moved_off_is_released():
    """Two steps (or a for_each group) reading one model variable, and a
    rerun that changes it: every step's PRIOR key is the old key and every
    step's current key is the new one. Judged on prior keys, the first
    step saw its sibling "still" on the old key and held both stacks while
    its replacement loaded - the OOM transition #150 fixed. Judged on the
    siblings' current keys, nobody is on the old key and it is released
    before the load."""
    from dw.workflow import pipeline_cache_key

    old_key = pipeline_cache_key(_model_step("x", "old-model")["pipeline"])
    changed_step = _model_step("gen", "new-model")
    new_key = pipeline_cache_key(changed_step["pipeline"])
    cache = {old_key: MagicMock()}

    workflow = Workflow({"id": "shared", "steps": []}, "/tmp/test_output", "t.json")
    workflow._prior_step_keys = {"gen": old_key, "gen_again": old_key}
    workflow._running_pipeline_keys = {"gen": new_key, "gen_again": new_key}

    assert _load_with(workflow, changed_step, cache, old_key) is False, (
        "a key no running step still maps to must be released before the load"
    )


def test_run_records_the_current_key_of_every_pipeline_step():
    """The wiring under the guard: Workflow.run records, from the realized
    steps it hands to create_step_action, the key each pipeline step loads
    under this run. create_step_action records the key it hashed per step
    (_pipeline_keys_by_step), so the two maps must be identical - computed
    by the same function over the same dicts."""
    from dw.workflow import pipeline_cache_key

    definition = {
        "id": "keys",
        "steps": [
            # Both save, so both run: a step that saves nothing and which
            # nothing reads is elided before the run (#122)
            {**_model_step("gen", "model-a"), "result": {"content_type": "image/png"}},
            {
                **_model_step("gen_again", "model-b"),
                "result": {"content_type": "image/png"},
            },
        ],
    }
    workflow = Workflow(definition, "/tmp/test_output", "t.json")
    hashed_by_create_step_action = {}
    original = Workflow.create_step_action

    def spy(self, step_definition, *args):
        if "pipeline" in step_definition:
            hashed_by_create_step_action[step_definition["name"]] = pipeline_cache_key(
                step_definition["pipeline"]
            )
        return original(self, step_definition, *args)

    def mock_load(self, shared_components):
        self.pipeline = MagicMock()

    with (
        patch.object(Pipeline, "load", mock_load),
        patch.object(Workflow, "create_step_action", spy),
        patch.object(Step, "run", lambda self, *a, **k: MagicMock(result_list=[])),
    ):
        workflow.run({})

    assert set(workflow._running_pipeline_keys) == {"gen", "gen_again"}
    assert workflow._running_pipeline_keys == hashed_by_create_step_action
    assert workflow._running_pipeline_keys == workflow._pipeline_keys_by_step
    assert (
        workflow._running_pipeline_keys["gen"]
        != (workflow._running_pipeline_keys["gen_again"])
    )


def test_pipeline_released_is_reported_on_the_event_stream():
    """The release is announced, and before the step's files are written.

    A consumer cannot time the release by polling memory: it happens inside
    the window between generation ending and the files appearing, which is
    sub-second for an image step. The event is the ordered record that makes
    it readable - see issue #75.
    """
    from dw.events import RunContext

    events = []
    workflow = Workflow(_release_workflow_def(), "/tmp/test_output", "test.json")

    def mock_pipeline_load(self, shared_components):
        self.pipeline = MagicMock()

    with patch.object(Pipeline, "load", mock_pipeline_load):
        with patch.object(
            Step, "run", lambda self, *args, **kwargs: MagicMock(result_list=[])
        ):
            with patch("dw.workflow.empty_device_cache"):
                workflow.run(
                    {},
                    previous_pipelines={},
                    context=RunContext(on_event=events.append),
                )

    names = [e["event"] for e in events]
    released = [e for e in events if e["event"] == "pipeline_released"]
    assert len(released) == 1, "only the step that asked for it reports a release"
    assert released[0]["step"] == "generate"
    assert "gpu_memory_allocated_mb" in released[0]
    assert "gpu_memory_allocated_before_mb" in released[0]

    ends = [i for i, e in enumerate(events) if e["event"] == "step_end"]
    assert names.index("pipeline_released") < ends[0], (
        "the release is reported before the step reports its files - which is "
        "the ordering the event exists to make visible"
    )


def test_superseded_release_is_reported_on_the_event_stream():
    """A release the caller never asked for still reaches the event stream.

    Without it a reload on top of a resident model looks exactly like a cold
    load, which is what made #150 take three jobs to diagnose.
    """
    from dw.events import RunContext, activate_context, deactivate_context
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

    events = []
    context = RunContext(on_event=lambda event: events.append(event))
    token = activate_context(context)
    try:
        with patch.object(Pipeline, "load", lambda self, shared: None):
            workflow.create_step_action(new_step, {}, cache, 1, "cpu")
    finally:
        deactivate_context(token)

    released = [e for e in events if e["event"] == "pipeline_released"]
    assert len(released) == 1, f"expected one release event, got {events}"
    assert released[0]["step"] == "gen"
    assert released[0]["reason"] == "superseded"


def test_release_pipeline_returns_host_caches_before_announcing_it():
    """The release hands pinned/arena host memory back then, not at job end.

    `pipeline_released` dropped the device memory but left ~10 GB of pinned
    staging buffers and heap arenas in RSS through every later step of the
    job, until the worker's between-run cleanup finally returned them (#368).
    The host caches are emptied after the pipeline has left the cache, and
    before the event, so the memory reading that follows it shows the drop.
    """
    from dw.events import RunContext

    events = []
    pipeline_cache = {}
    workflow = Workflow(_release_workflow_def(), "/tmp/test_output", "test.json")
    at_release = []

    def fake_release():
        key = workflow._pipeline_keys_by_step.get("generate")
        at_release.append(
            {
                "released_cached": key in pipeline_cache,
                "announced": any(e["event"] == "pipeline_released" for e in events),
            }
        )
        return 0.0

    def mock_pipeline_load(self, shared_components):
        self.pipeline = MagicMock()

    with patch.object(Pipeline, "load", mock_pipeline_load):
        with patch.object(
            Step, "run", lambda self, *args, **kwargs: MagicMock(result_list=[])
        ):
            with patch("dw.workflow.empty_device_cache"):
                with patch("dw.workflow.release_host_caches", fake_release):
                    workflow.run(
                        {},
                        previous_pipelines=pipeline_cache,
                        context=RunContext(on_event=events.append),
                    )

    assert at_release == [{"released_cached": False, "announced": False}]
    # the step that did not ask for a release leaves its pipeline warm
    assert workflow._pipeline_keys_by_step["keep"] in pipeline_cache


def test_release_models_returns_host_caches():
    """release_models is the task-model sibling of release_pipeline (#368)."""
    workflow = Workflow(
        _release_models_workflow_def(True), "/tmp/test_output", "test.json"
    )
    calls = []

    def mock_pipeline_load(self, shared_components):
        self.pipeline = MagicMock()

    cached_model(("text_generation", "some-model", "cuda"), MagicMock)
    try:
        with patch.object(Pipeline, "load", mock_pipeline_load):
            with patch.object(
                Step, "run", lambda self, *args, **kwargs: MagicMock(result_list=[])
            ):
                with patch("dw.workflow.empty_device_cache"):
                    with patch(
                        "dw.workflow.release_host_caches",
                        lambda: calls.append(bool(_model_cache)) or 0.0,
                    ):
                        workflow.run({}, previous_pipelines={})
    finally:
        clear_model_cache()

    assert calls == [False], "emptied once, after the task models were dropped"


def test_superseded_release_returns_host_caches():
    """A redefined step's old pipeline hands its host memory back before the
    new one loads, not once the job ends (#368)."""
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
    order = []

    def mock_load(self, shared_components):
        order.append("load")
        self.pipeline = MagicMock()

    with patch.object(Pipeline, "load", mock_load):
        with patch(
            "dw.workflow.release_host_caches",
            lambda: order.append(("release", old_key in cache)) or 0.0,
        ):
            workflow.create_step_action(new_step, {}, cache, 1, "cpu")

    assert order == [("release", False), "load"]


def test_release_host_caches_runs_for_real_on_the_release_path():
    """The unmocked call is harmless where there is no CUDA or glibc."""
    from dw.host_memory import release_host_caches

    workflow = Workflow(_release_workflow_def(), "/tmp/test_output", "test.json")

    def mock_pipeline_load(self, shared_components):
        self.pipeline = MagicMock()

    with patch.object(Pipeline, "load", mock_pipeline_load):
        with patch.object(
            Step, "run", lambda self, *args, **kwargs: MagicMock(result_list=[])
        ):
            with patch(
                "dw.workflow.release_host_caches",
                wraps=release_host_caches,
            ) as real:
                workflow.run({}, previous_pipelines={})

    assert real.call_count == 1
