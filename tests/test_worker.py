#!/usr/bin/env python3
"""
Tests for the worker-based REPL implementation.
This tests the persistent worker subprocess with GPU memory management.

All tests spawn a real `multiprocessing.Process` running `worker_main`. The
`worker_process` fixture owns the full lifecycle of that child process so
that no test failure can ever leave it orphaned: a spawned worker that never
receives (or never gets to process) a "shutdown" command blocks forever on
its command queue, and since it is non-daemon, multiprocessing's atexit
handler joins it forever at interpreter exit - hanging the whole pytest run.
"""

import sys
import os
import multiprocessing

import pytest

# CRITICAL: Set spawn method before any other imports that might use multiprocessing
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dw.worker import worker_main
import torch

# These two run a real SD 1.5 fp16 generation through the worker - fp16
# doesn't run on CPU, and CI has no accelerator (or the model), so they
# only run where one exists
requires_accelerator = pytest.mark.skipif(
    not (torch.cuda.is_available() or torch.backends.mps.is_available()),
    reason="runs a real fp16 generation; needs an accelerator",
)

# The first response after spawning must tolerate the child process importing
# torch + diffusers before it can reply to anything - a few seconds warm,
# much longer cold or when the machine is under load. This timeout is
# generous specifically to absorb that one-time cost.
WORKER_READY_TIMEOUT = 60

# Subsequent commands go to an already-imported, idle worker, so they should
# be fast - but keep this a bit above the old 5s to tolerate a loaded machine.
COMMAND_TIMEOUT = 10

TEST_WORKFLOW_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dw",
    "workflows",
    "test.json",
)


@pytest.fixture
def worker_process():
    """
    Spawn a worker process plus its command/result queues, yield them to the
    test, and unconditionally tear the worker down afterward.

    Teardown never assumes the test left things in a clean state: it first
    tries a graceful "shutdown" command (short join), then escalates to
    terminate() and finally kill() if the process is still alive. This runs
    in a `finally` so an assertion failure - or any other exception - can
    never orphan the child.
    """
    cmd_queue = multiprocessing.Queue()
    res_queue = multiprocessing.Queue()

    worker = multiprocessing.Process(
        target=worker_main, args=(cmd_queue, res_queue, "INFO")
    )
    # Belt-and-braces: even if the teardown logic below were somehow skipped,
    # a daemon process is killed automatically when the test process exits
    # instead of being joined forever.
    worker.daemon = True
    worker.start()

    try:
        yield cmd_queue, res_queue, worker
    finally:
        if worker.is_alive():
            try:
                cmd_queue.put({"type": "shutdown"})
                res_queue.get(timeout=COMMAND_TIMEOUT)
            except Exception:
                pass  # best-effort; escalation below handles a stuck worker
            worker.join(timeout=5)

        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)

        if worker.is_alive():
            worker.kill()
            worker.join(timeout=5)

        cmd_queue.close()
        res_queue.close()


def test_worker_lifecycle(worker_process):
    """Worker starts, responds to ping, and shuts down gracefully."""
    cmd_queue, res_queue, worker = worker_process

    cmd_queue.put({"type": "ping"})
    result = res_queue.get(timeout=WORKER_READY_TIMEOUT)
    assert result["type"] == "pong"
    assert result["run_count"] == 0

    cmd_queue.put({"type": "shutdown"})
    result = res_queue.get(timeout=COMMAND_TIMEOUT)
    assert result["type"] == "shutdown_complete"

    worker.join(timeout=COMMAND_TIMEOUT)
    assert not worker.is_alive()


def test_worker_memory_status(worker_process):
    """Worker reports memory status on request."""
    cmd_queue, res_queue, worker = worker_process

    cmd_queue.put({"type": "memory_status"})
    result = res_queue.get(timeout=WORKER_READY_TIMEOUT)

    assert result["type"] == "memory_status"
    info = result["info"]
    assert "gpu_available" in info
    if info["gpu_available"]:
        assert "gpu_device_name" in info
        assert "gpu_memory_allocated_mb" in info
        assert "gpu_memory_reserved_mb" in info


def test_worker_clear_memory(worker_process):
    """Worker clears cached models/components on request."""
    cmd_queue, res_queue, worker = worker_process

    cmd_queue.put({"type": "clear_memory"})
    result = res_queue.get(timeout=WORKER_READY_TIMEOUT)

    assert result["type"] == "memory_cleared"
    info = result["info"]
    assert "gpu_available" in info


@pytest.mark.skipif(
    not os.path.exists(TEST_WORKFLOW_PATH),
    reason=f"test workflow not found: {TEST_WORKFLOW_PATH}",
)
@requires_accelerator
def test_worker_with_simple_workflow(worker_process, tmp_path):
    """Worker executes a real workflow and reuses cached models on a second run."""
    cmd_queue, res_queue, worker = worker_process

    output_dir = str(tmp_path / "test_outputs")
    os.makedirs(output_dir, exist_ok=True)

    cmd_queue.put(
        {
            "type": "execute",
            "workflow_path": TEST_WORKFLOW_PATH,
            "arguments": {},
            "output_dir": output_dir,
            "log_level": "INFO",
        }
    )

    # First message after spawn (or after a fresh command with no prior
    # traffic) must tolerate child import cost; workflow execution itself
    # (model load + inference) is also slow, so keep the generous timeout
    # for every message in this loop rather than switching to the short one.
    success = False
    saw_workflow_loaded = False
    while True:
        result = res_queue.get(timeout=WORKER_READY_TIMEOUT)
        result_type = result.get("type")

        if result_type == "workflow_loaded":
            saw_workflow_loaded = True
        elif result_type == "success":
            success = True
            break
        elif result_type == "error":
            pytest.fail(f"Workflow execution error: {result['message']}")

    assert success
    assert saw_workflow_loaded

    # Run again to exercise the model-reuse/caching path.
    cmd_queue.put(
        {
            "type": "execute",
            "workflow_path": TEST_WORKFLOW_PATH,
            "arguments": {},
            "output_dir": output_dir,
            "log_level": "INFO",
        }
    )

    second_run_count = None
    while True:
        result = res_queue.get(timeout=WORKER_READY_TIMEOUT)
        result_type = result.get("type")

        if result_type == "success":
            second_run_count = result["run_count"]
            break
        elif result_type == "error":
            pytest.fail(f"Workflow execution error on second run: {result['message']}")

    assert second_run_count == 2


@pytest.mark.skipif(
    not os.path.exists(TEST_WORKFLOW_PATH),
    reason=f"test workflow not found: {TEST_WORKFLOW_PATH}",
)
@requires_accelerator
def test_worker_cache_hit_applies_new_output_dir(worker_process, tmp_path):
    """
    A second execute for the same workflow keeps its models cached, but the
    caller can still pass a different output_dir (e.g. after `config set
    output_dir` in the REPL), and results must land there rather than
    silently continuing to save to the first directory.
    """
    cmd_queue, res_queue, worker = worker_process

    first_output_dir = str(tmp_path / "first_outputs")
    os.makedirs(first_output_dir, exist_ok=True)

    cmd_queue.put(
        {
            "type": "execute",
            "workflow_path": TEST_WORKFLOW_PATH,
            "arguments": {},
            "output_dir": first_output_dir,
            "log_level": "INFO",
        }
    )

    saw_workflow_loaded = False
    while True:
        result = res_queue.get(timeout=WORKER_READY_TIMEOUT)
        result_type = result.get("type")

        if result_type == "workflow_loaded":
            saw_workflow_loaded = True
        elif result_type == "success":
            break
        elif result_type == "error":
            pytest.fail(f"Workflow execution error: {result['message']}")

    assert saw_workflow_loaded

    first_run_files = os.listdir(first_output_dir)
    assert len(first_run_files) > 0

    # Same workflow file (unchanged content/path) -> cache hit, but a new
    # output_dir.
    second_output_dir = str(tmp_path / "second_outputs")

    cmd_queue.put(
        {
            "type": "execute",
            "workflow_path": TEST_WORKFLOW_PATH,
            "arguments": {},
            "output_dir": second_output_dir,
            "log_level": "INFO",
        }
    )

    saw_model_release = False
    second_run_count = None
    while True:
        result = res_queue.get(timeout=WORKER_READY_TIMEOUT)
        result_type = result.get("type")

        if result_type == "output" and "releasing cached models" in result.get(
            "message", ""
        ):
            saw_model_release = True
        elif result_type == "success":
            second_run_count = result["run_count"]
            break
        elif result_type == "error":
            pytest.fail(f"Workflow execution error on second run: {result['message']}")

    # Same workflow identity - the cached models must NOT have been dropped
    assert not saw_model_release
    assert second_run_count == 2

    # The bug: on a cache hit the workflow kept the OLD output_dir, so
    # results silently saved to first_output_dir instead of second_output_dir.
    assert os.path.isdir(second_output_dir), (
        "results were not saved to the new output_dir after a cache-hit "
        "execute with a changed output_dir"
    )
    second_run_files = os.listdir(second_output_dir)
    assert len(second_run_files) > 0, (
        "second_output_dir exists but is empty - results kept going to the "
        "original output_dir"
    )
