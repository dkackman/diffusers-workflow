"""Unit tests for WorkerManager - the process management used by
JobManager to run the GPU worker. Everything here runs against a fake
process; the real spawn/shutdown path is exercised by tests/test_worker.py
on a GPU box.
"""

import logging
import queue

import pytest

import dw.worker_manager as worker_manager
from dw.worker import (
    Failed,
    MemoryStatus,
    MemoryStatusReply,
)
from dw.worker_manager import WorkerManager


class FakeProcess:
    def __init__(self, alive=True, exitcode=None):
        self.alive = alive
        self.exitcode = exitcode

    def is_alive(self):
        return self.alive


@pytest.fixture
def manager(monkeypatch):
    # The liveness poll is one second in production; the tests should not
    # spend it
    monkeypatch.setattr(worker_manager, "WORKER_LIVENESS_POLL_SECONDS", 0.01)
    manager = WorkerManager()
    manager.worker_active = True
    manager.worker_process = FakeProcess()
    manager.command_queue = queue.Queue()
    manager.result_queue = queue.Queue()
    return manager


class TestGetResult:
    def test_a_message_already_queued_comes_straight_back(self, manager):
        manager.result_queue.put({"type": "output", "message": "hi"})
        assert manager.get_result() == {"type": "output", "message": "hi"}

    def test_waiting_on_a_dead_worker_raises_instead_of_blocking(self, manager):
        """A worker that dies mid-run with nothing left to say must surface
        as an error - the untimed wait exists for long generations, not for
        a process that is no longer there."""
        manager.worker_process.alive = False
        with pytest.raises(RuntimeError, match="died while waiting"):
            manager.get_result()

    def test_a_message_sent_before_death_is_still_delivered(self, manager):
        manager.result_queue.put({"type": "error", "message": "last words"})
        manager.worker_process.alive = False
        assert manager.get_result()["message"] == "last words"
        with pytest.raises(RuntimeError):
            manager.get_result()

    def test_an_explicit_timeout_raises_empty_not_runtime_error(self, manager):
        with pytest.raises(queue.Empty):
            manager.get_result(timeout=0.01)


class TestRequest:
    """One reply dispatcher: a request is answered by the reply carrying its
    own request_id, and nothing else on the queue is taken for it."""

    def test_sends_the_command_on_the_wire(self, manager):
        manager.result_queue.put(
            {"type": "memory_status", "request_id": "r-1", "info": {}}
        )
        manager.request(MemoryStatus(request_id="r-1"), timeout=1)
        assert manager.command_queue.get_nowait() == {
            "type": "memory_status",
            "request_id": "r-1",
        }

    def test_returns_the_typed_reply_carrying_its_id(self, manager):
        manager.result_queue.put(
            {"type": "memory_status", "request_id": "r-1", "info": {"a": 1}}
        )
        reply = manager.request(MemoryStatus(request_id="r-1"), timeout=1)
        assert reply == MemoryStatusReply(request_id="r-1", info={"a": 1})

    def test_discards_every_reply_that_is_not_its_own(self, manager, caplog):
        manager.result_queue.put(
            {"type": "probe_cache", "request_id": "gave-up", "cached": ["x"]}
        )
        manager.result_queue.put({"type": "output", "message": "late"})
        manager.result_queue.put(
            {"type": "memory_status", "request_id": "r-1", "info": {}}
        )
        with caplog.at_level(logging.DEBUG, logger="dw"):
            reply = manager.request(MemoryStatus(request_id="r-1"), timeout=1)
        assert isinstance(reply, MemoryStatusReply)
        discarded = [r for r in caplog.records if "Discarding" in r.getMessage()]
        assert [r.levelno for r in discarded] == [logging.DEBUG, logging.DEBUG]
        assert manager.result_queue.empty()

    def test_an_error_answering_the_request_is_its_reply(self, manager):
        manager.result_queue.put(
            {
                "type": "error",
                "message": "Command processing error",
                "request_id": "r-1",
            }
        )
        reply = manager.request(MemoryStatus(request_id="r-1"), timeout=1)
        assert reply == Failed(message="Command processing error", request_id="r-1")

    def test_a_reply_without_an_id_never_answers_a_request(self, manager):
        manager.result_queue.put({"type": "probe_cache", "cached": ["x"]})
        manager.result_queue.put({"type": "pong"})
        with pytest.raises(queue.Empty):
            manager.request(MemoryStatus(request_id="r-1"), timeout=0.05)

    def test_the_timeout_bounds_the_whole_wait_not_each_read(self, manager):
        """A worker still draining stale replies must not keep a request
        waiting past its timeout by answering just often enough."""
        import threading
        import time

        stop = threading.Event()

        def trickle():
            while not stop.is_set():
                manager.result_queue.put({"type": "output", "message": "noise"})
                time.sleep(0.01)

        thread = threading.Thread(target=trickle, daemon=True)
        thread.start()
        started = time.monotonic()
        try:
            with pytest.raises(queue.Empty):
                manager.request(MemoryStatus(request_id="r-1"), timeout=0.2)
        finally:
            stop.set()
            thread.join()
        assert time.monotonic() - started < 2

    def test_an_inactive_worker_raises_runtime_error(self):
        manager = WorkerManager()
        with pytest.raises(RuntimeError, match="not active"):
            manager.request(MemoryStatus(request_id="r-1"), timeout=0.05)


class TestCrashDetails:
    """A worker killed by a signal never reaches worker_main's handler, so
    no traceback exists and the exit code is the whole diagnosis."""

    def test_a_live_worker_has_nothing_to_report(self, manager):
        assert manager.crash_details() is None

    def test_a_worker_not_yet_reaped_has_nothing_to_report(self, manager):
        manager.worker_process.alive = False
        assert manager.crash_details() is None

    def test_sigkill_names_the_out_of_memory_killer(self, manager):
        manager.worker_process.alive = False
        manager.worker_process.exitcode = -9
        detail = manager.crash_details()
        assert "SIGKILL" in detail
        assert "out-of-memory" in detail

    def test_another_signal_is_named_without_the_oom_guess(self, manager):
        manager.worker_process.alive = False
        manager.worker_process.exitcode = -11
        detail = manager.crash_details()
        assert "SIGSEGV" in detail
        assert "out-of-memory" not in detail

    def test_a_nonzero_exit_reports_its_code(self, manager):
        manager.worker_process.alive = False
        manager.worker_process.exitcode = 1
        assert manager.crash_details() == "exited with code 1"

    def test_it_answers_for_a_manager_that_never_started_one(self):
        assert WorkerManager().crash_details() is None


class TestInactiveWorker:
    def test_send_and_receive_refuse_an_inactive_worker(self):
        manager = WorkerManager()
        with pytest.raises(RuntimeError, match="not active"):
            manager.send_command({"type": "memory_status"})
        with pytest.raises(RuntimeError, match="not active"):
            manager.get_result()

    def test_mark_crashed_clears_the_tracking_state(self, manager):
        manager.mark_crashed()
        assert manager.worker_active is False
        assert manager.worker_process is None
        with pytest.raises(RuntimeError, match="not active"):
            manager.send_command({"type": "memory_status"})

    def test_shutdown_of_a_dead_process_is_a_no_op(self, manager):
        manager.worker_process.alive = False
        manager.shutdown_worker()  # must not raise or try to join
