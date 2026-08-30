"""Unit tests for WorkerManager - the process-facing side of the REPL and
the server's job runner. Everything here runs against a fake process; the
real spawn/shutdown path is exercised by tests/test_worker.py on a GPU box.
"""

import queue

import pytest

import dw.repl_worker as repl_worker
from dw.repl_worker import WorkerManager


class FakeProcess:
    def __init__(self, alive=True):
        self.alive = alive

    def is_alive(self):
        return self.alive


@pytest.fixture
def manager(monkeypatch):
    # The liveness poll is one second in production; the tests should not
    # spend it
    monkeypatch.setattr(repl_worker, "WORKER_LIVENESS_POLL_SECONDS", 0.01)
    manager = WorkerManager()
    manager.worker_active = True
    manager.worker_process = FakeProcess()
    manager.command_queue = queue.Queue()
    manager.result_queue = queue.Queue()
    return manager


class TestGetResult:
    def test_a_message_already_queued_comes_straight_back(self, manager):
        manager.result_queue.put({"type": "pong"})
        assert manager.get_result() == {"type": "pong"}

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


class TestInactiveWorker:
    def test_send_and_receive_refuse_an_inactive_worker(self):
        manager = WorkerManager()
        with pytest.raises(RuntimeError, match="not active"):
            manager.send_command({"type": "ping"})
        with pytest.raises(RuntimeError, match="not active"):
            manager.get_result()

    def test_mark_crashed_clears_the_tracking_state(self, manager):
        manager.mark_crashed()
        assert manager.worker_active is False
        assert manager.worker_process is None
        with pytest.raises(RuntimeError, match="not active"):
            manager.send_command({"type": "ping"})

    def test_shutdown_of_a_dead_process_is_a_no_op(self, manager):
        manager.worker_process.alive = False
        manager.shutdown_worker()  # must not raise or try to join
