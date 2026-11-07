"""A job a server restart dropped is closed by the next startup (#1220).

Every admitted job has a history row from submit on; a startup closes the
open rows of every other server whose lock is free as `cancelled` with an
`error` starting `server restart:`, and leaves a live sibling's alone.

A dead server is simulated by releasing its lock without shutdown(): its
threads stay parked, as a SIGKILLed process's rows would.
"""

import queue
import sqlite3
import time

from dw.server.job_history import RESTART_QUEUED
from dw.server.jobs import JobManager

from .test_server import ScriptedWorkerManager, admitted_for, success_script
from .test_server import valid_workflow

RUN_DIR = "server_test/run-1"


def stuck_script(command):
    """A run that opens its directory and never ends."""
    yield {
        "type": "progress",
        "event": "run_start",
        "run_id": "run-1",
        "run_dir": RUN_DIR,
        "version": 1,
    }
    yield {
        "type": "progress",
        "event": "step_start",
        "step": "gen",
        "index": 0,
        "total_steps": 1,
    }


class StuckWorkerManager(ScriptedWorkerManager):
    """Blocks for good once the script runs out: no timeout, so the job
    thread never finishes the job and overwrites its row."""

    def get_result(self, timeout=None):
        return self._results.get()


def manager_on(tmp_path, worker_manager):
    return JobManager(
        str(tmp_path / "outputs"),
        worker_manager=worker_manager,
        history_path=str(tmp_path / "jobs.sqlite"),
    )


def submit(manager, prompt="hi"):
    return manager.submit(
        admitted=admitted_for(manager, valid_workflow()),
        workflow=valid_workflow(),
        arguments={"prompt": prompt},
    )


def wait_until(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition never held")


def kill(manager):
    """What SIGKILL leaves: the lock free, the rows as last written."""
    manager._server_lock.release()


def stored_status(tmp_path, job_id):
    connection = sqlite3.connect(tmp_path / "jobs.sqlite")
    try:
        row = connection.execute(
            "SELECT status FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    finally:
        connection.close()
    return row and row[0]


def a_dead_server_with_a_running_and_a_queued_job(tmp_path):
    dead = manager_on(tmp_path, StuckWorkerManager(stuck_script))
    running = submit(dead, "first")
    wait_until(lambda: running.run_dir == RUN_DIR)
    queued = submit(dead, "second")
    # One card, held by the stuck job: the second stays queued
    assert queued.status == "queued"
    assert stored_status(tmp_path, running.id) == "running"
    assert stored_status(tmp_path, queued.id) == "queued"
    kill(dead)
    return running, queued


def test_a_queued_job_on_a_dead_server_is_cancelled_with_the_queued_reason(
    tmp_path,
):
    _, queued = a_dead_server_with_a_running_and_a_queued_job(tmp_path)

    restarted = manager_on(tmp_path, ScriptedWorkerManager(success_script))
    try:
        detail = restarted.get(queued.id)
        assert detail["status"] == "cancelled"
        assert detail["error"] == RESTART_QUEUED
        assert detail["error"].startswith("server restart:")
        assert detail["finished_at"] is not None
        assert queued.id in [s["id"] for s in restarted.list()]
    finally:
        restarted.shutdown()


def test_a_running_job_on_a_dead_server_names_its_run_dir(tmp_path):
    running, _ = a_dead_server_with_a_running_and_a_queued_job(tmp_path)

    restarted = manager_on(tmp_path, ScriptedWorkerManager(success_script))
    try:
        detail = restarted.get(running.id)
        assert detail["status"] == "cancelled"
        assert detail["error"].startswith(
            "server restart: running when the server stopped"
        )
        assert RUN_DIR in detail["error"]
    finally:
        restarted.shutdown()


def test_the_sweep_removes_the_dead_servers_lock_file(tmp_path):
    dead = manager_on(tmp_path, StuckWorkerManager(stuck_script))
    submit(dead)
    lock_file = dead._server_lock.lock_file
    kill(dead)

    restarted = manager_on(tmp_path, ScriptedWorkerManager(success_script))
    try:
        assert not (tmp_path / "servers" / lock_file.rsplit("/", 1)[-1]).exists()
    finally:
        restarted.shutdown()


def test_a_live_siblings_jobs_are_left_alone_and_not_listed(tmp_path):
    live = manager_on(tmp_path, StuckWorkerManager(stuck_script))
    running = submit(live, "first")
    wait_until(lambda: running.run_dir == RUN_DIR)
    queued = submit(live, "second")

    sibling = manager_on(tmp_path, ScriptedWorkerManager(success_script))
    try:
        assert stored_status(tmp_path, running.id) == "running"
        assert stored_status(tmp_path, queued.id) == "queued"
        listed = [s["id"] for s in sibling.list()]
        assert running.id not in listed and queued.id not in listed
        assert sibling.get(running.id) is None
        assert sibling.get(queued.id) is None
    finally:
        sibling.shutdown()

    # Once it dies, the next startup closes them after all
    kill(live)
    later = manager_on(tmp_path, ScriptedWorkerManager(success_script))
    try:
        assert later.get(queued.id)["error"] == RESTART_QUEUED
    finally:
        later.shutdown()


def test_a_tombstone_reruns_as_a_fresh_job(tmp_path):
    _, queued = a_dead_server_with_a_running_and_a_queued_job(tmp_path)

    restarted = manager_on(tmp_path, ScriptedWorkerManager(success_script))
    try:
        rerun = restarted.rerun(
            queued.id, admitted=admitted_for(restarted, valid_workflow())
        )
        assert rerun is not None and rerun.id != queued.id
        assert rerun.spec["rerun_of"] == queued.id
        assert rerun.spec["arguments"] == {"prompt": "second"}
        wait_until(lambda: rerun.status == "succeeded")
    finally:
        restarted.shutdown()


def test_a_failed_history_write_at_dispatch_does_not_fail_the_job(tmp_path):
    manager = manager_on(tmp_path, ScriptedWorkerManager(success_script))
    real_record = manager.history.record
    open_writes = queue.Queue()

    def record(job):
        if job.status in ("queued", "running"):
            open_writes.put(job.status)
            raise sqlite3.OperationalError("database is locked")
        real_record(job)

    manager.history.record = record
    try:
        job = submit(manager)
        wait_until(lambda: job.status in ("succeeded", "failed"))
        assert job.status == "succeeded"
        assert "running" in list(open_writes.queue)
        # Its outcome is still recorded
        wait_until(lambda: stored_status(tmp_path, job.id) == "succeeded")
    finally:
        manager.shutdown()


def test_a_clean_shutdown_releases_its_lock(tmp_path):
    manager = manager_on(tmp_path, ScriptedWorkerManager(success_script))
    lock_file = manager._server_lock.lock_file
    manager.shutdown()
    assert not (tmp_path / "servers" / lock_file.rsplit("/", 1)[-1]).exists()
