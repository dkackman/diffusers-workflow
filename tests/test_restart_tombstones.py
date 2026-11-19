"""A job a server restart dropped is closed by the next startup (#1220).

Every admitted job has a history row from submit on; a startup closes the
open rows of every other server whose lock is free as `cancelled` with an
`error` starting `server restart:`, and leaves a live sibling's alone.

A dead server is simulated by releasing its lock without shutdown(): its
threads stay parked, as a SIGKILLed process's rows would.
"""

import contextlib
import queue
import sqlite3
import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from dw.server.app import create_app
from dw.server.job_history import DELETED_SUFFIX, RESTART_QUEUED
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


# ------------------------------------------------- the restart batch (#1266)


def a_dead_server_with_three_jobs(tmp_path):
    """One running and two queued, all left open when the server died."""
    workflow_dir = tmp_path / "workflows"
    workflow_dir.mkdir(exist_ok=True)
    dead = manager_on(tmp_path, StuckWorkerManager(stuck_script))

    def queue_one(prompt):
        # As the route submits: the workflow directory is the base
        return dead.submit(
            admitted=admitted_for(dead, valid_workflow()),
            workflow=valid_workflow(),
            arguments={"prompt": prompt},
            base_dir=str(workflow_dir),
            workflow_dir=str(workflow_dir),
        )

    running = queue_one("first")
    wait_until(lambda: running.run_dir == RUN_DIR)
    second = queue_one("second")
    third = queue_one("third")
    kill(dead)
    return [running.id, second.id, third.id]


def fake_admit(manager, delay=0.0):
    """What the route's admission answers, without the app around it."""

    def admit(member_id):
        _, arguments = manager.rerun_spec(member_id)
        if delay:
            time.sleep(delay)
        admission = SimpleNamespace(
            workflow=admitted_for(manager, valid_workflow()),
            warnings=[],
            vram_need=None,
            host_ram_need=None,
            repos_named=[],
            plan=None,
        )
        return arguments, admission

    return admit


def set_column(tmp_path, job_id, column, value):
    connection = sqlite3.connect(tmp_path / "jobs.sqlite")
    try:
        with connection:
            connection.execute(
                f"UPDATE jobs SET {column} = ? WHERE id = ?", (value, job_id)
            )
    finally:
        connection.close()


@contextlib.contextmanager
def restarted_client(tmp_path):
    manager = manager_on(tmp_path, ScriptedWorkerManager(success_script))
    workflow_dir = tmp_path / "workflows"
    workflow_dir.mkdir(exist_ok=True)
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    app = create_app(
        workflow_dir=str(workflow_dir),
        output_dir=str(tmp_path / "outputs"),
        job_manager=manager,
        prompt_dir=str(prompt_dir),
    )
    try:
        yield TestClient(app, base_url="http://localhost"), manager
    finally:
        manager.shutdown()


def reruns(manager):
    return [job for job in manager.jobs.values() if job.rerun_of]


def test_the_batch_holds_every_tombstone_oldest_first(tmp_path):
    ids = a_dead_server_with_three_jobs(tmp_path)
    restarted = manager_on(tmp_path, ScriptedWorkerManager(success_script))
    try:
        instance, members = restarted.history.restart_batch(ids[2])
        assert instance and [member["id"] for member in members] == ids
    finally:
        restarted.shutdown()


def test_the_batch_queues_in_created_at_order_each_with_rerun_of(tmp_path):
    ids = a_dead_server_with_three_jobs(tmp_path)
    with restarted_client(tmp_path) as (client, manager):
        response = client.post(
            f"/api/jobs/{ids[1]}/rerun-restart-batch",
            json={"acknowledged_cost": True},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert "would_queue" not in body
        assert body["skipped"] == []
        assert [entry["rerun_of"] for entry in body["queued"]] == ids
        queued = [manager.jobs[entry["id"]] for entry in body["queued"]]
        assert [job.rerun_of for job in queued] == ids
        created = [job.created_at for job in queued]
        assert created == sorted(created)
        assert [job.spec["arguments"]["prompt"] for job in queued] == [
            "first",
            "second",
            "third",
        ]


def test_a_second_call_skips_every_member_as_already_rerun(tmp_path):
    ids = a_dead_server_with_three_jobs(tmp_path)
    with restarted_client(tmp_path) as (client, manager):
        url = f"/api/jobs/{ids[0]}/rerun-restart-batch"
        first = client.post(url, json={"acknowledged_cost": True}).json()
        again = client.post(url, json={"acknowledged_cost": True}).json()
        assert again["queued"] == []
        rerun_as = {entry["rerun_of"]: entry["id"] for entry in first["queued"]}
        assert again["skipped"] == [
            {"id": member, "reason": f"already rerun as {rerun_as[member]}"}
            for member in ids
        ]
        assert len(reruns(manager)) == 3


def test_a_rerun_recorded_in_history_counts_after_another_restart(tmp_path):
    ids = a_dead_server_with_three_jobs(tmp_path)
    first = manager_on(tmp_path, ScriptedWorkerManager(success_script))
    try:
        answer = first.rerun_restart_batch(ids[0], fake_admit(first), queue=True)
        assert len(answer["queued"]) == 3
        for entry in answer["queued"]:
            wait_until(lambda: first.jobs[entry["id"]].status == "succeeded")
    finally:
        first.shutdown()
    later = manager_on(tmp_path, ScriptedWorkerManager(success_script))
    try:
        again = later.rerun_restart_batch(ids[0], fake_admit(later), queue=True)
        assert again["queued"] == []
        assert all(
            entry["reason"].startswith("already rerun as ")
            for entry in again["skipped"]
        )
    finally:
        later.shutdown()


def test_two_concurrent_calls_queue_each_member_once(tmp_path):
    ids = a_dead_server_with_three_jobs(tmp_path)
    manager = manager_on(tmp_path, ScriptedWorkerManager(success_script))
    try:
        admit = fake_admit(manager, delay=0.05)
        start = threading.Barrier(2)
        answers = []

        def call():
            start.wait()
            answers.append(manager.rerun_restart_batch(ids[0], admit, queue=True))

        threads = [threading.Thread(target=call) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        queued = [entry["rerun_of"] for answer in answers for entry in answer["queued"]]
        assert sorted(queued) == sorted(ids)
        assert sorted(job.rerun_of for job in reruns(manager)) == sorted(ids)
    finally:
        manager.shutdown()


def test_a_member_in_a_deleted_workspace_is_skipped_and_the_rest_queue(tmp_path):
    ids = a_dead_server_with_three_jobs(tmp_path)
    set_column(tmp_path, ids[1], "workspace", f"gone{DELETED_SUFFIX}")
    with restarted_client(tmp_path) as (client, manager):
        body = client.post(
            f"/api/jobs/{ids[0]}/rerun-restart-batch",
            json={"acknowledged_cost": True},
        ).json()
        assert [entry["rerun_of"] for entry in body["queued"]] == [ids[0], ids[2]]
        assert body["skipped"] == [
            {"id": ids[1], "reason": "workspace 'gone' was deleted"}
        ]


def test_a_member_admission_refuses_is_skipped_and_the_rest_queue(tmp_path):
    ids = a_dead_server_with_three_jobs(tmp_path)
    manager = manager_on(tmp_path, ScriptedWorkerManager(success_script))
    try:
        admit = fake_admit(manager)

        def refusing(member_id):
            if member_id == ids[0]:
                raise ValueError("Unknown pipeline component")
            return admit(member_id)

        answer = manager.rerun_restart_batch(ids[2], refusing, queue=True)
        assert [entry["rerun_of"] for entry in answer["queued"]] == ids[1:]
        assert answer["skipped"] == [
            {"id": ids[0], "reason": "Unknown pipeline component"}
        ]
    finally:
        manager.shutdown()


def test_a_job_that_is_not_a_tombstone_is_refused(tmp_path):
    with restarted_client(tmp_path) as (client, manager):
        job = submit(manager)
        wait_until(lambda: job.status == "succeeded")
        wait_until(lambda: stored_status(tmp_path, job.id) == "succeeded")
        response = client.post(
            f"/api/jobs/{job.id}/rerun-restart-batch",
            json={"acknowledged_cost": True},
        )
        assert response.status_code == 400
        assert "not a restart tombstone" in response.json()["detail"]
        assert reruns(manager) == []


def test_a_tombstone_with_no_server_instance_is_refused(tmp_path):
    ids = a_dead_server_with_three_jobs(tmp_path)
    with restarted_client(tmp_path) as (client, manager):
        set_column(tmp_path, ids[0], "server_instance", None)
        response = client.post(
            f"/api/jobs/{ids[0]}/rerun-restart-batch",
            json={"acknowledged_cost": True},
        )
        assert response.status_code == 400
        assert "not a restart tombstone" in response.json()["detail"]
        assert reruns(manager) == []


def test_an_unknown_job_is_404(tmp_path):
    with restarted_client(tmp_path) as (client, manager):
        response = client.post("/api/jobs/no-such-job/rerun-restart-batch", json={})
        assert response.status_code == 404
        assert reruns(manager) == []


def test_without_acknowledgement_the_preview_queues_nothing(tmp_path):
    ids = a_dead_server_with_three_jobs(tmp_path)
    with restarted_client(tmp_path) as (client, manager):
        before = set(manager.jobs)
        for body in ({}, {"acknowledged_cost": False}):
            response = client.post(f"/api/jobs/{ids[0]}/rerun-restart-batch", json=body)
            assert response.status_code == 200, response.text
            answer = response.json()
            assert answer["queued"] == []
            assert [entry["id"] for entry in answer["would_queue"]] == ids
            assert all("estimate" in entry for entry in answer["would_queue"])
            assert answer["skipped"] == []
        assert set(manager.jobs) == before
        assert reruns(manager) == []


def test_a_bound_acknowledgement_is_refused(tmp_path):
    ids = a_dead_server_with_three_jobs(tmp_path)
    with restarted_client(tmp_path) as (client, manager):
        response = client.post(
            f"/api/jobs/{ids[0]}/rerun-restart-batch",
            json={"acknowledged_cost": {"fingerprint": "abc", "minutes": 1}},
        )
        assert response.status_code == 400
        assert "bound acknowledgement" in response.json()["detail"]
        assert reruns(manager) == []


def test_a_live_siblings_rows_are_never_members(tmp_path):
    ids = a_dead_server_with_three_jobs(tmp_path)
    # Closes the dead server's rows, then stays up holding two open ones
    live = manager_on(tmp_path, StuckWorkerManager(stuck_script))
    running = submit(live, "sibling running")
    wait_until(lambda: running.run_dir == RUN_DIR)
    queued = submit(live, "sibling queued")
    restarted = manager_on(tmp_path, ScriptedWorkerManager(success_script))
    try:
        _, members = restarted.history.restart_batch(ids[0])
        assert [member["id"] for member in members] == ids
        answer = restarted.rerun_restart_batch(
            ids[0], fake_admit(restarted), queue=True
        )
        assert [entry["rerun_of"] for entry in answer["queued"]] == ids
        assert restarted.history.restart_batch(running.id) is None
        assert restarted.history.restart_batch(queued.id) is None
    finally:
        restarted.shutdown()


# ------------------------------------- the batch over MCP rerun_job (#1371)


def mcp_client(client):
    """The MCP tools' REST client, talking to the in-process app."""
    from dw_mcp.client import DwClient

    dw = DwClient(base_url="http://localhost")
    # The TestClient is an httpx.Client already wired to the app
    dw._http.close()
    dw._http = client
    return dw


def test_mcp_rerun_job_previews_queues_then_skips_a_restart_batch(tmp_path):
    from dw_mcp import diagnose

    ids = a_dead_server_with_three_jobs(tmp_path)
    with restarted_client(tmp_path) as (client, manager):
        dw = mcp_client(client)
        preview = diagnose.rerun_job(dw, ids[2], restart_batch=True)
        assert preview["queued"] == []
        assert [entry["id"] for entry in preview["would_queue"]] == ids
        assert reruns(manager) == []

        queued = diagnose.rerun_job(
            dw, ids[2], restart_batch=True, acknowledged_cost=True
        )
        assert [entry["rerun_of"] for entry in queued["queued"]] == ids

        again = diagnose.rerun_job(
            dw, ids[2], restart_batch=True, acknowledged_cost=True
        )
        assert again["queued"] == []
        assert all(
            entry["reason"].startswith("already rerun as ")
            for entry in again["skipped"]
        )
        assert len(reruns(manager)) == 3


def test_mcp_rerun_job_refuses_a_non_tombstone_with_or_without_consent(tmp_path):
    import pytest

    from dw_mcp import diagnose
    from dw_mcp.client import DwApiError

    with restarted_client(tmp_path) as (client, manager):
        job = submit(manager)
        wait_until(lambda: job.status == "succeeded")
        wait_until(lambda: stored_status(tmp_path, job.id) == "succeeded")
        dw = mcp_client(client)
        for acknowledged in (False, True):
            with pytest.raises(DwApiError) as caught:
                diagnose.rerun_job(
                    dw, job.id, restart_batch=True, acknowledged_cost=acknowledged
                )
            assert "not a restart tombstone" in str(caught.value)
        assert reruns(manager) == []


def test_list_rows_of_a_swept_batch_carry_restart_batch_and_reason(tmp_path):
    """The list says which rows are tombstones and of which batch (#1449):
    `restart_batch` is the `server_instance` the batch route reports."""
    ids = a_dead_server_with_three_jobs(tmp_path)
    with restarted_client(tmp_path) as (client, manager):
        preview = client.post(f"/api/jobs/{ids[1]}/rerun-restart-batch", json={})
        assert preview.status_code == 200, preview.text
        instance = preview.json()["server_instance"]

        listed = client.get("/api/jobs", params={"status": "cancelled"})
        assert listed.status_code == 200, listed.text
        rows = {row["id"]: row for row in listed.json()["jobs"]}
        for job_id in ids:
            assert rows[job_id]["restart_batch"] == instance
            assert rows[job_id]["reason"].startswith("server restart:")
            assert "\n" not in rows[job_id]["reason"]
        # The detail keeps the full error the reason summarises
        detail = client.get(f"/api/jobs/{ids[0]}").json()
        assert detail["error"].startswith(detail["reason"].rstrip("…"))
