"""A job linked to a requests-ledger request and item (#1482): admission,
the record, history, rerun inheritance, exports, filters and the derived
item state."""

import json
import os
import shutil
import sqlite3
import zipfile  # noqa: F401 - kept with the export helpers' imports

import pytest
from fastapi.testclient import TestClient

from dw import ledger
from dw.runs import new_run_id
from dw.server.app import create_app
from dw.server.job_record import TERMINAL_STATES
from dw.server.jobs import JobManager
from dw.workspace import Workspace

from .test_ledger import item, plan
from .test_server import (
    ScriptedWorkerManager,
    admitted_for,
    valid_workflow,
    wait_for_status,
)

WS = "default"


def run_id_for(job_script_name):
    return new_run_id({"workflow": job_script_name})


class Scripts:
    """A worker script that finishes with one file in a fresh run directory,
    or fails, by the prompt the job was given."""

    def __init__(self):
        self.count = 0
        self.fail_next = False

    def __call__(self, command):
        self.count += 1
        if self.fail_next:
            self.fail_next = False
            yield {"type": "error", "message": "boom"}
            return
        output_dir = command["output_dir"]
        rid = run_id_for(f"link{self.count}")
        rel = f"server_test/{rid}"
        run_dir = os.path.join(output_dir, "server_test", rid)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "still.png"), "wb") as file:
            file.write(b"an image")
        with open(os.path.join(run_dir, "manifest.json"), "w") as file:
            json.dump(
                {
                    "run_id": rid,
                    "version": 4,
                    "status": "completed",
                    "workflow": {"identity": "server_test"},
                    "steps": [{"step": "gen", "files": ["still.png"]}],
                },
                file,
            )
        yield {
            "type": "progress",
            "event": "run_start",
            "run_id": rid,
            "version": 4,
            "identity": "server_test",
            "run_dir": rel,
        }
        yield {
            "type": "success",
            "message": "ok",
            "run_count": 1,
            "manifest": [
                {"step": "gen", "files": [os.path.join(run_dir, "still.png")]}
            ],
        }


@pytest.fixture
def root(tmp_path):
    return Workspace(tmp_path / "studio", "flag").ensure()


@pytest.fixture
def client(root, tmp_path):
    script = Scripts()
    manager = JobManager(
        root.outputs,
        worker_manager=ScriptedWorkerManager(script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=root.workflows,
    )
    app = create_app(
        workflow_dir=root.workflows,
        output_dir=root.outputs,
        job_manager=manager,
        prompt_dir=root.prompts,
        asset_dir=root.assets,
        workspace=root.root,
    )
    c = TestClient(app, base_url="http://localhost")
    c.manager = manager
    c.script = script
    c.root = root
    return c


def write_plan(client, request_id="cats", revision=0, items=None, **fields):
    body = plan(revision=revision, items=items, **fields)
    body["by"] = "agent"
    r = client.put(f"/api/workspaces/{WS}/requests/{request_id}", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def view(client, request_id="cats"):
    r = client.get(f"/api/workspaces/{WS}/requests/{request_id}")
    assert r.status_code == 200, r.text
    return r.json()


def items_of(client, request_id="cats"):
    return {i["key"]: i for i in view(client, request_id)["items"]}


def submit(client, prompt="hi", request="cats", item_key="a", expect=201, **extra):
    body = {"workflow": valid_workflow(), "arguments": {"prompt": prompt}}
    if request is not None:
        body["request"] = request
    if item_key is not None:
        body["item"] = item_key
    body.update(extra)
    r = client.post("/api/jobs", json=body)
    assert r.status_code == expect, r.text
    return r.json()


def finish(client, job):
    return wait_for_status(client, job["id"], TERMINAL_STATES)


@pytest.fixture
def planned(client):
    write_plan(client)
    return client


# --- admission ------------------------------------------------------------


class TestAdmission:
    def test_a_linked_submit_carries_both_through_to_history(self, planned):
        job = submit(planned)
        assert job["request"] == "cats" and job["item"] == "a"
        done = finish(planned, job)
        assert done["request"] == "cats" and done["item"] == "a"
        planned.manager.jobs.pop(job["id"])
        again = planned.get(f"/api/jobs/{job['id']}").json()
        assert again["request"] == "cats" and again["item"] == "a"

    def test_an_unlinked_job_has_neither_key(self, planned):
        job = submit(planned, request=None, item_key=None)
        assert "request" not in job and "item" not in job

    def test_the_unknown_request(self, planned):
        r = planned.post(
            "/api/jobs",
            json={"workflow": valid_workflow(), "request": "nope", "item": "a"},
        )
        assert r.status_code == 422
        assert r.json()["detail"]["code"] == "unknown_request"
        assert planned.manager.jobs == {}

    def test_the_unknown_item(self, planned):
        r = planned.post(
            "/api/jobs",
            json={"workflow": valid_workflow(), "request": "cats", "item": "zzz"},
        )
        assert r.status_code == 422
        assert r.json()["detail"]["code"] == "unknown_item"
        assert planned.manager.jobs == {}

    def test_a_closed_request(self, planned):
        write_plan(planned, revision=1, state="done")
        r = planned.post(
            "/api/jobs",
            json={"workflow": valid_workflow(), "request": "cats", "item": "a"},
        )
        assert r.status_code == 422
        assert r.json()["detail"]["code"] == "request_closed"
        assert planned.manager.jobs == {}

    def test_a_dropped_item(self, planned):
        write_plan(planned, revision=1, items=[item("a")])
        submit(planned, item_key="a")
        write_plan(planned, revision=2, items=[item("b")])
        assert items_of(planned)["a"]["status"] == "dropped"
        r = planned.post(
            "/api/jobs",
            json={"workflow": valid_workflow(), "request": "cats", "item": "a"},
        )
        assert r.status_code == 422
        assert r.json()["detail"]["code"] == "item_dropped"

    def test_an_item_without_a_request_is_refused(self, planned):
        r = planned.post("/api/jobs", json={"workflow": valid_workflow(), "item": "a"})
        assert r.status_code == 422

    def test_a_request_without_an_item_is_refused(self, planned):
        r = planned.post(
            "/api/jobs", json={"workflow": valid_workflow(), "request": "cats"}
        )
        assert r.status_code == 422

    def test_a_bad_key_is_refused(self, planned):
        r = planned.post(
            "/api/jobs",
            json={"workflow": valid_workflow(), "request": "Demo", "item": "a"},
        )
        assert r.status_code == 422
        assert planned.manager.jobs == {}


# --- reruns ---------------------------------------------------------------


class TestRerun:
    def test_a_single_rerun_inherits_the_link(self, planned):
        job = submit(planned)
        finish(planned, job)
        r = planned.post(f"/api/jobs/{job['id']}/rerun", json={})
        assert r.status_code == 201, r.text
        rerun = r.json()
        assert rerun["request"] == "cats" and rerun["item"] == "a"
        assert not any("rerun_ledger_link" in w for w in rerun.get("warnings", []))

    def test_a_rerun_into_a_closed_request_runs_with_a_warning(self, planned):
        job = submit(planned)
        finish(planned, job)
        write_plan(planned, revision=1, state="done")
        r = planned.post(f"/api/jobs/{job['id']}/rerun", json={})
        assert r.status_code == 201, r.text
        rerun = r.json()
        assert rerun["request"] == "cats" and rerun["item"] == "a"
        assert any(
            w.startswith("rerun_ledger_link_request_closed") for w in rerun["warnings"]
        )

    def test_a_rerun_into_a_dropped_item_runs_with_a_warning(self, planned):
        job = submit(planned)
        finish(planned, job)
        write_plan(planned, revision=1, items=[item("b")])
        r = planned.post(f"/api/jobs/{job['id']}/rerun", json={})
        assert r.status_code == 201, r.text
        rerun = r.json()
        assert rerun["item"] == "a"
        assert any(
            w.startswith("rerun_ledger_link_item_dropped") for w in rerun["warnings"]
        )

    def test_a_restart_batch_rerun_inherits_the_link(self, planned, tmp_path):
        # A server that died with a linked job open: its tombstone's batch
        # is rerun by a fresh server over the same history
        from .test_restart_tombstones import (
            StuckWorkerManager,
            kill,
            stuck_script,
            wait_until,
        )

        root = planned.root
        dead = JobManager(
            root.outputs,
            worker_manager=StuckWorkerManager(stuck_script),
            history_path=str(tmp_path / "jobs.sqlite"),
            workflow_dir=root.workflows,
        )
        wf = valid_workflow()
        spec_job = dead.submit(
            admitted=admitted_for(dead, wf),
            workflow=wf,
            arguments={"prompt": "x"},
            base_dir=root.workflows,
            workflow_dir=root.workflows,
            request="cats",
            item="a",
        )
        wait_until(lambda: spec_job.run_dir is not None)
        kill(dead)

        manager = JobManager(
            root.outputs,
            worker_manager=ScriptedWorkerManager(Scripts()),
            history_path=str(tmp_path / "jobs.sqlite"),
            workflow_dir=root.workflows,
        )
        app = create_app(
            workflow_dir=root.workflows,
            output_dir=root.outputs,
            job_manager=manager,
            prompt_dir=root.prompts,
            asset_dir=root.assets,
            workspace=root.root,
        )
        c = TestClient(app, base_url="http://localhost")
        try:
            r = c.post(
                f"/api/jobs/{spec_job.id}/rerun-restart-batch",
                json={"acknowledged_cost": True},
            )
            assert r.status_code == 200, r.text
            queued = r.json()["queued"]
            assert len(queued) == 1
            made = manager.jobs[queued[0]["id"]]
            assert (made.request, made.item) == ("cats", "a")
        finally:
            manager.shutdown()


# --- history --------------------------------------------------------------


class TestHistory:
    def test_forget_workspace_keeps_the_strings(self, planned, tmp_path):
        job = submit(planned)
        finish(planned, job)
        history = planned.manager.history
        history.forget_workspace(WS)
        con = sqlite3.connect(tmp_path / "jobs.sqlite")
        try:
            rows = con.execute(
                "SELECT request, item FROM jobs WHERE id = ?", (job["id"],)
            ).fetchall()
        finally:
            con.close()
        assert rows == [("cats", "a")]


# --- exports --------------------------------------------------------------


def test_an_exported_job_json_carries_both(planned):
    job = submit(planned)
    finish(planned, job)
    r = planned.post(f"/api/jobs/{job['id']}/export")
    assert r.status_code == 201, r.text
    directory = r.json()["directory"]
    path = os.path.join(directory, "job.json")
    if not os.path.isabs(path):
        path = os.path.join(planned.root.root, path)
    with open(path) as file:
        record = json.load(file)
    assert record["request"] == "cats" and record["item"] == "a"


# --- filters --------------------------------------------------------------


class TestFilters:
    def test_the_job_list_filters_by_request_and_item(self, planned):
        write_plan(planned, revision=1, items=[item("a"), item("b")])
        write_plan(planned, request_id="dogs")
        a = submit(planned, item_key="a")
        b = submit(planned, item_key="b")
        d = submit(planned, request="dogs", item_key="a")
        loose = submit(planned, request=None, item_key=None)
        for j in (a, b, d, loose):
            finish(planned, j)

        def ids(query):
            r = planned.get(f"/api/jobs?{query}")
            assert r.status_code == 200, r.text
            return {j["id"] for j in r.json()["jobs"]}

        assert ids("request=cats") == {a["id"], b["id"]}
        assert ids("request=cats&item=b") == {b["id"]}
        assert ids("request=dogs") == {d["id"]}

    def test_item_without_request_is_422(self, planned):
        assert planned.get("/api/jobs?item=a").status_code == 422

    def test_a_bad_key_in_the_filter_is_422(self, planned):
        assert planned.get("/api/jobs?request=Demo").status_code == 422

    def test_the_gallery_filters_by_request(self, planned):
        write_plan(planned, request_id="dogs")
        a = submit(planned)
        d = submit(planned, request="dogs", item_key="a")
        loose = submit(planned, request=None, item_key=None)
        done = {j["id"]: finish(planned, j) for j in (a, d, loose)}
        r = planned.get("/api/gallery")
        assert r.status_code == 200, r.text
        assert r.json()["total"] >= 3
        r = planned.get("/api/gallery?request=cats")
        assert r.status_code == 200, r.text
        entries = r.json()["files"]
        assert len(entries) == 1
        assert (
            entries[0]["run_id"] == planned.manager.get(a["id"]).run_dir.split("/")[-1]
        )
        assert done
        assert planned.get("/api/gallery?request=Demo").status_code == 422

    def test_the_gallery_filters_by_item(self, planned):
        a = submit(planned, item_key="a")
        b = submit(planned, item_key="b")
        for j in (a, b):
            finish(planned, j)

        def runs(query):
            r = planned.get(f"/api/gallery?{query}")
            assert r.status_code == 200, r.text
            return {e["run_id"] for e in r.json()["files"]}

        run_of = {
            j["id"]: planned.manager.get(j["id"]).run_dir.split("/")[-1] for j in (a, b)
        }
        assert runs("request=cats") == {run_of[a["id"]], run_of[b["id"]]}
        assert runs("request=cats&item=b") == {run_of[b["id"]]}
        assert runs("request=cats&item=a") == {run_of[a["id"]]}
        assert planned.get("/api/gallery?item=a").status_code == 422
        assert planned.get("/api/gallery?request=cats&item=B").status_code == 422


# --- derived state --------------------------------------------------------


class TestDerivedState:
    def test_a_planned_item_has_no_attempts(self, planned):
        a = items_of(planned)["a"]
        assert a["status"] == "planned"
        assert a["attempts"] == [] and a["result"] is None
        assert a["spent_minutes"] == 0
        assert "accepted_missing" not in a

    def test_the_latest_succeeded_job_is_the_result_until_one_is_accepted(
        self, planned
    ):
        first = finish(planned, submit(planned))
        second = finish(planned, submit(planned))
        a = items_of(planned)["a"]
        assert a["status"] == "done"
        assert [x["job"] for x in a["attempts"]] == [first["id"], second["id"]]
        assert a["result"]["job"] == second["id"]

        body = view(planned)
        body["items"][0]["accepted"] = {"job": first["id"]}
        r = planned.put(
            f"/api/workspaces/{WS}/requests/cats",
            json={k: body[k] for k in ("title", "goal", "items", "revision")}
            | {"why": "pick the first", "by": "user"},
        )
        assert r.status_code == 200, r.text
        assert {i["key"]: i for i in r.json()["items"]}["a"]["result"]["job"] == first[
            "id"
        ]

    def test_accepted_as_a_bare_job_id_is_taken(self, planned):
        first = finish(planned, submit(planned))
        finish(planned, submit(planned))
        body = view(planned)
        body["items"][0]["accepted"] = first["id"]
        r = planned.put(
            f"/api/workspaces/{WS}/requests/cats",
            json={k: body[k] for k in ("title", "goal", "items", "revision")}
            | {"why": "pick", "by": "user"},
        )
        assert r.status_code == 200, r.text
        assert items_of(planned)["a"]["result"]["job"] == first["id"]

    def _accept(self, client, job_id, key="a"):
        body = view(client)
        for i in body["items"]:
            if i["key"] == key:
                i["accepted"] = {"job": job_id}
        return client.put(
            f"/api/workspaces/{WS}/requests/cats",
            json={k: body[k] for k in ("title", "goal", "items", "revision")}
            | {"why": "pick", "by": "user"},
        )

    def test_accepted_on_a_job_of_another_item_is_refused(self, planned):
        other = finish(planned, submit(planned, item_key="b"))
        r = self._accept(planned, other["id"], key="a")
        assert r.status_code in (400, 422), r.text
        assert "not linked" in json.dumps(r.json())

    def test_accepted_on_a_job_of_another_request_is_refused(self, planned):
        write_plan(planned, request_id="dogs")
        other = finish(planned, submit(planned, request="dogs", item_key="a"))
        r = self._accept(planned, other["id"])
        assert r.status_code in (400, 422), r.text

    def test_accepted_on_a_failed_job_is_refused(self, planned):
        planned.script.fail_next = True
        failed = finish(planned, submit(planned))
        assert failed["status"] == "failed"
        assert items_of(planned)["a"]["status"] == "failed"
        r = self._accept(planned, failed["id"])
        assert r.status_code in (400, 422), r.text
        assert "not succeeded" in json.dumps(r.json())

    def test_accepted_missing_when_the_accepted_run_is_deleted(self, planned):
        first = finish(planned, submit(planned))
        second = finish(planned, submit(planned))
        assert self._accept(planned, first["id"]).status_code == 200
        run_dir = planned.manager.get(first["id"]).run_dir
        shutil.rmtree(os.path.join(planned.root.outputs, run_dir))
        a = items_of(planned)["a"]
        assert a["accepted_missing"] is True
        assert a["result"]["job"] == second["id"]
        assert a["status"] == "done"

    def test_status_order(self):
        def job(status, key="a", **kw):
            return {
                "job": status,
                "item": key,
                "status": status,
                "run_exists": True,
                **kw,
            }

        def status_of(jobs, **item_fields):
            req = {"items": [{"key": "a", **item_fields}]}
            return ledger.request_view(req, jobs)["items"][0]["status"]

        assert status_of([]) == "planned"
        assert status_of([job("failed"), job("cancelled")]) == "failed"
        assert status_of([job("failed"), job("queued")]) == "queued"
        assert status_of([job("queued"), job("running")]) == "running"
        assert status_of([job("running"), job("succeeded")]) == "done"
        assert status_of([job("succeeded")], dropped=True) == "dropped"

    def test_spent_and_remaining(self):
        req = {
            "items": [
                {"key": "a", "estimate_minutes": 10},
                {"key": "b", "estimate_minutes": 5},
                {"key": "c", "estimate_minutes": 7},
                {"key": "d", "estimate_minutes": 3, "dropped": True},
            ]
        }
        jobs = [
            {
                "job": "j1",
                "item": "a",
                "status": "succeeded",
                "run_exists": True,
                "started_at": 100,
                "finished_at": 220,
            },
            {
                "job": "j2",
                "item": "b",
                "status": "failed",
                "started_at": 300,
                "finished_at": 330,
            },
            {"job": "j3", "item": "b", "status": "queued"},
        ]
        view_ = ledger.request_view(req, jobs)
        assert view_["spent_minutes"] == 2.5
        assert view_["remaining_minutes"] == 12
        by_key = {i["key"]: i for i in view_["items"]}
        assert by_key["a"]["spent_minutes"] == 2
        assert by_key["b"]["spent_minutes"] == 0.5
        assert by_key["b"]["attempts"][1]["minutes"] is None
        summary = ledger.summary_view({"id": "r", **req}, jobs)
        assert summary["spent_minutes"] == 2.5
        assert summary["remaining_minutes"] == 12

    def test_the_request_view_and_list_carry_the_totals(self, planned):
        finish(planned, submit(planned))
        body = view(planned)
        assert "spent_minutes" in body and "remaining_minutes" in body
        listed = planned.get(f"/api/workspaces/{WS}/requests").json()["requests"]
        assert listed[0]["spent_minutes"] >= 0
        assert listed[0]["status_counts"].get("done") == 1


# --- one real-HTTP walk ---------------------------------------------------


def test_submit_then_history_then_request_view(planned):
    job = submit(planned, note="the first take")
    finish(planned, job)
    listed = planned.get("/api/jobs?request=cats").json()["jobs"]
    assert [j["id"] for j in listed] == [job["id"]]
    a = items_of(planned)["a"]
    assert a["status"] == "done"
    assert a["attempts"][0]["job"] == job["id"]
    assert a["attempts"][0]["note"] == "the first take"
    assert a["result"]["job"] == job["id"]
    assert items_of(planned)["b"]["status"] == "planned"
