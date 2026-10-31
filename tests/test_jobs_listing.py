"""Bounding the jobs list.

An unbounded listing is a dead tool over MCP: a server holding a few months
of history spilled 176 entries and the call failed before a single job id
could be read. The API grew `limit` and `status` (and always reports
`total`); the MCP tool bounds by default and answers newest first, since the
job worth looking at is almost always the last one.
"""

import httpx
import pytest

from dw_mcp import catalog
from dw_mcp.client import DwClient
from dw.server.job_history import JobHistory

from .test_server import (  # noqa: F401 - `server` is a fixture
    server,
    success_script,
    valid_workflow,
    wait_for_status,
)


def stored_job(history, job_id, status, created_at, workspace="default"):
    """A finished job straight into history - the rows a long-lived server
    accumulates, without running anything."""

    class Row:
        id = job_id
        workflow_name = "w"
        catalog_name = None
        created_at = None
        started_at = None
        finished_at = None
        manifest = []
        warnings = []
        error = None
        events = []
        run_id = None
        run_dir = None
        acknowledged = "none"

    row = Row()
    row.status = status
    row.created_at = created_at
    row.spec = {"workspace": workspace}
    history.record(row)


@pytest.fixture
def history(tmp_path):
    return JobHistory(str(tmp_path / "jobs.sqlite"))


def test_history_filters_by_status_in_sql(history):
    for index in range(5):
        stored_job(history, f"j{index}", "succeeded", index)
    stored_job(history, "bad", "failed", 99)

    failed = history.recent_summaries(statuses=["failed"])

    assert [row["id"] for row in failed] == ["bad"]
    # and the cap is spent on matching rows, not on rows the filter drops
    assert len(history.recent_summaries(limit=2, statuses=["succeeded"])) == 2


def test_history_filters_by_status_and_workspace_together(history):
    stored_job(history, "a", "failed", 1, workspace="qa")
    stored_job(history, "b", "failed", 2, workspace="other")

    rows = history.recent_summaries(statuses=["failed"], workspace="qa")

    assert [row["id"] for row in rows] == ["a"]


def test_listing_is_unchanged_without_the_new_parameters(server):
    """The web UI polls this route: oldest first, every job, same shape."""
    with server(success_script) as client:
        client.post("/api/jobs", json={"workflow": valid_workflow()})

        body = client.get("/api/jobs").json()

        assert [job["id"] for job in body["jobs"]]
        assert body["total"] == len(body["jobs"])


def test_limit_keeps_the_newest_and_total_reports_the_rest(server):
    with server(success_script) as client:
        ids = []
        for _ in range(3):
            job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
            wait_for_status(client, job["id"], {"succeeded"})
            ids.append(job["id"])

        body = client.get("/api/jobs", params={"limit": 1}).json()

        assert body["total"] == 3
        # the cut comes off the front: the newest job is what is kept
        assert [job["id"] for job in body["jobs"]] == [ids[-1]]
        assert client.get("/api/jobs", params={"limit": 0}).json()["jobs"] == []


def test_status_filter_narrows_the_listing(server):
    with server(success_script) as client:
        job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        wait_for_status(client, job["id"], {"succeeded"})

        succeeded = client.get("/api/jobs", params={"status": "succeeded"}).json()
        queued = client.get("/api/jobs", params={"status": "queued,running"}).json()

        assert [entry["id"] for entry in succeeded["jobs"]] == [job["id"]]
        assert queued["jobs"] == [] and queued["total"] == 0


def test_unknown_status_is_a_bad_request(server):
    with server(success_script) as client:
        response = client.get("/api/jobs", params={"status": "finished"})

        assert response.status_code == 400
        assert "finished" in response.json()["detail"]


def test_unknown_workspace_is_not_found_not_an_empty_listing(server, tmp_path):
    from dw.workspace import Workspace

    with server(success_script) as client:
        (tmp_path / "ws").mkdir()
        client.app.state.workspace_root = Workspace(str(tmp_path / "ws"), "flag")
        response = client.get("/api/jobs", params={"workspace": "qa-typo"})

        assert response.status_code == 404
        assert "No such workspace: qa-typo" in response.json()["detail"]
        assert client.get("/api/jobs", params={"workspace": "default"}).status_code == 200


def mcp_client(body):
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=body)

    return DwClient(transport=httpx.MockTransport(handler)), seen


def test_mcp_list_jobs_bounds_and_reverses():
    client, seen = mcp_client({"jobs": [{"id": "old"}, {"id": "new"}], "total": 176})

    answer = catalog.list_jobs(client)

    assert seen["params"]["limit"] == "20"
    assert [job["id"] for job in answer["jobs"]] == ["new", "old"]
    assert answer["returned"] == 2 and answer["total"] == 176
    assert answer["truncated"] is True
    assert "174 older jobs" in answer["next"]


def test_mcp_list_jobs_says_nothing_about_truncation_when_complete():
    client, _ = mcp_client({"jobs": [{"id": "a"}], "total": 1})

    answer = catalog.list_jobs(client)

    assert "truncated" not in answer and "next" not in answer


def test_mcp_list_jobs_passes_its_filters():
    client, seen = mcp_client({"jobs": [], "total": 0})

    catalog.list_jobs(client, limit=5, status=["failed", "cancelled"], workspace="qa")

    assert seen["params"] == {
        "limit": "5",
        "status": "failed,cancelled",
        "workspace": "qa",
    }


def test_limit_larger_than_the_match_returns_every_match(server):
    """A limit above the number of matching jobs is not a cut at all.

    The slice used to be taken from `len(jobs) - limit`, which goes negative
    once the limit exceeds what matched and then counts back from the end -
    so a limit just above what matched answered with only the difference -
    `limit=12` against 9 matching jobs returned 3 of them and called
    the other 6 "not listed".
    """
    with server(success_script) as client:
        ids = []
        for _ in range(3):
            job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
            wait_for_status(client, job["id"], {"succeeded"})
            ids.append(job["id"])

        body = client.get("/api/jobs", params={"limit": 4}).json()

        assert body["total"] == 3
        assert [job["id"] for job in body["jobs"]] == ids


def test_limit_larger_than_the_match_under_a_status_filter(server):
    """The shape the report came in: limit above the count, plus `status`."""
    with server(success_script) as client:
        ids = []
        for _ in range(2):
            job = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
            wait_for_status(client, job["id"], {"succeeded"})
            ids.append(job["id"])

        body = client.get(
            "/api/jobs", params={"limit": 3, "status": "succeeded"}
        ).json()

        assert body["total"] == 2
        assert [entry["id"] for entry in body["jobs"]] == ids


def test_history_summaries_survive_a_malformed_spec_row(history):
    stored_job(history, "good", "succeeded", 1)
    stored_job(history, "bad", "succeeded", 2)
    with history._connect() as conn:
        conn.execute("UPDATE jobs SET spec = '{not json' WHERE id = 'bad'")

    rows = history.recent_summaries()

    assert {row["id"] for row in rows} == {"good", "bad"}
    assert next(row for row in rows if row["id"] == "bad")["_output_dir"] is None


def test_list_page_reads_notes_only_for_the_page(tmp_path):
    from dw.server.jobs import JobManager

    manager = JobManager.__new__(JobManager)
    import threading

    manager._lock = threading.RLock()
    manager.jobs = {}
    manager._pending = []
    manager.history = JobHistory(str(tmp_path / "jobs.sqlite"))
    for index in range(6):
        stored_job(manager.history, f"j{index}", "succeeded", index)
    seen = []
    manager._current_note = lambda summary, out, run: (
        seen.append(summary["id"]) or summary
    )

    jobs, total = manager.list_page(limit=2)

    assert total == 6
    assert [j["id"] for j in jobs] == ["j4", "j5"]
    assert sorted(seen) == ["j4", "j5"]
