"""A list row's `needs` (RQ-4): what in a request wants the user, by reason,
and how many notes wait on the agent - only when the list is asked
`needs=true`. The rule is the request page's `planState`
(ui/src/lib/request/itemState.ts), pinned by the shared cases in
tests/fixtures/request_needs_cases.json that needs.twin.test.ts runs too."""

import json
import pathlib
from datetime import datetime, timedelta, timezone

import pytest

from dw import ledger, ledger_views

from . import test_ledger_job_link as link
from .test_ledger import item, plan, write
from .test_ledger_job_link import WS, write_plan

client, root, planned = link.client, link.root, link.planned

CASES = json.loads(
    (
        pathlib.Path(__file__).resolve().parent
        / "fixtures"
        / "request_needs_cases.json"
    ).read_text()
)["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_every_shared_case_gives_its_needs(case):
    assert ledger_views.needs_counts(case["request"]) == {
        **case["needs"],
        "notes_waiting": case["notes_waiting"],
    }


def job(key, n, status="succeeded", **fields):
    return {
        "job": f"job-{key}-{n}",
        "item": key,
        "status": status,
        "run_id": f"r-{key}-{n}",
        "run_exists": True,
        "created_at": n,
        "started_at": n,
        "finished_at": n + 60,
        "output": None,
        **fields,
    }


def stored_request(tmp_path):
    """A hold, a runnable failed item, a failed item blocked on it, two
    unaccepted takes, and a note left after the agent's write."""
    until = datetime.now(timezone.utc) + timedelta(hours=1)
    items = [
        item("held", workflow="shot", review_until=until.isoformat(timespec="seconds")),
        item("plate", workflow="shot"),
        item("comp", ["plate"], workflow="shot"),
        item("takes", workflow="shot"),
    ]
    write(tmp_path, plan(items=items))
    ledger.add_note(
        str(tmp_path),
        "cats",
        item="takes",
        text="the second, please",
        refs=[],
        check_ref=lambda request_id, ref: ref,
    )
    jobs = [
        job("plate", 1, "failed"),
        job("comp", 2, "failed"),
        job("takes", 3),
        job("takes", 4),
    ]
    return ledger._load(str(tmp_path))["cats"], jobs


def test_the_summary_needs_match_the_view(tmp_path):
    request, jobs = stored_request(tmp_path)
    needs = ledger_views.summary_view(request, jobs, needs=True)["needs"]
    assert needs == ledger_views.needs_counts(ledger_views.request_view(request, jobs))
    assert needs == {
        "held": 1,
        "ready": 0,
        "failed": 1,
        "revised": 0,
        "choose": 1,
        "accepted_missing": 0,
        "notes_waiting": 1,
    }


def test_a_summary_reads_no_proof_for_needs(tmp_path):
    request, jobs = stored_request(tmp_path)

    def proof():
        raise AssertionError("a proof was read")

    # the succeeded takes' runs are gone, so the cover reads no proof either
    jobs = [
        {**j, "output": proof, "run_exists": j["status"] != "succeeded"} for j in jobs
    ]
    needs = ledger_views.summary_view(request, jobs, needs=True)["needs"]
    assert (needs["held"], needs["failed"], needs["choose"]) == (1, 1, 1)


def test_the_list_says_nothing_about_needs_unless_asked(client):
    write_plan(client, items=[item("a", workflow="shot")])
    url = f"/api/workspaces/{WS}/requests"
    [row] = client.get(url).json()["requests"]
    assert "needs" not in row
    [row] = client.get(url, params={"needs": "true"}).json()["requests"]
    assert row["needs"] == {
        "held": 0,
        "ready": 1,
        "failed": 0,
        "revised": 0,
        "choose": 0,
        "accepted_missing": 0,
        "notes_waiting": 0,
    }


def test_the_open_requests_list_never_carries_needs(planned):
    for params in ({}, {"needs": "true"}):
        rows = planned.get("/api/requests", params=params).json()["requests"]
        assert rows and all("needs" not in row for row in rows)


def test_a_closed_request_needs_nothing(planned):
    found = link.view(planned)
    write_plan(planned, revision=found["revision"], state="done")
    url = f"/api/workspaces/{WS}/requests"
    [row] = planned.get(url, params={"needs": "true"}).json()["requests"]
    assert set(row["needs"].values()) == {0}


def test_a_real_ledger_counts_ready_revised_and_accepted_missing(tmp_path):
    """The rest of the reasons, off a stored request rather than the shared
    fixture: a runnable item, one whose only take ran another workflow than its plan names, and
    one whose accepted run is gone."""
    write(
        tmp_path,
        plan(
            items=[
                item("fresh", workflow="shot"),
                item("stale", workflow="shot"),
                item("kept", workflow="shot"),
            ]
        ),
    )
    request = ledger._load(str(tmp_path))["cats"]
    for stored in request["items"]:
        if stored["key"] == "kept":
            stored["accepted"] = {"job": "job-gone"}
    later = 4_000_000_000  # after any plan stamp
    jobs = [
        job("stale", 1, workflow="older-workflow"),
        job("kept", 1, created_at=later, started_at=later, finished_at=later),
    ]
    assert ledger_views.needs_counts(ledger_views.request_view(request, jobs)) == {
        "held": 0,
        "ready": 1,
        "failed": 0,
        "revised": 1,
        "choose": 0,
        "accepted_missing": 1,
        "notes_waiting": 0,
    }
