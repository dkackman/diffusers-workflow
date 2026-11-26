"""A take's `note` in a request's view is its run's current note (#1733):
an edit made after the run shows, a cleared one is gone, and the polled
lists read no notes.json."""

import pytest

from dw.server import jobs as jobs_module

from . import test_ledger_job_link as link
from .test_ledger_job_link import WS, finish, submit, view
from .test_ledger_page_actions import put

client, root, planned = link.client, link.root, link.planned

URL = f"/api/workspaces/{WS}/requests/cats"


def taken(client, live):
    """One finished take on item `a` submitted with a note; held by the
    manager (`live`) or only in history. Returns its run directory name."""
    job = finish(client, submit(client, note="tested X"))
    run_dir = client.manager.jobs[job["id"]].run_dir
    if not live:
        client.manager.jobs.pop(job["id"])
    return run_dir


def set_note(client, run_dir, note):
    r = client.put("/api/notes", json={"target": run_dir, "note": note})
    assert r.status_code == 200, r.text


def attempt(found):
    return found["items"][0]["attempts"][0]


@pytest.mark.parametrize("live", [True, False])
def test_an_edited_note_shows(planned, live):
    run_dir = taken(planned, live)
    assert attempt(view(planned))["note"] == "tested X"
    set_note(planned, run_dir, "tested X, edited")
    assert attempt(view(planned))["note"] == "tested X, edited"


@pytest.mark.parametrize("live", [True, False])
def test_a_cleared_note_is_gone(planned, live):
    set_note(planned, taken(planned, live), None)
    assert "note" not in attempt(view(planned))


def test_a_write_answers_with_the_current_note(planned):
    set_note(planned, taken(planned, False), "edited")
    assert attempt(put(planned, view(planned), "reorder"))["note"] == "edited"


def test_the_lists_read_no_notes(planned, monkeypatch):
    taken(planned, False)

    def read(*args):
        raise AssertionError("a list read a run's notes")

    monkeypatch.setattr(jobs_module, "with_run_note", read)
    for url in (f"/api/workspaces/{WS}/requests?needs=true", "/api/requests"):
        r = planned.get(url)
        assert r.status_code == 200, r.text
        assert r.json()["requests"]
