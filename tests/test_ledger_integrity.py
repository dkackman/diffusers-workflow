"""The requests ledger's integrity (0.21.0 review): the atomic write, the
file's shape, the cross-process lock, the caps that no longer wedge a
workspace, and deleting a request."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from dw import ledger, runs
from dw.runs import write_json_atomic
from dw.server.app import create_app
from dw.server.jobs import JobManager
from dw.workspace import REQUESTS_FILE_NAME, Workspace

from .test_ledger import URL, item, plan, refused, write
from .test_server import ScriptedWorkerManager, success_script


@pytest.fixture
def workspace_root(tmp_path):
    return Workspace(tmp_path / "studio", "flag").ensure()


@pytest.fixture
def client(workspace_root, tmp_path):
    manager = JobManager(
        workspace_root.outputs,
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=workspace_root.workflows,
    )
    app = create_app(
        workflow_dir=workspace_root.workflows,
        output_dir=workspace_root.outputs,
        job_manager=manager,
        prompt_dir=workspace_root.prompts,
        asset_dir=workspace_root.assets,
        workspace=workspace_root.root,
    )
    with TestClient(app, base_url="http://localhost") as c:
        c.manager = manager
        yield c


class TestAtomicWrite:
    def test_the_file_is_synced_before_it_replaces_the_old_one(
        self, tmp_path, monkeypatch
    ):
        order = []
        real_fsync, real_replace = os.fsync, os.replace
        monkeypatch.setattr(
            runs.os, "fsync", lambda fd: (order.append("fsync"), real_fsync(fd))[1]
        )
        monkeypatch.setattr(
            runs.os,
            "replace",
            lambda a, b: (order.append("replace"), real_replace(a, b))[1],
        )
        write_json_atomic(str(tmp_path / "x.json"), {"a": 1})
        assert order == ["fsync", "replace"]
        assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1}

    def test_a_value_json_cannot_hold_leaves_no_temp_file(self, tmp_path):
        (tmp_path / "x.json").write_text('{"kept": true}')
        with pytest.raises(ValueError):
            # a cycle: json raises ValueError, which is not an OSError
            cycle = {}
            cycle["self"] = cycle
            write_json_atomic(str(tmp_path / "x.json"), cycle)
        assert os.listdir(tmp_path) == ["x.json"]
        assert json.loads((tmp_path / "x.json").read_text()) == {"kept": True}

    def test_a_failed_replace_leaves_no_temp_file(self, tmp_path, monkeypatch):
        def refuse(a, b):
            raise PermissionError("no")

        monkeypatch.setattr(runs.os, "replace", refuse)
        with pytest.raises(OSError):
            write_json_atomic(str(tmp_path / "x.json"), {"a": 1})
        assert os.listdir(tmp_path) == []


class TestReservedNames:
    """A workspace is a folder at the root, and the default workspace keeps
    its ledger and its note as files there: a workspace of either name would
    be a directory where the file goes."""

    @pytest.mark.parametrize("name", ["requests.json", "workspace.json"])
    def test_a_workspace_cannot_take_a_root_file_name(self, tmp_path, name):
        from dw.security import InvalidInputError
        from dw.workspace import create_workspace

        root = Workspace(tmp_path / "studio", "flag").ensure()
        with pytest.raises(InvalidInputError):
            create_workspace(root, name)
        assert not os.path.exists(os.path.join(root.root, name))

    def test_the_reserved_names_include_both_files(self):
        from dw.workspace import (
            REQUESTS_FILE_NAME,
            RESERVED_WORKSPACE_NAMES,
            WORKSPACE_FILE_NAME,
        )

        assert REQUESTS_FILE_NAME in RESERVED_WORKSPACE_NAMES
        assert WORKSPACE_FILE_NAME in RESERVED_WORKSPACE_NAMES


# --- the file's shape -----------------------------------------------------

GOOD = {
    "id": "cats",
    "title": "t",
    "state": "open",
    "revision": 1,
    "prompt": None,
    "goal": "",
    "items": [],
    "history": [],
}

WRONG_SHAPES = [
    [],
    "text",
    5,
    None,
    {},
    {"requests": []},
    {"requests": {"cats": 3}},
    {"requests": {"cats": {**GOOD, "items": 5}}},
    {"requests": {"cats": {**GOOD, "items": ["a"]}}},
    {"requests": {"cats": {**GOOD, "items": [{"title": "no key"}]}}},
    {"requests": {"cats": {**GOOD, "history": "x"}}},
    {"requests": {"cats": {**GOOD, "state": "paused"}}},
    {"requests": {"cats": {**GOOD, "revision": "1"}}},
    {"requests": {"cats": {**GOOD, "id": "dogs"}}},
]


class TestShape:
    @pytest.mark.parametrize("document", WRONG_SHAPES, ids=repr)
    def test_valid_json_of_the_wrong_shape_is_never_replaced(self, tmp_path, document):
        path = tmp_path / REQUESTS_FILE_NAME
        text = json.dumps(document)
        path.write_text(text)
        assert ledger.list_requests(str(tmp_path)) == []
        error = refused(tmp_path, plan(), code="ledger_unreadable")
        assert error.status == 409
        assert str(tmp_path) not in str(error)
        refused(tmp_path, plan(), code="ledger_unreadable", request_id="other")
        assert path.read_text() == text

    def test_a_well_shaped_file_reads(self, tmp_path):
        (tmp_path / REQUESTS_FILE_NAME).write_text(json.dumps({"requests": {}}))
        assert write(tmp_path, plan())["revision"] == 1

    @pytest.mark.parametrize("document", WRONG_SHAPES, ids=repr)
    def test_over_http_a_read_says_so_and_never_500s(
        self, client, workspace_root, document
    ):
        path = os.path.join(workspace_root.root, REQUESTS_FILE_NAME)
        with open(path, "w") as file:
            json.dump(document, file)
        listing = client.get(URL)
        assert listing.status_code == 200, listing.text
        body = listing.json()
        assert body["requests"] == []
        assert REQUESTS_FILE_NAME in body["unreadable"]
        assert workspace_root.root not in listing.text
        for reply in (
            client.get(f"{URL}/cats"),
            client.put(f"{URL}/cats", json=plan()),
        ):
            assert reply.status_code == 409, reply.text
            assert reply.json()["detail"]["code"] == "ledger_unreadable"
            assert workspace_root.root not in reply.text

    def test_a_readable_ledger_lists_without_the_key(self, client):
        assert "unreadable" not in client.get(URL).json()

    def test_a_job_cannot_link_into_an_unreadable_ledger(self, tmp_path):
        (tmp_path / REQUESTS_FILE_NAME).write_text("[]")
        problem = ledger.link_problem(str(tmp_path), "cats", "a")
        assert (problem.status, problem.code) == (409, "ledger_unreadable")


# --- two servers on one workspace -------------------------------------------

_WRITER = """
import sys
from dw import ledger
root, prefix, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
for n in range(count):
    ledger.put_request(
        root,
        f"{prefix}{n}",
        {"title": "t", "goal": "", "items": [], "revision": 0, "why": "w"},
        by="agent",
    )
"""


class TestCrossProcess:
    def test_a_write_waits_for_another_process_holding_the_lock(
        self, tmp_path, monkeypatch
    ):
        from filelock import FileLock

        from dw.workspace import REQUESTS_LOCK_NAME

        write(tmp_path, plan())
        before = (tmp_path / REQUESTS_FILE_NAME).read_text()
        monkeypatch.setattr(ledger, "LOCK_TIMEOUT", 0.2)
        # another FileLock object on the same file is what a second server
        # holds: it excludes this one exactly as another process does
        with FileLock(str(tmp_path / REQUESTS_LOCK_NAME), thread_local=False):
            error = refused(tmp_path, plan(revision=1), code="ledger_busy")
            assert error.status == 503
            assert str(tmp_path) not in str(error)
            assert (tmp_path / REQUESTS_FILE_NAME).read_text() == before
        assert write(tmp_path, plan(revision=1))["revision"] == 2

    def test_two_processes_writing_different_requests_lose_none(self, tmp_path):
        import subprocess
        import sys

        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = {**os.environ, "PYTHONPATH": repo}
        writers = [
            subprocess.Popen(
                [sys.executable, "-c", _WRITER, str(tmp_path), prefix, "15"], env=env
            )
            for prefix in ("a", "b", "c")
        ]
        assert [w.wait(timeout=120) for w in writers] == [0, 0, 0]
        stored = ledger.read_requests(str(tmp_path))
        assert len(stored) == 45

    def test_the_lock_file_does_not_stop_a_workspace_being_deleted(self, client):
        client.post("/api/workspaces", json={"name": "shots"})
        assert (
            client.put("/api/workspaces/shots/requests/cats", json=plan()).status_code
            == 200
        )
        deleted = client.delete("/api/workspaces/shots?acknowledged=true")
        assert deleted.status_code == 200, deleted.text


# --- deleting a request ------------------------------------------------------


class TestDelete:
    def test_only_the_ledger_entry_goes(self, tmp_path):
        write(tmp_path, plan())
        write(tmp_path, plan(), request_id="dogs")
        answer = ledger.delete_request(str(tmp_path), "cats", 1)
        assert answer == {
            "id": "cats",
            "deleted": True,
            "revision": 1,
            "linked_jobs": 0,
        }
        assert list(ledger.read_requests(str(tmp_path))) == ["dogs"]

    def test_refusals(self, tmp_path):
        def refuses(code, status, *args, **kw):
            with pytest.raises(ledger.LedgerError) as e:
                ledger.delete_request(*args, **kw)
            assert (e.value.code, e.value.status) == (code, status), str(e.value)
            return e.value

        refuses("no_workspace_root", 409, None, "cats", 1)
        refuses("unknown_request", 404, str(tmp_path), "cats", 1)
        write(tmp_path, plan())
        refuses("invalid_request", 400, str(tmp_path), "Cats", 1)
        refuses("invalid_request", 400, str(tmp_path), "cats", True)
        stale = refuses("stale_revision", 409, str(tmp_path), "cats", 2)
        assert stale.detail()["current_revision"] == 1
        jobs = [
            {"job": "j1", "item": "a", "status": "succeeded"},
            {"job": "j2", "item": "a", "status": "queued"},
            {"job": "j3", "item": "b", "status": "running"},
        ]
        active = refuses(
            "request_has_active_jobs",
            409,
            str(tmp_path),
            "cats",
            1,
            jobs_for=lambda rid: jobs,
        )
        assert active.detail()["jobs"] == ["j2", "j3"]
        assert "cats" in ledger.read_requests(str(tmp_path))

    def test_an_unreadable_ledger_is_not_touched(self, tmp_path):
        (tmp_path / REQUESTS_FILE_NAME).write_text("[]")
        with pytest.raises(ledger.LedgerError) as e:
            ledger.delete_request(str(tmp_path), "cats", 1)
        assert e.value.code == "ledger_unreadable"
        assert (tmp_path / REQUESTS_FILE_NAME).read_text() == "[]"


# --- caps that do not wedge a workspace ------------------------------------


def _linked(*keys):
    return lambda rid, key: (
        [{"job": f"job-{key}", "status": "succeeded", "run_id": "r"}]
        if key in keys
        else []
    )


class TestCaps:
    def test_closed_requests_do_not_count_toward_the_open_cap(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(ledger, "MAX_REQUESTS", 3)
        for n in range(3):
            write(tmp_path, plan(items=[]), request_id=f"r{n}")
        error = refused(tmp_path, plan(items=[]), code="ledger_full")
        assert error.status == 409 and "delete" in str(error)
        write(tmp_path, plan(items=[], revision=1, state="done"), request_id="r0")
        write(tmp_path, plan(items=[], revision=1, state="abandoned"), request_id="r1")
        write(tmp_path, plan(items=[]))
        write(tmp_path, plan(items=[]), request_id="more")
        refused(tmp_path, plan(items=[]), code="ledger_full", request_id="again")
        # reopening takes an open place too
        refused(
            tmp_path,
            plan(items=[], revision=2, state="open"),
            code="ledger_full",
            request_id="r0",
        )
        # and an open request is still writable at the cap
        assert write(tmp_path, plan(items=[], revision=1))["revision"] == 2

    def test_closed_requests_have_their_own_higher_cap(self, tmp_path, monkeypatch):
        assert ledger.MAX_CLOSED_REQUESTS >= 10 * ledger.MAX_REQUESTS
        monkeypatch.setattr(ledger, "MAX_CLOSED_REQUESTS", 2)
        for n in range(3):
            write(tmp_path, plan(items=[]), request_id=f"r{n}")
        for n in range(2):
            write(
                tmp_path, plan(items=[], revision=1, state="done"), request_id=f"r{n}"
            )
        error = refused(
            tmp_path,
            plan(items=[], revision=1, state="done"),
            code="ledger_full",
            request_id="r2",
        )
        assert "delete" in str(error)
        refused(
            tmp_path,
            plan(items=[], state="abandoned"),
            code="ledger_full",
            request_id="new",
        )
        # deleting one makes room, and a closed request stays writable
        ledger.delete_request(str(tmp_path), "r0", 2)
        write(tmp_path, plan(items=[], revision=1, state="done"), request_id="r2")
        write(tmp_path, plan(items=[], revision=2, state="abandoned"), request_id="r1")

    def test_dropped_items_do_not_count_toward_the_item_cap(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(ledger, "MAX_ITEMS", 3)
        linked = _linked("a", "b", "c")
        write(tmp_path, plan(items=[item("a"), item("b"), item("c")]))
        view = write(
            tmp_path,
            plan(revision=1, items=[item("d"), item("e"), item("f")]),
            linked_jobs=linked,
        )
        assert [(i["key"], i["dropped"]) for i in view["items"]] == [
            ("d", False),
            ("e", False),
            ("f", False),
            ("a", True),
            ("b", True),
            ("c", True),
        ]
        # the plan as read - dropped items and all - writes back
        again = write(
            tmp_path, plan(revision=2, items=view["items"]), linked_jobs=linked
        )
        assert len(again["items"]) == 6
        refused(
            tmp_path,
            plan(revision=3, items=[item(k) for k in "defg"]),
            linked_jobs=linked,
        )

    def test_dropped_items_have_a_cap_that_purge_relieves(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ledger, "MAX_DROPPED_ITEMS", 2)
        linked = _linked("a", "b", "c")
        write(tmp_path, plan(items=[item("a"), item("b"), item("c"), item("d")]))
        error = refused(
            tmp_path,
            plan(revision=1, items=[item("d")]),
            code="too_many_dropped_items",
            linked_jobs=linked,
        )
        assert error.status == 409 and "purge" in str(error)
        view = write(
            tmp_path,
            plan(revision=1, items=[item("d")], purge=["a"]),
            linked_jobs=linked,
        )
        assert [(i["key"], i["dropped"]) for i in view["items"]] == [
            ("d", False),
            ("b", True),
            ("c", True),
        ]
        # an item already dropped is purged the same way
        view = write(
            tmp_path,
            plan(revision=2, items=[item("d")], purge=["b"]),
            linked_jobs=linked,
        )
        assert [i["key"] for i in view["items"]] == ["d", "c"]

    def test_purge_names_only_items_leaving_the_plan(self, tmp_path):
        write(tmp_path, plan(items=[item("a"), item("b")]))
        refused(tmp_path, plan(revision=1, items=[item("a")], purge=["a"]))
        refused(tmp_path, plan(revision=1, items=[item("a")], purge=["nope"]))
        refused(tmp_path, plan(revision=1, items=[item("a")], purge="b"))
        refused(tmp_path, plan(revision=1, items=[item("a")], purge=["B!"]))
        assert write(tmp_path, plan(revision=1, items=[item("a")], purge=["b"]))

    def test_a_purged_key_written_again_reads_its_jobs(self, tmp_path):
        linked = _linked("a")
        jobs = [
            {"job": "job-a", "item": "a", "status": "succeeded", "run_exists": True}
        ]
        write(tmp_path, plan(items=[item("a"), item("b")]))
        write(
            tmp_path,
            plan(revision=1, items=[item("b")], purge=["a"]),
            linked_jobs=linked,
        )
        view = write(
            tmp_path,
            plan(revision=2, items=[item("a"), item("b")]),
            linked_jobs=linked,
            jobs_for=lambda rid: jobs,
        )
        assert view["items"][0]["status"] == "done"

    def test_the_file_cap_says_what_frees_room(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ledger, "MAX_FILE_BYTES", 2000)
        error = refused(tmp_path, plan(goal="x" * 4000), code="ledger_too_large")
        assert "delete" in str(error) and "purge" in str(error)
        assert ledger.MAX_FILE_BYTES == 2000  # the real one is pinned in test_ledger

    def test_purge_over_http(self, client):
        assert client.put(f"{URL}/cats", json=plan()).status_code == 200
        r = client.put(f"{URL}/cats", json=plan(revision=1, items=[], purge=["a", "b"]))
        assert r.status_code == 200, r.text
        assert r.json()["items"] == []
        bad = client.put(f"{URL}/cats", json=plan(revision=2, items=[], purge=["a"]))
        assert bad.status_code == 400
