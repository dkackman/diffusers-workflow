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

from .test_ledger import URL, plan, refused, write
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
