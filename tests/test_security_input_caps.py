"""The documented size limits, refused at validation - before any work.

`MAX_VARIABLE_VALUE_LENGTH` (20,000 characters), the variable-name cap, the
32-entry `for_each` ceiling, the 50MB workflow-file ceiling and the 200MB
upload ceiling are each a promise that an oversized input costs the server
nothing but the refusal. The live suite can see the refusal; it cannot see
whether a worker was handed the job first, a file was read whole, or a body
was buffered. These tests watch for exactly that: a ScriptedWorkerManager
that records every command it is sent, a sparse file that would be 50MB to
read, and an ASGI receive() that records whether the body was pulled.
"""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from dw.security import (
    MAX_JSON_SIZE,
    MAX_VARIABLE_NAME_LENGTH,
    MAX_VARIABLE_VALUE_LENGTH,
    InvalidInputError,
)
from dw.server.app import create_app
from dw.server.jobs import JobManager
from dw.variables import argument_errors

from .test_server import ScriptedWorkerManager, success_script

OVERSIZED = "x" * 25_000


def _workflow(variables=None, for_each=None):
    step = {
        "name": "t",
        "task": {"command": "compose_text", "arguments": {"parts": ["variable:p"]}},
        "result": {"content_type": "text/plain"},
    }
    if for_each is not None:
        step["for_each"] = for_each
        step["task"]["arguments"]["parts"] = ["item:prompt"]
    return {
        "id": "caps",
        "variables": {"p": "short", "shots": [], **(variables or {})},
        "steps": [step],
    }


@pytest.fixture
def server(tmp_path):
    (tmp_path / "workflows").mkdir()
    worker = ScriptedWorkerManager(success_script)
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=worker,
        history_path=str(tmp_path / "jobs.sqlite"),
    )
    app = create_app(
        workflow_dir=str(tmp_path / "workflows"),
        output_dir=str(tmp_path / "outputs"),
        job_manager=manager,
        prompt_dir=str(tmp_path / "prompts"),
        asset_dir=str(tmp_path / "assets"),
    )
    with TestClient(app, base_url="http://localhost") as client:
        yield client, worker


def _executed(worker):
    return [c for c in worker.commands if c.get("type") == "execute"]


def test_the_documented_limit_is_the_one_under_test():
    assert MAX_VARIABLE_VALUE_LENGTH == 20_000
    assert len(OVERSIZED) > MAX_VARIABLE_VALUE_LENGTH


class TestVariableValueLength:
    def test_the_limit_itself_is_accepted(self):
        assert (
            argument_errors(_workflow(), {"p": "x" * MAX_VARIABLE_VALUE_LENGTH}) == []
        )

    def test_one_over_is_refused(self):
        errors = argument_errors(
            _workflow(), {"p": "x" * (MAX_VARIABLE_VALUE_LENGTH + 1)}
        )
        assert [e["path"] for e in errors] == ["arguments.p"]

    @pytest.mark.parametrize(
        "arguments, path",
        [
            ({"p": OVERSIZED}, "arguments.p"),
            ({"shots": [{"name": "a", "prompt": OVERSIZED}]}, "arguments.shots"),
            ({"shots": [[OVERSIZED]]}, "arguments.shots"),
            (
                {"shots": [{"name": "a", "nested": {"deep": OVERSIZED}}]},
                "arguments.shots",
            ),
        ],
    )
    def test_25000_characters_is_refused_wherever_it_sits(self, arguments, path):
        errors = argument_errors(_workflow(), arguments)
        assert [e["path"] for e in errors] == [path]
        assert "too long" in errors[0]["message"]

    def test_the_validate_route_refuses_it(self, server):
        client, worker = server
        response = client.post(
            "/api/validate",
            json={"workflow": _workflow(), "arguments": {"p": OVERSIZED}},
        )
        body = response.json()
        assert body["valid"] is False
        assert any(e["path"] == "arguments.p" for e in body["errors"])
        assert _executed(worker) == []

    def test_the_jobs_route_refuses_it_before_the_worker_sees_it(self, server):
        client, worker = server
        response = client.post(
            "/api/jobs", json={"workflow": _workflow(), "arguments": {"p": OVERSIZED}}
        )
        assert response.status_code == 400
        assert _executed(worker) == []
        assert client.get("/api/jobs").json() in ([], {"jobs": []}) or not (
            client.get("/api/jobs").json().get("jobs")
        )

    def test_the_cli_refuses_it_before_loading_anything(self, monkeypatch, tmp_path):
        import dw.run as run_module

        loaded = []
        monkeypatch.setattr(
            run_module,
            "workflow_from_file",
            lambda *a, **k: loaded.append(a),
            raising=False,
        )
        workflow = tmp_path / "w.json"
        workflow.write_text(json.dumps(_workflow()))
        monkeypatch.setattr("sys.argv", ["dw-run", str(workflow), f"p={OVERSIZED}"])
        with pytest.raises(SystemExit) as exit_info:
            run_module.main()
        assert exit_info.value.code != 0
        assert loaded == []

    @pytest.mark.xfail(
        strict=True,
        reason="the cap is applied to caller arguments only - a 25,000-character "
        "default written into an inline workflow's own variables validates clean",
    )
    def test_a_default_in_the_definition_is_held_to_the_same_cap(self, server):
        client, worker = server
        response = client.post(
            "/api/validate", json={"workflow": _workflow({"p": OVERSIZED})}
        )
        assert response.json()["valid"] is False


class TestOtherDocumentedLimits:
    def test_an_over_long_variable_name_is_refused(self):
        name = "v" * (MAX_VARIABLE_NAME_LENGTH + 1)
        errors = argument_errors(_workflow({name: "d"}), {name: "value"})
        assert [e["path"] for e in errors] == [f"arguments.{name}"]

    def test_33_for_each_entries_are_a_validation_error(self, server):
        client, worker = server
        entries = [{"name": f"e{i}", "prompt": "p"} for i in range(33)]
        response = client.post(
            "/api/validate",
            json={"workflow": _workflow(for_each=entries)},
        )
        body = response.json()
        assert body["valid"] is False
        assert any(e["path"] == "steps[0].for_each" for e in body["errors"])
        assert _executed(worker) == []

    def test_33_entries_through_arguments_are_refused_before_the_queue(self, server):
        client, worker = server
        entries = [{"name": f"e{i}", "prompt": "p"} for i in range(33)]
        response = client.post(
            "/api/jobs",
            json={
                "workflow": _workflow(for_each="variable:shots"),
                "arguments": {"shots": entries},
            },
        )
        assert response.status_code == 400
        assert _executed(worker) == []

    def test_an_oversized_workflow_file_is_refused_without_reading_it(
        self, tmp_path, monkeypatch
    ):
        """A sparse file costs no disk; the check is on its size, so a
        refusal that opened it first would show up as an open() call."""
        from dw.workflow import workflow_from_file

        big = tmp_path / "big.json"
        with open(big, "wb") as file:
            file.truncate(MAX_JSON_SIZE + 1)

        opened = []
        real_open = open

        def recording_open(path, *args, **kwargs):
            if str(path) == str(big) or str(path).endswith("big.json"):
                opened.append(path)
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", recording_open)
        with pytest.raises(InvalidInputError, match="too large"):
            workflow_from_file(str(big), str(tmp_path / "out"))
        assert opened == []

    def test_an_upload_over_the_cap_is_refused_from_its_declared_length(self, server):
        """The 200MB ceiling is checked on Content-Length before the body is
        read - driven over raw ASGI so the declared length can exceed what
        is actually sent, and so the test sees whether receive() was called."""
        client, worker = server
        app = client.app
        pulled = []
        sent = []

        async def receive():
            pulled.append(True)
            return {"type": "http.request", "body": b"x", "more_body": False}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/uploads",
            "raw_path": b"/api/uploads",
            "query_string": b"filename=huge.png",
            "root_path": "",
            "headers": [
                (b"host", b"localhost"),
                (b"content-length", str(300 * 1024 * 1024).encode()),
                (b"content-type", b"application/octet-stream"),
            ],
            "client": ("127.0.0.1", 1),
            "server": ("localhost", 80),
        }
        asyncio.run(app(scope, receive, send))
        start = next(m for m in sent if m["type"] == "http.response.start")
        assert start["status"] == 413
        assert pulled == []
