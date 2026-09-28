"""Tests for the console-script entry points.

dw.run and dw.validate are what most people actually type, and everything
they do before the workflow executes - name=value parsing, path validation,
the exit codes a script checks - had no coverage at all.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from dw import run as run_module
from dw import validate as validate_module
from dw_mcp.client import DwClient
from tests.test_server import (  # noqa: F401
    failing_script,
    server,
    success_script,
    valid_workflow,
)


@pytest.fixture
def workflow_file(tmp_path):
    definition = {
        "id": "cli_test",
        "variables": {"prompt": "a landscape", "steps": 25},
        "steps": [
            {
                "name": "gen",
                "task": {"command": "gather_images", "arguments": {}},
                "result": {"content_type": "image/png", "save": False},
            }
        ],
    }
    path = tmp_path / "cli_test.json"
    path.write_text(json.dumps(definition))
    return path


def invoke(module, monkeypatch, argv):
    """Run an entry point's main() with argv, returning its exit code."""
    monkeypatch.setattr("sys.argv", argv)
    try:
        module.main()
    except SystemExit as exit_call:
        return exit_call.code or 0
    return 0


class TestValidateEntryPoint:
    def test_a_valid_workflow_reports_success(self, workflow_file, monkeypatch, capsys):
        code = invoke(validate_module, monkeypatch, ["dw-validate", str(workflow_file)])
        assert code == 0
        assert "validated successfully" in capsys.readouterr().out

    def test_a_schema_violation_exits_nonzero(self, tmp_path, monkeypatch, capsys):
        path = tmp_path / "broken.json"
        path.write_text(json.dumps({"id": "broken"}))  # no steps

        code = invoke(validate_module, monkeypatch, ["dw-validate", str(path)])
        assert code == 1
        out = capsys.readouterr().out
        # Printed once - not doubled as "Error validating workflow: Validation error: ..."
        assert out.count("Validation error") == 1
        assert "Error validating workflow" not in out

    def test_a_schema_violation_names_the_json_path(
        self, workflow_file, monkeypatch, capsys
    ):
        definition = json.loads(workflow_file.read_text())
        # "seed" on a step must be an integer per the schema - give it a string
        definition["steps"][0]["seed"] = "not-a-number"
        workflow_file.write_text(json.dumps(definition))

        code = invoke(validate_module, monkeypatch, ["dw-validate", str(workflow_file)])
        assert code == 1
        out = capsys.readouterr().out
        assert "steps[0]" in out
        assert "seed" in out

    def test_a_traversing_path_is_refused_before_anything_is_read(
        self, monkeypatch, capsys
    ):
        code = invoke(
            validate_module, monkeypatch, ["dw-validate", "../../etc/passwd.json"]
        )
        assert code == 1
        assert "Security validation failed" in capsys.readouterr().out

    def test_trust_workflows_flag_is_accepted_and_wired(
        self, workflow_file, monkeypatch, capsys
    ):
        """--trust-workflows must parse (dw.run has long had it; dw.validate
        did not) and actually flip the trust gate before the workflow is
        loaded/validated, since validation now realizes 'constant:' defaults
        through the same gate a run would."""
        seen = []
        monkeypatch.setattr(
            validate_module, "set_trust_workflows", lambda v: seen.append(v)
        )

        code = invoke(
            validate_module,
            monkeypatch,
            ["dw-validate", "--trust-workflows", str(workflow_file)],
        )
        assert code == 0
        assert seen == [True]
        assert "validated successfully" in capsys.readouterr().out


def _bridge(app, calls=None):
    """Wire a DwClient to a FastAPI app in-process, over a real httpx
    transport that never opens a socket - the shape tests/test_server.py's
    own `server` fixture builds, minus the process boundary. `calls`, when
    given, collects each request's path, so a test can assert how many
    requests dw.run actually made rather than only what the last one did."""
    # The server's own host-check middleware only accepts a loopback name;
    # "testserver" (TestClient's own default) isn't one, so this inner
    # client is pinned to "localhost" the way tests/test_server.py's own
    # `server` fixture is.
    local = TestClient(app, base_url="http://localhost")

    def handler(request):
        if calls is not None:
            calls.append(request.url.path)
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        response = local.request(
            request.method,
            request.url.path,
            params=request.url.params,
            content=request.content,
            headers=headers,
        )
        return httpx.Response(
            response.status_code, headers=response.headers, content=response.content
        )

    # TestClient's own host, so host-checking middleware sees what it expects
    return DwClient(
        base_url="http://testserver", transport=httpx.MockTransport(handler)
    )


class TestRunEntryPoint:
    def test_a_catalog_name_runs_and_exits_0(self, server, capsys):
        with server(success_script) as local:
            client = _bridge(local.app)
            code = run_module.main(["Basic"], client=client)
        assert code == 0
        assert "run_dir" in capsys.readouterr().out

    def test_name_value_pairs_arrive_as_arguments(self, server):
        with server(success_script) as local:
            client = _bridge(local.app)
            code = run_module.main(["Basic", "prompt=a cat"], client=client)
            assert code == 0
            manager = local.app.state.job_manager
            executes = [
                c for c in manager.worker_manager.commands if c["type"] == "execute"
            ]
        assert len(executes) == 1
        assert executes[0]["arguments"] == {"prompt": "a cat"}

    def test_a_bad_pair_is_refused_before_any_request(self, capsys):
        calls = []

        def handler(request):
            calls.append(request.url.path)
            pytest.fail("no request should have been made for an invalid pair")

        client = DwClient(
            base_url="http://testserver", transport=httpx.MockTransport(handler)
        )

        code = run_module.main(["Basic", "oops"], client=client)
        assert code == 1
        assert "not in name=value format" in capsys.readouterr().out
        assert calls == []

    def test_a_file_the_server_cannot_reach_is_sent_inline_with_a_notice(
        self, server, tmp_path, capsys
    ):
        # Outside the server's workflow_dir (tmp_path/"workflows"), so
        # resolve_workflow_reference can't confine it to a source
        external = tmp_path / "external.json"
        external.write_text(json.dumps(valid_workflow("external")))

        calls = []
        with server(success_script) as local:
            client = _bridge(local.app, calls=calls)
            code = run_module.main([str(external)], client=client)

        assert code == 0
        # Two POST /api/jobs calls - workflow_path, then the inline retry -
        # among whatever polling the run also did meanwhile
        assert [c for c in calls if c == "/api/jobs"] == ["/api/jobs", "/api/jobs"]
        out = capsys.readouterr().out
        assert f"note: the server cannot read {external}" in out

    def test_a_400_for_bad_arguments_on_a_local_file_is_not_resent_inline(
        self, server, tmp_path, capsys
    ):
        # Inside the server's workflow_dir, so the path itself resolves -
        # the 400 here is about the arguments, not about reaching the file
        local_file = tmp_path / "workflows" / "Basic.json"

        calls = []
        with server(success_script) as local:
            client = _bridge(local.app, calls=calls)
            code = run_module.main(
                [str(local_file), "totally_bogus_variable=1"], client=client
            )

        assert code == 1
        assert calls == ["/api/jobs"]
        assert "totally_bogus_variable" in capsys.readouterr().out

    def test_a_failed_job_exits_1_with_its_error(self, server, capsys):
        with server(failing_script) as local:
            client = _bridge(local.app)
            code = run_module.main(["Basic"], client=client)
        assert code == 1
        assert "CUDA out of memory" in capsys.readouterr().out

    def test_no_server_is_one_line_and_exit_2(self, capsys):
        def handler(request):
            raise httpx.ConnectError("refused")

        client = DwClient(
            base_url="http://127.0.0.1:19999",
            transport=httpx.MockTransport(handler),
        )

        code = run_module.main(["Basic"], client=client)
        assert code == 2
        out = capsys.readouterr().out
        assert out.startswith("error: no dw.serve at")
        assert "Traceback" not in out
