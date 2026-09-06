"""dw.serve's argument handling, without starting uvicorn."""

import pytest


@pytest.fixture
def serve(monkeypatch, tmp_path):
    """dw.serve.main with uvicorn and the app factory replaced, so a call
    returns what it would have served instead of serving it."""
    import uvicorn

    import dw.serve as serve_module
    from dw.server import app as app_module

    calls = {}

    def fake_create_app(**kwargs):
        calls["create_app"] = kwargs
        return object()

    def fake_run(app, **kwargs):
        calls["uvicorn"] = kwargs

    monkeypatch.setattr(app_module, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.delenv("DW_API_TOKEN", raising=False)
    # main() pins DW_PROMPT_DIR in os.environ; monkeypatch restores it
    monkeypatch.setenv("DW_PROMPT_DIR", str(tmp_path / "prompts"))
    # And a workspace of its own: without one main() resolves the working
    # directory, which for the test suite is the checkout - and then creates
    # the asset library inside it
    monkeypatch.setenv("DW_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("DW_WORKSPACE_SOURCE", "flag")
    (tmp_path / "workflows").mkdir()

    def run(*argv):
        monkeypatch.setattr(
            "sys.argv",
            ["dw-serve", "--workflow-dir", str(tmp_path / "workflows"), *argv],
        )
        serve_module.main()
        return calls

    return run


def test_mcp_is_off_by_default(serve):
    calls = serve()
    assert calls["create_app"]["mcp"] is False
    assert calls["create_app"]["port"] == 8765


def test_mcp_flag_is_passed_through(serve, capsys):
    calls = serve("--mcp", "--port", "9000", "--token", "t")
    assert calls["create_app"]["mcp"] is True
    assert calls["create_app"]["port"] == 9000
    # the banner tells the operator where the MCP endpoint is
    assert "/mcp" in capsys.readouterr().out


def test_mcp_on_a_non_loopback_bind_requires_a_token(serve, capsys):
    with pytest.raises(SystemExit) as exit_info:
        serve("--mcp", "--host", "0.0.0.0")
    assert exit_info.value.code == 2
    assert "--token" in capsys.readouterr().err


def test_the_refusal_comes_before_the_worker_is_started(serve, monkeypatch):
    """The point of a hard error is that nothing has happened yet - no
    startup(), no spawned worker to leave behind."""
    import dw

    def fail(*args, **kwargs):
        raise AssertionError("startup() ran before the --mcp check")

    monkeypatch.setattr(dw, "startup", fail)
    with pytest.raises(SystemExit):
        serve("--mcp", "--host", "0.0.0.0")


def test_mcp_on_loopback_needs_no_token(serve):
    calls = serve("--mcp")
    assert calls["create_app"]["mcp"] is True


def test_a_non_loopback_bind_without_mcp_is_only_warned_about(serve):
    calls = serve("--host", "0.0.0.0")
    assert calls["create_app"]["mcp"] is False
    assert calls["uvicorn"]["host"] == "0.0.0.0"
