"""`dw-mcp` startup: what it refuses, what it checks, what it prints."""

import json

import httpx
import pytest

pytest.importorskip("mcp", reason="the mcp extra is not installed")

from dw_mcp import __main__ as cli  # noqa: E402
from dw_mcp.client import is_loopback_url  # noqa: E402


@pytest.fixture(autouse=True)
def no_ambient_config(monkeypatch):
    monkeypatch.delenv("DW_API_TOKEN", raising=False)
    monkeypatch.delenv("DW_MCP_URL", raising=False)


@pytest.fixture
def no_stdio(monkeypatch):
    """Stop main() from actually serving stdio once startup checks pass."""
    ran = {}

    class FakeServer:
        def run(self, transport):
            ran["transport"] = transport

    monkeypatch.setattr(cli, "build_server", lambda client: FakeServer())
    return ran


def healthy(request):
    return httpx.Response(
        200,
        json={
            "status": "ok",
            "hostname": "gpu-box",
            "version": "1.2.3",
            "device": "cuda",
        },
    )


@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://127.0.0.1:8765", True),
        ("http://localhost:8765", True),
        ("http://[::1]:8765", True),
        ("http://192.168.1.50:8765", False),
        ("http://gpu-box.local:8765", False),
    ],
)
def test_is_loopback_url(url, expected):
    assert is_loopback_url(url) is expected


def test_a_remote_url_without_a_token_is_refused_before_any_request(capsys):
    requests = []

    def handler(request):
        requests.append(request)
        return healthy(request)

    code = cli.main(
        ["--url", "http://192.168.1.50:8765"], transport=httpx.MockTransport(handler)
    )

    assert code == 2
    assert requests == []
    err = capsys.readouterr().err
    assert "--token" in err and "DW_API_TOKEN" in err
    assert "192.168.1.50" in err


def test_a_loopback_url_without_a_token_is_allowed(no_stdio):
    code = cli.main([], transport=httpx.MockTransport(healthy))
    assert code == 0
    assert no_stdio["transport"] == "stdio"


def test_the_probe_reports_a_server_that_wants_a_token(capsys):
    def handler(request):
        return httpx.Response(401, json={"detail": "Missing or invalid bearer token"})

    code = cli.main(
        ["--url", "http://192.168.1.50:8765", "--token", "wrong"],
        transport=httpx.MockTransport(handler),
    )
    assert code == 2
    assert "token" in capsys.readouterr().err.lower()


def test_the_probe_reports_an_unreachable_server(capsys):
    def handler(request):
        raise httpx.ConnectError("refused")

    code = cli.main(
        ["--url", "http://192.168.1.50:8765", "--token", "t"],
        transport=httpx.MockTransport(handler),
    )
    assert code == 2
    assert "http://192.168.1.50:8765" in capsys.readouterr().err


def test_a_successful_probe_prints_the_server_identity(no_stdio, capsys):
    seen = []

    def handler(request):
        seen.append(
            (request.method, request.url.path, request.headers.get("authorization"))
        )
        return healthy(request)

    code = cli.main(
        ["--url", "http://192.168.1.50:8765", "--token", "s3cr3t"],
        transport=httpx.MockTransport(handler),
    )
    assert code == 0
    assert seen == [("GET", "/api/health", "Bearer s3cr3t")]
    err = capsys.readouterr().err
    assert "gpu-box" in err and "1.2.3" in err and "cuda" in err


def test_no_probe_sends_nothing(no_stdio):
    seen = []

    def handler(request):
        seen.append(request)
        return healthy(request)

    code = cli.main(
        ["--url", "http://192.168.1.50:8765", "--token", "t", "--no-probe"],
        transport=httpx.MockTransport(handler),
    )
    assert code == 0
    assert seen == []
