"""MCP over Streamable HTTP, mounted inside dw.serve at /mcp."""

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("mcp", reason="the mcp extra is not installed")

from dw.server.app import create_app  # noqa: E402
from dw.server.jobs import JobManager  # noqa: E402

from tests.test_mcp_server import EXPECTED_TOOLS  # noqa: E402
from tests.test_server import ScriptedWorkerManager, success_script  # noqa: E402

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}
MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def make_app(tmp_path, token=None, mcp=True):
    workflow_dir = tmp_path / "workflows"
    workflow_dir.mkdir(exist_ok=True)
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
    )
    return create_app(
        workflow_dir=str(workflow_dir),
        output_dir=str(tmp_path / "outputs"),
        job_manager=manager,
        prompt_dir=str(tmp_path / "prompts"),
        host="0.0.0.0",
        token=token,
        mcp=mcp,
        port=8765,
    )


def test_mcp_is_not_mounted_by_default(tmp_path):
    app = make_app(tmp_path, mcp=False)
    with TestClient(app, base_url="http://localhost") as client:
        assert client.get("/api/health").json()["mcp"] is False
        assert "mcp" not in {route.name for route in app.routes}
        # nothing under /mcp: the request falls through to whatever else
        # claims the path - the SPA catch-all answers a POST with 405, and
        # a build without the SPA has nothing there at all
        assert client.post(
            "/mcp", json=INITIALIZE, headers=MCP_HEADERS
        ).status_code in (404, 405)


def test_mcp_mount_answers_initialize(tmp_path):
    app = make_app(tmp_path)
    with TestClient(app, base_url="http://localhost") as client:
        assert client.get("/api/health").json()["mcp"] is True
        response = client.post(
            "/mcp", json=INITIALIZE, headers=MCP_HEADERS, follow_redirects=False
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["result"]["serverInfo"]["name"] == "diffusers-workflow"


def test_mcp_mount_answers_the_trailing_slash_spelling_too(tmp_path):
    """`/mcp` is what the docs configure; `/mcp/` is what a client that
    normalizes the URL may send. Both must answer without a redirect."""
    app = make_app(tmp_path)
    with TestClient(app, base_url="http://localhost") as client:
        response = client.post(
            "/mcp/", json=INITIALIZE, headers=MCP_HEADERS, follow_redirects=False
        )
        assert response.status_code == 200, response.text


def test_mcp_mount_is_gated_by_the_bearer_token(tmp_path):
    app = make_app(tmp_path, token="s3cr3t")
    with TestClient(app, base_url="http://localhost") as client:
        assert (
            client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS).status_code == 401
        )
        assert (
            client.post("/mcp/", json=INITIALIZE, headers=MCP_HEADERS).status_code
            == 401
        )
        wrong = {**MCP_HEADERS, "Authorization": "Bearer nope"}
        assert client.post("/mcp", json=INITIALIZE, headers=wrong).status_code == 401
        # never as a query parameter - that allowance is for <img>/<a> only
        assert (
            client.post(
                "/mcp?token=s3cr3t", json=INITIALIZE, headers=MCP_HEADERS
            ).status_code
            == 401
        )
        right = {**MCP_HEADERS, "Authorization": "Bearer s3cr3t"}
        assert client.post("/mcp", json=INITIALIZE, headers=right).status_code == 200
        # a lookalike path must not reach the tool surface behind the gate,
        # which covers "/mcp" and "/mcp/..." only
        assert client.post("/mcpfoo", json=INITIALIZE, headers=right).status_code != 200


@pytest.mark.asyncio
async def test_a_real_mcp_client_lists_every_tool_over_http(tmp_path):
    """The SDK's own client, speaking Streamable HTTP into the mounted app
    in-process, sees the same tool surface the stdio server exposes."""
    import httpx2
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    app = make_app(tmp_path, token="s3cr3t")
    # TestClient drives the lifespan (the MCP session manager); the SDK
    # client rides an ASGI transport straight into the same app object.
    with TestClient(app, base_url="http://localhost"):
        http = httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": "Bearer s3cr3t"},
        )
        async with http:
            async with streamable_http_client(
                "http://localhost/mcp", http_client=http
            ) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await session.list_tools()
    assert {t.name for t in tools.tools} == EXPECTED_TOOLS


def test_download_output_says_it_writes_on_the_server_side():
    """Over /mcp the tool runs on the GPU box; the description has to say so
    or an agent on another machine will look for the file locally."""
    import asyncio

    from dw_mcp.client import DwClient
    from dw_mcp.server import build_server

    server = build_server(DwClient())
    tools = asyncio.run(server.list_tools())
    description = next(t.description for t in tools if t.name == "download_output")
    assert "machine running the MCP server" in description
