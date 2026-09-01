"""The assembled MCP server: what a client actually sees when it connects."""

import json

import httpx
import pytest

pytest.importorskip("mcp", reason="the mcp extra is not installed")

from dw.mcp.client import DwClient  # noqa: E402
from dw.mcp.server import build_server  # noqa: E402

EXPECTED_TOOLS = {
    "list_workflows",
    "get_workflow",
    "get_schema",
    "list_pipelines",
    "get_pipeline_signature",
    "list_classes",
    "get_class",
    "list_tasks",
    "get_task",
    "list_models",
    "get_memory",
    "get_health",
    "list_jobs",
    "list_gallery",
    "get_gallery_metadata",
    "get_output_image",
    "validate_workflow",
    "save_workflow",
    "delete_workflow",
    "run_workflow",
    "get_job",
    "get_job_events",
    "cancel_job",
    "rerun_job",
    "move_job",
}

READ_ONLY_TOOLS = EXPECTED_TOOLS - {
    "save_workflow",
    "delete_workflow",
    "run_workflow",
    "cancel_job",
    "rerun_job",
    "move_job",
}

DESTRUCTIVE_TOOLS = {"save_workflow", "delete_workflow"}


def server_over(handler):
    client = DwClient(transport=httpx.MockTransport(handler))
    return build_server(client)


def ok(body):
    def handler(request):
        return httpx.Response(200, json=body)

    return handler


async def tools_of(server):
    return {tool.name: tool for tool in await server.list_tools()}


def _text_of(result):
    """2.x call_tool returns a CallToolResult; the payload is its content."""
    return [getattr(item, "text", "") for item in result.content]


@pytest.mark.asyncio
async def test_every_designed_tool_is_registered():
    tools = await tools_of(server_over(ok({})))

    assert set(tools) == EXPECTED_TOOLS


@pytest.mark.asyncio
async def test_every_tool_has_a_description():
    tools = await tools_of(server_over(ok({})))

    missing = [
        name for name, tool in tools.items() if not (tool.description or "").strip()
    ]
    assert missing == []


@pytest.mark.asyncio
async def test_read_only_tools_are_annotated_read_only():
    tools = await tools_of(server_over(ok({})))

    for name in READ_ONLY_TOOLS:
        assert tools[name].annotations is not None, name
        assert tools[name].annotations.read_only_hint is True, name


@pytest.mark.asyncio
async def test_writing_tools_are_not_annotated_read_only():
    tools = await tools_of(server_over(ok({})))

    for name in EXPECTED_TOOLS - READ_ONLY_TOOLS:
        assert tools[name].annotations.read_only_hint is not True, name


@pytest.mark.asyncio
async def test_overwriting_tools_are_annotated_destructive():
    tools = await tools_of(server_over(ok({})))

    for name in DESTRUCTIVE_TOOLS:
        assert tools[name].annotations.destructive_hint is True, name


@pytest.mark.asyncio
async def test_no_tool_claims_an_open_world():
    """Every tool talks to one known local server and nothing else."""
    tools = await tools_of(server_over(ok({})))

    for name, tool in tools.items():
        assert tool.annotations.open_world_hint is False, name


@pytest.mark.asyncio
async def test_no_tool_exposes_base_dir():
    tools = await tools_of(server_over(ok({})))

    for name, tool in tools.items():
        assert "base_dir" not in json.dumps(tool.input_schema), name


@pytest.mark.asyncio
async def test_run_workflow_takes_an_acknowledged_cost_flag():
    tools = await tools_of(server_over(ok({})))

    assert "acknowledged_cost" in tools["run_workflow"].input_schema["properties"]


@pytest.mark.asyncio
async def test_run_workflow_advertises_its_cost():
    """The description is what an agent reads before spending GPU minutes."""
    tools = await tools_of(server_over(ok({})))

    description = tools["run_workflow"].description
    assert "COSTS GPU TIME" in description
    assert "acknowledged_cost" in description


@pytest.mark.asyncio
async def test_a_read_only_tool_round_trips_to_the_api():
    server = server_over(ok({"workflows": ["a"], "details": {}}))

    result = await server.call_tool("list_workflows", {})

    assert "workflows" in json.dumps(_text_of(result))


@pytest.mark.asyncio
async def test_run_workflow_refuses_without_acknowledgement():
    server = server_over(ok({"id": "job-1", "status": "queued"}))

    with pytest.raises(Exception) as caught:
        await server.call_tool("run_workflow", {"workflow_path": "w.json"})

    assert "acknowledged_cost" in str(caught.value)


@pytest.mark.asyncio
async def test_an_unreachable_server_reports_how_to_start_it():
    def refusing(request):
        raise httpx.ConnectError("refused", request=request)

    server = server_over(refusing)

    with pytest.raises(Exception) as caught:
        await server.call_tool("get_health", {})

    assert "dw-serve" in str(caught.value)


@pytest.mark.asyncio
async def test_an_image_comes_back_as_an_image_block():
    png = (
        b"\x89PNG\r\n\x1a\n"  # a real 1x1 PNG, so the handler can open it
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
        b"\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01"
        b"\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    def serving_png(request):
        return httpx.Response(200, content=png, headers={"content-type": "image/png"})

    server = server_over(serving_png)

    result = await server.call_tool("get_output_image", {"name": "out.png"})

    block = result.content[0]
    assert block.type == "image"
    assert block.mime_type.startswith("image/")
    assert block.data
