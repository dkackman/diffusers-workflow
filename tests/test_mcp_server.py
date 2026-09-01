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

# a real 1x1 PNG, so the image handler can actually decode it
PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01"
    b"\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


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
    def serving_png(request):
        return httpx.Response(
            200, content=PNG_1X1, headers={"content-type": "image/png"}
        )

    server = server_over(serving_png)

    result = await server.call_tool("get_output_image", {"name": "out.png"})

    block = result.content[0]
    assert block.type == "image"
    assert block.mime_type.startswith("image/")
    assert block.data


# Every tool, the arguments a client would send, and the one API call it is
# expected to make. This is the wiring: a tool bound to the wrong handler or
# handed its arguments in the wrong order shows up here and nowhere else.
TOOL_WIRING = [
    ("list_workflows", {}, "GET", "/api/workflows"),
    ("get_workflow", {"name": "w"}, "GET", "/api/workflows/w"),
    ("get_schema", {}, "GET", "/api/schema"),
    ("list_pipelines", {}, "GET", "/api/pipelines"),
    (
        "get_pipeline_signature",
        {"name": "FluxPipeline"},
        "GET",
        "/api/pipelines/FluxPipeline",
    ),
    ("list_classes", {"kind": "models"}, "GET", "/api/classes"),
    (
        "get_class",
        {"name": "FluxTransformer2DModel"},
        "GET",
        "/api/classes/FluxTransformer2DModel",
    ),
    ("list_tasks", {}, "GET", "/api/tasks"),
    ("get_task", {"command": "resize"}, "GET", "/api/tasks/resize"),
    ("list_models", {}, "GET", "/api/models"),
    ("get_memory", {}, "GET", "/api/memory"),
    ("get_health", {}, "GET", "/api/health"),
    ("list_jobs", {}, "GET", "/api/jobs"),
    ("list_gallery", {"limit": 5}, "GET", "/api/gallery"),
    (
        "get_gallery_metadata",
        {"name": "out.png"},
        "GET",
        "/api/gallery/out.png/metadata",
    ),
    ("get_output_image", {"name": "out.png"}, "GET", "/outputs/out.png"),
    ("validate_workflow", {"workflow": {"id": "w"}}, "POST", "/api/validate"),
    (
        "save_workflow",
        {"name": "w", "workflow": {"id": "w"}},
        "PUT",
        "/api/workflows/w",
    ),
    ("delete_workflow", {"name": "w"}, "DELETE", "/api/workflows/w"),
    (
        "run_workflow",
        {"workflow_path": "w.json", "acknowledged_cost": True},
        "POST",
        "/api/jobs",
    ),
    ("get_job", {"job_id": "j1"}, "GET", "/api/jobs/j1"),
    ("get_job_events", {"job_id": "j1"}, "GET", "/api/jobs/j1/event-log"),
    ("cancel_job", {"job_id": "j1"}, "POST", "/api/jobs/j1/cancel"),
    ("rerun_job", {"job_id": "j1"}, "POST", "/api/jobs/j1/rerun"),
    ("move_job", {"job_id": "j1", "direction": "up"}, "POST", "/api/jobs/j1/move"),
]


def test_the_wiring_table_covers_every_registered_tool():
    assert {name for name, _, _, _ in TOOL_WIRING} == EXPECTED_TOOLS


@pytest.mark.asyncio
@pytest.mark.parametrize("name,arguments,method,path", TOOL_WIRING)
async def test_each_tool_calls_its_endpoint(name, arguments, method, path):
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.url.path.startswith("/outputs/"):
            return httpx.Response(
                200, content=PNG_1X1, headers={"content-type": "image/png"}
            )
        return httpx.Response(200, json={"id": "j1", "status": "queued"})

    result = await server_over(handler).call_tool(name, arguments)

    assert seen == [(method, path)]
    assert result.content


class RecordingServer:
    def __init__(self):
        self.transport = None
        self.raises = None

    def run(self, transport=None):
        self.transport = transport
        if self.raises is not None:
            raise self.raises


def entry_point_over(monkeypatch, server):
    """`main` with the real client construction but a stub server."""
    from dw.mcp import __main__ as entry

    built = {}

    def build(client):
        built["client"] = client
        return server

    monkeypatch.setattr(entry, "build_server", build)
    return entry, built


def test_the_entry_point_serves_over_stdio(monkeypatch):
    monkeypatch.delenv("DW_MCP_URL", raising=False)
    server = RecordingServer()
    entry, built = entry_point_over(monkeypatch, server)

    assert entry.main([]) == 0

    assert server.transport == "stdio"
    assert built["client"].base_url == "http://127.0.0.1:8765"
    assert built["client"].timeout == 30.0


def test_the_entry_point_takes_a_url_and_a_timeout(monkeypatch):
    server = RecordingServer()
    entry, built = entry_point_over(monkeypatch, server)

    entry.main(["--url", "http://127.0.0.1:9000/", "--timeout", "5"])

    assert built["client"].base_url == "http://127.0.0.1:9000"
    assert built["client"].timeout == 5.0


def test_the_entry_point_closes_the_client_when_serving_fails(monkeypatch):
    """The connection is released on the way out, however the loop ends."""
    server = RecordingServer()
    server.raises = KeyboardInterrupt()
    entry, built = entry_point_over(monkeypatch, server)

    with pytest.raises(KeyboardInterrupt):
        entry.main([])

    assert built["client"]._http.is_closed


@pytest.mark.asyncio
async def test_validate_workflow_accepts_explicit_nulls_and_says_pick_one():
    """A model filling in every declared property with null is routine, and
    for a 'give exactly one of' tool it is the likely call. A non-optional
    annotation with a None default makes the schema reject it before the
    handler runs, replacing the written message with a validation dump."""
    server = server_over(ok({}))

    with pytest.raises(Exception) as caught:
        await server.call_tool("validate_workflow", {"workflow": None, "name": None})

    assert "exactly one" in str(caught.value)


@pytest.mark.asyncio
async def test_validate_workflow_reaches_the_handler_past_an_explicit_null():
    """A null filled in beside a real value must not be a schema error - the
    handler is what decides whether the combination makes sense."""
    server = server_over(ok({"valid": True, "errors": []}))

    result = await server.call_tool(
        "validate_workflow", {"workflow": None, "name": "w"}
    )

    assert "valid" in json.dumps(_text_of(result))


@pytest.mark.asyncio
async def test_run_workflow_accepts_explicit_nulls_and_says_pick_one():
    server = server_over(ok({"id": "job-1", "status": "queued"}))

    with pytest.raises(Exception) as caught:
        await server.call_tool(
            "run_workflow",
            {
                "workflow_path": "w.json",
                "inline_workflow": None,
                "arguments": None,
                "acknowledged_cost": False,
            },
        )

    assert "acknowledged_cost" in str(caught.value)


@pytest.mark.asyncio
async def test_optional_parameters_are_declared_nullable():
    """A `{"type": "object", "default": null}` schema contradicts itself."""
    tools = await tools_of(server_over(ok({})))

    for name, tool in tools.items():
        for parameter, schema in tool.input_schema["properties"].items():
            if schema.get("default", "missing") is None:
                assert (
                    "anyOf" in schema or schema.get("type") == "null"
                ), f"{name}.{parameter} defaults to null but is not nullable"
