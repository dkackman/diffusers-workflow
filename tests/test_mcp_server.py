"""The assembled MCP server: what a client actually sees when it connects."""

import inspect
import json
import os
import tempfile

import httpx
import pytest

pytest.importorskip("mcp", reason="the mcp extra is not installed")

from dw_mcp import authoring, catalog, diagnose, media, models, prompts  # noqa: E402
from dw_mcp.client import DwClient  # noqa: E402
from dw_mcp.server import build_server  # noqa: E402

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
    "get_server_info",
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
    "wait_for_job",
    "cancel_job",
    "rerun_job",
    "move_job",
    "download_model",
    "list_downloads",
    "cancel_download",
    "delete_model",
    "get_diffusers_state",
    "update_diffusers",
    "list_prompts",
    "get_prompt",
    "get_prompt_schema",
    "save_prompt",
    "delete_prompt",
    "list_enhancers",
    "enhance_prompt",
    "get_output_text",
    "download_output",
    "delete_output",
    "list_assets",
    "upload_asset",
}

READ_ONLY_TOOLS = EXPECTED_TOOLS - {
    "save_workflow",
    "delete_workflow",
    "run_workflow",
    "cancel_job",
    "rerun_job",
    "move_job",
    "download_model",
    "cancel_download",
    "delete_model",
    "update_diffusers",
    "save_prompt",
    "delete_prompt",
    "enhance_prompt",
    "download_output",
    "delete_output",
    "upload_asset",
}

DESTRUCTIVE_TOOLS = {
    "save_workflow",
    "delete_workflow",
    "delete_model",
    "save_prompt",
    "delete_prompt",
    "download_output",
    "delete_output",
}

# Tools that refuse until the caller passes acknowledged_cost=true. The
# annotations are a hint a client may or may not surface; this flag is the
# floor that holds on every client, so which tools carry it is pinned here.
GATED_TOOLS = {
    "run_workflow",
    "rerun_job",
    "download_model",
    "delete_model",
    "update_diffusers",
    "enhance_prompt",
}

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

    image_block = result.content[0]
    assert image_block.type == "image"
    assert image_block.mime_type.startswith("image/")
    assert image_block.data

    text_block = result.content[1]
    assert text_block.type == "text"
    assert "out.png" in text_block.text
    assert "original_size" in text_block.text
    assert "returned_size" in text_block.text
    assert "bytes" in text_block.text


# Every tool, the arguments a client would send, and the one API call it is
# expected to make. This is the wiring: a tool bound to the wrong handler or
# handed its arguments in the wrong order shows up here and nowhere else.
def _a_file_to_upload():
    """A real file on disk: upload_asset reads it before it calls out, so
    the wiring test needs one that exists."""
    path = os.path.join(tempfile.mkdtemp(), "iris.png")
    with open(path, "wb") as handle:
        handle.write(b"png-bytes")
    return path


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
    ("get_server_info", {}, "GET", "/api/server"),
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
    (
        # timeout_seconds=0 keeps this to the single poll the wiring test
        # expects: wait_for_job has no route of its own, it reuses get_job's.
        # Its polling loop across several calls is exercised separately in
        # test_mcp_diagnose.py, with WAIT_POLL_SECONDS shrunk so the tests
        # stay fast.
        "wait_for_job",
        {"job_id": "j1", "timeout_seconds": 0},
        "GET",
        "/api/jobs/j1",
    ),
    ("cancel_job", {"job_id": "j1"}, "POST", "/api/jobs/j1/cancel"),
    (
        "rerun_job",
        {"job_id": "j1", "acknowledged_cost": True},
        "POST",
        "/api/jobs/j1/rerun",
    ),
    ("move_job", {"job_id": "j1", "direction": "up"}, "POST", "/api/jobs/j1/move"),
    (
        "download_model",
        {"repo_id": "org/model", "acknowledged_cost": True},
        "POST",
        "/api/models/download",
    ),
    ("list_downloads", {}, "GET", "/api/models/downloads"),
    (
        "cancel_download",
        {"download_id": "d1"},
        "POST",
        "/api/models/downloads/d1/cancel",
    ),
    (
        "delete_model",
        {"repo": "org/model", "acknowledged_cost": True},
        "DELETE",
        "/api/models",
    ),
    ("get_diffusers_state", {}, "GET", "/api/system/diffusers"),
    (
        "update_diffusers",
        {"acknowledged_cost": True},
        "POST",
        "/api/system/diffusers/update",
    ),
    ("list_prompts", {}, "GET", "/api/prompts"),
    ("get_prompt", {"name": "duke"}, "GET", "/api/prompts/duke"),
    ("get_prompt_schema", {}, "GET", "/api/prompt-schema"),
    (
        "save_prompt",
        {"name": "duke", "prompt": {"text": "a duke"}},
        "PUT",
        "/api/prompts/duke",
    ),
    ("delete_prompt", {"name": "duke"}, "DELETE", "/api/prompts/duke"),
    ("list_enhancers", {}, "GET", "/api/enhancers"),
    (
        "enhance_prompt",
        {"idea": "a duke", "acknowledged_cost": True},
        "POST",
        "/api/enhance",
    ),
    ("get_output_text", {"name": "enhanced.txt"}, "GET", "/outputs/enhanced.txt"),
    (
        "download_output",
        {
            "name": "out.png",
            "destination": os.path.join(tempfile.mkdtemp(), "out.png"),
        },
        "GET",
        "/outputs/out.png",
    ),
    ("delete_output", {"name": "out.png"}, "DELETE", "/api/gallery/out.png"),
    ("list_assets", {}, "GET", "/api/assets"),
    (
        "upload_asset",
        {"file_path": _a_file_to_upload()},
        "POST",
        "/api/uploads",
    ),
]


def test_the_wiring_table_covers_every_registered_tool():
    assert {name for name, _, _, _ in TOOL_WIRING} == EXPECTED_TOOLS


@pytest.mark.asyncio
@pytest.mark.parametrize("name,arguments,method,path", TOOL_WIRING)
async def test_each_tool_calls_its_endpoint(name, arguments, method, path):
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.url.path.endswith(".txt"):
            return httpx.Response(
                200, content=b"a duke", headers={"content-type": "text/plain"}
            )
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
    from dw_mcp import __main__ as entry

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


@pytest.mark.asyncio
async def test_rerun_job_takes_an_acknowledged_cost_flag():
    tools = await tools_of(server_over(ok({})))

    assert "acknowledged_cost" in tools["rerun_job"].input_schema["properties"]


@pytest.mark.asyncio
async def test_rerun_job_advertises_its_cost():
    tools = await tools_of(server_over(ok({})))

    description = tools["rerun_job"].description
    assert "COSTS GPU TIME" in description
    assert "acknowledged_cost" in description


@pytest.mark.asyncio
async def test_rerun_job_refuses_without_acknowledgement_and_sends_nothing():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(201, json={"id": "job-2", "status": "queued"})

    server = server_over(handler)

    with pytest.raises(Exception) as caught:
        await server.call_tool("rerun_job", {"job_id": "job-1"})

    assert "acknowledged_cost" in str(caught.value)
    assert seen == []


class TestStartupWeight:
    def test_the_server_starts_without_importing_the_engine(self):
        """`dw_mcp` is a top-level package rather than `dw.mcp` precisely so
        that starting it does not drag in torch and diffusers: it is an HTTP
        client of a server that owns the models, and needs none of them.

        Importing any `dw.*` submodule runs `dw/__init__.py`, which imports
        torch - so a single convenience import from the engine would quietly
        put ~1s of model-framework startup back into every client session.
        This asserts the boundary rather than the timing, which is the part
        a future edit can actually break.
        """
        import subprocess
        import sys

        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import dw_mcp.server; "
                "print(','.join(m for m in ('torch', 'diffusers', 'dw') "
                "if m in sys.modules))",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        assert probe.stdout.strip() == ""


# The arguments each gated tool needs apart from the acknowledgement itself
GATED_ARGUMENTS = {
    "run_workflow": {"workflow_path": "w.json"},
    "rerun_job": {"job_id": "j1"},
    "download_model": {"repo_id": "org/model"},
    "delete_model": {"repo": "org/model"},
    "update_diffusers": {},
    "enhance_prompt": {"idea": "a duke"},
}


def test_every_gated_tool_has_arguments_to_test_with():
    assert set(GATED_ARGUMENTS) == GATED_TOOLS


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(GATED_TOOLS))
async def test_a_gated_tool_refuses_through_the_session(name):
    """The gate has to survive the trip through the SDK.

    A refusal is raised as a DwApiError and only becomes a ToolError at the
    tool boundary; anything else the SDK treats as a crash and replaces the
    text with "Error executing tool <name>". That would leave the model with
    no idea a flag exists, so this asserts the message a client really sees.
    """

    def refusing(request):
        raise AssertionError(f"gate let {request.method} {request.url.path} through")

    server = server_over(refusing)

    with pytest.raises(Exception) as caught:
        await server.call_tool(name, dict(GATED_ARGUMENTS[name]))

    assert "acknowledged_cost=true" in str(caught.value)


# Explicit mapping of wrapper tool name -> (handler_module, handler_function_name)
# Only includes tools whose handlers declare parameter defaults (beyond the client arg).
WRAPPER_HANDLER_MAP = {
    "get_job_events": (diagnose, "get_job_events"),
    "wait_for_job": (diagnose, "wait_for_job"),
    "get_output_image": (media, "get_output_image"),
    "get_class": (catalog, "get_class"),
    "list_gallery": (catalog, "list_gallery"),
    "download_model": (models, "download_model"),
    "delete_model": (models, "delete_model"),
    "update_diffusers": (models, "update_diffusers"),
    "validate_workflow": (authoring, "validate_workflow"),
    "run_workflow": (diagnose, "run_workflow"),
    "rerun_job": (diagnose, "rerun_job"),
    "get_output_text": (media, "get_output_text"),
    "enhance_prompt": (prompts, "enhance_prompt"),
    "download_output": (media, "download_output"),
}


@pytest.mark.asyncio
async def test_wrapper_and_handler_defaults_match():
    """Wrapper functions always pass every argument, so handler defaults
    should never be reached. This test pinpoints any drift: if a developer
    changes a handler default, the wrapper must be updated to match, or the
    change is silently lost.

    Compares inspect.signature defaults for identically named parameters
    between wrapper (from MCP schema) and handler (from inspect.signature).
    """
    server = server_over(ok({}))
    tools = await tools_of(server)

    mismatches = []

    for tool_name, (handler_module, handler_fn_name) in WRAPPER_HANDLER_MAP.items():
        if tool_name not in tools:
            mismatches.append(
                f"{tool_name}: mapped in WRAPPER_HANDLER_MAP but not a registered tool"
            )
            continue

        tool = tools[tool_name]
        handler_fn = getattr(handler_module, handler_fn_name)
        handler_sig = inspect.signature(handler_fn)

        # Extract defaults from handler signature (skip 'client' parameter)
        handler_defaults = {
            param_name: param.default
            for param_name, param in handler_sig.parameters.items()
            if param_name != "client" and param.default is not inspect.Parameter.empty
        }

        # Extract defaults from wrapper tool schema
        tool_schema_defaults = {
            param_name: schema.get("default", inspect.Parameter.empty)
            for param_name, schema in tool.input_schema.get("properties", {}).items()
            if "default" in schema
        }

        # Compare: every handler default must match the tool schema default
        for param_name, handler_default in handler_defaults.items():
            if param_name not in tool_schema_defaults:
                mismatches.append(
                    f"{tool_name}.{param_name}: handler has default {handler_default!r} "
                    f"but wrapper schema has no default"
                )
            elif tool_schema_defaults[param_name] != handler_default:
                mismatches.append(
                    f"{tool_name}.{param_name}: handler default {handler_default!r} "
                    f"does not match wrapper schema default {tool_schema_defaults[param_name]!r}"
                )

    assert not mismatches, "\n".join(mismatches)


def test_wrapper_handler_map_covers_every_defaulted_handler():
    """WRAPPER_HANDLER_MAP is hand-maintained, so a handler gaining a default
    parameter and never being added to the map would go unnoticed - the
    comparison above only checks tools that are already listed. Walk every
    handler module and require any registered-tool handler with a default on
    a non-client parameter to be a mapped key."""
    modules = [catalog, diagnose, media, models, authoring, prompts]

    missing = []
    for module in modules:
        for fn_name, fn in inspect.getmembers(module, inspect.isfunction):
            if fn_name not in EXPECTED_TOOLS:
                continue
            sig = inspect.signature(fn)
            has_default = any(
                param_name != "client" and param.default is not inspect.Parameter.empty
                for param_name, param in sig.parameters.items()
            )
            if has_default and fn_name not in WRAPPER_HANDLER_MAP:
                missing.append(f"{module.__name__}.{fn_name}")

    assert not missing, (
        "handlers with defaulted parameters missing from WRAPPER_HANDLER_MAP: "
        + ", ".join(missing)
    )


def test_the_instructions_explain_how_a_workflow_reaches_a_prompt():
    """The prompt tools are only useful to a model that knows a workflow
    argument reaches the library by `prompt:name` - nothing else in the tool
    surface connects the two halves of authoring."""
    server = server_over(ok({}))

    assert "prompt:" in server.instructions


def test_the_instructions_send_an_agent_to_the_catalog_before_authoring():
    """An agent asked to "run a zimage workflow" should reach for the
    hundred workflows already on disk rather than write a new one. Only
    run_workflow's own docstring says so today, and a model that has
    decided to author never reads it."""
    server = server_over(ok({}))

    assert "list_workflows" in server.instructions
