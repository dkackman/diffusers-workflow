"""Snapshot the server's public surface as one JSON file.

For the Phase 3 stage 3b/3c surface diff: run it at a task's base and at its
head, and the two files must be identical (3b), or differ by exactly the
listed breaking changes (3c). It records the route set, the order inside
each greedy `{name:path}` family, the tail entries (/mcp, SPA mount), the
middleware stack, the OpenAPI document and the MCP tool list.

    venv/bin/python scripts/surface_snapshot.py OUT.json
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from dw.introspection import describe_task, list_tasks  # noqa: E402
from dw.server.app import create_app  # noqa: E402
from dw.server.jobs import JobManager  # noqa: E402
from dw_mcp.client import DwClient  # noqa: E402
from dw_mcp.server import build_server  # noqa: E402


class StubWorkerManager:
    """Enough of WorkerManager for a JobManager that never runs a job."""

    worker_active = False

    def shutdown_worker(self):
        pass

    def shutdown(self):
        pass


def route_entry(route):
    endpoint = getattr(route, "endpoint", None)
    return {
        "methods": sorted(getattr(route, "methods", None) or []),
        "name": route.name,
        "path": route.path,
        "query_token_ok": bool(getattr(endpoint, "query_token_ok", False)),
        "type": type(route).__name__,
    }


def greedy_families(routes):
    """For each `{name:path}` prefix, the (method, path) members under it in
    router order. Methods are kept apart in the member list rather than in
    separate families, so a DELETE moving relative to a GET shows as a diff
    too (stricter than the per-method rule Starlette needs)."""
    prefixes = sorted({r.path.split("{", 1)[0] for r in routes if ":path}" in r.path})
    families = {}
    for prefix in prefixes:
        members = [
            f"{method} {r.path}"
            for r in routes
            if r.path.startswith(prefix)
            for method in sorted(getattr(r, "methods", None) or ["*"])
        ]
        if len(members) > 1:
            families[prefix] = members
    return families


def middleware_entry(middleware):
    dispatch = middleware.kwargs.get("dispatch")
    return {
        "class": middleware.cls.__name__,
        "dispatch": getattr(dispatch, "__name__", None),
        "kwargs": sorted(k for k in middleware.kwargs if k != "dispatch"),
    }


def mcp_server():
    def refuse(request):
        return httpx.Response(500)

    return build_server(DwClient(transport=httpx.MockTransport(refuse)))


def mcp_tools(server):
    tools = asyncio.run(server.list_tools())
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.input_schema,
            "annotations": tool.annotations.model_dump(mode="json")
            if tool.annotations
            else None,
        }
        for tool in tools
    ]


SNAPSHOT_WORKFLOW = {
    "id": "snapshot",
    "variables": {"prompt": "d"},
    "steps": [
        {
            "name": "gen",
            "pipeline": {
                "configuration": {"component_type": "{Fake}", "no_generator": True},
                "from_pretrained_arguments": {"model_name": "m"},
                "arguments": {"prompt": "variable:prompt"},
            },
        }
    ],
}


def write_json(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))


def build_library_fixture(root):
    """A workspace and an examples tree that exercise every rule the library
    listings apply: a workspace workflow shadowing an example one, a prompt in
    each, and an asset in the workspace, in common/assets and in the example
    (one name in two of them, so `shadowed` has an entry).

    Returns the examples directory (the workflows/ tree) to hand create_app.
    """
    examples = root / "examples"
    write_json(root / "workflows" / "shared-name.json", SNAPSHOT_WORKFLOW)
    write_json(examples / "workflows" / "shared-name.json", SNAPSHOT_WORKFLOW)
    write_json(examples / "workflows" / "example-only.json", SNAPSHOT_WORKFLOW)
    write_json(root / "prompts" / "own.json", {"text": "the workspace's prompt"})
    write_json(examples / "prompts" / "example.json", {"text": "the example's prompt"})
    for path, body in (
        (root / "assets" / "own.png", b"own"),
        (root / "assets" / "shared.png", b"workspace copy"),
        (root / "common" / "assets" / "common.png", b"common"),
        (examples / "assets" / "example.png", b"example"),
        (examples / "assets" / "shared.png", b"example copy"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    return str(examples / "workflows")


def first_entry_keys(items):
    """Sorted keys of one entry of a listing (a list's first, a dict's first
    value), or None when the listing holds none."""
    if isinstance(items, dict):
        items = list(items.values())
    return sorted(items[0]) if items else None


def listing_keys(client):
    """The shape of the three library listings: the sorted key set of each
    body and of one entry per listing. Keys only - no paths, no times - so
    the snapshot is the same on every run and every machine."""
    workflows = client.get("/api/workflows").json()
    prompts = client.get("/api/prompts").json()
    assets = client.get("/api/assets").json()
    return {
        "workflows": {
            "body": sorted(workflows),
            "details_entry": first_entry_keys(workflows["details"]),
        },
        "prompts": {
            "body": sorted(prompts),
            "details_entry": first_entry_keys(prompts["details"]),
        },
        "assets": {
            "body": sorted(assets),
            "assets_entry": first_entry_keys(assets["assets"]),
            "shadowed_entry": first_entry_keys(assets["shadowed"]),
        },
    }


def task_surface():
    """`list_tasks()` and `describe_task(c)` for every command. A command whose
    description raises is recorded as its error string rather than crashing."""
    listing = list_tasks()
    described = {}
    for command in listing["commands"]:
        try:
            described[command] = describe_task(command)
        except Exception as error:
            described[command] = f"error: {type(error).__name__}: {error}"
    return {"list": listing, "describe": described}


def workflow_schema():
    path = Path(__file__).resolve().parent.parent / "dw" / "workflow_schema.json"
    return json.loads(path.read_text())


def snapshot(root):
    root = Path(root)
    examples_dir = build_library_fixture(root)
    server = mcp_server()
    ui_dir = root / "ui"
    ui_dir.mkdir()
    (ui_dir / "index.html").write_text("<html></html>")
    workflows = str(root / "workflows")
    manager = JobManager(
        root / "outputs",
        worker_manager=StubWorkerManager(),
        history_path=root / "jobs.sqlite",
        workflow_dir=workflows,
    )
    app = create_app(
        workflow_dir=workflows,
        output_dir=str(root / "outputs"),
        job_manager=manager,
        ui_dir=str(ui_dir),
        prompt_dir=str(root / "prompts"),
        asset_dir=str(root / "assets"),
        workspace=str(root),
        examples_dirs=[examples_dir],
        mcp=True,
    )
    routes = list(app.router.routes)
    entries = [route_entry(r) for r in routes]
    tail_start = next(i for i, e in enumerate(entries) if e["path"] == "/mcp")
    return {
        "routes": sorted(entries, key=lambda e: (e["path"], e["methods"])),
        "greedy_families": greedy_families(routes),
        "tail": entries[tail_start:],
        "middleware": [middleware_entry(m) for m in app.user_middleware],
        "openapi": app.openapi(),
        "mcp_instructions": server.instructions,
        "mcp_tools": mcp_tools(server),
        "library_listings": listing_keys(TestClient(app, base_url="http://localhost")),
        "tasks": task_surface(),
        "workflow_schema": workflow_schema(),
    }


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: surface_snapshot.py OUT.json")
    with tempfile.TemporaryDirectory() as root:
        data = snapshot(root)
    Path(sys.argv[1]).write_text(
        json.dumps(data, indent=2, sort_keys=True, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
