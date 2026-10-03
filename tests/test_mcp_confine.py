"""One confinement rule for the mounted MCP surface's file reads and
writes: the roots the server names, and containment on the resolved real
path. Two copies of it drifted once (#389 reached the write side only)."""

import os

import httpx
import pytest

from dw_mcp import confine
from dw_mcp.client import DwApiError, DwClient


def mounted(handler):
    client = DwClient(transport=httpx.MockTransport(handler))
    client.mounted = True
    return client


def serving_roots(workspace_root, seen=None, libraries=()):
    def handler(request):
        if seen is not None:
            seen.append((request.url.path, request.url.params.get("workspace")))
        if request.url.path == "/api/server":
            return httpx.Response(
                200, json={"directories": {"workspace": str(workspace_root)}}
            )
        return httpx.Response(200, json={"libraries": list(libraries)})

    return handler


def test_a_stdio_client_is_not_confined():
    client = DwClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert confine.remote_roots(client, None, ("workspace",), refusal="x") is None


def test_the_per_call_workspace_reaches_both_routes(tmp_path):
    seen = []
    client = mounted(serving_roots(tmp_path, seen))
    confine.remote_roots(
        client, "other", ("workspace",), writable_libraries=True, refusal="x"
    )
    assert seen == [("/api/server", "other"), ("/api/assets", "other")]


def test_a_writable_library_is_a_root_and_a_read_only_one_is_not(tmp_path):
    shared, examples = tmp_path / "common", tmp_path / "examples"
    shared.mkdir()
    examples.mkdir()
    libraries = [
        {"root": str(shared), "writable": True},
        {"root": str(examples), "writable": False},
    ]
    client = mounted(serving_roots(tmp_path / "ws", libraries=libraries))
    roots = confine.remote_roots(
        client, None, ("workspace",), writable_libraries=True, refusal="x"
    )
    assert os.path.realpath(str(shared)) in roots
    assert os.path.realpath(str(examples)) not in roots


def test_a_server_naming_no_roots_is_refused_with_the_callers_message():
    client = mounted(lambda r: httpx.Response(200, json={"directories": {}}))
    with pytest.raises(DwApiError, match="cannot say"):
        confine.remote_roots(client, None, ("workspace",), refusal="cannot say")


def test_a_symlink_out_of_a_root_is_outside(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "link").symlink_to(outside)
    roots = [os.path.realpath(str(root))]
    assert not confine.contains(str(root / "link" / "secret.png"), roots)


def test_a_file_not_yet_written_under_a_root_is_inside(tmp_path):
    roots = [os.path.realpath(str(tmp_path))]
    assert confine.contains(str(tmp_path / "new" / "deeper" / "file.png"), roots)
    assert not confine.contains(str(tmp_path.parent / "elsewhere.png"), roots)
