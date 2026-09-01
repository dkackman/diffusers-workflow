"""Authoring tools: validate before saving, and never guess which workflow
the caller meant."""

import httpx
import pytest

from dw.mcp import authoring
from dw.mcp.client import DwApiError, DwClient

WORKFLOW = {"id": "w", "steps": []}


def scripted(routes):
    """routes: {(method, path): (status, json_body)}"""
    seen = []

    def handler(request):
        key = (request.method, request.url.path)
        seen.append(key)
        if key not in routes:
            return httpx.Response(404, json={"detail": f"unrouted {key}"})
        status, body = routes[key]
        return httpx.Response(status, json=body)

    return DwClient(transport=httpx.MockTransport(handler)), seen


def test_validate_posts_an_inline_workflow():
    client, seen = scripted(
        {
            ("POST", "/api/validate"): (
                200,
                {"valid": True, "error": None, "warnings": []},
            )
        }
    )

    result = authoring.validate_workflow(client, workflow=WORKFLOW)

    assert result["valid"] is True
    assert seen == [("POST", "/api/validate")]


def test_validate_by_name_fetches_then_posts_inline():
    """/api/validate refuses a request that carries only a path, so a
    name has to be resolved to a definition first."""
    client, seen = scripted(
        {
            ("GET", "/api/workflows/mine"): (200, WORKFLOW),
            ("POST", "/api/validate"): (
                200,
                {"valid": True, "error": None, "warnings": []},
            ),
        }
    )

    authoring.validate_workflow(client, name="mine")

    assert seen == [("GET", "/api/workflows/mine"), ("POST", "/api/validate")]


def test_validate_refuses_both_sources_at_once():
    client, _seen = scripted({})

    with pytest.raises(DwApiError, match="exactly one"):
        authoring.validate_workflow(client, workflow=WORKFLOW, name="mine")


def test_validate_refuses_neither_source():
    client, _seen = scripted({})

    with pytest.raises(DwApiError, match="exactly one"):
        authoring.validate_workflow(client)


def test_validate_returns_an_invalid_verdict_rather_than_raising():
    """An invalid workflow is the answer the agent asked for, not a failure."""
    client, _seen = scripted(
        {
            ("POST", "/api/validate"): (
                200,
                {"valid": False, "error": "steps must not be empty", "warnings": []},
            )
        }
    )

    result = authoring.validate_workflow(client, workflow=WORKFLOW)

    assert result["valid"] is False
    assert "steps" in result["error"]


def test_save_puts_the_definition_under_its_name():
    body_seen = {}

    def handler(request):
        body_seen["method"] = request.method
        body_seen["path"] = request.url.path
        body_seen["body"] = request.read()
        return httpx.Response(
            200, json={"name": "mine", "path": "/w/mine.json", "warnings": []}
        )

    client = DwClient(transport=httpx.MockTransport(handler))

    result = authoring.save_workflow(client, "mine", WORKFLOW)

    assert body_seen["method"] == "PUT"
    assert body_seen["path"] == "/api/workflows/mine"
    assert b'"workflow"' in body_seen["body"]
    assert result["name"] == "mine"


def test_save_surfaces_a_rejected_definition():
    client, _seen = scripted(
        {("PUT", "/api/workflows/mine"): (400, {"detail": "steps must be a list"})}
    )

    with pytest.raises(DwApiError, match="steps must be a list"):
        authoring.save_workflow(client, "mine", WORKFLOW)


def test_save_surfaces_a_path_the_server_refuses():
    client, _seen = scripted(
        {
            ("PUT", "/api/workflows/../escape"): (
                400,
                {"detail": "Path traversal is not allowed"},
            )
        }
    )

    with pytest.raises(DwApiError, match="traversal"):
        authoring.save_workflow(client, "../escape", WORKFLOW)


def test_delete_calls_delete():
    client, seen = scripted(
        {("DELETE", "/api/workflows/mine"): (200, {"name": "mine", "deleted": True})}
    )

    assert authoring.delete_workflow(client, "mine")["deleted"] is True
    assert seen == [("DELETE", "/api/workflows/mine")]


def test_delete_surfaces_a_missing_workflow():
    client, _seen = scripted(
        {("DELETE", "/api/workflows/ghost"): (404, {"detail": "No such workflow"})}
    )

    with pytest.raises(DwApiError, match="No such workflow"):
        authoring.delete_workflow(client, "ghost")
