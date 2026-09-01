"""The read-only tools: each is a thin pass-through, so the tests pin the
route, the parameters, and that nothing is reshaped on the way back."""

import httpx
import pytest

from dw_mcp import catalog
from dw_mcp.client import DwApiError, DwClient


def recording_client(body=None, status=200):
    """A client whose transport records the request it was given."""
    seen = {}

    def handler(request):
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(status, json=body if body is not None else {})

    return DwClient(transport=httpx.MockTransport(handler)), seen


@pytest.mark.parametrize(
    "call, path",
    [
        (lambda c: catalog.list_workflows(c), "/api/workflows"),
        (lambda c: catalog.get_workflow(c, "folder/w"), "/api/workflows/folder/w"),
        (lambda c: catalog.get_schema(c), "/api/schema"),
        (lambda c: catalog.list_pipelines(c), "/api/pipelines"),
        (
            lambda c: catalog.get_pipeline_signature(c, "FluxPipeline"),
            "/api/pipelines/FluxPipeline",
        ),
        (lambda c: catalog.list_classes(c, "schedulers"), "/api/classes"),
        (
            lambda c: catalog.get_class(c, "diffusers.AutoencoderKL"),
            "/api/classes/diffusers.AutoencoderKL",
        ),
        (lambda c: catalog.list_tasks(c), "/api/tasks"),
        (lambda c: catalog.get_task(c, "upscale"), "/api/tasks/upscale"),
        (lambda c: catalog.list_models(c), "/api/models"),
        (lambda c: catalog.get_memory(c), "/api/memory"),
        (lambda c: catalog.get_health(c), "/api/health"),
        (lambda c: catalog.list_jobs(c), "/api/jobs"),
        (lambda c: catalog.list_gallery(c), "/api/gallery"),
        (
            lambda c: catalog.get_gallery_metadata(c, "a.png"),
            "/api/gallery/a.png/metadata",
        ),
    ],
)
def test_each_catalog_tool_calls_its_route(call, path):
    client, seen = recording_client()

    call(client)

    assert seen["method"] == "GET"
    assert seen["path"] == path


def test_list_classes_sends_the_required_kind():
    client, seen = recording_client()

    catalog.list_classes(client, "quantization")

    assert seen["params"]["kind"] == "quantization"


def test_get_class_sends_the_target():
    client, seen = recording_client()

    catalog.get_class(client, "diffusers.FluxPipeline", target="call")

    assert seen["params"]["target"] == "call"


def test_list_gallery_sends_its_limit():
    client, seen = recording_client()

    catalog.list_gallery(client, limit=7)

    assert seen["params"]["limit"] == "7"


def test_a_pass_through_tool_returns_the_body_unchanged():
    client, _seen = recording_client({"workflows": ["a"], "details": {}})

    assert catalog.list_workflows(client) == {"workflows": ["a"], "details": {}}


def test_a_missing_workflow_propagates_the_api_error():
    def handler(request):
        return httpx.Response(404, json={"detail": "No such workflow: ghost"})

    client = DwClient(transport=httpx.MockTransport(handler))

    with pytest.raises(DwApiError, match="ghost"):
        catalog.get_workflow(client, "ghost")


def test_get_workflow_sends_the_name_percent_encoded_on_the_wire():
    """httpx.URL.path decodes escapes back for display, so an unquoted and
    a quoted request can look identical on `.path` - only the wire bytes
    (`.raw_path`) tell them apart. A name with '..' has to survive intact
    onto the wire so the server's own traversal check is what refuses it,
    rather than httpx silently normalizing the dot-segment away first."""
    seen = {}

    def handler(request):
        seen["raw_path"] = request.url.raw_path
        return httpx.Response(200, json={})

    client = DwClient(transport=httpx.MockTransport(handler))

    catalog.get_workflow(client, "../escape")

    assert seen["raw_path"] == b"/api/workflows/..%2Fescape"
