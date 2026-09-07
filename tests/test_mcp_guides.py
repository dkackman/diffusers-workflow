"""The guide tools are proxies: the guides an agent reads have to be the
guides for the engine it is about to drive, so nothing is read locally."""

import httpx
import pytest

from dw_mcp import guides
from dw_mcp.client import DwApiError, DwClient


def scripted(routes):
    """routes: {(method, path): (status, json_body)}; records params and wire path."""
    seen = []

    def handler(request):
        key = (request.method, request.url.path)
        seen.append(
            {
                "key": key,
                "params": dict(request.url.params),
                "raw_path": request.url.raw_path,
            }
        )
        if key not in routes:
            return httpx.Response(404, json={"detail": f"unrouted {key}"})
        status, body = routes[key]
        return httpx.Response(status, json=body)

    return DwClient(transport=httpx.MockTransport(handler)), seen


LISTING = {
    "guides": [
        {
            "name": "tasks",
            "file": "TASKS.md",
            "summary": "The utility task commands.",
            "sections": ["Speech Generation"],
        }
    ]
}


def test_list_guides_passes_the_listing_through():
    client, seen = scripted({("GET", "/api/guides"): (200, LISTING)})

    assert guides.list_guides(client) == LISTING
    assert seen[0]["key"] == ("GET", "/api/guides")


def test_get_guide_whole_sends_no_section():
    body = {"name": "tasks", "section": None, "content": "## Speech Generation\n"}
    client, seen = scripted({("GET", "/api/guides/tasks"): (200, body)})

    assert guides.get_guide(client, "tasks") == body
    assert seen[0]["params"] == {}


def test_get_guide_section_is_a_query_parameter():
    body = {"name": "tasks", "section": "Speech Generation", "content": "..."}
    client, seen = scripted({("GET", "/api/guides/tasks"): (200, body)})

    assert guides.get_guide(client, "tasks", section="speech-generation") == body
    assert seen[0]["params"] == {"section": "speech-generation"}


def test_a_404_detail_reaches_the_model_as_the_message():
    # The server writes "No guide named 'x'. The guides are: ..." - that text
    # is the answer, and it must not be replaced by an HTTP status
    client, _seen = scripted(
        {
            ("GET", "/api/guides/nonexistent"): (
                404,
                {"detail": "No guide named 'nonexistent'. The guides are: tasks."},
            )
        }
    )

    with pytest.raises(DwApiError, match="The guides are: tasks"):
        guides.get_guide(client, "nonexistent")


def test_a_guide_name_is_path_encoded():
    """A name with '..' must reach the server intact, so its own validation
    - not httpx's dot-segment normalisation - decides what it means.
    httpx.URL.path decodes escapes for display; only raw_path shows the
    wire bytes."""
    client, seen = scripted({})

    with pytest.raises(DwApiError):
        guides.get_guide(client, "../escape")

    assert seen[0]["raw_path"] == b"/api/guides/..%2Fescape"
