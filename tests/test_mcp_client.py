"""The HTTP layer under the MCP tools: URL resolution and the one place
every API failure is turned into a message a non-developer can act on."""

import httpx
import pytest

from dw.mcp.client import DwApiError, DwClient, path_segment, resolve_base_url


def client_with(handler, **kwargs):
    return DwClient(transport=httpx.MockTransport(handler), **kwargs)


def test_resolve_base_url_prefers_the_explicit_value(monkeypatch):
    monkeypatch.setenv("DW_MCP_URL", "http://127.0.0.1:9999")
    assert resolve_base_url("http://127.0.0.1:1234") == "http://127.0.0.1:1234"


def test_resolve_base_url_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("DW_MCP_URL", "http://127.0.0.1:9999/")
    assert resolve_base_url(None) == "http://127.0.0.1:9999"


def test_resolve_base_url_defaults_to_the_serve_port(monkeypatch):
    monkeypatch.delenv("DW_MCP_URL", raising=False)
    assert resolve_base_url(None) == "http://127.0.0.1:8765"


def test_get_json_returns_the_decoded_body():
    def handler(request):
        assert request.url.path == "/api/health"
        return httpx.Response(200, json={"status": "ok"})

    assert client_with(handler).get_json("/api/health") == {"status": "ok"}


def test_get_json_passes_query_parameters():
    def handler(request):
        assert request.url.params["limit"] == "5"
        return httpx.Response(200, json={"files": []})

    client_with(handler).get_json("/api/gallery", params={"limit": 5})


def test_post_put_and_delete_send_the_right_method_and_body():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, request.read()))
        return httpx.Response(200, json={"ok": True})

    client = client_with(handler)
    client.post_json("/api/validate", {"workflow": {"id": "w"}})
    client.put_json("/api/workflows/w", {"workflow": {"id": "w"}})
    client.delete_json("/api/workflows/w")

    assert [entry[0] for entry in seen] == ["POST", "PUT", "DELETE"]
    assert b'"id":"w"' in seen[0][2].replace(b" ", b"")


def test_get_bytes_returns_the_body_and_content_type():
    def handler(request):
        return httpx.Response(
            200, content=b"\x89PNG", headers={"content-type": "image/png"}
        )

    body, content_type = client_with(handler).get_bytes("/outputs/a.png")

    assert body == b"\x89PNG"
    assert content_type == "image/png"


def test_a_refused_connection_says_how_to_start_the_server():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    client = client_with(handler, base_url="http://127.0.0.1:8765")
    with pytest.raises(DwApiError) as caught:
        client.get_json("/api/health")

    message = str(caught.value)
    assert "http://127.0.0.1:8765" in message
    assert "dw-serve" in message


def test_a_timeout_names_the_request():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(DwApiError) as caught:
        client_with(handler).get_json("/api/models")

    assert "/api/models" in str(caught.value)
    assert "timed out" in str(caught.value).lower()


def test_a_400_surfaces_the_servers_detail_verbatim():
    def handler(request):
        return httpx.Response(400, json={"detail": "steps must be a list"})

    with pytest.raises(DwApiError) as caught:
        client_with(handler).post_json("/api/validate", {})

    assert str(caught.value) == "steps must be a list"


def test_a_404_names_what_was_missing():
    def handler(request):
        return httpx.Response(404, json={"detail": "Unknown job"})

    with pytest.raises(DwApiError) as caught:
        client_with(handler).get_json("/api/jobs/nope")

    assert "Unknown job" in str(caught.value)


def test_a_500_is_labelled_a_server_side_failure():
    def handler(request):
        return httpx.Response(500, text="boom")

    with pytest.raises(DwApiError) as caught:
        client_with(handler).get_json("/api/memory")

    message = str(caught.value)
    assert "500" in message
    assert "boom" in message


def test_a_5xx_with_a_json_detail_is_still_labelled_a_server_side_failure():
    def handler(request):
        return httpx.Response(500, json={"detail": "Could not read workflow: boom"})

    with pytest.raises(DwApiError) as caught:
        client_with(handler).get_json("/api/workflows/x")

    message = str(caught.value)
    assert "500" in message
    assert message != "Could not read workflow: boom"


def test_path_segment_encodes_characters_a_url_path_cannot_carry_raw():
    # Pins the exact encoding so a later "simplification" to the default
    # safe="/" (which would leave '/' - and so a traversal segment like
    # "../escape" - unescaped) breaks visibly.
    assert path_segment("a b#1") == "a%20b%231"


def test_path_segment_encodes_its_own_slashes():
    assert path_segment("folder/w") == "folder%2Fw"


def test_a_non_json_error_body_does_not_mask_the_status():
    def handler(request):
        return httpx.Response(404, text="<html>not found</html>")

    with pytest.raises(DwApiError) as caught:
        client_with(handler).get_json("/api/workflows/x")

    assert "404" in str(caught.value)
