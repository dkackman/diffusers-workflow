"""The HTTP layer under the MCP tools: URL resolution and the one place
every API failure is turned into a message a non-developer can act on."""

import re
from pathlib import Path

import httpx
import pytest

from dw_mcp.client import DwApiError, DwClient, api_path, path_segment, resolve_base_url


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


def test_get_bytes_if_downloads_the_body_when_accepted():
    def handler(request):
        return httpx.Response(
            200, content=b"\x89PNG", headers={"content-type": "image/png"}
        )

    body, content_type = client_with(handler).get_bytes_if(
        "/outputs/a.png", lambda ct: ct.startswith("image/")
    )

    assert body == b"\x89PNG"
    assert content_type == "image/png"


def test_get_bytes_if_rejects_without_reading_the_body():
    class TrackingStream(httpx.SyncByteStream):
        def __init__(self, chunks):
            self.chunks = chunks
            self.iterated = False

        def __iter__(self):
            self.iterated = True
            yield from self.chunks

        def close(self):
            pass

    stream = TrackingStream([b"\x00\x00\x00\x18ftypmp42" * 1000])

    def handler(request):
        return httpx.Response(200, headers={"content-type": "video/mp4"}, stream=stream)

    client = client_with(handler)

    body, content_type = client.get_bytes_if(
        "/outputs/clip.mp4", lambda ct: ct.startswith("image/")
    )

    assert body is None
    assert content_type == "video/mp4"
    assert stream.iterated is False


def test_get_bytes_if_translates_a_body_read_failure():
    """Anything httpx raises has to arrive as a DwApiError, even when it
    happens while reading the body after the headers already came back
    clean - not just on the initial send()."""

    class FailingStream(httpx.SyncByteStream):
        def __iter__(self):
            raise httpx.ReadError("connection reset", request=None)
            yield b""  # pragma: no cover

        def close(self):
            pass

    def handler(request):
        return httpx.Response(
            200, headers={"content-type": "image/png"}, stream=FailingStream()
        )

    client = client_with(handler)

    with pytest.raises(DwApiError) as caught:
        client.get_bytes_if("/outputs/a.png", lambda ct: ct.startswith("image/"))

    assert "/outputs/a.png" in str(caught.value)


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


def test_api_path_joins_literal_and_encoded_segments():
    assert api_path("api", "jobs", "j1", "cancel") == "/api/jobs/j1/cancel"


def test_api_path_quotes_a_slash_in_a_segment():
    assert (
        api_path("api", "workflows", "flux/FluxDev") == "/api/workflows/flux%2FFluxDev"
    )


def test_api_path_quotes_dot_segments_so_they_cannot_traverse():
    assert api_path("api", "workflows", "../escape") == "/api/workflows/..%2Fescape"


def test_api_path_quotes_a_hash():
    assert api_path("api", "gallery", "a#1") == "/api/gallery/a%231"


def test_api_path_quotes_a_space():
    assert api_path("outputs", "a b.png") == "/outputs/a%20b.png"


# Every dw_mcp handler module that talks to the API must build request paths
# through `api_path`, not by hand - `path_segment` alone is easy to forget on
# one call site among many, and a bare f-string skips quoting entirely. This
# scans source text rather than testing behavior, because the bug this guards
# against is a *new* call site written the old way, which no behavioral test
# would catch until someone thought to write it.
HANDLER_MODULES = [
    "authoring.py",
    "catalog.py",
    "diagnose.py",
    "media.py",
    "models.py",
    "server.py",
]

# An f-string whose literal part begins with /api or /outputs and contains an
# interpolation, e.g. f"/api/jobs/{job_id}" or f"/outputs/{name}/thumb".
RAW_INTERPOLATED_PATH = re.compile(r"""f(['"])/(?:api|outputs)[^'"]*\{[^'"]*\1""")


@pytest.mark.parametrize("module_name", HANDLER_MODULES)
def test_handler_modules_do_not_build_paths_by_hand(module_name):
    source = (Path(__file__).parent.parent / "dw_mcp" / module_name).read_text()

    assert "path_segment(" not in source
    assert not RAW_INTERPOLATED_PATH.search(source)


def test_a_non_json_error_body_does_not_mask_the_status():
    def handler(request):
        return httpx.Response(404, text="<html>not found</html>")

    with pytest.raises(DwApiError) as caught:
        client_with(handler).get_json("/api/workflows/x")

    assert "404" in str(caught.value)


def test_a_transport_level_failure_still_names_the_request():
    """Anything httpx raises has to arrive as a DwApiError - a bare
    httpx exception reaching the tool layer would be reported as a crash."""

    def handler(request):
        raise httpx.ReadError("connection reset", request=request)

    with pytest.raises(DwApiError) as caught:
        client_with(handler).get_json("/api/health")

    assert "/api/health" in str(caught.value)
    assert "connection reset" in str(caught.value)


def test_a_successful_response_that_is_not_json_is_reported_as_such():
    def handler(request):
        return httpx.Response(200, text="<html>not the API</html>")

    with pytest.raises(DwApiError) as caught:
        client_with(handler).get_json("/api/health")

    assert "non-JSON" in str(caught.value)
    assert "not the API" in str(caught.value)


def test_close_releases_the_underlying_http_client():
    client = client_with(lambda request: httpx.Response(200, json={}))

    client.close()

    assert client._http.is_closed


def test_a_connect_timeout_says_how_to_start_the_server():
    """ConnectTimeout is a subclass of TimeoutException, so it must be caught
    before TimeoutException to show the "cannot reach" message instead of the
    "server may be busy" timeout message."""

    def handler(request):
        raise httpx.ConnectTimeout("connection timed out", request=request)

    client = client_with(handler, base_url="http://127.0.0.1:8765")
    with pytest.raises(DwApiError) as caught:
        client.get_json("/api/health")

    message = str(caught.value)
    assert "Cannot reach" in message
    assert "http://127.0.0.1:8765" in message
    assert "dw-serve" in message


def test_a_422_with_list_detail_formats_it_as_human_readable():
    """FastAPI validation errors (422) have detail as a list of dicts.
    Format each as 'field: message' instead of showing the Python repr."""

    def handler(request):
        return httpx.Response(
            422,
            json={
                "detail": [
                    {
                        "loc": ["body", "model_name"],
                        "msg": "Field required",
                        "type": "value_error.missing",
                    },
                    {
                        "loc": ["body", "num_steps"],
                        "msg": "ensure this value is greater than 0",
                        "type": "value_error.number.not_gt",
                    },
                ]
            },
        )

    with pytest.raises(DwApiError) as caught:
        client_with(handler).post_json("/api/validate", {})

    message = str(caught.value)
    # Should have readable field names, not Python repr
    assert "model_name" in message
    assert "num_steps" in message
    # Should have the error messages
    assert "Field required" in message
    assert "ensure this value is greater than 0" in message
    # Should not contain Python dict repr fragments
    assert "[{" not in message
    assert "'loc'" not in message


def test_a_configured_token_rides_along_as_a_bearer_header(monkeypatch):
    """dw.serve --token gates /api/*; the MCP client must be able to present
    it. DW_API_TOKEN is the same variable dw.serve itself reads, so one
    export configures both ends."""
    monkeypatch.setenv("DW_API_TOKEN", "s3cr3t")
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    client_with(handler).get_json("/api/health")
    assert seen["auth"] == "Bearer s3cr3t"


def test_an_explicit_token_wins_over_the_environment(monkeypatch):
    monkeypatch.setenv("DW_API_TOKEN", "from-env")
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    client_with(handler, token="explicit").get_json("/api/health")
    assert seen["auth"] == "Bearer explicit"


def test_no_token_means_no_authorization_header(monkeypatch):
    monkeypatch.delenv("DW_API_TOKEN", raising=False)
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    client_with(handler).get_json("/api/health")
    assert seen["auth"] is None
