"""Getting an input file to the server: the asset library over MCP.

An agent that can only name assets already on the box can author workflows
it cannot supply inputs for - these are the two tools that close that.
"""

import json

import httpx
import pytest

import base64

from dw_mcp.assets import (
    MAX_INLINE_UPLOAD_BYTES,
    MAX_UPLOAD_BYTES,
    delete_asset,
    keep_output,
    list_assets,
    upload_asset,
)
from dw_mcp.client import DwApiError, DwClient


def client_over(handler):
    return DwClient(transport=httpx.MockTransport(handler))


def recording():
    """A client that answers every upload, recording what it was sent."""
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = request.content
        return httpx.Response(
            201,
            json={
                "reference": "asset:uploads/deadbeef.png",
                "url": "/inputs/uploads/deadbeef.png",
            },
        )

    return client_over(handler), seen


class TestListing:
    def test_assets_come_back_by_reference(self):
        def handler(request):
            assert request.url.path == "/api/assets"
            return httpx.Response(
                200,
                json={
                    "asset_dir": "/studio/assets",
                    "assets": [
                        {
                            "name": "uploads/iris.png",
                            "reference": "asset:uploads/iris.png",
                            "kind": "image",
                            "size": 1234,
                        }
                    ],
                    "folders": ["", "uploads"],
                },
            )

        result = list_assets(client_over(handler))
        assert result["assets"][0]["reference"] == "asset:uploads/iris.png"

    def test_a_workspace_can_be_named_for_one_request(self):
        """#463: listing another workspace's library for one call must not
        depend on switching the session's own pin."""
        seen = {}

        def handler(request):
            seen["params"] = dict(request.url.params)
            return httpx.Response(
                200, json={"asset_dir": "/w/A/assets", "assets": [], "folders": []}
            )

        list_assets(client_over(handler), workspace="A")
        assert seen["params"]["workspace"] == "A"


class TestUpload:
    def test_a_file_is_pushed_and_the_reference_returned(self, tmp_path):
        source = tmp_path / "iris.png"
        source.write_bytes(b"png-bytes")
        client, seen = recording()

        result = upload_asset(client, str(source))

        assert seen["body"] == b"png-bytes"
        assert "filename=iris.png" in seen["url"]
        assert result["reference"] == "asset:uploads/deadbeef.png"
        assert result["uploaded"] == "iris.png"
        assert result["size"] == len(b"png-bytes")

    def test_a_name_is_passed_through_when_one_is_given(self, tmp_path):
        source = tmp_path / "clip-01.wav"
        source.write_bytes(b"wav-bytes")
        client, seen = recording()

        upload_asset(client, str(source), asset_name="cast/priya-voice")

        assert "asset_name=cast" in seen["url"].replace("%2F", "/")

    def test_no_name_leaves_the_server_to_choose_one(self, tmp_path):
        source = tmp_path / "clip-01.wav"
        source.write_bytes(b"wav-bytes")
        client, seen = recording()

        upload_asset(client, str(source))

        assert "asset_name" not in seen["url"]

    def test_only_the_base_name_is_sent(self, tmp_path):
        # The server generates the stored name; the directory this file sits
        # in is this machine's business and means nothing over there
        nested = tmp_path / "deep" / "tree"
        nested.mkdir(parents=True)
        source = nested / "frame.png"
        source.write_bytes(b"x")
        client, seen = recording()

        upload_asset(client, str(source))
        assert "deep" not in seen["url"]

    def test_a_missing_file_says_so(self, tmp_path):
        client, _seen = recording()
        with pytest.raises(DwApiError, match="No such file"):
            upload_asset(client, str(tmp_path / "absent.png"))

    def test_a_kind_the_library_does_not_take_is_refused(self, tmp_path):
        source = tmp_path / "script.py"
        source.write_text("print(1)")
        client, _seen = recording()
        with pytest.raises(DwApiError, match="not a kind"):
            upload_asset(client, str(source))

    def test_audio_is_a_kind_the_library_takes(self, tmp_path):
        # A workflow's audio reference is built from a .wav, so refusing it
        # would leave one input kind with no way onto the machine
        source = tmp_path / "voice.wav"
        source.write_bytes(b"riff")
        client, seen = recording()
        upload_asset(client, str(source))
        assert seen["body"] == b"riff"

    def test_an_oversized_file_fails_before_it_is_read(self, tmp_path, monkeypatch):
        source = tmp_path / "huge.mp4"
        source.write_bytes(b"0")
        monkeypatch.setattr("os.path.getsize", lambda _path: MAX_UPLOAD_BYTES + 1)
        client, seen = recording()
        with pytest.raises(DwApiError, match="upload limit"):
            upload_asset(client, str(source))
        assert "body" not in seen

    def test_a_server_with_no_library_still_reports_the_url(self, tmp_path):
        """A server with no asset library configured gives no reference at
        all - there is nothing an `asset:` argument could name - only a URL
        to fetch the file back from. Report that rather than inventing a
        reference out of it (#527)."""
        source = tmp_path / "iris.png"
        source.write_bytes(b"png")

        def handler(request):
            return httpx.Response(201, json={"url": "/outputs/x.png"})

        result = upload_asset(client_over(handler), str(source))
        assert result["reference"] is None
        assert result["url"] == "/outputs/x.png"

    def test_a_workspace_can_be_named_for_one_request(self, tmp_path):
        """#463: an upload pinned to another workspace must not depend on
        switching the session's own pin."""
        source = tmp_path / "iris.png"
        source.write_bytes(b"png-bytes")
        seen = {}

        def handler(request):
            seen["params"] = dict(request.url.params)
            return httpx.Response(
                201,
                json={
                    "reference": "asset:uploads/deadbeef.png",
                    "workspace": "A",
                    "url": "/inputs/uploads/deadbeef.png",
                },
            )

        result = upload_asset(client_over(handler), str(source), workspace="A")
        assert seen["params"]["workspace"] == "A"
        assert result["workspace"] == "A"


class TestKeeping:
    def test_keeping_sends_no_bytes(self):
        """The whole point: the copy happens on the server, so a render is
        not downloaded here only to be uploaded back."""
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["body"] = request.content
            return httpx.Response(
                201,
                json={
                    "reference": "asset:gyre/hero.png",
                    "name": "gyre/hero.png",
                    "linked": True,
                },
            )

        result = keep_output(
            client_over(handler),
            "Gyre/20260905-101500-aaaaaaaa/still.png",
            asset_name="gyre/hero.png",
        )
        assert result["reference"] == "asset:gyre/hero.png"
        # only the two names travelled, not the file
        body = json.loads(seen["body"])
        assert body["name"] == "Gyre/20260905-101500-aaaaaaaa/still.png"
        assert body["asset_name"] == "gyre/hero.png"
        assert "/api/assets/keep" in seen["url"]

    def test_a_refusal_reaches_the_caller(self):
        def handler(request):
            return httpx.Response(409, json={"detail": "asset:hero.png already exists"})

        with pytest.raises(DwApiError, match="already exists"):
            keep_output(client_over(handler), "Gyre/run/still.png")


class TestDeleting:
    """The counterpart to upload and keep: without it a mistaken name could
    only be cleaned up on the box (T014)."""

    def test_the_name_travels_as_a_path_segment(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            seen["method"] = request.method
            return httpx.Response(
                200,
                json={
                    "name": "qa-cast/priya-voice",
                    "deleted": True,
                    "origin": "common",
                },
            )

        result = delete_asset(client_over(handler), "qa-cast/priya-voice")
        assert result["deleted"] is True
        assert seen["method"] == "DELETE"
        assert "/api/assets/" in seen["url"]

    def test_a_read_only_asset_refusal_reaches_the_caller(self):
        def handler(request):
            return httpx.Response(403, json={"detail": "asset:iris.png is read-only"})

        with pytest.raises(DwApiError, match="read-only"):
            delete_asset(client_over(handler), "iris.png")

    def test_a_workspace_can_be_named_for_one_request(self):
        """#463: the exact incident this issue fixes - a delete pinned to the
        workspace named on the call must not follow the session's own pin,
        which another connection on a shared mount can move underneath it."""
        seen = {}

        def handler(request):
            seen["params"] = dict(request.url.params)
            return httpx.Response(
                200,
                json={
                    "name": "qa-cast/priya-voice",
                    "workspace": "A",
                    "reference": "asset:qa-cast/priya-voice.wav",
                    "deleted": True,
                    "origin": "workspace",
                },
            )

        result = delete_asset(
            client_over(handler), "qa-cast/priya-voice", workspace="A"
        )
        assert seen["params"]["workspace"] == "A"
        assert result["workspace"] == "A"
        assert result["deleted"] is True


class TestUploadInlineContent:
    """The content= path (#203): bytes an agent holds itself, with no path
    behind them at all, for a mounted endpoint with no filesystem in common
    with the caller."""

    def test_content_is_decoded_and_posted(self):
        client, seen = recording()
        body = b"riff-bytes"

        result = upload_asset(
            client,
            content=base64.b64encode(body).decode("ascii"),
            asset_name="cast/priya-voice.wav",
        )

        assert seen["body"] == body
        assert "asset_name=cast" in seen["url"].replace("%2F", "/")
        assert "filename=cast" in seen["url"].replace("%2F", "/")
        assert result["reference"] == "asset:uploads/deadbeef.png"
        assert result["uploaded"] == "cast/priya-voice.wav"
        assert result["size"] == len(body)

    def test_both_file_path_and_content_is_refused(self, tmp_path):
        source = tmp_path / "iris.png"
        source.write_bytes(b"x")
        client, _seen = recording()
        with pytest.raises(DwApiError, match="exactly one of file_path or content"):
            upload_asset(client, str(source), content="AAAA", asset_name="x.png")

    def test_neither_file_path_nor_content_is_refused(self):
        client, _seen = recording()
        with pytest.raises(DwApiError, match="exactly one of file_path or content"):
            upload_asset(client)

    def test_content_without_asset_name_is_refused(self):
        client, _seen = recording()
        with pytest.raises(DwApiError, match="content requires asset_name"):
            upload_asset(client, content=base64.b64encode(b"x").decode("ascii"))

    def test_a_kind_the_library_does_not_take_is_refused(self):
        client, seen = recording()
        with pytest.raises(DwApiError, match="not a kind"):
            upload_asset(
                client,
                content=base64.b64encode(b"x").decode("ascii"),
                asset_name="script.py",
            )
        assert "body" not in seen

    def test_invalid_base64_says_so(self):
        client, seen = recording()
        with pytest.raises(DwApiError, match="could not be decoded as base64"):
            upload_asset(client, content="not-valid-base64!!", asset_name="x.png")
        assert "body" not in seen

    def test_content_over_the_inline_limit_is_refused(self):
        client, seen = recording()
        oversized = base64.b64encode(b"0" * (MAX_INLINE_UPLOAD_BYTES + 1)).decode(
            "ascii"
        )
        with pytest.raises(DwApiError, match="inline upload"):
            upload_asset(client, content=oversized, asset_name="x.png")
        assert "body" not in seen

    def test_shared_is_passed_through(self):
        client, seen = recording()
        upload_asset(
            client,
            content=base64.b64encode(b"x").decode("ascii"),
            asset_name="cast/priya.png",
            shared=True,
        )
        assert "shared=true" in seen["url"]


class TestUploadContainmentOverAMountedEndpoint:
    """Served by dw.serve, 'local file' means the operator's box - so an
    unconfined file_path is an arbitrary read of the server's filesystem plus
    a path-existence oracle for it (#138), the mirror of download_output's
    write direction (#113). A stdio dw-mcp keeps reading the user's own disk."""

    def _mounted_client(self, workspace, tmp_path):
        def handler(request):
            if request.url.path == "/api/server":
                return httpx.Response(
                    200,
                    json={
                        "directories": {
                            "workspace": str(workspace),
                            "workflows": str(workspace / "workflows"),
                            "assets": str(workspace / "assets"),
                            "outputs": str(workspace / "outputs"),
                            "prompts": None,
                        }
                    },
                )
            return httpx.Response(
                201,
                json={
                    "reference": "asset:uploads/deadbeef.png",
                    "url": "/inputs/uploads/deadbeef.png",
                },
            )

        client = client_over(handler)
        client.mounted = True
        return client

    def test_a_file_outside_the_roots_is_refused(self, tmp_path):
        workspace = tmp_path / "workspace"
        (workspace / "assets").mkdir(parents=True)
        outside = tmp_path / "elsewhere" / "secret.png"
        outside.parent.mkdir()
        outside.write_bytes(b"png-bytes")

        client = self._mounted_client(workspace, tmp_path)
        with pytest.raises(DwApiError, match="Refusing to read"):
            upload_asset(client, str(outside))

    def test_the_refusal_does_not_say_whether_the_file_exists(self, tmp_path):
        """The containment check comes before the existence and extension
        checks, so the tool cannot be used to probe the box for paths."""
        workspace = tmp_path / "workspace"
        (workspace / "assets").mkdir(parents=True)
        present = tmp_path / "elsewhere" / "there.png"
        present.parent.mkdir()
        present.write_bytes(b"png-bytes")
        client = self._mounted_client(workspace, tmp_path)

        with pytest.raises(DwApiError) as there:
            upload_asset(client, str(present))
        with pytest.raises(DwApiError) as not_there:
            upload_asset(client, str(tmp_path / "elsewhere" / "missing.png"))
        assert str(there.value).replace("there.png", "X") == str(
            not_there.value
        ).replace("missing.png", "X")

        # and a non-media extension outside the roots reads the same way, so
        # the allowlist is not an oracle either
        with pytest.raises(DwApiError, match="Refusing to read"):
            upload_asset(client, "/etc/hostname")

    def test_a_file_inside_the_roots_still_uploads(self, tmp_path):
        workspace = tmp_path / "workspace"
        (workspace / "assets").mkdir(parents=True)
        source = workspace / "assets" / "iris.png"
        source.write_bytes(b"png-bytes")

        client = self._mounted_client(workspace, tmp_path)
        result = upload_asset(client, str(source))
        assert result["reference"] == "asset:uploads/deadbeef.png"

    def test_a_stdio_client_is_unconfined(self, tmp_path):
        """There 'local' is genuinely the caller's own machine."""
        source = tmp_path / "elsewhere" / "iris.png"
        source.parent.mkdir()
        source.write_bytes(b"png-bytes")
        client, _seen = recording()

        assert upload_asset(client, str(source))["uploaded"] == "iris.png"

    def _mounted_client_with_shared_library(self, workspace, common, tmp_path):
        """A workspace root plus the shared 'common/assets' library every
        workspace's asset search path includes (#448) - only visible via
        /api/assets's 'libraries', not /api/server's 'directories'."""

        def handler(request):
            if request.url.path == "/api/server":
                return httpx.Response(
                    200,
                    json={
                        "directories": {
                            "workspace": str(workspace),
                            "workflows": str(workspace / "workflows"),
                            "assets": str(workspace / "assets"),
                            "outputs": str(workspace / "outputs"),
                            "prompts": None,
                        }
                    },
                )
            if request.url.path == "/api/assets":
                return httpx.Response(
                    200,
                    json={
                        "asset_dir": str(workspace / "assets"),
                        "assets": [],
                        "folders": [],
                        "libraries": [
                            {
                                "origin": "workspace",
                                "dir": str(workspace / "assets"),
                                "writable": True,
                            },
                            {
                                "origin": "common",
                                "dir": str(common),
                                "writable": True,
                            },
                            {
                                "origin": "examples",
                                "dir": str(tmp_path / "examples" / "assets"),
                                "writable": False,
                            },
                        ],
                        "shadowed": [],
                    },
                )
            return httpx.Response(
                201,
                json={
                    "reference": "asset:uploads/deadbeef.png",
                    "url": "/inputs/uploads/deadbeef.png",
                },
            )

        client = client_over(handler)
        client.mounted = True
        return client

    def test_a_file_in_the_shared_common_library_is_a_legal_source(self, tmp_path):
        """SE-F027: the shared asset library is writable and on every
        workspace's asset search path, so it must be as legal a file_path
        source as the four directories /api/server names - not just the
        workspace's own assets/ folder (#448)."""
        workspace = tmp_path / "workspace"
        (workspace / "assets").mkdir(parents=True)
        common = tmp_path / "common" / "assets"
        common.mkdir(parents=True)
        source = common / "qa-cast" / "ep10-bed.wav"
        source.parent.mkdir()
        source.write_bytes(b"wav-bytes")

        client = self._mounted_client_with_shared_library(workspace, common, tmp_path)
        result = upload_asset(client, str(source))
        assert result["reference"] == "asset:uploads/deadbeef.png"

    def test_a_read_only_examples_library_is_still_refused(self, tmp_path):
        """A writable library is a legal source; a read-only examples one is
        not - only #448's shared/writable gap is closed, not confinement
        itself."""
        workspace = tmp_path / "workspace"
        (workspace / "assets").mkdir(parents=True)
        common = tmp_path / "common" / "assets"
        common.mkdir(parents=True)
        examples = tmp_path / "examples" / "assets"
        examples.mkdir(parents=True)
        source = examples / "sample.png"
        source.write_bytes(b"png-bytes")

        client = self._mounted_client_with_shared_library(workspace, common, tmp_path)
        with pytest.raises(DwApiError, match="Refusing to read"):
            upload_asset(client, str(source))

    def test_traversal_out_of_the_shared_library_is_still_refused(self, tmp_path):
        """#448 widens the legal roots; it must not widen containment - '..'
        out of the shared library is still refused."""
        workspace = tmp_path / "workspace"
        (workspace / "assets").mkdir(parents=True)
        common = tmp_path / "common" / "assets"
        common.mkdir(parents=True)
        outside = tmp_path / "elsewhere" / "secret.png"
        outside.parent.mkdir()
        outside.write_bytes(b"png-bytes")
        traversal = str(common) + "/../../elsewhere/secret.png"

        client = self._mounted_client_with_shared_library(workspace, common, tmp_path)
        with pytest.raises(DwApiError, match="Refusing to read"):
            upload_asset(client, traversal)


class TestUploadRouteInTheRefusals:
    """Both refusals end in a curl command for POST /api/uploads that the
    caller can run from its own machine (#481): a mounted endpoint reaches
    its server at 127.0.0.1, which on a remote caller's machine is that
    machine, so a loopback origin is printed as <host>; the workspace is the
    one the call would have uploaded to; and the ceiling is the route's real
    one, not 'no size cap'."""

    def _client(self, base_url, workspace=None, roots=None):
        def handler(request):
            if request.url.path == "/api/server":
                return httpx.Response(200, json={"directories": roots or {}})
            if request.url.path == "/api/assets":
                return httpx.Response(200, json={"assets": []})
            raise AssertionError(f"nothing should be sent: {request.url}")

        client = DwClient(
            base_url=base_url,
            workspace=workspace,
            transport=httpx.MockTransport(handler),
        )
        client.mounted = roots is not None
        return client

    def _file_path_refusal(self, tmp_path, base_url, **kwargs):
        workspace = tmp_path / "workspace"
        (workspace / "assets").mkdir(parents=True)
        roots = {"workspace": str(workspace), "assets": str(workspace / "assets")}
        client = self._client(
            base_url, workspace=kwargs.pop("session", None), roots=roots
        )
        with pytest.raises(DwApiError) as refused:
            upload_asset(
                client,
                file_path="/Users/don/Pictures/portrait.jpg",
                asset_name="qa-cast/portrait.jpg",
                **kwargs,
            )
        return str(refused.value)

    def test_a_loopback_origin_is_printed_as_host(self, tmp_path):
        # what dw/server/mcp_mount.py's client_base_url gives a mounted
        # endpoint on a loopback or wildcard bind
        text = self._file_path_refusal(tmp_path, "http://127.0.0.1:8765")
        assert "127.0.0.1:8765" not in text
        assert '"http://<host>:8765/api/uploads?filename=<name>' in text
        assert "<host> is this server's address as your machine reaches it" in text

    def test_a_reachable_origin_is_printed_verbatim(self, tmp_path):
        text = self._file_path_refusal(tmp_path, "http://100.64.0.7:8765")
        assert '"http://100.64.0.7:8765/api/uploads?' in text
        assert "<host>" not in text

    def test_the_command_carries_token_workspace_and_return(self, tmp_path):
        text = self._file_path_refusal(
            tmp_path, "http://127.0.0.1:8765", workspace="qa-series"
        )
        assert "Authorization: Bearer $DW_API_TOKEN" in text
        assert "--data-binary @<file>" in text
        assert '&asset_name=<folder/name>&workspace=qa-series"' in text
        assert "201 with 'reference', the 'asset:' reference" in text
        assert f"up to {MAX_UPLOAD_BYTES // (1024 * 1024)}MB" in text
        assert "no size cap" not in text

    def test_the_session_workspace_fills_the_command(self, tmp_path):
        text = self._file_path_refusal(
            tmp_path, "http://127.0.0.1:8765", session="episode-4"
        )
        assert '&workspace=episode-4"' in text

    def test_the_default_workspace_is_named_rather_than_left_blank(self, tmp_path):
        text = self._file_path_refusal(tmp_path, "http://127.0.0.1:8765")
        assert '&workspace=default"' in text
        assert "<ws>" not in text

    def test_the_inline_limit_refusal_carries_the_same_command(self):
        client = self._client("http://127.0.0.1:8765", workspace="qa-series")
        oversized = base64.b64encode(b"0" * (MAX_INLINE_UPLOAD_BYTES + 1)).decode(
            "ascii"
        )
        with pytest.raises(DwApiError) as refused:
            upload_asset(client, content=oversized, asset_name="cut/final.mp4")
        text = str(refused.value)
        assert '"http://<host>:8765/api/uploads?filename=<name>' in text
        assert '&workspace=qa-series"' in text
        assert "201 with 'reference', the 'asset:' reference" in text
        assert "no size cap" not in text
