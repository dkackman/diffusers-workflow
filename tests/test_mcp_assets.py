"""Getting an input file to the server: the asset library over MCP.

An agent that can only name assets already on the box can author workflows
it cannot supply inputs for - these are the two tools that close that.
"""

import json

import httpx
import pytest

from dw_mcp.assets import MAX_UPLOAD_BYTES, keep_output, list_assets, upload_asset
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
                "path": "asset:uploads/deadbeef.png",
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

    def test_a_server_with_no_library_still_reports_what_it_got(self, tmp_path):
        """Older servers answer with an absolute path rather than a
        reference - report whatever came back rather than inventing one."""
        source = tmp_path / "iris.png"
        source.write_bytes(b"png")

        def handler(request):
            return httpx.Response(
                201,
                json={"path": "/srv/outputs/uploads/x.png", "url": "/outputs/x.png"},
            )

        result = upload_asset(client_over(handler), str(source))
        assert result["reference"] == "/srv/outputs/uploads/x.png"


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
