"""Returning a generated image to the agent: downscaled enough to be worth
a context window, honest about what it refuses."""

import base64
import io
import os

import httpx
import numpy as np
import pytest
from PIL import Image

import dw_mcp.media as media
from dw_mcp.client import DwApiError, DwClient
from dw_mcp.media import (
    MAX_RETURNED_BYTES,
    download_output,
    get_output_audio,
    get_output_frames,
    get_output_image,
)


def noise_png_bytes(width, height, seed=0):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def png_bytes(width, height, color=(120, 30, 200)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def jpeg_bytes(width, height):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 10, 10)).save(buffer, format="JPEG")
    return buffer.getvalue()


def serving(content, content_type):
    def handler(request):
        return httpx.Response(
            200, content=content, headers={"content-type": content_type}
        )

    return DwClient(transport=httpx.MockTransport(handler))


def serving_with_headers(content, content_type, headers):
    def handler(request):
        return httpx.Response(
            200, content=content, headers={"content-type": content_type, **headers}
        )

    return DwClient(transport=httpx.MockTransport(handler))


def decoded(result):
    return Image.open(io.BytesIO(base64.b64decode(result["data"])))


def image_route(body=b"img", content_type="image/png", headers=None, status=200):
    """A server whose /api/gallery/<name>/image answers `body` with the
    sizing headers the route sends; records each request."""
    seen = []
    sizing = {"x-dw-original-size": "2048,1024", "x-dw-returned-size": "512,256"}

    def handler(request):
        seen.append(request)
        if status >= 400:
            return httpx.Response(status, json={"detail": body})
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": content_type, **sizing, **(headers or {})},
        )

    return DwClient(transport=httpx.MockTransport(handler)), seen


class TestGetOutputImage:
    """The server crops, fits and budgets (GET /api/gallery/<name>/image);
    the tool forwards the request and reshapes the answer."""

    def test_it_asks_the_image_route_with_the_size_and_budget(self):
        client, seen = image_route()

        get_output_image(client, "w/run/big.png", max_dimension=512)

        assert seen[0].url.path == "/api/gallery/w/run/big.png/image"
        params = dict(seen[0].url.params)
        assert params["max_dimension"] == "512"
        assert params["max_bytes"] == str(MAX_RETURNED_BYTES)
        assert "crop" not in params

    def test_it_forwards_a_crop(self):
        client, seen = image_route(headers={"x-dw-crop": "10,20,30,40"})

        result = get_output_image(client, "big.png", crop=[10, 20, 30, 40])

        assert dict(seen[0].url.params)["crop"] == "10,20,30,40"
        assert result["crop"] == [10, 20, 30, 40]

    def test_it_reshapes_the_answer(self):
        client, _ = image_route(body=b"jpeg-bytes", content_type="image/jpeg")

        result = get_output_image(client, "photo.jpg")

        assert base64.b64decode(result["data"]) == b"jpeg-bytes"
        assert result["mime_type"] == "image/jpeg"
        assert result["original_size"] == [2048, 1024]
        assert result["returned_size"] == [512, 256]
        assert result["crop"] is None
        assert result["bytes"] == len(b"jpeg-bytes")

    def test_the_servers_refusal_reaches_the_caller(self):
        client, _ = image_route(body="clip.mp4 is not an image", status=404)

        with pytest.raises(DwApiError, match="not an image"):
            get_output_image(client, "clip.mp4")


class TestGetOutputAudio:
    def test_a_clip_under_budget_comes_back_as_base64(self):
        body = b"riff-audio-bytes"
        client = serving(body, "audio/wav")

        result = get_output_audio(client, "voice.wav")

        assert base64.b64decode(result["data"]) == body
        assert result["mime_type"] == "audio/wav"
        assert result["bytes"] == len(body)
        assert result["name"] == "voice.wav"

    def test_an_image_output_is_refused_by_content_type(self):
        client = serving(png_bytes(10, 10), "image/png")

        with pytest.raises(DwApiError) as caught:
            get_output_audio(client, "still.png")

        assert "image/png" in str(caught.value)

    def test_a_response_with_no_content_type_is_refused(self):
        def handler(request):
            return httpx.Response(200, content=b"???")

        client = DwClient(transport=httpx.MockTransport(handler))

        with pytest.raises(DwApiError):
            get_output_audio(client, "mystery")

    def test_a_clip_over_the_byte_ceiling_is_refused(self, monkeypatch):
        monkeypatch.setattr(media, "MAX_RETURNED_BYTES", 16)
        client = serving(b"x" * 32, "audio/wav")

        with pytest.raises(DwApiError, match="byte limit"):
            get_output_audio(client, "long.wav")

    def test_the_budget_is_checked_against_the_base64_size_not_the_raw_bytes(
        self, monkeypatch
    ):
        # 12 raw bytes base64-encode to 16 - set the cap between the two so a
        # raw-bytes-only comparison would wrongly accept it.
        monkeypatch.setattr(media, "MAX_RETURNED_BYTES", 13)
        client = serving(b"x" * 12, "audio/wav")

        with pytest.raises(DwApiError, match="byte limit"):
            get_output_audio(client, "clip.wav")


def test_audio_is_fetched_from_the_gallery_audio_route():
    seen = []

    def handler(request):
        seen.append((request.url.path, dict(request.url.params)))
        return httpx.Response(
            200,
            content=b"riff",
            headers={"content-type": "audio/wav", "x-dw-duration": "2.0"},
        )

    client = DwClient(transport=httpx.MockTransport(handler))
    result = get_output_audio(client, "run/shot.mp4")

    # request.url.path decodes percent-escapes back for display (see the
    # comment in test_the_name_is_url_quoted_in_the_request above); the
    # encoding itself is api_path's job and is covered by
    # tests/test_mcp_client.py.
    assert seen == [("/api/gallery/run/shot.mp4/audio", {})]
    assert result["mime_type"] == "audio/wav"
    assert result["duration_seconds"] == 2.0
    assert result["excerpt"] is None


def test_an_excerpt_is_asked_for_and_reported():
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(
            200,
            content=b"riff",
            headers={
                "content-type": "audio/wav",
                "x-dw-duration": "240.0",
                "x-dw-excerpt-start": "10.0",
                "x-dw-excerpt-duration": "2.0",
            },
        )

    client = DwClient(transport=httpx.MockTransport(handler))
    result = get_output_audio(client, "cut.mp4", start=10.0, duration=2.0)

    assert seen == [{"start": "10.0", "duration": "2.0"}]
    assert result["excerpt"] == {"start": 10.0, "duration": 2.0, "of": 240.0}


def test_a_whole_track_over_budget_is_refused_and_told_to_excerpt():
    big = b"\0" * (MAX_RETURNED_BYTES * 3 // 4 + 1024)
    client = serving_with_headers(big, "audio/wav", {"x-dw-duration": "240.0"})

    with pytest.raises(DwApiError) as caught:
        get_output_audio(client, "cut.mp4")

    assert "start" in str(caught.value) and "duration" in str(caught.value)


def test_a_non_audio_answer_is_refused():
    client = serving(b"{}", "application/json")

    with pytest.raises(DwApiError, match="not audio"):
        get_output_audio(client, "thing.json")


def test_the_name_is_url_quoted_in_the_request():
    # "#" starts a URL fragment when left unescaped - an unquoted name would
    # arrive at the server truncated. The escaping itself is checked on
    # url.raw_path, the bytes actually placed on the wire.
    client, seen = image_route()

    get_output_image(client, "a b#1.png")

    assert "%23" in seen[0].url.raw_path.decode("ascii")
    assert seen[0].url.path == "/api/gallery/a b#1.png/image"


def test_a_dot_segment_name_survives_intact_onto_the_wire():
    """httpx normalizes `..` out of a request path client-side, which would
    skip the server's own confinement. Quoting the separator keeps the
    literal bytes on the wire so it is the server that refuses the name."""
    client, seen = image_route()

    get_output_image(client, "../api/models")

    assert seen[0].url.raw_path.startswith(b"/api/gallery/..%2Fapi%2Fmodels")


# ------------------------------------------------------------- text output


def test_a_text_output_comes_back_as_text():
    client = serving(b"a duke on a velvet sofa", "text/plain; charset=utf-8")

    result = media.get_output_text(client, "enhanced.txt")

    assert result["text"] == "a duke on a velvet sofa"
    assert result["name"] == "enhanced.txt"
    assert result["truncated"] is False


def test_a_text_output_reports_its_length():
    client = serving(b"four", "text/plain")

    assert media.get_output_text(client, "e.txt")["characters"] == 4


def test_a_json_output_is_text_too():
    """The manifest can hold a .json result, and it is as readable as a
    .txt one - refusing it would send the agent to the raw HTTP API."""
    client = serving(b'{"a": 1}', "application/json")

    assert media.get_output_text(client, "e.json")["text"] == '{"a": 1}'


def test_a_long_text_output_is_truncated_and_says_so():
    client = serving(b"x" * 500, "text/plain")

    result = media.get_output_text(client, "long.txt", max_characters=100)

    assert len(result["text"]) == 100
    assert result["truncated"] is True
    assert result["characters"] == 500


def test_an_image_is_refused_by_the_text_tool():
    """Rejection happens on the content type, before the body is read - a
    video output could be gigabytes."""
    client = serving(png_bytes(8, 8), "image/png")

    with pytest.raises(DwApiError, match="not text"):
        media.get_output_text(client, "out.png")


def test_the_text_tool_names_the_tool_that_can_read_an_image():
    client = serving(png_bytes(8, 8), "image/png")

    with pytest.raises(DwApiError, match="get_output_image"):
        media.get_output_text(client, "out.png")


def test_undecodable_bytes_do_not_crash_the_tool():
    """A file the server labels text but that is not valid UTF-8 should read
    as damaged output, not as a tool that blew up."""
    client = serving(b"\xff\xfe\x00bad", "text/plain")

    result = media.get_output_text(client, "odd.txt")

    assert isinstance(result["text"], str)


# ----------------------------------------------------------- output removal


def test_delete_output_calls_delete_on_the_gallery_route():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"name": "out.png", "deleted": True})

    client = DwClient(transport=httpx.MockTransport(handler))

    assert media.delete_output(client, "out.png")["deleted"] is True
    assert seen == [("DELETE", "/api/gallery/out.png")]


def deleting_by_job(status=200, body=None):
    """DELETE /api/jobs/job-1/run answers as the server does."""
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, dict(request.url.params)))
        if request.url.path != "/api/jobs/job-1/run":
            return httpx.Response(404, json={"detail": "Unknown job"})
        return httpx.Response(
            status,
            json=body
            or {
                "job_id": "job-1",
                "run_dir": "ltx2/Gyre",
                "deleted": True,
                "run_swept": "Gyre",
            },
        )

    return DwClient(transport=httpx.MockTransport(handler)), seen


def test_delete_output_by_job_id_is_one_call_to_the_job_run_route():
    """The server reads the job's run directory and its own root; the
    client derives neither."""
    client, seen = deleting_by_job()
    client.workspace = "elsewhere"

    result = media.delete_output(client, job_id="job-1")

    assert [entry[:2] for entry in seen] == [("DELETE", "/api/jobs/job-1/run")]
    assert result["run_dir"] == "ltx2/Gyre"
    assert result["job_id"] == "job-1"


def test_delete_output_refuses_neither_name_nor_job_id():
    client, seen = deleting_by_job()

    with pytest.raises(DwApiError, match="exactly one"):
        media.delete_output(client)

    assert seen == []


def test_delete_output_refuses_both_name_and_job_id():
    client, seen = deleting_by_job()

    with pytest.raises(DwApiError, match="exactly one"):
        media.delete_output(client, "out.png", job_id="job-1")

    assert seen == []


def test_delete_output_by_job_id_surfaces_an_unknown_job():
    client, _seen = deleting_by_job()

    with pytest.raises(DwApiError, match="Unknown job"):
        media.delete_output(client, job_id="ghost")


def test_delete_output_by_job_id_surfaces_the_servers_refusal():
    """A job with no run directory, or one still running, is the server's
    to refuse; its detail reaches the caller as written."""
    client, _seen = deleting_by_job(
        404, {"detail": "Job job-1 (failed) has no run directory to delete"}
    )

    with pytest.raises(DwApiError, match="no run directory"):
        media.delete_output(client, job_id="job-1")


# --------------------------------------------------------- output download


def test_download_output_writes_bytes_to_explicit_file_path(tmp_path):
    client = serving(png_bytes(64, 48), "image/png")
    destination = tmp_path / "saved.png"

    result = download_output(client, "run-step.0-0.0.png", destination=str(destination))

    assert destination.read_bytes() == png_bytes(64, 48)
    assert result == {
        "name": "run-step.0-0.0.png",
        "saved_to": str(destination),
        "content_type": "image/png",
        "bytes": len(png_bytes(64, 48)),
    }


def test_download_output_into_a_directory_uses_the_output_basename(tmp_path):
    client = serving(png_bytes(10, 10), "image/png")

    result = download_output(
        client, "sub/run-step.0-0.0.png", destination=str(tmp_path)
    )

    saved = tmp_path / "run-step.0-0.0.png"
    assert saved.read_bytes() == png_bytes(10, 10)
    assert result["saved_to"] == str(saved)


def test_download_output_with_no_destination_saves_to_current_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    client = serving(png_bytes(10, 10), "image/png")

    result = download_output(client, "run-step.0-0.0.png")

    assert (tmp_path / "run-step.0-0.0.png").read_bytes() == png_bytes(10, 10)
    assert result["saved_to"] == str(tmp_path / "run-step.0-0.0.png")


def test_download_output_creates_missing_parent_directories(tmp_path):
    client = serving(png_bytes(10, 10), "image/png")
    destination = tmp_path / "renders" / "today" / "spoons.png"

    download_output(client, "spoons.png", destination=str(destination))

    assert destination.read_bytes() == png_bytes(10, 10)


def test_download_output_expands_user_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    client = serving(png_bytes(5, 5), "image/png")

    result = download_output(client, "spoons.png", destination="~/spoons.png")

    assert result["saved_to"] == str(tmp_path / "spoons.png")


def test_download_output_refuses_to_overwrite_an_existing_file_by_default(tmp_path):
    destination = tmp_path / "existing.png"
    destination.write_bytes(b"already here")
    client = serving(png_bytes(10, 10), "image/png")

    with pytest.raises(DwApiError, match=str(destination)):
        download_output(client, "new.png", destination=str(destination))

    assert destination.read_bytes() == b"already here"


def test_download_output_reports_an_unwritable_destination_as_the_servers_disk(
    tmp_path,
):
    # A cold agent over dw.serve --mcp passed a path on its own machine; on
    # the server that parent is a file, so makedirs raises. The message has
    # to say the write happened server-side and name the ways to see the
    # file from the client, instead of an anonymous tool error (drill
    # 2026-09-08).
    blocker = tmp_path / "private"
    blocker.write_bytes(b"not a directory")
    destination = blocker / "scratch" / "clip.mp4"
    client = serving(png_bytes(10, 10), "image/png")

    with pytest.raises(DwApiError) as excinfo:
        download_output(client, "clip.mp4", destination=str(destination))

    message = str(excinfo.value)
    assert str(destination) in message
    assert "machine running the MCP server" in message
    assert "list_gallery" in message and "keep_output" in message
    assert not destination.exists()


def test_download_output_overwrite_true_replaces_an_existing_file(tmp_path):
    destination = tmp_path / "existing.png"
    destination.write_bytes(b"already here")
    client = serving(png_bytes(10, 10), "image/png")

    result = download_output(
        client, "new.png", destination=str(destination), overwrite=True
    )

    assert destination.read_bytes() == png_bytes(10, 10)
    assert result["saved_to"] == str(destination)


def test_download_output_rejects_a_dot_dot_segment_in_destination(tmp_path):
    client = serving(png_bytes(10, 10), "image/png")
    escaping = str(tmp_path / ".." / "escaped.png")

    with pytest.raises(DwApiError, match=r"\.\."):
        download_output(client, "new.png", destination=escaping)

    assert not os.path.exists(os.path.join(str(tmp_path), "..", "escaped.png"))


def test_download_output_streams_the_body_in_chunks(tmp_path, monkeypatch):
    """The tool exists for files get_output_image can't return - large
    videos. Buffering the whole body defeats the point, so the write has to
    go through iter_bytes in chunks rather than one `.content` blob."""

    class ChunkedStream(httpx.SyncByteStream):
        def __init__(self, chunks):
            self.chunks = chunks

        def __iter__(self):
            yield from self.chunks

        def close(self):
            pass

    chunks = [b"a" * 1000, b"b" * 1000, b"c" * 1000]

    def handler(request):
        return httpx.Response(
            200, headers={"content-type": "video/mp4"}, stream=ChunkedStream(chunks)
        )

    client = DwClient(transport=httpx.MockTransport(handler))
    destination = tmp_path / "chunked.mp4"

    writes = []
    real_fdopen = os.fdopen

    def tracking_fdopen(fd, mode="r", *args, **kwargs):
        # Chunks land in a temp file next to `destination`, opened via
        # os.fdopen on a descriptor from tempfile.mkstemp - not via
        # builtins.open - so that is what has to be intercepted to observe
        # each write.
        file = real_fdopen(fd, mode, *args, **kwargs)
        if "b" in mode:
            original_write = file.write

            def tracked_write(data):
                writes.append(len(data))
                return original_write(data)

            file.write = tracked_write
        return file

    monkeypatch.setattr("dw_mcp.client.os.fdopen", tracking_fdopen)

    result = download_output(client, "big.bin", destination=str(destination))

    assert destination.read_bytes() == b"".join(chunks)
    assert len(writes) >= 2
    assert result["bytes"] == len(b"".join(chunks))
    assert result["content_type"] == "video/mp4"


def test_download_output_leaves_no_file_when_the_stream_breaks_mid_body(tmp_path):
    """A connection drop after some chunks have already arrived (the normal
    failure mode for a large video) must not leave a torn partial file -
    that file would then "exist" for a later overwrite=False call and
    silently mask the failure."""

    class BreakingStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"a" * 1000
            raise httpx.ReadError("connection dropped")

        def close(self):
            pass

    def handler(request):
        return httpx.Response(
            200, headers={"content-type": "video/mp4"}, stream=BreakingStream()
        )

    client = DwClient(transport=httpx.MockTransport(handler))
    destination = tmp_path / "broken.mp4"

    with pytest.raises(DwApiError):
        download_output(client, "big.bin", destination=str(destination))

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------- a mounted server's writes
#
# SE-F016 (#113): over a `dw.serve --mcp` endpoint the tool runs on the GPU
# box, so `destination` is a path on the operator's machine rather than on the
# calling agent's. The '..' check it had could not see that - an absolute or
# '~' path needs no '..' to reach anywhere the server process can write.


def mounted(content, content_type, workspace_root):
    """A client shaped like the one dw.serve builds for its own /mcp."""

    def handler(request):
        if request.url.path == "/api/server":
            return httpx.Response(
                200,
                json={"directories": {"workspace": str(workspace_root)}},
                headers={"content-type": "application/json"},
            )
        return httpx.Response(
            200, content=content, headers={"content-type": content_type}
        )

    client = DwClient(transport=httpx.MockTransport(handler))
    client.mounted = True
    return client


def test_a_mounted_server_refuses_an_absolute_destination_outside_the_workspace(
    tmp_path,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = mounted(png_bytes(4, 4), "image/png", workspace)
    outside = tmp_path / "elsewhere" / "probe.jpg"

    with pytest.raises(DwApiError) as refusal:
        download_output(client, "run/probe.jpg", destination=str(outside))

    assert "confined to the workspace" in str(refusal.value)
    assert not outside.exists()


def test_a_mounted_server_refuses_a_home_relative_destination(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    client = mounted(png_bytes(4, 4), "image/png", workspace)

    with pytest.raises(DwApiError):
        download_output(client, "run/probe.jpg", destination="~/probe.jpg")

    assert not (home / "probe.jpg").exists()


def test_a_mounted_server_refuses_an_overwrite_outside_the_workspace(tmp_path):
    """The escalation SE-F016 flagged but would not probe: the same absolute
    destination with overwrite=true is an arbitrary file overwrite."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    victim = tmp_path / "bashrc"
    victim.write_text("mine")
    client = mounted(png_bytes(4, 4), "image/png", workspace)

    with pytest.raises(DwApiError):
        download_output(
            client, "run/probe.jpg", destination=str(victim), overwrite=True
        )

    assert victim.read_text() == "mine"


def test_a_mounted_server_confines_a_per_call_workspace_override(tmp_path):
    """#389: download_output's own `workspace` argument overrides the
    session's pin for the download itself (stream_to_file already forwarded
    it), but _remote_root asked /api/server with no workspace at all, so the
    confinement root stayed the session's - a relative destination under a
    workspace= override landed in the wrong tree with no error."""
    default_ws = tmp_path / "default"
    default_ws.mkdir()
    other_ws = tmp_path / "other"
    other_ws.mkdir()

    def handler(request):
        if request.url.path == "/api/server":
            requested = httpx.QueryParams(request.url.query.decode())
            root = other_ws if requested.get("workspace") == "other" else default_ws
            return httpx.Response(
                200,
                json={"directories": {"workspace": str(root)}},
                headers={"content-type": "application/json"},
            )
        return httpx.Response(
            200, content=png_bytes(4, 4), headers={"content-type": "image/png"}
        )

    client = DwClient(transport=httpx.MockTransport(handler))
    client.mounted = True

    result = download_output(
        client, "run/probe.jpg", destination="kept/probe.jpg", workspace="other"
    )

    assert result["saved_to"] == str(other_ws / "kept" / "probe.jpg")
    assert (other_ws / "kept" / "probe.jpg").read_bytes() == png_bytes(4, 4)
    assert not (default_ws / "kept" / "probe.jpg").exists()


def test_a_mounted_server_writes_a_relative_destination_into_its_workspace(tmp_path):
    """And the default keeps working: a relative destination is joined onto
    the workspace rather than onto whatever the server's cwd happens to be."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = mounted(png_bytes(4, 4), "image/png", workspace)

    result = download_output(client, "run/probe.jpg", destination="kept/probe.jpg")

    assert result["saved_to"] == str(workspace / "kept" / "probe.jpg")
    assert (workspace / "kept" / "probe.jpg").read_bytes() == png_bytes(4, 4)


def test_a_mounted_server_refuses_an_omitted_destination(tmp_path):
    """#353: defaulting into the workspace root stranded a file nothing could
    later find or delete - so a mounted endpoint now requires an explicit
    destination rather than picking one."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = mounted(png_bytes(4, 4), "image/png", workspace)

    with pytest.raises(DwApiError) as refusal:
        download_output(client, "run/probe.jpg")

    assert "destination is required" in str(refusal.value)
    assert list(workspace.iterdir()) == []


def test_a_stdio_client_still_writes_wherever_the_user_can(tmp_path):
    """Unmounted, 'local disk' is genuinely the caller's own machine."""
    client = serving(png_bytes(4, 4), "image/png")
    destination = tmp_path / "anywhere" / "probe.jpg"

    download_output(client, "run/probe.jpg", destination=str(destination))

    assert destination.read_bytes() == png_bytes(4, 4)


def tile_json(width, height, label="00:00.0 (frame 0)", frame=0, seconds=0.0):
    return {
        "label": label,
        "frame": frame,
        "seconds": seconds,
        "data": base64.b64encode(png_bytes(width, height)).decode("ascii"),
        "mime_type": "image/png",
        "width": width,
        "height": height,
    }


def frames_server(tiles, seen=None, crop=None, status=200, detail=None):
    def handler(request):
        if seen is not None:
            seen.append((request.url.path, list(request.url.params.multi_items())))
        if status != 200:
            return httpx.Response(status, json={"detail": detail})
        return httpx.Response(
            200,
            json={
                "name": "x.mp4",
                "frame_count": 24,
                "fps": 6.0,
                "width": 64,
                "height": 32,
                "tiles": tiles,
                "crop": crop,
            },
        )

    return DwClient(transport=httpx.MockTransport(handler))


def test_frames_are_asked_for_by_moment_and_come_back_labelled():
    seen = []
    client = frames_server(
        [tile_json(64, 32), tile_json(64, 32, "00:02.0 (frame 12)", 12, 2.0)], seen
    )

    result = get_output_frames(client, "run/x.mp4", at=[0.0, "frame:12"])

    # request.url.path decodes percent-escapes back for display (see the
    # comment on test_audio_is_fetched_from_the_gallery_audio_route above);
    # the encoding itself is api_path's job, covered by tests/test_mcp_client.py.
    assert seen[0][0] == "/api/gallery/run/x.mp4/frames"
    assert dict(seen[0][1])["at"] == "0.0,frame:12"
    assert [t["label"] for t in result["tiles"]] == [
        "00:00.0 (frame 0)",
        "00:02.0 (frame 12)",
    ]
    assert result["downscaled_to"] is None


def test_seams_send_boundaries_and_names():
    seen = []
    client = frames_server([tile_json(128, 32, "seam 1: a | b", 8, 1.33)], seen)

    get_output_frames(
        client, "cut.mp4", seams=[1], boundaries=[8, 16], names=["a", "b", "c"]
    )

    params = dict(seen[0][1])
    assert params["seams"] == "1"
    assert params["boundaries"] == "8,16"
    assert params["names"] == "a,b,c"


def test_two_selectors_are_both_sent_and_the_servers_refusal_reaches_the_caller():
    """The server owns the one-selector rule; the tool sends what it was
    given rather than quietly dropping the second."""
    seen = []
    detail = "Pass exactly one of `at`, `count` or `seams` - got at, count"
    client = frames_server([], seen, status=400, detail=detail)

    with pytest.raises(DwApiError, match="one of"):
        get_output_frames(client, "x.mp4", at=[0.0], count=4)

    sent = dict(seen[0][1])
    assert "at" in sent and "count" in sent


def test_names_without_seams_is_refused_before_any_request():
    seen = []
    client = frames_server([], seen)

    with pytest.raises(DwApiError, match="only appl.* alongside `seams`"):
        get_output_frames(client, "x.mp4", names=["a", "b"], count=3)
    with pytest.raises(DwApiError, match="only appl.* alongside `seams`"):
        get_output_frames(client, "x.mp4", boundaries=[8], at=[0.0])
    with pytest.raises(DwApiError, match="only appl.* alongside `seams`"):
        get_output_frames(client, "x.mp4", boundaries=[8], names=["a", "b"], count=3)

    assert seen == []  # refused before any request reached the server


def test_frames_ask_the_server_to_budget_the_tiles_and_echo_what_it_did():
    """The server shrinks the tiles together under max_total_bytes; the
    tool forwards the budget and reports `downscaled_to` as answered."""
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(
            200,
            json={
                "frame_count": 24,
                "fps": 6.0,
                "tiles": [tile_json(64, 32)],
                "crop": None,
                "downscaled_to": 64,
            },
        )

    client = DwClient(transport=httpx.MockTransport(handler))
    result = get_output_frames(client, "x.mp4", at=[0.0], max_dimension=2048)

    assert seen[0]["max_total_bytes"] == str(MAX_RETURNED_BYTES)
    assert seen[0]["max_dimension"] == "2048"
    assert result["downscaled_to"] == 64


def two_tone_tile(width, height, **kwargs):
    image = Image.new("RGB", (width, height), (255, 0, 0))
    image.paste((0, 0, 255), (width // 2, 0, width, height))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {
        **tile_json(width, height, **kwargs),
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def test_a_crop_is_forwarded_to_the_server_and_its_resolved_box_echoed_back():
    # Cropping now happens server-side (dw/media_frames.py), in source
    # pixels, before any downscale/composition - see test_media_frames.py
    # and test_server.py for the actual crop math. This only covers the
    # MCP layer's job: send `crop` as a query param, and read back the
    # server's resolved box rather than echoing the caller's own.
    seen = []
    client = frames_server([two_tone_tile(80, 60)], seen, crop=[100, 0, 80, 60])

    result = get_output_frames(client, "x.mp4", at=[0.0], crop=[100, 0, 80, 60])

    assert dict(seen[0][1])["crop"] == "100,0,80,60"
    assert result["crop"] == [100, 0, 80, 60]


def test_a_frame_crop_the_server_refuses_is_reported_to_the_caller():
    client = frames_server(
        [], status=400, detail="crop origin (150, 0) lies outside the 64x32 frame."
    )

    with pytest.raises(DwApiError, match="crop"):
        get_output_frames(client, "x.mp4", at=[0.0], crop=[150, 0, 100, 100])


def test_a_whole_track_over_budget_is_refused_from_its_content_length():
    """The refusal above must not have downloaded the track to make it:
    the server declares the body's length, and the tool refuses on that
    header without reading past it (#193 review)."""
    length = MAX_RETURNED_BYTES * 3 // 4 + 1024

    class Unread(httpx.SyncByteStream):
        iterated = False

        def __iter__(self):
            Unread.iterated = True
            yield b"\0" * 1024

        def close(self):
            pass

    def handler(request):
        return httpx.Response(
            200,
            headers={
                "content-type": "audio/wav",
                "content-length": str(length),
                "x-dw-duration": "240.0",
            },
            stream=Unread(),
        )

    client = DwClient(transport=httpx.MockTransport(handler))

    with pytest.raises(DwApiError) as caught:
        get_output_audio(client, "cut.mp4")

    assert "start" in str(caught.value) and "duration" in str(caught.value)
    assert str(length) in str(caught.value)
    assert Unread.iterated is False


def test_a_413_from_the_server_reads_as_the_same_excerpt_advice():
    """The gallery route refuses a long video's whole track with a 413
    before decoding it; the tool must surface that as the excerpt advice,
    not as an opaque HTTP failure."""

    def handler(request):
        return httpx.Response(
            413,
            json={
                "detail": "cut.mp4's whole soundtrack would be 5000000 bytes "
                "base64-encoded as WAV - over the 4194304 byte limit for an "
                "inline clip. Ask for an excerpt with `start` and `duration` "
                "(seconds), or download the file."
            },
        )

    client = DwClient(transport=httpx.MockTransport(handler))

    with pytest.raises(DwApiError) as caught:
        get_output_audio(client, "cut.mp4")

    message = str(caught.value)
    assert "start" in message and "duration" in message
    assert "HTTP 413" not in message


def test_hear_fetches_an_excerpt_around_each_moment():
    calls = []

    def handler(request):
        calls.append((request.url.path, dict(request.url.params)))
        if request.url.path.endswith("/frames"):
            return httpx.Response(
                200,
                json={
                    "frame_count": 48,
                    "fps": 24.0,
                    "tiles": [
                        tile_json(64, 32, "00:01.0 (frame 24)", 24, 1.0),
                        tile_json(64, 32, "00:00.2 (frame 5)", 5, 0.2),
                    ],
                },
            )
        return httpx.Response(
            200,
            content=b"RIFF" + b"\0" * 64,
            headers={
                "content-type": "audio/wav",
                "x-dw-duration": "2.0",
                "x-dw-excerpt-start": request.url.params["start"],
                "x-dw-excerpt-duration": request.url.params["duration"],
            },
        )

    client = DwClient(transport=httpx.MockTransport(handler))

    result = get_output_frames(client, "x.mp4", at=[1.0, 0.2], hear=1.0)

    audio_calls = [c for c in calls if c[0].endswith("/audio")]
    assert [c[1]["start"] for c in audio_calls] == ["0.5", "0.0"]  # never before 0
    assert [c[1]["duration"] for c in audio_calls] == ["1.0", "1.0"]
    assert all(t["audio"]["mime_type"] == "audio/wav" for t in result["tiles"])
    assert result["tiles"][0]["audio"]["excerpt"]["start"] == 0.5


def test_hear_on_a_mute_clip_keeps_the_frames_and_says_so():
    def handler(request):
        if request.url.path.endswith("/frames"):
            return httpx.Response(
                200,
                json={
                    "frame_count": 48,
                    "fps": 24.0,
                    "tiles": [tile_json(64, 32, "00:01.0 (frame 24)", 24, 1.0)],
                },
            )
        return httpx.Response(404, json={"detail": "x.mp4 carries no soundtrack"})

    client = DwClient(transport=httpx.MockTransport(handler))

    result = get_output_frames(client, "x.mp4", at=[1.0], hear=1.0)

    assert "audio" not in result["tiles"][0]
    assert "no soundtrack" in result["tiles"][0]["audio_error"]


def test_hear_is_refused_without_at():
    client = frames_server([])

    with pytest.raises(DwApiError, match="hear"):
        get_output_frames(client, "x.mp4", count=4, hear=1.0)


def test_hear_stops_fetching_once_the_aggregate_budget_is_spent(monkeypatch):
    """Each excerpt is well under get_output_audio's own per-clip cap, but
    three of them together are not under the response's overall budget:
    fetching must stop rather than blow the aggregate, and the tiles it
    stopped on say so rather than silently losing their audio (#193 review,
    finding 1). The tiles spend the same budget first, so 20 bytes are
    left for the excerpts."""
    tiles = [
        tile_json(64, 32, "00:00.0 (frame 0)", 0, 0.0),
        tile_json(64, 32, "00:01.0 (frame 24)", 24, 1.0),
        tile_json(64, 32, "00:02.0 (frame 48)", 48, 2.0),
    ]
    tile_bytes = sum(len(tile["data"]) for tile in tiles)
    monkeypatch.setattr(media, "MAX_RETURNED_BYTES", tile_bytes + 20)

    def handler(request):
        if request.url.path.endswith("/frames"):
            return httpx.Response(
                200,
                json={"frame_count": 72, "fps": 24.0, "tiles": tiles},
            )
        return httpx.Response(
            200,
            content=b"x" * 8,  # base64-encodes to 12 bytes: under the cap alone
            headers={"content-type": "audio/wav", "x-dw-duration": "2.0"},
        )

    client = DwClient(transport=httpx.MockTransport(handler))

    result = get_output_frames(client, "x.mp4", at=[0.0, 1.0, 2.0], hear=1.0)

    assert "audio" in result["tiles"][0]
    assert "audio_error" not in result["tiles"][0]
    assert result["tiles"][1]["audio_error"] == (
        "skipped - would exceed the response size budget"
    )
    assert result["tiles"][2]["audio_error"] == (
        "skipped - would exceed the response size budget"
    )
    assert "audio" not in result["tiles"][1]
    assert "audio" not in result["tiles"][2]
    assert result["audio_truncated"] is True


def test_hear_counts_the_tiles_against_the_response_budget(monkeypatch):
    """The tiles and the excerpts are one reply: excerpts that fit a budget
    of their own would still double what reaches the conversation."""
    tiles = [tile_json(64, 32, "00:00.0 (frame 0)", 0, 0.0)]
    monkeypatch.setattr(media, "MAX_RETURNED_BYTES", len(tiles[0]["data"]) + 4)

    def handler(request):
        if request.url.path.endswith("/frames"):
            return httpx.Response(
                200, json={"frame_count": 24, "fps": 24.0, "tiles": tiles}
            )
        return httpx.Response(
            200,
            content=b"x" * 3,  # base64-encodes to 4 bytes: exactly what is left
            headers={"content-type": "audio/wav", "x-dw-duration": "1.0"},
        )

    client = DwClient(transport=httpx.MockTransport(handler))
    result = get_output_frames(client, "x.mp4", at=[0.0], hear=1.0)
    assert "audio" in result["tiles"][0]

    monkeypatch.setattr(media, "MAX_RETURNED_BYTES", len(tiles[0]["data"]) + 3)
    result = get_output_frames(client, "x.mp4", at=[0.0], hear=1.0)
    assert result["audio_truncated"] is True


def _assess_client(body=None):
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, dict(request.url.params)))
        return httpx.Response(200, json=body or {"findings": []})

    return DwClient(transport=httpx.MockTransport(handler)), seen


def test_assess_output_surfaces_the_servers_unknown_probe_refusal():
    """#388: the probe is whitelisted before anything else is read. The
    server owns the whitelist (tests/test_server_assess.py); a copy here
    would be a second owner, so its 400 reaches the caller as is."""
    from dw.server.assess import PROBES, unknown_probe

    detail = unknown_probe("analyze_vibes")

    def handler(request):
        return httpx.Response(400, json={"detail": detail})

    client = DwClient(transport=httpx.MockTransport(handler))
    with pytest.raises(DwApiError) as refused:
        media.assess_output(client, "cut.mp4", probe="analyze_vibes")

    for probe in PROBES:
        assert probe in str(refused.value)


def test_assess_output_passes_probe_detail_and_an_asset_name_through():
    client, seen = _assess_client({"probe": "analyze_seams", "seams": []})

    result = media.assess_output(
        client, "asset:episode.mp4", probe="analyze_seams", detail=True
    )

    assert result == {"probe": "analyze_seams", "seams": []}
    method, path, params = seen[0]
    assert method == "GET"
    assert path == "/api/gallery/asset:episode.mp4/assess"
    assert params == {"probe": "analyze_seams", "detail": "true"}


def test_assess_output_sends_no_parameters_by_default():
    client, seen = _assess_client()

    media.assess_output(client, "cut.mp4")

    assert seen[0][2] == {}
