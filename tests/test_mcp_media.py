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


def test_a_large_image_is_downscaled_to_max_dimension():
    client = serving(png_bytes(2048, 1024), "image/png")

    result = get_output_image(client, "big.png", max_dimension=512)

    assert decoded(result).size == (512, 256)
    assert result["original_size"] == [2048, 1024]
    assert result["returned_size"] == [512, 256]


def test_the_taller_side_governs_the_downscale():
    client = serving(png_bytes(600, 1200), "image/png")

    result = get_output_image(client, "tall.png", max_dimension=600)

    assert decoded(result).size == (300, 600)


def test_a_small_image_is_returned_at_its_own_size():
    client = serving(png_bytes(64, 48), "image/png")

    result = get_output_image(client, "small.png", max_dimension=768)

    assert decoded(result).size == (64, 48)
    assert result["returned_size"] == [64, 48]


class TestCrop:
    """A 2K still cannot be checked for whether a small element reads, or
    for a decode-tiling seam, through a 768-pixel downscale. `crop` is a
    box in the original's pixels, cut before the downscale, so a region
    of the full-resolution file comes back at 100%."""

    def two_tone(self):
        # left half red, right half blue, 2048 wide
        image = Image.new("RGB", (2048, 1024), (255, 0, 0))
        image.paste((0, 0, 255), (1024, 0, 2048, 1024))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return serving(buffer.getvalue(), "image/png")

    def test_a_crop_returns_that_region_at_full_resolution(self):
        result = get_output_image(
            self.two_tone(), "big.png", max_dimension=768, crop=[1000, 0, 400, 300]
        )
        image = decoded(result)
        assert image.size == (400, 300)
        assert image.getpixel((0, 0)) == (255, 0, 0)
        assert image.getpixel((399, 0)) == (0, 0, 255)
        assert result["original_size"] == [2048, 1024]
        assert result["crop"] == [1000, 0, 400, 300]
        assert result["returned_size"] == [400, 300]

    def test_a_crop_wider_than_max_dimension_is_still_downscaled(self):
        result = get_output_image(
            self.two_tone(), "big.png", max_dimension=256, crop=[0, 0, 1024, 512]
        )
        assert decoded(result).size == (256, 128)
        assert result["crop"] == [0, 0, 1024, 512]

    def test_a_crop_is_clamped_to_the_image(self):
        result = get_output_image(
            self.two_tone(), "big.png", max_dimension=768, crop=[1900, 900, 500, 500]
        )
        assert decoded(result).size == (148, 124)
        assert result["crop"] == [1900, 900, 148, 124]

    @pytest.mark.parametrize(
        "crop", [[0, 0, 0, 10], [-1, 0, 10, 10], [2048, 0, 10, 10], [0, 0, 10], "x"]
    )
    def test_an_empty_or_malformed_crop_is_refused(self, crop):
        with pytest.raises(DwApiError, match="crop"):
            get_output_image(self.two_tone(), "big.png", crop=crop)


def test_a_jpeg_source_comes_back_as_jpeg():
    client = serving(jpeg_bytes(300, 300), "image/jpeg")

    result = get_output_image(client, "photo.jpg")

    assert result["mime_type"] == "image/jpeg"


def test_a_png_source_comes_back_as_png():
    client = serving(png_bytes(300, 300), "image/png")

    assert get_output_image(client, "a.png")["mime_type"] == "image/png"


def test_the_result_stays_under_the_byte_ceiling():
    """A hard cap matters more than fidelity - a payload over the ceiling
    would crowd out the conversation it is meant to inform."""
    import random

    noise = Image.new("RGB", (4000, 4000))
    noise.putdata(
        [
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(4000 * 4000)
        ]
    )
    buffer = io.BytesIO()
    noise.save(buffer, format="PNG")
    client = serving(buffer.getvalue(), "image/png")

    result = get_output_image(client, "noise.png", max_dimension=4000)

    assert result["bytes"] <= MAX_RETURNED_BYTES
    assert len(base64.b64decode(result["data"])) == result["bytes"]


def test_a_video_output_is_refused_by_name():
    client = serving(b"\x00\x00\x00\x18ftypmp42", "video/mp4")

    with pytest.raises(DwApiError) as caught:
        get_output_image(client, "clip.mp4")

    assert "video/mp4" in str(caught.value)


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


def test_a_non_image_output_is_refused_without_reading_the_body():
    """A video can be arbitrarily large - the content-type header alone
    should be enough to refuse it, before the body is ever downloaded."""

    class TrackingStream(httpx.SyncByteStream):
        def __init__(self, chunks):
            self.chunks = chunks
            self.iterated = False

        def __iter__(self):
            self.iterated = True
            yield from self.chunks

        def close(self):
            pass

    stream = TrackingStream([b"\x00\x00\x00\x18ftypmp42" * 100000])

    def handler(request):
        return httpx.Response(200, headers={"content-type": "video/mp4"}, stream=stream)

    client = DwClient(transport=httpx.MockTransport(handler))

    with pytest.raises(DwApiError):
        get_output_image(client, "clip.mp4")

    assert stream.iterated is False


def test_the_name_is_url_quoted_in_the_request():
    # "#" starts a URL fragment when left unescaped - an unquoted name would
    # arrive at the server truncated ("a b", with "1.png" silently dropped as
    # a fragment). httpx's request.url.path decodes percent-escapes back for
    # display, so the escaping itself is checked on url.raw_path, the bytes
    # actually placed on the wire; url.path then confirms the full name
    # (not a truncated one) is what the server would see.
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["raw_path"] = request.url.raw_path.decode("ascii")
        return httpx.Response(
            200, content=png_bytes(10, 10), headers={"content-type": "image/png"}
        )

    get_output_image(DwClient(transport=httpx.MockTransport(handler)), "a b#1.png")

    assert "%23" in seen["raw_path"]
    assert seen["path"] == "/outputs/a b#1.png"


def test_bytes_that_are_not_a_decodable_image_are_refused():
    """A truncated or corrupt file is served with an image content type like
    any other - only the decode tells us it is unusable."""
    client = serving(b"\x89PNG\r\n\x1a\ntruncated", "image/png")

    with pytest.raises(DwApiError) as caught:
        get_output_image(client, "broken.png")

    assert "could not be decoded" in str(caught.value)


def test_the_budget_is_checked_against_the_base64_size_not_the_raw_bytes(monkeypatch):
    """The payload the caller actually receives is base64 text (4/3 the raw
    bytes). A cap that only looked at the raw encoded bytes would let a
    payload through that is over budget once encoded."""
    client = serving(noise_png_bytes(300, 300), "image/png")

    raw_result = get_output_image(client, "noise.png", max_dimension=300)
    raw_bytes = raw_result["bytes"]

    # Set the cap strictly between the raw size and its base64 expansion, so
    # a raw-bytes comparison would accept the first encoding while a
    # base64-aware comparison must keep shrinking.
    budget = raw_bytes + 1
    assert budget < 4 * -(-raw_bytes // 3)
    monkeypatch.setattr(media, "MAX_RETURNED_BYTES", budget)

    client = serving(noise_png_bytes(300, 300), "image/png")
    result = get_output_image(client, "noise.png", max_dimension=300)

    encoded_len = len(base64.b64decode(result["data"]))
    base64_len = 4 * -(-encoded_len // 3)
    assert base64_len <= budget
    assert result["returned_size"] != raw_result["returned_size"]


def test_the_downscale_loop_resizes_from_the_previous_result_not_the_original(
    monkeypatch,
):
    calls = []
    original_fit = media._fit

    def tracking_fit(image, limit):
        result = original_fit(image, limit)
        calls.append((image, result))
        return result

    monkeypatch.setattr(media, "_fit", tracking_fit)
    monkeypatch.setattr(media, "MAX_RETURNED_BYTES", 1)

    client = serving(noise_png_bytes(600, 600), "image/png")
    get_output_image(client, "noise.png", max_dimension=600)

    assert len(calls) >= 2
    for previous, current in zip(calls, calls[1:]):
        _, previous_sized = previous
        current_source, _ = current
        assert current_source is previous_sized


def test_a_jpeg_in_an_unencodable_mode_is_converted_before_re_encoding():
    """A CMYK JPEG cannot be re-saved as JPEG without a conversion first."""
    buffer = io.BytesIO()
    Image.new("CMYK", (200, 100)).save(buffer, format="JPEG")
    client = serving(buffer.getvalue(), "image/jpeg")

    result = get_output_image(client, "cmyk.jpg", max_dimension=64)

    assert result["mime_type"] == "image/jpeg"
    assert decoded(result).mode == "RGB"


def test_a_dot_segment_name_survives_intact_onto_the_wire():
    """httpx normalizes `..` out of a request path client-side, which would
    escape the /outputs prefix entirely and skip the static mount's own
    confinement. Quoting the separator keeps the literal bytes on the wire so
    it is the server that refuses the name. Gallery names come from a listing
    of one flat directory, so a legitimate one never contains '/'."""
    seen = {}

    def handler(request):
        seen["raw_path"] = request.url.raw_path
        return httpx.Response(
            200, content=png_bytes(10, 10), headers={"content-type": "image/png"}
        )

    get_output_image(DwClient(transport=httpx.MockTransport(handler)), "../api/models")

    assert seen["raw_path"] == b"/outputs/..%2Fapi%2Fmodels"


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


def deleting_by_job(job):
    """GET /api/jobs/job-1 answers `job`; DELETE on the gallery answers as
    the server's run-directory form does."""
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, dict(request.url.params)))
        if request.method == "GET" and request.url.path == "/api/jobs/job-1":
            return httpx.Response(200, json=job)
        if request.method == "GET":
            return httpx.Response(404, json={"detail": "Unknown job"})
        name = request.url.path.removeprefix("/api/gallery/")
        return httpx.Response(
            200, json={"name": name, "deleted": True, "run_swept": name.split("/")[-1]}
        )

    return DwClient(transport=httpx.MockTransport(handler)), seen


def test_delete_output_by_job_id_deletes_the_run_directory_the_job_wrote():
    client, seen = deleting_by_job(
        {
            "id": "job-1",
            "status": "succeeded",
            "run_dir": "ltx2/Gyre",
            "workspace": "default",
        }
    )

    result = media.delete_output(client, job_id="job-1")

    assert [entry[:2] for entry in seen] == [
        ("GET", "/api/jobs/job-1"),
        ("DELETE", "/api/gallery/ltx2/Gyre"),
    ]
    assert result == {
        "name": "ltx2/Gyre",
        "deleted": True,
        "run_swept": "Gyre",
        "job_id": "job-1",
        "run_dir": "ltx2/Gyre",
    }


def test_delete_output_by_job_id_goes_to_the_workspace_the_job_ran_in():
    """The run directory is wherever the job wrote it, so with no pin the
    delete follows the job's own workspace rather than the session's."""
    client, seen = deleting_by_job(
        {"id": "job-1", "status": "failed", "run_dir": "w/Run", "workspace": "shots"}
    )
    client.workspace = "elsewhere"

    media.delete_output(client, job_id="job-1")

    assert seen[-1][1] == "/api/gallery/w/Run"
    assert seen[-1][2] == {"workspace": "shots"}


def test_delete_output_by_job_id_honours_an_explicit_workspace():
    client, seen = deleting_by_job(
        {"id": "job-1", "status": "failed", "run_dir": "w/Run", "workspace": "shots"}
    )

    media.delete_output(client, job_id="job-1", workspace="pinned")

    assert seen[-1][2] == {"workspace": "pinned"}


def test_delete_output_refuses_neither_name_nor_job_id():
    client, seen = deleting_by_job({})

    with pytest.raises(DwApiError, match="exactly one"):
        media.delete_output(client)

    assert seen == []


def test_delete_output_refuses_both_name_and_job_id():
    client, seen = deleting_by_job({})

    with pytest.raises(DwApiError, match="exactly one"):
        media.delete_output(client, "out.png", job_id="job-1")

    assert seen == []


def test_delete_output_by_job_id_surfaces_an_unknown_job():
    client, seen = deleting_by_job({})

    with pytest.raises(DwApiError, match="Unknown job"):
        media.delete_output(client, job_id="ghost")

    assert [entry[0] for entry in seen] == ["GET"], "nothing is deleted"


def test_delete_output_by_job_id_refuses_a_job_with_no_run_directory():
    """A job refused before it started (or one from before run tracking)
    has no run_dir; that is an error, not a delete of nothing."""
    client, seen = deleting_by_job(
        {"id": "job-1", "status": "failed", "run_dir": None, "workspace": "default"}
    )

    with pytest.raises(DwApiError, match="no run directory"):
        media.delete_output(client, job_id="job-1")

    assert [entry[0] for entry in seen] == ["GET"], "nothing is deleted"


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


def test_two_selectors_are_refused_before_any_request():
    client = frames_server([])

    with pytest.raises(DwApiError, match="one of"):
        get_output_frames(client, "x.mp4", at=[0.0], count=4)


def test_tiles_over_budget_are_shrunk_together_and_say_so():
    # three noisy 2048x1024 tiles: well over 4MB base64 between them
    tiles = []
    for n in range(3):
        tiles.append(
            {
                **tile_json(2048, 1024, frame=n, seconds=float(n)),
                "data": base64.b64encode(noise_png_bytes(2048, 1024, seed=n)).decode(
                    "ascii"
                ),
            }
        )
    client = frames_server(tiles)

    result = get_output_frames(client, "x.mp4", at=[0, 1, 2], max_dimension=2048)

    total = sum(len(t["data"]) for t in result["tiles"])
    assert total <= MAX_RETURNED_BYTES
    assert len(result["tiles"]) == 3  # shrunk, not dropped
    assert result["downscaled_to"] is not None and result["downscaled_to"] < 2048
    assert all(decoded(t).width == result["tiles"][0]["width"] for t in result["tiles"])


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
    finding 1)."""
    monkeypatch.setattr(media, "MAX_RETURNED_BYTES", 20)

    def handler(request):
        if request.url.path.endswith("/frames"):
            return httpx.Response(
                200,
                json={
                    "frame_count": 72,
                    "fps": 24.0,
                    "tiles": [
                        tile_json(64, 32, "00:00.0 (frame 0)", 0, 0.0),
                        tile_json(64, 32, "00:01.0 (frame 24)", 24, 1.0),
                        tile_json(64, 32, "00:02.0 (frame 48)", 48, 2.0),
                    ],
                },
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
