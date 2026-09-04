"""Returning a generated image to the agent: downscaled enough to be worth
a context window, honest about what it refuses."""

import base64
import io

import httpx
import numpy as np
import pytest
from PIL import Image

import dw_mcp.media as media
from dw_mcp.client import DwApiError, DwClient
from dw_mcp.media import MAX_RETURNED_BYTES, get_output_image


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


def test_an_undecodable_body_is_refused_clearly():
    client = serving(b"not an image at all", "image/png")

    with pytest.raises(DwApiError, match="could not be decoded"):
        get_output_image(client, "broken.png")


def test_a_missing_file_propagates_the_api_error():
    def handler(request):
        return httpx.Response(404, json={"detail": "Unknown file"})

    client = DwClient(transport=httpx.MockTransport(handler))

    with pytest.raises(DwApiError, match="Unknown file"):
        get_output_image(client, "ghost.png")


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


def test_a_missing_text_output_surfaces_the_error():
    def handler(request):
        return httpx.Response(404, json={"detail": "Unknown file"})

    client = DwClient(transport=httpx.MockTransport(handler))

    with pytest.raises(DwApiError):
        media.get_output_text(client, "ghost.txt")


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


def test_delete_output_surfaces_a_missing_file():
    def handler(request):
        return httpx.Response(404, json={"detail": "Unknown file"})

    client = DwClient(transport=httpx.MockTransport(handler))

    with pytest.raises(DwApiError, match="Unknown file"):
        media.delete_output(client, "ghost.png")


# --------------------------------------------------------- output download


import os

from dw_mcp.media import download_output


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
