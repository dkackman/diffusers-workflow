"""Returning a generated image to the agent: downscaled enough to be worth
a context window, honest about what it refuses."""

import base64
import io

import httpx
import pytest
from PIL import Image

from dw.mcp.client import DwApiError, DwClient
from dw.mcp.media import MAX_RETURNED_BYTES, get_output_image


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
