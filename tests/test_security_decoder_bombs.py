"""A crafted PNG that is small on disk and enormous once decoded.

`get_output_image` (MCP) and the gallery thumbnail route both decode a whole
image a caller names. A PNG header can claim any size; the pixels are a
zlib stream that compresses a flat colour about a thousand to one. Pillow's
own guard (`Image.MAX_IMAGE_PIXELS`, ~89.5M) raises only above *twice* that
and only warns in between - so without a clamp of dw's own, an image just
under 179M pixels is decoded in full: half a gigabyte of RGB per request.

The probe never lets a decode happen, whatever the boundary does: the
headers here are real, but `ImageFile.load` is replaced for the test with
one that records the attempt and raises before allocating. A test fails when
that record is non-empty - "refused" means refused *before* the decode.
"""

import struct
import zlib

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageFile

from dw.server.app import create_app
from dw.server.jobs import JobManager
from dw_mcp.client import DwApiError, DwClient
from dw_mcp.media import get_output_image

from .test_server import ScriptedWorkerManager, success_script

# Well above anything the tool legitimately returns (a 4K frame is 8.3M) and
# well below Pillow's default, where only dw's own clamp can stop it
UNDER_PILLOWS_ERROR = (12_000, 12_000)  # 144M pixels: Pillow only warns
OVER_PILLOWS_ERROR = (20_000, 20_000)  # 400M pixels: Pillow refuses on open
DECODE_LIMIT = 50_000_000


def _chunk(kind, data):
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def bomb_png(width, height):
    """A syntactically valid PNG header claiming width x height grayscale
    pixels, with a token IDAT. Tiny on disk; never decoded by this file."""
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(b"\x00" * 64))
        + _chunk(b"IEND", b"")
    )


@pytest.fixture
def decodes(monkeypatch):
    """Every attempt to decode an image over DECODE_LIMIT pixels, recorded
    and stopped before the allocation."""
    attempts = []
    original = ImageFile.ImageFile.load

    def guarded(self):
        width, height = self.size
        if width * height > DECODE_LIMIT:
            attempts.append(self.size)
            raise MemoryError("probe: a decompression bomb was decoded")
        return original(self)

    monkeypatch.setattr(ImageFile.ImageFile, "load", guarded)
    return attempts


def test_the_probe_png_parses_to_the_size_it_claims():
    """The header is honest enough for Pillow to believe it - otherwise the
    tests below would pass on a parse error rather than a clamp."""
    import io
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(bomb_png(*UNDER_PILLOWS_ERROR))) as image:
            assert image.size == UNDER_PILLOWS_ERROR


def test_pillows_guard_has_not_been_disabled():
    """Everything above 2 x MAX_IMAGE_PIXELS rests on Pillow's default; a
    module that set it to None would open every size below."""
    assert Image.MAX_IMAGE_PIXELS is not None
    assert Image.MAX_IMAGE_PIXELS <= 100_000_000


def _serving(body):
    def handler(request):
        return httpx.Response(200, content=body, headers={"content-type": "image/png"})

    return DwClient(transport=httpx.MockTransport(handler))


class TestGetOutputImage:
    def test_a_bomb_over_pillows_limit_is_refused_before_decode(self, decodes):
        with pytest.raises((DwApiError, Image.DecompressionBombError)):
            get_output_image(_serving(bomb_png(*OVER_PILLOWS_ERROR)), "bomb.png")
        assert decodes == []

    @pytest.mark.xfail(
        strict=True,
        reason="no pixel clamp - gap: get_output_image calls image.load() on "
        "whatever Pillow opens, and Pillow only warns below 2x MAX_IMAGE_PIXELS",
    )
    def test_a_bomb_under_pillows_limit_is_refused_before_decode(self, decodes):
        try:
            get_output_image(_serving(bomb_png(*UNDER_PILLOWS_ERROR)), "bomb.png")
        except DwApiError:
            pass
        assert decodes == []

    @pytest.mark.xfail(
        strict=True,
        reason="no pixel clamp - gap: a crop is cut from the fully decoded "
        "image, so a small crop still decodes the whole bomb",
    )
    def test_a_crop_does_not_decode_the_whole_bomb(self, decodes):
        try:
            get_output_image(
                _serving(bomb_png(*UNDER_PILLOWS_ERROR)), "bomb.png", crop=[0, 0, 8, 8]
            )
        except DwApiError:
            pass
        assert decodes == []


@pytest.fixture
def gallery(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (tmp_path / "workflows").mkdir()
    manager = JobManager(
        str(outputs),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
    )
    app = create_app(
        workflow_dir=str(tmp_path / "workflows"),
        output_dir=str(outputs),
        job_manager=manager,
        prompt_dir=str(tmp_path / "prompts"),
    )
    client = TestClient(app, base_url="http://localhost", raise_server_exceptions=False)
    return client, outputs


class TestGalleryThumbnail:
    def test_a_bomb_over_pillows_limit_is_refused_before_decode(self, gallery, decodes):
        client, outputs = gallery
        (outputs / "bomb.png").write_bytes(bomb_png(*OVER_PILLOWS_ERROR))
        with client:
            response = client.get("/api/gallery/bomb.png/thumbnail")
        assert response.status_code >= 400
        assert decodes == []

    @pytest.mark.xfail(
        strict=True,
        reason="no pixel clamp - gap: gallery_thumbnail's draft() is a no-op "
        "for PNG, so thumbnail() decodes the full image first",
    )
    def test_a_bomb_under_pillows_limit_is_refused_before_decode(
        self, gallery, decodes
    ):
        client, outputs = gallery
        (outputs / "bomb.png").write_bytes(bomb_png(*UNDER_PILLOWS_ERROR))
        with client:
            client.get("/api/gallery/bomb.png/thumbnail")
        assert decodes == []

    @pytest.mark.xfail(
        strict=True,
        reason="no pixel clamp - gap: read_embedded_metadata (dw/result.py) "
        "reads PngImageFile.text, which makes Pillow load() the whole image "
        "to reach text chunks after IDAT",
    )
    def test_the_gallery_metadata_route_does_not_decode_it(self, gallery, decodes):
        """Width and height are in the header; nothing about a request for
        metadata needs every pixel of a bomb."""
        client, outputs = gallery
        (outputs / "bomb.png").write_bytes(bomb_png(*UNDER_PILLOWS_ERROR))
        with client:
            client.get("/api/gallery/bomb.png/metadata")
        assert decodes == []

    def test_the_gallery_listing_does_not_decode_it(self, gallery, decodes):
        client, outputs = gallery
        (outputs / "bomb.png").write_bytes(bomb_png(*UNDER_PILLOWS_ERROR))
        with client:
            assert client.get("/api/gallery").status_code == 200
        assert decodes == []
