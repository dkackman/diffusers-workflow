"""Inline images on the server: one helper for frames and the gallery's
image route, so a client asks for a size and a byte budget instead of
decoding and re-encoding what the server already holds."""

import base64
import io

import numpy
import pytest
from PIL import Image

from dw.server import inline_media


def noise(width, height, seed=0):
    rng = numpy.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=numpy.uint8)
    return Image.fromarray(pixels, "RGB")


def test_base64_size_is_the_encoded_length():
    for n in (0, 1, 2, 3, 4, 1000):
        assert inline_media.base64_size(n) == len(base64.b64encode(b"x" * n))


def test_fit_longest_shrinks_to_the_limit_and_never_upscales():
    assert inline_media.fit_longest(noise(400, 200), 100).size == (100, 50)
    small = noise(40, 20)
    assert inline_media.fit_longest(small, 100) is small


def test_encode_within_budget_halves_until_it_fits():
    image = noise(512, 512)
    data, fitted = inline_media.encode_within_budget(
        image, 512, "PNG", max_base64_bytes=60_000
    )
    assert inline_media.base64_size(len(data)) <= 60_000
    assert max(fitted.size) < 512
    assert Image.open(io.BytesIO(data)).size == fitted.size


def test_encode_within_budget_stops_at_its_floor():
    data, fitted = inline_media.encode_within_budget(
        noise(512, 512), 512, "PNG", max_base64_bytes=10, floor=64
    )
    assert max(fitted.size) == 64


def test_fit_tiles_within_budget_shrinks_every_tile_together():
    tiles = [noise(256, 128, seed=n) for n in range(3)]
    fitted, downscaled_to = inline_media.fit_tiles_within_budget(
        tiles, 256, max_base64_bytes=80_000
    )
    assert downscaled_to is not None and downscaled_to < 256
    assert len({tile.size for tile in fitted}) == 1
    total = sum(
        inline_media.base64_size(len(inline_media.png_bytes(tile))) for tile in fitted
    )
    assert total <= 80_000


def test_fit_tiles_within_budget_leaves_tiles_that_fit():
    tiles = [noise(32, 32)]
    fitted, downscaled_to = inline_media.fit_tiles_within_budget(
        tiles, 32, max_base64_bytes=10_000_000
    )
    assert downscaled_to is None and fitted[0].size == (32, 32)


@pytest.fixture
def gallery(tmp_path):
    """A server over an outputs and an assets directory, and a writer of
    images into either."""
    from fastapi.testclient import TestClient

    from dw.server.app import create_app
    from dw.server.jobs import JobManager

    from .test_server import ScriptedWorkerManager, success_script

    (tmp_path / "workflows").mkdir()
    (tmp_path / "assets").mkdir()
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    manager = JobManager(
        str(outputs),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
    )
    app = create_app(
        workflow_dir=str(tmp_path / "workflows"),
        output_dir=str(outputs),
        job_manager=manager,
        asset_dir=str(tmp_path / "assets"),
    )
    with TestClient(app, base_url="http://localhost") as client:

        def write(relative, image, fmt="PNG", root=outputs):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, format=fmt)
            return relative

        yield client, write, tmp_path


def decoded(response):
    return Image.open(io.BytesIO(response.content))


class TestGalleryImage:
    def test_a_large_image_is_downscaled_to_max_dimension(self, gallery):
        client, write, _ = gallery
        name = write("w/big.png", noise(1200, 600))
        answer = client.get(f"/api/gallery/{name}/image", params={"max_dimension": 300})
        assert answer.status_code == 200, answer.text
        assert decoded(answer).size == (300, 150)
        assert answer.headers["x-dw-original-size"] == "1200,600"
        assert answer.headers["x-dw-returned-size"] == "300,150"

    def test_a_crop_returns_that_region_at_full_resolution(self, gallery):
        client, write, _ = gallery
        name = write("w/big.png", noise(1200, 600))
        answer = client.get(
            f"/api/gallery/{name}/image",
            params={"crop": "100,50,200,100", "max_dimension": 2048},
        )
        assert decoded(answer).size == (200, 100)
        assert answer.headers["x-dw-crop"] == "100,50,200,100"

    def test_a_crop_is_clamped_to_the_image(self, gallery):
        client, write, _ = gallery
        name = write("w/big.png", noise(400, 200))
        answer = client.get(
            f"/api/gallery/{name}/image", params={"crop": "300,100,500,500"}
        )
        assert decoded(answer).size == (100, 100)
        assert answer.headers["x-dw-crop"] == "300,100,100,100"

    def test_a_malformed_crop_is_a_400(self, gallery):
        client, write, _ = gallery
        name = write("w/big.png", noise(400, 200))
        answer = client.get(f"/api/gallery/{name}/image", params={"crop": "a,b"})
        assert answer.status_code == 400

    def test_max_bytes_halves_until_the_base64_fits(self, gallery):
        client, write, _ = gallery
        name = write("w/big.png", noise(1024, 1024))
        answer = client.get(
            f"/api/gallery/{name}/image",
            params={"max_dimension": 1024, "max_bytes": 200_000},
        )
        assert inline_media.base64_size(len(answer.content)) <= 200_000
        assert "x-dw-downscaled-to" in answer.headers

    def test_a_jpeg_source_comes_back_jpeg(self, gallery):
        client, write, _ = gallery
        name = write("w/photo.jpg", noise(200, 100), fmt="JPEG")
        answer = client.get(f"/api/gallery/{name}/image")
        assert answer.headers["content-type"] == "image/jpeg"

    def test_a_video_is_404(self, gallery):
        client, _, tmp_path = gallery
        (tmp_path / "outputs" / "w").mkdir(parents=True, exist_ok=True)
        (tmp_path / "outputs" / "w" / "clip.mp4").write_bytes(b"not really")
        assert client.get("/api/gallery/w/clip.mp4/image").status_code == 404

    def test_an_asset_reference_is_read(self, gallery):
        client, write, tmp_path = gallery
        write("cast/priya.png", noise(64, 32), root=tmp_path / "assets")
        answer = client.get("/api/gallery/asset:cast/priya.png/image")
        assert answer.status_code == 200, answer.text
        assert decoded(answer).size == (64, 32)
