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
