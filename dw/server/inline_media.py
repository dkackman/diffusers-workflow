"""Images sized for an inline answer: fitted to a longest side and, when
asked, halved until their base64 fits a byte budget.

The frames route and the gallery's image route both answer with pixels
a client sends straight into a conversation, so both shrink here, on the
server that already holds the decoded image - a client that decoded and
re-encoded what it was sent did the same work twice and kept a second
copy of these rules.
"""

import io

from fastapi import HTTPException

from ..media import base64_size
from ..security import MAX_DECODE_PIXELS

__all__ = [
    "base64_size",
    "encode_within_budget",
    "fit_longest",
    "fit_tiles_within_budget",
    "open_bounded",
    "png_bytes",
]


def fit_longest(image, limit):
    """A copy no larger than `limit` on its longest side, aspect kept. An
    image already inside the limit comes back as-is: upscaling would invent
    detail the reader would then reason about."""
    from PIL import Image

    longest = max(image.width, image.height)
    if longest <= limit:
        return image
    scale = limit / longest
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.LANCZOS,
    )


def open_bounded(path, name):
    """The image at `path`, loaded, or a 413 when its header declares more
    pixels than MAX_DECODE_PIXELS - refused before the pixels are decoded."""
    from PIL import Image

    try:
        image = Image.open(path)
        if image.width * image.height > MAX_DECODE_PIXELS:
            raise HTTPException(
                status_code=413,
                detail=f"{name} is {image.width}x{image.height}, more than the "
                f"{MAX_DECODE_PIXELS:,} pixels an image is decoded from",
            )
        image.load()
        return image
    except Image.DecompressionBombError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except (OSError, ValueError, SyntaxError):
        # What Pillow raises for a truncated, corrupt or unknown file
        # (UnidentifiedImageError is an OSError): the file's problem, not
        # the server's, so a refusal that says so rather than a 500
        raise HTTPException(
            status_code=422, detail=f"{name} could not be decoded as an image"
        )


def _encoded(image, fmt):
    buffer = io.BytesIO()
    # Each format writes some modes only: JPEG has no alpha or palette, PNG
    # no CMYK - a print-ready JPEG asked for as PNG would otherwise fail
    if fmt == "JPEG" and image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    elif fmt == "PNG" and image.mode not in (
        "1",
        "L",
        "LA",
        "P",
        "RGB",
        "RGBA",
        "I",
        "I;16",
    ):
        image = image.convert("RGBA" if "A" in image.mode else "RGB")
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def png_bytes(image):
    return _encoded(image, "PNG")


def encode_within_budget(image, limit, fmt, max_base64_bytes=None, floor=64):
    """`image` fitted to `limit` and encoded as `fmt`; with a budget, halved
    until its base64 fits or its longest side reaches `floor`. Two loops
    rather than a calculation: compressed size does not follow from pixel
    count, and noise and flat colour differ by an order of magnitude.
    Returns (bytes, the image those bytes encode)."""
    sized = fit_longest(image, limit)
    data = _encoded(sized, fmt)
    while (
        max_base64_bytes is not None
        and base64_size(len(data)) > max_base64_bytes
        and max(sized.size) > floor
    ):
        limit = max(floor, max(sized.size) // 2)
        sized = fit_longest(sized, limit)
        data = _encoded(sized, fmt)
    return data, sized


def fit_tiles_within_budget(images, limit, max_base64_bytes, floor=64):
    """Every tile shrunk to one shared longest side, halved until their PNG
    base64 sizes sum within the budget or the side reaches `floor`. A seam
    pair at half size is still a seam pair; a tile dropped is a different
    answer. Returns (tiles, the side they were shrunk to, or None when
    nothing had to shrink)."""

    def total(tiles):
        return sum(base64_size(len(png_bytes(tile))) for tile in tiles)

    if not images or total(images) <= max_base64_bytes:
        return images, None
    side = max(max(image.size) for image in images)
    while True:
        side = max(floor, side // 2)
        fitted = [fit_longest(image, side) for image in images]
        if total(fitted) <= max_base64_bytes or side <= floor:
            return fitted, side
