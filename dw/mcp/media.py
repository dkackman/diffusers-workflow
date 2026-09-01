"""Hand a generated image back to the agent.

Output media is served from the /outputs static mount rather than an /api
route, so this is the one tool that reaches outside /api. Everything is
downscaled and capped before it is returned: a full-resolution render would
cost more context than the answer it is meant to support.
"""

import base64
import io
from urllib.parse import quote

from PIL import Image

from dw.mcp.client import DwApiError

# Roughly 4MB of encoded image. Past this the payload crowds out the
# conversation it is supposed to inform
MAX_RETURNED_BYTES = 4 * 1024 * 1024
MIN_DIMENSION = 64


def get_output_image(client, name, max_dimension=768):
    """One image from the output directory, downscaled, as base64 plus the
    sizes it went in and came out at."""
    body, content_type = client.get_bytes(f"/outputs/{quote(name)}")
    if content_type and not content_type.startswith("image/"):
        raise DwApiError(
            f"{name} is {content_type}, not an image - this tool returns "
            "images only. Use get_gallery_metadata to inspect other media."
        )
    try:
        image = Image.open(io.BytesIO(body))
        image.load()
    except Exception:
        raise DwApiError(f"{name} could not be decoded as an image.")

    original_size = [image.width, image.height]
    fmt = "JPEG" if (image.format or "").upper() == "JPEG" else "PNG"
    if image.mode not in ("RGB", "L") and fmt == "JPEG":
        image = image.convert("RGB")

    limit = max(MIN_DIMENSION, int(max_dimension))
    encoded, sized = _encode_within_budget(image, limit, fmt)
    return {
        "name": name,
        "data": base64.b64encode(encoded).decode("ascii"),
        "mime_type": "image/jpeg" if fmt == "JPEG" else "image/png",
        "original_size": original_size,
        "returned_size": [sized.width, sized.height],
        "bytes": len(encoded),
    }


def _encode_within_budget(image, limit, fmt):
    """Shrink until the encoded bytes fit the ceiling. Two loops rather than
    one calculation because compressed size does not follow from pixel count
    - noise and flat colour differ by an order of magnitude."""
    while True:
        sized = _fit(image, limit)
        buffer = io.BytesIO()
        sized.save(buffer, format=fmt)
        encoded = buffer.getvalue()
        if len(encoded) <= MAX_RETURNED_BYTES or limit <= MIN_DIMENSION:
            return encoded, sized
        limit = max(MIN_DIMENSION, limit // 2)


def _fit(image, limit):
    """A copy no larger than `limit` on its longest side, aspect preserved.
    An image already inside the limit is returned as-is - upscaling would
    invent detail the model would then reason about."""
    longest = max(image.width, image.height)
    if longest <= limit:
        return image
    scale = limit / longest
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.LANCZOS,
    )
