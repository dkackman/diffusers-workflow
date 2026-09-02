"""The output directory: hand a generated file back to the agent, or remove
one.

Output media is served from the /outputs static mount rather than an /api
route, so these are the tools that reach outside /api. Everything returned
is downscaled or truncated first, and says so: a full-resolution render or
an unbounded text file would cost more context than the answer it is meant
to support.
"""

import base64
import io
import math

from PIL import Image

from dw_mcp.client import DwApiError, api_path

# Roughly 4MB. The cap is on the returned payload's base64 size - the bytes
# actually sent over MCP - not the raw encoded image, which is smaller by a
# factor of 3/4. Past this the payload crowds out the conversation it is
# supposed to inform.
MAX_RETURNED_BYTES = 4 * 1024 * 1024
MIN_DIMENSION = 64

# Text is cheap next to an image, but an unbounded output file is not:
# a job that logged its way to a megabyte would otherwise arrive whole.
MAX_RETURNED_CHARACTERS = 20000


def get_output_image(client, name, max_dimension=768):
    """One image from the output directory, downscaled, as base64 plus the
    sizes it went in and came out at."""

    def is_image(content_type):
        return not content_type or content_type.startswith("image/")

    body, content_type = client.get_bytes_if(api_path("outputs", name), is_image)
    if body is None:
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
    """Shrink until the base64-encoded bytes fit the ceiling. Two loops
    rather than one calculation because compressed size does not follow
    from pixel count - noise and flat colour differ by an order of
    magnitude. After the first pass, each resize starts from the previous
    pass's already-shrunk result rather than the full-resolution original -
    LANCZOS-from-LANCZOS at half size is fine, and it is never an upscale
    since the limit only ever shrinks."""
    source = image
    while True:
        sized = _fit(source, limit)
        buffer = io.BytesIO()
        sized.save(buffer, format=fmt)
        encoded = buffer.getvalue()
        base64_size = 4 * math.ceil(len(encoded) / 3)
        if base64_size <= MAX_RETURNED_BYTES or limit <= MIN_DIMENSION:
            return encoded, sized
        limit = max(MIN_DIMENSION, limit // 2)
        source = sized


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


def get_output_text(client, name, max_characters=MAX_RETURNED_CHARACTERS):
    """One text output from the output directory - the form a prompt
    enhancement and any `text/plain` result arrive in."""

    def is_text(content_type):
        kind = content_type.split(";")[0].strip().lower()
        return kind.startswith("text/") or kind == "application/json"

    body, content_type = client.get_bytes_if(api_path("outputs", name), is_text)
    if body is None:
        raise DwApiError(
            f"{name} is {content_type or 'of no declared type'}, not text - "
            "this tool returns text only. Use get_output_image for an image, "
            "or get_gallery_metadata for other media."
        )
    # A file the server labels text but that is not valid UTF-8 is damaged
    # output, and reading it that way is more use than a decoding traceback
    text = body.decode("utf-8", errors="replace")
    limit = max(1, int(max_characters))
    return {
        "name": name,
        "text": text[:limit],
        "content_type": content_type,
        "characters": len(text),
        "truncated": len(text) > limit,
    }


def delete_output(client, name):
    """Remove one file from the output directory. The gallery is the output
    directory read back, so this is where a delete belongs."""
    return client.delete_json(api_path("api", "gallery", name))
