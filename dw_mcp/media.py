"""The output directory: hand a generated file back to the agent, or remove
one.

Most output media is served from the /outputs static mount rather than an
/api route, so these are the tools that reach outside /api - audio is the
exception, served from the gallery's own `/audio` route so it can answer an
excerpt (#193). Everything returned
is downscaled or truncated first, and says so: a full-resolution render or
an unbounded text file would cost more context than the answer it is meant
to support.
"""

import base64
import io
import math
import os
import pathlib

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


def get_output_image(client, name, max_dimension=768, workspace=None):
    """One image from the output directory, downscaled, as base64 plus the
    sizes it went in and came out at."""

    def is_image(content_type):
        return not content_type or content_type.startswith("image/")

    body, content_type = client.get_bytes_if(
        api_path("outputs", name), is_image, workspace=workspace
    )
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


def get_output_audio(client, name, start=None, duration=None, workspace=None):
    """One soundtrack from the gallery as base64 - an audio output, or the
    track muxed into a video (#193) - for a clip short enough to fit
    MAX_RETURNED_BYTES whole, or an excerpt of one that is not. In its own
    encoding when an audio file is served whole, WAV when extracted from a
    video or excerpted; `mime_type` says which.

    Audio is not resized the way an image is - there is no downscale of a
    waveform that keeps it meaningful to listen to - so a whole clip over
    budget is refused rather than truncated (#204). The way to hear part of
    a long track is to *ask* for the part: `start` and `duration` in
    seconds, and the answer names what it cut in `excerpt`, so a slice is
    never mistaken for the whole."""

    def is_audio(content_type):
        return bool(content_type) and content_type.startswith("audio/")

    params = {}
    if start is not None:
        params["start"] = start
    if duration is not None:
        params["duration"] = duration
    body, content_type, headers = client.get_media_if(
        api_path("api", "gallery", name, "audio"),
        is_audio,
        workspace=workspace,
        params=params,
        max_bytes=MAX_RETURNED_BYTES,
    )
    if body is None and not is_audio(content_type):
        raise DwApiError(
            f"{name} answered {content_type or 'no declared type'}, not audio - "
            "this tool returns a soundtrack only. Use get_output_image for "
            "an image, or get_gallery_metadata for other media."
        )

    # Sized from the declared content-length when the client refused to
    # read the body on it, else from the body it read (an answer that
    # declared no length). The server refuses a whole track it can size
    # from the file's headers with a 413 before either, and the client
    # surfaces that detail as is; this is the same advice for the rest.
    raw_size = len(body) if body is not None else int(headers["content-length"])
    base64_size = 4 * math.ceil(raw_size / 3)
    if base64_size > MAX_RETURNED_BYTES:
        raise DwApiError(
            f"{name} is {raw_size} bytes, which would be {base64_size} "
            f"bytes base64-encoded - over the {MAX_RETURNED_BYTES} byte "
            "limit for an inline clip. Ask for an excerpt with `start` and "
            "`duration` (seconds) - get_gallery_metadata's envelope says "
            "where to look - or use download_output for the whole file."
        )

    excerpt = None
    if "x-dw-excerpt-start" in headers:
        excerpt = {
            "start": float(headers["x-dw-excerpt-start"]),
            "duration": float(headers["x-dw-excerpt-duration"]),
            "of": _float_header(headers, "x-dw-duration"),
        }
    return {
        "name": name,
        "data": base64.b64encode(body).decode("ascii"),
        "mime_type": content_type,
        "bytes": len(body),
        "duration_seconds": _float_header(headers, "x-dw-duration"),
        "excerpt": excerpt,
    }


def _float_header(headers, key):
    value = headers.get(key)
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def get_output_frames(
    client,
    name,
    at=None,
    seams=None,
    count=None,
    boundaries=None,
    names=None,
    max_dimension=512,
    hear=None,
    workspace=None,
):
    """Frames of a generated video as images - the way to *see* a clip when
    there is no video content type to return it as (#193, #210). One
    selector per call: `at` (moments: seconds, or "frame:N"), `count` (an
    evenly spaced contact sheet) or `seams` (True, or seam numbers from 1:
    the last frame before and the first frame after each boundary, side by
    side). `boundaries` is the list of frame indexes each shot after the
    first starts at - the running sum of the shots' `frame_count` from
    `get_gallery_metadata` on their own files - `names` the shots' names -
    both needed with `seams` until a joined file carries its own.

    Every tile is fitted to `max_dimension`; when the whole answer would
    still exceed MAX_RETURNED_BYTES the tiles are shrunk *together* - the
    same dimension for all, halved until they fit - rather than any being
    dropped, and `downscaled_to` says what they were shrunk to. A seam
    pair at half size is still a seam pair; a seam pair missing is a
    different answer."""
    chosen = [key for key, value in (("at", at), ("count", count), ("seams", seams)) if value]
    if len(chosen) != 1:
        raise DwApiError(
            "Pass exactly one of `at`, `count` or `seams`"
            + (f" - got {', '.join(chosen)}" if chosen else "")
        )
    if hear is not None:
        if not at:
            raise DwApiError("`hear` takes seconds of soundtrack around each `at` moment - pass `at`")
        if float(hear) <= 0:
            raise DwApiError("`hear` is a positive number of seconds")
    params = [("max_dimension", str(max(MIN_DIMENSION, int(max_dimension))))]
    # a list of pairs, turned into a dict by the client - so no key repeats
    if at:
        params.append(("at", ",".join(str(moment) for moment in at)))
    elif count:
        params.append(("count", str(int(count))))
    else:
        params.append(("seams", "true" if seams is True else ",".join(str(s) for s in seams)))
        if boundaries:
            params.append(("boundaries", ",".join(str(int(b)) for b in boundaries)))
        if names:
            params.append(("names", ",".join(names)))

    body = client.get_json(
        api_path("api", "gallery", name, "frames"), params=params, workspace=workspace
    )
    tiles, downscaled_to = _fit_tiles_within_budget(body.get("tiles", []))
    if hear is not None:
        span = float(hear)
        for tile in tiles:
            start = max(0.0, float(tile["seconds"]) - span / 2)
            try:
                audio = get_output_audio(
                    client, name, start=start, duration=span, workspace=workspace
                )
            except DwApiError as e:
                tile["audio_error"] = str(e)
                continue
            tile["audio"] = {
                "data": audio["data"],
                "mime_type": audio["mime_type"],
                "excerpt": audio["excerpt"],
            }
    return {
        "name": name,
        "frame_count": body.get("frame_count"),
        "fps": body.get("fps"),
        "tiles": tiles,
        "downscaled_to": downscaled_to,
        "hear": hear,
    }


def _fit_tiles_within_budget(tiles):
    """Shrink every tile by the same factor until their base64 sizes sum
    to MAX_RETURNED_BYTES or less. Returns (tiles, downscaled_to) with
    downscaled_to None when nothing had to shrink."""
    total = sum(len(tile["data"]) for tile in tiles)
    if total <= MAX_RETURNED_BYTES or not tiles:
        return tiles, None
    images = [Image.open(io.BytesIO(base64.b64decode(tile["data"]))) for tile in tiles]
    for image in images:
        image.load()
    limit = max(max(image.width, image.height) for image in images)
    while True:
        limit = max(MIN_DIMENSION, limit // 2)
        shrunk = []
        for tile, image in zip(tiles, images):
            sized = _fit(image, limit)
            buffer = io.BytesIO()
            sized.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            shrunk.append({**tile, "data": encoded, "width": sized.width, "height": sized.height})
        if sum(len(t["data"]) for t in shrunk) <= MAX_RETURNED_BYTES or limit <= MIN_DIMENSION:
            return shrunk, limit
        images = [Image.open(io.BytesIO(base64.b64decode(t["data"]))) for t in shrunk]


def get_output_text(
    client, name, max_characters=MAX_RETURNED_CHARACTERS, workspace=None
):
    """One text output from the output directory - the form a prompt
    enhancement and any `text/plain` result arrive in."""

    def is_text(content_type):
        kind = content_type.split(";")[0].strip().lower()
        return kind.startswith("text/") or kind == "application/json"

    body, content_type = client.get_bytes_if(
        api_path("outputs", name), is_text, workspace=workspace
    )
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


def delete_output(client, name, workspace=None):
    """Remove one file from the output directory. The gallery is the output
    directory read back, so this is where a delete belongs.

    The run directory goes too once its last media file is gone, sidecars
    included, and a `<workflow>/<run id>` name removes a whole run - what a
    failed run, which has a manifest and nothing else, needs (#134)."""
    return client.delete_json(api_path("api", "gallery", name), workspace=workspace)


def _remote_root(client):
    """The workspace a remote write is confined to, or None when local.

    Only the mounted MCP surface is remote: there the tool runs inside
    dw.serve, so the path a caller names is a path on the operator's box
    rather than on its own machine. A stdio `dw-mcp` returns None and keeps
    writing wherever the user can.
    """
    if not getattr(client, "mounted", False):
        return None

    directories = (client.get_json("/api/server").get("directories")) or {}
    root = directories.get("workspace")
    if not root:
        raise DwApiError(
            "This server cannot say where its workspace is, so it will not "
            "write a file for you. Use the url list_gallery reports, "
            "get_output_image / get_output_audio / get_output_text, or "
            "keep_output."
        )
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(root))))


def _confine(destination, root):
    """Refuse a destination outside `root`, on the resolved real path.

    Containment is on realpath, not on a substring: an absolute path or a
    '~' needs no '..' to reach anywhere the server process can write (#113),
    and a symlink inside the workspace would otherwise carry the write out.
    """
    # realpath of the nearest existing ancestor: the file itself usually does
    # not exist yet, and realpath of a missing path leaves symlinks in its
    # existing prefix unresolved on some platforms
    probe = destination
    while not os.path.exists(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    resolved = os.path.join(
        os.path.realpath(probe), os.path.relpath(destination, probe)
    )
    resolved = os.path.normpath(resolved)
    if resolved != root and not resolved.startswith(root + os.sep):
        raise DwApiError(
            f"Refusing to write {destination} - this MCP endpoint is served "
            f"by dw.serve, so the file would land on the server, where a "
            f"destination is confined to the workspace ({root}). Pass a "
            f"relative destination, or - to see the file where you are - use "
            f"the url list_gallery reports, get_output_image / "
            f"get_output_audio / get_output_text for inline content, or "
            f"keep_output to make it an asset for a later workflow."
        )


def download_output(client, name, destination=None, overwrite=False, workspace=None):
    """Fetch one output file and save it to local disk, for an agent that
    wants the artifact itself rather than a description of it.

    Unlike get_output_image/get_output_audio/get_output_text, this accepts
    any content type and returns nothing to the conversation but a manifest
    of where the file landed - the point is a file on disk, not a payload
    in context. It is also the one tool in this package that writes a
    local file, and the body is streamed to disk in chunks rather than
    buffered whole, since it exists for files (large videos) the inline
    tools can't return.

    `destination` may be a full file path, a directory (the output's own
    basename is used inside it), or omitted (saved to the current working
    directory under its own basename). '~' expands to the user's home
    directory. Missing parent directories are created. A `destination`
    containing a '..' path segment is refused. An existing file at the
    resolved path is left alone unless `overwrite=True`.

    Over a `dw.serve --mcp` endpoint the file lands on the *server*, not on
    the calling agent's machine, so there the destination is confined to that
    workspace: an absolute or '~' path outside it is refused rather than
    written (#113). A stdio `dw-mcp` keeps writing anywhere the user can,
    because there "local disk" is genuinely their own.
    """
    if destination is None:
        destination = os.path.basename(name)
    destination = os.path.expanduser(destination)
    if ".." in pathlib.PurePath(destination).parts:
        raise DwApiError(
            f"destination {destination!r} contains a '..' path segment, "
            "which is refused."
        )
    if os.path.isdir(destination) or destination.endswith(os.sep):
        destination = os.path.join(destination, os.path.basename(name))
    # A relative destination is joined onto whichever directory is "here" for
    # this transport: the caller's own working directory for stdio, the
    # server's workspace when the tool runs inside dw.serve - where the
    # process's cwd is an implementation detail the caller never chose
    root = _remote_root(client)
    destination = (
        os.path.abspath(os.path.join(root, destination))
        if root and not os.path.isabs(destination)
        else os.path.abspath(destination)
    )
    if root:
        _confine(destination, root)

    if os.path.exists(destination) and not overwrite:
        raise DwApiError(
            f"{destination} already exists. Pass overwrite=True to replace it."
        )

    parent = os.path.dirname(destination)
    try:
        if parent:
            os.makedirs(parent, exist_ok=True)
        content_type, bytes_written = client.stream_to_file(
            api_path("outputs", name), destination, workspace=workspace
        )
    except OSError as e:
        # A client-side path handed to a `dw.serve --mcp` endpoint lands
        # here: the write happens on the server, so a permission or
        # missing-volume error is the surest sign the agent is on another
        # machine. Say so, rather than letting the OSError surface as an
        # anonymous "Error executing tool".
        raise DwApiError(
            f"Could not write {destination} on the machine running the MCP "
            f"server ({e.strerror or e}). This tool saves on that machine - "
            "over a dw.serve --mcp endpoint that is the GPU box, not where "
            "you are. To see the file from here use the url list_gallery "
            "reports, get_output_image / get_output_audio / get_output_text "
            "for inline content, or keep_output to make it an asset for a "
            "later workflow."
        ) from e

    return {
        "name": name,
        "saved_to": destination,
        "content_type": content_type,
        "bytes": bytes_written,
    }
