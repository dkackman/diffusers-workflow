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
import os
import pathlib

from dw_mcp import confine
from dw_mcp.client import DwApiError, api_path, base64_size

# Roughly 4MB. The cap is on the returned payload's base64 size - the bytes
# actually sent over MCP - not the raw encoded image, which is smaller by a
# factor of 3/4. Past this the payload crowds out the conversation it is
# supposed to inform.
MAX_RETURNED_BYTES = 4 * 1024 * 1024

# Text is cheap next to an image, but an unbounded output file is not:
# a job that logged its way to a megabyte would otherwise arrive whole.
MAX_RETURNED_CHARACTERS = 20000


def get_output_image(client, name, max_dimension=768, workspace=None, crop=None):
    """One image from the output directory, or an `asset:`, downscaled, as
    base64 plus the sizes it went in and came out at.

    `crop` is `[x, y, width, height]` in the original's pixels, cut before
    the downscale, so a region of a 2K still comes back at 100% where the
    whole would be shrunk past what a seam or a small element can be read
    at. Clamped to the image; the box actually cut is reported. The server
    crops, fits and keeps the answer under MAX_RETURNED_BYTES
    (GET /api/gallery/<name>/image), so nothing is decoded here."""
    params = {
        "max_dimension": str(int(max_dimension)),
        "max_bytes": str(MAX_RETURNED_BYTES),
    }
    if crop is not None:
        params["crop"] = ",".join(str(value) for value in crop)
    body, content_type, headers = client.get_media_if(
        api_path("api", "gallery", name, "image"),
        lambda _content_type: True,
        workspace=workspace,
        params=params,
    )

    def size(header):
        value = headers.get(header)
        return [int(part) for part in value.split(",")] if value else None

    return {
        "name": name,
        "data": base64.b64encode(body).decode("ascii"),
        "mime_type": content_type.split(";")[0].strip(),
        "original_size": size("x-dw-original-size"),
        "crop": size("x-dw-crop"),
        "returned_size": size("x-dw-returned-size"),
        "bytes": len(body),
    }


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
    encoded_size = base64_size(raw_size)
    if encoded_size > MAX_RETURNED_BYTES:
        raise DwApiError(
            f"{name} is {raw_size} bytes, which would be {encoded_size} "
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
    crop=None,
):
    """Frames of a generated video as images - the way to *see* a clip when
    there is no video content type to return it as (#193, #210). One
    selector per call: `at` (moments: seconds, or "frame:N"), `count` (an
    evenly spaced contact sheet) or `seams` (True, or seam numbers from 1:
    the last frame before and the first frame after each boundary, side by
    side). `boundaries` and `names` are modifiers of `seams` only, for a
    file with no `media.shots` of its own - they do nothing alongside `at`
    or `count`, and passing either without `seams` is refused. `boundaries`
    is the list of frame indexes each shot after the first starts at - the
    running sum of the shots' `frame_count` from `get_gallery_metadata` on
    their own files - `names` the shots' names. Without `boundaries`, an
    output joined from shots uses the boundaries its run recorded
    (`get_gallery_metadata`'s `media.shots`).

    `crop` is `[x, y, width, height]` in the video's own source pixels -
    the same convention `get_output_image` uses - resolved once against
    the clip's actual dimensions and cut from every sampled frame before
    any stamping, fitting or composing, so it names the same region
    whatever `max_dimension` (or a contact sheet's own tiling) does to the
    result.

    Every tile is fitted to `max_dimension`; when the whole answer would
    still exceed MAX_RETURNED_BYTES the tiles are shrunk *together* - the
    same dimension for all, halved until they fit - rather than any being
    dropped, and `downscaled_to` says what they were shrunk to. A seam
    pair at half size is still a seam pair; a seam pair missing is a
    different answer.

    `hear`'s excerpts share that budget with the tiles: each one is
    already under `get_output_audio`'s own per-clip budget, but with up to
    MAX_FRAME_MOMENTS tiles the excerpts summed could still dwarf
    MAX_RETURNED_BYTES, so fetching stops once the tiles plus the excerpts
    so far would push past it - the remaining tiles keep their frame but
    carry an `audio_error` saying so, and `audio_truncated` is true."""
    if (boundaries or names) and not seams:
        modifiers = [n for n, v in (("boundaries", boundaries), ("names", names)) if v]
        raise DwApiError(
            f"`{'` and `'.join(modifiers)}` only appl{'y' if len(modifiers) > 1 else 'ies'} "
            "alongside `seams` - pass `seams=true` (or seam numbers) to use "
            f"{'them' if len(modifiers) > 1 else 'it'}"
        )
    if hear is not None:
        if not at:
            raise DwApiError(
                "`hear` takes seconds of soundtrack around each `at` moment - pass `at`"
            )
        if float(hear) <= 0:
            raise DwApiError("`hear` is a positive number of seconds")
    # Every selector given is sent: the server refuses anything but exactly
    # one, and its 400 reaches the caller. A list of pairs, turned into a
    # dict by the client - so no key repeats
    params = [
        ("max_dimension", str(int(max_dimension))),
        ("max_total_bytes", str(MAX_RETURNED_BYTES)),
    ]
    if at:
        params.append(("at", ",".join(str(moment) for moment in at)))
    if count:
        params.append(("count", str(int(count))))
    if seams:
        params.append(
            ("seams", "true" if seams is True else ",".join(str(s) for s in seams))
        )
    if boundaries:
        params.append(("boundaries", ",".join(str(int(b)) for b in boundaries)))
    if names:
        params.append(("names", ",".join(names)))
    if crop is not None:
        params.append(("crop", ",".join(str(v) for v in crop)))

    body = client.get_json(
        api_path("api", "gallery", name, "frames"), params=params, workspace=workspace
    )
    tiles = body.get("tiles", [])
    downscaled_to = body.get("downscaled_to")
    audio_truncated = False
    if hear is not None:
        span = float(hear)
        # The tiles already spent part of the one response budget
        audio_bytes_so_far = sum(len(tile["data"]) for tile in tiles)
        budget_exceeded = False
        for tile in tiles:
            if budget_exceeded:
                tile["audio_error"] = "skipped - would exceed the response size budget"
                audio_truncated = True
                continue
            start = max(0.0, float(tile["seconds"]) - span / 2)
            try:
                audio = get_output_audio(
                    client, name, start=start, duration=span, workspace=workspace
                )
            except DwApiError as e:
                tile["audio_error"] = str(e)
                continue
            # A per-tile cap (get_output_audio's own MAX_RETURNED_BYTES check)
            # bounds one excerpt; this is the whole response's cap, tiles
            # included, on top of - not instead of - the per-tile one.
            if audio_bytes_so_far + len(audio["data"]) > MAX_RETURNED_BYTES:
                tile["audio_error"] = "skipped - would exceed the response size budget"
                audio_truncated = True
                budget_exceeded = True
                continue
            audio_bytes_so_far += len(audio["data"])
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
        "audio_truncated": audio_truncated,
        "crop": body.get("crop"),
    }


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


def assess_output(client, name, probe=None, detail=False, workspace=None):
    """Measure a finished output or asset and say where to look (#388).

    The server runs the assessment probes on one decode, beside any GPU
    job rather than behind it; it refuses a probe outside its whitelist
    with a 400 naming the valid ones."""
    params = {}
    if probe is not None:
        params["probe"] = probe
    if detail:
        params["detail"] = "true"
    return client.get_json(
        api_path("api", "gallery", name, "assess"),
        params=params or None,
        workspace=workspace,
    )


def delete_output(client, name=None, workspace=None, job_id=None):
    """Remove one file from the output directory. The gallery is the output
    directory read back, so this is where a delete belongs.

    The run directory goes too once its last media file is gone, sidecars
    included, and a `<workflow>/<run id>` name removes a whole run - what a
    failed run, which has a manifest and nothing else, needs (#134).

    `job_id` is the other handle on a whole run: the job record carries
    the `<workflow>/<run id>` its run wrote (`run_dir`, relative to the
    output root), so the run is deleted without the caller listing the
    gallery to find the name. Exactly one of `name` / `job_id`. The server
    resolves a job's run against the root the job ran in, and refuses one
    with no run directory or still running. `workspace` pins a `name`
    delete to another workspace."""
    if (name is None) == (job_id is None):
        raise DwApiError(
            "Provide exactly one of `name` (a gallery name or a "
            "`<workflow>/<run id>` run directory) or `job_id` (the run that "
            "job wrote, deleted whole)."
        )
    if job_id is None:
        return client.delete_json(api_path("api", "gallery", name), workspace=workspace)

    # The server reads the job's run directory and the root it ran against
    return client.delete_json(api_path("api", "jobs", job_id, "run"))


# What a mounted surface says when its server names no workspace to
# confine a write to
NO_WORKSPACE_REFUSAL = (
    "This server cannot say where its workspace is, so it will not write a "
    "file for you. Use the url list_gallery reports, get_output_image / "
    "get_output_audio / get_output_text, or keep_output."
)


def _write_refusal(destination, root):
    return DwApiError(
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
    written (#113). An omitted `destination` is refused outright there
    rather than defaulting into the workspace root - a file dropped loose in
    the root has no run to delete it with and nothing names it back as an
    output (#353); pass an explicit destination inside the workspace to save
    one anyway. A stdio `dw-mcp` keeps writing anywhere the user can, and an
    omitted `destination` keeps defaulting to the current working directory,
    because there "local disk" is genuinely their own.
    """
    # The workspace a mounted surface's write is confined to; None for stdio
    roots = confine.remote_roots(
        client, workspace, ("workspace",), refusal=NO_WORKSPACE_REFUSAL
    )
    root = roots[0] if roots else None
    if destination is None:
        if root:
            raise DwApiError(
                "destination is required over a dw.serve --mcp endpoint - "
                "omitting it would drop the file loose in the workspace "
                "root, where nothing can find or delete it later. Pass an "
                "explicit destination inside the workspace, or use the url "
                "list_gallery reports, get_output_image / get_output_audio / "
                "get_output_frames for inline content, or keep_output to "
                "make it a named asset instead."
            )
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
    destination = (
        os.path.abspath(os.path.join(root, destination))
        if root and not os.path.isabs(destination)
        else os.path.abspath(destination)
    )
    if root and not confine.contains(destination, roots):
        raise _write_refusal(destination, root)

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
