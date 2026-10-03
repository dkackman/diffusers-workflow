"""The media routes under `/api/gallery/{name:path}`: metadata, assessment,
audio, frames, thumbnail and download of one output or asset.

Registered before the gallery router, so these GETs sit ahead of the greedy
`DELETE /api/gallery/{name:path}`, as they always have. `video_shape` is looked
up here, at call time.
"""

import base64
import io
import mimetypes
import os
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, Response

from ..api_models import (
    GalleryMetadata,
)
from ...assets import is_asset_reference
from ...media import (
    MAX_INLINE_AUDIO_BYTES,
    NoSoundtrack,
    audio_shape,
    extract_audio,
    media_duration,
    probe_media,
    projected_wav_base64_size,
    video_shape,
)
from ...media_frames import (
    contact_sheet,
    frames_at,
    resolve_crop_box,
    seam_tiles,
)
from ...writers import read_embedded_metadata
from ...runs import kept_provenance, recorded_shots, shots_beside
from ...security import MAX_DECODE_PIXELS
from ...workspace import Workspace
from ..assess import assess, level_findings, unknown_probe
from ..deps import selected_workspace
from ..http_security import query_token_ok
from ..inline_media import (
    encode_within_budget,
    fit_longest,
    fit_tiles_within_budget,
    open_bounded,
    png_bytes,
)
from ..outputs import (
    MEDIA_KINDS,
    asset_file,
    job_provenance,
    resolve_output_file,
    strip_output_prefix,
)

router = APIRouter()

# Longest side of an on-demand gallery thumbnail, in pixels
GALLERY_THUMBNAIL_MAX_DIM = 320


@router.get(
    "/api/gallery/{name:path}/metadata",
    response_model=GalleryMetadata,
    response_model_exclude_unset=True,
)
def gallery_metadata(
    request: Request,
    name: str,
    envelope: bool = False,
    ws: Workspace = Depends(selected_workspace),
):
    """Generation metadata embedded in a saved image ('workflow' inside
    it is the full definition the editor can reopen), plus the job that
    produced the file when history remembers one, plus - for audio and
    video - what the file itself holds: duration, format and level,
    which is how an agent that cannot listen checks a track. Only an
    image embeds 'metadata' this way - it is always null for audio and
    video, since neither format has a slot this writer uses; recover
    the recipe from 'job' (GET /api/jobs/{id}/workflow) when one is
    known, or from nothing when it isn't (a kept asset has no job).

    `envelope=true` adds the soundtrack's level second by second, which
    is what says *where* in a track something is - whether a shot is
    still voiced at its last frame, how deep the hole at a seam goes.
    Opt-in: a ten-minute track is 600 numbers, and the default call has
    to stay small.

    `name` may also be an 'asset:' reference, and then it is the input
    asset of that name that is described rather than an output (#127).
    The numbers here - duration, frame count, fps, sample rate - are
    what decide whether a call will work at all, and for a file the
    caller is about to *consume* they were previously unobtainable:
    the only way to read a wav's length was to run a job that copied it
    into the output directory. `job` is null for an asset with no
    recorded provenance, and `source` says which of the two roots
    answered.

    A kept asset (`keep_output`) is not always provenance-blind: when
    the file it was kept from still had a job in history, `keep_output`
    recorded `{job, run_id, version}` beside it, and this route reads it
    back the same way it reads a kept file's shots (#556). `run_id`/
    `version` are that source run's own - `job.id` is still the one to
    pass `get_job_workflow`, since a pruned run leaves no `workflow.json`
    of its own to reopen."""
    run_id, version = "", None
    name = strip_output_prefix(name)
    if is_asset_reference(name):
        path = asset_file(request.app.state, name, ws)
        source, job = "asset", None
        kept = kept_provenance(path)
        if kept:
            job = kept.get("job")
            run_id = kept.get("run_id") or ""
            version = kept.get("version")
    else:
        path = resolve_output_file(request.app.state, name, ws.outputs)
        source = "output"
        # Which run wrote it, and that run's ordinal - the same 'v4' the
        # listing reports. After "look at version 3" this is the next
        # call, so it confirms the right file was reached rather than
        # sending the caller back to the listing
        job, run_id, version = job_provenance(request.app.state, name, ws)
    metadata = read_embedded_metadata(path)
    extension = os.path.splitext(path)[1].lower()
    media = (
        probe_media(path, envelope=envelope)
        if MEDIA_KINDS.get(extension) in ("audio", "video")
        else None
    )
    if media is not None and source == "output":
        # Where each shot of a joined video sits, as the run that wrote
        # it recorded (dw/shots.py) - null for a file not joined from shots
        media["shots"] = recorded_shots(ws.outputs, name)
    elif media is not None and source == "asset":
        # keep_output carries the source run's shots into a sidecar
        # manifest beside the asset (#393); a file kept before that fix,
        # or never joined from shots, has none
        media["shots"] = shots_beside(path)
    return {
        "name": name,
        "source": source,
        "metadata": metadata,
        "job": job,
        "run_id": run_id,
        "version": version,
        "media": media,
        "findings": level_findings(media),
    }


@router.get("/api/gallery/{name:path}/assess")
def gallery_assess(
    request: Request,
    name: str,
    probe: Optional[str] = None,
    detail: bool = False,
    ws: Workspace = Depends(selected_workspace),
):
    """Measure a finished cut and say where to look (#388): every
    assessment probe that applies to the file, run here in the server
    process on one decode - a sync route, so it runs beside a GPU job
    rather than queueing behind it. Findings are places to look, not
    verdicts; nothing acts on one (dw/assessment_rules.py).

    The default answer merges the probes' `findings`, `rules_applied`
    and `rules_skipped`, and names each probe the file cannot feed in
    `not_applicable` (a still, no soundtrack, no recorded shots);
    `detail=true` adds each probe's full answer under `probes`.
    `probe` names one - analyze_shots, analyze_seams or
    analyze_sync_drift - and answers with its full body. It is checked
    before the name is resolved. `name` may be an `asset:` reference,
    and then the shots are the ones keep_output carried beside it."""
    rejected = unknown_probe(probe)
    if rejected:
        raise HTTPException(status_code=400, detail=rejected)
    name = strip_output_prefix(name)
    if is_asset_reference(name):
        path = asset_file(request.app.state, name, ws)
        source, shots = "asset", shots_beside(path)
    else:
        path = resolve_output_file(request.app.state, name, ws.outputs)
        source, shots = "output", recorded_shots(ws.outputs, name)
    kind = MEDIA_KINDS.get(os.path.splitext(path)[1].lower())
    try:
        body = assess(path, kind, shots, probe=probe, detail=detail)
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=422, detail=f"{name} could not be read: {e}")
    return {"name": name, "source": source, "kind": kind, **body}


@router.get("/api/gallery/{name:path}/audio")
def gallery_audio(
    request: Request,
    name: str,
    start: Optional[float] = None,
    duration: Optional[float] = None,
    ws: Workspace = Depends(selected_workspace),
):
    """The soundtrack of an output or asset, as WAV - a muxed video's
    track, which `get_output_audio` used to refuse outright, or an
    excerpt (`start` + `duration`, seconds) of a track too long to send
    whole (#193). An excerpt names itself in the response headers
    (`X-DW-Excerpt-Start`, `X-DW-Excerpt-Duration`) beside the whole
    track's `X-DW-Duration` - omitted only when a container carries no
    duration in its own header - so a cut is never silent (#204).

    An audio-only file asked for whole is served as its own bytes in its
    own encoding - there is nothing to extract, and a transcode would
    change what the agent hears."""
    name = strip_output_prefix(name)
    if is_asset_reference(name):
        path = asset_file(request.app.state, name, ws)
    else:
        path = resolve_output_file(request.app.state, name, ws.outputs)
    extension = os.path.splitext(path)[1].lower()
    kind = MEDIA_KINDS.get(extension)
    if kind not in ("audio", "video"):
        raise HTTPException(status_code=404, detail=f"{name} carries no soundtrack")

    excerpt = start is not None or duration is not None
    if kind == "audio" and not excerpt:
        # The container's own header has the duration - reading it does
        # not decode a single frame, unlike probe_media (which measures
        # level and would pay for a full decode just for one number).
        headers = {}
        duration_seconds = media_duration(path)
        if duration_seconds is not None:
            headers["X-DW-Duration"] = str(duration_seconds)
        if extension == ".wav":
            # mimetypes says audio/x-wav on macOS, audio/vnd.wave from
            # Python 3.14's builtin table on a box with no system mime
            # file; an extract says audio/wav, and a whole WAV must not
            # read as a different kind
            media_type = "audio/wav"
        else:
            media_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type, headers=headers)

    # A track over the cap is refused at the header, not after it has
    # been decoded and shipped: the MCP side would refuse the same bytes
    # for the same reason, having paid for all of them. An excerpt is
    # sized by its own span - `duration`, clipped to what is left of the
    # track after `start` - so a whole-length "excerpt" is not a way
    # around the gate.
    shape = audio_shape(path)
    if shape is not None and shape["duration_seconds"] is not None:
        span = shape["duration_seconds"]
        if excerpt and duration is not None:
            span = max(0.0, min(float(duration), span - float(start or 0.0)))
        projected = projected_wav_base64_size({**shape, "duration_seconds": span})
        if projected > MAX_INLINE_AUDIO_BYTES:
            what = (
                f"a {span:.1f}s excerpt of {name}"
                if excerpt
                else f"{name}'s whole soundtrack"
            )
            advice = (
                "Ask for a shorter `duration`"
                if excerpt
                else "Ask for an excerpt with `start` and `duration` (seconds)"
            )
            raise HTTPException(
                status_code=413,
                detail=(
                    f"{what} would be {projected} bytes base64-encoded as WAV "
                    f"- over the {MAX_INLINE_AUDIO_BYTES} byte limit for an "
                    f"inline clip. {advice}, or download the file."
                ),
            )

    try:
        data, info = extract_audio(path, start=start, duration=duration)
    except NoSoundtrack:
        raise HTTPException(status_code=404, detail=f"{name} carries no soundtrack")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    headers = {"X-DW-Duration": str(info["of_seconds"])}
    if info["excerpt"]:
        headers["X-DW-Excerpt-Start"] = str(info["start"])
        headers["X-DW-Excerpt-Duration"] = str(info["duration_seconds"])
    return Response(content=data, media_type="audio/wav", headers=headers)


FRAME_MIN_DIMENSION = 64
# The most moments one `at` may name: each is a seek, a decode and a
# PNG encode in the server process, and a contact sheet is the shape
# for seeing more of a clip at once
MAX_FRAME_MOMENTS = 32


def _require_one_selector(at, count, seams):
    """400 unless exactly one of `at`, `count` and `seams` was passed."""
    chosen = [
        key for key, value in (("at", at), ("count", count), ("seams", seams)) if value
    ]
    if len(chosen) != 1:
        raise HTTPException(
            status_code=400,
            detail="Pass exactly one of `at`, `count` or `seams`"
            + (f" - got {', '.join(chosen)}" if chosen else ""),
        )


def _moments_named(at):
    """The moments an `at` list names - seconds, or "frame:N" - capped at
    MAX_FRAME_MOMENTS."""
    moments = [
        m.strip() if m.strip().startswith("frame:") else float(m)
        for m in at.split(",")
        if m.strip()
    ]
    if len(moments) > MAX_FRAME_MOMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"`at` names {len(moments)} moments; the most is "
            f"{MAX_FRAME_MOMENTS} - ask for a contact sheet (`count`) "
            "to see more of the clip at once",
        )
    return moments


def _seams_requested(path, name, ws, seams, boundaries, names):
    """Where each shot after the first starts, the shots' names, and which
    seams were asked for (None for every one)."""
    recorded = (
        None
        if boundaries
        else shots_beside(path)
        if is_asset_reference(name)
        else recorded_shots(ws.outputs, name)
    )
    if recorded:
        # The file's own seams, from its run's manifest (or, for
        # a linked asset, the sidecar `record_kept_shots` wrote
        # beside it)
        starts = [shot["start_frame"] for shot in recorded[1:]]
        shot_names = (
            [n.strip() for n in names.split(",")]
            if names
            else [shot["name"] for shot in recorded]
        )
    elif not boundaries:
        raise HTTPException(
            status_code=400,
            detail="`seams` needs `boundaries`: the frame index each "
            "shot after the first starts at - this file's run "
            "recorded no shots for it",
        )
    else:
        starts = [int(b) for b in boundaries.split(",") if b.strip()]
        shot_names = [n.strip() for n in names.split(",")] if names else None
    wanted = (
        None
        if seams.lower() == "true"
        else {int(s) for s in seams.split(",") if s.strip()}
    )
    if wanted is not None and not wanted:
        raise HTTPException(
            status_code=400,
            detail="`seams` names no seam - pass `true` for every seam, "
            "or seam numbers from 1",
        )
    return starts, shot_names, wanted


def _select_tiles(path, name, ws, shape, crop_box, sub_tile_width, selectors):
    """The tiles the one selector asks for: `at` moments, a `count` contact
    sheet, or the `seams`. `selectors` is (at, count, seams, boundaries,
    names) as the route received them."""
    at, count, seams, boundaries, names = selectors
    if at:
        return frames_at(path, _moments_named(at), shape=shape, crop_box=crop_box)
    if count:
        return [
            contact_sheet(
                path, count, tile_width=sub_tile_width, shape=shape, crop_box=crop_box
            )
        ]
    starts, shot_names, wanted = _seams_requested(
        path, name, ws, seams, boundaries, names
    )
    return seam_tiles(
        path,
        starts,
        names=shot_names,
        tile_width=sub_tile_width,
        shape=shape,
        wanted=wanted,
        crop_box=crop_box,
    )


def _crop_rectangle(crop_box):
    """`crop_box` (x0, y0, x1, y1) as the `x,y,width,height` the caller wrote."""
    if not crop_box:
        return None
    return [
        crop_box[0],
        crop_box[1],
        crop_box[2] - crop_box[0],
        crop_box[3] - crop_box[1],
    ]


@router.get("/api/gallery/{name:path}/frames")
def gallery_frames(
    request: Request,
    name: str,
    at: Optional[str] = None,
    count: Optional[int] = None,
    seams: Optional[str] = None,
    boundaries: Optional[str] = None,
    names: Optional[str] = None,
    max_dimension: int = 512,
    crop: Optional[str] = None,
    max_total_bytes: Optional[int] = None,
    ws: Workspace = Depends(selected_workspace),
):
    """Frames of a video output or asset, as PNG tiles - the way an
    agent with no video content type sees what a run made (#193).
    Exactly one selector: `at` (a comma list of seconds or "frame:N"),
    `count` (an evenly spaced contact sheet, `frame_grid` without a
    workflow), or `seams` ("true", or a comma list of 1-based seam
    numbers) for the last frame before and first frame after each
    boundary, side by side. `boundaries` is the comma list of frame
    indexes each shot after the first starts at, and `names` the
    shots' names. Without `boundaries`, an output's seams are the shots
    its run's manifest recorded for it (a `concat_videos`,
    `dissolve_videos` or chained step), named as recorded unless `names`
    is given; a linked asset (`keep_output(shared=true)`) uses the same
    shots `get_gallery_metadata`'s `media.shots` reports for it, from the
    sidecar manifest kept beside it. A file with none recorded still
    needs `boundaries`.
    Tiles are downscaled to `max_dimension` on their longest side.
    `crop` is `x,y,width,height` in the video's own source pixels
    (`video_shape`'s `width`/`height`) - resolved once and cut from
    every sampled frame before any stamping, fitting or composing, so
    it names the same region whatever `max_dimension` downscales the
    result to. `max_total_bytes` is a budget on the tiles' summed base64
    size: over it, every tile shrinks to one shared size (never below
    FRAME_MIN_DIMENSION) rather than any being dropped, and
    `downscaled_to` says the side they came out at (null when nothing
    had to shrink)."""
    state = request.app.state
    name = strip_output_prefix(name)
    if is_asset_reference(name):
        path = asset_file(state, name, ws)
    else:
        path = resolve_output_file(state, name, ws.outputs)
    if MEDIA_KINDS.get(os.path.splitext(path)[1].lower()) != "video":
        raise HTTPException(status_code=404, detail=f"{name} is not a video")

    _require_one_selector(at, count, seams)
    # A floor on each *sub-tile* of a composite (contact sheet / seam
    # pair) - a caller asking for a small max_dimension still gets a
    # legible grid, which is then fit to max_dimension as a whole below.
    sub_tile_width = max(FRAME_MIN_DIMENSION, int(max_dimension))
    limit = max(1, int(max_dimension))

    try:
        # Computed once and threaded through every selector: each of
        # frames_at/contact_sheet/seam_tiles would otherwise call
        # video_shape itself, opening the container (and, lacking a
        # header frame count, decoding it whole to count) a second time
        # just to answer the same frame_count/fps/width/height (#193).
        shape = video_shape(path)
        crop_box = (
            resolve_crop_box(
                [c.strip() for c in crop.split(",")], shape["width"], shape["height"]
            )
            if crop
            else None
        )
        tiles = _select_tiles(
            path,
            name,
            ws,
            shape,
            crop_box,
            sub_tile_width,
            (at, count, seams, boundaries, names),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    images = [fit_longest(tile["image"], limit) for tile in tiles]
    downscaled_to = None
    if max_total_bytes is not None:
        images, downscaled_to = fit_tiles_within_budget(
            images, limit, max_total_bytes, floor=FRAME_MIN_DIMENSION
        )
    return {
        "name": name,
        **shape,
        "tiles": [_encoded_tile(tile, image) for tile, image in zip(tiles, images)],
        "crop": _crop_rectangle(crop_box),
        "downscaled_to": downscaled_to,
    }


def _encoded_tile(tile, image):
    """A tile as the frames answer carries it: PNG, base64, and the size
    `image` - the tile's picture, already fitted - came out at."""
    encoded = {key: value for key, value in tile.items() if key != "image"}
    encoded.update(
        {
            "data": base64.b64encode(png_bytes(image)).decode("ascii"),
            "mime_type": "image/png",
            "width": image.width,
            "height": image.height,
        }
    )
    return encoded


@router.get("/api/gallery/{name:path}/image")
@query_token_ok
def gallery_image(
    request: Request,
    name: str,
    max_dimension: int = 768,
    crop: Optional[str] = None,
    max_bytes: Optional[int] = None,
    format: Literal["auto", "png", "jpeg"] = "auto",
    ws: Workspace = Depends(selected_workspace),
):
    """An image output or asset sized for an inline answer: `crop`
    (`x,y,width,height` in the image's own pixels, clamped to it) first,
    then fitted to `max_dimension` on its longest side, then - with
    `max_bytes`, a budget on the base64 size - halved until it fits. The
    format follows the source (JPEG stays JPEG, anything else is PNG)
    unless `format` names one. Headers say what came back:
    X-DW-Original-Size, X-DW-Returned-Size, and when they apply
    X-DW-Crop and X-DW-Downscaled-To (the budget shrank it)."""
    state = request.app.state
    name = strip_output_prefix(name)
    if is_asset_reference(name):
        path = asset_file(state, name, ws)
    else:
        path = resolve_output_file(state, name, ws.outputs)
    if MEDIA_KINDS.get(os.path.splitext(path)[1].lower()) != "image":
        raise HTTPException(status_code=404, detail=f"{name} is not an image")
    image = open_bounded(path, name)
    original = image.size
    headers = {"X-DW-Original-Size": f"{original[0]},{original[1]}"}
    if crop:
        try:
            box = resolve_crop_box([c.strip() for c in crop.split(",")], *original)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        image = image.crop(box)
        headers["X-DW-Crop"] = _crop_header(box)
    fmt = (
        format.upper()
        if format != "auto"
        else (
            "JPEG"
            if image.format == "JPEG" or path.lower().endswith((".jpg", ".jpeg"))
            else "PNG"
        )
    )
    limit = max(1, int(max_dimension))
    fitted_size = fit_longest(image, limit).size
    data, sized = encode_within_budget(
        image, limit, fmt, max_bytes, floor=FRAME_MIN_DIMENSION
    )
    if sized.size != fitted_size:
        headers["X-DW-Downscaled-To"] = str(max(sized.size))
    headers["X-DW-Returned-Size"] = f"{sized.width},{sized.height}"
    return Response(
        content=data,
        media_type="image/jpeg" if fmt == "JPEG" else "image/png",
        headers=headers,
    )


def _crop_header(box):
    left, upper, right, lower = box
    return f"{left},{upper},{right - left},{lower - upper}"


@router.get("/api/gallery/{name:path}/thumbnail")
@query_token_ok
def gallery_thumbnail(
    name: str, request: Request, ws: Workspace = Depends(selected_workspace)
):
    """A small JPEG rendition of an image output, for the grid - the
    full-resolution file is only fetched for the detail/lightbox view.
    Generated on demand rather than cached to disk, so it never grows
    the output directory the gallery itself scans."""
    path = resolve_output_file(request.app.state, name, ws.outputs)
    extension = os.path.splitext(path)[1].lower()
    if MEDIA_KINDS.get(extension) != "image":
        raise HTTPException(
            status_code=404, detail="Thumbnails are only generated for images"
        )
    # The file's mtime and size are the validator: the grid re-requests
    # every visible thumbnail on each visit, and a 304 skips the
    # decode/resize/encode; a rerun that overwrites the file changes it
    stat = os.stat(path)
    etag = f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'
    cache_headers = {"ETag": etag, "Cache-Control": "private, no-cache"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=cache_headers)
    try:
        from PIL import Image

        with Image.open(path) as image:
            if image.width * image.height > MAX_DECODE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail=f"{name} is {image.width}x{image.height}, more "
                    f"than the {MAX_DECODE_PIXELS:,} pixels a thumbnail "
                    "is decoded from",
                )
            # shrink first (JPEGs decode at reduced size via draft), then
            # convert - converting a full-resolution image only to
            # discard most of it is the expensive order
            image.draft("RGB", (GALLERY_THUMBNAIL_MAX_DIM, GALLERY_THUMBNAIL_MAX_DIM))
            image.thumbnail((GALLERY_THUMBNAIL_MAX_DIM, GALLERY_THUMBNAIL_MAX_DIM))
            image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=80)
    except Image.DecompressionBombError as e:
        # Pillow's own refusal, on open, of a header past twice its limit
        raise HTTPException(status_code=413, detail=str(e))
    except (OSError, ValueError) as e:
        # what PIL raises for an unreadable or corrupt file
        raise HTTPException(
            status_code=500, detail=f"Could not generate thumbnail: {e}"
        )
    return Response(
        content=buffer.getvalue(), media_type="image/jpeg", headers=cache_headers
    )


@router.get("/api/gallery/{name:path}/download")
@query_token_ok
def download_output(
    request: Request, name: str, ws: Workspace = Depends(selected_workspace)
):
    """Serve one output file as a forced download rather than an inline view."""
    path = resolve_output_file(request.app.state, name, ws.outputs)
    return FileResponse(path, filename=os.path.basename(name))
