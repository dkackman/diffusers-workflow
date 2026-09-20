"""Frames out of a video file by seeking to them - a moment, an evenly
spaced contact sheet, or the frame pair either side of a seam - without
decoding the clip whole. The server process runs this per request; a
four-minute 1080p clip materialised as PIL frames is tens of gigabytes,
so nothing here ever holds more than the frames it returns, each already
fitted to its tile where a caller asked for many (#193).
"""

import logging
import math

import av
import numpy
from PIL import Image

from .tasks.video_utils import (
    _compose_grid,
    _default_columns,
    _evenly_spaced_indices,
    _format_timestamp,
    _grid_tile,
)

logger = logging.getLogger("dw")

# Each cell of a contact sheet and each side of a seam pair is a seek, a
# decode and a resize in the server process; a sheet past 64 cells is
# unreadable anyway, and `at` has its own cap in the route
# (MAX_FRAME_MOMENTS). Both are ValueErrors, so the route answers 400.
MAX_CONTACT_SHEET_FRAMES = 64
MAX_SEAMS = 32


def video_shape(path):
    """Frame count, fps and size, from the container's own headers where
    they are written and by counting otherwise."""
    with av.open(path) as container:
        if not container.streams.video:
            raise ValueError(f"{path} has no video stream")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        count = int(stream.frames) if stream.frames else None
        if count is None:
            count = sum(1 for _ in container.decode(stream))
        return {
            "frame_count": count,
            "fps": fps,
            "width": int(stream.width),
            "height": int(stream.height),
        }


def frames_at(path, moments, shape=None):
    """One tile per moment - a float in seconds or "frame:N" - in the order
    asked for. Each tile is {label, frame, seconds, image}.

    `shape` reuses an already-computed `video_shape(path)` - a caller such
    as the gallery route that also reports the clip's own frame_count/fps
    would otherwise pay for `video_shape`'s container open (and, lacking a
    header frame count, a full decode) a second time for the same answer."""
    shape = shape if shape is not None else video_shape(path)
    indexes = [_moment_to_index(moment, shape) for moment in moments]
    images = _read_frames(path, indexes)
    return [_tile(index, images[index], shape) for index in indexes]


def contact_sheet(path, count, tile_width=320, shape=None):
    """N evenly spaced frames (first and last included) tiled into one
    image - frame_grid without a workflow. `shape` reuses an
    already-computed `video_shape(path)`; see `frames_at`."""
    if int(count) < 1:
        raise ValueError("count must be at least 1")
    shape = shape if shape is not None else video_shape(path)
    # Clamp to the clip's own length before checking the cap: a count that
    # would only ever produce a handful of cells (a short clip) should not
    # be refused for the raw number the caller asked for.
    count = min(int(count), shape["frame_count"])
    if count > MAX_CONTACT_SHEET_FRAMES:
        raise ValueError(
            f"count {count} is more than a contact sheet holds "
            f"({MAX_CONTACT_SHEET_FRAMES}); ask for a smaller one, or `at` for moments"
        )
    indexes = _evenly_spaced_indices(shape["frame_count"], count)
    images = _read_frames(
        path,
        indexes,
        fit=lambda image, index: _stamped_tile(image, index, shape["fps"], tile_width),
    )
    tiles = [images[index] for index in indexes]
    grid = _compose_grid(tiles, _default_columns(len(tiles)))
    return {
        "label": f"contact sheet, {len(tiles)} frames",
        "frame": indexes[0],
        "seconds": _seconds(indexes[0], shape),
        "image": grid,
        "frames": list(indexes),
    }


def seam_tiles(path, boundaries, names=None, tile_width=320, shape=None, wanted=None):
    """For each boundary (the frame index a shot *starts* at), the last
    frame before it and the first frame at it, side by side - the seam and
    continuity evidence in one image. Seam i sits between shot i and shot
    i+1; `names` names the shots, "shot 1".. by default.

    `boundaries` and `names` are validated in full regardless of `wanted` -
    they describe the whole cut, and a seam's label names the shots either
    side of it by position in that full list. `wanted` (a set of 1-based
    seam numbers, or None for all) then limits which seams are actually
    decoded and composed: a caller after seam 2 of twelve should not pay to
    decode and tile the other eleven. `shape` reuses an already-computed
    `video_shape(path)`; see `frames_at`."""
    shape = shape if shape is not None else video_shape(path)
    total = shape["frame_count"]
    for boundary in boundaries:
        if not 1 <= int(boundary) <= total - 1:
            raise ValueError(
                f"boundary {boundary} is not inside the clip (1..{total - 1})"
            )
    boundaries = [int(b) for b in boundaries]
    names = list(names or [f"shot {n + 1}" for n in range(len(boundaries) + 1)])
    if len(names) != len(boundaries) + 1:
        raise ValueError(
            f"{len(boundaries)} boundaries make {len(boundaries) + 1} shots, "
            f"but {len(names)} names were given"
        )
    if wanted is not None:
        off = sorted(int(s) for s in wanted if not 1 <= int(s) <= len(boundaries))
        if off:
            raise ValueError(
                f"seam {', '.join(str(s) for s in off)} is not in this cut - "
                f"{len(boundaries)} boundaries make seams 1..{len(boundaries)}"
            )
    chosen = [
        (seam, boundary)
        for seam, boundary in enumerate(boundaries, start=1)
        if wanted is None or seam in wanted
    ]
    if len(chosen) > MAX_SEAMS:
        raise ValueError(
            f"{len(chosen)} seams is more than one call serves ({MAX_SEAMS}); "
            "name the seams wanted (`seams=1,2,...`)"
        )
    frame_indexes = sorted({b - 1 for _, b in chosen} | {b for _, b in chosen})
    images = _read_frames(
        path, frame_indexes, fit=lambda image, _index: _fit_width(image, tile_width)
    )
    tiles = []
    for seam, boundary in chosen:
        before = images[boundary - 1]
        after = images[boundary]
        pair = _compose_grid([before, after], 2)
        difference = float(
            numpy.abs(
                numpy.asarray(before, dtype=numpy.int16) - numpy.asarray(after, dtype=numpy.int16)
            ).mean()
        )
        tiles.append(
            {
                "label": f"seam {seam}: {names[seam - 1]} | {names[seam]}",
                "frame": boundary,
                "seconds": _seconds(boundary, shape),
                "difference": round(difference, 2),
                "image": pair,
            }
        )
    return tiles


def _moment_to_index(moment, shape):
    total = shape["frame_count"]
    if isinstance(moment, str) and moment.startswith("frame:"):
        index = int(moment[len("frame:") :])
    else:
        fps = shape["fps"]
        if fps is None:
            raise ValueError("this clip has no frame rate, so name a frame: 'frame:N'")
        seconds = float(moment)
        if not math.isfinite(seconds):
            raise ValueError(f"{moment!r} is not a moment in seconds")
        index = int(round(seconds * fps))
    if index < 0:
        index += total
    if not 0 <= index < total:
        raise ValueError(f"{moment!r} is past the end of a {total}-frame clip")
    return index


def _seconds(index, shape):
    return index / shape["fps"] if shape["fps"] else float(index)


def _tile(index, image, shape):
    fps = shape["fps"]
    stamp = _format_timestamp(index, fps) if fps else f"#{index}"
    return {
        "label": f"{stamp} (frame {index})",
        "frame": index,
        "seconds": _seconds(index, shape),
        "image": image,
    }


def _stamped_tile(image, index, fps, tile_width):
    """A contact-sheet cell: fitted like `_fit_width` (never upscaled) and
    stamped with its timestamp by `frame_grid`'s own tile maker, so the
    sheet says which cell is which without the text part."""
    return _grid_tile(image, index, fps, min(int(tile_width), image.width), label=True)


def _fit_width(image, tile_width):
    # never upscaled: a tile wider than its source is a blurred enlargement
    # that costs bytes and shows nothing the source holds
    tile_width = min(int(tile_width), image.width)
    height = max(1, round(image.height * tile_width / image.width))
    return image.resize((tile_width, height), Image.LANCZOS).convert("RGB")


def _read_frames(path, indexes, fit=None):
    """The frames at these indexes, as {index: PIL image}, in one forward
    pass that seeks to the keyframe before each wanted frame rather than
    decoding from the top. Decodes are dropped as soon as they are past.

    `fit(image, index)`, when given, is applied to each frame as it is
    decoded, so a caller tiling many frames never holds one at source size.

    This assumes a constant frame rate, which every file this engine writes
    has (`encode_video` / `export_to_video` write a fixed `fps`): a
    `backward=True` seek lands on the keyframe at or before the target
    timestamp, and `frame.pts * stream.time_base` converts that keyframe's
    own presentation time back to an exact frame index (`round(seconds *
    fps)`) - verified against PyAV 18.1's actual seek landings (a
    single-keyframe short clip, where every seek lands on frame 0; a `g=10`
    multi-keyframe clip, where a seek to frame 95 lands exactly on frame 90;
    and a clip whose packets carry a 5-frame pts offset - an edit list or a
    non-zero start, which real muxers write - where the raw pts arithmetic
    landed 5 frames off until it was anchored on `stream.start_time`).
    `start_pts` is that anchor: pts is a timestamp against the *container's*
    clock, not a frame count from this stream's first frame, so it has to be
    zeroed against wherever this stream actually starts before it means a
    frame index. `position` is then a plain frame counter from that
    landing, decoding forward to the target and dropping what is skipped
    past.
    """
    wanted = sorted(set(int(i) for i in indexes))
    found = {}
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        start_pts = stream.start_time if stream.start_time is not None else 0
        position = 0  # index of the next frame decode() will yield
        for target in wanted:
            if target < position or target - position > 2 * (int(fps) if fps else 24):
                # seek back or a long way forward: land on the keyframe at
                # or before the target, then read up to it. The seek target
                # is a container timestamp too, so it needs the same anchor.
                seconds = target / fps if fps else 0.0
                container.seek(
                    int(seconds / stream.time_base) + start_pts,
                    stream=stream,
                    backward=True,
                )
                position = None
            recovered = False
            for frame in container.decode(stream):
                if position is None:
                    # first frame after a seek says where we landed
                    position = (
                        int(
                            round(
                                float((frame.pts - start_pts) * stream.time_base)
                                * fps
                            )
                        )
                        if fps and frame.pts is not None
                        else 0
                    )
                    if position > target and not recovered:
                        # Landed past the target: the keyframe estimate was
                        # wrong for this file (off-rate or VFR). Reading on
                        # would scan to EOF and blame the caller; read from
                        # the top once instead, which is always correct.
                        container.seek(start_pts, stream=stream, backward=True)
                        position = None
                        recovered = True
                        continue
                if position == target:
                    image = frame.to_image()
                    found[target] = fit(image, target) if fit is not None else image
                    position += 1
                    break
                position += 1
        missing = [i for i in wanted if i not in found]
        if missing:
            raise ValueError(f"could not decode frame(s) {missing} of {path}")
    return found
