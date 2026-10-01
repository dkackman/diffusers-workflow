"""Frames out of a video file by seeking to them - a moment, an evenly
spaced contact sheet, or the frame pair either side of a seam - without
decoding the clip whole. The server process runs this per request; a
four-minute 1080p clip materialised as PIL frames is tens of gigabytes,
so nothing here ever holds more than the frames it returns, each already
fitted to its tile where a caller asked for many (#193).
"""

import logging
import math

import numpy
from PIL import Image

from . import media

logger = logging.getLogger("dw")

# Each cell of a contact sheet and each side of a seam pair is a seek, a
# decode and a resize in the server process; a sheet past 64 cells is
# unreadable anyway, and `at` has its own cap in the route
# (MAX_FRAME_MOMENTS). Both are ValueErrors, so the route answers 400.
MAX_CONTACT_SHEET_FRAMES = 64
MAX_SEAMS = 32


def resolve_crop_box(crop, width, height):
    """`[x, y, w, h]` as Pillow's `(left, upper, right, lower)`, clamped to
    a frame. Every frame of one video shares the same dimensions, so a
    caller resolves this once against `video_shape(path)` and reuses it
    across every tile - the same `[x, y, width, height]` convention
    `get_output_image`'s crop uses. Refused when it is not four
    non-negative integers, starts outside the frame, or has nothing in it."""
    try:
        x, y, w, h = (int(v) for v in crop)
    except (TypeError, ValueError):
        raise ValueError(f"crop must be [x, y, width, height] in pixels, got {crop!r}.")
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(
            f"crop must have a non-negative origin and a positive size, got {crop!r}."
        )
    if x >= width or y >= height:
        raise ValueError(
            f"crop origin ({x}, {y}) lies outside the {width}x{height} frame."
        )
    return (x, y, min(x + w, width), min(y + h, height))


def frames_at(path, moments, shape=None, crop_box=None):
    """One tile per moment - a float in seconds or "frame:N" - in the order
    asked for. Each tile is {label, frame, seconds, image}.

    `shape` reuses an already-computed `video_shape(path)` - a caller such
    as the gallery route that also reports the clip's own frame_count/fps
    would otherwise pay for `video_shape`'s container open (and, lacking a
    header frame count, a full decode) a second time for the same answer.

    `crop_box` (`resolve_crop_box`'s Pillow box) is cut from each frame
    right after decode, before it is returned - full source resolution,
    not whatever a caller's `max_dimension` later downscales it to."""
    shape = shape if shape is not None else media.video_shape(path)
    indexes = [_moment_to_index(moment, shape) for moment in moments]
    fit = (lambda image, index: image.crop(crop_box)) if crop_box else None
    images = media.read_frames(path, indexes, fit=fit)
    return [_tile(index, images[index], shape) for index in indexes]


def contact_sheet(path, count, tile_width=320, shape=None, crop_box=None):
    """N evenly spaced frames (first and last included) tiled into one
    image - frame_grid without a workflow. `shape` reuses an
    already-computed `video_shape(path)`; see `frames_at`.

    `crop_box`, as in `frames_at`, is cut from each frame before it is
    stamped and tiled into the sheet - the sheet is built from cropped
    frames rather than cropped after assembly, so the box means the same
    source-pixel region whatever the sheet's own `tile_width` ends up."""
    if int(count) < 1:
        raise ValueError("count must be at least 1")
    shape = shape if shape is not None else media.video_shape(path)
    # Clamp to the clip's own length before checking the cap: a count that
    # would only ever produce a handful of cells (a short clip) should not
    # be refused for the raw number the caller asked for.
    count = min(int(count), shape["frame_count"])
    if count > MAX_CONTACT_SHEET_FRAMES:
        raise ValueError(
            f"count {count} is more than a contact sheet holds "
            f"({MAX_CONTACT_SHEET_FRAMES}); ask for a smaller one, or `at` for moments"
        )
    indexes = evenly_spaced_indices(shape["frame_count"], count)

    def fit(image, index):
        if crop_box:
            image = image.crop(crop_box)
        return _stamped_tile(image, index, shape["fps"], tile_width)

    images = media.read_frames(path, indexes, fit=fit)
    tiles = [images[index] for index in indexes]
    grid = compose_grid(tiles, default_columns(len(tiles)))
    return {
        "label": f"contact sheet, {len(tiles)} frames",
        "frame": indexes[0],
        "seconds": _seconds(indexes[0], shape),
        "image": grid,
        "frames": list(indexes),
    }


def seam_tiles(
    path, boundaries, names=None, tile_width=320, shape=None, wanted=None, crop_box=None
):
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
    `video_shape(path)`; see `frames_at`.

    `crop_box`, as in `frames_at`, is cut from each source frame before it
    is fit to `tile_width` and paired into a seam - the pair is built from
    cropped frames rather than cropped after pairing."""
    shape = shape if shape is not None else media.video_shape(path)
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

    def fit(image, _index):
        if crop_box:
            image = image.crop(crop_box)
        return _fit_width(image, tile_width)

    images = media.read_frames(path, frame_indexes, fit=fit)
    tiles = []
    for seam, boundary in chosen:
        before = images[boundary - 1]
        after = images[boundary]
        pair = compose_grid([before, after], 2)
        difference = float(
            numpy.abs(
                numpy.asarray(before, dtype=numpy.int16)
                - numpy.asarray(after, dtype=numpy.int16)
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
        raw = moment[len("frame:") :]
        try:
            index = int(raw)
        except ValueError:
            raise ValueError(f'"{moment}" - frame index must be a whole number')
        if index < 0:
            index += total
        if not 0 <= index < total:
            raise ValueError(
                f"frame {index} is past the end of a {total}-frame clip "
                f"(frames 0-{total - 1})"
            )
        return index
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
        duration = total / fps
        message = (
            f"{moment!r} s is past the end of a {duration:.2f} s "
            f"({total}-frame) clip - bare numbers in 'at' are seconds"
        )
        # A fractional value is a seconds overshoot, not a frame index in
        # disguise - the frame: hint only makes sense for a whole number
        # that would itself be a valid frame index.
        if seconds.is_integer() and 0 <= int(seconds) < total:
            message += f'; use "frame:{_format_moment(moment)}" for a frame index'
        raise ValueError(message)
    return index


def _format_moment(moment):
    if isinstance(moment, float) and moment.is_integer():
        return str(int(moment))
    return str(moment)


def _seconds(index, shape):
    return index / shape["fps"] if shape["fps"] else float(index)


def _tile(index, image, shape):
    fps = shape["fps"]
    stamp = format_timestamp(index, fps) if fps else f"#{index}"
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
    return grid_tile(image, index, fps, min(int(tile_width), image.width), label=True)


def _fit_width(image, tile_width):
    # never upscaled: a tile wider than its source is a blurred enlargement
    # that costs bytes and shows nothing the source holds
    tile_width = min(int(tile_width), image.width)
    height = max(1, round(image.height * tile_width / image.width))
    return image.resize((tile_width, height), Image.LANCZOS).convert("RGB")


def evenly_spaced_indices(total, count):
    """`count` frame indices spaced evenly across [0, total - 1], inclusive
    of both ends. Rounding can coincide two spacings on one index in a short
    clip; those collapse rather than repeating the same frame as a tile."""
    if count == 1:
        return [0]
    raw = numpy.linspace(0, total - 1, num=count)
    seen = []
    for value in raw.round().astype(int).tolist():
        if not seen or seen[-1] != value:
            seen.append(value)
    return seen


def default_columns(count):
    """A grid biased wide: rows no more than columns, columns >= sqrt(count)."""
    rows = math.isqrt(count) or 1
    return math.ceil(count / rows)


def grid_tile(frame, index, fps, tile_width, label):
    tile_height = max(1, round(frame.height * tile_width / frame.width))
    tile = frame.resize((tile_width, tile_height), Image.LANCZOS).convert("RGB")
    if not label:
        return tile

    from PIL import ImageDraw, ImageFont

    text = format_timestamp(index, fps) if fps else f"#{index}"
    draw = ImageDraw.Draw(tile)
    font_size = max(10, tile_width // 16)
    try:
        font = ImageFont.truetype("Arial", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default(size=font_size)
    draw.text(
        (4, 4), text, font=font, fill="white", stroke_width=2, stroke_fill="black"
    )
    return tile


def format_timestamp(index, fps):
    seconds = index / fps
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes):02d}:{remainder:04.1f}"


def compose_grid(tiles, columns):
    tile_width, tile_height = tiles[0].size
    rows = math.ceil(len(tiles) / columns)
    grid = Image.new("RGB", (columns * tile_width, rows * tile_height), (0, 0, 0))
    for position, tile in enumerate(tiles):
        row, col = divmod(position, columns)
        grid.paste(tile, (col * tile_width, row * tile_height))
    return grid
