"""Two to eight outputs side by side at the same moments, with how far each
is from a baseline - the read-only `compare_outputs` (#871). One grid (a
row per output, a column per moment) and, per non-baseline output, PSNR,
SSIM and mean absolute difference against the baseline at source
resolution.

Metrics always compare the same pixels: one shared crop is the region they
cover, and anything else - no crop, or one crop per output, which only
changes what each row *shows* - leaves them on the full frame. Outputs
whose frames differ in size get no number, and say why: a resize to force
one would itself be a difference. A moment past an output's end is a
blank cell, and says so.

The server process runs this per request, like `media_frames`: the
baseline's frames are held at source size for the whole call, every other
output's only while its own row is measured and fitted to tiles.
"""

import math
import os

import numpy
from PIL import Image, ImageDraw, ImageFont

from . import media
from .media_frames import compose_grid, evenly_spaced_indices, moment_index, parse_crop

MAX_COMPARE_NAMES = 8
MAX_COMPARE_MOMENTS = 8
DEFAULT_MOMENT_COUNT = 4

FULL_FRAME = "full frame"
PER_OUTPUT_CROPS = "full frame (per-output crops are display only)"


class UnreadableMedia(Exception):
    """A file that resolved but whose frames could not be decoded - the
    route answers 422, the file's problem rather than the caller's."""


def psnr(a, b):
    """Peak signal-to-noise ratio of two uint8 arrays in dB, `math.inf`
    when they are identical."""
    difference = a.astype(numpy.float64) - b.astype(numpy.float64)
    mse = float(numpy.mean(difference * difference))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(255.0 * 255.0 / mse)


def ssim(a, b):
    """Structural similarity of two uint8 RGB arrays, on their luma, with
    the usual 11x11 Gaussian window (sigma 1.5) - 1.0 when identical."""
    import cv2

    x = _luma(a)
    y = _luma(b)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    def blur(image):
        return cv2.GaussianBlur(image, (11, 11), 1.5)

    mu_x = blur(x)
    mu_y = blur(y)
    var_x = blur(x * x) - mu_x * mu_x
    var_y = blur(y * y) - mu_y * mu_y
    covariance = blur(x * y) - mu_x * mu_y
    numerator = (2 * mu_x * mu_y + c1) * (2 * covariance + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    return float(numpy.mean(numerator / denominator))


def mean_abs_diff(a, b):
    """Mean absolute difference of two uint8 arrays, on the 0-255 scale."""
    return float(numpy.mean(numpy.abs(a.astype(numpy.int16) - b.astype(numpy.int16))))


def _luma(rgb):
    # ITU-R 601, the weights Pillow's "L" conversion uses
    rgb = rgb.astype(numpy.float64)
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def check_request(names, at, crop, baseline):
    """Refuse a request by its shape before anything is resolved or
    decoded: name and moment counts, the baseline, the crop's form.
    Returns (the baseline's row, crop mode: None, "shared" or "each")."""
    if not isinstance(names, list) or not 2 <= len(names) <= MAX_COMPARE_NAMES:
        count = len(names) if isinstance(names, list) else 0
        raise ValueError(
            f"`names` holds {count} outputs; compare takes 2 to {MAX_COMPARE_NAMES}"
        )
    if at is not None and not 1 <= len(at) <= MAX_COMPARE_MOMENTS:
        raise ValueError(
            f"`at` names {len(at)} moments; compare takes 1 to {MAX_COMPARE_MOMENTS}"
        )
    if baseline is None:
        row = 0
    elif baseline in names:
        row = names.index(baseline)
    else:
        raise ValueError(f"baseline {baseline!r} is not one of `names`")
    if crop is None:
        return row, None
    if crop and all(isinstance(box, (list, tuple)) for box in crop):
        if len(crop) != len(names):
            raise ValueError(
                f"`crop` gives {len(crop)} boxes for {len(names)} names - give "
                "one [x, y, width, height] for every output, or one box for all"
            )
        return row, "each"
    return row, "shared"


def crop_box(crop, width, height, name):
    """`[x, y, w, h]` as Pillow's (left, upper, right, lower), refused unless
    it lies wholly inside this output's `width` x `height` frame - a box
    clamped differently per output would measure different regions."""
    x, y, w, h = parse_crop(crop)
    if x + w > width or y + h > height:
        raise ValueError(
            f"crop {[x, y, w, h]} is out of bounds for {name}, which is "
            f"{width}x{height}"
        )
    return (x, y, x + w, y + h)


def resolve_moments(at, reference, shortest):
    """The frame index each moment names, as [(moment, frame)]. Seconds
    convert at the reference video's fps (the baseline's, or the first
    video's), and a negative frame counts back from its end; every output
    is then read at that same frame number. Without `at`, 4 evenly spaced
    frames of the shortest video."""
    if at is None:
        return [
            (f"frame:{index}", index)
            for index in evenly_spaced_indices(shortest, DEFAULT_MOMENT_COUNT)
        ]
    resolved = []
    for moment in at:
        index = moment_index(moment, reference)
        if index < 0:
            raise ValueError(f"{moment!r} is before the start of the clip")
        resolved.append((moment, index))
    return resolved


def compare(sources, at, crop, baseline_row, crop_mode, max_dimension):
    """The grid and metrics for `sources`, one dict per output in the
    caller's order: {name, kind, and for a still `image` (decoded, PIL),
    for a video `path` and `shape` (`video_shape`)}. Returns the answer's
    body without the encoded grid, plus the grid itself under "grid"."""
    for source in sources:
        source["width"], source["height"] = _size(source)
        source["box"] = None
    if crop_mode == "shared":
        for source in sources:
            source["box"] = crop_box(
                crop, source["width"], source["height"], source["name"]
            )
    elif crop_mode == "each":
        for source, box in zip(sources, crop):
            source["box"] = crop_box(
                box, source["width"], source["height"], source["name"]
            )

    videos = [source for source in sources if source["kind"] == "video"]
    notes = []
    if videos:
        reference = (
            sources[baseline_row]
            if sources[baseline_row]["kind"] == "video"
            else videos[0]
        )
        shortest = min(video["shape"]["frame_count"] for video in videos)
        columns = resolve_moments(at, reference["shape"], shortest)
        if all(
            frame >= video["shape"]["frame_count"]
            for _, frame in columns
            for video in videos
        ):
            raise ValueError(
                f"every moment is past the end of every video (the shortest has "
                f"{shortest} frames)"
            )
    else:
        columns = [("still", 0)]
        if at is not None:
            notes.append("`at` ignored: every output is a still image")
    frames = [frame for _, frame in columns]

    metrics_region = (
        PER_OUTPUT_CROPS
        if crop_mode == "each"
        else f"crop {_rectangle(sources[0]['box'])}"
        if crop_mode == "shared"
        else FULL_FRAME
    )
    measure_box = sources[0]["box"] if crop_mode == "shared" else None

    baseline = sources[baseline_row]
    base_frames = _frames(baseline, frames)
    base_arrays = [
        None if frame is None else _measured(frame, measure_box)
        for frame in base_frames
    ]

    rows = []
    metrics = []
    for row, source in enumerate(sources):
        source_frames = base_frames if row == baseline_row else _frames(source, frames)
        if row != baseline_row:
            metrics.append(
                _metrics_for(
                    row, source, source_frames, baseline, base_arrays, measure_box
                )
            )
        rows.append(
            [
                None if frame is None else _cell(frame, source["box"], max_dimension)
                for frame in source_frames
            ]
        )
        del source_frames

    fps_values = {round(video["shape"]["fps"] or 0.0, 3) for video in videos}
    grid = _grid(rows, sources, baseline_row, frames, bool(videos))
    return {
        "grid": grid,
        "baseline": baseline["name"],
        "region": metrics_region,
        "fps_differs": len(fps_values) > 1,
        "columns": [
            {"moment": moment, "frame": frame if videos else None}
            for moment, frame in columns
        ],
        "rows": [_row_summary(source) for source in sources],
        "metrics": metrics,
        "notes": notes,
    }


def _size(source):
    if source["kind"] == "video":
        return source["shape"]["width"], source["shape"]["height"]
    return source["image"].size


def _frames(source, frames):
    """The source's frames at these indexes as RGB PIL images, None for a
    moment past its end. A still is the same image at every moment."""
    if source["kind"] == "image":
        image = source["image"].convert("RGB")
        return [image for _ in frames]
    total = source["shape"]["frame_count"]
    inside = [frame for frame in frames if frame < total]
    try:
        found = media.read_frames(source["path"], inside) if inside else {}
    except Exception as e:
        # PyAV's own errors, or read_frames' ValueError for a frame it
        # could not reach: either way the file, not the request
        raise UnreadableMedia(f"{source['name']} could not be decoded: {e}")
    return [found[frame].convert("RGB") if frame < total else None for frame in frames]


def _measured(image, box):
    return numpy.asarray(image.crop(box) if box else image)


def _metrics_for(row, source, source_frames, baseline, base_arrays, measure_box):
    entry = {"name": source["name"], "row": row, "reason": None, "cells": []}
    same_size = (source["width"], source["height"]) == (
        baseline["width"],
        baseline["height"],
    )
    if not same_size:
        entry["reason"] = (
            f"size differs from baseline: {source['width']}x{source['height']} "
            f"vs {baseline['width']}x{baseline['height']}"
        )
    for frame, base in zip(source_frames, base_arrays):
        cell = {"psnr_db": None, "ssim": None, "mean_abs_diff": None, "reason": None}
        if frame is None:
            cell["reason"] = f"past end ({source['shape']['frame_count']} frames)"
        elif base is None:
            cell["reason"] = (
                f"baseline past end ({baseline['shape']['frame_count']} frames)"
            )
        elif not same_size:
            cell["reason"] = entry["reason"]
        else:
            ours = _measured(frame, measure_box)
            value = psnr(ours, base)
            cell.update(
                psnr_db="identical" if math.isinf(value) else round(value, 2),
                ssim=round(ssim(ours, base), 4),
                mean_abs_diff=round(mean_abs_diff(ours, base), 3),
            )
        entry["cells"].append(cell)
    measured = [cell for cell in entry["cells"] if cell["ssim"] is not None]
    for key in ("psnr_db", "ssim", "mean_abs_diff"):
        entry[key] = _summary([cell[key] for cell in measured])
    if not measured and entry["reason"] is None:
        entry["reason"] = "no moment holds a frame of both this output and the baseline"
    return entry


def _summary(values):
    """mean, min and max of one metric over the measured columns, None when
    none was measured. PSNR's "identical" columns are left out of the
    arithmetic (they are infinite) and counted; all identical is
    "identical" throughout."""
    if not values:
        return None
    finite = [value for value in values if value != "identical"]
    identical = len(values) - len(finite)
    if not finite:
        return {"mean": "identical", "min": "identical", "max": "identical"}
    summary = {
        "mean": round(sum(finite) / len(finite), 4),
        "min": min(finite),
        "max": "identical" if identical else max(finite),
    }
    if identical:
        summary["identical_columns"] = identical
    return summary


def _rectangle(box):
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]


def _row_summary(source):
    summary = {
        "name": source["name"],
        "kind": source["kind"],
        "width": source["width"],
        "height": source["height"],
        "crop": _rectangle(source["box"]) if source["box"] else None,
    }
    if source["kind"] == "video":
        summary["frame_count"] = source["shape"]["frame_count"]
        summary["fps"] = source["shape"]["fps"]
    return summary


def _cell(frame, box, max_dimension):
    """One grid cell: the frame cut to its row's box at source pixels, then
    fitted to `max_dimension` on its longest side - never enlarged."""
    image = frame.crop(box) if box else frame
    limit = max(1, int(max_dimension))
    longest = max(image.size)
    if longest > limit:
        scale = limit / longest
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.LANCZOS,
        )
    return image.convert("RGB")


def short_name(name):
    """A row's label: an asset reference as given, an output as its run id
    and file (the workflow folder is the same across a comparison)."""
    if name.startswith("asset:"):
        return name
    parts = name.replace(os.sep, "/").split("/")
    return "/".join(parts[-2:])


def _grid(rows, sources, baseline_row, frames, stamped):
    """Rows of cells, every cell letterboxed to the largest cell's size so
    rows of different aspect still line up in columns; each row under a
    band naming its output, each cell stamped with its frame number."""
    cells = [cell for row in rows for cell in row if cell is not None]
    width = max(cell.width for cell in cells)
    height = max(cell.height for cell in cells)
    font_size = max(10, min(width // 16, 20))
    font = _font(font_size)
    band = font_size + 8
    tiles = []
    for row, source in zip(rows, sources):
        for cell, frame in zip(row, frames):
            tile = Image.new("RGB", (width, height), (32, 32, 32))
            if cell is not None:
                tile.paste(
                    cell, ((width - cell.width) // 2, (height - cell.height) // 2)
                )
                if stamped:
                    _stamp(tile, f"frame {frame}", font)
            else:
                _stamp(tile, f"frame {frame}: past end", font)
            tiles.append(tile)
    body = compose_grid(tiles, len(frames))
    grid = Image.new("RGB", (body.width, body.height + band * len(rows)), (0, 0, 0))
    draw = ImageDraw.Draw(grid)
    for index, source in enumerate(sources):
        top = index * (height + band)
        label = f"{index + 1}. {short_name(source['name'])}"
        if index == baseline_row:
            label += "  (baseline)"
        draw.text((4, top + 4), label, font=font, fill="white")
        grid.paste(
            body.crop((0, index * height, body.width, (index + 1) * height)),
            (0, top + band),
        )
    return grid


def _stamp(tile, text, font):
    ImageDraw.Draw(tile).text(
        (4, 4), text, font=font, fill="white", stroke_width=2, stroke_fill="black"
    )


def _font(size):
    try:
        return ImageFont.truetype("Arial", size)
    except (IOError, OSError):
        return ImageFont.load_default(size=size)
