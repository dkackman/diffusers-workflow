"""Frames out of a video file by seeking to them - a moment, an evenly
spaced contact sheet, or the frame pair either side of a seam - without
decoding the clip whole. The server process runs this per request; a
four-minute 1080p clip materialised as PIL frames is tens of gigabytes,
so nothing here ever holds more than the frames it returns (#193).
"""

import logging

import av
from PIL import Image

from .tasks.video_utils import (
    _compose_grid,
    _default_columns,
    _evenly_spaced_indices,
    _format_timestamp,
)

logger = logging.getLogger("dw")


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


def frames_at(path, moments):
    """One tile per moment - a float in seconds or "frame:N" - in the order
    asked for. Each tile is {label, frame, seconds, image}."""
    shape = video_shape(path)
    indexes = [_moment_to_index(moment, shape) for moment in moments]
    images = _read_frames(path, indexes)
    return [_tile(index, images[index], shape) for index in indexes]


def contact_sheet(path, count, tile_width=320):
    """N evenly spaced frames (first and last included) tiled into one
    image - frame_grid without a workflow."""
    if int(count) < 1:
        raise ValueError("count must be at least 1")
    shape = video_shape(path)
    count = min(int(count), shape["frame_count"])
    indexes = _evenly_spaced_indices(shape["frame_count"], count)
    images = _read_frames(path, indexes)
    tiles = [_fit_width(images[index], tile_width) for index in indexes]
    grid = _compose_grid(tiles, _default_columns(len(tiles)))
    return {
        "label": f"contact sheet, {len(tiles)} frames",
        "frame": indexes[0],
        "seconds": _seconds(indexes[0], shape),
        "image": grid,
        "frames": list(indexes),
    }


def seam_tiles(path, boundaries, names=None, tile_width=320):
    """For each boundary (the frame index a shot *starts* at), the last
    frame before it and the first frame at it, side by side - the seam and
    continuity evidence in one image. Seam i sits between shot i and shot
    i+1; `names` names the shots, "shot 1".. by default."""
    shape = video_shape(path)
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
    wanted = sorted({b - 1 for b in boundaries} | set(boundaries))
    images = _read_frames(path, wanted)
    tiles = []
    for seam, boundary in enumerate(boundaries):
        before = _fit_width(images[boundary - 1], tile_width)
        after = _fit_width(images[boundary], tile_width)
        pair = _compose_grid([before, after], 2)
        tiles.append(
            {
                "label": f"seam {seam + 1}: {names[seam]} | {names[seam + 1]}",
                "frame": boundary,
                "seconds": _seconds(boundary, shape),
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
        index = int(round(float(moment) * fps))
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


def _fit_width(image, tile_width):
    height = max(1, round(image.height * tile_width / image.width))
    return image.resize((tile_width, height), Image.LANCZOS).convert("RGB")


def _read_frames(path, indexes):
    """The frames at these indexes, as {index: PIL image}, in one forward
    pass that seeks to the keyframe before each wanted frame rather than
    decoding from the top. Decodes are dropped as soon as they are past.

    This assumes a constant frame rate, which every file this engine writes
    has (`encode_video` / `export_to_video` write a fixed `fps`): a
    `backward=True` seek lands on the keyframe at or before the target
    timestamp, and `frame.pts * stream.time_base` converts that keyframe's
    own presentation time back to an exact frame index (`round(seconds *
    fps)`) - verified against PyAV 18.1's actual seek landings (both a
    single-keyframe short clip, where every seek lands on frame 0, and a
    `g=10` multi-keyframe clip, where a seek to frame 95 lands exactly on
    frame 90) before trusting the arithmetic here. `position` is then a
    plain frame counter from that landing, decoding forward to the target
    and dropping what is skipped past.
    """
    wanted = sorted(set(int(i) for i in indexes))
    found = {}
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        position = 0  # index of the next frame decode() will yield
        for target in wanted:
            if target < position or target - position > 2 * (int(fps) if fps else 24):
                # seek back or a long way forward: land on the keyframe at
                # or before the target, then read up to it
                seconds = target / fps if fps else 0.0
                container.seek(
                    int(seconds / stream.time_base), stream=stream, backward=True
                )
                position = None
            for frame in container.decode(stream):
                if position is None:
                    # first frame after a seek says where we landed
                    position = (
                        int(round(float(frame.pts * stream.time_base) * fps))
                        if fps and frame.pts is not None
                        else 0
                    )
                if position == target:
                    found[target] = frame.to_image()
                    position += 1
                    break
                position += 1
        missing = [i for i in wanted if i not in found]
        if missing:
            raise ValueError(f"could not decode frame(s) {missing} of {path}")
    return found
