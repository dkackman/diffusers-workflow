"""export_timeline: a joined piece as an editable OpenTimelineIO bundle (#1197).

A joined video is one file to an editor, and a recut of it means finding every
seam again. The shots a join records say where each one starts, so this writes
them as the timeline they already are: an `.otioz` bundle that DaVinci Resolve
(18.5+) imports as a timeline with one clip per shot.

The bundle is a zip holding `version.txt` (`1.0.0`), `content.otio` (OTIO's
JSON, written directly - no `opentimelineio` dependency) and a `media/` folder
with the files the clips point at, each clip's `target_url` being the relative
path `media/<file>`. By default there is one video track and, when the file
has a soundtrack, one audio track; each shot is a clip whose media is the
joined video itself with `source_range` set to the shot's span, so the bundle
holds one media file and every clip points into it. With `sources` (one file
per shot, in order) each clip points at its own file over that file's full
range, the shot lengths of the join giving way to the files' own.

Frame values are integers on a track whose rate is the exact rational the
engine writes (23.976 is 24000/1001); OTIO stores the rate as a float.

The concept and the schema shape come from leonardofrito/Directors-Cut (MIT);
see docs/ATTRIBUTIONS.md.
"""

import json
import logging
import os
import zipfile
from types import SimpleNamespace

from ..locations import validate_media_path
from ..security import (
    InvalidInputError,
    validate_output_path,
    validate_string_input,
)
from ..writers import exact_frame_rate
from .assess import resolve_shots

logger = logging.getLogger("dw")

COMMAND = "export_timeline"
OTIO_VERSION = "1.0.0"
MEDIA_FOLDER = "media"
MAX_NAME_LENGTH = 200


def rational_time(frames, rate):
    return {"OTIO_SCHEMA": "RationalTime.1", "rate": rate, "value": int(frames)}


def time_range(start, duration, rate):
    return {
        "OTIO_SCHEMA": "TimeRange.1",
        "duration": rational_time(duration, rate),
        "start_time": rational_time(start, rate),
    }


def _gap(frames, rate):
    return {
        "OTIO_SCHEMA": "Gap.1",
        "name": "",
        "metadata": {},
        "source_range": time_range(0, frames, rate),
        "effects": [],
        "markers": [],
        "enabled": True,
    }


def _clip(name, start, duration, target_url, available_frames, rate):
    return {
        "OTIO_SCHEMA": "Clip.1",
        "name": name,
        "metadata": {},
        "source_range": time_range(start, duration, rate),
        "effects": [],
        "markers": [],
        "enabled": True,
        "media_reference": {
            "OTIO_SCHEMA": "ExternalReference.1",
            "name": os.path.basename(target_url),
            "metadata": {},
            "available_range": time_range(0, available_frames, rate),
            "target_url": target_url,
        },
    }


def _track(kind, children):
    return {
        "OTIO_SCHEMA": "Track.1",
        "name": kind,
        "metadata": {},
        "source_range": None,
        "effects": [],
        "markers": [],
        "enabled": True,
        "children": children,
        "kind": kind,
    }


def _timeline(name, tracks):
    return {
        "OTIO_SCHEMA": "Timeline.1",
        "name": name,
        "metadata": {},
        "global_start_time": None,
        "tracks": {
            "OTIO_SCHEMA": "Stack.1",
            "name": "tracks",
            "metadata": {},
            "source_range": None,
            "effects": [],
            "markers": [],
            "enabled": True,
            "children": tracks,
        },
    }


class TimelineBundle:
    """The timeline and the files it points at; `save` writes the `.otioz`.

    It has a `save`, so the result writer takes it for any content type it
    has no other writer for - `application/x-otioz` names the extension.
    """

    def __init__(self, document, media):
        self.document = document
        # {arcname: absolute source path}
        self.media = media

    def save(self, path):
        """Write the zip at `path`. The member names are fixed here, never
        taken from a path, so no source file can place itself outside
        `media/`."""
        directory = os.path.dirname(os.path.abspath(path))
        path = validate_output_path(path, directory)
        with zipfile.ZipFile(path, "w") as bundle:
            bundle.writestr("version.txt", OTIO_VERSION, zipfile.ZIP_DEFLATED)
            bundle.writestr(
                "content.otio",
                json.dumps(self.document, indent=4),
                zipfile.ZIP_DEFLATED,
            )
            for arcname, source in self.media.items():
                bundle.write(source, arcname, zipfile.ZIP_STORED)
        return path


def _media_name(source, taken):
    """`media/<file>` for a source, unique among the bundle's members."""
    base = os.path.basename(source) or "media"
    arcname = f"{MEDIA_FOLDER}/{base}"
    stem, extension = os.path.splitext(base)
    counter = 1
    while arcname in taken and taken[arcname] != source:
        counter += 1
        arcname = f"{MEDIA_FOLDER}/{stem}-{counter}{extension}"
    return arcname


def _stored_file(value, what):
    """The confined, existing file a video argument names."""
    from .video_utils import VideoFileReference

    if isinstance(value, VideoFileReference):
        value = value.path
    if isinstance(value, dict) and "location" in value:
        value = value["location"]
    if not isinstance(value, str):
        raise InvalidInputError(
            f"{COMMAND}: {what} must be a stored video (asset:, output: or a "
            f"path) - the bundle carries the file itself, so a clip still in "
            f"memory has to be saved first, not {type(value).__name__}"
        )
    return validate_media_path(value, None, what)


def _facts(path):
    from ..media import probe_metadata

    info = probe_metadata(path)
    if not info or not info.get("fps") or not info.get("frame_count"):
        raise InvalidInputError(f"{COMMAND}: '{os.path.basename(path)}' is not a video")
    return info


def export_timeline(video, sources=None, shots=None, name="timeline"):
    """Task command: the shots of a joined video as an OpenTimelineIO bundle.

    Args:
        video: The joined video - a stored file (asset:, output: or a path)
        sources: One file per shot, in order, for each clip to point at its
            own file over that file's whole range. Default: every clip points
            into `video` over its shot's span
        shots: Shot records, else the video's own from the run manifest beside
            the file, else the whole file is one clip
        name: The timeline's name

    Returns:
        A bundle the result writes as `<name>.otioz` - content type
        `application/x-otioz`
    """
    name = validate_string_input(str(name), max_length=MAX_NAME_LENGTH)
    path = _stored_file(video, "the video")
    info = _facts(path)
    rate = float(exact_frame_rate(info["fps"]))
    total = int(info["frame_count"])

    records, _source = resolve_shots(video, SimpleNamespace(shots=None), shots)
    if not records:
        records = [{"name": "whole", "start_frame": 0, "num_frames": total}]
    records = sorted(records, key=lambda shot: int(shot["start_frame"]))

    media = {}
    video_arc = _media_name(path, media)
    media[video_arc] = path

    if sources:
        if len(sources) != len(records):
            raise InvalidInputError(
                f"{COMMAND}: sources names {len(sources)} files for "
                f"{len(records)} shots - give one per shot, in order"
            )
        children = []
        for index, (shot, source) in enumerate(zip(records, sources)):
            source_path = _stored_file(source, f"sources[{index}]")
            source_info = _facts(source_path)
            arc = _media_name(source_path, media)
            media[arc] = source_path
            frames = int(source_info["frame_count"])
            children.append(
                _clip(
                    str(shot.get("name", f"shot {index + 1}")),
                    0,
                    frames,
                    arc,
                    frames,
                    rate,
                )
            )
        has_audio = bool(info.get("sample_rate"))
        tracks = [_track("Video", children)]
        if has_audio:
            tracks.append(_track("Audio", json.loads(json.dumps(children))))
        return TimelineBundle(_timeline(name, tracks), media)

    def lay_out():
        children = []
        cursor = 0
        for index, shot in enumerate(records):
            start = int(shot["start_frame"])
            frames = int(shot["num_frames"])
            if start > cursor:
                children.append(_gap(start - cursor, rate))
            children.append(
                _clip(
                    str(shot.get("name", f"shot {index + 1}")),
                    start,
                    frames,
                    video_arc,
                    total,
                    rate,
                )
            )
            cursor = start + frames
        if cursor < total:
            children.append(_gap(total - cursor, rate))
        return children

    tracks = [_track("Video", lay_out())]
    if info.get("sample_rate"):
        tracks.append(_track("Audio", lay_out()))
    return TimelineBundle(_timeline(name, tracks), media)
