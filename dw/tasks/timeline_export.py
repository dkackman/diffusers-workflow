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
range and at its own frame rate, the shot lengths of the join giving way
to the files' own. `target_url` is percent-encoded.

Frame values are integers on a track whose rate is the exact rational the
engine writes (23.976 is 24000/1001); OTIO stores the rate as a float.

The concept and the schema shape come from leonardofrito/Directors-Cut (MIT);
see docs/ATTRIBUTIONS.md.
"""

import json
import logging
import os
import zipfile
import zlib
from urllib.parse import quote, unquote
from types import SimpleNamespace

from .. import references
from ..locations import validate_media_path
from ..runs import output_root, split_run_path
from ..security import (
    InvalidInputError,
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


def _clip(name, start, duration, target_url, available_frames, rate, source=None):
    return {
        "OTIO_SCHEMA": "Clip.1",
        "name": name,
        "metadata": {"dw": {"source": source}} if source else {},
        "source_range": time_range(start, duration, rate),
        "effects": [],
        "markers": [],
        "enabled": True,
        "media_reference": {
            "OTIO_SCHEMA": "ExternalReference.1",
            "name": os.path.basename(target_url),
            "metadata": {},
            "available_range": time_range(0, available_frames, rate),
            # A URL: spaces, '#' and '%' in a file name are percent-encoded
            "target_url": quote(target_url),
        },
    }


def _source_reference(path):
    """The `output:` reference of a file inside a run directory of the
    output root, else None - a clip's `metadata.dw.source`, what
    `chain_take` and `chain_output` take. A file outside any run (an asset,
    a loose path) has no reference to record."""
    try:
        root = os.path.realpath(output_root())
        relative = os.path.relpath(os.path.realpath(path), root).replace(os.sep, "/")
    except (OSError, ValueError):
        return None
    if relative.startswith("../") or not split_run_path(relative)[1]:
        return None
    return references.make_ref(references.OUTPUT, relative)


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
        `media/`. `path` is the result writer's, already confined to the
        step's output directory (dw/writers.py `output_file_path`) - the
        only base there is to confine it to, which a bundle never learns."""
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


def _span(shot, index):
    """A shot record with its start and length as checked integers."""
    where = f"{COMMAND}: shots[{index}]"
    if not isinstance(shot, dict):
        raise InvalidInputError(
            f"{where} must be a shot record, not {type(shot).__name__}"
        )
    span = dict(shot)
    for key, least in (("start_frame", 0), ("num_frames", 1)):
        value = shot.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidInputError(f"{where} needs an integer {key}, got {value!r}")
        if value != int(value) or value < least:
            raise InvalidInputError(
                f"{where} {key} must be a whole number of at least {least}, "
                f"got {value!r}"
            )
        span[key] = int(value)
    return span


def _refuse_overlaps(records):
    """A track lays its clips end to end, so shots that overlap would
    lengthen the timeline past the video and repeat the shared frames:
    clips and gaps must add up to the video's length."""
    for before, after in zip(records, records[1:]):
        end = before["start_frame"] + before["num_frames"]
        if after["start_frame"] < end:
            raise InvalidInputError(
                f"{COMMAND}: shot {after.get('name', '')!r} starts at frame "
                f"{after['start_frame']}, inside {before.get('name', '')!r} "
                f"(frames {before['start_frame']}-{end - 1}) - shots on a "
                f"timeline cannot overlap"
            )


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

    if shots is not None and not isinstance(shots, (list, tuple)):
        raise InvalidInputError(
            f"{COMMAND}: shots must be a list of shot records, not "
            f"{type(shots).__name__}"
        )
    if sources is not None and not isinstance(sources, (list, tuple)):
        raise InvalidInputError(
            f"{COMMAND}: sources must be a list of files, one per shot, not "
            f"{type(sources).__name__}"
        )
    records = _shot_records(path, shots, total)
    if sources:
        return _sources_bundle(name, records, sources, info)
    return _joined_bundle(name, records, path, info, rate, total)


def _shot_records(path, shots, total):
    """The shots to lay out, in frame order: the caller's, else the ones the
    run manifest beside `path` records, else the whole file as one."""
    if shots:
        records = shots
    else:
        # The resolved path, so a {"location": ...} video still finds the
        # shots its run's manifest records beside the file
        records, _source = resolve_shots(path, SimpleNamespace(shots=None))
    if not records:
        records = [{"name": "whole", "start_frame": 0, "num_frames": total}]
    return sorted(
        (_span(shot, index) for index, shot in enumerate(records)),
        key=lambda shot: shot["start_frame"],
    )


def _sources_bundle(name, records, sources, info):
    """Sources mode: each clip points at its own file over its whole range,
    at that file's own rate. The joined video would be dead weight in the
    bundle, so it is left out."""
    if len(sources) != len(records):
        raise InvalidInputError(
            f"{COMMAND}: sources names {len(sources)} files for "
            f"{len(records)} shots - give one per shot, in order"
        )
    media = {}
    children = []
    for index, (shot, source) in enumerate(zip(records, sources)):
        source_path = _stored_file(source, f"sources[{index}]")
        source_info = _facts(source_path)
        arc = _media_name(source_path, media)
        media[arc] = source_path
        frames = int(source_info["frame_count"])
        # Each clip runs at its own file's rate, not the join's
        source_rate = float(exact_frame_rate(source_info["fps"]))
        children.append(
            _clip(
                str(shot.get("name", f"shot {index + 1}")),
                0,
                frames,
                arc,
                frames,
                source_rate,
                _source_reference(source_path),
            )
        )
    tracks = [_track("Video", children)]
    if info.get("sample_rate"):
        tracks.append(_track("Audio", json.loads(json.dumps(children))))
    return TimelineBundle(_timeline(name, tracks), media)


def _joined_bundle(name, records, path, info, rate, total):
    """Every clip points into the joined video over its shot's span, with a
    gap wherever the shots leave one."""
    media = {}
    video_arc = _media_name(path, media)
    media[video_arc] = path
    source = _source_reference(path)
    _refuse_overlaps(records)
    for shot in records:
        end = shot["start_frame"] + shot["num_frames"]
        if end > total:
            raise InvalidInputError(
                f"{COMMAND}: shot {shot.get('name', '')!r} runs to frame "
                f"{end} (starts at {shot['start_frame']}, {shot['num_frames']} "
                f"frames) but the video has only {total} frames - the shots "
                f"must fit inside the video"
            )

    def lay_out():
        children = []
        cursor = 0
        for index, shot in enumerate(records):
            start = shot["start_frame"]
            frames = shot["num_frames"]
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
                    source,
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


MAX_CONTENT_BYTES = 8 * 1024 * 1024


def _summarize_bundle(path):
    """A read-only summary of an `.otioz` this module wrote (#1254): the
    tracks with their clips and gaps in order, each clip's name, source
    range and media member, and the bundle's member list. `path` is a file
    the caller already confined; nothing is extracted, and `content.otio`
    is read bounded so a hostile zip cannot expand in memory.
    Raises ValueError for a file that is not a timeline bundle - and, read
    through `summarize_bundle`, for nothing else."""
    try:
        bundle = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as error:
        raise ValueError(f"not an .otioz bundle: {error}") from error
    with bundle:
        infos = bundle.infolist()
        members = [i.filename for i in infos if not i.filename.endswith("/")]
        try:
            info = bundle.getinfo("content.otio")
        except KeyError as error:
            raise ValueError("not an .otioz bundle: no content.otio") from error
        if info.file_size > MAX_CONTENT_BYTES:
            raise ValueError("content.otio is too large to summarize")
        with bundle.open(info) as handle:
            raw = handle.read(MAX_CONTENT_BYTES + 1)
        if len(raw) > MAX_CONTENT_BYTES:
            raise ValueError("content.otio is too large to summarize")
        version = None
        if "version.txt" in members:
            # Opened and read 64 bytes: bundle.read would inflate the whole
            # member first, however large a hostile one expands to
            with bundle.open("version.txt") as handle:
                version = handle.read(64).decode("utf-8", "replace")
    try:
        document = json.loads(raw)
    except ValueError as error:
        raise ValueError(f"content.otio is not JSON: {error}") from error
    stack = document.get("tracks") or {}
    tracks = []
    for track in stack.get("children") or []:
        clips, total = [], 0.0
        for child in track.get("children") or []:
            span = child.get("source_range") or {}
            duration = span.get("duration") or {}
            rate = duration.get("rate") or 0
            frames = duration.get("value") or 0
            seconds = round(frames / rate, 4) if rate else None
            if seconds:
                total += seconds
            entry = {
                "type": str(child.get("OTIO_SCHEMA", "")).split(".")[0],
                "name": child.get("name", ""),
                "start_frame": (span.get("start_time") or {}).get("value"),
                "frames": frames,
                "rate": rate,
                "seconds": seconds,
            }
            reference = child.get("media_reference") or {}
            if reference.get("target_url"):
                entry["media"] = unquote(reference["target_url"])
            source = ((child.get("metadata") or {}).get("dw") or {}).get("source")
            entry["source"] = source if isinstance(source, str) else None
            clips.append(entry)
        tracks.append(
            {
                "kind": track.get("kind", ""),
                "name": track.get("name", ""),
                "clip_count": sum(1 for c in clips if c["type"] == "Clip"),
                "seconds": round(total, 4),
                "items": clips,
            }
        )
    return {
        "kind": "timeline",
        "name": document.get("name", ""),
        "otio_version": version,
        "has_audio_track": any(t["kind"] == "Audio" for t in tracks),
        "tracks": tracks,
        "members": members,
    }


# What reading a damaged or foreign member can raise: a CRC mismatch
# (BadZipFile), an encrypted member (RuntimeError), an unsupported
# compression method (NotImplementedError), a truncated stream (EOFError,
# zlib.error); and what valid JSON of another shape raises when read as a
# bundle's dict (AttributeError, TypeError, ZeroDivisionError)
_MALFORMED = (
    zipfile.BadZipFile,
    RuntimeError,
    NotImplementedError,
    EOFError,
    zlib.error,
    OSError,
    AttributeError,
    TypeError,
    ZeroDivisionError,
)


def summarize_bundle(path):
    """`_summarize_bundle`, with every way a malformed bundle fails read as
    the ValueError the gallery route maps to 422."""
    try:
        return _summarize_bundle(path)
    except ValueError:
        raise
    except _MALFORMED as error:
        raise ValueError(f"not a readable timeline bundle: {error}") from error
