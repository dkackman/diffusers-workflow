"""chain_repair_cut and chain_repair_splice: re-render one segment of a
finished chain in place (#1169, `templates/minimax/repair-segment`).

A guide chain run with `save_segments` + `keep_segments` leaves each segment
as a playable mp4 beside the take, `<prefix>.<iteration>.segment-NNN.mp4`,
holding the frames it put on the timeline (chain_spill.py), and names each
file on the take's shots (`segment_file`, in the run manifest). Segment k is
repaired by one H3 render held at both ends: segment k-1's last `head_frames`
frames as a guide at frame 0, segment k+1's frames as a guide ending at the
render's last frame, both with their audio. The render's new frames between
the guides replace segment k, and the splice puts them into the take.

The splice is cut from the take itself, never rebuilt from the segment files:
apart from the repaired span the output IS the take - its frame count (a
duration_seconds take is shorter than its files, cut on replay), its
soundtrack (a match_audio take's own track is kept whole; a generated one
keeps every seam's crossfade or declick bit for bit) and its picture. The
segment files are only the guides' material. The repaired span's audio is the
render's own, blended in and out of the take's track inside the span over
the window the chain blended that seam with (the shot's `crossfade_ms`, or
the declick a guide chain's held audio gets).

The geometry is the 17-frame grid's. A guide starts at frame 17j and is 17m + 5
frames long, and a render is 17n + 5, so the tail guide ending at the render's
end starts at `num_frames - tail_frames`, always on 17j - and the frames between
the guides number 12 (mod 17). An interior segment of a 17n + 5 chain with a
22- or 39-frame carry is a multiple of 17 long, so the new frames cannot be
exactly segment k: they run `overrun` frames on into segment k+1 (12 at the
smallest render), the tail guide is segment k+1's frames from there, and the
take resumes at that frame.

A workflow has no arithmetic, so the cut does it (`repair_geometry`, pure);
the template composes cut, the H3 step and the splice.

Concepts only from H3MD Director (AGPL-3.0) and roycho87/geiru_context_suite
(GPL-3.0): a tail hold that outnumbers the head hold, no tail hold on the last
segment, a new seed for a re-render (docs/ATTRIBUTIONS.md).
"""

import logging
import os
import re

import numpy

from ..dsp import as_channels_samples, equal_power_ramps, resample_waveform
from ..locations import validate_media_path
from ..media import probe_metadata
from ..media_types import AudioVideo
from ..pipeline_processors.h3_rules import GUIDE_FRAMES_PER_CHUNK
from ..runs import manifest_beside, shots_beside
from ..security import validate_path
from ..shots import shot_record, shots_for_file
from ..task_domains import (
    NON_NEGATIVE,
    POSITIVE,
    check_arguments,
    frames_to_samples,
    whole_number,
)
from .joins import DECLICK_MS
from .registry import register_command
from .trim import trim_video
from .video_utils import VideoFileReference, frames_as_pil_list, load_audio_video

logger = logging.getLogger("dw")

COMMAND = "chain_repair_cut"
SPLICE = "chain_repair_splice"

# The name SegmentSpill gives a segment file (chain_spill.py)
_SEGMENT_FILE = re.compile(r"^(?P<base>.+)\.segment-(?P<index>\d{3})\.mp4$")
# The shortest whole-latent guide a hold is made of
MIN_HOLD = 22


def _hold_problem(name, frames):
    if frames < MIN_HOLD or (frames - 5) % GUIDE_FRAMES_PER_CHUNK:
        return (
            f"{COMMAND}: '{name}' must be a whole-latent guide length (17m + 5, "
            f"from {MIN_HOLD}: 22, 39, 56), got {frames}"
        )
    return None


def repair_geometry(segment_frames, segment, num_frames, head_frames, tail_frames):
    """Where the repair render's guides sit and how the splice is cut.

    Args:
        segment_frames: Each segment file's frame count, in chain order
        segment: The segment to repair, 1-based; not the first or the last
        num_frames: The repair render's length, on the 17n + 5 grid
        head_frames: Frames of segment k-1's tail held at frame 0
        tail_frames: Frames of segment k+1 held at the render's end

    Returns:
        A dict: `head_start` (where in segment k-1 the head guide starts),
        `tail_start` (where in segment k+1 the tail guide starts, which is
        also `overrun`, the frames of k+1 the repair replaces), `tail_frame`
        (the render frame the tail guide is placed at), `keep_start` and
        `keep_frames` (the render span spliced in), `minimum_num_frames`
        and `total_frames` (the take's, unchanged by the splice).

    Raises:
        ValueError: naming the argument and the value that would work.
    """
    count = len(segment_frames)
    if count < 3:
        raise ValueError(
            f"{COMMAND}: the chain has {count} segment file(s); repairing one in "
            "place needs a segment on each side of it, so at least 3"
        )
    if segment < 2 or segment > count - 1:
        raise ValueError(
            f"{COMMAND}: 'segment' must be 2 to {count - 1} of this {count}-segment "
            f"chain, got {segment}. Segment 1 has no segment before it to hold "
            f"and segment {count} none after it - the first and the last need a "
            "one-sided repair (one guide), which this command does not cut"
        )
    for name, frames in (("head_frames", head_frames), ("tail_frames", tail_frames)):
        problem = _hold_problem(name, frames)
        if problem:
            raise ValueError(problem)
    if (num_frames - 5) % GUIDE_FRAMES_PER_CHUNK:
        raise ValueError(
            f"{COMMAND}: 'num_frames' must be on H3's 17n + 5 grid, got {num_frames}"
        )

    before, replaced, after = segment_frames[segment - 2 : segment + 1]
    if before < head_frames:
        raise ValueError(
            f"{COMMAND}: segment {segment - 1} has {before} frames, fewer than "
            f"the {head_frames}-frame head guide"
        )
    shortest = head_frames + replaced + tail_frames
    over = (5 - shortest) % GUIDE_FRAMES_PER_CHUNK
    minimum = shortest + over
    overrun = num_frames - shortest
    if overrun < 0:
        raise ValueError(
            f"{COMMAND}: 'num_frames' {num_frames} cannot hold segment {segment} "
            f"({replaced} frames) between a {head_frames}-frame head and a "
            f"{tail_frames}-frame tail guide - pass num_frames={minimum}"
        )
    if overrun + tail_frames > after:
        raise ValueError(
            f"{COMMAND}: the tail guide would be segment {segment + 1}'s frames "
            f"{overrun} to {overrun + tail_frames}, but it has {after} - "
            f"lower num_frames (the smallest is {minimum}) or tail_frames"
        )
    return {
        "head_start": before - head_frames,
        "tail_start": overrun,
        "overrun": overrun,
        "tail_frame": num_frames - tail_frames,
        "keep_start": head_frames,
        "keep_frames": replaced + overrun,
        "minimum_num_frames": minimum,
        "total_frames": sum(segment_frames),
    }


def segment_files(chain_output):
    """Every segment file named like the one `chain_output` names, in order -
    the fallback for a take whose shots name no files (no manifest beside
    it), checked by the caller against each other."""
    path = validate_media_path(chain_output, what="chain_output")
    match = _SEGMENT_FILE.match(os.path.basename(path))
    if match is None:
        raise ValueError(
            f"{COMMAND}: 'chain_output' must name one of a chain's segment files "
            "(<prefix>.<n>.segment-000.mp4, left by a chain run with "
            f"save_segments and keep_segments), got {os.path.basename(path)!r}"
        )
    directory = os.path.dirname(path)
    found = {}
    for name in os.listdir(directory):
        sibling = _SEGMENT_FILE.match(name)
        if sibling and sibling["base"] == match["base"]:
            found[int(sibling["index"])] = validate_path(
                os.path.join(directory, name), directory
            )
    indices = sorted(found)
    if indices != list(range(len(indices))):
        missing = sorted(set(range(indices[-1] + 1)) - set(indices))
        raise ValueError(
            f"{COMMAND}: the chain's segment files are not all there - "
            f"segment-{missing[0]:03d} is missing beside {match['base']}"
        )
    return [found[index] for index in indices]


def _take_naming(segment_path):
    """The take whose run manifest names `segment_path` on one of its shots,
    or None."""
    found = manifest_beside(segment_path)
    if found is None:
        return None
    run_dir, manifest = found
    name, directory = os.path.basename(segment_path), os.path.dirname(segment_path)
    for entry in manifest.get("steps") or []:
        if not isinstance(entry, dict) or entry.get("reused"):
            continue
        files = entry.get("files") or []
        for file in files:
            take = validate_path(os.path.join(run_dir, file), run_dir)
            shots = shots_for_file(entry.get("shots"), file, files) or []
            if os.path.dirname(take) == directory and any(
                shot.get("segment_file") == name for shot in shots
            ):
                return take
    return None


def _named_files(take, shots):
    """The segment files the take's shots name, beside it - or None when a
    shot names none."""
    if not shots or not all(shot.get("segment_file") for shot in shots):
        return None
    directory = os.path.dirname(take)
    paths = [
        validate_path(os.path.join(directory, shot["segment_file"]), directory)
        for shot in shots
    ]
    for path in paths:
        if not os.path.isfile(path):
            raise ValueError(
                f"{COMMAND}: the take's segment file {os.path.basename(path)} is "
                "gone - a repair needs every one of them kept"
            )
    return paths


def _locate(chain_take, chain_output):
    """(take path, take shots or None, segment file paths)."""
    if chain_take is None and chain_output is None:
        raise ValueError(
            f"{COMMAND}: pass 'chain_take' (an 'output:' of the chain's final "
            "video) or 'chain_output' (one of its segment files)"
        )
    output = (
        None
        if chain_output is None
        else validate_media_path(chain_output, what="chain_output")
    )
    if chain_take is None:
        chain_take = _take_naming(output)
        if chain_take is None:
            raise ValueError(
                f"{COMMAND}: no run manifest beside {os.path.basename(output)} "
                "names it on a take's shots - pass 'chain_take', the chain's "
                "final video, as well"
            )
    take = validate_media_path(chain_take, what="chain_take")
    shots = shots_beside(take)
    paths = _named_files(take, shots)
    if paths is not None:
        if output is not None and output not in paths:
            raise ValueError(
                f"{COMMAND}: 'chain_output' {os.path.basename(output)} is not one "
                f"of the segment files {os.path.basename(take)}'s shots name"
            )
        return take, shots, paths
    if output is None:
        raise ValueError(
            f"{COMMAND}: {os.path.basename(take)}'s shots name no segment files "
            "(a chain run before they were recorded, or a take copied away from "
            "its run) - pass 'chain_output', one of its segment files, as well"
        )
    paths = segment_files(output)
    if shots is not None and len(shots) != len(paths):
        shots = None
    _check_siblings_agree(paths)
    return take, shots, paths


def _check_siblings_agree(paths):
    """Refuse segment files found by name that disagree in rate or size - two
    runs' files mixed in one directory by the flat layout."""
    seen = {}
    for path in paths:
        info = probe_metadata(path) or {}
        own = (info.get("fps"), info.get("width"), info.get("height"))
        for key, value in seen.items():
            if value != own:
                raise ValueError(
                    f"{COMMAND}: segment files {os.path.basename(key)} and "
                    f"{os.path.basename(path)} disagree (fps, width, height: "
                    f"{value} against {own}) - files of two runs are mixed; "
                    "pass the take from a run directory"
                )
        seen = {path: own}


def _header_frames(path):
    info = probe_metadata(path) or {}
    count = info.get("frame_count")
    if not isinstance(count, int) or count <= 0:
        raise ValueError(f"{COMMAND}: cannot count the frames of {path}")
    return count


def _take_spans(take, shots, paths, segment, neighbours):
    """[(start, frames)] of every segment on the take, and its frame count:
    the take's shots, else the segment files' counts (the neighbours' from
    their decode) cut to the take's length."""
    if shots is not None:
        spans = [(int(s["start_frame"]), int(s["num_frames"])) for s in shots]
        return spans, sum(frames for _, frames in spans)
    counts = [
        neighbours.get(index) or _header_frames(path)
        for index, path in enumerate(paths)
    ]
    total = _header_frames(take)
    if not sum(counts[:-1]) < total <= sum(counts):
        raise ValueError(
            f"{COMMAND}: {os.path.basename(take)} has {total} frames, and its "
            f"segment files hold {sum(counts)} - they are not this take's"
        )
    spans, start = [], 0
    for count in counts:
        spans.append((start, min(count, total - start)))
        start += count
    return spans, total


def chain_repair_cut(
    chain_take=None,
    chain_output=None,
    segment=None,
    num_frames=None,
    head_frames=22,
    tail_frames=39,
    width=None,
    height=None,
):
    """Task command: cut a finished chain's guides for a repair of one segment,
    and where the repair sits on the take.

    Args:
        chain_take: The chain's final video (an `output:` reference). Found
            from `chain_output` through the run manifest when not given
        chain_output: One of the chain's segment files - needed only when the
            take's shots name none (no manifest beside it); then the others
            are found beside it by name
        segment: The segment to re-render, 1-based; 2 to N-1
        num_frames: The repair render's length - at least segment k's frames
            plus both guides, on the 17n + 5 grid; a smaller one is refused
            naming the smallest that fits
        head_frames: Segment k-1's last frames held at the render's frame 0
        tail_frames: Segment k+1's frames held at the render's end - more
            than head_frames by default (39), the tail hold outnumbering the
            head
        width, height: The render's canvas, when given checked against the
            segment files' own size - a repair renders on the chain's canvas

    Returns:
        A dict for the template's later steps: `head_guide` and `tail_guide`
        (AudioVideo clips with their audio), `keyframe` (the head guide's
        first frame), `tail_frame`, `keep_start`, `chain_take` (the take's
        path) and `start_frame`/`end_frame` (the take's frames the render's
        span replaces), with the geometry's numbers beside them.
    """
    segment = whole_number(segment, "segment", COMMAND, required=True)
    num_frames = whole_number(num_frames, "num_frames", COMMAND, required=True)
    head_frames = whole_number(head_frames, "head_frames", COMMAND, required=True)
    tail_frames = whole_number(tail_frames, "tail_frames", COMMAND, required=True)
    width = whole_number(width, "width", COMMAND)
    height = whole_number(height, "height", COMMAND)
    check_arguments(
        COMMAND,
        segment=segment,
        num_frames=num_frames,
        head_frames=head_frames,
        tail_frames=tail_frames,
        width=width,
        height=height,
    )

    take, shots, paths = _locate(chain_take, chain_output)
    if not 2 <= segment <= len(paths) - 1:
        # The geometry's own refusal, before anything is decoded
        repair_geometry([0] * len(paths), segment, num_frames, 22, 22)
    previous = load_audio_video(paths[segment - 2])
    following = load_audio_video(paths[segment])
    for clip, path in ((previous, paths[segment - 2]), (following, paths[segment])):
        if clip.audio is None:
            raise ValueError(
                f"{COMMAND}: {os.path.basename(path)} has no soundtrack, and the "
                "repair holds the neighbours' audio over both guides"
            )
    _check_canvas(previous.frames[0].size, width, height, paths[segment - 2])
    spans, total = _take_spans(
        take,
        shots,
        paths,
        segment,
        {segment - 2: len(previous.frames), segment: len(following.frames)},
    )
    counts = [frames for _, frames in spans]
    for index, clip in ((segment - 2, previous), (segment, following)):
        last = index == len(spans) - 1
        if counts[index] > len(clip.frames) or (
            not last and counts[index] != len(clip.frames)
        ):
            raise ValueError(
                f"{COMMAND}: the take's segment {index + 1} is {counts[index]} "
                f"frames, but {os.path.basename(paths[index])} holds "
                f"{len(clip.frames)} - they are not the same run's"
            )
    counts[segment - 2] = len(previous.frames)
    counts[segment] = len(following.frames)
    geometry = repair_geometry(counts, segment, num_frames, head_frames, tail_frames)
    start = spans[segment - 1][0]
    end = min(start + geometry["keep_frames"], total)

    head_guide = trim_video(previous, geometry["head_start"], head_frames)
    tail_guide = trim_video(following, geometry["tail_start"], tail_frames)
    logger.info(
        f"{COMMAND}: segment {segment} of {len(paths)} - render {num_frames} "
        f"frames, head guide {head_frames} at 0, tail guide {tail_frames} at "
        f"{geometry['tail_frame']}; take frames {start}..{end - 1} of {total} "
        "replaced"
    )
    return {
        **geometry,
        "keep_frames": end - start,
        "total_frames": total,
        "segments": len(paths),
        "segment_frames": counts,
        "segment_files": list(paths),
        "chain_take": take,
        "start_frame": start,
        "end_frame": end,
        "head_guide": head_guide,
        "tail_guide": tail_guide,
        "keyframe": head_guide.frames[0],
    }


def _check_canvas(size, width, height, path):
    for name, wanted, own in (("width", width, size[0]), ("height", height, size[1])):
        if wanted is not None and own != wanted:
            raise ValueError(
                f"{COMMAND}: '{name}' is {wanted}, but the chain was rendered at "
                f"{own} ({os.path.basename(path)}) - a repair renders on the "
                "chain's canvas, so the splice joins frames of one size"
            )


def chain_repair_splice(chain_take, video, segment, start_frame, end_frame, keep_start):
    """Task command: put a repair render's span into the take in place.

    Args:
        chain_take: The take (`chain_repair_cut`'s `chain_take`)
        video: The repair render
        segment: The segment repaired, 1-based - which of the take's shots
            the span opens
        start_frame, end_frame: The take's frames the span replaces
        keep_start: The render frame the span starts at (after the head guide)

    Returns:
        An AudioVideo that is the take apart from frames
        `[start_frame, end_frame)`: the take's own frames, frame count and
        track outside them. Inside, the render's frames and - unless the
        take's track is a match_audio one, kept whole - its audio, blended
        from the take's track and back into it within the span over the
        window the chain blended those seams with. Its shots are the take's,
        with the span as one `(repaired)` shot.
    """
    segment = whole_number(segment, "segment", SPLICE, required=True)
    start = whole_number(start_frame, "start_frame", SPLICE, required=True)
    end = whole_number(end_frame, "end_frame", SPLICE, required=True)
    keep_start = whole_number(keep_start, "keep_start", SPLICE, required=True)
    take = load_audio_video(validate_media_path(chain_take, what="chain_take"))
    if isinstance(video, VideoFileReference):
        video = load_audio_video(video.path)
    elif isinstance(video, str):
        video = load_audio_video(video)
    frames = frames_as_pil_list(take)
    rendered = frames_as_pil_list(video)
    if not 0 < start < end <= len(frames):
        raise ValueError(
            f"{SPLICE}: frames {start}..{end} are not inside the take's {len(frames)}"
        )
    if keep_start + end - start > len(rendered):
        raise ValueError(
            f"{SPLICE}: the render has {len(rendered)} frames, short of the "
            f"{end - start} from frame {keep_start} the span needs"
        )
    if rendered[0].size != frames[0].size:
        raise ValueError(
            f"{SPLICE}: the render's canvas {rendered[0].size} is not the "
            f"take's {frames[0].size}"
        )
    spliced = frames[:start] + rendered[keep_start : keep_start + end - start]
    spliced += frames[end:]
    fps = take.fps or getattr(video, "fps", None)
    shots = take.shots
    if not shots or len(shots) <= segment or shots[segment - 1]["start_frame"] != start:
        shots = None
    audio, blends, samples = _spliced_track(
        take, video, fps, shots, segment, start, end, keep_start
    )
    logger.info(
        f"{SPLICE}: segment {segment} - take frames {start}..{end - 1} of "
        f"{len(frames)} from the render's {keep_start}.."
        f"{keep_start + end - start - 1}"
    )
    return AudioVideo(
        spliced,
        audio,
        take.sample_rate if audio is not None else None,
        fps=fps,
        shots=_spliced_shots(shots, segment, start, end, len(frames), blends, samples),
    )


def _blend_ms(shots, index):
    """The window the chain blended the seam opening shot `index` with: its
    realized crossfade, else the declick a held or butt-joined seam gets."""
    if shots is None:
        return DECLICK_MS
    return shots[index].get("crossfade_ms") or DECLICK_MS


def _spliced_track(take, video, fps, shots, segment, start, end, keep_start):
    """(track, (in ms, out ms) or None, (first sample, end sample) or None)."""
    if take.audio is None:
        return None, None, None
    if shots is not None and all(shot.get("start_sample") is None for shot in shots):
        # match_audio: the caller's own track, laid under the take whole
        return take.audio, None, None
    if getattr(video, "audio", None) is None:
        raise ValueError(
            f"{SPLICE}: the render has no soundtrack to put under the span"
        )
    rate = take.sample_rate
    track = as_channels_samples(take.audio)
    if shots is not None:
        after = shots[segment]
        first = int(shots[segment - 1]["start_sample"])
        last = int(after["start_sample"]) + frames_to_samples(
            end - after["start_frame"], fps, rate
        )
    else:
        first = frames_to_samples(start, fps, rate)
        last = frames_to_samples(end, fps, rate)
    last = min(last, track.shape[1])
    own = as_channels_samples(video.audio)
    if video.sample_rate != rate:
        own = resample_waveform(own, video.sample_rate, rate)
    offset = frames_to_samples(keep_start, getattr(video, "fps", None) or fps, rate)
    span = numpy.zeros((track.shape[0], last - first), dtype=track.dtype)
    piece = own[:, offset : offset + span.shape[1]]
    span[:, : piece.shape[1]] = (
        piece if piece.shape[0] == span.shape[0] else piece.mean(axis=0, keepdims=True)
    )
    blends = []
    for side, index in (("in", segment - 1), ("out", segment)):
        window = min(int(_blend_ms(shots, index) / 1000.0 * rate), span.shape[1] // 2)
        blends.append(round(window / rate * 1000, 1))
        if not window:
            continue
        fade_out, fade_in = equal_power_ramps(window)
        if side == "in":
            span[:, :window] = (
                track[:, first : first + window] * fade_out + span[:, :window] * fade_in
            )
        else:
            span[:, -window:] = (
                span[:, -window:] * fade_out + track[:, last - window : last] * fade_in
            )
    audio = numpy.concatenate([track[:, :first], span, track[:, last:]], axis=1)
    return audio, tuple(blends), (first, last)


def _spliced_shots(shots, segment, start, end, total, blends, samples):
    """The take's shots with the span as one repaired shot, the shot after it
    starting where the take resumes."""
    first, last = samples if samples else (None, None)
    if shots is None:
        return [
            shot_record("take", 0, start),
            shot_record(f"segment {segment} (repaired)", start, end - start),
            *([shot_record("take", end, total - end)] if end < total else []),
        ]
    seam = ("crossfade_ms", "trim_frames", "segment_file")
    clean = [{k: v for k, v in shot.items() if k not in seam} for shot in shots]
    repaired = shot_record(
        f"{shots[segment - 1]['name']} (repaired)",
        start,
        end - start,
        first,
        None if first is None else last - first,
    )
    result = [*clean[: segment - 1], repaired]
    after = clean[segment]
    after_end = after["start_frame"] + after["num_frames"]
    if after_end > end:
        resumed = {**after, "start_frame": end, "num_frames": after_end - end}
        if first is not None:
            resumed["start_sample"] = last
            resumed["num_samples"] = after["start_sample"] + after["num_samples"] - last
        result.append(resumed)
    result.extend(clean[segment + 1 :])
    if blends:
        repaired["crossfade_ms"] = blends[0]
        if after_end > end:
            result[segment]["crossfade_ms"] = blends[1]
    return result


_WHOLE = ("segment", "num_frames", "head_frames", "tail_frames", "width", "height")


@register_command(
    COMMAND,
    implementation="dw.tasks.chain_repair.chain_repair_cut",
    domains={name: POSITIVE for name in _WHOLE},
    whole_numbers=_WHOLE,
    media_arguments=("chain_take", "chain_output"),
)
def _handle_chain_repair_cut(task, arguments, previous_pipelines):
    """Cut a finished chain's guides for an in-place repair of one segment"""
    return chain_repair_cut(**arguments)


@register_command(
    SPLICE,
    implementation="dw.tasks.chain_repair.chain_repair_splice",
    domains={
        "segment": POSITIVE,
        "start_frame": POSITIVE,
        "end_frame": POSITIVE,
        "keep_start": NON_NEGATIVE,
    },
    whole_numbers=("segment", "start_frame", "end_frame", "keep_start"),
    media_arguments=("chain_take",),
)
def _handle_chain_repair_splice(task, arguments, previous_pipelines):
    """Put a repair render's span back into the chain's take in place"""
    return chain_repair_splice(**arguments)
