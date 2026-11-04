"""chain_repair_cut: the pieces that re-render one segment of a finished
chain in place (#1169, `templates/minimax/repair-segment`).

A guide chain run with `save_segments` + `keep_segments` leaves each segment
as a playable mp4, `<prefix>.<iteration>.segment-NNN.mp4`, holding the frames
it put on the timeline (chain_spill.py). Segment k is repaired by one H3 render
held at both ends: segment k-1's last `head_frames` frames as a guide at frame
0, segment k+1's frames as a guide ending at the render's last frame, both with
their audio. The render's new frames between the guides replace segment k, and
the splice puts it back between its neighbours.

The geometry is the 17-frame grid's. A guide starts at frame 17j and is 17m + 5
frames long, and a render is 17n + 5, so the tail guide ending at the render's
end starts at `num_frames - tail_frames`, always on 17j - and the frames between
the guides number 12 (mod 17). An interior segment of a 17n + 5 chain with a
22- or 39-frame carry is a multiple of 17 long, so the new frames cannot be
exactly segment k: they run `overrun` frames on into segment k+1 (12 at the
smallest render), the tail guide is segment k+1's frames from there, and the
splice takes segment k+1 back from that frame on. The take keeps its frame
count exactly; segments other than k and the head of k+1 are the files' own
frames, re-encoded once by the join.

A workflow has no arithmetic, so this command does it (`repair_geometry`,
pure) and cuts the media; the template composes it with the H3 step,
`trim_video` and `concat_videos(continuous_joins=true)`, which keeps the two
repaired seams measured by analyze_seams rather than skipped as cuts.

Concepts only from H3MD Director (AGPL-3.0) and roycho87/geiru_context_suite
(GPL-3.0): a tail hold that outnumbers the head hold, no tail hold on the last
segment, a new seed for a re-render (docs/ATTRIBUTIONS.md).
"""

import logging
import os
import re

from ..locations import validate_media_path
from ..media import probe_metadata
from ..pipeline_processors.h3_rules import GUIDE_FRAMES_PER_CHUNK
from ..security import validate_path
from ..task_domains import POSITIVE, check_arguments, whole_number
from .registry import register_command
from .trim import trim_video
from .video_utils import load_audio_video

logger = logging.getLogger("dw")

COMMAND = "chain_repair_cut"

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
    """Every segment file of the chain `chain_output` names one of, in order."""
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


def _frame_count(path):
    info = probe_metadata(path) or {}
    count = info.get("frame_count")
    if not isinstance(count, int) or count <= 0:
        raise ValueError(f"{COMMAND}: cannot count the frames of {path}")
    return count, info


def chain_repair_cut(
    chain_output,
    segment,
    num_frames,
    head_frames=22,
    tail_frames=39,
    width=None,
    height=None,
):
    """Task command: cut a finished chain's segment files for a repair of one.

    Args:
        chain_output: Any one of the chain's segment files (an `output:`
            reference) - the others are found beside it
        segment: The segment to re-render, 1-based; 2 to N-1
        num_frames: The repair render's length - at least segment k's frames
            plus both guides, on the 17n + 5 grid; a smaller one is refused
            naming the smallest that fits
        head_frames: Segment k-1's last frames held at the render's frame 0
            (17m + 5 from 22)
        tail_frames: Segment k+1's frames held at the render's end - more
            than head_frames by default (39), the tail hold outnumbering the
            head
        width, height: The render's canvas, when given checked against the
            segment files' own size - a repair renders on the chain's canvas

    Returns:
        A dict for the template's later steps: `head_guide` and `tail_guide`
        (AudioVideo clips with their audio), `keyframe` (the head guide's
        first frame, the fl2va keyframe a guide chain gives its segments),
        `tail_frame`, `keep_start`, `keep_frames`, `before` (segment files
        1..k-1) and `after` (segment k+1 from the overrun on, then the files
        after it), with the geometry's numbers beside them.
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

    paths = segment_files(chain_output)
    counts, infos = zip(*(_frame_count(path) for path in paths))
    geometry = repair_geometry(
        list(counts), segment, num_frames, head_frames, tail_frames
    )
    _check_canvas(infos[segment - 1], width, height, paths[segment - 1])

    previous = load_audio_video(paths[segment - 2])
    following = load_audio_video(paths[segment])
    for clip, path in ((previous, paths[segment - 2]), (following, paths[segment])):
        if clip.audio is None:
            raise ValueError(
                f"{COMMAND}: {os.path.basename(path)} has no soundtrack, and the "
                "repair holds the neighbours' audio over both guides"
            )
    head_guide = trim_video(previous, geometry["head_start"], head_frames)
    tail_guide = trim_video(following, geometry["tail_start"], tail_frames)
    rest = trim_video(
        following, geometry["tail_start"], counts[segment] - geometry["tail_start"]
    )
    logger.info(
        f"{COMMAND}: segment {segment} of {len(paths)} - render {num_frames} "
        f"frames, head guide {head_frames} at 0, tail guide {tail_frames} at "
        f"{geometry['tail_frame']}, {geometry['overrun']} frames into segment "
        f"{segment + 1}"
    )
    return {
        **geometry,
        "segments": len(paths),
        "segment_frames": list(counts),
        "head_guide": head_guide,
        "tail_guide": tail_guide,
        "keyframe": head_guide.frames[0],
        "before": list(paths[: segment - 1]),
        "after": [rest, *paths[segment + 1 :]],
    }


def _check_canvas(info, width, height, path):
    for name, wanted in (("width", width), ("height", height)):
        if wanted is None:
            continue
        own = info.get(name)
        if own is not None and own != wanted:
            raise ValueError(
                f"{COMMAND}: '{name}' is {wanted}, but the chain was rendered at "
                f"{own} ({os.path.basename(path)}) - a repair renders on the "
                "chain's canvas, so the splice joins frames of one size"
            )


@register_command(
    COMMAND,
    implementation="dw.tasks.chain_repair.chain_repair_cut",
    domains={
        name: POSITIVE
        for name in (
            "segment",
            "num_frames",
            "head_frames",
            "tail_frames",
            "width",
            "height",
        )
    },
    whole_numbers=(
        "segment",
        "num_frames",
        "head_frames",
        "tail_frames",
        "width",
        "height",
    ),
    media_arguments=("chain_output",),
)
def _handle_chain_repair_cut(task, arguments, previous_pipelines):
    """Cut a finished chain's segment files for an in-place repair of one"""
    return chain_repair_cut(**arguments)
