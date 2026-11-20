"""Keep a span of a video's frames, and its audio over the same span (#627).

`trim_video` keeps frames `[start_frame, start_frame + num_frames)`. The clip's
soundtrack, if it has one, is cut to the same span at the track's own rate:
each end is `frames_to_samples` of a frame index, rounded on its own (the
cumulative rule `slice_audio` and `window_video` follow), so consecutive trims
tile the track with no sample lost or repeated.

`pad_to` lengthens the kept span to a render length (#1194): the last kept
frame is held up to `pad_to` frames and the track runs on in silence to the
same end, so a cut `detect_cuts` laid out can be handed to a model whose frame
grid it falls short of, and trimmed back to its own length after.

`replace_cuts` is the way back: each re-rendered cut's frames go over the
take's own frames at the cut's `start_frame`, and everything else - the cuts
nobody listed, and the whole soundtrack - stays the take's. The result is the
take's length whatever the list holds, so picture and sound cannot part: the
join it replaced (`concat_videos` of the rendered cuts, the take's track paired
on) lost a cut left out of the middle and slid every later cut under the wrong
sound. `cut_list_problem` owns the list's one rule - in the take's order, no
two overlapping - for the task and for validation.
"""

import logging
import numbers

import numpy as np

from ..dsp import as_channels_samples
from ..media_types import AudioVideo
from ..shots import carried_shots, shot_record, trimmed_shots
from ..task_domains import (
    NON_NEGATIVE,
    POSITIVE,
    check_arguments,
    frames_to_samples,
    real_number,
    whole_number,
)
from .registry import register_command
from .joins import video_names
from .video_utils import VideoFileReference, frames_as_pil_list, load_audio_video

logger = logging.getLogger("dw")

COMMAND = "trim_video"
REPLACE = "replace_cuts"


def trim_video(video, start_frame, num_frames, fps=None, pad_to=None):
    """Task command: keep one span of a video's frames and its audio.

    Args:
        video: The clip - a frame list or array, an AudioVideo (from a
            previous_result reference), or the path or URL of a video file,
            read with its audio
        start_frame: The first frame kept, from 0
        num_frames: How many frames are kept. The span must lie inside the
            clip: asking past its end is refused rather than shortened
        fps: Frame rate of the clip. Defaults to the rate the clip carries;
            needed only to cut its audio, which is refused without one
        pad_to: A frame count at least num_frames: the last kept frame is
            held until the clip is this long, and its track padded with
            silence to the same end. Omitted, the span is kept as it is

    Returns:
        The kept frames as one AudioVideo - the clip's track cut to the same
        span, at its own rate, fps and sample rate kept; a clip that came in
        as bare frames comes out silent, at 'fps'. The shots the clip carried are clipped to
        the span (sample side cleared); a clip with none carries one shot
        spanning the kept frames, named after the file it was read from.
        With pad_to, the clip is pad_to frames long and its last shot runs
        over the held frames
    """
    start_frame = whole_number(start_frame, "start_frame", COMMAND)
    num_frames = whole_number(num_frames, "num_frames", COMMAND)
    pad_to = whole_number(pad_to, "pad_to", COMMAND)
    fps = real_number(fps, "fps", COMMAND)
    whole = [("start_frame", start_frame), ("num_frames", num_frames)]
    if pad_to is not None:
        whole.append(("pad_to", pad_to))
    for name, value in whole:
        if not isinstance(value, numbers.Integral) or isinstance(value, bool):
            raise ValueError(
                f"{COMMAND} needs '{name}' as a whole number, got {value!r}"
            )
    start_frame, num_frames = int(start_frame), int(num_frames)
    pad_to = None if pad_to is None else int(pad_to)
    check_arguments(
        COMMAND, start_frame=start_frame, num_frames=num_frames, fps=fps, pad_to=pad_to
    )
    problem = pad_to_problem(num_frames, pad_to)
    if problem:
        raise ValueError(problem)

    name = video_names([video])[0]
    if isinstance(video, VideoFileReference):
        name = video_names([video.path])[0]
        video = load_audio_video(video.path)
    elif isinstance(video, str):
        video = load_audio_video(video)

    frames = frames_as_pil_list(video)
    total = len(frames)
    end = start_frame + num_frames
    problem = span_problem(start_frame, num_frames, total)
    if problem:
        raise ValueError(f"{COMMAND}: {problem}")
    kept = frames[start_frame:end]
    logger.info(f"Trimmed to frames {start_frame}..{end - 1} of {total}")
    held = (pad_to or num_frames) - num_frames
    if held:
        kept = list(kept) + [kept[-1]] * held
        logger.info(f"Held the last frame {held} more, to {pad_to} frames")

    # Bare frames come out as one silent clip, not a bare list: a task's list
    # result is one result per entry, so the step saved each frame as its own
    # video (#1199)
    clip = video if isinstance(video, AudioVideo) else None
    audio = clip.audio if clip else None
    sample_rate = clip.sample_rate if clip else None
    fps = fps or (clip.fps if clip else None)
    if audio is not None:
        audio = _span_of_audio(audio, sample_rate, fps, start_frame, end, held)
    shots = trimmed_shots(
        clip.shots if clip else None, start_frame, keep_frames=num_frames
    )
    if shots and held:
        # The held frames belong to the shot they repeat
        shots[-1] = {**shots[-1], "num_frames": shots[-1]["num_frames"] + held}
    if not shots:
        # A clip with no shots record is one shot: the trim is still a clip
        # a consumer reads spans from (#627). The track was cut at its own
        # rate, so its samples are measured here, not derived
        shots = [
            shot_record(
                name,
                0,
                num_frames + held,
                None if audio is None else 0,
                None if audio is None else audio.shape[1],
            )
        ]
    return AudioVideo(kept, audio, sample_rate, fps=fps, shots=shots)


def span_problem(start_frame, num_frames, total):
    """Why frames [start_frame, start_frame + num_frames) are not all in a
    clip of `total` frames, or None. The one owner of the span rule: a span
    past the clip's end is refused, never shortened. `trim_video` raises it,
    and the static check of a regional edit whose source is a trim of a file
    that resolves now (dw/hold_audio.py) asks it before anything runs."""
    end = start_frame + num_frames
    if end <= total:
        return None
    return (
        f"start_frame {start_frame} + num_frames {num_frames} reaches frame "
        f"{end}, but the clip has only {total} frames"
    )


def pad_to_problem(num_frames, pad_to):
    """Why a pad_to cannot lengthen a num_frames span, or None."""
    if pad_to is None or num_frames is None or pad_to >= num_frames:
        return None
    return (
        f"{COMMAND}: pad_to {pad_to} is shorter than num_frames {num_frames} - "
        "pad_to only lengthens the span, so it must be at least num_frames"
    )


def _literal_whole(value):
    return isinstance(value, int) and not isinstance(value, bool)


def trim_video_errors(arguments):
    """[(argument, message)] for a literal pad_to below a literal num_frames."""
    num_frames, pad_to = arguments.get("num_frames"), arguments.get("pad_to")
    if not (_literal_whole(num_frames) and _literal_whole(pad_to)):
        return []
    if pad_to < 1 or num_frames < 1:
        return []  # the domain's to refuse, once
    problem = pad_to_problem(num_frames, pad_to)
    return [("pad_to", problem)] if problem else []


def _span_of_audio(audio, sample_rate, fps, start_frame, end, held=0):
    """The track over frames [start_frame, end), cut at the track's own rate,
    and run on in silence over `held` more frames."""
    if not sample_rate:
        raise ValueError(
            f"{COMMAND}: the clip has audio with no sample rate, so its track "
            "cannot be cut to the span"
        )
    if not fps:
        raise ValueError(
            f"{COMMAND} needs 'fps' to cut the clip's audio to the span - "
            "the clip does not carry a frame rate"
        )
    waveform = as_channels_samples(audio)
    first = frames_to_samples(start_frame, fps, sample_rate)
    last = frames_to_samples(end, fps, sample_rate)
    span = waveform[:, first:last]
    if not held:
        return span
    padded = frames_to_samples(end + held, fps, sample_rate) - first
    return np.pad(span, ((0, 0), (0, max(0, padded - span.shape[1]))))


# ------------------------------------------------------------ replace_cuts


def _whole(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _cut_label(cuts, index):
    name = cuts[index].get("name")
    return f"cuts[{index}] ('{name}')" if isinstance(name, str) else f"cuts[{index}]"


def cut_list_problem(cuts):
    """Why a list of cuts ({start_frame, num_frames, ...}) cannot be put back
    into one take, or None: each must start at or after the end of the one
    before it. Entries whose span is not a literal whole number are not judged
    (a reference resolves later, and the task refuses a bad one itself)."""
    if not isinstance(cuts, list):
        return None
    previous = None  # (index, start, end) of the last entry with a known span
    for index, cut in enumerate(cuts):
        if not isinstance(cut, dict):
            continue
        start, count = cut.get("start_frame"), cut.get("num_frames")
        if not (_whole(start) and _whole(count)) or start < 0 or count < 1:
            continue
        if previous is not None and start < previous[2]:
            return _order_problem(cuts, index, start, previous)
        previous = (index, start, start + count)
    return None


def _order_problem(cuts, index, start, previous):
    before, before_start, before_end = previous
    here, there = _cut_label(cuts, index), _cut_label(cuts, before)
    if start < before_start:
        return (
            f"{here} starts at frame {start}, before {there} at frame "
            f"{before_start}: list the cuts in the take's order, each starting "
            "at or after the end of the one before. A cut to leave as it is "
            "is left out of the list - its frames stay the take's"
        )
    return (
        f"{here} starts at frame {start}, inside {there}, which runs to frame "
        f"{before_end - 1}: two cuts cannot cover the same frames. Start it at "
        f"frame {before_end} or later, shorten {there}, or merge the two into "
        "one cut"
    )


def replace_cuts_errors(arguments):
    """[(argument, message)] for a literal cut list out of the take's order or
    overlapping, and a literal clip list of another length."""
    cuts, clips = arguments.get("cuts"), arguments.get("clips")
    errors = []
    problem = cut_list_problem(cuts)
    if problem:
        errors.append(("cuts", f"{REPLACE}: {problem}"))
    if isinstance(cuts, list) and isinstance(clips, list) and len(cuts) != len(clips):
        errors.append(("clips", _count_problem(len(clips), len(cuts))))
    return errors


def _count_problem(clips, cuts):
    return (
        f"{REPLACE}: {clips} clip(s) for {cuts} cut(s) - 'clips' holds one "
        "re-rendered clip per entry of 'cuts', in the same order"
    )


def _cut_span(cuts, index, total):
    """(start_frame, num_frames) of one cut, inside a take of `total` frames."""
    cut = cuts[index]
    label = _cut_label(cuts, index) if isinstance(cut, dict) else f"cuts[{index}]"
    if not isinstance(cut, dict):
        raise ValueError(
            f"{REPLACE}: {label} must be an object with start_frame and num_frames"
        )
    start = whole_number(cut.get("start_frame"), "start_frame", REPLACE)
    count = whole_number(cut.get("num_frames"), "num_frames", REPLACE)
    if not (_whole(start) and _whole(count)) or start < 0 or count < 1:
        raise ValueError(
            f"{REPLACE}: {label} needs a whole start_frame of 0 or more and a "
            f"whole num_frames of 1 or more, got {cut.get('start_frame')!r} "
            f"and {cut.get('num_frames')!r}"
        )
    problem = span_problem(int(start), int(count), total)
    if problem:
        raise ValueError(f"{REPLACE}: {label}: {problem}")
    return int(start), int(count)


def _read_clip(video):
    if isinstance(video, VideoFileReference):
        return load_audio_video(video.path)
    if isinstance(video, str):
        return load_audio_video(video)
    return video


def replace_cuts(video, clips, cuts):
    """Task command: put re-rendered cuts back into the take they came from.

    Args:
        video: The take - a path, an asset:/output: reference (read with its
            audio), or an earlier step's video
        clips: One re-rendered clip per entry of 'cuts', in the same order.
            A clip's first num_frames frames are used, so a clip rendered
            longer (trim_video's pad_to) needs no trim first
        cuts: The cuts the clips were rendered from, as detect_cuts lists
            them: objects with start_frame and num_frames (other fields are
            ignored), in the take's order and not overlapping. Any subset of
            a take's cuts: one left out stays as the take has it

    Returns:
        The take as one AudioVideo, the same length: frames [start_frame,
        start_frame + num_frames) of each cut are the clip's, every other
        frame, the whole track, the fps and the shots are the take's
    """
    if video is None:
        raise ValueError(f"{REPLACE} needs 'video' - the take the cuts came from")
    if not isinstance(cuts, list) or not cuts:
        raise ValueError(f"{REPLACE} needs 'cuts' - a list of the cuts to put back")
    clips = list(clips) if isinstance(clips, (list, tuple)) else [clips]
    if len(clips) != len(cuts):
        raise ValueError(_count_problem(len(clips), len(cuts)))
    problem = cut_list_problem(cuts)
    if problem:
        raise ValueError(f"{REPLACE}: {problem}")

    take = _read_clip(video)
    frames = list(frames_as_pil_list(take))
    if not frames:
        raise ValueError(f"{REPLACE}: the take has no frames")
    size = frames[0].size
    for index, clip in enumerate(clips):
        start, count = _cut_span(cuts, index, len(frames))
        rendered = frames_as_pil_list(_read_clip(clip))
        label = _cut_label(cuts, index)
        if len(rendered) < count:
            raise ValueError(
                f"{REPLACE}: the clip for {label} has {len(rendered)} frames, "
                f"fewer than the cut's num_frames {count}"
            )
        if rendered[0].size != size:
            raise ValueError(
                f"{REPLACE}: the clip for {label} is "
                f"{rendered[0].size[0]}x{rendered[0].size[1]} and the take is "
                f"{size[0]}x{size[1]} - render the cuts at the take's size"
            )
        frames[start : start + count] = rendered[:count]
    logger.info(f"Put {len(clips)} cut(s) back into a take of {len(frames)} frames")

    clip = take if isinstance(take, AudioVideo) else None
    fps = (clip.fps if clip else None) or getattr(clips[0], "fps", None)
    return AudioVideo(
        frames,
        clip.audio if clip else None,
        clip.sample_rate if clip else None,
        fps=fps,
        shots=carried_shots(take),
    )


@register_command(
    REPLACE,
    implementation="dw.tasks.trim.replace_cuts",
    static_check=replace_cuts_errors,
)
def _handle_replace_cuts(task, arguments, previous_pipelines):
    """Put re-rendered cuts back into the take, its other frames and audio kept"""
    logger.debug("Replacing cuts in a take")
    return replace_cuts(**arguments)


@register_command(
    COMMAND,
    implementation="dw.tasks.trim.trim_video",
    domains={
        "start_frame": NON_NEGATIVE,
        "num_frames": POSITIVE,
        "fps": POSITIVE,
        "pad_to": POSITIVE,
    },
    whole_numbers=("start_frame", "num_frames", "pad_to"),
    static_check=trim_video_errors,
)
def _handle_trim_video(task, arguments, previous_pipelines):
    """Keep a span of a video's frames, and its audio over the same span"""
    logger.debug("Trimming a video")
    return trim_video(**arguments)
