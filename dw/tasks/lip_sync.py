"""measure_lip_sync: does a speaker's mouth move with the audio?

`measure_sync` lines an audio track up against its source slice; this lines
the picture up against the audio. In each frame YuNet finds the face, and the
mouth box is cut from its two mouth-corner landmarks - the corners' span
across, from the higher corner down to the lower one plus the corner
distance, which takes in the lower lip and jaw that open with speech. The
mouth signal is the mean absolute grey-level change inside that box from the
frame before (the box of the later frame, applied to both). The audio signal
is its RMS envelope at the frame rate: one value per frame's span of
samples. The two are Pearson-correlated over the frames that had a face, at
whole-frame lags. `correlation` is the best r within SYNC_WINDOW_FRAMES of
lag 0; `lag_seconds` is the best lag within ±`max_lag_seconds` (positive when
the audio is LATE against the mouth, as `measure_sync`'s is) and
`best_correlation` its r.

A coarse take-level indicator, not a verdict, and not a chain measure (ruling
R9): on 70 real H3 dialogue segments no threshold separated one segment's
own audio from audio 0.5 s late or from another segment's
(bench/records/narrated.md, "Sync calibration"). The best lag tells a late
track (it clusters at +12..+14 frames) far better than one correlation does.
A still face, a face in profile or a mouth hidden by a hand reads low on
good sync, and a mouth that moves to music reads high. `lip_sync_score` is
the measure on frames and a waveform in memory, which a calibration script
calls; `measure_lip_sync` is the task.

Models: YuNet through `face_track.face_detector_model`, the one loader.
"""

import logging
import math

import numpy as np

from .registry import register_command
from ..task_domains import POSITIVE

logger = logging.getLogger("dw")

COMMAND = "measure_lip_sync"

DEFAULT_MAX_LAG_SECONDS = 1.0
# The most a task may search, either way: the search is one Pearson r per
# whole-frame lag, and a sync error past two seconds is no lip-sync question
MAX_LAG_SECONDS_CAP = 2.0
# The lags either side of 0 the take-level `correlation` is the best of: the
# calibration's good segments clustered at 0..+2 frames
SYNC_WINDOW_FRAMES = 3
# The detector score below which a face is ignored, as face_drift's
DETECTOR_CONFIDENCE = 0.6
# The fewest frame pairs a lag's correlation is read over
MIN_PAIRS = 8
PLACES = 3

# YuNet's row: x, y, w, h, then (x, y) of the right eye, left eye, nose tip,
# right mouth corner and left mouth corner, then the score
MOUTH_CORNERS = slice(10, 14)


def _faces(rgb, device=None):
    """Every face YuNet finds in an RGB array, as full rows."""
    import cv2

    from .face_track import (
        DEFAULT_DETECTOR_FILE,
        DEFAULT_DETECTOR_REPO,
        face_detector_model,
    )

    model = face_detector_model(
        DEFAULT_DETECTOR_REPO, DEFAULT_DETECTOR_FILE, DETECTOR_CONFIDENCE, device
    )
    height, width = rgb.shape[:2]
    model.setInputSize((width, height))
    _, faces = model.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return [] if faces is None else list(faces)


def _mouth_box(rgb, device=None):
    """(x0, y0, x1, y1) of the mouth of the largest face in an RGB array -
    the mouth corners' span across, run down by the corner distance - or
    None when no face is found."""
    faces = _faces(rgb, device)
    if not faces:
        return None
    row = max(faces, key=lambda face: float(face[2]) * float(face[3]))
    rx, ry, lx, ly = (float(v) for v in row[MOUTH_CORNERS])
    distance = math.hypot(lx - rx, ly - ry)
    height, width = rgb.shape[:2]
    x0, x1 = max(0, int(round(min(rx, lx)))), min(width, int(round(max(rx, lx))))
    y0 = max(0, int(round(min(ry, ly))))
    y1 = min(height, int(round(max(ry, ly) + distance)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _grey(frame):
    """A frame (PIL image or array) as a float32 grey array."""
    rgb = np.asarray(frame.convert("RGB") if hasattr(frame, "convert") else frame)
    if rgb.ndim == 2:
        return rgb.astype(np.float32), np.stack([rgb] * 3, axis=-1)
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).round().astype(np.uint8)
    rgb = np.ascontiguousarray(rgb[..., :3])
    return rgb.astype(np.float32).mean(axis=-1), rgb


def _mouth_signal(frames, device=None):
    """(signal, frames_with_face): per frame, the mean absolute grey change
    inside its mouth box from the frame before - NaN for frame 0 and for a
    frame with no face."""
    signal = np.full(len(frames), np.nan)
    found, before = 0, None
    for index, frame in enumerate(frames):
        grey, rgb = _grey(frame)
        box = _mouth_box(rgb, device)
        if box is not None:
            found += 1
            if before is not None and before.shape == grey.shape:
                x0, y0, x1, y1 = box
                change = np.abs(grey[y0:y1, x0:x1] - before[y0:y1, x0:x1])
                signal[index] = float(change.mean())
        before = grey
    return signal, found


def _envelope(waveform, sample_rate, fps, count):
    """The audio's RMS over each frame's span of samples, `count` frames."""
    mono = np.asarray(waveform, dtype=np.float64)
    if mono.ndim == 2:
        mono = mono.mean(axis=0)
    per_frame = sample_rate / float(fps)
    envelope = np.full(count, np.nan)
    for index in range(count):
        window = mono[
            int(round(index * per_frame)) : int(round((index + 1) * per_frame))
        ]
        if window.size:
            envelope[index] = math.sqrt(float(np.mean(np.square(window))))
    return envelope


def _correlation(first, second):
    """Pearson's r over the pairs where both are finite, or None."""
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < MIN_PAIRS:
        return None
    a, b = first[valid], second[valid]
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _lag_correlations(signal, envelope, reach):
    """{lag: r} for every whole-frame lag within ±reach with an answer: the
    mouth at frame i against the audio at frame i + lag."""
    count = len(signal)
    answers = {}
    for lag in range(-reach, reach + 1):
        if lag >= 0:
            mouth, audio = signal[: count - lag], envelope[lag:]
        else:
            mouth, audio = signal[-lag:], envelope[: count + lag]
        r = _correlation(mouth, audio)
        if r is not None:
            answers[lag] = r
    return answers


def _best(answers, reach):
    """(lag, r) of the highest r within ±reach, or None."""
    within = [(r, lag) for lag, r in answers.items() if abs(lag) <= reach]
    if not within:
        return None
    r, lag = max(within)
    return lag, r


def lip_sync_score(
    frames,
    waveform,
    sample_rate,
    fps,
    max_lag_seconds=DEFAULT_MAX_LAG_SECONDS,
    device=None,
):
    """How well the mouth's motion follows the audio's loudness.

    Args:
        frames: The clip's frames, PIL images or RGB arrays
        waveform: Its audio, (channels, samples) or 1-D, aligned to frame 0
        sample_rate: The waveform's rate
        fps: The frames' rate
        max_lag_seconds: The largest lag searched, either way; never more
            lags than the frames leave MIN_PAIRS pairs at
        device: Where YuNet runs

    Returns:
        {correlation, lag_seconds, best_correlation, frames_with_face}: the
        best Pearson r of the mouth signal and the audio envelope within
        SYNC_WINDOW_FRAMES of lag 0, the best lag within max_lag_seconds in
        seconds (positive when the audio is late) and its r - each None with
        too few frames with a face, or a flat signal - and how many frames had
        a face
    """
    signal, found = _mouth_signal(list(frames), device)
    answer = {
        "correlation": None,
        "lag_seconds": None,
        "best_correlation": None,
        "frames_with_face": found,
    }
    if found == 0 or not sample_rate or not fps:
        return answer
    envelope = _envelope(waveform, sample_rate, fps, len(signal))
    # A lag past len - MIN_PAIRS leaves too few pairs to read: never search it
    ceiling = max(0, len(signal) - MIN_PAIRS)
    reach = min(int(round(float(max_lag_seconds) * float(fps))), ceiling)
    window = min(SYNC_WINDOW_FRAMES, ceiling)
    answers = _lag_correlations(signal, envelope, max(reach, window))
    near = _best(answers, window)
    best = _best(answers, reach)
    if near is not None:
        answer["correlation"] = round(near[1], PLACES)
    if best is not None:
        answer["lag_seconds"] = round(best[0] / float(fps), PLACES)
        answer["best_correlation"] = round(best[1], PLACES)
    return answer


def _clip(video):
    """A stored video (a path or VideoFileReference) as an AudioVideo; an
    in-memory one as it is."""
    from .video_utils import VideoFileReference, load_audio_video

    if isinstance(video, VideoFileReference):
        video = video.path
    if isinstance(video, str):
        from ..locations import validate_media_path

        return load_audio_video(validate_media_path(video, None, "a video to measure"))
    return video


def measure_lip_sync(
    video, audio=None, max_lag_seconds=DEFAULT_MAX_LAG_SECONDS, device=None
):
    """Task command: how well a speaker's mouth moves with the audio.

    Args:
        video: A stored video (asset:, output: or a path) or the video an
            earlier step returned
        audio: The audio to hold the mouth to - a file, a generated audio
            result, or a video carrying a soundtrack; by default the video's
            own track
        max_lag_seconds: The largest lag searched, either way - above 0 and
            at most MAX_LAG_SECONDS_CAP
        device: Where the face detector runs

    Returns:
        {correlation, lag_seconds, best_correlation, frames_with_face, frames,
        max_lag_seconds}: `lip_sync_score`'s answer, with the frames read and
        the search reach
    """
    from ..task_domains import real_number
    from .audio_utils import waveform_and_rate
    from .video_utils import frames_as_pil_list

    if video is None:
        raise ValueError(f"{COMMAND} needs 'video' - the clip of the speaker")
    reach = real_number(max_lag_seconds, "max_lag_seconds", COMMAND, required=True)
    if reach <= 0:
        raise ValueError(f"{COMMAND}: max_lag_seconds must be above 0")
    if reach > MAX_LAG_SECONDS_CAP:
        raise ValueError(
            f"{COMMAND}: max_lag_seconds must be at most {MAX_LAG_SECONDS_CAP}, "
            f"got {reach}"
        )
    clip = _clip(video)
    fps = getattr(clip, "fps", None)
    if not fps:
        raise ValueError(f"{COMMAND}: the video carries no frame rate to line up by")
    wave, rate = waveform_and_rate(clip if audio is None else audio, None, COMMAND)
    frames = frames_as_pil_list(clip)
    answer = lip_sync_score(frames, wave, rate, fps, reach, device)
    return {**answer, "frames": len(frames), "max_lag_seconds": float(reach)}


@register_command(
    COMMAND,
    implementation="dw.tasks.lip_sync.measure_lip_sync",
    returns="json",
    consumes_device=True,
    required_weights="dw.tasks.face_track.required_weights",
    domains={"max_lag_seconds": POSITIVE},
)
def _handle_measure_lip_sync(task, arguments, previous_pipelines):
    """Measure how well a speaker's mouth moves with the audio"""
    logger.debug("Measuring lip sync")
    return measure_lip_sync(device=task.device_for(arguments), **arguments)
