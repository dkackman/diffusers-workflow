"""Hold a generated clip's framing still.

A video model drifts. Ask it for a locked camera and the framing still wanders
over a few seconds - a slow slide of the whole picture that nothing in the
prompt asked for. It goes unnoticed inside a single shot, and becomes obvious
the moment two shots are cut together and the subject jumps back to where it
started.

The wander is a global translation, so phase correlation measures it: the
cross-power spectrum of consecutive frames peaks at the shift between them.
Accumulating those shifts gives the clip's drift, and shifting every frame back
by it puts the framing where it began.
"""

import logging

import numpy as np
from PIL import Image

from ..result import AudioVideo
from .video_utils import frames_as_pil_list, load_audio_video

logger = logging.getLogger("dw")


def _pair_shift(previous, following, window):
    """The translation between two grayscale frames, in whole pixels."""
    height, width = previous.shape
    a = np.fft.rfft2(previous * window)
    b = np.fft.rfft2(following * window)
    cross = a * np.conj(b)
    cross /= np.abs(cross) + 1e-8
    correlation = np.fft.irfft2(cross, s=previous.shape)

    # The peak of this cross-power spectrum sits at the negative of the
    # displacement, so it is negated here and every caller reads (dx, dy) as
    # "how far the picture moved between these two frames".
    peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    dy = peak[0] - height if peak[0] > height // 2 else peak[0]
    dx = peak[1] - width if peak[1] > width // 2 else peak[1]
    return -dx, -dy


def _moving_average(values, window):
    """Trajectory smoothed over `window` frames, with the ends held."""
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    return np.stack(
        [
            np.convolve(padded[:, axis], kernel, mode="valid")[: len(values)]
            for axis in (0, 1)
        ],
        axis=1,
    )


def stabilize_video(clip, smooth=0):
    """Task command: remove a generated clip's accumulated framing drift.

    Args:
        clip: The video - a frame list, a frame array or tensor, an
            AudioVideo, whose soundtrack is carried through untouched, or the
            path or URL of a video file, which is read with its audio, so a
            shot an earlier run wrote can be steadied without regenerating it.
            The argument is deliberately not called "video": the engine loads
            an argument by that name itself, as bare frames, which would strip
            the soundtrack off before this ever saw it
        smooth: 0 locks the framing to the first frame, which is what a shot
            generated from a pinned keyframe wants. A window in frames instead
            removes only the wander faster than that window, so a slow
            deliberate move survives and the drift around it does not
    Returns:
        The stabilized clip, cropped to the region every frame covers and
        resized back to its original size - an AudioVideo when one came in
    """
    if isinstance(clip, str):
        clip = load_audio_video(clip)

    frames = frames_as_pil_list(clip)
    if len(frames) < 2:
        return clip

    width, height = frames[0].size
    gray = [np.asarray(f.convert("L"), dtype=np.float32) for f in frames]
    window = np.outer(np.hanning(height), np.hanning(width))

    trajectory = np.zeros((len(frames), 2), dtype=np.float32)
    for index in range(1, len(frames)):
        dx, dy = _pair_shift(gray[index - 1], gray[index], window)
        trajectory[index] = trajectory[index - 1] + (dx, dy)

    target = _moving_average(trajectory, smooth) if smooth > 1 else 0.0
    correction = np.rint(trajectory - target).astype(np.int32)
    logger.debug(
        f"Stabilizing {len(frames)} frames, peak drift "
        f"{np.abs(correction).max()}px of {max(width, height)}"
    )
    if not correction.any():
        return clip

    # Shifting frames back uncovers their edges, so keep only the rectangle
    # every frame still covers, then put that back at the original size.
    shifts = -correction
    left = int(max(0, shifts[:, 0].max()))
    right = int(width + min(0, shifts[:, 0].min()))
    top = int(max(0, shifts[:, 1].max()))
    bottom = int(height + min(0, shifts[:, 1].min()))

    held = []
    for frame, (sx, sy) in zip(frames, shifts):
        moved = np.roll(np.asarray(frame), (int(sy), int(sx)), axis=(0, 1))
        held.append(
            Image.fromarray(moved[top:bottom, left:right]).resize(
                (width, height), Image.LANCZOS
            )
        )

    if isinstance(clip, AudioVideo):
        return AudioVideo(held, clip.audio, clip.sample_rate)
    return held
