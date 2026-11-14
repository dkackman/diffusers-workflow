"""Picking the sharpest of a segment's last frames as the next keyframe (#1032).

A motion-blurred final frame passes its softness to the next segment as that
segment's keyframe. `carry_pick: {"sharpest": N}` looks at the last N frames
instead. Numpy only: the sharpness is the variance of a 3x3 Laplacian on luma.

The idea comes from harlixay7/Kinoloom (MIT; concept only, no code).
"""

import numpy


def _luma(frame):
    pixels = numpy.asarray(frame, dtype=numpy.float64)
    if pixels.ndim == 3 and pixels.shape[-1] >= 3:
        return 0.299 * pixels[..., 0] + 0.587 * pixels[..., 1] + 0.114 * pixels[..., 2]
    if pixels.ndim == 3:
        return pixels[..., 0]
    return pixels


def sharpness(frame):
    """The variance of a 3x3 Laplacian on the frame's luma - higher is sharper.

    Accepts a PIL image or an array. A frame under 3 pixels either way has no
    interior to measure and scores 0.
    """
    luma = _luma(frame)
    if luma.shape[0] < 3 or luma.shape[1] < 3:
        return 0.0
    laplacian = (
        luma[:-2, 1:-1]
        + luma[2:, 1:-1]
        + luma[1:-1, :-2]
        + luma[1:-1, 2:]
        - 4.0 * luma[1:-1, 1:-1]
    )
    return float(laplacian.var())


def sharpest_index(frames, window):
    """Index from the end of the sharpest of the last `window` frames.

    0 is the last frame. Ties go to the latest frame; a window longer than the
    clip uses the whole clip.
    """
    count = min(int(window), len(frames))
    best_back, best_score = 0, None
    for back in range(count):
        score = sharpness(frames[len(frames) - 1 - back])
        if best_score is None or score > best_score:
            best_back, best_score = back, score
    return best_back
