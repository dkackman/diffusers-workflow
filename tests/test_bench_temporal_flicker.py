"""bench/temporal_flicker.py (#1749): the filter and its two measures on
synthetic clips - static grain flickers less after it, a translating textured
square keeps its gradient energy and leaves no ghost trail."""

import sys
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))
import temporal_flicker as tf  # noqa: E402

H, W = 96, 128


def _texture(h, w, seed):
    """A smooth-but-structured texture DIS can lock onto: blurred noise."""
    import cv2

    rng = np.random.default_rng(seed)
    noise = rng.uniform(0, 255, size=(h, w)).astype(np.float32)
    return cv2.GaussianBlur(noise, (0, 0), 2.0)


def _rgb(luma):
    luma = np.clip(np.rint(luma), 0, 255).astype(np.uint8)
    return np.repeat(luma[..., None], 3, axis=-1)


def static_grain(count=16, sigma=6.0, seed=1749):
    rng = np.random.default_rng(seed)
    base = 60 + 0.5 * _texture(H, W, seed)
    return np.stack(
        [_rgb(base + rng.normal(0, sigma, size=(H, W))) for _ in range(count)]
    )


def translating_square(count=12, step=3, size=32, seed=1747):
    """A textured square sliding right `step` px a frame over a flat background."""
    texture = 40 + 0.7 * _texture(size, size, seed)
    frames = []
    boxes = []
    for t in range(count):
        frame = np.full((H, W), 128.0, dtype=np.float32)
        x = 10 + step * t
        y = (H - size) // 2
        frame[y : y + size, x : x + size] = texture
        frames.append(_rgb(frame))
        boxes.append((y, x))
    return np.stack(frames), boxes, size


def test_static_grain_flicker_falls():
    before = static_grain()
    after = tf.denoise(before)
    result = tf.measure(before, after)
    assert result["static_share"] > 0.5
    assert result["flicker"]["before"] > 3.0
    assert result["flicker"]["ratio"] < 0.7


def test_translating_square_keeps_gradient_and_leaves_no_trail():
    before, boxes, size = translating_square()
    after = tf.denoise(before)
    result = tf.measure(before, after)
    assert result["moving_share"] > 0.0
    assert result["gradient_energy"]["ratio"] >= tf.GAIN_KEEP
    assert result["edge_energy"]["ratio"] >= tf.GAIN_KEEP
    # The background the square has just left must come back as background:
    # a ghost trail is the square's texture blended into those pixels.
    for t in range(1, len(before)):
        (y, x0), (_, x1) = boxes[t - 1], boxes[t]
        trail = after[t, y : y + size, x0:x1, 0].astype(np.float32)
        assert np.abs(trail - 128.0).max() <= 4.0, t


def test_zero_strength_is_identity():
    before = static_grain(count=4)
    after = tf.denoise(before, luma=0.0, chroma=0.0)
    assert np.array_equal(after, before)


def test_one_frame_is_returned_unchanged():
    before = static_grain(count=1)
    assert np.array_equal(tf.denoise(before), before)


def test_gain_needs_both_measures():
    before = static_grain(count=6)
    result = tf.measure(before, before)
    assert result["flicker"]["ratio"] == 1.0
    assert result["gain"] is False
