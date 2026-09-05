"""Tests for dw.tasks.stabilize - removing a generated clip's framing drift.

The sign convention gets its own test: the peak of the cross-power spectrum
sits at the negative of the displacement, and reading it backwards doubles the
drift instead of removing it, which still produces a plausible-looking number.
"""

import numpy
import pytest
from PIL import Image

from dw.result import AudioVideo
from dw.tasks.stabilize import _pair_shift, stabilize_video


def _texture(size=192, seed=0):
    rng = numpy.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=numpy.uint8)


def _drifting_clip(frames=12, per_frame=(2, 3), size=192):
    """A still picture slid by a fixed amount every frame."""
    base = _texture(size)
    return [
        Image.fromarray(
            numpy.roll(base, (i * per_frame[0], i * per_frame[1]), axis=(0, 1))
        )
        for i in range(frames)
    ]


def _frame_to_frame_change(frames):
    gray = numpy.stack(
        [numpy.asarray(f.convert("L"), dtype=numpy.float32) for f in frames]
    )
    return float(numpy.abs(numpy.diff(gray, axis=0)).mean())


class TestPairShift:
    def test_reports_the_displacement_not_its_negative(self):
        base = _texture(seed=1)[:, :, 0].astype(numpy.float32)
        window = numpy.outer(numpy.hanning(192), numpy.hanning(192))
        moved = numpy.roll(base, (5, 7), axis=(0, 1))
        assert _pair_shift(base, moved, window) == (7, 5)

    def test_reports_negative_movement(self):
        base = _texture(seed=2)[:, :, 0].astype(numpy.float32)
        window = numpy.outer(numpy.hanning(192), numpy.hanning(192))
        moved = numpy.roll(base, (-4, -9), axis=(0, 1))
        assert _pair_shift(base, moved, window) == (-9, -4)

    def test_a_still_pair_reports_no_movement(self):
        base = _texture(seed=3)[:, :, 0].astype(numpy.float32)
        window = numpy.outer(numpy.hanning(192), numpy.hanning(192))
        assert _pair_shift(base, base, window) == (0, 0)


class TestStabilizeVideo:
    def test_a_steadily_drifting_clip_is_held_still(self):
        frames = _drifting_clip()
        assert _frame_to_frame_change(frames) > 1.0
        assert _frame_to_frame_change(stabilize_video(frames)) == pytest.approx(
            0.0, abs=1e-6
        )

    def test_output_keeps_the_original_frame_count_and_size(self):
        frames = _drifting_clip()
        held = stabilize_video(frames)
        assert len(held) == len(frames)
        assert held[0].size == frames[0].size

    def test_a_clip_that_does_not_drift_is_returned_unchanged(self):
        base = Image.fromarray(_texture(seed=4))
        frames = [base] * 6
        assert stabilize_video(frames) is frames

    def test_a_single_frame_is_returned_unchanged(self):
        frames = [Image.fromarray(_texture(seed=5))]
        assert stabilize_video(frames) is frames

    def test_a_soundtrack_is_carried_through(self):
        audio = numpy.zeros((2, 4410), dtype=numpy.float32)
        clip = AudioVideo(_drifting_clip(), audio, 44100)
        held = stabilize_video(clip)
        assert isinstance(held, AudioVideo)
        assert held.sample_rate == 44100
        assert held.audio.shape == audio.shape

    def test_smoothing_keeps_a_deliberate_move(self):
        # A long window treats the whole slide as intended camera motion and
        # leaves it alone, rather than locking the framing to the first frame.
        frames = _drifting_clip(frames=16)
        held = stabilize_video(frames, smooth=64)
        assert _frame_to_frame_change(held) > 1.0


class TestStabilizeVideoFromAFile:
    def test_a_path_is_loaded_with_its_audio(self, monkeypatch):
        import numpy as np

        from dw.result import AudioVideo
        from dw.tasks import stabilize as stabilize_module

        clip = AudioVideo(
            _drifting_clip(), np.zeros((2, 4410), dtype=np.float32), 44100
        )
        monkeypatch.setattr(stabilize_module, "load_audio_video", lambda path: clip)
        held = stabilize_video("shot.mp4")
        assert isinstance(held, AudioVideo)
        assert _frame_to_frame_change(held.frames) == pytest.approx(0.0, abs=1e-6)
