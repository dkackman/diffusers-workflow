"""Tests for lip_sync_score and measure_lip_sync (dw/tasks/lip_sync.py).

The mouth box comes from YuNet's landmarks in the real thing; here the
module's `_mouth_box` seam is replaced, so no weights are read. A synthetic
clip's mouth box steps in brightness by a known envelope each frame, and the
audio is a tone whose RMS follows the same envelope: in sync, the two
correlate at lag 0; with the audio delayed, at the delay.
"""

import numpy as np
import pytest

import dw.tasks.lip_sync as ls
from dw.media_types import AudioVideo
from dw.tasks.task import task_command_info

FPS = 24
RATE = 16000
FRAMES = 96
BOX = (16, 16, 32, 32)
STEP = 60


def envelope(seed=3):
    """One loudness per frame, in [0.1, 1], independent frame to frame."""
    return np.random.default_rng(seed).uniform(0.1, 1.0, FRAMES)


def mouth_frames(levels):
    """Frames whose mouth box changes by STEP * level from the frame before:
    the step turns back toward mid-grey, so the brightness stays in range."""
    frames, brightness = [], 128.0
    for level in levels:
        brightness += -STEP * level if brightness > 128 else STEP * level
        pixels = np.full((48, 48, 3), 128, dtype=np.uint8)
        x0, y0, x1, y1 = BOX
        pixels[y0:y1, x0:x1] = int(round(brightness))
        frames.append(pixels)
    return frames


def tone(levels, delay_seconds=0.0):
    """A 220 Hz tone whose amplitude is `levels`, one per frame, delayed."""
    per_frame = RATE / FPS
    count = int(round(FRAMES * per_frame))
    times = np.arange(count) / RATE
    amplitude = levels[
        np.minimum((np.arange(count) / per_frame).astype(int), FRAMES - 1)
    ]
    wave = amplitude * np.sin(2 * np.pi * 220 * times)
    shift = int(round(delay_seconds * RATE))
    if shift:
        wave = np.concatenate([np.zeros(shift), wave])[:count]
    return wave.astype(np.float32)[np.newaxis, :]


@pytest.fixture
def fixed_mouth(monkeypatch):
    monkeypatch.setattr(ls, "_mouth_box", lambda rgb, device=None: BOX)


def test_synced_mouth_scores_high_at_zero_lag(fixed_mouth):
    levels = envelope()
    answer = ls.lip_sync_score(mouth_frames(levels), tone(levels), RATE, FPS)
    assert answer["correlation"] > 0.8
    assert abs(answer["lag_seconds"]) < 1 / FPS
    assert answer["frames_with_face"] == FRAMES


def test_shifted_audio_scores_at_its_shift(fixed_mouth):
    levels = envelope()
    answer = ls.lip_sync_score(
        mouth_frames(levels), tone(levels, delay_seconds=0.3), RATE, FPS
    )
    assert answer["lag_seconds"] == pytest.approx(0.3, abs=1.5 / FPS)
    assert answer["correlation"] > 0.8


def test_audio_from_another_clip_scores_low(fixed_mouth):
    levels = envelope()
    answer = ls.lip_sync_score(mouth_frames(levels), tone(envelope(9)), RATE, FPS)
    assert answer["correlation"] < 0.5


def test_no_face_returns_none(monkeypatch):
    monkeypatch.setattr(ls, "_mouth_box", lambda rgb, device=None: None)
    levels = envelope()
    answer = ls.lip_sync_score(mouth_frames(levels), tone(levels), RATE, FPS)
    assert answer == {"correlation": None, "lag_seconds": None, "frames_with_face": 0}


def test_the_mouth_box_comes_from_the_mouth_corners(monkeypatch):
    # YuNet's row: box, right eye, left eye, nose, right and left mouth corner, score
    row = np.array(
        [0, 0, 40, 40, 10, 10, 30, 10, 20, 20, 12, 30, 28, 32, 0.9], dtype=np.float32
    )
    monkeypatch.setattr(ls, "_faces", lambda rgb, device=None: [row])
    x0, y0, x1, y1 = ls._mouth_box(np.zeros((48, 48, 3), dtype=np.uint8))
    # The corners span x 12-28 (distance ~16.1); the box runs down that far
    assert (x0, x1) == (12, 28)
    assert y0 == 30 and y1 == pytest.approx(32 + 16, abs=1)


def test_the_task_reads_the_videos_own_track(fixed_mouth):
    levels = envelope()
    video = AudioVideo(mouth_frames(levels), tone(levels), RATE, fps=FPS)
    answer = ls.measure_lip_sync(video)
    assert answer["correlation"] > 0.8
    assert answer["frames_with_face"] == FRAMES


def test_the_task_takes_a_separate_audio(fixed_mouth):
    levels = envelope()
    video = AudioVideo(mouth_frames(levels), None, None, fps=FPS)
    shifted = AudioVideo(None, tone(levels, delay_seconds=0.3), RATE, fps=FPS)
    answer = ls.measure_lip_sync(video, audio=shifted)
    assert answer["lag_seconds"] == pytest.approx(0.3, abs=1.5 / FPS)


def test_the_task_is_registered_as_json():
    info = task_command_info("measure_lip_sync")
    assert info["returns"] == "json"
    assert info["implementation"] == "dw.tasks.lip_sync.measure_lip_sync"
