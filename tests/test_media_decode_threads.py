"""Every picture-decoding open in dw/media.py decodes on every core."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dw import media


class FakeFrame:
    def __init__(self):
        self.pts = 0

    def to_ndarray(self, format="rgb24"):
        return np.zeros((2, 2, 3), dtype=np.uint8)


def _container(frames=1):
    container = MagicMock()
    stream = MagicMock()
    stream.average_rate = 8
    stream.frames = 0
    stream.start_time = 0
    stream.time_base = 1
    stream.width = 2
    stream.height = 2
    del stream.thread_type  # set by the code under test, asserted below
    container.streams.video = [stream]
    container.streams.audio = []
    container.decode.return_value = [FakeFrame() for _ in range(frames)]
    container.__enter__.return_value = container
    container.__exit__.return_value = False
    return container, stream


@pytest.mark.parametrize(
    "call",
    [
        lambda: media.decode_rgb_frames("x.mp4"),
        lambda: media.decode_audio_video("x.mp4"),
        lambda: media.video_shape("x.mp4"),
        lambda: media.count_video_frames("x.mp4"),
        lambda: media.read_thumbnails_and_track("x.mp4", 2, 2),
    ],
)
def test_decoders_set_auto_threading(call):
    container, stream = _container()
    with patch.object(media.av, "open", return_value=container):
        try:
            call()
        except Exception:
            pass  # the fake need not satisfy the rest of the function
    assert stream.thread_type == "AUTO"


def test_video_shape_header_read_does_not_thread():
    container, stream = _container()
    stream.frames = 10
    with patch.object(media.av, "open", return_value=container):
        media.video_shape("x.mp4")
    assert not hasattr(stream, "thread_type")


@pytest.mark.parametrize(
    "call",
    [
        lambda: media.read_frames("x.mp4", [0]),
        lambda: media.read_frame_range("x.mp4", 0, 1),
    ],
)
def test_seeking_readers_stay_single_threaded(call):
    """Frame threads hold frames in flight; a backward seek after a decoded
    frame then yields stale ones (the recovery seek of read_frames)."""
    container, stream = _container()
    with patch.object(media.av, "open", return_value=container):
        try:
            call()
        except Exception:
            pass
    assert not hasattr(stream, "thread_type")
