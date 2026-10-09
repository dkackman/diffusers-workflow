"""`read_frame_range` is byte for byte a full decode's slice, from one
keyframe seek, and refuses a range the file cannot fill (#695)."""

import av
import numpy
import pytest

from dw.media import ShortFrameRange, count_video_frames, read_frame_range
from tests.test_media_frames import write_ramp_mp4, write_shifted_ramp_mp4


def full_decode(path):
    """Every frame of the file, stacked: the reference for byte parity."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return numpy.stack(
            [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
        )


def assert_parity(path, ranges):
    reference = full_decode(path)
    for start, stop in ranges:
        got = read_frame_range(str(path), start, stop)
        assert got.dtype == numpy.uint8
        assert got.shape == (stop - start,) + reference.shape[1:]
        assert numpy.array_equal(got, reference[start:stop]), (start, stop)


class TestSingleKeyframe:
    def test_ranges_equal_the_full_decode(self, tmp_path):
        """24 frames under x264's default gop: one keyframe, at the top."""
        path = tmp_path / "ramp.mp4"
        write_ramp_mp4(path, frames=24)
        assert_parity(path, [(0, 24), (0, 1), (5, 9), (10, 20), (22, 24)])


class TestKeyframeStraddling:
    def test_g10_ranges_equal_the_full_decode(self, tmp_path):
        path = tmp_path / "g10.mp4"
        write_shifted_ramp_mp4(path, offset=0)
        assert_parity(path, [(8, 13), (9, 21), (19, 20), (30, 45), (0, 60)])


class TestNonZeroStart:
    def test_shifted_clip_equals_the_full_decode(self, tmp_path):
        path = tmp_path / "shifted.mp4"
        write_shifted_ramp_mp4(path, offset=5)
        with av.open(str(path)) as container:
            assert container.streams.video[0].start_time
        assert_parity(path, [(0, 4), (0, 12), (8, 13), (19, 20), (30, 45)])


class TestLastFrames:
    @pytest.mark.parametrize("offset", [0, 5])
    def test_a_range_ending_at_the_count(self, tmp_path, offset):
        path = tmp_path / "clip.mp4"
        write_shifted_ramp_mp4(path, offset=offset)
        total = count_video_frames(str(path))
        assert_parity(path, [(total - 3, total), (total - 1, total)])


class TestCountMismatch:
    def test_count_video_frames_is_the_real_count(self, tmp_path):
        path = tmp_path / "clip.mp4"
        write_shifted_ramp_mp4(path, frames=60)
        assert count_video_frames(str(path)) == 60

    @pytest.mark.parametrize("start, stop", [(55, 61), (0, 100), (60, 61), (70, 80)])
    def test_a_stop_past_the_end_raises_short(self, tmp_path, start, stop):
        path = tmp_path / "clip.mp4"
        write_shifted_ramp_mp4(path, frames=60)
        with pytest.raises(ShortFrameRange) as raised:
            read_frame_range(str(path), start, stop)
        assert raised.value.decoded == 60
        assert str(path) in str(raised.value)
        assert isinstance(raised.value, ValueError)


class TestBadArguments:
    @pytest.mark.parametrize("start, stop", [(-1, 5), (5, 5), (6, 5), (0, 0)])
    def test_an_empty_or_negative_range_is_refused(self, tmp_path, start, stop):
        path = tmp_path / "clip.mp4"
        write_ramp_mp4(path)
        with pytest.raises(ValueError) as raised:
            read_frame_range(str(path), start, stop)
        assert not isinstance(raised.value, ShortFrameRange)
