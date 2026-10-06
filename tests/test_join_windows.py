"""join_windows (#601): processed windows blended back into one video the
source's length, with the source's soundtrack and one shot per window."""

import numpy
import pytest
import torch
from PIL import Image

from dw.media_types import AudioVideo
from dw.task_domains import (
    frames_to_samples,
    task_argument_errors,
    window_count,
    window_count_problem,
)
from dw.tasks.video_utils import VideoFileReference, frames_as_array
from dw.tasks.windows import _weights, join_windows, window_video

CURVES = ["cosine", "smoothstep", "linear"]


def _validate(arguments):
    """[(argument, message)] validate reports for one join_windows step."""
    step = {
        "name": "join",
        "task": {"command": "join_windows", "arguments": arguments},
        "result": {"content_type": "video/mp4"},
    }
    errors = task_argument_errors({"id": "windows", "steps": [step]})
    return [(error["path"].rsplit(".", 1)[-1], error["message"]) for error in errors]


def _ramp_clip(total, fps=24.0, sample_rate=48000, audio=True, size=(6, 4)):
    """A clip whose every frame differs: pixel values are a function of the
    frame, row, column and channel."""
    width, height = size
    index = numpy.arange(total).reshape(-1, 1, 1, 1)
    rows = numpy.arange(height).reshape(1, -1, 1, 1)
    cols = numpy.arange(width).reshape(1, 1, -1, 1)
    chans = numpy.arange(3).reshape(1, 1, 1, -1)
    frames = ((index * 7 + rows * 3 + cols * 5 + chans * 11) % 256).astype(numpy.uint8)
    if not audio:
        return AudioVideo(frames, None, None, fps=fps)
    samples = frames_to_samples(total, fps, sample_rate)
    track = numpy.arange(1, samples + 1, dtype=numpy.float32)
    return AudioVideo(frames, numpy.stack([track, -track]), sample_rate, fps=fps)


def _cut(clip, num_frames, overlap):
    count = window_count(len(clip.frames), num_frames, overlap)
    return [window_video(clip, i, num_frames, overlap) for i in range(count)]


def _flat(num_frames, level, size=(4, 4), audio=None, sample_rate=None):
    frames = numpy.full((num_frames, size[1], size[0], 3), level, dtype=numpy.uint8)
    return AudioVideo(frames, audio, sample_rate, fps=24.0)


class TestRoundTrip:
    @pytest.mark.parametrize("curve", CURVES)
    @pytest.mark.parametrize(
        "total, num_frames, overlap",
        [
            (50, 17, 4),  # 13-frame stride, 50 is not a multiple
            (52, 17, 4),  # exact multiple of the stride
            (40, 9, 0),  # no overlap
            (33, 9, 8),  # stride 1, a heavy overlap
            (10, 17, 4),  # one window carrying a long pad
        ],
    )
    def test_unprocessed_windows_rejoin_to_the_source(
        self, curve, total, num_frames, overlap
    ):
        clip = _ramp_clip(total)
        windows = _cut(clip, num_frames, overlap)

        joined = join_windows(windows, clip, num_frames, overlap, curve=curve)

        assert len(joined.frames) == total
        assert numpy.array_equal(frames_as_array(joined), frames_as_array(clip))

    def test_overlap_wider_than_the_stride_still_rejoins(self):
        clip = _ramp_clip(30)
        windows = _cut(clip, 9, 6)

        joined = join_windows(windows, clip, 9, 6)

        assert numpy.array_equal(frames_as_array(joined), frames_as_array(clip))

    def test_the_output_keeps_the_sources_fps(self):
        clip = _ramp_clip(50, fps=25.0)
        joined = join_windows(_cut(clip, 17, 4), clip, 17, 4)
        assert joined.fps == 25.0

    def test_string_arguments_are_coerced(self):
        clip = _ramp_clip(50)
        joined = join_windows(_cut(clip, 17, 4), clip, "17", "4")
        assert len(joined.frames) == 50


class TestWeights:
    @pytest.mark.parametrize("curve", CURVES)
    @pytest.mark.parametrize("overlap", [1, 2, 4, 8, 16])
    def test_open_ascending_and_symmetric(self, curve, overlap):
        weights = _weights(overlap, curve)

        assert len(weights) == overlap
        assert (weights > 0).all() and (weights < 1).all()
        assert (numpy.diff(weights) > 0).all() or overlap == 1
        assert numpy.allclose(weights + weights[::-1], 1.0, atol=1e-6)

    @pytest.mark.parametrize("overlap", [1, 4, 9])
    def test_linear_is_the_open_ramp(self, overlap):
        expected = [(k + 1) / (overlap + 1) for k in range(overlap)]
        assert numpy.allclose(_weights(overlap, "linear"), expected)

    def test_cosine_and_smoothstep_follow_their_formulas(self):
        t = numpy.array([1, 2, 3, 4], dtype=numpy.float64) / 5
        assert numpy.allclose(
            _weights(4, "cosine"), (1 - numpy.cos(numpy.pi * t)) / 2, atol=1e-6
        )
        assert numpy.allclose(_weights(4, "smoothstep"), 3 * t**2 - 2 * t**3, atol=1e-6)

    def test_overlap_zero_has_no_weights(self):
        assert len(_weights(0, "cosine")) == 0

    @pytest.mark.parametrize("curve", CURVES)
    def test_the_blend_follows_the_curve(self, curve):
        """Window 0 is all 0 and window 1 all 255: the overlap frames come
        out at 255 * w."""
        num_frames, overlap = 9, 4
        clip = _flat(14, 0)  # stride 5: 14 frames -> 3 windows
        count = window_count(14, num_frames, overlap)
        assert count == 3
        windows = [_flat(num_frames, 0), _flat(num_frames, 255), _flat(num_frames, 255)]

        joined = join_windows(windows, clip, num_frames, overlap, curve=curve)

        levels = frames_as_array(joined)[:, 0, 0, 0]
        weights = _weights(overlap, curve)
        # Window 0 owns source frame 0 up to window 1's head, which starts at
        # 1 * stride - overlap = 1 and blends over frames 1..4
        assert levels[0] == 0
        assert list(levels[1:5]) == [int(round(255 * w)) for w in weights]
        assert (levels[5:] == 255).all()
        assert len(levels) == 14


class TestRefusals:
    def test_too_few_windows_name_both_counts_and_what_to_add(self):
        clip = _ramp_clip(50)
        windows = _cut(clip, 17, 4)[:-1]

        with pytest.raises(ValueError) as error:
            join_windows(windows, clip, 17, 4)

        message = str(error.value)
        assert "needs 4 windows" in message
        assert "50-frame" in message
        assert "got 3" in message
        assert "add 1 entry (index 3)" in message

    def test_too_many_windows_name_what_to_drop(self):
        clip = _ramp_clip(50)
        windows = _cut(clip, 17, 4)
        windows += [windows[-1], windows[-1]]

        with pytest.raises(ValueError) as error:
            join_windows(windows, clip, 17, 4)

        message = str(error.value)
        assert "needs 4 windows" in message
        assert "got 6" in message
        assert "drop 2 entries (index 4..5)" in message

    def test_a_window_of_the_wrong_length_is_named(self):
        clip = _ramp_clip(50)
        windows = _cut(clip, 17, 4)
        windows[2] = _flat(16, 0, size=(6, 4))

        with pytest.raises(ValueError, match=r"window 2 has 16") as error:
            join_windows(windows, clip, 17, 4)
        assert "window 1 " not in str(error.value)

    def test_windows_of_different_sizes_are_refused(self):
        clip = _ramp_clip(50)
        windows = _cut(clip, 17, 4)
        windows[1] = _flat(17, 0, size=(8, 8))

        with pytest.raises(ValueError, match=r"window 1 is 8x8") as error:
            join_windows(windows, clip, 17, 4)
        assert "6x4" in str(error.value)

    def test_a_window_size_other_than_the_sources_is_allowed(self):
        clip = _ramp_clip(50)
        windows = [
            AudioVideo(
                frames_as_array(w).repeat(2, axis=1).repeat(2, axis=2),
                None,
                None,
                fps=24.0,
            )
            for w in _cut(clip, 17, 4)
        ]

        joined = join_windows(windows, clip, 17, 4)

        assert len(joined.frames) == 50
        assert frames_as_array(joined).shape == (50, 8, 12, 3)
        expected = frames_as_array(clip).repeat(2, axis=1).repeat(2, axis=2)
        assert numpy.array_equal(frames_as_array(joined), expected)

    def test_an_unknown_curve_is_refused(self):
        clip = _ramp_clip(50)
        with pytest.raises(ValueError, match="'curve' as one of"):
            join_windows(_cut(clip, 17, 4), clip, 17, 4, curve="bogus")

    def test_an_overlap_as_long_as_the_window_is_refused(self):
        clip = _ramp_clip(50)
        with pytest.raises(ValueError, match="'overlap' below 'num_frames'"):
            join_windows(_cut(clip, 17, 4), clip, 17, 17)

    def test_a_negative_overlap_is_refused(self):
        clip = _ramp_clip(50)
        with pytest.raises(ValueError):
            join_windows(_cut(clip, 17, 4), clip, 17, -1)

    def test_a_zero_num_frames_is_refused(self):
        clip = _ramp_clip(50)
        with pytest.raises(ValueError):
            join_windows(_cut(clip, 17, 4), clip, 0, 0)

    def test_a_fractional_overlap_is_refused(self):
        clip = _ramp_clip(50)
        with pytest.raises(ValueError, match="whole number"):
            join_windows(_cut(clip, 17, 4), clip, 17, 2.5)

    @pytest.mark.parametrize("videos", [[], None])
    def test_no_windows_is_refused(self, videos):
        with pytest.raises(ValueError, match="non-empty list"):
            join_windows(videos, _ramp_clip(50), 17, 4)


class TestWindowCount:
    @pytest.mark.parametrize(
        "frames, num_frames, overlap, expected",
        [
            (50, 17, 4, 4),
            (52, 17, 4, 4),
            (53, 17, 4, 5),
            (13, 17, 4, 1),
            (1, 17, 4, 1),
            (40, 9, 0, 5),
        ],
    )
    def test_ceil_of_frames_over_stride(self, frames, num_frames, overlap, expected):
        assert window_count(frames, num_frames, overlap) == expected

    def test_the_exact_count_is_no_problem(self):
        assert window_count_problem(4, 50, 17, 4) is None

    def test_one_missing_entry_is_singular(self):
        message = window_count_problem(3, 50, 17, 4)
        assert "needs 4 windows" in message and "50-frame" in message
        assert "got 3" in message
        assert "add 1 entry (index 3)" in message

    def test_several_missing_entries_are_plural(self):
        message = window_count_problem(1, 50, 17, 4)
        assert "add 3 entries (index 1..3)" in message

    def test_one_extra_entry_is_singular(self):
        assert "drop 1 entry (index 4)" in window_count_problem(5, 50, 17, 4)

    def test_several_extra_entries_are_plural(self):
        assert "drop 3 entries (index 4..6)" in window_count_problem(7, 50, 17, 4)


class TestValidate:
    GOOD = {
        "videos": "x",
        "source": "y",
        "num_frames": 17,
        "overlap": 4,
        "curve": "cosine",
    }

    def test_a_good_literal_step_has_no_errors(self):
        assert not _validate(self.GOOD)

    def test_the_default_curve_is_fine(self):
        good = {k: v for k, v in self.GOOD.items() if k != "curve"}
        assert not _validate(good)

    def test_an_unknown_curve_is_reported_on_curve(self):
        errors = _validate(dict(self.GOOD, curve="bogus"))
        assert any(name == "curve" for name, _ in errors), errors

    def test_an_overlap_as_long_as_the_window_is_reported_on_overlap(self):
        errors = _validate(dict(self.GOOD, overlap=17, num_frames=17))
        assert any(name == "overlap" for name, _ in errors), errors

    def test_a_negative_overlap_is_reported_on_overlap(self):
        errors = _validate(dict(self.GOOD, overlap=-1))
        assert any(name == "overlap" for name, _ in errors), errors

    def test_a_zero_num_frames_is_reported_on_num_frames(self):
        errors = _validate(dict(self.GOOD, num_frames=0, overlap=0))
        assert any(name == "num_frames" for name, _ in errors), errors


class TestShots:
    @pytest.mark.parametrize("fps", [24.0, 25.0])
    @pytest.mark.parametrize("sample_rate", [44100, 48000])
    def test_spans_partition_the_frames_and_tile_the_audio(self, fps, sample_rate):
        total, num_frames, overlap = 50, 17, 4
        clip = _ramp_clip(total, fps, sample_rate)
        joined = join_windows(
            _cut(clip, num_frames, overlap), clip, num_frames, overlap
        )
        shots = joined.shots

        assert len(shots) == 4
        assert sum(s["num_frames"] for s in shots) == total
        assert shots[0]["start_frame"] == 0
        for before, after in zip(shots, shots[1:]):
            assert after["start_frame"] == before["start_frame"] + before["num_frames"]
            assert (
                after["start_sample"] == before["start_sample"] + before["num_samples"]
            )
        assert shots[0]["start_sample"] == 0
        for shot in shots:
            assert shot["start_sample"] == frames_to_samples(
                shot["start_frame"], fps, sample_rate
            )
        assert sum(s["num_samples"] for s in shots) == joined.audio.shape[1]
        assert joined.audio.shape[1] == frames_to_samples(total, fps, sample_rate)

    def test_the_seams_carry_overlap_frames_but_not_the_first_shot(self):
        clip = _ramp_clip(50)
        shots = join_windows(_cut(clip, 17, 4), clip, 17, 4).shots

        assert "overlap_frames" not in shots[0]
        assert [s["overlap_frames"] for s in shots[1:]] == [4, 4, 4]
        # Window i > 0 starts at i * stride - overlap
        assert [s["start_frame"] for s in shots] == [0, 9, 22, 35]

    def test_overlap_zero_has_no_overlap_frames(self):
        clip = _ramp_clip(40)
        shots = join_windows(_cut(clip, 9, 0), clip, 9, 0).shots
        assert all("overlap_frames" not in s for s in shots)
        assert [s["start_frame"] for s in shots] == [0, 9, 18, 27, 36]

    def test_every_shot_has_a_distinct_name(self):
        clip = _ramp_clip(50)
        shots = join_windows(_cut(clip, 17, 4), clip, 17, 4).shots
        names = [s["name"] for s in shots]
        assert len(set(names)) == len(names) == 4


class TestNoAudio:
    def test_a_silent_source_gives_a_silent_result_without_fps(self):
        clip = _ramp_clip(50, audio=False)
        silent = AudioVideo(clip.frames, None, None)  # no fps either
        windows = _cut(clip, 17, 4)

        joined = join_windows(windows, silent, 17, 4)

        assert joined.audio is None
        assert joined.sample_rate is None
        assert len(joined.frames) == 50
        for shot in joined.shots:
            assert shot.get("start_sample") is None
            assert shot.get("num_samples") is None


class TestAudio:
    def test_the_source_track_comes_back_sample_for_sample(self):
        clip = _ramp_clip(50, 25.0, 44100)
        joined = join_windows(_cut(clip, 17, 4), clip, 17, 4)

        length = frames_to_samples(50, 25.0, 44100)
        assert joined.sample_rate == 44100
        assert joined.audio.shape == (2, length)
        assert numpy.array_equal(joined.audio, clip.audio[:, :length])

    def test_a_short_track_is_padded_to_the_frames_length(self):
        clip = _ramp_clip(50, 24.0, 48000)
        windows = _cut(clip, 17, 4)
        short = AudioVideo(clip.frames, clip.audio[:, :-3], 48000, fps=24.0)

        joined = join_windows(windows, short, 17, 4)

        assert joined.audio.shape == (2, frames_to_samples(50, 24.0, 48000))
        assert numpy.array_equal(joined.audio[:, :-3], short.audio)
        assert not joined.audio[:, -3:].any()

    def test_a_long_track_is_cut_to_the_frames_length(self):
        clip = _ramp_clip(50, 24.0, 48000)
        long = AudioVideo(
            clip.frames,
            numpy.concatenate([clip.audio, clip.audio[:, :500]], axis=1),
            48000,
            fps=24.0,
        )
        joined = join_windows(_cut(clip, 17, 4), long, 17, 4)
        assert numpy.array_equal(joined.audio, clip.audio)

    def test_the_windows_own_audio_is_ignored(self):
        clip = _ramp_clip(50, 24.0, 48000)
        windows = _cut(clip, 17, 4)
        for i, window in enumerate(windows):
            window.audio = numpy.full_like(window.audio, 0.5 + i)

        joined = join_windows(windows, clip, 17, 4)

        assert numpy.array_equal(joined.audio, clip.audio)

    def test_audio_with_no_frame_rate_is_refused(self):
        clip = _ramp_clip(50)
        windows = _cut(clip, 17, 4)
        no_fps = AudioVideo(clip.frames, clip.audio, clip.sample_rate)
        with pytest.raises(ValueError, match="needs 'fps'"):
            join_windows(windows, no_fps, 17, 4)

    def test_an_fps_argument_supplies_the_missing_rate(self):
        clip = _ramp_clip(50)
        windows = _cut(clip, 17, 4)
        no_fps = AudioVideo(clip.frames, clip.audio, clip.sample_rate)
        joined = join_windows(windows, no_fps, 17, 4, fps=24)
        assert joined.audio.shape[1] == frames_to_samples(50, 24, 48000)


def _write_video(path, num_frames=20, fps=8, sample_rate=8000):
    from diffusers.utils.export_utils import encode_video

    video = [
        Image.new("RGB", (16, 16), (10 * index, 0, 0)) for index in range(num_frames)
    ]
    samples = int(num_frames / fps * sample_rate)
    audio = torch.full((2, samples), 0.25, dtype=torch.float32)
    encode_video(
        video,
        fps=fps,
        output_path=str(path),
        audio=audio,
        audio_sample_rate=sample_rate,
    )
    return str(path)


class TestRealPath:
    def test_a_file_source_keeps_its_audio_and_frame_count(self, tmp_path):
        import dw.arguments as arguments_module

        path = _write_video(tmp_path / "long.mp4")
        windows = [window_video(VideoFileReference(path), i, 9, 2) for i in range(3)]
        task = {
            "command": "join_windows",
            "arguments": {
                "videos": windows,
                "source": path,
                "num_frames": 9,
                "overlap": 2,
            },
        }
        arguments_module.realize_args(task, base_dir=str(tmp_path))

        joined = join_windows(**task["arguments"])

        assert len(joined.frames) == 20
        assert joined.fps == 8
        assert joined.sample_rate == 8000
        assert joined.audio.shape == (2, 20 * 1000)
        assert joined.audio.mean() == pytest.approx(0.25, abs=0.05)
        assert sum(s["num_frames"] for s in joined.shots) == 20

    def test_a_source_path_given_directly_is_read_with_its_audio(self, tmp_path):
        path = _write_video(tmp_path / "long.mp4")
        windows = [window_video(VideoFileReference(path), i, 9, 2) for i in range(3)]

        joined = join_windows(windows, path, 9, 2)

        assert len(joined.frames) == 20
        assert joined.audio.shape == (2, 20000)


class TestWorkflow:
    def test_window_then_join_runs_end_to_end(self, tmp_path, monkeypatch):
        """The real Workflow.run: `for_each` over a windows list drives
        window_video, `gather:window` hands the windows to join_windows."""
        from dw.workflow import Workflow

        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
        path = _write_video(tmp_path / "long.mp4")
        definition = {
            "id": "window_join",
            "steps": [
                {
                    "name": "window",
                    "for_each": [
                        {"name": "a", "index": 0},
                        {"name": "b", "index": 1},
                        {"name": "c", "index": 2},
                    ],
                    "task": {
                        "command": "window_video",
                        "arguments": {
                            "video": path,
                            "index": "item:index",
                            "num_frames": 9,
                            "overlap": 2,
                        },
                    },
                    "result": {"content_type": "video/mp4"},
                },
                {
                    "name": "join",
                    "task": {
                        "command": "join_windows",
                        "arguments": {
                            "videos": "gather:window",
                            "source": path,
                            "num_frames": 9,
                            "overlap": 2,
                        },
                    },
                    "result": {"content_type": "video/mp4"},
                },
            ],
        }
        workflow = Workflow(definition, str(tmp_path / "out"), str(tmp_path / "w.json"))
        workflow.run({})

        (entry,) = [e for e in workflow.manifest if e["step"] == "join"]
        assert [s["name"] for s in entry["shots"]] == [
            "window@a",
            "window@b",
            "window@c",
        ]
        assert sum(s["num_frames"] for s in entry["shots"]) == 20
