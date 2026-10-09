"""window_video (#601): one overlapping, fixed-length window of a long video,
its audio cut on the source's own frame boundaries."""

import numpy
import pytest
import torch
from PIL import Image

from dw.media_types import AudioVideo
from dw.task_domains import frames_to_samples, task_argument_errors
from dw.tasks.windows import window_span, window_video


def _validate(arguments):
    """[(argument, message)] validate reports for one window_video step."""
    step = {
        "name": "window",
        "task": {"command": "window_video", "arguments": arguments},
        "result": {"content_type": "video/mp4"},
    }
    errors = task_argument_errors({"id": "windows", "steps": [step]})
    return [(error["path"].rsplit(".", 1)[-1], error["message"]) for error in errors]


def _marked_clip(num_frames, fps=24.0, sample_rate=48000, audio=True):
    """A clip whose frame f is a flat grey of level f, and whose track is a
    ramp counting samples, so a window's frames and samples name where in
    the source they came from."""
    frames = numpy.stack(
        [numpy.full((4, 4, 3), f, dtype=numpy.uint8) for f in range(num_frames)]
    )
    if not audio:
        return AudioVideo(frames, None, None, fps=fps)
    samples = frames_to_samples(num_frames, fps, sample_rate)
    track = numpy.arange(1, samples + 1, dtype=numpy.float32)
    return AudioVideo(frames, numpy.stack([track, -track]), sample_rate, fps=fps)


def _levels(window):
    """The source frame each window frame repeats, read back off its grey."""
    return [int(round(float(frame[0, 0, 0]) * 255)) for frame in window.frames]


class TestSpans:
    """The plan's acceptance example: a 50-frame clip, 17 frames, overlap 4."""

    def test_the_first_window_opens_on_repeats_of_frame_0(self):
        window = window_video(_marked_clip(50), 0, 17, 4)

        assert len(window.frames) == 17
        assert _levels(window) == [0] * 4 + list(range(13))

    def test_a_middle_window_shares_its_overlap_with_the_one_before(self):
        before = window_video(_marked_clip(50), 1, 17, 4)
        window = window_video(_marked_clip(50), 2, 17, 4)

        assert _levels(window) == list(range(22, 39))
        assert _levels(window)[:4] == _levels(before)[-4:]

    def test_the_last_window_pads_with_the_last_frame(self):
        window = window_video(_marked_clip(50), 3, 17, 4)

        assert window_span(3, 17, 4) == (35, 52)
        assert len(window.frames) == 17
        assert _levels(window) == list(range(35, 50)) + [49, 49]

    def test_frames_are_float32_in_0_1(self):
        window = window_video(_marked_clip(50), 1, 17, 4)

        assert window.frames.dtype == numpy.float32
        assert window.frames.shape == (17, 4, 4, 3)
        assert window.frames.min() >= 0.0 and window.frames.max() <= 1.0

    def test_every_real_frame_is_one_windows_own_stride(self):
        strides = []
        for index in range(4):
            levels = _levels(window_video(_marked_clip(50), index, 17, 4))
            strides += levels[4:]
        assert strides[:50] == list(range(50))

    def test_string_arguments_are_coerced(self):
        window = window_video(_marked_clip(50), "1", "17", "4")

        assert _levels(window) == list(range(9, 26))


class TestAudio:
    @pytest.mark.parametrize("fps", [24.0, 25.0])
    @pytest.mark.parametrize("sample_rate", [44100, 48000])
    def test_strides_tile_the_source_track_exactly(self, fps, sample_rate):
        """Each window's own stride, after its overlap, is the source's samples
        on the source's frame boundaries - concatenated they are the whole
        track, nothing lost or repeated (#401's cumulative rule)."""
        total, num_frames, overlap = 50, 17, 4
        clip = _marked_clip(total, fps, sample_rate)
        tiled = []
        for index in range(4):
            window = window_video(clip, index, num_frames, overlap)
            start, end = window_span(index, num_frames, overlap)
            assert window.sample_rate == sample_rate
            assert window.fps == fps
            # The window's own length on the extended timeline
            expected = frames_to_samples(end, fps, sample_rate) - frames_to_samples(
                start, fps, sample_rate
            )
            assert window.audio.shape == (2, expected)
            own = frames_to_samples(index * 13, fps, sample_rate) - frames_to_samples(
                start, fps, sample_rate
            )
            tiled.append(window.audio[0, own:])
        joined = numpy.concatenate(tiled)
        track = clip.audio[0]
        assert numpy.array_equal(joined[: len(track)], track)
        assert not joined[len(track) :].any()

    def test_the_first_windows_prefix_is_silent(self):
        clip = _marked_clip(50, 24.0, 48000)
        window = window_video(clip, 0, 17, 4)
        head = frames_to_samples(4, 24.0, 48000)

        assert not window.audio[:, :head].any()
        assert numpy.array_equal(window.audio[0, head:], clip.audio[0, : 13 * 2000])

    def test_the_last_windows_pad_is_silent(self):
        clip = _marked_clip(50, 25.0, 44100)
        window = window_video(clip, 3, 17, 4)
        pad = frames_to_samples(52, 25.0, 44100) - frames_to_samples(50, 25.0, 44100)

        assert pad > 0
        assert not window.audio[:, -pad:].any()
        assert window.audio[0, -pad - 1] == clip.audio[0, -1]

    def test_a_short_track_is_padded_to_the_windows_length(self):
        clip = _marked_clip(50, 24.0, 48000)
        clip = AudioVideo(clip.frames, clip.audio[:, :-100], 48000, fps=24.0)
        window = window_video(clip, 3, 17, 4)

        assert window.audio.shape[1] == frames_to_samples(
            52, 24.0, 48000
        ) - frames_to_samples(35, 24.0, 48000)

    def test_an_fps_argument_overrides_the_sources(self):
        clip = _marked_clip(50, 24.0, 48000)
        window = window_video(clip, 0, 17, 4, fps=25)

        assert window.fps == 25
        assert window.audio.shape[1] == frames_to_samples(17, 25, 48000)

    def test_a_silent_source_gives_a_silent_window(self):
        window = window_video(_marked_clip(50, audio=False), 1, 17, 4)

        assert window.audio is None
        assert window.sample_rate is None
        assert len(window.frames) == 17

    def test_audio_with_no_frame_rate_is_refused(self):
        clip = _marked_clip(50, 24.0, 48000)
        clip = AudioVideo(clip.frames, clip.audio, 48000)
        with pytest.raises(ValueError, match="needs 'fps'"):
            window_video(clip, 0, 17, 4)


class TestRefusals:
    def test_a_window_past_the_end_names_the_count_and_last_index(self):
        with pytest.raises(ValueError) as error:
            window_video(_marked_clip(50), 4, 17, 4)
        message = str(error.value)
        assert "50 frames" in message
        assert "last window is index 3" in message

    def test_an_overlap_as_long_as_the_window_is_refused(self):
        with pytest.raises(ValueError, match="'overlap' below 'num_frames'"):
            window_video(_marked_clip(50), 0, 17, 17)

    @pytest.mark.parametrize(
        "arguments",
        [
            {"index": 0, "num_frames": 17, "overlap": -1},
            {"index": -1, "num_frames": 17, "overlap": 4},
            {"index": 0, "num_frames": 0, "overlap": 0},
        ],
    )
    def test_out_of_domain_values_are_refused_at_run_time(self, arguments):
        with pytest.raises(ValueError):
            window_video(_marked_clip(50), **arguments)

    def test_a_fractional_index_is_refused(self):
        with pytest.raises(ValueError, match="whole number"):
            window_video(_marked_clip(50), 1.5, 17, 4)

    @pytest.mark.parametrize(
        "arguments, argument",
        [
            ({"index": 0, "num_frames": 17, "overlap": 17}, "overlap"),
            ({"index": 0, "num_frames": 17, "overlap": 20}, "overlap"),
            ({"index": 0, "num_frames": 17, "overlap": -1}, "overlap"),
            ({"index": -1, "num_frames": 17, "overlap": 4}, "index"),
            ({"index": 0, "num_frames": 0, "overlap": 0}, "num_frames"),
        ],
    )
    def test_validate_refuses_a_literal(self, arguments, argument):
        errors = _validate(dict(video="x", **arguments))
        assert any(name == argument for name, _ in errors), errors

    def test_validate_passes_a_good_window(self):
        assert not _validate({"video": "x", "index": 3, "num_frames": 17, "overlap": 4})


class TestRealPath:
    """A file-based `video` reaches the task by reference and is read with
    its audio - the ordinary `video` load reads frames only."""

    def write_video(self, path, num_frames=20, fps=8, sample_rate=8000):
        from diffusers.utils.export_utils import encode_video

        video = [
            Image.new("RGB", (16, 16), (10 * index, 0, 0))
            for index in range(num_frames)
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

    def test_a_path_argument_keeps_its_audio(self, tmp_path):
        import dw.arguments as arguments_module
        from dw.tasks.video_utils import VideoFileReference

        path = self.write_video(tmp_path / "long.mp4")
        task = {
            "command": "window_video",
            "arguments": {"video": path, "index": 1, "num_frames": 9, "overlap": 2},
        }
        arguments_module.realize_args(task, base_dir=str(tmp_path))
        assert isinstance(task["arguments"]["video"], VideoFileReference)

        window = window_video(**task["arguments"])

        assert len(window.frames) == 9
        assert window.fps == 8
        assert window.sample_rate == 8000
        # Window 1 is source frames 5..13, all real: 9 frames of 1000 samples
        assert window.audio.shape == (2, 9000)
        assert window.audio[:, 500:8500].mean() == pytest.approx(0.25, abs=0.05)


def _write_gradient_video(path, num_frames=37, fps=8, sample_rate=8000, size=24):
    """An mp4 with sound whose every frame differs: a gradient shifted by the
    frame index, and a track that is not constant."""
    from diffusers.utils.export_utils import encode_video

    cols = numpy.arange(size).reshape(1, -1)
    rows = numpy.arange(size).reshape(-1, 1)
    video = []
    for index in range(num_frames):
        pixels = numpy.stack(
            [
                (cols * 8 + index * 5 + rows * 0) % 256,
                (rows * 8 + index * 3 + cols * 0) % 256,
                (rows * 2 + cols * 2 + index * 7) % 256,
            ],
            axis=-1,
        ).astype(numpy.uint8)
        video.append(Image.fromarray(pixels))
    samples = int(num_frames / fps * sample_rate)
    audio = torch.sin(torch.arange(samples, dtype=torch.float32) / 20).repeat(2, 1)
    encode_video(
        video,
        fps=fps,
        output_path=str(path),
        audio=audio * 0.5,
        audio_sample_rate=sample_rate,
    )
    return str(path)


class TestFileMatchesInMemory:
    """#695: a file source is read a range at a time, and must cut the same
    window the whole-file decode does."""

    @pytest.mark.parametrize(
        "num_frames, overlap",
        [(9, 2), (9, 6), (17, 4), (8, 0), (40, 8)],
    )
    def test_every_window_equals_the_in_memory_window(
        self, tmp_path, num_frames, overlap
    ):
        from dw.task_domains import window_count
        from dw.tasks.video_utils import VideoFileReference, load_audio_video

        path = _write_gradient_video(tmp_path / "long.mp4")
        clip = load_audio_video(path)
        total = len(clip.frames)
        assert total >= 30

        for index in range(window_count(total, num_frames, overlap)):
            via_file = window_video(
                VideoFileReference(path), index, num_frames, overlap
            )
            via_memory = window_video(clip, index, num_frames, overlap)

            assert via_file.frames.dtype == via_memory.frames.dtype
            assert numpy.array_equal(via_file.frames, via_memory.frames), (
                f"window {index} of {num_frames}/{overlap}: max diff "
                f"{numpy.abs(via_file.frames - via_memory.frames).max()}"
            )
            assert via_file.sample_rate == via_memory.sample_rate
            assert via_file.fps == via_memory.fps
            assert via_file.audio.shape == via_memory.audio.shape
            assert numpy.array_equal(via_file.audio, via_memory.audio)

    def test_a_window_past_the_end_gives_the_in_memory_message(self, tmp_path):
        from dw.tasks.video_utils import VideoFileReference, load_audio_video

        path = _write_gradient_video(tmp_path / "long.mp4")
        clip = load_audio_video(path)
        with pytest.raises(ValueError) as in_memory:
            window_video(clip, 9, 9, 2)
        with pytest.raises(ValueError) as from_file:
            window_video(VideoFileReference(path), 9, 9, 2)

        assert str(from_file.value) == str(in_memory.value)
        assert "last window is index" in str(from_file.value)


class TestFileMemory:
    def test_a_window_of_a_long_file_never_decodes_the_whole_source(self, tmp_path):
        import tracemalloc

        import av

        from dw.tasks.video_utils import VideoFileReference

        total, size, fps, sample_rate = 2000, 64, 25, 8000
        path = str(tmp_path / "long.mp4")
        rng = numpy.random.default_rng(0)
        with av.open(path, "w") as container:
            video = container.add_stream("libx264", rate=fps)
            video.width = video.height = size
            video.pix_fmt = "yuv420p"
            video.options = {"g": "10", "preset": "ultrafast"}
            audio = container.add_stream("aac", rate=sample_rate)
            audio.layout = "mono"
            for index in range(total):
                pixels = numpy.full((size, size, 3), index % 256, dtype=numpy.uint8)
                pixels[:, : index % size] = rng.integers(0, 255, 3, dtype=numpy.uint8)
                for packet in video.encode(av.VideoFrame.from_ndarray(pixels, "rgb24")):
                    container.mux(packet)
            for packet in video.encode():
                container.mux(packet)
            samples = total // fps * sample_rate
            wave = numpy.sin(numpy.arange(samples) / 20).astype(numpy.float32) * 0.5
            frame = av.AudioFrame.from_ndarray(
                wave.reshape(1, -1), format="flt", layout="mono"
            )
            frame.sample_rate = sample_rate
            for packet in audio.encode(frame):
                container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)

        num_frames, overlap = 33, 8
        window_bytes = num_frames * size * size * 3 * 4
        audio_bytes = samples * 4
        # The soundtrack is decoded whole, and its decode holds a few copies
        # of it at once (chunks, joined, transposed, fitted)
        bound = 2 * window_bytes + 4 * audio_bytes
        assert bound < total * size * size * 3 * 4 / 4  # a quarter of the source

        tracemalloc.start()
        try:
            window = window_video(VideoFileReference(path), 20, num_frames, overlap)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        assert len(window.frames) == num_frames
        assert peak < bound, f"peak {peak} over bound {bound}"


class TestHeaderCountMismatch:
    """#695: the frame count comes from the header; a range read that runs
    out before it is retried on a counted total, and refused naming both
    counts when the file still ends short."""

    def _overstated(self, monkeypatch, frame_count):
        import dw.tasks.windows as windows
        from dw.media import video_shape

        def shape(path):
            return dict(video_shape(path), frame_count=frame_count)

        monkeypatch.setattr(windows, "video_shape", shape)

    def test_an_overstated_header_is_recounted(self, tmp_path, monkeypatch):
        from dw.tasks.video_utils import VideoFileReference, load_audio_video

        path = _write_gradient_video(tmp_path / "long.mp4")
        expected = window_video(load_audio_video(path), 5, 9, 2)
        self._overstated(monkeypatch, 45)

        # Window 5 is source frames 33..41 of 37: a header of 45 promises
        # frames the file doesn't have, and clamping must use the real end
        window = window_video(VideoFileReference(path), 5, 9, 2)

        assert numpy.array_equal(window.frames, expected.frames)
        assert numpy.array_equal(window.audio, expected.audio)

    def test_a_count_that_still_disagrees_is_refused_naming_both(
        self, tmp_path, monkeypatch
    ):
        import dw.tasks.windows as windows
        from dw.tasks.video_utils import VideoFileReference

        path = _write_gradient_video(tmp_path / "long.mp4")
        self._overstated(monkeypatch, 45)
        monkeypatch.setattr(windows, "count_video_frames", lambda path: 40)

        with pytest.raises(ValueError) as error:
            window_video(VideoFileReference(path), 5, 9, 2)
        message = str(error.value)
        assert "45 frames" in message and "counted 40" in message
