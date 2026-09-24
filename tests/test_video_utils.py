"""
Unit tests for video frame extraction and its task-command registration.

process_video dispatches through a hand-written if-chain while Task keeps a
separate hand-written list of the names it accepts, so the two are tested
against each other here.
"""

import numpy
import pytest
import torch
from PIL import Image

from dw.result import AudioVideo
from dw.tasks.task import _VIDEO_PROCESSOR_COMMANDS, Task
from dw.tasks.video_utils import (
    _fit_audio_to_frames,
    extract_frame,
    frame_count,
    get_frame,
    loop_frames,
    process_video,
)


@pytest.fixture
def video():
    """Four frames, each a distinct solid color so identity is checkable."""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    return [Image.new("RGB", (8, 8), color) for color in colors]


class TestGetFrame:
    def test_defaults_to_the_first_frame(self, video):
        assert get_frame(video) is video[0]

    def test_returns_the_indexed_frame(self, video):
        assert get_frame(video, 2) is video[2]

    def test_negative_indexes_count_from_the_end(self, video):
        assert get_frame(video, -1) is video[3]

    def test_an_out_of_range_index_raises(self, video):
        with pytest.raises(IndexError):
            get_frame(video, 99)


class TestProcessVideo:
    def test_get_frame_uses_the_frame_index_argument(self, video):
        assert process_video(video, "get_frame", "cpu", {"frame_index": 2}) is video[2]

    def test_get_frame_without_an_index_returns_the_first(self, video):
        assert process_video(video, "get_frame", "cpu", {}) is video[0]

    def test_get_first_frame(self, video):
        assert process_video(video, "get_first_frame", "cpu", {}) is video[0]

    def test_get_last_frame(self, video):
        assert process_video(video, "get_last_frame", "cpu", {}) is video[3]

    def test_get_last_frame_ignores_a_frame_index(self, video):
        # get_last_frame computes its own index; a stray argument must not win
        assert (
            process_video(video, "get_last_frame", "cpu", {"frame_index": 0})
            is (video[3])
        )

    @pytest.mark.parametrize("name", ["GET_LAST_FRAME", "Get_Last_Frame"])
    def test_processor_names_are_case_insensitive(self, video, name):
        assert process_video(video, name, "cpu", {}) is video[3]

    def test_an_unknown_processor_raises_with_its_name(self, video):
        with pytest.raises(Exception, match="Unknown video processor type: get_middle"):
            process_video(video, "get_middle", "cpu", {})

    def test_a_single_frame_video_works(self):
        frames = [Image.new("RGB", (8, 8))]

        assert process_video(frames, "get_last_frame", "cpu", {}) is frames[0]


class TestExtractFrame:
    """extract_frame must pull a PIL frame out of every shape results carry."""

    def test_pil_frames_are_returned_by_identity(self, video):
        assert extract_frame(video, 2) is video[2]

    def test_negative_indexes_count_from_the_end(self, video):
        assert extract_frame(video, -1) is video[3]

    def test_uint8_numpy_frames(self):
        frames = numpy.zeros((4, 8, 8, 3), dtype=numpy.uint8)
        frames[2, :, :, 0] = 255  # frame 2 is solid red

        frame = extract_frame(frames, 2)

        assert isinstance(frame, Image.Image)
        assert frame.getpixel((0, 0)) == (255, 0, 0)

    def test_float_numpy_frames_are_scaled_from_unit_range(self):
        frames = numpy.zeros((4, 8, 8, 3), dtype=numpy.float32)
        frames[-1, :, :, 1] = 1.0  # last frame is solid green

        frame = extract_frame(frames, -1)

        assert frame.getpixel((0, 0)) == (0, 255, 0)

    def test_float_values_are_clipped_before_scaling(self):
        frames = numpy.full((1, 8, 8, 3), 1.5, dtype=numpy.float32)

        assert extract_frame(frames, 0).getpixel((0, 0)) == (255, 255, 255)

    def test_channels_first_tensor_frames(self):
        frames = torch.zeros((4, 3, 8, 8))
        frames[1, 2] = 1.0  # frame 1 is solid blue

        frame = extract_frame(frames, 1)

        assert frame.getpixel((0, 0)) == (0, 0, 255)

    def test_channels_last_tensor_frames(self):
        frames = torch.zeros((4, 8, 8, 3))
        frames[0, :, :, 0] = 1.0

        assert extract_frame(frames, 0).getpixel((0, 0)) == (255, 0, 0)

    def test_audio_video_unwraps_to_its_frames(self, video):
        artifact = AudioVideo(video, audio=None, sample_rate=None)

        assert extract_frame(artifact, -1) is video[3]

    def test_a_one_video_batch_list_unwraps(self, video):
        assert extract_frame([video], -1) is video[3]

    def test_a_one_video_batch_of_numpy_frames_unwraps(self):
        frames = numpy.zeros((1, 4, 8, 8, 3), dtype=numpy.uint8)

        assert isinstance(extract_frame(frames, 3), Image.Image)

    def test_a_single_frame_video_is_not_unwrapped(self):
        frames = [Image.new("RGB", (8, 8))]

        assert extract_frame(frames, 0) is frames[0]

    def test_an_out_of_range_index_raises(self, video):
        with pytest.raises(IndexError):
            extract_frame(video, 99)

    def test_an_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Cannot extract frames"):
            extract_frame("not a video", 0)


class TestFrameCount:
    def test_counts_a_pil_list(self, video):
        assert frame_count(video) == 4

    def test_counts_numpy_frames(self):
        assert frame_count(numpy.zeros((7, 8, 8, 3), dtype=numpy.uint8)) == 7

    def test_counts_tensor_frames(self):
        assert frame_count(torch.zeros((5, 3, 8, 8))) == 5

    def test_counts_through_audio_video(self, video):
        assert frame_count(AudioVideo(video, None, None)) == 4

    def test_a_lone_numpy_frame_counts_as_one(self):
        assert frame_count(numpy.zeros((8, 8, 3), dtype=numpy.uint8)) == 1


class TestProcessVideoOverArtifactShapes:
    """The registered task commands must accept what pipelines actually return."""

    def test_get_last_frame_of_an_audio_video(self, video):
        artifact = AudioVideo(video, audio=None, sample_rate=None)

        assert process_video(artifact, "get_last_frame", "cpu", {}) is video[3]

    def test_get_last_frame_of_numpy_frames(self):
        frames = numpy.zeros((4, 8, 8, 3), dtype=numpy.float32)
        frames[-1, :, :, 0] = 1.0

        frame = process_video(frames, "get_last_frame", "cpu", {})

        assert frame.getpixel((0, 0)) == (255, 0, 0)


class TestVideoCommandRegistration:
    """Task's command list and process_video's if-chain are maintained by hand"""

    def test_every_registered_command_is_handled_by_process_video(self, video):
        for command in _VIDEO_PROCESSOR_COMMANDS:
            # Raises "Unknown video processor type" if the branch is missing
            assert process_video(video, command, "cpu", {}) in video

    def test_a_video_command_runs_through_task(self, video):
        task = Task(
            {"command": "get_last_frame", "arguments": {"video": "previous"}}, "cpu"
        )

        assert task.run({"video": video}) is video[3]

    def test_a_video_command_honors_a_device_override(self, video):
        # device_for pops "device" so it never reaches process_video as a
        # duplicate keyword argument
        task = Task({"command": "get_frame", "arguments": {}}, "cuda")

        assert task.run({"video": video, "device": "cpu", "frame_index": 1}) is video[1]

    def test_an_unregistered_video_command_is_reported_as_unknown(self, video):
        task = Task({"command": "get_middle_frame", "arguments": {}}, "cpu")

        with pytest.raises(ValueError, match="Unknown task command"):
            task.run({"video": video})

    def test_the_command_list_is_sorted_and_unique(self):
        assert _VIDEO_PROCESSOR_COMMANDS == sorted(set(_VIDEO_PROCESSOR_COMMANDS))

    def test_video_and_image_processor_names_do_not_collide(self):
        # Task checks image processors first, so a shared name would silently
        # route a video command into process_image
        from dw.tasks.image_utils import available_processors

        assert not set(_VIDEO_PROCESSOR_COMMANDS) & set(available_processors())


class TestFramesAsArray:
    """The frames of a video as one array, for an argument that takes frames"""

    def test_a_float_array_is_scaled_without_going_through_pil(self):
        from dw.tasks.video_utils import frames_as_array

        frames = numpy.full((4, 8, 8, 3), 0.5, dtype=numpy.float32)

        array = frames_as_array(frames)

        assert array.shape == (4, 8, 8, 3)
        assert array.dtype == numpy.uint8
        assert array[0, 0, 0, 0] == 128

    def test_a_uint8_array_is_carried_through_untouched(self):
        from dw.tasks.video_utils import frames_as_array

        frames = numpy.zeros((4, 8, 8, 3), dtype=numpy.uint8)

        assert frames_as_array(frames) is frames

    def test_a_one_video_batch_unwraps_to_the_video(self):
        from dw.tasks.video_utils import frames_as_array

        frames = numpy.zeros((1, 4, 8, 8, 3), dtype=numpy.uint8)

        assert frames_as_array(frames).shape == (4, 8, 8, 3)

    def test_pil_frames_are_stacked(self):
        from dw.tasks.video_utils import frames_as_array

        frames = [Image.new("RGB", (8, 6), (255, 0, 0)) for _ in range(3)]

        array = frames_as_array(frames)

        assert array.shape == (3, 6, 8, 3)
        assert tuple(array[0, 0, 0]) == (255, 0, 0)

    def test_channels_first_tensor_frames_are_transposed(self):
        from dw.tasks.video_utils import frames_as_array

        array = frames_as_array(torch.zeros(5, 3, 8, 6))

        assert array.shape == (5, 8, 6, 3)
        assert array.dtype == numpy.uint8

    def test_an_audio_video_gives_its_frames(self):
        from dw.result import AudioVideo
        from dw.tasks.video_utils import frames_as_array

        video = AudioVideo([Image.new("RGB", (8, 8))] * 2, "waveform", 24000)

        assert frames_as_array(video).shape == (2, 8, 8, 3)

    def test_registered_as_a_task_command(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        assert "video_frames" in _COMMAND_REGISTRY


class TestLoadAudioVideo:
    """Reading a video file back with the audio muxed into it."""

    def write_video(self, path, num_frames=8, fps=4, sample_rate=8000, level=0.25):
        from diffusers.utils.export_utils import encode_video

        video = [
            Image.new("RGB", (16, 16), (index, 0, 0)) for index in range(num_frames)
        ]
        samples = int(num_frames / fps * sample_rate)
        audio = torch.full((2, samples), level, dtype=torch.float32)
        encode_video(
            video,
            fps=fps,
            output_path=str(path),
            audio=audio,
            audio_sample_rate=sample_rate,
        )
        return str(path)

    def test_frames_and_audio_come_back_together(self, tmp_path):
        from dw.tasks.video_utils import load_audio_video

        path = self.write_video(tmp_path / "shot.mp4")

        video = load_audio_video(path)

        assert len(video.frames) == 8
        assert video.sample_rate == 8000
        assert video.audio.shape[0] == 2

    def test_the_files_own_frame_rate_comes_back_with_it(self, tmp_path):
        """A step that joins videos read from disk knows what to write them
        back at without being told - see issue #84."""
        from dw.tasks.video_utils import load_audio_video

        path = self.write_video(tmp_path / "shot.mp4", fps=24, num_frames=24)

        assert load_audio_video(path).fps == 24

    def test_a_loaded_video_argument_carries_the_files_rate(self, tmp_path):
        """The fps that reaches pair_audio: a `video` argument is loaded by
        fetch_video, which used to hand on a bare frame list - so a 24 fps
        file was written back at the 8 fps default, three times long, with
        nothing said about it (#104)."""
        from dw.arguments import fetch_video
        from dw.tasks.video_utils import FrameList

        path = self.write_video(tmp_path / "shot.mp4", fps=24, num_frames=24)

        frames = fetch_video(path)

        assert isinstance(frames, FrameList) and len(frames) == 24
        assert frames.fps == 24

    def test_a_file_that_states_no_rate_stays_a_plain_list(self, tmp_path):
        """Not knowing the rate is a state, not an error - the frames are
        already read by the time it is asked for."""
        from dw.arguments import _with_frame_rate

        assert _with_frame_rate(["a", "b"], str(tmp_path / "gone.mp4")) == ["a", "b"]

    def test_the_rate_reaches_the_paired_video(self, tmp_path):
        """End to end over the two functions #104 sits between."""
        from dw.arguments import fetch_video
        from dw.tasks.pair_audio import pair_audio

        path = self.write_video(tmp_path / "shot.mp4", fps=24, num_frames=24)

        paired = pair_audio(
            video=fetch_video(path),
            audio=numpy.zeros((2, 16000), dtype=numpy.float32),
            sample_rate=16000,
        )

        assert paired.fps == 24

    def test_a_loaded_video_argument_carries_the_run_s_shots(self, tmp_path):
        """A video loaded by path (an asset:/output: reference, already
        resolved to a local file by the time fetch_video sees it) carries
        the shots its run's manifest recorded, the way it already carries
        the file's fps - #398."""
        import json

        from dw.arguments import fetch_video
        from dw.runs import MANIFEST_FILE_NAME
        from dw.tasks.video_utils import FrameList

        run_dir = tmp_path / "ep42" / "20260101-000000-abcdef01"
        run_dir.mkdir(parents=True)
        path = self.write_video(run_dir / "ep42-film.mp4", fps=24, num_frames=24)
        shots = [
            {
                "name": "shot@accuse",
                "start_frame": 0,
                "num_frames": 12,
                "start_sample": None,
                "num_samples": None,
            },
            {
                "name": "shot@deflect",
                "start_frame": 12,
                "num_frames": 12,
                "start_sample": None,
                "num_samples": None,
            },
        ]
        manifest = {
            "steps": [
                {"step": "concat_videos", "files": ["ep42-film.mp4"], "shots": shots}
            ]
        }
        (run_dir / MANIFEST_FILE_NAME).write_text(json.dumps(manifest))

        frames = fetch_video(path)

        assert isinstance(frames, FrameList)
        assert [shot["name"] for shot in frames.shots] == [
            "shot@accuse",
            "shot@deflect",
        ]

    def test_pair_audio_remeasures_the_shots_a_loaded_video_carries(self, tmp_path):
        """The other half of #398: pair_audio's own remeasuring, fed a
        video loaded from a path rather than built by an earlier step in
        the same workflow."""
        import json

        from dw.arguments import fetch_video
        from dw.runs import MANIFEST_FILE_NAME
        from dw.tasks.pair_audio import pair_audio

        run_dir = tmp_path / "ep42" / "20260101-000000-abcdef01"
        run_dir.mkdir(parents=True)
        path = self.write_video(run_dir / "ep42-film.mp4", fps=24, num_frames=24)
        shots = [
            {
                "name": "shot@accuse",
                "start_frame": 0,
                "num_frames": 12,
                "start_sample": None,
                "num_samples": None,
            },
            {
                "name": "shot@deflect",
                "start_frame": 12,
                "num_frames": 12,
                "start_sample": None,
                "num_samples": None,
            },
        ]
        manifest = {
            "steps": [
                {"step": "concat_videos", "files": ["ep42-film.mp4"], "shots": shots}
            ]
        }
        (run_dir / MANIFEST_FILE_NAME).write_text(json.dumps(manifest))

        paired = pair_audio(
            video=fetch_video(path),
            audio=numpy.zeros((2, 16000), dtype=numpy.float32),
            sample_rate=16000,
        )

        assert [shot["name"] for shot in paired.shots] == [
            "shot@accuse",
            "shot@deflect",
        ]
        assert paired.shots[0]["start_sample"] == 0
        assert paired.shots[1]["start_sample"] == round(12 / 24 * 16000)

    def test_audio_is_fitted_to_the_frames_own_duration(self, tmp_path):
        """The codec pads the last block; joined shot after shot that padding
        would walk the sound off the picture."""
        from dw.tasks.video_utils import load_audio_video

        path = self.write_video(tmp_path / "shot.mp4")

        video = load_audio_video(path)

        assert video.audio.shape[1] == 8 / 4 * 8000

    def test_a_silent_video_comes_back_without_audio(self, tmp_path):
        from diffusers.utils.export_utils import encode_video

        from dw.tasks.video_utils import load_audio_video

        path = str(tmp_path / "silent.mp4")
        encode_video(
            [Image.new("RGB", (16, 16)) for _ in range(4)], fps=4, output_path=path
        )

        video = load_audio_video(path)

        assert len(video.frames) == 4
        assert video.audio is None
        assert video.sample_rate is None

    def test_a_disallowed_extension_is_refused(self, tmp_path):
        from dw.security import SecurityError
        from dw.tasks.video_utils import load_audio_video

        payload = tmp_path / "payload.txt"
        payload.write_text("not a video")

        with pytest.raises(SecurityError):
            load_audio_video(str(payload))


class TestVideoFileReference:
    """#367. get_frame/get_first_frame/get_last_frame only need one frame; a
    VideoFileReference lets get_frame seek to it with PyAV instead of
    decoding the whole clip through fetch_video/load_video."""

    def write_long_clip(self, path, num_frames=300, fps=30, marked=()):
        """A clip whose frames are black except the given indexes, which are
        pure red - a marker robust to a lossy codec's compression noise,
        unlike a unique near-black shade per frame."""
        from diffusers.utils.export_utils import encode_video

        marked = set(marked)
        frames = [
            Image.new("RGB", (8, 8), (255, 0, 0) if index in marked else (0, 0, 0))
            for index in range(num_frames)
        ]
        encode_video(frames, fps=fps, output_path=str(path))
        return str(path)

    def assert_is_red(self, frame):
        r, g, b = frame.getpixel((0, 0))
        assert r > 128 and r > g + 64 and r > b + 64

    def assert_is_black(self, frame):
        r, g, b = frame.getpixel((0, 0))
        assert r < 96

    def test_get_frame_seeks_rather_than_decoding_the_whole_clip(self, tmp_path):
        from dw.tasks.video_utils import VideoFileReference

        path = self.write_long_clip(tmp_path / "long.mp4", marked=[250])
        ref = VideoFileReference(path)

        self.assert_is_red(get_frame(ref, 250))
        self.assert_is_black(get_frame(ref, 100))

    def test_negative_indexes_count_from_the_end(self, tmp_path):
        from dw.tasks.video_utils import VideoFileReference

        path = self.write_long_clip(tmp_path / "long.mp4", marked=[299])
        ref = VideoFileReference(path)

        self.assert_is_red(get_frame(ref, -1))

    def test_an_out_of_range_index_names_the_frame_count(self, tmp_path):
        from dw.tasks.video_utils import VideoFileReference

        path = self.write_long_clip(tmp_path / "long.mp4")
        ref = VideoFileReference(path)

        with pytest.raises(ValueError, match="past the end of a 300-frame clip"):
            get_frame(ref, 999999)

    def test_process_video_dispatches_first_and_last_through_the_reference(
        self, tmp_path
    ):
        from dw.tasks.video_utils import VideoFileReference

        path = self.write_long_clip(tmp_path / "long.mp4", marked=[0, 299])
        ref = VideoFileReference(path)

        first = process_video(ref, "get_first_frame", "cpu", {})
        last = process_video(ref, "get_last_frame", "cpu", {})

        self.assert_is_red(first)
        self.assert_is_red(last)

    def test_realize_args_builds_a_reference_without_calling_load_video(
        self, tmp_path, monkeypatch
    ):
        """The whole point of #367: a get_frame step's 'video' must not go
        through the eager, whole-clip fetch_video/load_video path."""
        import dw.arguments as arguments_module
        from dw.tasks.video_utils import VideoFileReference

        path = self.write_long_clip(tmp_path / "long.mp4", marked=[250])

        def _boom(*args, **kwargs):
            raise AssertionError("load_video must not be called for get_frame (#367)")

        monkeypatch.setattr(arguments_module, "load_video", _boom)

        task = {
            "command": "get_frame",
            "arguments": {"video": path, "frame_index": 250},
        }
        arguments_module.realize_args(task, base_dir=str(tmp_path))

        video = task["arguments"]["video"]
        assert isinstance(video, VideoFileReference)
        self.assert_is_red(get_frame(video, 250))

    def test_a_deferred_previous_result_reference_is_left_unchanged(self, tmp_path):
        import dw.arguments as arguments_module

        task = {
            "command": "get_frame",
            "arguments": {"video": "previous_result:shot", "frame_index": 0},
        }
        arguments_module.realize_args(task, base_dir=str(tmp_path))

        assert task["arguments"]["video"] == "previous_result:shot"

    def test_a_variable_reference_is_left_unchanged(self, tmp_path):
        import dw.arguments as arguments_module

        task = {
            "command": "get_frame",
            "arguments": {"video": "variable:my_video"},
        }
        arguments_module.realize_args(task, base_dir=str(tmp_path))

        assert task["arguments"]["video"] == "variable:my_video"


class TestIsVideo:
    def test_the_shapes_that_are_videos(self):
        import numpy
        import torch
        from PIL import Image

        from dw.result import AudioVideo
        from dw.tasks.video_utils import is_video

        assert is_video(AudioVideo([Image.new("RGB", (2, 2))], None, None))
        assert is_video([Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))])
        assert is_video(numpy.zeros((3, 2, 2, 3)))
        assert is_video(torch.zeros((1, 3, 2, 2, 3)))

    def test_the_shapes_that_are_not(self):
        import numpy
        from PIL import Image

        from dw.tasks.video_utils import is_video

        assert not is_video(Image.new("RGB", (2, 2)))
        assert not is_video(numpy.zeros((2, 2, 3)))
        assert not is_video([])
        assert not is_video(["a", "b"])
        assert not is_video("clip.mp4")


class TestLoopFrames:
    """#151. A conditioning input has a length its model was trained to
    read: LTX-2.5's Ingredients IC-LoRA wants its reference sheet as a
    static video of at least 121 frames, and a sheet is one still."""

    def test_a_still_becomes_a_run_of_the_asked_for_length(self):
        looped = loop_frames(Image.new("RGB", (8, 4), "red"), 121)

        assert looped.shape == (121, 4, 8, 3)
        assert (looped[0] == looped[120]).all()

    def test_a_short_clip_laps_round_and_the_last_lap_is_trimmed(self):
        frames = numpy.stack(
            [numpy.full((2, 2, 3), value, dtype=numpy.uint8) for value in (1, 2, 3)]
        )

        looped = loop_frames(frames, 7)

        assert [int(frame[0][0][0]) for frame in looped] == [1, 2, 3, 1, 2, 3, 1]

    def test_a_clip_longer_than_the_request_is_trimmed(self):
        frames = numpy.zeros((10, 2, 2, 3), dtype=numpy.uint8)

        assert len(loop_frames(frames, 4)) == 4

    def test_a_pil_list_is_taken_too(self):
        looped = loop_frames([Image.new("RGB", (2, 2))] * 3, 5)

        assert looped.shape == (5, 2, 2, 3)

    def test_a_count_below_one_is_refused(self):
        with pytest.raises(ValueError) as caught:
            loop_frames(Image.new("RGB", (2, 2)), 0)

        assert "at least 1" in str(caught.value)

    def test_a_count_that_is_not_a_number_is_refused(self):
        with pytest.raises(ValueError):
            loop_frames(Image.new("RGB", (2, 2)), "many")

    def test_a_numeric_string_is_taken(self):
        """A count that arrived through a `variable:` may still be a string."""
        assert len(loop_frames(Image.new("RGB", (2, 2)), "121")) == 121


class TestFrameGrid:
    """#245. A contact sheet for previewing a clip's shape without authoring
    a frames-extraction workflow."""

    @pytest.fixture
    def clip(self):
        from dw.tasks.video_utils import FrameList

        frames = FrameList(
            Image.new("RGB", (64, 32), (index, 0, 0)) for index in range(30)
        )
        frames.fps = 24
        return frames

    def test_default_tiling(self, clip):
        from dw.tasks.video_utils import frame_grid

        grid = frame_grid(clip)

        # count=12 -> rows=isqrt(12)=3, columns=ceil(12/3)=4, tile 320 wide
        # (source 64x32 halved-aspect at width 320 -> height 160)
        assert grid.size == (4 * 320, 3 * 160)

    def test_custom_columns(self, clip):
        from dw.tasks.video_utils import frame_grid

        grid = frame_grid(clip, count=5, columns=2, tile_width=100)

        # 5 tiles over 2 columns is 3 rows, last row left-justified
        assert grid.size == (2 * 100, 3 * 50)

    def test_count_is_clamped_to_the_available_frames(self, clip):
        from dw.tasks.video_utils import frame_grid

        grid = frame_grid(clip, count=100, tile_width=64)

        # clamped to 30 -> rows=isqrt(30)=5, columns=ceil(30/5)=6
        assert grid.size == (6 * 64, 5 * 32)

    def test_a_single_tile(self, clip):
        from dw.tasks.video_utils import frame_grid

        assert frame_grid(clip, count=1, tile_width=160).size == (160, 80)

    def test_label_true_burns_in_a_timestamp(self, clip):
        from dw.tasks.video_utils import frame_grid

        labeled = frame_grid(clip, count=4, tile_width=64, label=True)
        unlabeled = frame_grid(clip, count=4, tile_width=64, label=False)

        assert numpy.asarray(labeled).tobytes() != numpy.asarray(unlabeled).tobytes()

    def test_label_falls_back_to_frame_index_without_fps(self):
        from dw.tasks.video_utils import frame_grid

        frames = [Image.new("RGB", (64, 32), (index, 0, 0)) for index in range(10)]

        # No .fps attribute on a bare list - must not raise
        assert frame_grid(frames, count=4, tile_width=32).size == (2 * 32, 2 * 16)

    def test_a_count_below_one_is_refused(self, clip):
        from dw.tasks.video_utils import frame_grid

        with pytest.raises(ValueError, match="count"):
            frame_grid(clip, count=0)

    def test_a_negative_count_is_refused(self, clip):
        from dw.tasks.video_utils import frame_grid

        with pytest.raises(ValueError, match="count"):
            frame_grid(clip, count=-1)

    def test_a_non_integer_columns_is_refused(self, clip):
        from dw.tasks.video_utils import frame_grid

        with pytest.raises(ValueError, match="columns"):
            frame_grid(clip, columns="many")

    def test_a_non_integer_tile_width_is_refused(self, clip):
        from dw.tasks.video_utils import frame_grid

        with pytest.raises(ValueError, match="tile_width"):
            frame_grid(clip, tile_width=0)

    def test_a_non_bool_label_is_refused(self, clip):
        from dw.tasks.video_utils import frame_grid

        with pytest.raises(ValueError, match="label"):
            frame_grid(clip, label="yes")

    def test_registered_as_a_task_command(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        assert "frame_grid" in _COMMAND_REGISTRY

    def test_a_numeric_string_count_is_taken(self, clip):
        """A count that arrived through a `variable:` may still be a string."""
        from dw.tasks.video_utils import frame_grid

        assert frame_grid(clip, count="4", tile_width=32).size == (2 * 32, 2 * 16)


class TestFitAudioToFrames:
    """Codec padding trimmed off a generated track, in whatever layout the
    waveform arrived in.

    It used to index shape[1] outright, which is only the sample axis for a
    (channels, samples) track: a mono waveform written (samples,) raised
    IndexError on the way to writing a video, and (samples, 1) - a layout
    _as_stereo explicitly supports - measured one channel against a sample
    count and silently fitted nothing.
    """

    # 24 frames at 24 fps of 48 kHz audio is 48000 samples
    FRAMES, FPS, RATE, EXPECTED = 24, 24.0, 48000, 48000

    def fit(self, audio):
        return _fit_audio_to_frames(audio, self.FRAMES, self.FPS, self.RATE)

    @pytest.mark.parametrize(
        "shape, axis",
        [
            ((2, 48100), 1),
            ((48100, 2), 0),
            ((48100,), 0),
            ((48100, 1), 0),
            ((1, 48100), 1),
        ],
    )
    def test_every_layout_is_trimmed_on_its_own_sample_axis(self, shape, axis):
        for audio in (numpy.zeros(shape, numpy.float32), torch.zeros(shape)):
            assert self.fit(audio).shape[axis] == self.EXPECTED

    @pytest.mark.parametrize(
        "shape, axis",
        [
            ((2, 47900), 1),
            ((47900, 2), 0),
            ((47900,), 0),
            ((47900, 1), 0),
            ((1, 47900), 1),
        ],
    )
    def test_every_layout_is_padded_on_its_own_sample_axis(self, shape, axis):
        for audio in (numpy.zeros(shape, numpy.float32), torch.zeros(shape)):
            fitted = self.fit(audio)
            assert fitted.shape[axis] == self.EXPECTED
            # Padding lands on the sample axis only - the channel axis, where
            # there is one, comes back the size it went in
            if len(shape) == 2:
                assert fitted.shape[1 - axis] == shape[1 - axis]

    def test_a_track_of_its_own_length_is_left_alone(self):
        audio = numpy.zeros((2, self.EXPECTED), numpy.float32)

        assert self.fit(audio) is audio

    def test_a_genuinely_different_length_is_left_alone(self):
        # A song laid over a short clip, not codec padding
        audio = numpy.zeros((2, self.EXPECTED * 3), numpy.float32)

        assert self.fit(audio) is audio

    def test_the_samples_are_carried_over_not_just_the_shape(self):
        audio = numpy.arange(48100, dtype=numpy.float32)

        fitted = self.fit(audio)

        assert fitted[0] == 0.0 and fitted[-1] == 47999.0
