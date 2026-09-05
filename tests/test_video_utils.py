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
from dw.tasks.video_utils import extract_frame, frame_count, get_frame, process_video


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
        assert process_video(video, "get_last_frame", "cpu", {"frame_index": 0}) is (
            video[3]
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
