"""
Unit tests for video frame extraction and its task-command registration.

process_video dispatches through a hand-written if-chain while Task keeps a
separate hand-written list of the names it accepts, so the two are tested
against each other here.
"""

import pytest
from PIL import Image

from dw.tasks.task import _VIDEO_PROCESSOR_COMMANDS, Task
from dw.tasks.video_utils import get_frame, process_video


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
