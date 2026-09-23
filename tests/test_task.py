import pytest
from unittest.mock import patch
from dw.tasks.task import Task
from PIL import Image


def test_qr_code_task():
    task_def = {"command": "qr_code", "arguments": {"qr_code_contents": "test content"}}

    task = Task(task_def, "cpu")
    result = task.run({"qr_code_contents": "test content"})

    assert isinstance(result, Image.Image)
    assert result.size == (768, 768)  # Default size


def test_unknown_task():
    task_def = {"command": "unknown_command", "arguments": {}}

    task = Task(task_def, "cpu")
    with pytest.raises(ValueError) as exc_info:
        task.run({})
    assert "Unknown task" in str(exc_info.value)


def test_unknown_task_with_an_image_argument_still_raises():
    # A typo'd command used to be silently routed into process_image whenever
    # an "image" argument was present, dying with a misleading "Unknown image
    # processor type" error instead of the task-level "Unknown task command".
    task_def = {"command": "resize_rescle", "arguments": {}}  # typo: rescle

    task = Task(task_def, "cpu")
    with pytest.raises(ValueError) as exc_info:
        task.run({"image": Image.new("RGB", (4, 4))})

    assert "Unknown task command" in str(exc_info.value)
    assert "Unknown image processor" not in str(exc_info.value)


def test_unknown_task_error_lists_known_commands():
    task_def = {"command": "totally_bogus_command", "arguments": {}}

    task = Task(task_def, "cpu")
    with pytest.raises(ValueError) as exc_info:
        task.run({})

    message = str(exc_info.value)
    assert "Registered commands" in message
    assert "qr_code" in message  # a registered command
    assert "Image processors" in message
    assert "resize_rescale" in message  # a known image processor
    assert "Video processors" in message
    assert "get_frame" in message  # a known video processor


def test_image_processor_command_dispatches_without_registry_entry():
    # resize_rescale is not in _COMMAND_REGISTRY - it must still resolve via
    # image_utils.available_processors() rather than raising.
    task_def = {"command": "resize_rescale", "arguments": {}}
    task = Task(task_def, "cpu")

    result = task.run({"image": Image.new("RGB", (8, 8)), "height": 4, "width": 4})

    assert isinstance(result, Image.Image)
    assert result.size == (4, 4)


def test_gather_images_task_dispatches_to_gather():
    urls = ["https://example.com/a.jpg", "https://example.com/b.jpg"]
    images = [Image.new("RGB", (4, 4)), Image.new("RGB", (8, 8))]
    task_def = {"command": "gather_images", "arguments": {"urls": urls}}
    task = Task(task_def, "cpu")

    with (
        patch(
            "dw.tasks.gather.validate_media_url", side_effect=lambda url, what=None: url
        ),
        patch("dw.tasks.gather.load_image", side_effect=images) as load_image,
    ):
        result = task.run(task_def["arguments"])

    assert result == images
    assert [call.args[0] for call in load_image.call_args_list] == urls


def test_format_chat_message_task():
    task_def = {
        "command": "format_chat_message",
        "arguments": {"system_prompt": "Hello, world!", "user_message": "unit_test"},
    }
    task = Task(task_def, "cpu")
    result = task.run(task_def["arguments"])

    # Check the overall structure
    assert isinstance(result, dict), (
        "Expected a formatted dict from format_chat_message"
    )
    assert "text_inputs" in result, "Result should contain text_inputs key"

    # Check the text_inputs array structure
    text_inputs = result["text_inputs"]
    assert isinstance(text_inputs, list), "text_inputs should be a list"
    assert len(text_inputs) == 2, "text_inputs should contain exactly 2 messages"

    # Check system message
    assert text_inputs[0]["role"] == "system", "First message should have role 'system'"
    assert text_inputs[0]["content"] == "Hello, world!", (
        "System message content mismatch"
    )

    # Check user message
    assert text_inputs[1]["role"] == "user", "Second message should have role 'user'"
    assert text_inputs[1]["content"] == "unit_test", "User message content mismatch"


def test_batch_decode_post_process_task():
    # We use a mock pipeline to simulate previous_pipelines behavior.
    class MockPipeline:
        def batch_decode(self, generated_ids, skip_special_tokens=False):
            return [f"decoded-{inp}" for inp in generated_ids]

        def post_process_generation(self, generated_text, task):
            return {task: generated_text}

    mock_previous_pipelines = {
        "test_pipe_ref": type(
            "MockPipelineWrapper", (object,), {"pipeline": MockPipeline()}
        )()
    }

    task_def = {
        "command": "batch_decode_post_process",
        "pipeline_reference": "test_pipe_ref",
        "arguments": {
            "generated_ids": ["foo", "bar"],
            "task": "<DETAILED_CAPTION>",
        },
    }
    task = Task(task_def, "cpu")
    result = task.run(task_def["arguments"], previous_pipelines=mock_previous_pipelines)
    # The first decoded sequence, post-processed and read back under the task key
    assert result == "decoded-foo"


class TestTaskDevice:
    """Test the device a task runs on"""

    def test_defaults_to_the_workflow_device(self, all_backends_available):
        task = Task({"command": "upscale", "arguments": {}}, "cuda")
        assert task.device_for({}) == "cuda"

    def test_arguments_can_override_the_device(self):
        task = Task({"command": "upscale", "arguments": {}}, "cuda")
        assert task.device_for({"device": "cpu"}) == "cpu"

    def test_the_override_is_consumed(self):
        # Left in place it would reach the command as a duplicate argument
        task = Task({"command": "upscale", "arguments": {}}, "cuda")
        arguments = {"device": "cpu", "model_name": "test"}

        task.device_for(arguments)

        assert arguments == {"model_name": "test"}

    def test_a_command_accepts_a_device_argument(self):
        # image_to_text takes device as a keyword - a device in the arguments used to
        # collide with it rather than override it
        task = Task({"command": "image_to_text", "arguments": {}}, "cuda")

        # Patched at its source - task.py imports it inside the handler
        with patch("dw.tasks.image_to_text.image_to_text") as image_to_text:
            task.run({"image": "an image", "device": "cpu"})

        assert image_to_text.call_args.kwargs["device"] == "cpu"


class TestTaskSeed:
    """Test the seed a task run consumes - mirrors TestTaskDevice (#261)"""

    def test_defaults_to_the_workflow_seed(self):
        task = Task({"command": "generate_speech", "arguments": {}}, "cpu", seed=7)
        assert task.seed_for({}) == 7

    def test_no_seed_anywhere_is_none(self):
        task = Task({"command": "generate_speech", "arguments": {}}, "cpu")
        assert task.seed_for({}) is None

    def test_arguments_can_override_the_seed(self):
        task = Task({"command": "generate_speech", "arguments": {}}, "cpu", seed=7)
        assert task.seed_for({"seed": 42}) == 42

    def test_the_override_is_consumed(self):
        # Left in place it would reach the command as a duplicate argument
        task = Task({"command": "generate_speech", "arguments": {}}, "cpu", seed=7)
        arguments = {"seed": 42, "model_name": "test"}

        task.seed_for(arguments)

        assert arguments == {"model_name": "test"}


class TestImageTasksTakeAVideo:
    """An image command bound to a video runs over its frames and hands the
    soundtrack through, so a generated clip can be upscaled or processed
    without losing what was generated alongside it."""

    def video(self):
        import numpy
        from dw.result import AudioVideo

        frames = [Image.new("RGB", (8, 8), (i * 40, 0, 0)) for i in range(3)]
        return AudioVideo(frames, numpy.zeros((2, 50), dtype=numpy.float32), 100)

    def test_an_image_processor_maps_over_the_frames(self):
        from dw.result import AudioVideo

        task = Task({"command": "resize_rescale", "arguments": {}}, "cpu")
        result = task.run({"image": self.video(), "height": 4, "width": 4})

        assert isinstance(result, AudioVideo)
        assert [f.size for f in result.frames] == [(4, 4)] * 3
        assert [f.getpixel((0, 0))[0] for f in result.frames] == [0, 40, 80]
        assert result.audio.shape == (2, 50) and result.sample_rate == 100

    def test_a_model_backed_command_maps_over_the_frames(self):
        from dw.result import AudioVideo

        with patch("dw.tasks.upscale.upscale_image") as upscale:
            upscale.side_effect = lambda image, model_name, device, **kw: image.resize(
                (16, 16)
            )
            task = Task({"command": "upscale", "arguments": {}}, "cpu")
            result = task.run({"image": self.video(), "model_name": "x/y"})

        assert isinstance(result, AudioVideo)
        assert upscale.call_count == 3
        assert all(f.size == (16, 16) for f in result.frames)
        assert result.sample_rate == 100

    def test_a_frame_array_becomes_one_video_artifact(self):
        import numpy
        from dw.result import AudioVideo

        frames = numpy.zeros((3, 8, 8, 3), dtype=numpy.uint8)
        task = Task({"command": "resize_rescale", "arguments": {}}, "cpu")
        result = task.run({"image": frames, "height": 4, "width": 4})

        assert isinstance(result, AudioVideo)
        assert len(result.frames) == 3 and result.audio is None

    def test_a_single_image_is_processed_as_itself(self):
        task = Task({"command": "resize_rescale", "arguments": {}}, "cpu")
        result = task.run({"image": Image.new("RGB", (8, 8)), "height": 4, "width": 4})

        assert isinstance(result, Image.Image)
