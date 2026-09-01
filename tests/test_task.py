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


@pytest.mark.skip(reason="Requires network access to external URLs which may be flaky")
def test_gather_images_task():
    task_def = {
        "command": "gather_images",
        "arguments": {
            "urls": [
                "https://pbs.twimg.com/media/Gf5iaDGXsAA0R30?format=jpg&name=small",
                "https://pbs.twimg.com/media/Gf7vNQJXoAAY5Cm?format=jpg&name=small",
            ]
        },
    }
    task = Task(task_def, "cpu")
    result = task.run(task_def["arguments"])
    assert isinstance(result, list), "Expected a list of images from gather_images"


def test_gather_inputs_task():
    task_def = {"command": "gather_inputs", "inputs": ["value1", "value2"]}
    task = Task(task_def, "cpu")
    result = task.run(task_def["inputs"])
    assert isinstance(result, list), "Expected a list of inputs from gather_inputs"
    assert "value1" in result and "value2" in result, "Should gather all passed inputs"


def test_format_chat_message_task():
    task_def = {
        "command": "format_chat_message",
        "arguments": {"system_prompt": "Hello, world!", "user_message": "unit_test"},
    }
    task = Task(task_def, "cpu")
    result = task.run(task_def["arguments"])

    # Check the overall structure
    assert isinstance(
        result, dict
    ), "Expected a formatted dict from format_chat_message"
    assert "text_inputs" in result, "Result should contain text_inputs key"

    # Check the text_inputs array structure
    text_inputs = result["text_inputs"]
    assert isinstance(text_inputs, list), "text_inputs should be a list"
    assert len(text_inputs) == 2, "text_inputs should contain exactly 2 messages"

    # Check system message
    assert text_inputs[0]["role"] == "system", "First message should have role 'system'"
    assert (
        text_inputs[0]["content"] == "Hello, world!"
    ), "System message content mismatch"

    # Check user message
    assert text_inputs[1]["role"] == "user", "Second message should have role 'user'"
    assert text_inputs[1]["content"] == "unit_test", "User message content mismatch"


@pytest.mark.skip(reason="Test not fully implemented yet")
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
    assert result == [
        "decoded-foo",
        "decoded-bar",
    ], "Should return batch-decoded strings"


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
