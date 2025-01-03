import pytest
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
