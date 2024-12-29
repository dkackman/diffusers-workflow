import pytest
from dh.tasks.task import Task
from PIL import Image

def test_qr_code_task():
    task_def = {
        "command": "qr_code",
        "arguments": {
            "qr_code_contents": "test content"
        }
    }
    
    task = Task(task_def, "cpu")
    result = task.run({"qr_code_contents": "test content"})
    
    assert isinstance(result, Image.Image)
    assert result.size == (768, 768)  # Default size

def test_unknown_task():
    task_def = {
        "command": "unknown_command",
        "arguments": {}
    }
    
    task = Task(task_def, "cpu")
    with pytest.raises(ValueError) as exc_info:
        task.run({})
    assert "Unknown task" in str(exc_info.value)
