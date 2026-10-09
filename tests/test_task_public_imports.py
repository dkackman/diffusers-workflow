"""#784: no dw/tasks module imports a private (underscore) name from a sibling
task module. A shared helper is public in its owning module."""

import ast
from pathlib import Path

# The registry's own state, read by task.py; not a task helper (#692).
ALLOWED = {"task.py: registry._COMMAND_INFO", "task.py: registry._COMMAND_REGISTRY"}

TASKS = Path(__file__).parent.parent / "dw" / "tasks"


def _private_sibling_imports():
    found = []
    for path in sorted(TASKS.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and (
                node.level == 1
                and node.module
                or (node.module or "").startswith("dw.tasks.")
            ):
                found += [
                    f"{path.name}: {node.module}.{a.name}"
                    for a in node.names
                    if a.name.startswith("_")
                ]
    return found


def test_no_private_cross_task_imports():
    assert set(_private_sibling_imports()) - ALLOWED == set()


def test_renamed_names_are_public():
    from dw.tasks.text_generation import DEFAULT_VISION_MODEL
    from dw.tasks.upscale import resolve_model_path
    from dw.tasks.video_utils import frames_of

    assert DEFAULT_VISION_MODEL and callable(resolve_model_path) and callable(frames_of)
