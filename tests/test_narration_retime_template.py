"""templates/narration-retime exposes `boundary_slack_seconds` (#1311, stage
#1336): the template description, audio.md and TASKS.md all advise raising it
when many `mid_word_boundaries` appear, so the template must carry it to the
`retime_narration` task, and refuse a value the task would refuse."""

import json
from pathlib import Path

import pytest

from dw.validation import workflow_errors
from dw.variables import replace_variables, resolve_variable_values
from dw.workflow import workflow_from_definition

TEMPLATE = (
    Path(__file__).resolve().parent.parent / "workflows/templates/narration-retime.json"
)
SCRIPT = "We the People of the United States, in Order to form a more perfect Union."


def load():
    return json.loads(TEMPLATE.read_text())


def retime_arguments(variables):
    definition = load()
    merged = resolve_variable_values({**definition["variables"], **variables})
    substituted = replace_variables(definition, merged)
    steps = {s["name"]: s for s in substituted["steps"]}
    return steps["retime"]["task"]["arguments"]


def messages(tmp_path, arguments):
    workflow = workflow_from_definition(load(), str(tmp_path))
    return [str(e.get("message")) for e in workflow_errors(workflow, arguments)]


def test_the_variable_defaults_to_the_tasks_own_default():
    import inspect

    from dw.tasks.narration_retime import retime_narration

    default = inspect.signature(retime_narration).parameters["boundary_slack_seconds"]
    assert load()["variables"]["boundary_slack_seconds"] == default.default


def test_boundary_slack_seconds_reaches_the_retime_task():
    assert retime_arguments({})["boundary_slack_seconds"] == 0.25
    assert retime_arguments({"boundary_slack_seconds": 0.6})[
        "boundary_slack_seconds"
    ] == pytest.approx(0.6)


def test_the_registered_handler_passes_it_to_the_implementation(monkeypatch):
    """The real dispatch path: the substituted step arguments go through the
    registered `retime_narration` handler into the implementation."""
    import dw.tasks  # noqa: F401  (registers the commands)
    from dw.tasks import narration_retime
    from dw.tasks.registry import _COMMAND_REGISTRY

    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(narration_retime, "retime_narration", capture)

    class Task:
        def device_for(self, arguments):
            return "cpu"

    arguments = retime_arguments({"script": SCRIPT, "boundary_slack_seconds": 0.5})
    _COMMAND_REGISTRY["retime_narration"](Task(), dict(arguments), {})
    assert seen["boundary_slack_seconds"] == pytest.approx(0.5)


def test_a_valid_value_validates(tmp_path):
    assert messages(tmp_path, {"script": SCRIPT, "boundary_slack_seconds": 0.5}) == []


@pytest.mark.parametrize("value", [-0.1, "wide"])
def test_an_invalid_value_is_refused_at_validate(tmp_path, value):
    found = messages(tmp_path, {"script": SCRIPT, "boundary_slack_seconds": value})
    assert any("boundary_slack_seconds" in m for m in found), found
