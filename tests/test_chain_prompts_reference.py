"""A chain's `prompts` and `lines` may each be one `previous_result:` reference
to a JSON list, the way a `narration_lines` task step hands them on."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dw.pipeline_processors.chain_config import ChainConfig
from dw.previous_results import resolve_chain_lists


class FakeResult:
    """Stands in for a stored step result holding one JSON dict."""

    def __init__(self, payload):
        self.payload = payload

    def get_artifacts(self):
        return [self.payload]

    def get_artifact_properties(self, name):
        return [self.payload[name]]


def _action(**chain):
    return SimpleNamespace(name="video", pipeline_definition={"chain": chain})


def test_prompts_resolve_from_one_list_reference():
    action = _action(segments=3, prompts="previous_result:lines.prompts")
    previous = {"lines": FakeResult({"lines": ["x"], "prompts": ["a", "b", "c"]})}
    resolve_chain_lists(action, previous)
    assert action.chain_prompts == ["a", "b", "c"]


def test_lines_resolve_from_one_list_reference():
    action = _action(segments=2, lines="previous_result:lines.lines")
    previous = {"lines": FakeResult({"lines": ["x", "y"], "prompts": ["a", "b"]})}
    resolve_chain_lists(action, previous)
    assert action.chain_lines == ["x", "y"]


def test_resolved_lines_reach_the_chain_config():
    config = ChainConfig({"segments": 2}, {"prompt": "p"}, None, None, ["x", "y"])
    assert config.lines == ["x", "y"]


@pytest.mark.parametrize("key", ["prompts", "lines"])
@pytest.mark.parametrize("value", ["just text", {"a": 1}, [1, 2]])
def test_list_reference_to_a_non_list_is_refused(key, value):
    action = _action(segments=2, **{key: "previous_result:lines.value"})
    previous = {"lines": FakeResult({"value": value})}
    with pytest.raises(ValueError) as raised:
        resolve_chain_lists(action, previous)
    assert key in str(raised.value)
    assert "previous_result:lines.value" in str(raised.value)


def test_entry_references_keep_working():
    action = _action(segments=2, prompts=["previous_result:lines.first", "literal"])
    previous = {"lines": FakeResult({"first": "opening"})}
    resolve_chain_lists(action, previous)
    assert action.chain_prompts == ["opening", "literal"]


def test_a_literal_string_prompts_is_still_refused():
    with pytest.raises(ValueError, match="one prompt per segment"):
        resolve_chain_lists(_action(prompts="abc"), {})


def test_validate_accepts_an_unresolved_list_reference(tmp_path):
    from dw.validation import workflow_errors
    from dw.workflow import workflow_from_definition

    template = (
        Path(__file__).parent.parent / "workflows/templates/ltx2/chained-segments.json"
    )
    definition = json.loads(template.read_text())
    # The step that hands the lists on, ahead of the chain step
    definition["steps"].insert(
        0,
        {
            "name": "lines",
            "task": {
                "command": "check_narration",
                "arguments": {"text": "hello", "audio": "asset:narration.wav"},
            },
            "result": {"content_type": "application/json"},
        },
    )
    block = definition["steps"][1]["pipeline"]["chain"]
    block.pop("duration_seconds", None)
    block["segments"] = 3
    block["prompts"] = "previous_result:lines.prompts"
    block["lines"] = "previous_result:lines.lines"
    errors = workflow_errors(workflow_from_definition(definition, str(tmp_path)))
    assert not [e for e in errors if "chain" in str(e.get("path"))], errors
