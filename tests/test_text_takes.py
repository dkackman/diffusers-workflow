"""split_takes: a long text into takes a speech model can say in one go."""

import re

import pytest

from dw.tasks.text_takes import split_takes

SENTENCES = [
    "Somebody finished the pistachio and nobody is saying anything.",
    "The fridge hums.",
    "Nobody looks up from their phone, and nobody admits to the crime.",
    "It was a quiet kind of guilt.",
    "Eventually someone laughs.",
]
TEXT = " ".join(SENTENCES * 4)


def words(text):
    return len(text.split())


def test_no_take_is_over_the_limit():
    takes = split_takes(TEXT, max_seconds=10, words_per_second=2.5)
    assert len(takes) > 1
    assert all(words(t) <= 25 for t in takes)


def test_takes_end_on_sentence_boundaries():
    for take in split_takes(TEXT, max_seconds=10, words_per_second=2.5):
        assert re.search(r"[.!?]$", take)


def test_takes_rejoin_to_the_text():
    takes = split_takes(TEXT, max_seconds=10, words_per_second=2.5)
    assert " ".join(takes).split() == TEXT.split()


def test_short_text_is_one_take():
    assert split_takes("Hello there.", 60, 2.5) == ["Hello there."]


def test_a_long_sentence_splits_at_its_last_comma_before_the_limit():
    sentence = "one two three, four five six, seven eight nine ten eleven twelve."
    takes = split_takes(sentence, max_seconds=4, words_per_second=2.5)  # 10 words
    assert takes[0] == "one two three, four five six,"
    assert " ".join(takes).split() == sentence.split()
    assert all(words(t) <= 10 for t in takes)


def test_a_long_sentence_without_commas_splits_at_a_word():
    sentence = " ".join(f"w{i}" for i in range(25)) + "."
    takes = split_takes(sentence, max_seconds=4, words_per_second=2.5)
    assert all(words(t) <= 10 for t in takes)
    assert " ".join(takes).split() == sentence.split()


def test_paragraph_breaks_are_whitespace():
    takes = split_takes("First one.\n\nSecond one.", 60, 2.5)
    assert " ".join(takes).split() == ["First", "one.", "Second", "one."]


@pytest.mark.parametrize("bad", [0, -1, float("nan")])
def test_bad_limits_are_refused(bad):
    with pytest.raises(ValueError):
        split_takes("Hi.", max_seconds=bad, words_per_second=2.5)
    with pytest.raises(ValueError):
        split_takes("Hi.", max_seconds=10, words_per_second=bad)


def test_empty_text_is_refused():
    with pytest.raises(ValueError):
        split_takes("   ", 60, 2.5)


def test_the_registered_task_answers_the_takes():
    import dw.tasks.task  # noqa: F401
    from dw.tasks.registry import _COMMAND_INFO

    assert _COMMAND_INFO["split_takes"]["returns"] == "json"
    from dw.tasks.text_takes import split_takes_task

    out = split_takes_task(TEXT, max_seconds=10)
    assert out["count"] == len(out["takes"]) > 1


def test_abbreviations_are_not_sentence_ends():
    takes = split_takes("Hi there now. Dr. Smith went home. Mr. Lee left.", 3, 1)
    assert all(not t.endswith(("Dr.", "Mr.")) for t in takes)
    takes = split_takes("see e.g. this one. Next.", 3, 1)
    assert all(not t.endswith("e.g.") for t in takes)


def test_an_initial_is_not_a_sentence_end():
    from dw.tasks.text_takes import _sentences

    assert _sentences("J. R. Smith spoke.") == ["J. R. Smith spoke."]
    assert _sentences("It ended. Then J. Smith spoke.") == [
        "It ended.",
        "Then J. Smith spoke.",
    ]


def test_narration_long_validates_and_names_its_child():
    import json
    from pathlib import Path

    from dw.workflow import Workflow

    root = Path(__file__).resolve().parent.parent / "workflows" / "templates"
    path = root / "narration-long.json"
    definition = json.loads(path.read_text())
    assert Workflow(definition, str(root), str(path)).validation_errors() == []
    child = definition["steps"][1]["workflow"]
    assert child["path"] == "./narration-from-takes.json"
    assert child["arguments"]["takes"] == "previous_result:split.takes"
    assert (root / "narration-from-takes.json").exists()


def test_a_child_receives_the_parents_takes(tmp_path):
    import json

    from dw.workflow import Workflow

    child = {
        "id": "child",
        "variables": {"takes": ["x"]},
        "steps": [
            {
                "name": "each",
                "for_each": "variable:takes",
                "task": {
                    "command": "split_takes",
                    "arguments": {"text": "item:"},
                },
                "result": {"content_type": "application/json", "save": False},
            }
        ],
    }
    (tmp_path / "child.json").write_text(json.dumps(child))
    parent = {
        "id": "parent",
        "variables": {"text": "One short. Two short. Three short."},
        "steps": [
            {
                "name": "split",
                "task": {
                    "command": "split_takes",
                    "arguments": {
                        "text": "variable:text",
                        "max_seconds": 2,
                        "words_per_second": 1,
                    },
                },
                "result": {"content_type": "application/json", "save": False},
            },
            {
                "name": "run",
                "workflow": {
                    "path": "./child.json",
                    "arguments": {"takes": "previous_result:split.takes"},
                },
                "result": {"content_type": "application/json", "save": False},
            },
        ],
    }
    workflow = Workflow(parent, str(tmp_path), str(tmp_path / "parent.json"))
    assert workflow.validation_errors() == []
    seen = []
    from dw.workflow import Workflow as W

    original = W.run

    def spy(self, arguments, *a, **k):
        seen.append(arguments)
        return original(self, arguments, *a, **k)

    from unittest.mock import patch

    with patch.object(W, "run", spy):
        workflow.run({})
    child_args = [s for s in seen if isinstance(s, dict) and "takes" in s]
    assert child_args[0]["takes"] == ["One short.", "Two short.", "Three short."]


def test_music_long_refuses_a_section_over_300_seconds():
    import json
    from pathlib import Path

    from dw.workflow import Workflow

    root = Path(__file__).resolve().parent.parent / "workflows" / "templates"
    path = root / "minimax" / "music-long.json"
    definition = json.loads(path.read_text())
    assert Workflow(definition, str(path.parent), str(path)).validation_errors() == []
    definition["variables"]["sections"][0]["audio_duration"] = 301
    errors = Workflow(definition, str(path.parent), str(path)).validation_errors()
    assert any("300" in json.dumps(e) for e in errors), errors


def test_music_long_slice_audio_steps_carry_a_sample_rate():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "workflows" / "templates"
    definition = json.loads((root / "minimax" / "music-long.json").read_text())
    slices = [
        s
        for s in definition["steps"]
        if s.get("task", {}).get("command") == "slice_audio"
    ]
    assert slices
    for step in slices:
        assert "sample_rate" in step["task"]["arguments"], step["name"]
