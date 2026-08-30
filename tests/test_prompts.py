"""The prompt library: 'prompt:' references resolve to stored text, rooted
at the prompt directory, and never to anything else."""

import json
import os

import pytest

from dw.arguments import is_prompt_reference, realize_args
from dw.prompts import fetch_prompt, get_prompt_dir, load_prompt
from dw.security import InvalidInputError, SecurityError


@pytest.fixture
def prompt_dir(tmp_path, monkeypatch):
    """A prompt library with one top-level and one foldered prompt."""
    library = tmp_path / "prompts"
    (library / "minimax").mkdir(parents=True)
    (library / "scenic.json").write_text(
        json.dumps({"text": "a scenic landscape", "tags": ["landscape"]})
    )
    (library / "minimax" / "fox.json").write_text(
        json.dumps({"text": "a red fox at dawn", "intended_model": "minimax-h3"})
    )
    monkeypatch.setenv("DW_PROMPT_DIR", str(library))
    return library


class TestPromptDir:
    def test_the_environment_names_the_prompt_dir(self, prompt_dir):
        assert get_prompt_dir() == str(prompt_dir)

    def test_without_the_environment_it_is_prompts_in_the_working_dir(
        self, monkeypatch
    ):
        monkeypatch.delenv("DW_PROMPT_DIR", raising=False)
        assert get_prompt_dir() == os.path.abspath("./prompts")


class TestPromptReferences:
    def test_a_reference_resolves_to_its_text(self, prompt_dir):
        assert fetch_prompt("prompt:scenic") == "a scenic landscape"

    def test_a_reference_resolves_one_folder_deep(self, prompt_dir):
        assert fetch_prompt("prompt:minimax/fox") == "a red fox at dawn"

    def test_a_reference_resolves_under_any_argument_name(self, prompt_dir):
        args = {"prompt": "prompt:scenic", "negative_prompt": "prompt:minimax/fox"}
        realize_args(args)
        assert args["prompt"] == "a scenic landscape"
        assert args["negative_prompt"] == "a red fox at dawn"

    def test_a_reference_resolves_inside_a_list(self, prompt_dir):
        args = {"prompts": ["prompt:scenic", "plain text"]}
        realize_args(args)
        assert args["prompts"] == ["a scenic landscape", "plain text"]

    def test_a_reference_resolves_in_a_variable_default(self, prompt_dir):
        # Workflow.run realizes the variables dict before substitution, so a
        # variable whose default is a prompt reference carries the text
        variables = {"prompt": "prompt:scenic"}
        realize_args(variables)
        assert variables["prompt"] == "a scenic landscape"

    def test_a_string_that_is_not_a_reference_is_left_alone(self, prompt_dir):
        args = {"prompt": "a prompt about prompts"}
        realize_args(args)
        assert args["prompt"] == "a prompt about prompts"

    def test_an_unknown_prompt_names_itself(self, prompt_dir):
        with pytest.raises(ValueError, match="missing"):
            fetch_prompt("prompt:missing")

    def test_a_traversing_name_is_rejected(self, prompt_dir):
        with pytest.raises(SecurityError):
            fetch_prompt("prompt:../conftest")

    def test_an_absolute_name_is_rejected(self, prompt_dir):
        with pytest.raises(SecurityError):
            fetch_prompt("prompt:/etc/passwd")

    def test_a_deeper_than_one_folder_name_is_rejected(self, prompt_dir):
        with pytest.raises(InvalidInputError):
            fetch_prompt("prompt:a/b/c")

    def test_an_empty_name_is_rejected(self, prompt_dir):
        with pytest.raises(InvalidInputError):
            fetch_prompt("prompt:")

    def test_is_prompt_reference_only_matches_the_prefix(self):
        assert is_prompt_reference("prompt:scenic")
        assert not is_prompt_reference("a prompt: colon in prose")
        assert not is_prompt_reference(42)


class TestPromptFiles:
    def test_a_file_that_is_not_json_is_rejected(self, prompt_dir):
        (prompt_dir / "broken.json").write_text("not json {")
        with pytest.raises(ValueError, match="not valid JSON"):
            fetch_prompt("prompt:broken")

    def test_a_file_without_text_is_rejected(self, prompt_dir):
        (prompt_dir / "empty.json").write_text(json.dumps({"description": "no text"}))
        with pytest.raises(ValueError, match="not a valid prompt"):
            fetch_prompt("prompt:empty")

    def test_text_that_is_itself_a_reference_is_rejected(self, prompt_dir):
        # Resolved text is substituted where the reference stood - text that
        # begins like a reference would be resolved again, or expand a
        # step's iterations
        for prefix in ("previous_result:gen", "variable:x", "constant:a.B", "prompt:s"):
            (prompt_dir / "sneaky.json").write_text(json.dumps({"text": prefix}))
            with pytest.raises(ValueError, match="may not itself be a reference"):
                fetch_prompt("prompt:sneaky")

    def test_load_prompt_returns_the_whole_definition(self, prompt_dir):
        definition = load_prompt(str(prompt_dir / "minimax" / "fox.json"))
        assert definition["intended_model"] == "minimax-h3"
