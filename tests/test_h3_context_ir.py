"""The H3 Context-IR builtin teaches what MiniMax's prompt-writing guides say.

The system prompt is a compression of the two guides on the model card
(docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md and _ref_en.md). The 2026-09-07
audit found three things it got wrong and one it left out; these tests keep
the corrections in place and the unsourced lines out.
"""

import glob
import json
import os

from dw.security import MAX_VARIABLE_VALUE_LENGTH
from tests.test_examples import BUILTIN_DIR, REPO_ROOT


def _system_prompt():
    path = os.path.join(BUILTIN_DIR, "h3_context_ir.json")
    return json.load(open(path, encoding="utf-8"))["variables"]["system_prompt"]


def test_the_sources_are_named():
    prompt = _system_prompt()

    assert "VIDEO_PROMPT_WRITING_GUIDE_base_en.md" in prompt
    assert "VIDEO_PROMPT_WRITING_GUIDE_ref_en.md" in prompt


def test_silent_audio_fields_are_written_as_not_applicable():
    """Without this the enhancer invents a soundscape for a silent brief."""
    prompt = _system_prompt()

    assert "N/A" in prompt
    assert "overall_soundscape" in prompt[prompt.index("N/A") - 600 : prompt.index("N/A") + 600]


def test_video_and_audio_references_are_numbered_within_their_own_category():
    prompt = _system_prompt()

    assert "numbered independently" in prompt
    assert "does not by itself" in prompt


def test_continuity_modes_are_labelled_as_this_engine_convention():
    prompt = _system_prompt()

    assert "not part of Context-IR" in prompt


def test_the_dialogue_fidelity_rules_are_present():
    prompt = _system_prompt()

    assert "[unclear]" in prompt
    assert "retention_analysis" in prompt and "(Sx)" in prompt


def test_the_unsourced_lines_are_gone():
    prompt = _system_prompt()

    assert "degrades on anything else" not in prompt
    assert "8k" not in prompt
    assert "artist names" not in prompt
    # The confirmed part of that line stays
    assert "no negative prompt" in prompt


def test_the_context_ir_system_prompt_fits_the_variable_length_limit():
    """The prompt reaches set_variables as a sub-workflow argument whenever a template
    passes it explicitly, so it must fit MAX_VARIABLE_VALUE_LENGTH."""
    prompt = _system_prompt()

    assert len(prompt) <= MAX_VARIABLE_VALUE_LENGTH, (
        f"the Context-IR system prompt is {len(prompt)} chars, "
        f"over the {MAX_VARIABLE_VALUE_LENGTH} limit"
    )


def _system_prompt_values(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "system_prompt":
                yield value
            yield from _system_prompt_values(value)
    elif isinstance(node, list):
        for item in node:
            yield from _system_prompt_values(item)


def test_no_catalog_template_overrides_the_context_ir_system_prompt():
    """The builtin carries the corrected prompt; a stored copy would go stale."""
    pattern = os.path.join(REPO_ROOT, "workflows", "templates", "minimax", "*.json")
    paths = glob.glob(pattern)
    assert paths

    for path in paths:
        with open(path, encoding="utf-8") as f:
            spec = json.load(f)

        for value in _system_prompt_values(spec):
            assert not (
                isinstance(value, str)
                and value.startswith("prompt:prompt_enhancement/minimax_h3")
            ), f"{path}: overrides the builtin's system prompt with a stored copy"
