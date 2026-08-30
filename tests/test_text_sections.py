"""Tests for the extract_sections task."""

import glob
import json
import os

import pytest

from dw.tasks.text_sections import extract_sections

SECTIONS = ["alpha", "beta", "gamma"]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The layout the MiniMax H3 prompts use, covering both the three-field tasks
# and the six-field reference tasks
H3_SECTIONS = [
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
]


def test_keeps_the_named_sections():
    text = "alpha: one\n\nbeta: two\n\ngamma: three"
    assert extract_sections(text, SECTIONS) == text


def test_drops_an_unlabelled_trailing_block():
    text = "alpha: one\n\nbeta: two\n\ngamma: three\n\nAnd then it kept writing."
    assert extract_sections(text, SECTIONS) == "alpha: one\n\nbeta: two\n\ngamma: three"


def test_drops_a_repeated_section():
    # A model that starts over emits the labels again - the first one wins
    text = "alpha: one\n\nbeta: two\n\nbeta: two again\n\ngamma: three"
    assert extract_sections(text, SECTIONS) == "alpha: one\n\nbeta: two\n\ngamma: three"


def test_keeps_the_preamble():
    text = "An instruction line.\n\nalpha: one\n\nbeta: two"
    assert extract_sections(text, SECTIONS).startswith("An instruction line.")


def test_preamble_can_be_dropped():
    text = "An instruction line.\n\nalpha: one"
    result = extract_sections(text, SECTIONS, keep_preamble=False)
    assert result == "alpha: one"


def test_a_label_on_its_own_line_keeps_its_body():
    # The models put the body on the next line as often as not
    text = "alpha: \nthe body\n\nbeta: two"
    assert extract_sections(text, SECTIONS) == "alpha: the body\n\nbeta: two"


def test_sections_come_out_in_declared_order():
    text = "gamma: three\n\nalpha: one"
    assert extract_sections(text, SECTIONS) == "alpha: one\n\ngamma: three"


def test_multi_line_section_survives():
    # A single newline is inside a section - only a blank line ends one, which
    # is what keeps a one-line-per-item field intact
    text = "alpha: first\nsecond\nthird\n\nbeta: two"
    assert (
        extract_sections(text, SECTIONS) == "alpha: first\nsecond\nthird\n\nbeta: two"
    )


def test_missing_sections_are_skipped():
    text = "alpha: one\n\ngamma: three"
    assert extract_sections(text, SECTIONS) == "alpha: one\n\ngamma: three"


def test_text_without_any_section_is_left_alone():
    text = "Nothing here looks like a field."
    assert extract_sections(text, SECTIONS) == text


def test_no_sections_requested_is_a_passthrough():
    text = "alpha: one\n\nleftovers"
    assert extract_sections(text, []) == text


def h3_prompts():
    """Every hand-written H3 prompt in the examples, as (file, key, text)."""
    found = []
    pattern = os.path.join(REPO_ROOT, "workflows", "minimax", "MiniMaxH3*.json")
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as handle:
            workflow = json.load(handle)
        for key, value in workflow.get("variables", {}).items():
            if isinstance(value, str) and "non_diegetic_music:" in value:
                found.append((os.path.basename(path), key, value))
    return found


@pytest.mark.parametrize("name,key,prompt", h3_prompts())
def test_a_hand_written_prompt_passes_through_unchanged(name, key, prompt):
    """The trim must be a no-op on a prompt that is already well formed.

    These are the prompts the examples ship, in both the three-field and
    six-field layouts - if trimming rewrites one of them it is removing content
    rather than trailing junk.
    """
    assert extract_sections(prompt, H3_SECTIONS).split() == prompt.split()


class TestExtractSectionsRegistration:
    def test_command_registered(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        assert "extract_sections" in _COMMAND_REGISTRY


def test_labels_match_regardless_of_case():
    # Models capitalise labels however they please - one wrote
    # Overall_soundscape where the spec says overall_soundscape, and dropping
    # the section over its first letter would lose real content
    text = "Alpha: one\n\nBETA: two\n\ngamma: three"
    assert extract_sections(text, SECTIONS) == "alpha: one\n\nbeta: two\n\ngamma: three"


def test_a_case_variant_repeat_is_still_a_repeat():
    text = "alpha: one\n\nAlpha: one again\n\nbeta: two"
    assert extract_sections(text, SECTIONS) == "alpha: one\n\nbeta: two"
