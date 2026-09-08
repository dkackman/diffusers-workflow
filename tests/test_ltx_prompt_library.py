"""The stored LTX-2.5 prompts are in the genre the model was trained on.

Lightricks' training captions are one paragraph of roughly 150-220 words in
the present progressive, opening on the action, carrying a shot type, a
camera motion and a viewpoint in prose, with the soundscape interleaved
rather than appended. A stored prompt is what every template runs by
default and what an agent copies, so one in image-generation tag style
teaches the wrong thing twice.
"""

import glob
import json
import os

import pytest

from tests.test_examples import REPO_ROOT

PROMPTS = sorted(glob.glob(os.path.join(REPO_ROOT, "prompts", "ltx2", "*.json")))

# Tag-style and preamble phrases the training spec rules out
FORBIDDEN = (
    "8k",
    "ultra-detailed",
    "photorealistic",
    "vibrant colors",
    "highly detailed",
    "The scene opens",
    "We see",
    "The image is",
)


def _prompt(path):
    return json.load(open(path, encoding="utf-8"))


def test_there_are_ltx_prompts():
    assert len(PROMPTS) == 6


@pytest.mark.parametrize("path", PROMPTS, ids=os.path.basename)
def test_a_prompt_is_one_paragraph_of_caption_length(path):
    text = _prompt(path)["text"]

    assert "\n" not in text.strip(), f"{path} is more than one paragraph"
    words = len(text.split())
    assert (
        140 <= words <= 240
    ), f"{path} is {words} words; the trained caption is 150-220"


@pytest.mark.parametrize("path", PROMPTS, ids=os.path.basename)
def test_a_prompt_carries_no_tag_style_phrase(path):
    text = _prompt(path)["text"]

    for phrase in FORBIDDEN:
        assert phrase.lower() not in text.lower(), f"{path} contains {phrase!r}"


@pytest.mark.parametrize("path", PROMPTS, ids=os.path.basename)
def test_a_prompt_names_the_model_it_is_for(path):
    assert _prompt(path)["intended_model"] == "ltx-2.5"


def test_no_ltx_template_summary_names_the_older_model():
    templates = glob.glob(
        os.path.join(REPO_ROOT, "workflows", "templates", "ltx2", "*.json")
    )
    for path in templates:
        summary = json.load(open(path, encoding="utf-8")).get("summary", "")
        assert "LTX-2 " not in summary and not summary.endswith("LTX-2"), path
