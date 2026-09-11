"""The dialogue-short template's optional voice references.

A recurring cast's voice was carried only as a description repeated verbatim
in every shot - about thirty hand-copied strings across three episodes, with
nothing checking that they matched (2026-09-11). `character_a_voice` /
`character_b_voice` name a clip instead, and each shot appends an audio
reference for whoever speaks in it. Null is the default, and null means the
reference is not there at all - so the template still generates exactly what
it generated before the variables existed.
"""

import json
import os

import pytest

from dw.arguments import realize_args
from dw.variables import replace_variables

TEMPLATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "workflows",
    "templates",
    "minimax",
    "dialogue-short.json",
)

# Which character speaks in which shot - the template appends a voice
# reference per speaker, and this is the mapping it is asserted against
SPEAKERS = {
    "shot_1_cold_open": ("a", "b"),
    "shot_2_deflect": ("b",),
    "shot_3_react": ("a",),
    "shot_4_button": ("b",),
    "shot_5_tag": ("a", "b"),
}


@pytest.fixture
def definition():
    with open(TEMPLATE, encoding="utf-8") as file:
        return json.load(file)


def shot_references(definition, variables):
    """Every shot's references, as the engine realizes them for a run."""
    resolved = replace_variables(definition, variables)
    references = {}
    for step in resolved["steps"]:
        if step["name"] not in SPEAKERS:
            continue
        block = step.get("pipeline") or step.get("pipeline_reference")
        arguments = block["arguments"]
        realize_args(arguments)
        references[step["name"]] = arguments["references"]
    return references


class TestOptionalVoices:
    def test_the_voices_default_to_null(self, definition):
        variables = definition["variables"]
        assert variables["character_a_voice"] is None
        assert variables["character_b_voice"] is None

    def test_no_voice_named_leaves_only_the_portraits(self, definition):
        for name, references in shot_references(
            definition, dict(definition["variables"])
        ).items():
            assert len(references) == len(set(SPEAKERS[name])), name
            assert all(
                reference["from_previous_result"].startswith("draw_character")
                for reference in references
            ), name

    def test_a_named_voice_is_referenced_in_the_shots_it_speaks_in(
        self, definition, tmp_path
    ):
        from tests.test_media_info import write_wav

        voice = tmp_path / "priya.wav"
        write_wav(voice, seconds=1.0)
        variables = dict(definition["variables"])
        variables["character_a_voice"] = str(voice)

        for name, references in shot_references(definition, variables).items():
            built = [r for r in references if not isinstance(r, dict)]
            assert len(built) == (1 if "a" in SPEAKERS[name] else 0), name

    def test_the_variable_names_are_roles_rather_than_a_cast(self, definition):
        """Every run carried howie_portrait_prompt and shot_3_howie_incredulous
        through its arguments, manifest and export whatever the cast was."""
        names = " ".join(definition["variables"]) + " ".join(
            step["name"] for step in definition["steps"]
        )
        assert "howie" not in names.lower()
        assert "pat_" not in names.lower()
        assert "character_a_portrait_prompt" in definition["variables"]
        assert "shot_3_react" in definition["variables"]
