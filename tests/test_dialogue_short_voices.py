"""The dialogue-short template's optional voice references.

A recurring cast's voice was carried only as a description repeated verbatim
in every shot - about thirty hand-copied strings across three episodes, with
nothing checking that they matched (2026-09-11). `character_a_voice` /
`character_b_voice` name a clip instead, and each shot appends an audio
reference for whoever speaks in it. Null is the default, and null means the
reference is not there at all - so the template still generates exactly what
it generated before the variables existed. The shots are entries of a list
now, and an entry's voice reference names the variable, so the same one
variable still sets the voice everywhere.
"""

import json
import os

import pytest

from dw.arguments import realize_args

TEMPLATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "workflows",
    "templates",
    "minimax",
    "dialogue-short.json",
)

# Which character speaks in which shot - each entry lists a voice
# reference per speaker, and this is the mapping it is asserted against
SPEAKERS = {
    "shot@cold_open": ("a", "b"),
    "shot@deflect": ("b",),
    "shot@react": ("a",),
    "shot@button": ("b",),
    "shot@tag": ("a", "b"),
}


@pytest.fixture
def definition():
    with open(TEMPLATE, encoding="utf-8") as file:
        return json.load(file)


def shot_references(definition, arguments):
    """Every shot member's references, as the engine realizes them for a
    run: the caller's arguments folded, entries resolved, the list expanded,
    then each member's arguments realized as its step would."""
    from dw.workflow import workflow_from_file

    expanded = workflow_from_file(TEMPLATE, ".").expanded_definition(arguments)
    references = {}
    for step in expanded["steps"]:
        if step["name"] not in SPEAKERS:
            continue
        arguments = step["pipeline"]["arguments"]
        realize_args(arguments)
        references[step["name"]] = arguments["references"]
    assert set(references) == set(SPEAKERS)
    return references


class TestOptionalVoices:
    def test_the_voices_default_to_null(self, definition):
        variables = definition["variables"]
        assert variables["character_a_voice"] is None
        assert variables["character_b_voice"] is None

    def test_every_entry_names_the_voice_variables_rather_than_a_file(self, definition):
        for entry in definition["variables"]["shots"]:
            voices = [r["from_file"] for r in entry["references"] if "from_file" in r]
            assert voices, entry["name"]
            assert all(v.startswith("variable:character_") for v in voices), entry[
                "name"
            ]

    def test_no_voice_named_leaves_only_the_portraits(self, definition):
        for name, references in shot_references(definition, {}).items():
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

        for name, references in shot_references(
            definition, {"character_a_voice": str(voice)}
        ).items():
            built = [r for r in references if not isinstance(r, dict)]
            assert len(built) == (1 if "a" in SPEAKERS[name] else 0), name

    def test_the_tag_runs_longer(self, definition):
        frames = {e["name"]: e["num_frames"] for e in definition["variables"]["shots"]}
        assert frames == {
            "cold_open": 124,
            "deflect": 124,
            "react": 124,
            "button": 124,
            "tag": 141,
        }

    def test_the_variable_names_are_roles_rather_than_a_cast(self, definition):
        """Every run carried howie_portrait_prompt and shot_3_howie_incredulous
        through its arguments, manifest and export whatever the cast was."""
        names = (
            " ".join(definition["variables"])
            + " ".join(step["name"] for step in definition["steps"])
            + " ".join(e["name"] for e in definition["variables"]["shots"])
        )
        assert "howie" not in names.lower()
        assert "pat_" not in names.lower()
        assert "character_a_portrait_prompt" in definition["variables"]
        assert [e["name"] for e in definition["variables"]["shots"]] == [
            "cold_open",
            "deflect",
            "react",
            "button",
            "tag",
        ]
