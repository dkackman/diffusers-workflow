import copy
import json
import os

import pytest

from dw.for_each import (
    MAX_FOR_EACH_ENTRIES,
    ForEachError,
    expand_for_each,
    member_name,
)


def definition(*steps, **extra):
    return {"id": "test", "steps": list(steps), **extra}


TEMPLATES = os.path.join(
    os.path.dirname(__file__), "..", "workflows", "templates", "minimax"
)


def load_template(name):
    with open(os.path.join(TEMPLATES, name)) as f:
        return json.load(f)


def steps_by_name(definition):
    return {s["name"]: s for s in definition["steps"]}


def without_pipeline_reference(step, pipeline_step):
    """A hand-written 'pipeline_reference' shot as the full pipeline block
    the expansion produces: the reference's arguments over the referenced
    step's pipeline."""
    rebuilt = {k: v for k, v in step.items() if k != "pipeline_reference"}
    pipeline = copy.deepcopy(pipeline_step["pipeline"])
    pipeline["arguments"] = step["pipeline_reference"]["arguments"]
    rebuilt["pipeline"] = pipeline
    return rebuilt


class TestNaming:
    def test_member_name_joins_with_at(self):
        assert member_name("shot", "wide_open") == "shot@wide_open"

    def test_an_object_entry_is_named_by_its_name(self):
        expanded = expand_for_each(
            definition(
                {
                    "name": "shot",
                    "for_each": [{"name": "wide_open"}, {"name": "closeup"}],
                    "task": {"command": "x", "arguments": {}},
                }
            )
        )
        assert [s["name"] for s in expanded["steps"]] == [
            "shot@wide_open",
            "shot@closeup",
        ]

    def test_a_nameless_entry_is_named_by_its_index(self):
        expanded = expand_for_each(
            definition(
                {"name": "shot", "for_each": ["a", "b"], "task": {"arguments": {}}}
            )
        )
        assert [s["name"] for s in expanded["steps"]] == ["shot@0", "shot@1"]

    def test_for_each_is_dropped_from_the_members(self):
        expanded = expand_for_each(
            definition({"name": "shot", "for_each": ["a"], "task": {}})
        )
        assert "for_each" not in expanded["steps"][0]

    def test_the_input_is_not_mutated(self):
        original = definition({"name": "shot", "for_each": ["a"], "task": {}})
        before = copy.deepcopy(original)
        expand_for_each(original)
        assert original == before

    def test_a_workflow_without_for_each_comes_back_equal(self):
        original = definition({"name": "plain", "task": {"arguments": {"a": 1}}})
        assert expand_for_each(original) == original

    def test_a_hand_written_at_sign_is_an_error(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(definition({"name": "shot@0", "task": {}}))
        assert e.value.path == "steps[0].name"
        assert "@" in str(e.value)

    def test_a_duplicate_entry_name_is_an_error(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition(
                    {
                        "name": "shot",
                        "for_each": [{"name": "closeup"}, {"name": "closeup"}],
                        "task": {},
                    }
                )
            )
        assert e.value.path == "steps[0].for_each[1].name"
        assert "closeup" in str(e.value)

    def test_an_invalid_entry_name_is_an_error(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition(
                    {"name": "shot", "for_each": [{"name": "wide open"}], "task": {}}
                )
            )
        assert e.value.path == "steps[0].for_each[0].name"

    def test_a_non_list_for_each_is_an_error(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(definition({"name": "shot", "for_each": 4, "task": {}}))
        assert e.value.path == "steps[0].for_each"
        assert "list" in str(e.value)

    def test_an_unsubstituted_variable_is_an_error_that_says_so(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition({"name": "shot", "for_each": "variable:shots", "task": {}})
            )
        assert "variable:shots" in str(e.value)

    def test_an_empty_list_expands_to_no_steps(self):
        expanded = expand_for_each(
            definition(
                {"name": "shot", "for_each": [], "task": {}},
                {"name": "after", "task": {}},
            )
        )
        assert [s["name"] for s in expanded["steps"]] == ["after"]

    def test_the_ceiling_is_enforced(self):
        entries = [{"name": f"s{i}"} for i in range(MAX_FOR_EACH_ENTRIES + 1)]
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition({"name": "shot", "for_each": entries, "task": {}})
            )
        assert str(MAX_FOR_EACH_ENTRIES) in str(e.value)

    def test_a_list_at_the_ceiling_is_fine(self):
        entries = [{"name": f"s{i}"} for i in range(MAX_FOR_EACH_ENTRIES)]
        expanded = expand_for_each(
            definition({"name": "shot", "for_each": entries, "task": {}})
        )
        assert len(expanded["steps"]) == MAX_FOR_EACH_ENTRIES


class TestItemSubstitution:
    def test_a_field_is_substituted(self):
        expanded = expand_for_each(
            definition(
                {
                    "name": "shot",
                    "for_each": [{"name": "a", "prompt": "the band walks on"}],
                    "pipeline": {"arguments": {"prompt": "item:prompt"}},
                }
            )
        )
        assert expanded["steps"][0]["pipeline"]["arguments"]["prompt"] == (
            "the band walks on"
        )

    def test_a_bare_item_is_the_whole_entry(self):
        expanded = expand_for_each(
            definition(
                {
                    "name": "shot",
                    "for_each": ["first prompt", "second prompt"],
                    "pipeline": {"arguments": {"prompt": "item:"}},
                }
            )
        )
        prompts = [s["pipeline"]["arguments"]["prompt"] for s in expanded["steps"]]
        assert prompts == ["first prompt", "second prompt"]

    def test_a_structured_field_is_spliced_whole(self):
        references = [
            {"reference_type": "T", "from_previous_result": "draw_a"},
            {"reference_type": "T", "from_previous_result": "draw_b"},
        ]
        expanded = expand_for_each(
            definition(
                {"name": "draw_a", "task": {}},
                {"name": "draw_b", "task": {}},
                {
                    "name": "shot",
                    "for_each": [{"name": "open", "references": references}],
                    "pipeline": {"arguments": {"references": "item:references"}},
                },
            )
        )
        assert expanded["steps"][2]["pipeline"]["arguments"]["references"] == references

    def test_a_number_keeps_its_type(self):
        expanded = expand_for_each(
            definition(
                {
                    "name": "slice",
                    "for_each": [{"name": "a", "start_frame": 124}],
                    "task": {"arguments": {"start_frame": "item:start_frame"}},
                }
            )
        )
        assert expanded["steps"][0]["task"]["arguments"]["start_frame"] == 124

    def test_item_is_substituted_at_any_depth(self):
        expanded = expand_for_each(
            definition(
                {
                    "name": "shot",
                    "for_each": [{"name": "a", "voice": "asset:a.wav"}],
                    "pipeline": {
                        "arguments": {"references": [{"from_file": "item:voice"}]}
                    },
                }
            )
        )
        assert expanded["steps"][0]["pipeline"]["arguments"]["references"] == [
            {"from_file": "asset:a.wav"}
        ]

    def test_a_missing_field_is_an_error_at_its_path(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition(
                    {
                        "name": "shot",
                        "for_each": [{"name": "a"}],
                        "pipeline": {"arguments": {"prompt": "item:prompt"}},
                    }
                )
            )
        assert e.value.path == "steps[0].pipeline.arguments.prompt"
        assert "prompt" in str(e.value) and "a" in str(e.value)

    def test_a_field_of_a_string_entry_is_an_error(self):
        with pytest.raises(ForEachError):
            expand_for_each(
                definition(
                    {
                        "name": "shot",
                        "for_each": ["just a prompt"],
                        "pipeline": {"arguments": {"prompt": "item:prompt"}},
                    }
                )
            )

    def test_item_outside_a_for_each_step_is_an_error(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition({"name": "plain", "task": {"arguments": {"p": "item:x"}}})
            )
        assert e.value.path == "steps[0].task.arguments.p"

    def test_members_do_not_share_mutable_values(self):
        expanded = expand_for_each(
            definition(
                {
                    "name": "shot",
                    "for_each": [{"name": "a"}, {"name": "b"}],
                    "pipeline": {"arguments": {"references": [{"k": 1}]}},
                }
            )
        )
        first, second = expanded["steps"]
        first["pipeline"]["arguments"]["references"][0]["k"] = 2
        assert second["pipeline"]["arguments"]["references"][0]["k"] == 1


class TestRelease:
    def test_release_flags_survive_on_the_last_member_only(self):
        expanded = expand_for_each(
            definition(
                {
                    "name": "shot",
                    "for_each": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
                    "release_pipeline": True,
                    "release_models": True,
                    "pipeline": {},
                }
            )
        )
        flags = [
            (s.get("release_pipeline"), s.get("release_models"))
            for s in expanded["steps"]
        ]
        assert flags == [(None, None), (None, None), (True, True)]


class TestGather:
    def group(self, *extra):
        return definition(
            {
                "name": "shot",
                "for_each": [{"name": "open"}, {"name": "close"}],
                "pipeline": {"arguments": {}},
            },
            *extra,
        )

    def test_a_scalar_gather_becomes_the_member_list(self):
        expanded = expand_for_each(
            self.group(
                {"name": "edit", "task": {"arguments": {"videos": "gather:shot"}}}
            )
        )
        assert expanded["steps"][-1]["task"]["arguments"]["videos"] == [
            "previous_result:shot@open",
            "previous_result:shot@close",
        ]

    def test_a_gather_inside_a_list_splices(self):
        expanded = expand_for_each(
            self.group(
                {"name": "intro", "task": {}},
                {
                    "name": "edit",
                    "task": {
                        "arguments": {
                            "videos": ["previous_result:intro", "gather:shot"]
                        }
                    },
                },
            )
        )
        assert expanded["steps"][-1]["task"]["arguments"]["videos"] == [
            "previous_result:intro",
            "previous_result:shot@open",
            "previous_result:shot@close",
        ]

    def test_gather_of_an_empty_group_is_an_empty_list(self):
        expanded = expand_for_each(
            definition(
                {"name": "shot", "for_each": [], "task": {}},
                {"name": "edit", "task": {"arguments": {"videos": "gather:shot"}}},
            )
        )
        assert expanded["steps"][-1]["task"]["arguments"]["videos"] == []

    def test_gather_of_a_plain_step_is_an_error(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition(
                    {"name": "one", "task": {}},
                    {"name": "edit", "task": {"arguments": {"videos": "gather:one"}}},
                )
            )
        assert e.value.path == "steps[1].task.arguments.videos"
        assert "no earlier for_each step" in str(e.value)

    def test_gather_of_a_later_group_is_an_error(self):
        with pytest.raises(ForEachError):
            expand_for_each(
                definition(
                    {"name": "edit", "task": {"arguments": {"videos": "gather:shot"}}},
                    {"name": "shot", "for_each": ["a"], "task": {}},
                )
            )

    def test_gather_in_a_sub_workflow_argument_map(self):
        expanded = expand_for_each(
            self.group(
                {
                    "name": "score",
                    "workflow": {"path": "builtin:x.json", "arguments": {"clips": "gather:shot"}},
                }
            )
        )
        assert expanded["steps"][-1]["workflow"]["arguments"]["clips"] == [
            "previous_result:shot@open",
            "previous_result:shot@close",
        ]


class TestGroupReferences:
    def test_previous_result_naming_a_group_says_to_gather(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition(
                    {"name": "shot", "for_each": ["a"], "task": {}},
                    {
                        "name": "edit",
                        "task": {"arguments": {"video": "previous_result:shot"}},
                    },
                )
            )
        assert e.value.path == "steps[1].task.arguments.video"
        assert "gather:shot" in str(e.value)

    def test_from_previous_result_naming_a_group_says_to_gather(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition(
                    {"name": "shot", "for_each": ["a"], "task": {}},
                    {
                        "name": "edit",
                        "task": {
                            "arguments": {"refs": [{"from_previous_result": "shot"}]}
                        },
                    },
                )
            )
        assert e.value.path == "steps[1].task.arguments.refs[0].from_previous_result"

    def test_a_property_reference_to_a_group_is_also_refused(self):
        with pytest.raises(ForEachError):
            expand_for_each(
                definition(
                    {"name": "shot", "for_each": ["a"], "task": {}},
                    {
                        "name": "edit",
                        "task": {"arguments": {"v": "previous_result:shot.frames"}},
                    },
                )
            )

    def test_a_reference_to_an_ordinary_step_is_untouched(self):
        expanded = expand_for_each(
            definition(
                {"name": "draw", "task": {}},
                {
                    "name": "shot",
                    "for_each": ["a"],
                    "pipeline": {
                        "arguments": {
                            "references": [{"from_previous_result": "draw"}],
                            "still": "previous_result:draw.image",
                        }
                    },
                },
            )
        )
        arguments = expanded["steps"][1]["pipeline"]["arguments"]
        assert arguments["references"] == [{"from_previous_result": "draw"}]
        assert arguments["still"] == "previous_result:draw.image"

    def test_a_variable_spelled_from_previous_result_is_untouched(self):
        expanded = expand_for_each(
            definition(
                {
                    "name": "edit",
                    "task": {"arguments": {"r": [{"from_previous_result": "variable:x"}]}},
                }
            )
        )
        assert expanded["steps"][0]["task"]["arguments"]["r"] == [
            {"from_previous_result": "variable:x"}
        ]


class TestSameKeySiblings:
    def shots(self):
        return [
            {"name": "open", "start_frame": 0},
            {"name": "close", "start_frame": 124},
        ]

    def test_a_sibling_over_the_same_list_resolves_to_the_same_key(self):
        shots = self.shots()
        expanded = expand_for_each(
            definition(
                {
                    "name": "slice",
                    "for_each": shots,
                    "task": {"arguments": {"start_frame": "item:start_frame"}},
                },
                {
                    "name": "shot",
                    "for_each": shots,
                    "pipeline": {
                        "arguments": {
                            "references": [{"from_previous_result": "slice"}],
                            "audio": "previous_result:slice.audio",
                        }
                    },
                },
            )
        )
        names = [s["name"] for s in expanded["steps"]]
        assert names == ["slice@open", "slice@close", "shot@open", "shot@close"]
        close = expanded["steps"][3]["pipeline"]["arguments"]
        assert close["references"] == [{"from_previous_result": "slice@close"}]
        assert close["audio"] == "previous_result:slice@close.audio"

    def test_the_same_list_means_equal_not_identical(self):
        expanded = expand_for_each(
            definition(
                {"name": "slice", "for_each": self.shots(), "task": {}},
                {
                    "name": "shot",
                    "for_each": self.shots(),
                    "task": {"arguments": {"a": "previous_result:slice"}},
                },
            )
        )
        assert expanded["steps"][2]["task"]["arguments"]["a"] == (
            "previous_result:slice@open"
        )

    def test_a_sibling_over_a_different_list_is_an_error(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition(
                    {"name": "slice", "for_each": ["a", "b"], "task": {}},
                    {
                        "name": "shot",
                        "for_each": ["a"],
                        "task": {"arguments": {"x": "previous_result:slice"}},
                    },
                )
            )
        assert "different list" in str(e.value)

    def test_a_step_referencing_its_own_group_is_an_error(self):
        with pytest.raises(ForEachError) as e:
            expand_for_each(
                definition(
                    {
                        "name": "shot",
                        "for_each": ["a", "b"],
                        "task": {"arguments": {"x": "previous_result:shot"}},
                    }
                )
            )
        assert "its own" in str(e.value)

    def test_a_gather_inside_a_member_still_gathers(self):
        expanded = expand_for_each(
            definition(
                {"name": "slice", "for_each": ["a", "b"], "task": {}},
                {
                    "name": "shot",
                    "for_each": ["x"],
                    "task": {"arguments": {"all": "gather:slice"}},
                },
            )
        )
        assert expanded["steps"][2]["task"]["arguments"]["all"] == [
            "previous_result:slice@0",
            "previous_result:slice@1",
        ]


class TestMusicVideoTemplate:
    """music-video's four slices and four shots, written as two for_each
    groups over one 'shots' list, expand to the steps the template holds
    by hand today."""

    def test_the_hand_written_shots_are_what_the_list_expands_to(self):
        template = load_template("music-video.json")
        today = steps_by_name(template)
        shots = [
            {"name": "wide_open", "prompt": "variable:shot_1_wide_open", "start_frame": 0},
            {"name": "closeup", "prompt": "variable:shot_2_closeup", "start_frame": 124},
            {"name": "room", "prompt": "variable:shot_3_room", "start_frame": 248},
            {"name": "finale", "prompt": "variable:shot_4_finale", "start_frame": 372},
        ]
        slice_template = copy.deepcopy(today["slice_1"])
        slice_template["name"] = "slice"
        slice_template["for_each"] = shots
        slice_template["task"]["arguments"]["start_frame"] = "item:start_frame"

        shot_template = copy.deepcopy(today["shot_1_wide_open"])
        shot_template["name"] = "shot"
        shot_template["for_each"] = shots
        shot_template["pipeline"]["arguments"]["prompt"] = "item:prompt"
        for reference in shot_template["pipeline"]["arguments"]["references"]:
            if reference.get("from_previous_result") == "slice_1":
                reference["from_previous_result"] = "slice"

        edit = copy.deepcopy(today["edit"])
        edit["task"]["arguments"]["videos"] = "gather:shot"

        expanded = expand_for_each(
            definition(
                today["draw_singer"], today["write_song"], slice_template,
                today["soundtrack"], shot_template, edit, today["music_video"],
            )
        )
        got = steps_by_name(expanded)

        # Each expanded slice is today's slice with the new name
        for key, old in zip(["wide_open", "closeup", "room", "finale"], range(1, 5)):
            expected = copy.deepcopy(today[f"slice_{old}"])
            expected["name"] = f"slice@{key}"
            assert got[f"slice@{key}"] == expected

        # Each expanded shot is today's shot (as a full pipeline block) with
        # the new name and its slice renamed
        hand_written = ["shot_1_wide_open", "shot_2_closeup", "shot_3_room", "shot_4_finale"]
        for key, old, index in zip(["wide_open", "closeup", "room", "finale"], hand_written, range(1, 5)):
            step = today[old]
            if "pipeline_reference" in step:
                step = without_pipeline_reference(step, today["shot_1_wide_open"])
            expected = copy.deepcopy(step)
            expected["name"] = f"shot@{key}"
            for reference in expected["pipeline"]["arguments"]["references"]:
                if reference.get("from_previous_result") == f"slice_{index}":
                    reference["from_previous_result"] = f"slice@{key}"
            assert got[f"shot@{key}"] == expected

        assert got["edit"]["task"]["arguments"]["videos"] == [
            f"previous_result:shot@{k}" for k in ["wide_open", "closeup", "room", "finale"]
        ]


class TestDialogueShortTemplate:
    """dialogue-short's five shots, whose reference lists and frame counts
    differ by shot, written as one for_each group whose entries carry every
    argument that differs between shots."""

    def test_the_hand_written_shots_are_what_the_list_expands_to(self):
        template = load_template("dialogue-short.json")
        today = steps_by_name(template)
        hand_written = [
            ("cold_open", "shot_1_cold_open"),
            ("deflect", "shot_2_deflect"),
            ("react", "shot_3_react"),
            ("button", "shot_4_button"),
            ("tag", "shot_5_tag"),
        ]
        first = today["shot_1_cold_open"]
        full = {
            old: (step if "pipeline_reference" not in step else without_pipeline_reference(step, first))
            for old, step in today.items()
            if old.startswith("shot_")
        }

        shots = []
        for key, old in hand_written:
            arguments = full[old]["pipeline"]["arguments"]
            shots.append(
                {
                    "name": key,
                    "prompt": arguments["prompt"],
                    "references": arguments["references"],
                    "num_frames": arguments["num_frames"],
                }
            )

        shot_template = copy.deepcopy(full["shot_1_cold_open"])
        shot_template["name"] = "shot"
        shot_template["for_each"] = shots
        shot_template["pipeline"]["arguments"]["prompt"] = "item:prompt"
        shot_template["pipeline"]["arguments"]["references"] = "item:references"
        shot_template["pipeline"]["arguments"]["num_frames"] = "item:num_frames"

        expanded = expand_for_each(
            definition(today["draw_character_a"], today["draw_character_b"], shot_template)
        )
        got = steps_by_name(expanded)
        for key, old in hand_written:
            expected = copy.deepcopy(full[old])
            expected["name"] = f"shot@{key}"
            assert got[f"shot@{key}"] == expected
