import copy
import json
import os
import tempfile

import pytest

from dw.for_each import (
    MAX_FOR_EACH_ENTRIES,
    ForEachError,
    expand_for_each,
    member_name,
)


def definition(*steps, **extra):
    return {"id": "test", "steps": list(steps), **extra}


TEMPLATES = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "workflows", "templates", "minimax")
)


def load_template(name):
    with open(os.path.join(TEMPLATES, name)) as f:
        return json.load(f)


def load_workflow(name, output_dir=None):
    from dw.workflow import workflow_from_file

    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    return workflow_from_file(os.path.join(TEMPLATES, name), output_dir)


def steps_by_name(definition):
    return {s["name"]: s for s in definition["steps"]}


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


class TestSourceIndices:
    """The expansion records where each step came from in the source file.

    A parallel list rather than a key on the step: step_data is what the
    step cache keys on and what the schema validates, so nothing new may
    appear in it.
    """

    def test_every_expanded_step_records_its_source_index(self):
        source_indices = []
        expanded = expand_for_each(
            definition(
                {"name": "draw", "task": {}},
                {"name": "shot", "for_each": ["a", "b", "c"], "task": {}},
                {"name": "edit", "task": {}},
            ),
            source_indices,
        )
        assert [s["name"] for s in expanded["steps"]] == [
            "draw",
            "shot@0",
            "shot@1",
            "shot@2",
            "edit",
        ]
        assert source_indices == [0, 1, 1, 1, 2]

    def test_the_list_is_optional(self):
        expanded = expand_for_each(
            definition({"name": "shot", "for_each": ["a"], "task": {}})
        )
        assert [s["name"] for s in expanded["steps"]] == ["shot@0"]


class TestLeafCopying:
    """A leaf is copied only where the copy is needed - inside a member.

    expand_for_each runs on every run of every workflow, after realize_args
    has turned 'asset:' and '*_image' arguments into loaded PIL images and
    decoded frame lists. Copying every leaf of every step would multiply
    that media, and a leaf that cannot be copied at all would fail a run
    that has always worked.
    """

    class Uncopyable:
        def __deepcopy__(self, memo):
            raise TypeError("cannot copy this")

    def test_a_leaf_outside_a_member_is_passed_through_by_identity(self):
        leaf = self.Uncopyable()
        expanded = expand_for_each(
            definition({"name": "plain", "task": {"arguments": {"model": leaf}}})
        )
        assert expanded["steps"][0]["task"]["arguments"]["model"] is leaf

    def test_a_leaf_inside_a_member_that_cannot_be_copied_is_the_object_itself(self):
        # The step cache makes the same choice for a realized argument it
        # cannot deep-copy: the run goes on, with the object as it is
        leaf = self.Uncopyable()
        expanded = expand_for_each(
            definition(
                {
                    "name": "shot",
                    "for_each": ["a", "b"],
                    "task": {"arguments": {"model": leaf}},
                }
            )
        )
        assert [s["task"]["arguments"]["model"] for s in expanded["steps"]] == [
            leaf,
            leaf,
        ]


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
                    "workflow": {
                        "path": "builtin:x.json",
                        "arguments": {"clips": "gather:shot"},
                    },
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
                    "task": {
                        "arguments": {"r": [{"from_previous_result": "variable:x"}]}
                    },
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
    """music-video's slices and shots are two for_each groups over one
    'shots' list, paired by entry name: shot@closeup reads slice@closeup."""

    KEYS = ["wide_open", "closeup", "room", "finale"]

    def expanded(self):
        return load_workflow("music-video.json").expanded_definition()

    def test_the_template_validates_as_it_will_run(self):
        assert load_workflow("music-video.json").validation_errors() == []

    def test_one_slice_and_one_shot_per_entry_in_list_order(self):
        names = [s["name"] for s in self.expanded()["steps"]]
        assert names == (
            ["draw_singer", "write_song"]
            + [f"slice@{k}" for k in self.KEYS]
            + ["soundtrack"]
            + [f"shot@{k}" for k in self.KEYS]
            + ["edit", "music_video"]
        )

    def test_each_slice_starts_where_its_entry_says(self):
        got = steps_by_name(self.expanded())
        starts = [
            got[f"slice@{k}"]["task"]["arguments"]["start_frame"] for k in self.KEYS
        ]
        assert starts == [0, 124, 248, 372]

    def test_each_shot_reads_its_own_slice_and_the_one_portrait(self):
        got = steps_by_name(self.expanded())
        for key in self.KEYS:
            references = got[f"shot@{key}"]["pipeline"]["arguments"]["references"]
            assert [r["from_previous_result"] for r in references] == [
                "draw_singer",
                f"slice@{key}",
            ]

    def test_each_shot_carries_its_entry_s_prompt(self):
        template = load_template("music-video.json")
        got = steps_by_name(self.expanded())
        for entry in template["variables"]["shots"]:
            prompt = got[f"shot@{entry['name']}"]["pipeline"]["arguments"]["prompt"]
            assert prompt == entry["prompt"]
            assert prompt.startswith("subject_definitions:")

    def test_every_shot_is_the_same_pipeline(self):
        """Full pipeline blocks rather than pipeline_reference: the identity
        cache reuses the loaded model, so this costs no reload."""
        from dw.workflow import pipeline_cache_key

        got = steps_by_name(self.expanded())
        keys = {pipeline_cache_key(got[f"shot@{k}"]["pipeline"]) for k in self.KEYS}
        assert len(keys) == 1

    def test_the_edit_gathers_the_shots_in_order(self):
        got = steps_by_name(self.expanded())
        assert got["edit"]["task"]["arguments"]["videos"] == [
            f"previous_result:shot@{k}" for k in self.KEYS
        ]

    def test_the_template_keeps_the_list(self):
        """The template on disk keeps 'for_each' and the 'shots' variable
        rather than expanded members - the realized workflow.json a run
        writes is built from this same definition, but that is not what
        this test reads."""
        template = load_template("music-video.json")
        assert "shots" in template["variables"]
        assert [s["name"] for s in template["steps"] if "for_each" in s] == [
            "slice",
            "shot",
        ]

    def test_each_shot_keeps_the_generation_settings(self):
        got = steps_by_name(self.expanded())
        for key in self.KEYS:
            arguments = got[f"shot@{key}"]["pipeline"]["arguments"]
            assert arguments["num_frames"] == 124
            assert arguments["width"] == 960
            assert arguments["height"] == 544
            assert arguments["num_inference_steps"] == 9
            assert arguments["output"] == ["videos", "audio", "sampling_rate"]
            loras = got[f"shot@{key}"]["pipeline"]["loras"]
            assert len(loras) == 1
            assert loras[0]["model_name"] == "lightx2v/Minimax-h3-Turbo"


class TestDialogueShortTemplate:
    """dialogue-short's five shots are one for_each group whose entries
    carry everything that differs between shots: prompt, references and
    length."""

    KEYS = ["cold_open", "deflect", "react", "button", "tag"]

    def expanded(self):
        return load_workflow("dialogue-short.json").expanded_definition()

    def test_the_template_validates_as_it_will_run(self):
        assert load_workflow("dialogue-short.json").validation_errors() == []

    def test_one_shot_per_entry_between_the_cast_and_the_edit(self):
        names = [s["name"] for s in self.expanded()["steps"]]
        assert names == (
            ["draw_character_a", "draw_character_b"]
            + [f"shot@{k}" for k in self.KEYS]
            + ["episode"]
        )

    def test_each_shot_references_the_portraits_its_entry_lists(self):
        got = steps_by_name(self.expanded())
        portraits = {
            key: [
                r["from_previous_result"]
                for r in got[f"shot@{key}"]["pipeline"]["arguments"]["references"]
                if "from_previous_result" in r
            ]
            for key in self.KEYS
        }
        assert portraits == {
            "cold_open": ["draw_character_a", "draw_character_b"],
            "deflect": ["draw_character_b"],
            "react": ["draw_character_a"],
            "button": ["draw_character_b"],
            "tag": ["draw_character_a", "draw_character_b"],
        }

    def test_the_reference_types_are_resolved_inside_the_entries(self):
        """An entry's "variable:subject_reference_type" is the dotted type
        name by the time the member exists - never the literal reference."""
        got = steps_by_name(self.expanded())
        for key in self.KEYS:
            for r in got[f"shot@{key}"]["pipeline"]["arguments"]["references"]:
                assert r["reference_type"].startswith("diffusers.modular_pipelines")

    def test_the_tag_runs_longer(self):
        got = steps_by_name(self.expanded())
        frames = [
            got[f"shot@{k}"]["pipeline"]["arguments"]["num_frames"] for k in self.KEYS
        ]
        assert frames == [124, 124, 124, 124, 141]

    def test_every_shot_is_the_same_pipeline(self):
        from dw.workflow import pipeline_cache_key

        got = steps_by_name(self.expanded())
        assert (
            len({pipeline_cache_key(got[f"shot@{k}"]["pipeline"]) for k in self.KEYS})
            == 1
        )

    def test_the_episode_gathers_the_shots_in_order(self):
        got = steps_by_name(self.expanded())
        assert got["episode"]["task"]["arguments"]["videos"] == [
            f"previous_result:shot@{k}" for k in self.KEYS
        ]

    def test_each_shot_keeps_the_generation_settings(self):
        got = steps_by_name(self.expanded())
        for key in self.KEYS:
            arguments = got[f"shot@{key}"]["pipeline"]["arguments"]
            assert arguments["width"] == 960
            assert arguments["height"] == 544
            assert arguments["num_inference_steps"] == 9
            assert arguments["output"] == ["videos", "audio", "sampling_rate"]
            loras = got[f"shot@{key}"]["pipeline"]["loras"]
            assert len(loras) == 1
            assert loras[0]["model_name"] == "lightx2v/Minimax-h3-Turbo"


class TestRunTimeRealizationOrder:
    """Workflow.run resolves in a fixed order: realize_constants ->
    set_variables -> resolve_variable_values -> realize_args(variables,
    base_dir) -> replace_variables -> expand_for_each. realize_args walks
    into the 'shots' variable and loads every 'reference_type' there, so
    resolve_variable_values must already have turned
    'variable:subject_reference_type' into a dotted name before realize_args
    runs - reordering those two would fail exactly what this test checks."""

    KEYS = {
        "cold_open": 2,
        "deflect": 1,
        "react": 1,
        "button": 1,
        "tag": 2,
    }

    def test_the_run_s_own_resolution_order_produces_loaded_types(self):
        from dw.arguments import realize_args, realize_constants
        from dw.for_each import expand_for_each
        from dw.variables import replace_variables, resolve_variable_values

        definition = load_template("dialogue-short.json")
        variables = definition["variables"]
        realize_constants(variables)
        variables = resolve_variable_values(variables)
        realize_args(variables, TEMPLATES)
        expanded = expand_for_each(replace_variables(definition, variables))

        got = steps_by_name(expanded)
        for key, count in self.KEYS.items():
            references = got[f"shot@{key}"]["pipeline"]["arguments"]["references"]
            assert len(references) == count, key
            for reference in references:
                assert isinstance(reference, dict)
                assert isinstance(reference["reference_type"], type)
                assert "from_previous_result" in reference
                assert "from_file" not in reference
