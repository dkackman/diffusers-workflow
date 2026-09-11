import copy

import pytest

from dw.for_each import (
    MAX_FOR_EACH_ENTRIES,
    ForEachError,
    expand_for_each,
    member_name,
)


def definition(*steps, **extra):
    return {"id": "test", "steps": list(steps), **extra}


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
