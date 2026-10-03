"""dw/lora_catalog.py: the entry schema, which bases a workflow loads, and
which entries fit them - exactly, never by family."""

import pytest

from dw.lora_catalog import (
    entry_errors,
    is_repo_id,
    matches,
    query_terms,
    ranked,
    rejection_reasons,
    workflow_bases,
)


def entry(**overrides):
    base = {
        "model_name": "prithivMLmods/Qwen-Image-2.1-Voxel-Style",
        "base_models": ["Qwen/Qwen-Image-2.1"],
        "description": "Blocky 3D look",
        "use_when": "voxel or blocky 3D requests",
        "status": "trial",
    }
    base.update(overrides)
    return base


class TestSchema:
    def test_a_minimal_trial_entry_is_valid(self):
        assert entry_errors(entry()) is None

    def test_proven_needs_evidence(self):
        assert entry_errors(entry(status="proven")) is not None
        assert (
            entry_errors(entry(status="proven", evidence=[{"issue": 585, "note": "best arm"}]))
            is None
        )

    def test_base_models_may_not_be_empty(self):
        assert entry_errors(entry(base_models=[])) is not None

    def test_an_unknown_status_is_refused(self):
        assert entry_errors(entry(status="recommended")) is not None

    def test_an_unknown_field_is_refused(self):
        assert entry_errors(entry(colour="blue")) is not None

    def test_the_partition_vocabulary_is_the_adapter_checks(self):
        assert entry_errors(entry(workflow="ref2va")) is None
        assert entry_errors(entry(workflow="i2v")) is not None


class TestRepoIds:
    @pytest.mark.parametrize("value", ["Qwen/Qwen-Image-2.1", "a/b.c_d-e"])
    def test_a_repo_id(self, value):
        assert is_repo_id(value)

    @pytest.mark.parametrize("value", ["noslash", "a/b/c", "../x", "a/..", None, 3, "variable:x"])
    def test_not_a_repo_id(self, value):
        assert not is_repo_id(value)


class TestWorkflowBases:
    def test_each_pipeline_step_names_its_base_and_partition(self):
        definition = {
            "steps": [
                {"pipeline": {"from_pretrained_arguments": {"model_name": "MiniMaxAI/MiniMax-H3", "workflow": "ref2va"}}},
                {"pipeline": {"from_pretrained_arguments": {"model_name": "Qwen/Qwen-Image-2.1"}}},
                {"task": {"command": "gather_images"}},
            ]
        }
        assert workflow_bases(definition) == [
            ("MiniMaxAI/MiniMax-H3", "ref2va"),
            ("Qwen/Qwen-Image-2.1", None),
        ]

    def test_a_variable_reference_takes_the_variables_default(self):
        definition = {
            "variables": {"model": "Qwen/Qwen-Image-2.1"},
            "steps": [{"pipeline": {"from_pretrained_arguments": {"model_name": "variable:model"}}}],
        }
        assert workflow_bases(definition) == [("Qwen/Qwen-Image-2.1", None)]

    def test_a_variable_with_no_repo_default_is_skipped(self):
        definition = {
            "variables": {"model": None},
            "steps": [{"pipeline": {"from_pretrained_arguments": {"model_name": "variable:model"}}}],
        }
        assert workflow_bases(definition) == []

    def test_a_repeated_base_is_listed_once(self):
        step = {"pipeline": {"from_pretrained_arguments": {"model_name": "a/b"}}}
        assert workflow_bases({"steps": [step, step]}) == [("a/b", None)]


class TestMatching:
    def test_the_base_must_match_exactly(self):
        assert matches(entry(), [("Qwen/Qwen-Image-2.1", None)])
        assert not matches(entry(), [("Qwen/Qwen-Image-2.1-2509", None)])
        assert not matches(entry(), [("qwen/qwen-image-2.1", None)])

    def test_a_partitioned_entry_fits_only_its_partition(self):
        h3 = entry(base_models=["MiniMaxAI/MiniMax-H3"], workflow="t2va")
        assert matches(h3, [("MiniMaxAI/MiniMax-H3", "t2va")])
        assert not matches(h3, [("MiniMaxAI/MiniMax-H3", "ref2va")])

    def test_a_bare_repo_lists_every_partition(self):
        h3 = entry(base_models=["MiniMaxAI/MiniMax-H3"], workflow="t2va")
        assert matches(h3, [("MiniMaxAI/MiniMax-H3", None)])


class TestRanking:
    def test_stop_words_and_short_words_are_dropped(self):
        assert query_terms("a Voxel style LoRA, isometric!") == ["voxel", "isometric"]
        assert query_terms("style") == []
        assert query_terms(None) == []

    def test_query_hits_rank_first_then_status_then_name(self):
        entries = {
            "b": entry(use_when="anything", status="proven", evidence=[{"note": "x"}]),
            "a": entry(use_when="voxel art"),
            "c": entry(use_when="voxel art", status="proven", evidence=[{"note": "x"}]),
        }
        assert [row["name"] for row in ranked(entries, ["voxel"])] == ["c", "a", "b"]

    def test_tags_count_toward_the_score(self):
        entries = {"x": entry(use_when="", tags=["voxel"]), "y": entry(use_when="")}
        assert ranked(entries, ["voxel"])[0]["name"] == "x"

    def test_rejection_reasons_come_from_the_first_evidence_note(self):
        entries = {
            "fast": entry(model_name="drozbay/FastH3", status="rejected", evidence=[{"note": ".diff keys"}]),
            "ok": entry(),
        }
        assert rejection_reasons(entries) == {"drozbay/FastH3": ".diff keys"}
