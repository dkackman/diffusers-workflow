"""vram_estimate refuses a shape against the device actually serving it.

Every catalog 'cost' entry is a CUDA card, so checking all of them refused a
64 GB Mac against a 24 GB RTX 3090 - and filtering to matching entries alone
would have switched the guard off on a Mac entirely, since none match.
"""

import copy
import tempfile
from unittest.mock import patch

import pytest

import dw.workflow
from dw.vram_estimate import apply_vram_estimate, reference_count, vram_estimate_errors
from dw.workflow import workflow_from_definition


def definition(extra_cost=()):
    # 768 * 512 * 121 voxels * 200 bytes = 8.86 GiB, + 20 base = 28.86 GiB
    return {
        "variables": {"width": 768, "height": 512, "num_frames": 121},
        "cost": [
            {"device": "cuda", "name": "RTX 3090", "vram_gb": 24, "minutes": 1.8},
            *extra_cost,
        ],
        "vram_estimate": {
            "voxel_variables": ["width", "height", "num_frames"],
            "base_gb": 20,
            "bytes_per_voxel": 200,
        },
    }


def test_no_device_checks_every_entry_as_before():
    errors = vram_estimate_errors(definition())
    assert len(errors) == 1 and "RTX 3090" in errors[0]["message"]


def test_a_cuda_box_checks_its_cuda_entries():
    errors = vram_estimate_errors(definition(), device_type="cuda", capacity_gb=24)
    assert len(errors) == 1 and "RTX 3090" in errors[0]["message"]


def test_a_mac_with_room_is_not_refused_by_a_cuda_card():
    assert vram_estimate_errors(definition(), device_type="mps", capacity_gb=62) == []


def test_a_mac_without_room_is_refused_against_its_own_capacity():
    errors = vram_estimate_errors(definition(), device_type="mps", capacity_gb=16)
    assert len(errors) == 1
    assert "mps" in errors[0]["message"]
    assert "RTX 3090" not in errors[0]["message"]


def test_a_mac_whose_capacity_is_unknown_keeps_the_conservative_check():
    errors = vram_estimate_errors(definition(), device_type="mps", capacity_gb=None)
    assert len(errors) == 1 and "RTX 3090" in errors[0]["message"]


def test_a_curated_mps_entry_wins_over_the_measured_capacity():
    mac = {"device": "mps", "name": "M5 Pro 64 GB", "vram_gb": 26, "minutes": 9}
    errors = vram_estimate_errors(definition([mac]), device_type="mps", capacity_gb=62)
    assert len(errors) == 1 and "M5 Pro 64 GB" in errors[0]["message"]


def test_an_indexed_cost_device_matches_its_backend():
    card = {"device": "cuda:1", "name": "second card", "vram_gb": 48, "minutes": 1}
    d = definition()
    d["cost"] = [card]
    assert vram_estimate_errors(d, device_type="cuda", capacity_gb=24) == []


def test_run_time_backstop_takes_the_device_too():
    d = definition()
    apply_vram_estimate(d, d["variables"], device_type="mps", capacity_gb=62)
    with pytest.raises(ValueError):
        apply_vram_estimate(d, d["variables"], device_type="mps", capacity_gb=16)


# --- Per-step projection over an expanded definition (#479) ---------------


def per_step_definition(steps, base_gb=0, gb_per_reference=None, ceiling=24):
    """A minimal expanded definition for exercising `_projections` directly:
    one voxel variable `n`, scaled so `n=1` costs exactly 1 GiB, which keeps
    the arithmetic in each test legible."""
    estimate = {
        "voxel_variables": ["n"],
        "base_gb": base_gb,
        "bytes_per_voxel": 1024**3,
    }
    if gb_per_reference is not None:
        estimate["gb_per_reference"] = gb_per_reference
    return {
        "cost": [{"device": "cuda", "name": "Card", "vram_gb": ceiling, "minutes": 1}],
        "vram_estimate": estimate,
        "steps": steps,
    }


def test_step_arguments_override_the_workflow_variables():
    steps = [{"name": "s1", "pipeline": {"arguments": {"n": 30}}}]
    d = per_step_definition(steps)
    d["variables"] = {"n": 5}
    errors = vram_estimate_errors(d)
    assert len(errors) == 1
    assert "n = 30" in errors[0]["message"]


def test_a_step_that_omits_the_value_falls_back_to_variables():
    steps = [{"name": "s1", "pipeline": {"arguments": {}}}]
    d = per_step_definition(steps)
    d["variables"] = {"n": 30}
    errors = vram_estimate_errors(d)
    assert len(errors) == 1
    assert "n = 30" in errors[0]["message"]


def test_a_task_step_is_not_projected():
    steps = [{"name": "t1", "task": {"command": "resample_audio", "arguments": {}}}]
    d = per_step_definition(steps)
    d["variables"] = {"n": 1000}
    assert vram_estimate_errors(d) == []


def test_a_non_numeric_value_is_skipped_not_erroring():
    steps = [{"name": "s1", "pipeline": {"arguments": {"n": "not-a-number"}}}]
    d = per_step_definition(steps, base_gb=100)
    assert vram_estimate_errors(d) == []


def test_reference_counting_skips_null_and_non_dict_entries():
    references = [
        {"from_file": "a"},
        {"from_file": None},
        None,
        "not-a-dict",
        {"from_previous_result": "step.x"},
    ]
    assert reference_count({"references": references}) == 2


def test_no_gb_per_reference_means_references_add_nothing_to_the_projection():
    steps = [
        {
            "name": "s1",
            "pipeline": {
                "arguments": {
                    "n": 1,
                    "references": [
                        {"from_file": "a"},
                        {"from_file": "b"},
                        {"from_file": "c"},
                    ],
                }
            },
        }
    ]
    d = per_step_definition(steps, base_gb=0, gb_per_reference=None, ceiling=0.5)
    errors = vram_estimate_errors(d)
    assert len(errors) == 1
    # 1 GB from `n` alone - the 3 references add nothing since gb_per_reference
    # is not declared, even though the message still names the count.
    assert "projects to 1.00 GB" in errors[0]["message"]


def test_only_the_largest_over_ceiling_step_is_reported_once_per_cost_entry():
    steps = [
        {"name": "s1", "pipeline": {"arguments": {"n": 30}}},
        {"name": "s2", "pipeline": {"arguments": {"n": 50}}},
    ]
    d = per_step_definition(steps, base_gb=0)
    d["cost"] = [
        {"device": "cuda", "name": "Card A", "vram_gb": 24, "minutes": 1},
        {"device": "cuda", "name": "Card B", "vram_gb": 40, "minutes": 1},
    ]
    errors = vram_estimate_errors(d)
    assert len(errors) == 2
    assert all("n = 50" in e["message"] for e in errors)
    assert not any("n = 30" in e["message"] for e in errors)


# --- Per-for_each-member projection through Workflow.validation_errors ----


def _reference(name):
    return {"from_file": f"asset:{name}.png"}


def shots_definition():
    """Two H3-shaped shots: 'a' stays under ceiling at every scenario below,
    'b' is the one whose num_frames/references/resolution combination is
    pushed over 24 GB by each test. Constants mirror reference-to-video.json.
    """
    return {
        "id": "shots-vram",
        "variables": {
            "width": 960,
            "height": 544,
            "shots": [
                {
                    "name": "a",
                    "num_frames": 124,
                    "references": [_reference("one"), _reference("two")],
                },
                {
                    "name": "b",
                    "num_frames": 209,
                    "references": [
                        _reference("three"),
                        _reference("four"),
                        _reference("five"),
                    ],
                },
            ],
        },
        "cost": [
            {"device": "cuda", "name": "RTX 3090", "vram_gb": 24, "minutes": 7.88}
        ],
        "vram_estimate": {
            "base_gb": 16.0,
            "bytes_per_voxel": 28.71,
            "gb_per_reference": 1.0,
            "voxel_variables": ["width", "height", "num_frames"],
        },
        "steps": [
            {
                "name": "shot",
                "for_each": "variable:shots",
                "pipeline": {
                    "configuration": {"component_type": "ModularPipeline"},
                    "from_pretrained_arguments": {
                        "model_name": "MiniMaxAI/MiniMax-H3",
                        "workflow": "ref2va",
                    },
                    "arguments": {
                        "prompt": "x",
                        "width": "variable:width",
                        "height": "variable:height",
                        "num_frames": "item:num_frames",
                        "references": "item:references",
                    },
                },
                "result": {"content_type": "video/mp4"},
            }
        ],
    }


def _vram_errors(errors):
    return [e for e in errors if "GB VRAM" in e["message"]]


def _build(definition):
    return workflow_from_definition(definition, tempfile.mkdtemp())


def test_for_each_defaults_have_no_vram_errors():
    with (
        patch.object(dw.workflow, "get_device_type", return_value="cuda"),
        patch.object(dw.workflow, "device_capacity_gb", return_value=24.0),
    ):
        workflow = _build(shots_definition())
        assert _vram_errors(workflow.validation_errors()) == []


def test_for_each_member_over_ceiling_names_variables_path_when_shots_is_a_default():
    with (
        patch.object(dw.workflow, "get_device_type", return_value="cuda"),
        patch.object(dw.workflow, "device_capacity_gb", return_value=24.0),
    ):
        workflow = _build(shots_definition())
        errors = _vram_errors(
            workflow.validation_errors(arguments={"width": 1344, "height": 768})
        )
        assert len(errors) == 1
        assert errors[0]["path"] == "variables.shots[1]"
        assert "shot@b" in errors[0]["message"]
        assert "with 3 references" in errors[0]["message"]


def test_for_each_member_over_ceiling_names_arguments_path_when_shots_is_supplied():
    with (
        patch.object(dw.workflow, "get_device_type", return_value="cuda"),
        patch.object(dw.workflow, "device_capacity_gb", return_value=24.0),
    ):
        workflow = _build(shots_definition())
        shots = copy.deepcopy(shots_definition()["variables"]["shots"])
        shots[1]["num_frames"] = 500
        errors = _vram_errors(workflow.validation_errors(arguments={"shots": shots}))
        assert len(errors) == 1
        assert errors[0]["path"] == "arguments.shots[1]"
        assert "shot@b" in errors[0]["message"]


def test_nulling_a_reference_brings_a_member_back_under_ceiling():
    with (
        patch.object(dw.workflow, "get_device_type", return_value="cuda"),
        patch.object(dw.workflow, "device_capacity_gb", return_value=24.0),
    ):
        d = shots_definition()
        d["variables"]["shots"][1]["references"][2]["from_file"] = None
        workflow = _build(d)
        errors = _vram_errors(
            workflow.validation_errors(arguments={"width": 1344, "height": 768})
        )
        assert errors == []


def test_a_literal_for_each_list_names_the_steps_path():
    with (
        patch.object(dw.workflow, "get_device_type", return_value="cuda"),
        patch.object(dw.workflow, "device_capacity_gb", return_value=24.0),
    ):
        d = shots_definition()
        d["steps"][0]["for_each"] = d["variables"].pop("shots")
        workflow = _build(d)
        errors = _vram_errors(
            workflow.validation_errors(arguments={"width": 1344, "height": 768})
        )
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].for_each[1]"
        assert "shot@b" in errors[0]["message"]


# --- Run-time backstop -----------------------------------------------------


def test_apply_vram_estimate_raises_on_an_expanded_definition():
    d = per_step_definition(
        [{"name": "s1", "pipeline": {"arguments": {"n": 30}}}], base_gb=0
    )
    with pytest.raises(ValueError):
        apply_vram_estimate(d, None, device_type="cuda", capacity_gb=24.0)


def test_apply_vram_estimate_raises_with_no_variables_block_at_all():
    d = per_step_definition(
        [{"name": "s1", "pipeline": {"arguments": {"n": 30}}}], base_gb=0
    )
    assert "variables" not in d
    with pytest.raises(ValueError):
        apply_vram_estimate(d, variables=None, device_type="cuda", capacity_gb=24.0)
