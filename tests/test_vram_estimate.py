"""vram_estimate refuses a shape against the device actually serving it.

Every catalog 'cost' entry is a CUDA card, so checking all of them refused a
64 GB Mac against a 24 GB RTX 3090 - and filtering to matching entries alone
would have switched the guard off on a Mac entirely, since none match.
"""

import pytest

from dw.vram_estimate import apply_vram_estimate, vram_estimate_errors


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
