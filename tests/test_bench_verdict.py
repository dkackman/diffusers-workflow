import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

import pytest  # noqa: E402
from verdict import combined_floor, noise_floor, rerun_gap, verdict  # noqa: E402


def test_verdict_wins_only_past_noise_and_never_worse_elsewhere():
    noise = {"seam_look_mean": 0.2, "texture_fine_seg_last": 0.1}
    control = {"seam_look_mean": 2.0, "texture_fine_seg_last": 1.0}
    assert (
        verdict(
            control,
            {"seam_look_mean": 1.5, "texture_fine_seg_last": 1.05},
            noise,
            "seam_look_mean",
        )
        == "wins"
    )
    assert (
        verdict(
            control,
            {"seam_look_mean": 1.9, "texture_fine_seg_last": 1.0},
            noise,
            "seam_look_mean",
        )
        == "tie"
    )
    assert (
        verdict(
            control,
            {"seam_look_mean": 1.5, "texture_fine_seg_last": 1.4},
            noise,
            "seam_look_mean",
        )
        == "mixed"
    )
    assert (
        verdict(
            control,
            {"seam_look_mean": 2.6, "texture_fine_seg_last": 1.0},
            noise,
            "seam_look_mean",
        )
        == "loses"
    )


def test_noise_floor_is_the_absolute_difference_per_measure():
    assert noise_floor({"a": 1.0, "b": 3.0}, {"a": 1.25, "b": 2.0}) == {
        "a": 0.25,
        "b": 1.0,
    }


def test_rerun_gap_is_the_largest_pairwise_difference_per_measure():
    pairs = [
        ({"look": 1.0, "level": 7.0}, {"look": 1.2, "level": 11.0}),
        ({"look": 2.0, "level": 5.0}, {"look": 1.5, "level": 4.0}),
    ]
    gap = rerun_gap(pairs)
    assert gap["look"] == pytest.approx(0.5)
    assert gap["level"] == pytest.approx(4.0)
    assert rerun_gap([]) == {}


def test_combined_floor_takes_the_larger_of_seed_floor_and_rerun_gap():
    seed = {"look": 0.12, "texture": 0.081}
    rerun = {"look": 0.44, "texture": 0.068, "level": 6.2}
    assert combined_floor(seed, rerun) == {"look": 0.44, "texture": 0.081, "level": 6.2}


def test_a_seed_floor_win_inside_the_rerun_gap_is_a_tie():
    control = {"look": 1.46, "texture": 0.536}
    arm = {"look": 1.31, "texture": 0.464}
    seed = {"look": 0.12, "texture": 0.081}
    assert verdict(control, arm, seed, "look") == "wins"
    floor = combined_floor(seed, rerun_gap([({"look": 1.53}, {"look": 1.09})]))
    assert verdict(control, arm, floor, "look") == "tie"
