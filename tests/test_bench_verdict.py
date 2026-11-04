import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))

from verdict import noise_floor, verdict  # noqa: E402


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
