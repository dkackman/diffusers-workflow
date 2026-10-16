"""The comparison engine behind compare_outputs (#871): the metrics on
synthetic pixels, and the checks that refuse a request by its shape."""

import math

import numpy
import pytest

from dw.media_compare import (
    check_request,
    crop_box,
    mean_abs_diff,
    psnr,
    resolve_moments,
    short_name,
    ssim,
)


def noise(seed=0, shape=(48, 48, 3)):
    return numpy.random.default_rng(seed).integers(0, 256, shape, dtype=numpy.uint8)


def test_identical_arrays_are_identical_on_every_metric():
    """No difference is an infinite PSNR, a perfect SSIM and a zero error."""
    a = noise()

    assert psnr(a, a.copy()) == math.inf
    assert ssim(a, a.copy()) == pytest.approx(1.0, abs=1e-9)
    assert mean_abs_diff(a, a.copy()) == 0


def test_a_constant_offset_gives_the_psnr_its_mse_implies():
    """An offset of 5 is an MSE of 25, so 10*log10(255^2/25) dB."""
    a = numpy.full((16, 16, 3), 100, numpy.uint8)
    b = a + 5

    assert psnr(a, b) == pytest.approx(10 * math.log10(255**2 / 25))
    assert psnr(a, b) == pytest.approx(34.15, abs=0.01)
    assert mean_abs_diff(a, b) == pytest.approx(5.0)


def test_psnr_does_not_wrap_at_the_uint8_edge():
    """255 against 0 is the largest error, not a wrapped small one."""
    a = numpy.zeros((4, 4, 3), numpy.uint8)
    b = numpy.full((4, 4, 3), 255, numpy.uint8)

    assert psnr(a, b) == pytest.approx(0.0)
    assert mean_abs_diff(a, b) == 255


def test_a_shift_lowers_ssim_far_more_than_a_brightness_change():
    """SSIM reads structure: a rolled texture is a different picture, a
    slight brightening of it is the same one."""
    a = noise(1)
    shifted = numpy.roll(a, 3, axis=1)
    brighter = numpy.clip(a.astype(int) + 4, 0, 255).astype(numpy.uint8)

    assert ssim(a, shifted) < 0.5
    assert ssim(a, brighter) > 0.95


def test_a_shifted_checkerboard_scores_low():
    """A checkerboard moved by one square inverts, so it shares no structure."""
    board = (numpy.indices((32, 32)).sum(axis=0) // 4 % 2 * 255).astype(numpy.uint8)
    a = numpy.stack([board] * 3, axis=-1)

    assert ssim(a, numpy.roll(a, 4, axis=1)) < 0.2


NAMES = ["a.mp4", "b.mp4"]


@pytest.mark.parametrize("names", [["a.mp4"], [f"{i}.mp4" for i in range(9)]])
def test_check_request_refuses_a_name_count_outside_2_to_8(names):
    """Compare takes two to eight outputs."""
    with pytest.raises(ValueError, match="compare takes 2 to 8"):
        check_request(names, None, None, None)


@pytest.mark.parametrize("at", [[], list(range(9))])
def test_check_request_refuses_a_moment_count_outside_1_to_8(at):
    """Compare takes one to eight moments."""
    with pytest.raises(ValueError, match="compare takes 1 to 8"):
        check_request(NAMES, at, None, None)


def test_check_request_refuses_a_baseline_that_is_not_a_name():
    """The baseline is one of the outputs compared."""
    with pytest.raises(ValueError, match="not one of `names`"):
        check_request(NAMES, None, None, "c.mp4")


def test_check_request_refuses_a_crop_list_of_the_wrong_length():
    """One box per output, or one box for all."""
    with pytest.raises(ValueError, match="3 boxes for 2 names"):
        check_request(NAMES, None, [[0, 0, 4, 4]] * 3, None)


def test_check_request_reports_the_baseline_row_and_crop_mode():
    """The baseline defaults to the first row; the crop form picks a mode."""
    assert check_request(NAMES, None, None, None) == (0, None)
    assert check_request(NAMES, [0, 1], [0, 0, 4, 4], "b.mp4") == (1, "shared")
    assert check_request(NAMES, None, [[0, 0, 4, 4], [1, 1, 2, 2]], None) == (
        0,
        "each",
    )


def test_crop_box_returns_a_pillow_box():
    """[x, y, w, h] becomes (left, upper, right, lower)."""
    assert crop_box([2, 3, 10, 5], 32, 16, "a.mp4") == (2, 3, 12, 8)


def test_crop_box_refuses_a_box_outside_the_frame_naming_the_output():
    """A clamped box would measure a different region per output."""
    with pytest.raises(ValueError, match="out of bounds for a.mp4"):
        crop_box([30, 0, 10, 4], 32, 16, "a.mp4")


@pytest.mark.parametrize("box", [[-1, 0, 4, 4], [0, -1, 4, 4], [0, 0, 0, 4]])
def test_crop_box_refuses_a_negative_origin_or_empty_size(box):
    """The origin is non-negative and the size positive."""
    with pytest.raises(ValueError, match="non-negative origin"):
        crop_box(box, 32, 16, "a.mp4")


REFERENCE = {"fps": 6.0, "frame_count": 24}


def test_resolve_moments_defaults_to_four_evenly_spaced_frames():
    """Without `at`, four frames spread over the shortest video."""
    assert resolve_moments(None, REFERENCE, 24) == [
        ("frame:0", 0),
        ("frame:8", 8),
        ("frame:15", 15),
        ("frame:23", 23),
    ]


@pytest.mark.parametrize("moment", ["frame:-1", -1.0, "-0.01"])
def test_resolve_moments_refuses_a_negative_moment(moment):
    """A negative moment is refused, never wrapped to the clip's end."""
    with pytest.raises(ValueError, match="is negative"):
        resolve_moments([moment], REFERENCE, 24)


def test_resolve_moments_converts_seconds_at_the_reference_fps():
    """1.5 s at 6 fps is frame 9."""
    assert resolve_moments([1.5, "0"], REFERENCE, 24) == [(1.5, 9), ("0", 0)]


def test_get_output_frames_still_counts_a_negative_frame_from_the_end():
    """frames keeps Python-style negatives; only compare refuses them."""
    from dw.media_frames import moment_index

    assert moment_index("frame:-1", REFERENCE) == 23


def test_short_name_keeps_an_asset_reference_and_trims_an_output_path():
    """An asset reads as given; an output as its run id and file."""
    assert short_name("asset:uploads/a.png") == "asset:uploads/a.png"
    assert short_name("wf/run1/out.png") == "run1/out.png"


@pytest.mark.parametrize("moment", ["50%", "frame:x", "nan"])
def test_resolve_moments_refuses_what_get_output_frames_refuses(moment):
    # One `at` grammar (media_frames.moment_index): compare and frames
    # refuse the same malformed moment with the same words.
    from dw.media_frames import frames_at

    with pytest.raises(ValueError) as frames_error:
        frames_at("unused.mp4", [moment], shape=REFERENCE)
    with pytest.raises(ValueError) as compare_error:
        resolve_moments([moment], REFERENCE, 24)
    assert str(compare_error.value) == str(frames_error.value)


def test_crop_box_refuses_a_malformed_box_as_resolve_crop_box_does():
    # One crop parse (media_frames.parse_crop); only the bounds policy differs.
    from dw.media_frames import resolve_crop_box

    with pytest.raises(ValueError) as frames_error:
        resolve_crop_box([1, 2, 3], 32, 16)
    with pytest.raises(ValueError) as compare_error:
        crop_box([1, 2, 3], 32, 16, "a.mp4")
    assert str(compare_error.value) == str(frames_error.value)
