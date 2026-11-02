"""Texture energy by wavelength band (#1152): what each band measures, on
synthetic arrays."""

import numpy
import pytest
from scipy.ndimage import gaussian_filter

from dw.texture_bands import CHANNELS, ratio, texture_bands

SIZE = 256


def grey(plane):
    """A float plane as a grey uint8 RGB image."""
    plane = numpy.clip(numpy.rint(plane), 0, 255).astype(numpy.uint8)
    return numpy.stack([plane] * 3, axis=-1)


def stripes(period, amplitude=60):
    x = numpy.arange(SIZE)[None, :].repeat(SIZE, axis=0)
    return grey(128 + amplitude * numpy.sin(2 * numpy.pi * x / period))


def textured(seed=0):
    rng = numpy.random.default_rng(seed)
    return gaussian_filter(rng.normal(128, 50, (SIZE, SIZE)), 1.0)


@pytest.mark.parametrize("period, band", [(4, "fine"), (16, "mid"), (64, "coarse")])
def test_a_stripe_puts_its_energy_in_the_band_of_its_wavelength(period, band):
    luma = texture_bands(stripes(period))["luma"]

    for other in set(luma) - {band}:
        assert luma[band] > 10 * luma[other]
    # a sine of amplitude 60 measures about 60/sqrt(2)
    assert luma[band] == pytest.approx(42.4, rel=0.15)


def test_a_linear_gradient_measures_nothing_in_any_band():
    y, x = numpy.mgrid[0:SIZE, 0:SIZE]
    bands = texture_bands(grey(20 + 0.5 * x + 0.3 * y))

    for channel in CHANNELS:
        for value in bands[channel].values():
            assert value < 0.5


def test_blurring_lowers_every_luma_band():
    sharp = textured()
    soft = gaussian_filter(sharp, 3.0)

    before = texture_bands(grey(sharp))["luma"]
    after = texture_bands(grey(soft))["luma"]

    for band in before:
        assert after[band] < before[band]


def test_a_hue_change_at_constant_luma_moves_chroma_not_luma():
    base = textured()
    pattern = 20 * numpy.sin(2 * numpy.pi * numpy.arange(SIZE) / 6)[None, :]
    pattern = pattern.repeat(SIZE, axis=0)
    # +d on R and -d*0.299/0.114 ... keep luma: 0.299*dr + 0.114*db = 0
    red = base + pattern
    blue = base - pattern * 0.299 / 0.114
    tinted = numpy.stack([red, base, blue], axis=-1)
    tinted = numpy.clip(numpy.rint(tinted), 0, 255).astype(numpy.uint8)

    plain = texture_bands(grey(base))
    changed = texture_bands(tinted)

    assert plain["chroma_rg"]["fine"] < 0.5
    assert changed["chroma_rg"]["fine"] > 5 * max(plain["chroma_rg"]["fine"], 1)
    assert changed["chroma_bg"]["fine"] > 5
    for band in ("fine", "mid", "coarse"):
        assert changed["luma"][band] == pytest.approx(plain["luma"][band], rel=0.05)


def test_a_small_box_holds_only_the_bands_that_fit():
    bands = texture_bands(numpy.zeros((20, 40, 3), numpy.uint8) + 7)

    for channel in CHANNELS:
        assert bands[channel]["fine"] is not None
        assert bands[channel]["mid"] is None
        assert bands[channel]["coarse"] is None


def test_a_box_under_16_px_a_side_is_refused():
    with pytest.raises(ValueError, match="too small"):
        texture_bands(numpy.zeros((15, 64, 3), numpy.uint8))


def test_ratio_is_none_for_a_zero_or_missing_band_and_one_for_itself():
    flat = texture_bands(numpy.full((20, 40, 3), 9, numpy.uint8))
    busy = texture_bands(grey(textured())[:20, :40])

    assert ratio(busy, busy)["luma"]["fine"] == 1.0
    assert ratio(busy, busy)["luma"]["mid"] is None  # None band stays None
    # flat baseline: every fine band is exactly 0
    assert flat["luma"]["fine"] == 0
    assert ratio(busy, flat)["luma"]["fine"] is None
    assert ratio(flat, busy)["luma"]["fine"] == 0
