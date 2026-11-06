"""Texture energy by wavelength band - `compare_outputs(texture_bands=true)`
(#1152). A measurement, not a score: how much of a region's variation sits
at fine (2-8 px), mid (8-32 px) and coarse (32-128 px) wavelengths, on luma
and on two chroma differences, so an A/B can tell coarsening or blotching
(mid up) from blur or over-smoothing (every band down).

Per channel (Y, R-G, B-G, on the 0-255 scale): a least-squares plane is
removed so a linear lighting gradient reads as nothing, a 2D Hann window
tapers the edges so the box's border is not a step, and `numpy.fft.rfft2`
gives the spectrum. Each band is the RMS of the components whose radial
frequency falls in that band's wavelengths, rescaled for the window's
energy, so a sine of amplitude A measures about A/sqrt(2) in its band. A
band whose longest wavelength exceeds the box's short side can't be held
by the box and is None.

The method is after knoic/ComfyUI-MiniMaxH3-PrefixStream (MIT); see
docs/ATTRIBUTIONS.md. Pure numpy; the caller decodes and crops.
"""

import numpy

# Band name -> (shortest, longest) wavelength in pixels
BANDS = {"fine": (2, 8), "mid": (8, 32), "coarse": (32, 128)}

# The smallest box side that still holds the fine band
MIN_SIDE = 16

CHANNELS = ("luma", "chroma_rg", "chroma_bg")


def texture_bands(rgb):
    """`{luma, chroma_rg, chroma_bg}`, each `{fine, mid, coarse}` (float or
    None), for an (h, w, 3) uint8 RGB array."""
    rgb = numpy.asarray(rgb, dtype=numpy.float64)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"texture_bands needs an (h, w, 3) array, got {rgb.shape}")
    height, width = rgb.shape[:2]
    if min(height, width) < MIN_SIDE:
        raise ValueError(
            f"a {width}x{height} region is too small for the 2-8 px band "
            f"(at least {MIN_SIDE} px a side)"
        )
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    planes = {
        "luma": red * 0.299 + green * 0.587 + blue * 0.114,
        "chroma_rg": red - green,
        "chroma_bg": blue - green,
    }
    masks = _band_masks(height, width)
    window = numpy.outer(numpy.hanning(height), numpy.hanning(width))
    window_power = float(numpy.mean(window * window))
    return {
        channel: _bands(planes[channel], window, window_power, masks)
        for channel in CHANNELS
    }


def ratio(value, base):
    """`value / base` per channel and band, None where either is None or
    the baseline band is 0."""
    return {
        channel: {
            band: (
                None
                if value[channel][band] is None or not base[channel][band]
                else round(value[channel][band] / base[channel][band], 4)
            )
            for band in BANDS
        }
        for channel in CHANNELS
    }


def _bands(plane, window, window_power, masks):
    tapered = _detrend(plane) * window
    spectrum = numpy.fft.rfft2(tapered)
    power = (spectrum.real**2 + spectrum.imag**2) * masks["weight"]
    count = plane.size
    out = {}
    for band, mask in masks["bands"].items():
        if mask is None:
            out[band] = None
            continue
        mean_square = float(power[mask].sum()) / (count * count)
        out[band] = round(float(numpy.sqrt(mean_square / window_power)), 4)
    return out


def _detrend(plane):
    """`plane` less its least-squares fit a + b*x + c*y. On a full grid the
    centred x and y are orthogonal to each other and to the constant, so
    each coefficient is its own projection - no solve needed."""
    height, width = plane.shape
    x = numpy.arange(width, dtype=numpy.float64) - (width - 1) / 2.0
    y = numpy.arange(height, dtype=numpy.float64) - (height - 1) / 2.0
    slope_x = float((plane * x[None, :]).sum()) / (height * float((x * x).sum()))
    slope_y = float((plane * y[:, None]).sum()) / (width * float((y * y).sum()))
    return plane - plane.mean() - slope_x * x[None, :] - slope_y * y[:, None]


def _band_masks(height, width):
    """Per-band boolean masks over the rfft2 grid, and the weight that
    counts each rfft column's mirrored twin (Parseval over a half
    spectrum)."""
    fy = numpy.fft.fftfreq(height)[:, None]
    fx = numpy.fft.rfftfreq(width)[None, :]
    radius = numpy.sqrt(fx * fx + fy * fy)
    weight = numpy.full((1, fx.shape[1]), 2.0)
    weight[0, 0] = 1.0
    if width % 2 == 0:
        weight[0, -1] = 1.0
    short_side = min(height, width)
    bands = {}
    for band, (shortest, longest) in BANDS.items():
        if longest > short_side:
            bands[band] = None
            continue
        # wavelength in [shortest, longest) px; the fine band keeps Nyquist
        bands[band] = (radius > 1.0 / longest) & (radius <= 1.0 / shortest)
    return {"weight": weight, "bands": bands}
