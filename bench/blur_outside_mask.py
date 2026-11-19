"""Blur a reference picture's backdrop outside a person mask (#1439, stage #1460).

Arm 3 of bench/records/ref-prep-blur.md: the square crop on the person (arm 2)
with everything outside them blurred, so an H3 reference keeps the subject
sharp and gives the model no legible backdrop to carry into the clip. The mask
is a still `segment` result (prompt "person", white = subject), binarized at
128 like the masks dw writes; it must be the picture's own size, so segment
the crop that is blurred, not the uncropped picture.

The mask is grown by --grow px (a (2N+1)-square maximum filter, so hair and
shoulder edges the segmenter clipped stay sharp), feathered by a gaussian of
--feather sigma into a 0..1 alpha, and the picture is composited as
alpha * sharp + (1 - alpha) * gaussian(picture, --blur sigma). A grow of 0 and
a feather of 0 are the hard mask itself.

usage: python bench/blur_outside_mask.py IMAGE MASK OUT
       [--grow 12] [--feather 6] [--blur 20]
"""

import argparse

import numpy as np
from PIL import Image
from scipy import ndimage

MASK_THRESHOLD = 128  # white at or above, as dw's own masks are binarized


def read_mask(path, size):
    """(H, W) bool, white = subject; `size` is the picture's (width, height)."""
    mask = Image.open(path).convert("L")
    if mask.size != size:
        raise SystemExit(f"mask is {mask.size}, picture is {size}: segment the crop")
    return np.asarray(mask) >= MASK_THRESHOLD


def grow(mask, px):
    """`mask` dilated by `px` with a (2px+1) square; 0 is the identity."""
    if px <= 0:
        return mask.copy()
    return ndimage.maximum_filter(mask.astype(np.uint8), size=2 * px + 1) > 0


def feather(mask, sigma):
    """(H, W) float alpha in 0..1, the mask's edge softened by a gaussian."""
    alpha = mask.astype(np.float32)
    if sigma > 0:
        alpha = ndimage.gaussian_filter(alpha, sigma=sigma)
    return np.clip(alpha, 0.0, 1.0)


def blur(picture, sigma):
    """(H, W, 3) float picture blurred per channel by a gaussian of `sigma`."""
    return np.stack(
        [ndimage.gaussian_filter(picture[..., c], sigma=sigma) for c in range(3)],
        axis=-1,
    )


def composite(picture, blurred, alpha):
    """Sharp over blurred through `alpha`, as (H, W, 3) uint8."""
    a = alpha[..., None]
    out = a * picture + (1.0 - a) * blurred
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("image")
    p.add_argument("mask")
    p.add_argument("out")
    p.add_argument("--grow", type=int, default=12, help="mask growth in px")
    p.add_argument("--feather", type=float, default=6.0, help="edge gaussian sigma")
    p.add_argument("--blur", type=float, default=20.0, help="backdrop gaussian sigma")
    a = p.parse_args()
    if a.grow < 0 or a.feather < 0 or a.blur <= 0:
        raise SystemExit("--grow and --feather are 0 or more, --blur above 0")
    image = Image.open(a.image).convert("RGB")
    picture = np.asarray(image).astype(np.float32)
    mask = read_mask(a.mask, image.size)
    alpha = feather(grow(mask, a.grow), a.feather)
    Image.fromarray(composite(picture, blur(picture, a.blur), alpha)).save(a.out)
    print(
        f"{image.size[0]}x{image.size[1]}; mask {mask.mean():.4f} of the picture, "
        f"grown {a.grow} px, feather sigma {a.feather:g}, blur sigma {a.blur:g}"
    )


if __name__ == "__main__":
    main()
