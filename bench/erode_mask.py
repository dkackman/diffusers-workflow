"""Erode a person-mask video by N px, before H3 pools it onto patch cells (#1382).

The eroded-mask arm of bench/records/performer-replacement.md ("The seam: an
eroded mask"). Each frame is binarized the way `h3_mask_cells.binarized` does
it - Pillow's L conversion, white at or above 128 (`h3_rules.MASK_THRESHOLD`) -
then eroded by N px with a (2N+1)-square minimum filter, so a filled square
shrinks by N on every side. The frame border does not erode (mode "nearest"):
a person cut off by the frame edge stays masked up to that edge. The output is
a 0/255 mask with the input's size, frame count and fps, written by the
diffusers `export_to_video` dw writes its own masks with.

usage: python bench/erode_mask.py MASK.mp4 OUT.mp4 --px N
"""

import argparse

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

MASK_THRESHOLD = 128  # dw.pipeline_processors.h3_rules.MASK_THRESHOLD


def read_mask(path):
    """(frames as a (T, H, W, 3) uint8 RGB array, fps)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise SystemExit(f"no frames in {path}")
    return np.stack(frames), fps


def binarize(frames):
    """(T, H, W[, C]) uint8 -> (T, H, W) bool, by h3_mask_cells.binarized's rule."""
    luma = np.stack(
        [np.asarray(Image.fromarray(frame).convert("L")) for frame in frames]
    )
    return luma >= MASK_THRESHOLD


def erode(mask, px):
    """(T, H, W) bool eroded by `px` per frame with a (2px+1) square; 0 is the identity."""
    if px <= 0:
        return mask.copy()
    size = 2 * px + 1
    return np.stack(
        [
            ndimage.minimum_filter(frame.astype(np.uint8), size=size, mode="nearest")
            > 0
            for frame in mask
        ]
    )


def removed_fraction(before, after):
    """Share of the mask's white pixels the erosion removed (0 for an empty mask)."""
    white = int(before.sum())
    return 0.0 if white == 0 else (white - int(after.sum())) / white


def write_mask(mask, path, fps):
    """Write (T, H, W) bool as a 0/255 RGB video at `fps`."""
    from diffusers.utils import export_to_video

    frames = [
        Image.fromarray(np.repeat((m * 255).astype(np.uint8)[..., None], 3, axis=-1))
        for m in mask
    ]
    export_to_video(frames, path, fps=fps)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("mask")
    p.add_argument("out")
    p.add_argument("--px", type=int, required=True, help="erosion in px, 0 or more")
    a = p.parse_args()
    if a.px < 0:
        raise SystemExit("--px is 0 or more")
    frames, fps = read_mask(a.mask)
    before = binarize(frames)
    after = erode(before, a.px)
    write_mask(after, a.out, fps)
    print(
        f"{len(after)} frames, {after.shape[2]}x{after.shape[1]} @ {fps:g} fps; "
        f"eroded {a.px} px, removed {removed_fraction(before, after):.4f} of the mask"
    )


if __name__ == "__main__":
    main()
