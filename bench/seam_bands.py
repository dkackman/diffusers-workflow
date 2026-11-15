"""Where a replacement edit stops: change by distance from the regenerated zone.

The measure from bench/records/performer-replacement.md ("Where the change
stops", "Temporal stability"). Given the source cut, the edited cut and the
mask video (all one size), build the union mask (mask > 128 on any frame),
take each pixel's distance to it, and per band (0 = inside the mask, then
1-16, 17-32, ... px outside) report:

  diff      mean |source - edit| over frames, 0-255 RGB
  flick_edit / flick_src
            mean frame-to-frame luma change in the edit / in the source
  step      the diff drop from the last band inside the regenerated zone
            (the nearest band outside the mask, band 1) to the next, so a hard
            seam is a number: a large step means the change stops at an edge.

usage: python bench/seam_bands.py SOURCE EDIT MASK [--band 16] [--bands N] [--json]
"""

import argparse
import json

import cv2
import numpy as np
from scipy import ndimage


def read_video(path):
    """All frames as a (T, H, W, 3) uint8 RGB array."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise SystemExit(f"no frames in {path}")
    return np.stack(frames)


def union_mask(mask_frames):
    """True where the mask exceeds 128 on any frame; (T,H,W[,C]) -> (H,W)."""
    m = mask_frames
    if m.ndim == 4:
        m = m.max(axis=-1)
    return (m > 128).any(axis=0)


def band_index(union, band=16):
    """Per-pixel band: 0 inside the union, k for distance in ((k-1)*band, k*band]."""
    dist = ndimage.distance_transform_edt(~union)
    return np.ceil(dist / band).astype(np.int32)


def luma(frames):
    f = frames.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def measure(source, edit, mask, band=16, bands=None):
    """Rows of {band, lo, hi, pixels, diff, flick_edit, flick_src} plus the step."""
    t = min(len(source), len(edit), len(mask))
    source, edit, mask = source[:t], edit[:t], mask[:t]
    idx = band_index(union_mask(mask), band)
    n = int(idx.max())
    if bands is not None:
        n = min(n, bands)
    diff = np.abs(source.astype(np.float32) - edit.astype(np.float32)).mean(
        axis=(0, -1)
    )
    fe = (
        np.abs(np.diff(luma(edit), axis=0)).mean(axis=0)
        if t > 1
        else np.zeros(idx.shape)
    )
    fs = (
        np.abs(np.diff(luma(source), axis=0)).mean(axis=0)
        if t > 1
        else np.zeros(idx.shape)
    )
    rows = []
    for k in range(n + 1):
        sel = idx == k
        if not sel.any():
            continue
        rows.append(
            {
                "band": k,
                "lo": 0 if k == 0 else (k - 1) * band + 1,
                "hi": 0 if k == 0 else k * band,
                "pixels": int(sel.sum()),
                "diff": float(diff[sel].mean()),
                "flick_edit": float(fe[sel].mean()),
                "flick_src": float(fs[sel].mean()),
            }
        )
    by = {r["band"]: r for r in rows}
    step = by[1]["diff"] - by[2]["diff"] if 1 in by and 2 in by else None
    return rows, step


def table(rows, step):
    out = [
        "| band px | pixels | diff | flick edit | flick src |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        label = "inside" if r["band"] == 0 else f"{r['lo']}-{r['hi']}"
        out.append(
            f"| {label} | {r['pixels']} | {r['diff']:.2f} | {r['flick_edit']:.2f} | {r['flick_src']:.2f} |"
        )
    out.append("")
    out.append(
        "hard-edge step (band 1 diff - band 2 diff): "
        + ("n/a" if step is None else f"{step:.2f}")
    )
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source")
    p.add_argument("edit")
    p.add_argument("mask")
    p.add_argument("--band", type=int, default=16, help="band width in px")
    p.add_argument("--bands", type=int, default=None, help="outside bands to report")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    src, edit, mask = read_video(a.source), read_video(a.edit), read_video(a.mask)
    if not (src.shape[1:3] == edit.shape[1:3] == mask.shape[1:3]):
        raise SystemExit(
            f"size mismatch: {src.shape[1:3]} {edit.shape[1:3]} {mask.shape[1:3]}"
        )
    rows, step = measure(src, edit, mask, a.band, a.bands)
    if a.json:
        print(json.dumps({"band_px": a.band, "step": step, "bands": rows}, indent=2))
    else:
        print(table(rows, step))


if __name__ == "__main__":
    main()
