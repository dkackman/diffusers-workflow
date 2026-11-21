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

--inner K (#1382) also splits the inside: K bands of --band px by distance to
the outside of the union mask, numbered -1 (1-16 px inside), -2, ..., with
band 0 the rest of the inside. It adds `jump`, band -1's diff minus band 1's,
so an edge that moved inside the silhouette is a number too. `step` keeps its
meaning, and without --inner the output is unchanged.

usage: python bench/seam_bands.py SOURCE EDIT MASK [--band 16] [--bands N]
       [--inner K] [--json]
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


def band_index(union, band=16, inner=0):
    """Per-pixel band: 0 inside the union, k for distance in ((k-1)*band, k*band].

    With `inner` K > 0, an inside pixel whose distance to the outside is in
    ((k-1)*band, k*band] for k <= K is band -k; band 0 is the rest."""
    dist = ndimage.distance_transform_edt(~union)
    idx = np.ceil(dist / band).astype(np.int32)
    if inner > 0:
        depth = np.ceil(ndimage.distance_transform_edt(union) / band).astype(np.int32)
        edge = union & (depth <= inner)
        idx[edge] = -depth[edge]
    return idx


def luma(frames):
    f = frames.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def measure(source, edit, mask, band=16, bands=None, inner=0):
    """Rows of {band, lo, hi, pixels, diff, flick_edit, flick_src} plus the step.

    Rows run from the inside out: 0, then -inner ... -1, then 1, 2, ..."""
    t = min(len(source), len(edit), len(mask))
    source, edit, mask = source[:t], edit[:t], mask[:t]
    idx = band_index(union_mask(mask), band, inner)
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
    for k in [0, *range(-inner, 0), *range(1, n + 1)]:
        sel = idx == k
        if not sel.any():
            continue
        rows.append(
            {
                "band": k,
                "lo": 0 if k == 0 else (abs(k) - 1) * band + 1,
                "hi": 0 if k == 0 else abs(k) * band,
                "pixels": int(sel.sum()),
                "diff": float(diff[sel].mean()),
                "flick_edit": float(fe[sel].mean()),
                "flick_src": float(fs[sel].mean()),
            }
        )
    by = {r["band"]: r for r in rows}
    step = by[1]["diff"] - by[2]["diff"] if 1 in by and 2 in by else None
    return rows, step


def inner_jump(rows):
    """Band -1's diff minus band 1's: the drop across the mask's edge, or None."""
    by = {r["band"]: r for r in rows}
    return by[-1]["diff"] - by[1]["diff"] if -1 in by and 1 in by else None


def _label(r):
    if r["band"] == 0:
        return "inside"
    if r["band"] < 0:
        return f"in {r['lo']}-{r['hi']}"
    return f"{r['lo']}-{r['hi']}"


def table(rows, step, jump=None):
    out = [
        "| band px | pixels | diff | flick edit | flick src |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {_label(r)} | {r['pixels']} | {r['diff']:.2f} | {r['flick_edit']:.2f} | {r['flick_src']:.2f} |"
        )
    out.append("")
    out.append(
        "hard-edge step (band 1 diff - band 2 diff): "
        + ("n/a" if step is None else f"{step:.2f}")
    )
    if jump is not None:
        out.append(f"inner jump (band -1 diff - band 1 diff): {jump:.2f}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("source")
    p.add_argument("edit")
    p.add_argument("mask")
    p.add_argument("--band", type=int, default=16, help="band width in px")
    p.add_argument("--bands", type=int, default=None, help="outside bands to report")
    p.add_argument(
        "--inner", type=int, default=0, help="inside bands to split off (#1382)"
    )
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    if a.inner < 0:
        raise SystemExit("--inner is 0 or more")
    src, edit, mask = read_video(a.source), read_video(a.edit), read_video(a.mask)
    if not (src.shape[1:3] == edit.shape[1:3] == mask.shape[1:3]):
        raise SystemExit(
            f"size mismatch: {src.shape[1:3]} {edit.shape[1:3]} {mask.shape[1:3]}"
        )
    rows, step = measure(src, edit, mask, a.band, a.bands, a.inner)
    if a.inner:
        jump = inner_jump(rows)
        if a.json:
            out = {"band_px": a.band, "step": step, "jump": jump, "bands": rows}
            print(json.dumps(out, indent=2))
        else:
            print(table(rows, step, jump))
    elif a.json:
        print(json.dumps({"band_px": a.band, "step": step, "bands": rows}, indent=2))
    else:
        print(table(rows, step))


if __name__ == "__main__":
    main()
