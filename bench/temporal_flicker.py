"""Temporal flicker probe (#1747 stage A, #1749): does a flow-compensated
temporal IIR take the frame-to-frame shimmer out of a finished H3 clip without
smearing what moves?

The filter (#1747 plan v2, *The design -> Algorithm*), CPU only:

1. Frames go to float YCbCr (OpenCV's YCrCb order).
2. For each neighbour pair, DIS optical flow (`PRESET_FAST`) runs on half-scale
   grey frames; the flow is upsampled to full size and the neighbour (the
   running accumulator) is `cv2.remap`-ed onto the current frame.
3. A forward-backward consistency check: a pixel whose round trip misses by
   `occlusion_tolerance` px or more, or whose match falls outside the frame,
   gets weight 0, so moving edges and occlusions don't ghost.
4. A forward IIR and a backward IIR, averaged, so neither end frame is biased.
   The forward pass is kept as float16, never a float32 clip; the backward
   pass recomputes its flow rather than storing it.
5. Luma and chroma blend at their own strengths: out = cur + s * w * (warped -
   cur). Strength 0 for both returns the frames unchanged.

The two measures, both on the luma of the clip as decoded:

- **flicker**: the mean absolute frame-to-frame difference (0-255) over a
  static-region mask, the pixels whose full-resolution DIS flow between the two
  *input* frames is under `STATIC_PX` (0.25 px). The mask comes from the input
  and is applied to both before and after, so the two numbers count the same
  pixels.
- **motion blur**: the mean Sobel gradient energy (gx^2 + gy^2) over the moving
  subject, the pixels whose input flow is over `MOVING_PX` (1 px). A filter that
  smears what moves lowers it. It also counts grain as detail, so removing
  grain lowers it too; `edge_energy`, a diagnostic outside the gate, measures
  the strong moving edges after a blur that takes the grain out.

The numbers show a gain when flicker falls to `GAIN_FLICKER` (0.9) of the input
or below and the moving-edge gradient energy stays at `GAIN_KEEP` (0.95) of the
input or above ("flicker down, moving-edge gradient energy within a few % of
the input", the plan's gate).

    python bench/temporal_flicker.py run CLIP --out-dir DIR
        [--luma 0.5] [--chroma 0.5] [--occlusion-tolerance 1.5]
    python bench/temporal_flicker.py report DIR/temporal-flicker.json

`run` reads CLIP read-only and writes, in DIR: `before.mp4` and `after.mp4`
(the input and the filtered frames, both encoded the same way, x264 crf 12, so
the encode treats them alike), `pair.mp4` (before | after side by side) and
`temporal-flicker.json` (the record's numbers). The videos carry no audio. The
measures are taken on the frames in memory, not on the re-encoded files, so a
rerun on the same clip reproduces them. `report` prints the record's table.

Reimplemented from the description of ComfyUI-H3-ReTime's `H3TemporalStabilizer`
(MIT); no code was copied (docs/ATTRIBUTIONS.md).
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

LUMA = 0.5
CHROMA = 0.5
OCCLUSION_TOLERANCE = 1.5
STATIC_PX = 0.25
MOVING_PX = 1.0
GAIN_FLICKER = 0.9
GAIN_KEEP = 0.95
EDGE_SIGMA = 1.5
EDGE_TOP = 0.1
CRF = "12"


# ------------------------------------------------------------------ the filter


def _dis():
    return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)


def grey(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def flow_half(a, b, dis):
    """Flow from grey frame a to grey frame b, solved at half scale and
    returned at full scale: a(x) ~ b(x + f(x))."""
    h, w = a.shape
    hs, ws = max(1, h // 2), max(1, w // 2)
    small = dis.calc(
        cv2.resize(a, (ws, hs), interpolation=cv2.INTER_AREA),
        cv2.resize(b, (ws, hs), interpolation=cv2.INTER_AREA),
        None,
    )
    flow = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    flow[..., 0] *= w / ws
    flow[..., 1] *= h / hs
    return flow


def flow_full(a, b, dis):
    """Flow from grey frame a to grey frame b at full resolution."""
    return dis.calc(a, b, None)


def _grid(h, w):
    return np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))


def warp(image, flow):
    """Sample `image` at x + flow(x): the neighbour brought onto the current frame."""
    h, w = flow.shape[:2]
    gx, gy = _grid(h, w)
    return cv2.remap(
        image,
        gx + flow[..., 0],
        gy + flow[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def consistent(forward, backward, tolerance):
    """The pixels whose forward flow (current -> neighbour), followed by the
    backward flow (neighbour -> current) from where it lands, returns within
    `tolerance` px, and whose match lies inside the frame."""
    h, w = forward.shape[:2]
    back = warp(backward, forward)
    error = np.hypot(forward[..., 0] + back[..., 0], forward[..., 1] + back[..., 1])
    gx, gy = _grid(h, w)
    x = gx + forward[..., 0]
    y = gy + forward[..., 1]
    inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    return (error < tolerance) & inside


def _iir_pass(ycc, greys, order, strengths, tolerance, dis):
    """One recursive pass over `order`, yielding (index, filtered float32 YCrCb).
    The accumulator is one frame; flow is computed on the input greys."""
    acc = None
    previous = None
    for i in order:
        current = ycc[i].astype(np.float32)
        if acc is None:
            out = current
        else:
            forward = flow_half(greys[i], greys[previous], dis)
            backward = flow_half(greys[previous], greys[i], dis)
            mask = consistent(forward, backward, tolerance).astype(np.float32)
            weight = mask[..., None] * strengths
            out = current + weight * (warp(acc, forward) - current)
        acc = out
        previous = i
        yield i, out


def denoise(frames, luma=LUMA, chroma=CHROMA, occlusion_tolerance=OCCLUSION_TOLERANCE):
    """The flow-compensated temporal IIR over uint8 RGB frames (T, H, W, 3);
    returns uint8 RGB frames of the same shape."""
    frames = np.asarray(frames)
    if (luma == 0 and chroma == 0) or len(frames) < 2:
        return frames.copy()
    dis = _dis()
    ycc = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb) for f in frames])
    greys = [grey(f) for f in frames]
    strengths = np.array([luma, chroma, chroma], dtype=np.float32)
    count = len(frames)
    forward = np.empty(ycc.shape, dtype=np.float16)
    for i, out in _iir_pass(
        ycc, greys, range(count), strengths, occlusion_tolerance, dis
    ):
        forward[i] = out
    result = np.empty_like(frames)
    for i, out in _iir_pass(
        ycc, greys, range(count - 1, -1, -1), strengths, occlusion_tolerance, dis
    ):
        both = 0.5 * (forward[i].astype(np.float32) + out)
        ycc8 = np.clip(np.rint(both), 0, 255).astype(np.uint8)
        result[i] = cv2.cvtColor(ycc8, cv2.COLOR_YCrCb2RGB)
    return result


# ------------------------------------------------------------------ the measures


def motion_masks(frames, static_px=STATIC_PX, moving_px=MOVING_PX):
    """Per neighbour pair (t-1, t), t >= 1: the static and moving masks from the
    full-resolution input flow. Returns (static, moving), each (T-1, H, W) bool."""
    dis = _dis()
    greys = [grey(f) for f in frames]
    static = []
    moving = []
    for t in range(1, len(frames)):
        flow = flow_full(greys[t], greys[t - 1], dis)
        magnitude = np.hypot(flow[..., 0], flow[..., 1])
        static.append(magnitude < static_px)
        moving.append(magnitude > moving_px)
    return np.array(static), np.array(moving)


def _luma(frames):
    return [
        cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32) for f in frames
    ]


def flicker(frames, static):
    """Mean |Y_t - Y_{t-1}| (0-255) over the static mask, pooled over every pair;
    None when no pixel is static."""
    luma = _luma(frames)
    total = 0.0
    count = 0
    for t in range(1, len(luma)):
        mask = static[t - 1]
        total += float(np.abs(luma[t] - luma[t - 1])[mask].sum())
        count += int(mask.sum())
    return total / count if count else None


def gradient_energy(frames, moving):
    """Mean Sobel gradient energy gx^2 + gy^2 of Y_t over the moving mask of
    pair (t-1, t), pooled over every t >= 1; None when nothing moves."""
    luma = _luma(frames)
    total = 0.0
    count = 0
    for t in range(1, len(luma)):
        mask = moving[t - 1]
        gx = cv2.Sobel(luma[t], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(luma[t], cv2.CV_32F, 0, 1, ksize=3)
        total += float((gx * gx + gy * gy)[mask].sum())
        count += int(mask.sum())
    return total / count if count else None


def edge_energy(frames, moving, reference):
    """A diagnostic beside the gate, not part of it: gradient energy of the
    Gaussian-blurred (sigma `EDGE_SIGMA`) luma over the strong moving edges -
    moving pixels in the top `EDGE_TOP` share of the *reference* (input) frame's
    blurred gradient. The blur takes grain out of the measure, so a drop here is
    a smeared edge, while a drop in `gradient_energy` alone can be grain
    removed. None when no pixel qualifies."""
    luma = _luma(frames)
    ref = _luma(reference)
    total = 0.0
    count = 0
    for t in range(1, len(luma)):
        r = cv2.GaussianBlur(ref[t], (0, 0), EDGE_SIGMA)
        rx = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3)
        ry = cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3)
        strength = rx * rx + ry * ry
        mask = moving[t - 1] & (strength >= np.quantile(strength, 1.0 - EDGE_TOP))
        b = cv2.GaussianBlur(luma[t], (0, 0), EDGE_SIGMA)
        gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
        total += float((gx * gx + gy * gy)[mask].sum())
        count += int(mask.sum())
    return total / count if count else None


def measure(before, after, masks=None):
    """Both measures on before and after, over the input's masks."""
    static, moving = masks if masks is not None else motion_masks(before)
    pixels = static.size or 1
    result = {
        "static_share": round(float(static.sum()) / pixels, 4),
        "moving_share": round(float(moving.sum()) / pixels, 4),
        "flicker": {"before": flicker(before, static), "after": flicker(after, static)},
        "gradient_energy": {
            "before": gradient_energy(before, moving),
            "after": gradient_energy(after, moving),
        },
        "edge_energy": {
            "before": edge_energy(before, moving, before),
            "after": edge_energy(after, moving, before),
        },
    }
    for key in ("flicker", "gradient_energy", "edge_energy"):
        pair = result[key]
        pair["ratio"] = (
            pair["after"] / pair["before"]
            if pair["before"] and pair["after"] is not None
            else None
        )
    f, g = result["flicker"]["ratio"], result["gradient_energy"]["ratio"]
    result["gain"] = bool(
        f is not None and g is not None and f <= GAIN_FLICKER and g >= GAIN_KEEP
    )
    return result


# ------------------------------------------------------------------ files


def read_clip(path):
    import imageio.v2 as imageio

    reader = imageio.get_reader(path, format="FFMPEG")
    try:
        fps = float(reader.get_meta_data()["fps"])
        frames = np.stack([np.asarray(f)[..., :3] for f in reader])
    finally:
        reader.close()
    return frames, fps


def write_clip(path, frames, fps):
    import imageio.v2 as imageio

    writer = imageio.get_writer(
        path,
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        quality=None,
        macro_block_size=1,
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", CRF, "-preset", "slow"],
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _round(value, places=4):
    return None if value is None else round(value, places)


def run(clip, out_dir, luma, chroma, tolerance):
    frames, fps = read_clip(clip)
    wall = time.perf_counter()
    cpu = time.process_time()
    after = denoise(frames, luma, chroma, tolerance)
    filter_wall = time.perf_counter() - wall
    filter_cpu = time.process_time() - cpu
    result = measure(frames, after)
    for key in ("flicker", "gradient_energy", "edge_energy"):
        result[key] = {k: _round(v) for k, v in result[key].items()}
    os.makedirs(out_dir, exist_ok=True)
    write_clip(os.path.join(out_dir, "before.mp4"), frames, fps)
    write_clip(os.path.join(out_dir, "after.mp4"), after, fps)
    write_clip(
        os.path.join(out_dir, "pair.mp4"),
        np.concatenate([frames, after], axis=2),
        fps,
    )
    record = {
        "clip": os.path.abspath(clip),
        "clip_sha256": _sha256(clip),
        "frames": int(len(frames)),
        "width": int(frames.shape[2]),
        "height": int(frames.shape[1]),
        "fps": fps,
        "strengths": {
            "luma": luma,
            "chroma": chroma,
            "occlusion_tolerance": tolerance,
        },
        "filter_wall_s": round(filter_wall, 1),
        "filter_cpu_s": round(filter_cpu, 1),
        "filter_wall_s_per_frame": round(filter_wall / len(frames), 3),
        "gate": {
            "flicker_ratio_at_most": GAIN_FLICKER,
            "gradient_ratio_at_least": GAIN_KEEP,
        },
        **result,
        "outputs": os.path.abspath(out_dir),
        "host": platform.node(),
        "cpu_threads": cv2.getNumThreads(),
        "opencv": cv2.__version__,
    }
    with open(os.path.join(out_dir, "temporal-flicker.json"), "w") as handle:
        json.dump(record, handle, indent=2)
    return record


def report(record):
    f, g, e = record["flicker"], record["gradient_energy"], record["edge_energy"]
    s = record["strengths"]
    lines = [
        f"clip: {record['clip']} ({record['frames']} frames, "
        f"{record['width']}x{record['height']} @ {record['fps']:g} fps)",
        f"strengths: luma {s['luma']}, chroma {s['chroma']}, "
        f"occlusion_tolerance {s['occlusion_tolerance']} px",
        f"filter time: {record['filter_wall_s']} s wall, {record['filter_cpu_s']} s CPU",
        f"static share {record['static_share']}, moving share {record['moving_share']}",
        "",
        "| measure | before | after | after/before |",
        "|---|---|---|---|",
        f"| flicker (static mean abs dY) | {f['before']} | {f['after']} | {f['ratio']} |",
        f"| moving gradient energy | {g['before']} | {g['after']} | {g['ratio']} |",
        f"| strong moving edges, grain blurred out (diagnostic) | {e['before']} "
        f"| {e['after']} | {e['ratio']} |",
        "",
        f"gain: {'yes' if record['gain'] else 'no'}",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("clip")
    run_parser.add_argument("--out-dir", required=True)
    run_parser.add_argument("--luma", type=float, default=LUMA)
    run_parser.add_argument("--chroma", type=float, default=CHROMA)
    run_parser.add_argument(
        "--occlusion-tolerance", type=float, default=OCCLUSION_TOLERANCE
    )
    report_parser = sub.add_parser("report")
    report_parser.add_argument("record")
    args = parser.parse_args(argv)
    if args.command == "run":
        record = run(
            args.clip, args.out_dir, args.luma, args.chroma, args.occlusion_tolerance
        )
    else:
        record = json.loads(Path(args.record).read_text())
    print(report(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
