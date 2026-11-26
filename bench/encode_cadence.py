"""Encode-cadence probe (#1748 stage A, #1752): does dw's H.264 output show a
periodic quality dip on crisp content because of x264's B-frames?

libx264 at its defaults (crf 23, 3 B-frames, pbratio 1.3) quantizes B-frames
harder than P-frames. On soft footage that is invisible; on hard black strokes
over flat fills it could show as a pulse every few frames. The probe encodes a
crisp frame source four ways, CPU only:

- **a_audio_path**: dw's audio path, `dw.writers.encode_video` (diffusers'
  `encode_video`, PyAV libx264 at its defaults), exactly as dw calls it.
- **b_silent_path**: dw's silent path, diffusers `export_to_video`
  (imageio-ffmpeg, `quality=5`) with dw's `macro_block_size`, as
  `dw/result.py` `_write_video_file` calls it.
- **c_bf0**: path (a) with libx264 `bf=0` (no B-frames).
- **d_crf18**: path (a) with libx264 `crf=18`.

(c) and (d) set the options in the `add_stream("libx264", ...)` call that
`writers._ExactRateContainer` already intercepts, the mechanism #1748's stage
B would use; everything else about the encode is (a)'s.

Each file is decoded with PyAV and compared with the source frame by frame:

- **PSNR** on RGB (peak 255), and **SSIM** on BT.601 luma (Gaussian window,
  sigma 1.5, the Wang et al. constants);
- the decoded frame's `pict_type` (I, P, B), so the mean PSNR of B-frames is
  set against the mean PSNR of P-frames: `b_gap_db = mean(P) - mean(B)`;
- the **period-N component**: the per-frame PSNR, less a running mean over
  `TREND_WINDOW` frames (so slow content change isn't counted), has its
  strongest period between 2 and `MAX_PERIOD` frames found by a periodogram;
  the series folded at that period gives `period_depth_db`, the highest
  phase mean less the lowest;
- the file size.

Q1's line (#1748, ruled 2026-11-26): on a real crisp clip, B-frames averaging
at least `THRESHOLD_DB` (1.0 dB) below P-frames, or a period-N dip of that
depth, on either of dw's two current paths (a, b), is **confirmed**. The
synthetic source is reported beside it but doesn't decide the line.

    python bench/encode_cadence.py run --out-dir DIR [--clip CLIP ...]
        [--no-synthetic] [--frames 96] [--fps 24]
    python bench/encode_cadence.py report DIR/encode-cadence.json

`run` writes the synthetic source as a lossless reference frame stack in
memory only, the four encodes of each source under DIR/<source>/, and
DIR/encode-cadence.json (the record's numbers). A CLIP is read read-only and
its decoded frames are the source: what is measured is the loss of one more
encode, which is what dw adds when it saves frames a pipeline produced.
`report` prints the record's table.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

THRESHOLD_DB = 1.0
MAX_PERIOD = 16
TREND_WINDOW = 9
FPS = 24
FRAMES = 96
WIDTH, HEIGHT = 640, 368
PAN_PX = 3
ENCODES = ("a_audio_path", "b_silent_path", "c_bf0", "d_crf18")
DW_PATHS = ("a_audio_path", "b_silent_path")
X264_OPTIONS = {"c_bf0": {"bf": "0"}, "d_crf18": {"crf": "18"}}

# ------------------------------------------------------------- the source


def synthetic_line_art(
    frames=FRAMES, width=WIDTH, height=HEIGHT, step=PAN_PX, seed=1748
):
    """A deterministic line-art pan: flat colour fills under hard black strokes,
    on a canvas wider than the frame, panned `step` px a frame to the right.

    Returns uint8 RGB frames, shape (frames, height, width, 3).
    """
    import cv2

    rng = np.random.default_rng(seed)
    canvas_w = width + step * max(frames - 1, 0) + 1
    canvas = np.full((height, canvas_w, 3), 236, np.uint8)
    palette = [
        (238, 201, 120),
        (120, 178, 222),
        (214, 104, 96),
        (132, 196, 140),
        (246, 232, 196),
        (176, 150, 210),
        (250, 250, 250),
    ]
    # Flat fills: polygons and circles in palette colours, outlined in black
    shapes = max(6, canvas_w // 40)
    for _ in range(shapes):
        colour = palette[int(rng.integers(len(palette)))]
        cx = int(rng.integers(0, canvas_w))
        cy = int(rng.integers(0, height))
        r = int(rng.integers(height // 10, height // 3))
        if rng.random() < 0.5:
            cv2.circle(canvas, (cx, cy), r, colour, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), r, (12, 12, 12), 3, cv2.LINE_AA)
        else:
            sides = int(rng.integers(3, 7))
            angles = np.sort(rng.uniform(0, 2 * np.pi, sides))
            radii = rng.uniform(0.5, 1.0, sides) * r
            pts = np.stack(
                [cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1
            ).astype(np.int32)
            cv2.fillPoly(canvas, [pts], colour, cv2.LINE_AA)
            cv2.polylines(canvas, [pts], True, (12, 12, 12), 3, cv2.LINE_AA)
    # Loose ink strokes and hatching on top
    for _ in range(shapes * 2):
        x0 = int(rng.integers(0, canvas_w))
        y0 = int(rng.integers(0, height))
        dx, dy = rng.integers(-80, 81, 2)
        cv2.line(
            canvas,
            (x0, y0),
            (x0 + int(dx), y0 + int(dy)),
            (12, 12, 12),
            int(rng.integers(1, 4)),
            cv2.LINE_AA,
        )
    return np.stack([canvas[:, i * step : i * step + width] for i in range(frames)])


def read_clip(path):
    """A clip's frames as uint8 RGB, (frames, height, width, 3), and its rate."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or FPS
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
    return np.stack(frames), float(rate)


def even_crop(frames):
    """Crop to even dimensions, which yuv420p needs, so every encode sees the
    same pixels."""
    h, w = frames.shape[1:3]
    return frames[:, : h - h % 2, : w - w % 2]


# ------------------------------------------------------------- the encodes


class _OptionsContainer:
    def __init__(self, container, options):
        self._container = container
        self._options = options

    def add_stream(self, codec, rate=None, **kwargs):
        if codec == "libx264":
            kwargs["options"] = {**kwargs.get("options", {}), **self._options}
        return self._container.add_stream(codec, rate=rate, **kwargs)

    def __getattr__(self, name):
        return getattr(self._container, name)


class _OptionsAv:
    def __init__(self, av, options):
        self._av = av
        self._options = options

    def open(self, *args, **kwargs):
        container = self._av.open(*args, **kwargs)
        if kwargs.get("mode", args[1] if len(args) > 1 else "r") != "w":
            return container
        return _OptionsContainer(container, self._options)

    def __getattr__(self, name):
        return getattr(self._av, name)


def encode(name, frames, fps, path):
    """Write `frames` to `path` the way encode `name` does."""
    from diffusers.utils import export_utils
    from PIL import Image

    from dw import writers

    pil = [Image.fromarray(f) for f in frames]
    if name == "b_silent_path":
        from dw.result import VIDEO_MACRO_BLOCK

        export_utils.export_to_video(
            pil, str(path), fps=fps, macro_block_size=VIDEO_MACRO_BLOCK
        )
        return
    options = X264_OPTIONS.get(name)
    if not options:
        writers.encode_video(pil, fps, output_path=str(path))
        return
    original = export_utils._import_av
    export_utils._import_av = lambda: _OptionsAv(original(), options)
    try:
        writers.encode_video(pil, fps, output_path=str(path))
    finally:
        export_utils._import_av = original


# FFmpeg's AVPictureType: PyAV 19 hands back the bare int, older PyAV an enum
_PICT_TYPES = {1: "I", 2: "P", 3: "B", 4: "S", 5: "SI", 6: "SP", 7: "BI"}


def _pict_type(value):
    name = getattr(value, "name", None)
    if name:
        return name
    return _PICT_TYPES.get(int(value), "?")


def decode(path):
    """Decoded uint8 RGB frames and each frame's picture type ("I", "P", "B")."""
    import av

    frames, types = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            types.append(_pict_type(frame.pict_type))
    return np.stack(frames), types


# ------------------------------------------------------------- the metrics


def psnr(ref, test):
    """RGB PSNR in dB, peak 255; identical frames give `inf`."""
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def _luma(rgb):
    rgb = rgb.astype(np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def ssim(ref, test):
    """SSIM on BT.601 luma, Gaussian window sigma 1.5."""
    import cv2

    x, y = _luma(ref), _luma(test)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2

    def blur(a):
        return cv2.GaussianBlur(a, (11, 11), 1.5)

    mx, my = blur(x), blur(y)
    vx = blur(x * x) - mx * mx
    vy = blur(y * y) - my * my
    cxy = blur(x * y) - mx * my
    num = (2 * mx * my + c1) * (2 * cxy + c2)
    den = (mx * mx + my * my + c1) * (vx + vy + c2)
    return float(np.mean(num / den))


def period_component(series, max_period=MAX_PERIOD, window=TREND_WINDOW):
    """The strongest short period in a per-frame series and its depth.

    The series less its running mean (`window` frames) is searched for its
    strongest period between 2 and `max_period` frames by periodogram; the
    detrended series folded at that period (rounded) gives `depth`, the
    highest phase mean less the lowest. Returns {"period", "depth", "power"};
    `power` is the peak's share of the detrended series' variance.
    """
    x = np.asarray(series, np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 6:
        return {"period": None, "depth": 0.0, "power": 0.0}
    w = min(window, n)
    trend = np.convolve(
        np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, "valid"
    )
    d = x - trend
    total = float(np.sum(d * d))
    if total == 0:
        return {"period": None, "depth": 0.0, "power": 0.0}
    t = np.arange(n)
    best = (None, 0.0)
    for k in range(1, n // 2 + 1):
        period = n / k
        if period < 2 or period > max_period:
            continue
        c = np.sum(d * np.cos(2 * np.pi * k * t / n))
        s = np.sum(d * np.sin(2 * np.pi * k * t / n))
        power = (c * c + s * s) * 2 / n
        if power > best[1]:
            best = (period, power)
    period, power = best
    if period is None:
        return {"period": None, "depth": 0.0, "power": 0.0}
    p = max(2, int(round(period)))
    phases = [d[i::p].mean() for i in range(p)]
    return {
        "period": round(float(period), 2),
        "depth": round(float(max(phases) - min(phases)), 3),
        "power": round(float(min(power / total, 1.0)), 3),
    }


def measure(source, decoded, types):
    """The per-encode numbers for decoded frames against the source."""
    n = min(len(source), len(decoded))
    if decoded.shape[1:3] != source.shape[1:3]:
        import cv2

        h, w = source.shape[1:3]
        decoded = np.stack(
            [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in decoded]
        )
    p = [psnr(source[i], decoded[i]) for i in range(n)]
    s = [ssim(source[i], decoded[i]) for i in range(n)]
    finite = [v for v in p if np.isfinite(v)]

    def mean_of(kind):
        vals = [p[i] for i in range(n) if types[i] == kind and np.isfinite(p[i])]
        return (round(float(np.mean(vals)), 3) if vals else None), len(vals)

    i_mean, i_count = mean_of("I")
    p_mean, p_count = mean_of("P")
    b_mean, b_count = mean_of("B")
    return {
        "frames": n,
        "psnr_mean": round(float(np.mean(finite)), 3) if finite else None,
        "psnr_min": round(float(np.min(finite)), 3) if finite else None,
        "ssim_mean": round(float(np.mean(s)), 5),
        "ssim_min": round(float(np.min(s)), 5),
        "types": "".join(t[:1] for t in types[:n]),
        "count": {"I": i_count, "P": p_count, "B": b_count},
        "psnr_by_type": {"I": i_mean, "P": p_mean, "B": b_mean},
        "b_gap_db": round(p_mean - b_mean, 3)
        if p_mean is not None and b_mean is not None
        else None,
        "period": period_component(p),
        "psnr": [round(v, 3) if np.isfinite(v) else None for v in p],
    }


def verdict(encodes, threshold=THRESHOLD_DB):
    """Q1's line on one source's encodes: confirmed when either dw path's
    B-gap or period depth reaches `threshold` dB."""
    hits = []
    for name in DW_PATHS:
        row = encodes.get(name)
        if not row:
            continue
        if (row["b_gap_db"] or 0) >= threshold:
            hits.append(f"{name} b_gap_db {row['b_gap_db']}")
        if row["period"]["depth"] >= threshold:
            hits.append(f"{name} period_depth {row['period']['depth']}")
    return {"confirmed": bool(hits), "hits": hits, "threshold_db": threshold}


# ------------------------------------------------------------- the run


def run_source(label, frames, fps, out_dir):
    frames = even_crop(frames)
    folder = Path(out_dir) / label
    folder.mkdir(parents=True, exist_ok=True)
    encodes = {}
    for name in ENCODES:
        path = folder / f"{name}.mp4"
        started = time.time()
        encode(name, frames, fps, path)
        decoded, types = decode(path)
        row = measure(frames, decoded, types)
        row["bytes"] = path.stat().st_size
        row["encode_seconds"] = round(time.time() - started, 2)
        row["file"] = str(path)
        encodes[name] = row
    h, w = frames.shape[1:3]
    return {
        "label": label,
        "frames": len(frames),
        "size": f"{w}x{h}",
        "fps": fps,
        "encodes": encodes,
        "verdict": verdict(encodes),
    }


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _versions():
    out = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "host": platform.node(),
    }
    for mod in ("av", "imageio_ffmpeg", "diffusers", "cv2", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import av

        out["libx264"] = str(av.codec.Codec("libx264", "w").long_name)
        out["ffmpeg_libs"] = {
            k: ".".join(map(str, v)) for k, v in av.library_versions.items()
        }
    except Exception:
        pass
    try:
        import imageio_ffmpeg

        out["imageio_ffmpeg_exe"] = imageio_ffmpeg.get_ffmpeg_version()
    except Exception:
        pass
    return out


def run(args, argv=None):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    if not args.no_synthetic:
        frames = synthetic_line_art(frames=args.frames)
        sources.append(run_source("synthetic", frames, args.fps, out_dir))
    for i, clip in enumerate(args.clip or []):
        frames, rate = read_clip(clip)
        record = run_source(f"clip{i + 1}", frames, rate, out_dir)
        record["clip"] = str(clip)
        record["clip_sha256"] = _sha256(clip)
        sources.append(record)
    record = {
        "probe": "encode-cadence",
        "issue": "#1748 stage A (#1752)",
        "date": time.strftime("%Y-%m-%d"),
        "command": " ".join(
            ["python bench/encode_cadence.py", *(argv or sys.argv[1:])]
        ),
        "versions": _versions(),
        "sources": sources,
    }
    path = out_dir / "encode-cadence.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(table(record))
    print(f"\nwrote {path}")


def table(record):
    lines = []
    for src in record["sources"]:
        name = src.get("clip", src["label"])
        lines.append(
            f"\n{src['label']}: {name} ({src['frames']} frames, {src['size']}, {src['fps']:g} fps)"
        )
        lines.append(
            "| encode | PSNR mean | PSNR min | SSIM mean | SSIM min | P PSNR | B PSNR | B gap dB "
            "| period | depth dB | I/P/B | bytes |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for enc, r in src["encodes"].items():
            bt = r["psnr_by_type"]
            c = r["count"]
            lines.append(
                f"| {enc} | {r['psnr_mean']} | {r['psnr_min']} | {r['ssim_mean']} | {r['ssim_min']} "
                f"| {bt['P']} | {bt['B']} | {r['b_gap_db']} | {r['period']['period']} "
                f"| {r['period']['depth']} | {c['I']}/{c['P']}/{c['B']} | {r['bytes']:,} |"
            )
        v = src["verdict"]
        state = "confirmed" if v["confirmed"] else "not confirmed"
        lines.append(
            f"Q1 line ({v['threshold_db']} dB, dw paths a/b): **{state}** {'; '.join(v['hits'])}"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out-dir", required=True)
    r.add_argument(
        "--clip", action="append", help="a real crisp clip (read-only); repeatable"
    )
    r.add_argument("--no-synthetic", action="store_true")
    r.add_argument("--frames", type=int, default=FRAMES)
    r.add_argument("--fps", type=float, default=FPS)
    rep = sub.add_parser("report")
    rep.add_argument("record")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        run(args, argv)
    else:
        print(table(json.loads(Path(args.record).read_text())))


if __name__ == "__main__":
    main()
