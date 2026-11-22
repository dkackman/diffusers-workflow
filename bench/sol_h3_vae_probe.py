"""Sol-H3-VAE against dw's full H3 video decoder, for #1583 (stage 1 of #1574; outside dw).

A measurement probe, not engine code. For each finished H3 take it encodes one latent
with dw's `encode_h3_latents`, then decodes that latent with:

- `full-fp32`: dw's full H3 decoder as `decode_h3_latents` loads it today (float32,
  resident), through dw's `streamed_decode` in 8-clip pieces;
- `full-bf16`: the same VAE cast to bfloat16, resident;
- `full-int8`: the VAE quantized as `workflows/templates/minimax/video-with-audio.json`
  quantizes it (SDNQ int8, return_device cpu), moved onto the card for the decode and
  off again (the template's `residency: on_demand`); the move is timed apart;
- `sol` / `sol-bf16w`: Sol-H3-VAE, `engine="ref"` (plain SDPA under bf16 autocast),
  whole, with the shipped fp32 weights or the weights cast to bf16;
- `sol-piece` / `sol-piece-ctx`: Sol in pieces of `--piece-clips` native blocks, cut
  naively, or with one block of context each side that is decoded and dropped.

Each decode reports wall time (one warm-up, then `--runs` timed runs, synchronized),
the weights resident on the card, the peak allocated and reserved memory during the
decode, and PSNR (uint8 RGB, all frames) against `full-fp32`; the pieces also against
`sol` whole. Frame grids (full | sol, matched frames) are written as PNG. Prints
RECORD lines and writes `<out>/<label>.json`.

The Sol code and weights are third-party: keep them in their own directory and run
this with `python -I`, the repo and the Sol code put on sys.path by flag.

usage: CUDA_VISIBLE_DEVICES=1 python -I sol_h3_vae_probe.py --repo ~/diffusers-workflow
           --sol-code ~/sol-probe/hf/code --sol-weights .../sol_h3_vae.safetensors
           --input take.mp4 --label 544p --out ~/sol-probe/out
           --variants full-fp32,full-bf16,full-int8,sol,sol-bf16w,sol-piece,sol-piece-ctx
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch

BLOCK = 5  # latents per native H3 block
BLOCK_FRAMES = 17  # frames a full block decodes to

ALL_VARIANTS = (
    "full-fp32",
    "full-bf16",
    "full-int8",
    "sol",
    "sol-bf16w",
    "sol-piece",
    "sol-piece-ctx",
)


def log(*parts):
    print(*parts, flush=True)


def record(label, variant, **fields):
    log("RECORD", json.dumps({"label": label, "variant": variant, **fields}))


def gib(n):
    return round(n / 2**30, 3)


def free():
    gc.collect()
    torch.cuda.empty_cache()


def psnr(a, b):
    """PSNR in dB of two uint8 (F, H, W, 3) CPU tensors, over every frame."""
    frames = min(a.shape[0], b.shape[0])
    total = 0.0
    count = 0
    for i in range(0, frames, 8):
        d = a[i : i + 8].float() - b[i : i + 8].float()
        total += float((d * d).sum())
        count += d.numel()
    mse = total / count
    return float("inf") if mse == 0 else 10 * math.log10(255.0**2 / mse)


def max_abs(a, b):
    frames = min(a.shape[0], b.shape[0])
    worst = 0
    for i in range(0, frames, 8):
        d = (a[i : i + 8].int() - b[i : i + 8].int()).abs().max()
        worst = max(worst, int(d))
    return worst


def per_frame_psnr(a, b):
    out = []
    for i in range(min(a.shape[0], b.shape[0])):
        d = a[i].float() - b[i].float()
        mse = float((d * d).mean())
        out.append(99.0 if mse == 0 else 10 * math.log10(255.0**2 / mse))
    return out


def timed(fn, runs):
    """One warm-up, then `runs` timed runs; returns (result, [seconds], peak bytes)."""
    result = fn()
    del result
    free()
    times = []
    peak_alloc = 0
    peak_reserved = 0
    result = None
    for _ in range(runs):
        if result is not None:
            del result
            free()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated())
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
    return result, times, peak_alloc, peak_reserved


# dw's full decoder


def load_full(repo, kind):
    from dw.pipeline_processors.h3_windows import DwH3StreamedVideoDecodeStep
    from dw import quantized_cache

    pipeline = DwH3StreamedVideoDecodeStep().init_pipeline(repo)
    if kind == "int8":
        from sdnq import SDNQConfig

        config = SDNQConfig(
            weights_dtype="int8",
            quant_conv=True,
            use_quantized_matmul_conv=True,
            quantization_device="cuda",
            return_device="cpu",
            dequantize_fp32=False,
        )
        quantized_cache.load_components(
            pipeline,
            {
                "names": ["vae"],
                "torch_dtype": torch.bfloat16,
                "quantization_config": {"vae": config},
            },
        )
        pipeline.vae.to("cpu")
    else:
        dtype = torch.float32 if kind == "fp32" else torch.bfloat16
        quantized_cache.load_components(
            pipeline, {"names": ["vae"], "torch_dtype": dtype}
        )
        pipeline.vae.to(device="cuda", dtype=dtype)
    return pipeline


def full_decode(pipeline, z, dtype):
    """dw's streamed decode of normalized latents z (1, 24, T, H, W): uint8 (F, H, W, 3) on the CPU."""
    from dw.pipeline_processors.h3_hold_steps import streamed_decode

    vae = pipeline.vae
    config = vae.config
    mean = torch.tensor(config.latents_mean, device="cuda").view(1, -1, 1, 1, 1)
    std = torch.tensor(config.latents_std, device="cuda").view(1, -1, 1, 1, 1)
    pmean = torch.tensor(pipeline.pixel_mean, device="cuda").view(1, -1, 1, 1, 1)
    pstd = torch.tensor(pipeline.pixel_std, device="cuda").view(1, -1, 1, 1, 1)
    latents = (z.to("cuda", torch.float32) * std + mean).to(dtype)
    frames = []

    def finish(video):
        video = (video.float() * pstd + pmean).clamp_(0, 1).mul_(255).round_()
        frames.append(video.to(torch.uint8)[0].permute(1, 2, 3, 0).cpu())

    with torch.no_grad():
        streamed_decode(vae, latents, finish)
    return torch.cat(frames)


# Sol-H3-VAE


def load_sol(config_path, weights, bf16_weights):
    from sol_h3_vae import SolH3VAEDecoder

    decoder = SolH3VAEDecoder(config_path, weights, engine="ref", device="cuda")
    if bf16_weights:
        # Only the blocks: the embedder and the head take float32 inputs
        # outside the autocast, and the model casts around its norms itself
        decoder.model.blocks.to(torch.bfloat16)
    return decoder


def sol_rgb(decoder, z_sol):
    with torch.no_grad():
        return decoder.decode_rgb_uint8(z_sol).cpu()


def sol_whole(decoder, z):
    """z: dw's normalized (1, 24, T, H, W); Sol takes (1, T, 24, H, W)."""
    return sol_rgb(decoder, z.transpose(1, 2).contiguous().to("cuda"))


def sol_pieces(decoder, z, piece_clips, context):
    """Sol over runs of `piece_clips` native blocks. With `context`, each piece is
    decoded with one more block on either side (where there is one) and the frames
    those blocks gave are dropped; every piece still starts on a block."""
    z_sol = z.transpose(1, 2).contiguous()
    total = z_sol.shape[1]
    piece = piece_clips * BLOCK
    frames = []
    start = 0
    while start < total:
        stop = min(start + piece, total)
        lo = max(start - BLOCK, 0) if context else start
        hi = min(stop + BLOCK, total) if context else stop
        rgb = sol_rgb(decoder, z_sol[:, lo:hi].to("cuda"))
        # a full block gives 17 frames; only the take's last block is ever partial
        skip = ((start - lo) // BLOCK) * BLOCK_FRAMES
        keep = ((stop - start) // BLOCK) * BLOCK_FRAMES
        if stop == total:
            frames.append(rgb[skip:])
        else:
            frames.append(rgb[skip : skip + keep])
        del rgb
        start = stop
    return torch.cat(frames)


# Grids


def grid_frames(count):
    picks = [0, count // 4, count // 2, (3 * count) // 4, count - 1]
    return sorted(set(picks))


def write_grid(path, rows, frames, scale):
    """rows: [(name, uint8 (F, H, W, 3))]; one row per decode, one column per frame."""
    tiles = []
    for _, video in rows:
        row = [video[i].numpy() for i in frames]
        tiles.append(np.concatenate(row, axis=1))
    image = PIL.Image.fromarray(np.concatenate(tiles, axis=0))
    if scale != 1:
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            PIL.Image.LANCZOS,
        )
    image.save(path, optimize=True)


def write_crops(path, rows, frame, box):
    """Full-resolution crops of one frame, side by side: the detail check."""
    x, y, w, h = box
    tiles = [video[frame, y : y + h, x : x + w].numpy() for _, video in rows]
    PIL.Image.fromarray(np.concatenate(tiles, axis=1)).save(path, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="diffusers-workflow checkout")
    parser.add_argument("--sol-code", required=True, help="dir holding sol_h3_vae/")
    parser.add_argument("--sol-weights", required=True)
    parser.add_argument("--vae-config", required=True, help="the H3 vae/config.json")
    parser.add_argument("--h3", default="MiniMaxAI/MiniMax-H3")
    parser.add_argument("--input", required=True, help="a finished H3 take")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variants", default=",".join(ALL_VARIANTS))
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--piece-clips", type=int, default=8)
    parser.add_argument(
        "--sol-bf16-weights",
        action="store_true",
        help="every Sol variant with its blocks' weights in bf16",
    )
    parser.add_argument(
        "--latents-from", default="", help="reuse another label's encoded latent"
    )
    parser.add_argument("--grid-scale", type=float, default=0.5)
    parser.add_argument(
        "--crop", default="", help="x,y,w,h of a full-resolution detail crop"
    )
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(os.path.expanduser(args.repo)))
    sys.path.insert(0, os.path.abspath(os.path.expanduser(args.sol_code)))
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    variants = [v for v in args.variants.split(",") if v]
    unknown = set(variants) - set(ALL_VARIANTS)
    if unknown:
        parser.error(f"unknown variants: {sorted(unknown)}")
    log(
        "device",
        torch.cuda.get_device_name(0),
        "torch",
        torch.__version__,
        "capability",
        torch.cuda.get_device_capability(0),
    )

    # Encode once, and keep the latent: every decode reads the same one
    latent_path = out / f"{args.latents_from or args.label}.latents.pt"
    if latent_path.exists():
        encoded = torch.load(latent_path)
    else:
        from dw.tasks.h3_latent_upscale import encode_h3_latents

        start = time.perf_counter()
        # A trusted local file, read as dw reads a realized video argument
        from dw.tasks.video_utils import VideoFileReference

        source = VideoFileReference(os.path.abspath(os.path.expanduser(args.input)))
        encoded = encode_h3_latents(source, device="cuda", model_name=args.h3)
        encoded = {k: v for k, v in encoded.items()}
        encoded["latents"] = encoded["latents"].cpu()
        torch.save(encoded, latent_path)
        log("encoded in", round(time.perf_counter() - start, 1), "s")
        from dw.tasks.model_cache import clear_model_cache

        clear_model_cache()
    z = encoded["latents"].to(torch.float32)
    log("latents", list(z.shape), "source_frames", encoded["source_frames"])
    free()

    results = {
        "label": args.label,
        "input": args.input,
        "latents": list(z.shape),
        "source_frames": encoded["source_frames"],
        "device": torch.cuda.get_device_name(0),
        "piece_clips": args.piece_clips,
        "sol_bf16_weights": args.sol_bf16_weights,
        "variants": {},
    }
    decoded = {}

    def keep(name, video, times, peak, reserved, resident, **extra):
        fields = {
            "frames": int(video.shape[0]),
            "size": [int(video.shape[2]), int(video.shape[1])],
            "seconds": [round(t, 3) for t in times],
            "best_s": round(min(times), 3),
            "resident_gib": gib(resident),
            "peak_alloc_gib": gib(peak),
            "peak_reserved_gib": gib(reserved),
            **extra,
        }
        if "full-fp32" in decoded and name != "full-fp32":
            fields["psnr_vs_full_fp32"] = round(psnr(video, decoded["full-fp32"]), 2)
            pf = sorted(per_frame_psnr(video, decoded["full-fp32"]))
            fields["frame_psnr_vs_full_fp32"] = {
                "min": round(pf[0], 2),
                "median": round(pf[len(pf) // 2], 2),
                "max": round(pf[-1], 2),
            }
        if name.startswith("sol-") and "sol" in decoded:
            fields["psnr_vs_sol_whole"] = round(psnr(video, decoded["sol"]), 2)
            fields["max_abs_vs_sol_whole"] = max_abs(video, decoded["sol"])
            pf = per_frame_psnr(video, decoded["sol"])
            fields["worst_frames_vs_sol_whole"] = sorted(
                ((round(p, 2), i) for i, p in enumerate(pf))
            )[:6]
        results["variants"][name] = fields
        record(args.label, name, **fields)
        decoded[name] = video

    for name in variants:
        free()
        log("--", name)
        try:
            if name.startswith("full-"):
                kind = name.split("-")[1]
                pipeline = load_full(args.h3, kind)
                dtype = torch.float32 if kind == "fp32" else torch.bfloat16
                if kind == "int8":
                    move_times = []

                    def run():
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        pipeline.vae.to("cuda")
                        torch.cuda.synchronize()
                        move_times.append(time.perf_counter() - t0)
                        video = full_decode(pipeline, z, dtype)
                        pipeline.vae.to("cpu")
                        return video

                    resident = torch.cuda.memory_allocated()
                    video, times, peak, reserved = timed(run, args.runs)
                    keep(
                        name,
                        video,
                        times,
                        peak,
                        reserved,
                        resident,
                        move_s=[round(t, 3) for t in move_times[1:]],
                        note="weights on the CPU between decodes; seconds include the move on",
                    )
                else:
                    resident = torch.cuda.memory_allocated()
                    video, times, peak, reserved = timed(
                        lambda: full_decode(pipeline, z, dtype), args.runs
                    )
                    keep(name, video, times, peak, reserved, resident)
                del pipeline
            else:
                decoder = load_sol(
                    args.vae_config,
                    args.sol_weights,
                    name == "sol-bf16w" or args.sol_bf16_weights,
                )
                resident = torch.cuda.memory_allocated()
                if name in ("sol", "sol-bf16w"):
                    fn = lambda: sol_whole(decoder, z)  # noqa: E731
                else:
                    ctx = name == "sol-piece-ctx"
                    fn = lambda: sol_pieces(decoder, z, args.piece_clips, ctx)  # noqa: E731
                video, times, peak, reserved = timed(fn, args.runs)
                keep(name, video, times, peak, reserved, resident)
                del decoder
        except torch.OutOfMemoryError as e:
            free()
            results["variants"][name] = {"oom": str(e).splitlines()[0]}
            record(args.label, name, oom=str(e).splitlines()[0])
        free()

    (out / f"{args.label}.json").write_text(json.dumps(results, indent=2))

    rows = [(n, decoded[n]) for n in ("full-fp32", "full-int8", "sol") if n in decoded]
    if len(rows) >= 2:
        count = min(v.shape[0] for _, v in rows)
        write_grid(
            out / f"{args.label}-grid.png", rows, grid_frames(count), args.grid_scale
        )
        if args.crop:
            box = [int(v) for v in args.crop.split(",")]
            write_crops(out / f"{args.label}-crop.png", rows, count // 2, box)
    pieces = [n for n in ("sol", "sol-piece", "sol-piece-ctx") if n in decoded]
    edge = args.piece_clips * BLOCK_FRAMES
    if len(pieces) >= 2 and "sol-piece" in decoded and edge < decoded["sol"].shape[0]:
        # The frames either side of the first piece boundary
        edge = args.piece_clips * BLOCK_FRAMES
        frames = [
            f
            for f in (edge - 2, edge - 1, edge, edge + 1)
            if f < decoded["sol"].shape[0]
        ]
        write_grid(
            out / f"{args.label}-seam.png",
            [(n, decoded[n]) for n in pieces],
            frames,
            args.grid_scale,
        )
    log("done", out / f"{args.label}.json")


if __name__ == "__main__":
    main()
