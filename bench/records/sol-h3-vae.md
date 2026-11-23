# Sol-H3-VAE probe against dw's full H3 decoder (#1583, feature #1574)

Measured 2026-11-22 by `claude-opus-5-5` via `anthropic` in a build session. This is
measurement only. No engine code changed.

- **Hardware:** lem GPU 1, an RTX 3090 (24 GB, SM86), idle. The dw tree is `develop` @ bf5d341b.
- **Script:** `bench/sol_h3_vae_probe.py`. It runs as `python -I` from its own directory
  (`~/sol-probe/` on lem), with the downloaded Sol code kept in a separate directory.
- **Sol:** `Efficient-Large-Model/Sol-H3-VAE`, revision
  `6a8a33fc61e092281f32bf28cde236760e2caaa4`. The weights file `sol_h3_vae.safetensors`
  is 6,471,256,328 bytes, sha256
  `8658f1e85b5997c4ba716406a2f148bfa36cf97d4d5fcd402c7bd341f54e77a1`.
- **Full decoder:** dw's own `DwH3StreamedVideoDecodeStep` + `streamed_decode` (8 latents
  per stream step), at MiniMax-H3 snapshot `0919fb68…`.
- **Latents:** each input mp4 is encoded once with `encode_h3_latents`. Every decoder
  then decodes the same latents.
- **Timing:** one warm-up, then the best of N timed runs: N=2 at 544p and 768p, and N=1
  at 1080p, for the 26 s take and for the 544p noise rerun. Peak is
  `torch.cuda.max_memory_allocated` during the decode, and resident is what stays
  allocated between decodes.
- **PSNR:** computed on uint8 RGB over the whole clip against the full fp32 decode
  (`full-fp32`, today's decode path), with per-frame min/median/max in the JSON.

## Variants

| name | what |
|---|---|
| `full-fp32` | dw's full H3 decoder in fp32: today's decode, and the quality reference |
| `full-bf16` | the same decoder with bf16 weights |
| `full-int8` | the same decoder, SDNQ int8 (conv + matmul), weights parked on the CPU and moved to the GPU for the decode (on-demand); the time includes the move |
| `sol` | Sol `engine="ref"` (bf16 autocast, fp32 weights), whole clip in one call |
| `sol-bf16w` | `sol` with the transformer blocks cast to bf16 (3.03 GiB resident instead of 6.03). A full cast breaks `x_embedder`, which takes fp32 input outside autocast |
| `sol-piece` | Sol in pieces of `--piece-clips` 5-latent blocks, cut naively |
| `sol-piece-ctx` | the same pieces with ±1 block (5 latents) of context, decoded and then dropped |

## Results

### 544p (960x544, 124 frames, 37 latents)

| decode | s | resident GiB | peak GiB | PSNR vs full-fp32 |
|---|---:|---:|---:|---:|
| full-fp32 | 77.9 | 9.71 | 11.90 | (reference) |
| full-bf16 | 16.6 | 4.86 | 6.68 | 50.5 |
| full-int8 on-demand | 18.2 | 0.01 | 4.77 | 50.0 |
| sol | **5.32** | 6.03 | 19.79 | 32.7 (frames 30.6–35.2) |
| sol-bf16w | 5.21 | 3.03 | 15.64 | 32.7 |

Pieces against whole, 544p, `--piece-clips 2` (cuts at frames 34 and 68):

| decode | s | peak GiB | PSNR vs sol whole | max abs (0–255) |
|---|---:|---:|---:|---:|
| sol-piece (naive) | 5.32 | 10.17 | 43.05 | 91, worst at frames 30–34 and 64–68 (the cuts) |
| sol-piece-ctx | 9.40 | 13.77 | **57.85** | 11 |
| *noise floor:* sol-bf16w vs sol | | | 56.35 | 6 |

The noise floor is the same Sol model with bf16 blocks in place of fp32 blocks. Its
difference from Sol is what bf16 arithmetic alone moves.

### 768p (1344x768, 124 frames)

| decode | s | peak GiB | PSNR vs full-fp32 |
|---|---:|---:|---:|
| full-fp32 | 145.7 | 14.02 | (reference) |
| full-bf16 | 30.6 | 8.45 | 53.6 |
| full-int8 on-demand | 33.1 | 6.54 | 52.8 |
| sol, whole | **OOM** (asked for 4.55 GiB more at 20.4 GiB allocated) | | |
| sol-bf16w, whole | **OOM** | | |
| sol-piece, 2 blocks | 9.83 | 13.61 | 39.9 |
| sol-piece-ctx, 2 blocks | 17.5 | 20.35 | 40.4 |

### 1080p (1920x1088, 124 frames; `--sol-bf16-weights`)

| decode | s | peak GiB | PSNR vs full-fp32 |
|---|---:|---:|---:|
| full-fp32 | 312.9 | 18.43 | (reference) |
| full-int8 on-demand | 68.4 | 10.20 | 52.4 |
| sol-piece, 1 block | 20.4 | 10.32 | 37.5 (frames 35.9–39.8) |
| sol-piece-ctx, 1 block | **OOM** | | |

### 26 s take (960x544, 634 frames, ~187 latents; Sol whole cannot fit)

| decode | s | peak GiB | PSNR vs full-fp32 |
|---|---:|---:|---:|
| full-fp32 | 413.2 | 12.42 | (reference) |
| full-int8 on-demand | 90.0 | 4.97 | 51.2 |
| sol-piece, 4 blocks | 26.7 | 13.78 | 35.0 (frames 31.8–40.7) |
| sol-piece-ctx, 4 blocks | 38.0 | 17.38 | 35.2 (frames 32.3–40.7) |

## Against the plan's bar

The bar is measured against today's lowest-memory full decode, int8 on-demand. fp32
figures appear above for reference.

| bar | result |
|---|---|
| ≥ 2.5x faster at 544p | **pass:** 3.4x (5.32 s vs 18.2 s), whole clip |
| ≥ 2.5x faster at 768p | **fail on a correct decode.** Whole OOMs. Pieces with context, the only seam-free way, are 1.9x (17.5 s vs 33.1 s). Naive pieces are 3.4x, but they leave seams (see the next row) |
| 1080p peak ≤ today + 1.5 GB | **pass only with naive one-block pieces:** 10.32 GiB vs 10.20 GiB. Whole and context pieces OOM on 24 GB |
| piecewise = whole within noise | **pass with context** at 544p: 57.85 dB / max 11 vs a 56.35 dB / max 6 floor. **Fail naive:** 43 dB, max 91 at the cuts. This is only measurable at 544p, the one size where whole fits |

No single Sol configuration on this card meets all three at once.
- A seam-free decode (context pieces) misses the 768p speed bar and OOMs at 1080p.
- A configuration fast and small enough (naive pieces) fails the seam check.

## Quality (Don's taste verdict)

Sol sits 32.7–40 dB from the full decoder. The full decoder's own bf16 and int8 variants
sit 50–53 dB from it. Sol is a distinct, lossier decoder, not a faster run of the same
one: its error is 10–30× our int8 error in MSE terms. Sol's README claims 38–45
dB. We get 32.7 at 544p, 37.5 at 1080p, 35.0 on the 26 s take and 39.9 at 768p.

These grids are q85 JPEGs, small enough for the repo. The full-resolution PNGs are on
lem in `~/sol-probe/out/`.

- `sol-h3-vae/544p-grid.jpg`, `768p-grid.jpg`, `1080p-grid.jpg`, `26s-grid.jpg`.
  - Rows: full-fp32, full-int8, then Sol. That is `sol` at 544p, `sol-piece-ctx` at 768p
    and on the 26 s take, and `sol-piece` at 1080p.
  - Columns: frames 0, F/4, F/2, 3F/4, F-1.
- `sol-h3-vae/544p-crop.jpg` and `768p-crop.jpg`: a 1:1 crop at the middle frame, in the
  same row order.
- `sol-h3-vae/544p-p2-seam.jpg`, `1080p-seam.jpg` and
  `26s-seam.jpg`: the four frames around the first cut.
  - At 544p the rows are `sol`, `sol-piece`, `sol-piece-ctx`.
  - Where whole Sol OOMed, the rows are full-int8, `sol-piece`, `sol-piece-ctx`.

Content: the 544p take is a bronze statue against foliage, the 768p take a montage shot,
the 1080p take GateArmD and the 26 s take a glass octopus. The 26 s take gives gloss and
transparency. **None of these four takes carries a close face or on-screen text.** Those
two checks are not covered (see Deviations).

## Deviations from the plan

1. **The weights are 6.47 GB of fp32, not ~3 GB.** Casting the blocks to bf16 gets the
   resident size to 3.03 GiB.
2. **Ref engine only.** Sol's fast path needs FlashAttention-4, which does not run on SM86
   (Ampere). Every number here is the ref engine. Hopper or Blackwell cards would differ.
3. **Whole-clip Sol decode OOMs at 768p, at 1080p and on the 26 s take on 24 GB.**
   - Ref-engine memory is about 184 KB per latent token plus the weights.
   - So the plan's "decode whole" was run at 544p only. The larger cases ran as pieces.
   - The 26 s take was cut into 4-block pieces, not 8: 8 blocks would need about 21 GB
     of activations.
4. **Piecewise = whole was measured at 544p with 2-block pieces.** That is the only size
   where whole fits and a cut exists. The 26 s comparison is naive vs context pieces,
   each against full-fp32. The 26 s take is 634 frames.
5. **Faces and on-screen text are not in the grids.** No existing H3 output on lem at
   these sizes carried them.
6. **Sol's PSNR is below its own README** (32.7–40 dB vs 38–45 dB).

## Reproduce

On lem, with Sol code and weights fetched at the pinned revision into their own
directories:

    CUDA_VISIBLE_DEVICES=1 DW_QUANTIZED_CACHE=1 ~/diffusers-workflow/venv/bin/python -I \
      bench/sol_h3_vae_probe.py --repo ~/diffusers-workflow \
      --sol-code <sol>/code --sol-weights <sol>/sol_h3_vae.safetensors \
      --vae-config <H3 snapshot>/vae/config.json \
      --input <mp4> --label 544p --out <dir> \
      [--variants ...] [--piece-clips N] [--sol-bf16-weights] [--runs N]

The JSON for each label is in `bench/records/sol-h3-vae/<label>.json`.
