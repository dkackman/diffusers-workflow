# MiniMax-H3 hybrid transformer_ref: fl2va + ref2va adaln_proj b20-49 (#1441, stage #1463)

Lem, RTX 3090 24 GB, 2026-11-19. Snapshot `42ed227ee7df40d41602854ae760620d6eb651fe`.

## Question

H3 ships two transformers of one architecture: `transformer` (fl2va) and
`transformer_ref` (ref2va). Does a ref2va shot get fl2va's look without losing
the reference's identity, if `transformer_ref` is fl2va everywhere except the
late blocks' `adaln_proj` (the timestep modulation), taken from ref2va?

Floor (plan v1, Q2): SFace median within 0.02 of stock, or better. Above the
floor, Don's look decides quality.

## Method

1. `bench/h3_hybrid_graft.py FL2VA REF2VA ~/bench-scratch/h3-hybrid-b20-49`
   (default `--first 20 --last 49`). It reads both parents lazily, asserts
   identical tensor names and shapes, and writes fl2va's shards with
   `transformer_blocks.{20..49}.adaln_proj.*` swapped in from ref2va, plus
   ref2va's `config.json` (the two configs are byte-identical anyway).
2. Two arms, one shot, run inline on the same server, identical apart from
   one key:
   - stock: `bench/workflows/probe-1441-stock-ref.json`;
   - hybrid: `bench/workflows/probe-1441-hybrid-ref.json`, which adds
     `load_components.pretrained_model_name_or_path.transformer_ref` set to the graft.

   Both are the shipped `templates/minimax/reference-to-video` (ref2v 8-step
   turbo LoRA, 9 steps, shift 12/3, int4 SDNQ, sparse attention, 960x544,
   124 frames). The shot uses subject `asset:cast/estela.jpg`, the template's
   voice clip, a garden to-camera prompt, and seed 1441.
3. SFace: `analyze_face_drift` (the `bench/workflows/t26-face-check.json` shape,
   16 samples) of each clip against `asset:cast/estela.jpg`.

### Graft report

| | |
|---|---|
| fl2va | `.../snapshots/42ed227e.../transformer` |
| ref2va | `.../snapshots/42ed227e.../transformer_ref` |
| out | `/home/don/bench-scratch/h3-hybrid-b20-49/transformer_ref` |
| blocks | 20-49 |
| tensors | 638 (names and shapes identical across parents) |
| from ref2va | 60: `transformer_blocks.{20..49}.adaln_proj.linear.{weight,bias}` |
| ref2va-only tensors | none |
| config | ref2va |
| total bytes | 66,280,430,080 (bf16, 14 shards) |

## Results

| Arm | Job | Wall (waited) | Load | Denoise (8 passes) | Decode | Host peak | SFace median | SFace min | Faces |
|---|---|---|---|---|---|---|---|---|---|
| stock | `7cd603a1922f` | 247.7 s | 5.3 s (cached) | 185.0 s (22.6 s/pass) | 19.8 s | 23.8 GB | **0.855** | 0.759 | 16/16 |
| hybrid b20-49 | `d5a6c03f9ee3` | 308.8 s | 48.8 s (uncached, fresh quantize) | 205.4 s (25.7 s/pass) | 22.0 s | 40.2 GB | **0.788** | 0.561 | 16/16 |

SFace jobs: stock `d7589d81e015`, hybrid `3a77473b74fa` (reference `cast/estela.jpg`).

Outputs (workspace `bench`):
- stock: `BenchProbe1441StockRef/20261119-075335-e5a85485/final/BenchProbe1441StockRef-reference_to_video_audio.0-0.0.mp4`
- hybrid: `BenchProbe1441HybridRef/20261119-075839-d44b817f/final/BenchProbe1441HybridRef-reference_to_video_audio.0-0.0.mp4`

**SFace: the hybrid misses the floor.** Its median is 0.067 below stock
(allowed: 0.02), and its minimum falls 0.20. Identity drifts across the clip.

Contact sheet (4 frames each). Both clips are coherent: the same character,
costume, garden and push-in. The hybrid is more saturated and contrasty, with
a warmer and stronger backlight, and the face reads a little more stylised
(larger eyes, sharper features). The stock clip stays closer to the portrait.
Whether the hybrid's look is a gain is Don's call.

The extra 7.3 s/pass and 43 s of load are the uncached local path (a fresh
SDNQ quantize, and no pinned cache). They are not a property of the graft.
The 40 GB host peak is the same quantize-from-bf16 load.

## b25-49

The plan runs b25-49 only if b20-49 shows a quality gain but loses identity.
Identity is lost; the quality gain is Don's verdict, so b25-49 was not run.
If Don sees a gain, rebuild the graft with `--first 25` (about a minute) and
run the hybrid probe again with the path changed.

## Deviations from the plan

- The stock arm is its own file (`probe-1441-stock-ref.json`), not the
  template run with arguments. This keeps the two arms diffable as JSON:
  `get_job_workflow` on the two jobs differs only in the
  `pretrained_model_name_or_path` key (and the id, summary and description).
- The subject is `cast/estela.jpg` with a to-camera garden prompt, not the
  template's default reference. This gives a large, frontal face for SFace.
- Both arms ran as `inline_workflow`. `run_workflow(workflow_path=...)` is
  refused for a file the server's catalog can't reach, and the probes live
  in the repo's `bench/`.
- With the local override, `validate_workflow` can list
  `MiniMaxAI/MiniMax-H3` (0.01 GB) in `downloads_required`, for the other
  components. The run's own cost ack had `downloads: []`, and nothing was
  fetched.
- The scratch graft (`/home/don/bench-scratch/h3-hybrid-b20-49`, 62 GB) was
  deleted after this record, as plan v1 says. The script rebuilds it in
  about a minute.

## Verdict

_For Don:_ ______ (SFace says identity is lost at b20-49: 0.788 vs 0.855.
Is the hybrid's look a gain worth trying b25-49 for, or drop the idea?)
