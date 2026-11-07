# Determinism check (2026-11-05)

> **Correction (2026-11-07, #1229).** The served worker did not read
> `enable_tf32`, `cudnn_benchmark` or `cudnn_deterministic` when these
> batches ran. Only `startup()` applied them, and `startup()` runs in the
> `dw.serve` parent, which runs no pipeline; each worker is a spawned process
> that never called it. So served jobs ran on PyTorch's own defaults
> (benchmark off, deterministic off, TF32 off) whatever `settings.json` said,
> unless something the worker imports set them, and the `cudnn` and `cudnn2`
> batches below cannot have got their result through those keys as this
> record says (*Answer*, *The suspects*). From #1229 the
> worker applies the keys itself, at start and at each job start, and the
> defaults are `cudnn_benchmark: false`, `cudnn_deterministic: true`,
> `enable_tf32: false`. The first job on a fresh worker narrates the flags it
> actually ran with; what lem reports is recorded on #1238. The record below
> stays as written.

The question from `.superpowers/sdd/2026-11-03-long-form-media/nondeterminism-report.md`:
why do same-seed, same-arguments runs on the bench server sometimes give a
different output? Records: `bench/records/determinism.json` (one row per job:
batch, setup, card, wall seconds, segment-1 and segment-2 hashes).

## Answer

- **Cause: cuDNN autotuning (`cudnn_benchmark: true`, the dw default).**
  cuDNN times its candidate algorithms and keeps the fastest. When the other
  card's worker is loading or running a job at the same time, the timings
  change, cuDNN can pick a different algorithm, and the numbers change with it.
  The picks are cached in the worker process, so the output then repeats bit
  for bit from job to job (the "persistent alternate" classes) until the
  process starts again.
- **The fix: `cudnn_benchmark: false` and `cudnn_deterministic: true`** in
  `~/.diffusers_helper/settings.json` (docs/ACCELERATION.md, *TF32 and
  cuDNN*). Both batches with this setting stayed on the canonical output, in
  the same job order on two cards that drifted with the default.
- **The cost: none that can be measured here.** A warm 2-segment control
  takes 195.2-195.4 s against 195.4-195.6 s with the default. The 6-segment
  arms take 629.5 s (backoff), 630.0 s (keystone) and 790.0 s (anchor 0.75),
  against 630.3, 630.8 and 791.0 s.
- **The trigger reproduced here is a second worker, not the job history on
  the first** (other host load fits the earlier cases; see *The earlier
  drifts*). On
  one card, 16 jobs in a row gave the canonical output (11 jobs including
  anchor 0.75, backoff and keystone arms, then a 2-minute anchor 0.25 run).
  With two cards, the first control after the cuda:1 worker loaded the model
  moved off canonical from frame 0.

## Method

- **The probe.** The S1 guide control at `segments: 2`, seed 1166 (226 video
  frames, 295 AAC frames). Every job has a fresh workflow-id suffix, so the
  step cache never serves one. The 6-segment arms are round 1's
  `r1-anchor75-s1`, `r1-backoff17-s1` and `r1-keystone-s1`. The 2-minute run
  is `r3-anchor25-s1-120s` (`duration_seconds: 120`, anchor 0.25).
- **The hashes.**
  - Segment 1 is `bench/segment1.py`'s fingerprint (video frames 0-59, AAC
    frames 0-129), taken with lem's ffmpeg 7.0.2 (the imageio-ffmpeg binary
    in the venv). The 2-segment control's segment 1 equals the 6-segment
    canonical `d59a1259/b2c2c63e`.
  - Segment 2 is the md5 of the decoded-frame framemd5 hashes from video
    frame 124 and AAC frame 166 (5.17 s, the seam) to the end. Canonical is
    `5c299afd/c6e90bd4`.
  - "First diff" is the first decoded video / AAC frame that differs from the
    canonical control.
- **Each batch starts on a fresh bench server** (`bench_server.sh stop`, then
  `start`), so every batch begins with a cold worker. The driver submits all
  of a batch's arms at once, and the server places each one on a free card.

## Results

| batch | setup | jobs | segment 1 | segment 2 | result |
|---|---|---|---|---|---|
| base | one card; defaults; SDNQ and Triton autotune logging on | c x5, anchor 0.75, c, backoff 17, c, keystone, c | all `d59a1259/b2c2c63e` | all 8 controls `5c299afd/c6e90bd4` | **reproducible** |
| long | one card; defaults | c, 2-minute anchor 0.25 (67 min), c, c | all canonical | all canonical | **reproducible** |
| dual | **two cards**; defaults | c (cuda:0, cold); anchor 0.75 (**cuda:1, cold load**); then c, backoff, c, keystone, c, anchor 0.75, c, c (all cuda:0) | c1 and the cuda:1 anchor run canonical; **every cuda:0 job after the cuda:1 load is `0a81547e/5f61a5f5`** | the 6 later controls are `4eb38d29/29cdd6df`, first diff at frame 0 / 0 | **drift reproduced**: it repeats bit for bit after it starts |
| nostream | two cards; transformer group_offload `use_stream`, `record_stream`, `low_cpu_mem_usage` all false | c (cuda:0) and anchor 0.75 (cuda:1) **started together**; then c, c | c1 canonical; **the anchor 0.75 cold load is `ff630564/5dfbe0da`**; the later controls are `b56226bd/ffb5374b` | c1 departs at segment 2 (video 86, audio 160, the x264 lookahead before the seam at 124); the later controls differ from frame 0 | **drift persists**, so stream offload is not the cause |
| cudnn | two cards; `cudnn_benchmark: false`, `cudnn_deterministic: true` | c, anchor 0.75 (cuda:1, cold), c, c | all canonical | all canonical | **reproducible** |
| cudnn2 | two cards; `cudnn_benchmark: false`, `cudnn_deterministic: true` | the dual batch's 10 jobs, placed on the same cards | all canonical | all 6 controls canonical | **reproducible** |

Both kinds of departure in the nondeterminism report showed up, and both had
another worker active at the time:

- The persistent drift in dual: the cuda:0 worker was idle while cuda:1 loaded
  and ran. The next cuda:0 job is about 4 s slower (199.6 s against 195.5 s),
  then runs at full speed and repeats its new output.
- In nostream, a control departed at segment 2 when the cuda:1 job started up
  next to it, and a cold load drifted from frame 0. A cold load is therefore
  not always canonical: it is canonical when nothing else runs while it
  loads and tunes.

The canonical output is the deterministic-algorithm output. With
`cudnn_deterministic: true`, every job reproduces the canonical hashes the
default gives on a quiet host. So on a quiet machine, the autotuner picks the
same algorithms the deterministic setting forces.

## The suspects

| suspect | how it is set | result |
|---|---|---|
| cuDNN benchmark (on) and cudnn_deterministic (off) | `~/.diffusers_helper/settings.json` keys, read once at worker start (`dw/settings.py`, `dw/__init__.py`) | **cause.** Turning both over gives canonical output under the same two-card order that drifts with the default |
| stream group offload | workflow: `components.transformer.group_offload.{use_stream, record_stream, low_cpu_mem_usage}` | **ruled out.** Drift persists with all three false, and turning them off costs **+59%** per warm control (310.7 s against 195.5 s) |
| SDNQ / Triton autotune | `cache_results=True`; `SDNQ_DEBUG_TRITON_AUTOTUNE=1` and `TRITON_PRINT_AUTOTUNING=1` in the worker environment | **ruled out for every drift after 2026-11-04 00:15 UTC.** The last autotune entry in `~/.triton/cache` was written then. The logged base batch tuned nothing, because every key was a cache hit, so each config is fixed per key across processes |
| sparse attention (`attention_backend: sparse`) | workflow variable | not run: drift appears and disappears with the cuDNN setting while sparse stays on |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | worker environment | not run, for the same reason |
| the job history on one worker (anchor, backoff, keystone, a 2-minute run) | | **ruled out.** 16 jobs on one card are all canonical |
| the cross-worker channel | | each worker sees only its own card (`CUDA_VISIBLE_DEVICES=0` or `1`), and nothing was written to disk while the drift began. What the two workers share is the host: CPU, PCIe, RAM and swap (4 GB of swap was in use after the dual batch). That changes timing, and timing is what the autotuner reads |

## Recommendation

- **Bench default: yes, now.** Set
  `{"devices": "cuda:0,cuda:1", "cudnn_benchmark": false, "cudnn_deterministic": true}`
  in `~/.diffusers_helper/settings.json` on lem, then restart the bench
  server. This file is shared with the main dw-serve. It was restored to
  `{"devices": "cuda:0,cuda:1"}` after these batches, because changing it
  changes the main server too. With the setting, two-card bench runs can be
  compared without segment-1 labels, and R35's rerun gap should collapse to
  zero for same-seed pairs. A canary control at the start and end of a batch
  is the cheap check that it held.
- **Engine default: recommend it, but measure first.** For H3 it costs
  nothing measurable: its convolutions are in the VAEs, and the transformer
  is matmul and attention. The default also has a real failure mode: any
  multi-card `dw.serve` gives timing-dependent output, and so does a single
  card next to other GPU work. Before flipping the default in
  `dw/settings.py`, time one conv-heavy family (an SD 1.5 or SDXL UNet, and
  the LTX VAE decode), where `benchmark` usually pays. If the cost is small
  there too, `cudnn_benchmark: false` and `cudnn_deterministic: true` should
  be the default, and speed becomes the opt-in.
- **What R35 still needs.** Records made before this setting keep their
  `clean` / `drifted` labels.

## The earlier drifts

The job list on lem shows when each earlier departure happened and what else
was running:

- **`cheap-a-s1` (cuda:1, 00:30-00:41 on 2026-11-04)** departed at
  segment 2 while cuda:0 ran `noise-b-s4`. That is the nostream case.
- **The round-2 drifts on cuda:1 (from 15:52)** and **round 3's second
  2-minute run** followed long jobs. Nothing ran on the other card at those
  times.
- **The first round-1 S1 drift (cuda:0, 06:39)** and the
  **`ctl-guide-s1-s1166-rerun` segment-2 departure (12:28)** had no job on
  the other card either.

All of those rounds ran the driver with `--measure`. That measure, the
server's `assess` and `compare` routes, decodes the finished output in the
server process while the next job starts and tunes. These determinism batches
ran without `--measure`. So "a second worker" is the trigger this check
reproduced, but "any other load on the host while a worker meets a new cuDNN
key" fits every case. Each case is consistent with the cause; only the
two-card one was rerun. The fix does not depend on which load it is, because
with the autotuner off, timing no longer selects algorithms.
