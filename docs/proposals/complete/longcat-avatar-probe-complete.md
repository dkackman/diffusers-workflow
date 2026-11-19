# LongCat-Video-Avatar 1.6 at INT8: a probe that failed its gate (#1443)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-19). Don filed the idea on 2026-11-18, from a ComfyUI wrapper's README
(Pixelailabs/ComfyUI-LongCat-Avatar-PixelAILabs, MIT). The model is
`meituan-longcat/LongCat-Video-Avatar-1.6`, which turns one picture and a voice
track into a lip-synced talking video, window by window. Its card claims the two
things the narrated long take lacked: stable lips and less error accumulation
over long continuations. The question was whether it runs on one 24 GB 3090 and
holds kf00's face past 30 s better than the H3 keeper on the same narration.
Plan v2 is the plan comment on #1443, with the full sweep. This file records
what was measured.

## Verdict: build (the limited probe), then stop

Don ruled a limited investigation: gate steps 1 and 2 only, outside dw, INT8
weights, and no engine, catalog, template or skill work. There was one probe
stage, with a 30 s gate before an 86 s take. **It fits the card but fails the
gate on identity and sharpness**, so there is no 86 s take and **no onboarding
plan**. Under Don's overnight delegation, the 2026-11-19 ruling took the
default: the LongCat-Avatar path stops here, with no 86 s take and no second
seed.

## Plan history

- **v1 → v2 (ruled 2026-11-18).** No scope change.
  - Q1 [spend]: stop if an extra 480p window takes over 6 min.
  - Q2 [spend]: moot, since 626 GB was free on `/`. The preflight `df` and the
    60 GB stop line stayed.
  - Q3 [none]: the gate to 86 s, at 30 s: SFace to kf00 ≥ .88 at every
    sample, saturation rise ≤ 5, 256 px Laplacian at 20-30 s ≥ 410, and at most
    3 still-mouth bins.
  - Q4 [none]: 480p only.
  - Also ruled: run only while `dw-serve` is idle, and stop at 90 GB host RSS.
- **Cut by the plan:** vendoring, onboarding, 720p, takes over 86 s, and a new
  lip-sync metric.
- **Ruled during the build (2026-11-19):** the gate failed, so stop. The
  spike's weights and venv stay on lem until Don says to delete them.

## What was built

### Stage A (#1465): INT8 probe on lem (bench only)

Commit `9bdded11`, merged to `develop` as `2c75bba2`. The gate script options
are in `fce10801`, and the record follow-up is `e57ebbef` (merged as
`74948813`). No deploy.

- **Spike:** `lem:~/spikes/longcat-avatar/`. Vendor `LongCat-Video` at
  `58eb336e`. Python 3.10, torch 2.6.0+cu124, flash_attn 2.7.4.post1 from the
  prebuilt wheel (no CUDA build), diffusers 0.35.1, transformers 4.41.0.
- **Downloads: about 46 GB,** not the planned 36. The LongCat-Video UMT5 text
  encoder is fp32 (22 GB). The INT8 parts only, from Avatar-1.6 @ `acfc6282`
  and LongCat-Video @ `03b55529`. Every weight is safetensors, except the
  vendor's ONNX vocal separator (64 MB).
- **`bench/longcat_avatar_probe.py`:** the CPU-staged driver, with
  `--offload blocks`.
- **`bench/ks2v_gate.py`:** gained `--crop256`, `--windows` and
  `--window-crop`. Calibrated on the H3 keeper, the Laplacian matches the
  narrated record; saturation and SFace read slightly higher, so every arm was
  compared within one run.
- **Record:** `bench/records/longcat-avatar.md`, with a pointer in
  `bench/records/narrated.md`.
- **On the server** (workspace `constitution-walk-3`), under
  `asset:uploads/bench/longcat-avatar/` (the plan wrote `bench/longcat-avatar/`;
  `upload_asset` adds `uploads/`): `kf00-30s-s1.mp4`, `kf00-30s-faces.jpg`,
  `kf00-30s-gate-faces.jpg` and `kf00-30s-ref-0.5s.png`.

**Fit.**
- The vendor demo OOMs at `pipe.to(cuda)`.
- Building every model on the CPU at once hit the RSS stop (90.9 GB).
- Loading in sequence with the DiT resident OOMs in window 1.
- **Block streaming fits:** the DiT blocks stay pinned on the host, and each is
  swapped onto the GPU for its own forward. The INT8 path stores int8 weights
  and dequantizes them in each forward, with bf16 matmuls (a 3090 has no INT8
  tensor-core path).

| 30 s kf00 take | |
|---|---|
| peak VRAM | 10.25 GB allocated (smi 13473 MiB) |
| peak host RSS | 80.2 GB (83.8 GB over the process tree) |
| per window | 255.9 s first, 255.4 s mean extra (stop line 360 s) |
| total | 10 windows, 45:08 wall |
| output | 832x480 at 25 fps |

An 86 s take would be about 27 windows, about 1 h 55 min.

**Gate** (kf00, seed 1, "a woman speaking to camera", 8 steps, `mix_steps` 4),
against the H3 keeper in the same `ks2v_gate.py` run:

| threshold | LongCat | H3 keeper | |
|---|---|---|---|
| SFace to kf00 ≥ .88 at 0.5/5/10/15/20/25/29.5 s | .93/.83/.89/.80/.90/.86/.89 | .93/.86/.92/.95/.93/.87/.91 | fails |
| saturation rise ≤ 5 | +3.9 | | passes |
| 256 px Laplacian at 20-30 s ≥ 410 | 254 | 537 | fails |
| still-mouth bins ≤ 3 | 0 of 62 | 0 of 60 | passes |

By eye it is the same woman, with scarf, earrings and background held. The skin
is smoother and the face darker than H3's, the eyelids get heavy at 5 s and
15 s, the mouth is very active, and the camera is static. The lips are the one
measure it wins, and it is coarse: still-mouth bins, not a sync metric.

**Bounces: 1.** The first hand-off left five items out of the build comment (the
`--include` patterns, file formats, which disk the preflight measured, that
`dw-serve` was idle, and which INT8 path ran). The re-hand-off added them, read
off lem and `list_jobs`. The tester then verified C-F1011 to C-F1018;
C-F1015's `analyze_face_drift` read within 0.01 of the record at every sample.

## Cost and bounces

| Stage | Bounces | Cost |
|---|---|---|
| #1465 | 1 (C-F1011, missing record items) | Session cost not recorded on the stage (plan estimate $4-7). GPU on lem: the fit attempts, then 45 min for the 30 s take. Download: about 46 GB |

## Deferred and open

- **No onboarding.** There is no LongCat video pipeline, template, catalog
  entry or skill, and nothing is vendored.
- **Disk:** the spike's weights (about 46 GB) and venv stay in
  `lem:~/spikes/longcat-avatar/` until Don says to delete them.
- **For any future avatar model on lem:** block streaming fits the card at
  ~10 GB, but host RAM is the limit (80 GB RSS with sequenced loading), so such
  a model can only run while `dw-serve` is idle.
- **The keyframe-free long talking take is still unsolved.** KS2V Lite, KS2V Pro
  and now LongCat-Avatar 1.6 all fail on the face before 30 s. Other routes are
  #1312, #1313 and #1188.
