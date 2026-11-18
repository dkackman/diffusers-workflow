# Kandinsky S2V 1.0 Pro at 4-bit: a probe that failed its gate (#1314)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-18). Don filed the idea from the diffusers 0.43.0 release notes after a
2026-11-12 spike outside dw. In that spike, Kandinsky S2V 1.0 Lite distilled
turned one still and a narration into an 86 s talking take, but it failed on
identity: SFace 0.71 by 85 s, saturation 71 to 91. The question here was whether
the 21B Pro model, quantized to fit one 24 GB card, holds the face. If it did,
the family would be onboarded. Plan v2 is the plan comment on #1314, and it
holds the full sweep. This file records what was measured.

## Verdict: build smaller, then stop

There was one probe stage, with a 20 s spend gate before the 86 s take, and a
hard stop either way. **Pro failed the gate on all three thresholds**, so the
86 s take was not rendered and the family is **not onboarded**. Per the plan,
that failure closes the feature. Don's Q3 viewing applied only to an 86 s take,
and there is none.

## Plan history

- **v1 → v2 (ruled 2026-11-14, by a lead session under Don's delegation).**
  - Q1 [spend]: approved a ~45-60 GB download and gated GPU time.
  - Q2 [default]: changed. The probe runs inside dw only after #1315 stage B
    (#1341) moves lem to diffusers 0.43. This feature does not move lem's
    diffusers. The fallback is the 2026-11-12 spike venv.
  - Q3 [taste]: Don's viewing of the 86 s take decides onboarding.
  - Q4 [scope]: onboarding would be a later plan version.
  - No scope change.
- **Cut by the plan:** Lite undistilled (the same 2B model that drifted),
  multiple seeds, takes over 86 s, and any engine change.
- **Rulings during the build:**
  - 2026-11-17: approved a re-download of Pro-distill, with a probe fix first.
    It was not spent: the weights were in the spike venv's default HF cache.
  - 2026-11-18: close the stage on both failures, with no third attempt and no
    further download.

## What was built

### Stage A (#1339): Pro 4-bit probe (bench only)

Merged to `develop` at 392edfe0, with a record follow-up at fdddcb00 (84f91dca).
No deploy.

- **The fallback path was taken, for the waveform reason.** S2V's encoder step
  takes `audio` as a 16 kHz mono float waveform. dw's `fetch_media`
  (`dw/argument_media.py`) decodes only images and video, so an `asset:` wav
  reaches the pipeline as a path. The probe therefore ran in
  `lem:~/spikes/ks2v` (diffusers 0.43.0, torch 2.14.1+cu130, sdnq 0.2.8). It
  did not run on the dw server, which reports 0.44.0.dev0.
- **`bench/ks2v_pro_probe.py`:** the probe. It loads the transformer at SDNQ
  int4 (with bnb nf4 as the fallback), puts the encoders on model CPU offload,
  tiles the VAE, and can group-offload the transformer with `--group-offload`.
- **`bench/ks2v_gate.py`:** the face table in the 2026-11-12 format, with
  `--ref`.
- **`bench/arms/ks2v-pro-4bit.json`** → **`bench/workflows/ks2v-pro-4bit.json`:**
  the dw shape (take → `pair_audio`). It validates, and
  `tests/test_bench_workflows.py` pins it. It can't run until dw has an `audio`
  media branch.
- **Model:** `kandinskylab/Kandinsky-S2V-1.0-Pro-distill-Diffusers` (57.8 GB).
  The plan's "Pro" estimate only holds for the 16-step distilled student. The
  undistilled repo is 62.1 GB, over the approved download.
- **Record:** `bench/records/narrated.md`, section "Kandinsky S2V Pro 4-bit
  probe (2026-11-16 to 11-18, #1339)". The arms row is in
  `narrated-long-take-complete.md`.
- **On the server** (workspace `constitution-walk-3`):
  - the take, `asset:bench/ks2v-pro/pro-gate-20s-go-s1.mp4`;
  - the contact sheet, `asset:bench/ks2v-pro/pro-gate-20s-faces.jpg`;
  - the inputs, `asset:uploads/bench/keyframes-v2/kf00.jpg` and
    `asset:bench/constitution-narration-retimed.wav`.

**Result.**
- **Attempt 1, no transformer offload:** GPU OOM in the denoise step at
  21.02 GiB allocated.
- **Attempt 2, transformer group-offloaded:**
  - SDNQ int4 loaded the transformer class, so bnb was never needed;
  - peak 17.1 GiB allocated and about 48 GB host RSS;
  - 143.9 min for 4 chunks. Chunk 1 took 89 min, swap-bound, and chunk 4 took
    12.5 min;
  - the host thrashed hard enough that the kernel OOM-killed the dw server's
    worker (not the probe).

| at 20 s | Pro 4-bit | gate | |
|---|---|---|---|
| SFace to first frame (to kf00) | 0.72 (0.69) | ≥ 0.88 | fails |
| saturation rise | +5.4 (92.0 → 97.4) | ≤ 5 | fails |
| face-crop Laplacian | 61-114 | H3's 206-375 | fails |

By 5 s the face is rounder and waxy, and by 10 s it reads as a different woman.
Pro drifts faster than Lite distilled did over the same 20 s (Lite's SFace was
.85 at 19.5 s), and it is softer than either H3 arm.

**Bounces: 1.** The first hand-off was missing the input `asset:` names and the
full face table. It also named the wrong interpreter (it said the dw venv; the
probe ran in the spike venv). Hand-off 2 fixed all three, and the tester then
verified C-F773 and C-F776 to C-F779. C-F776 re-measured SFace with
`analyze_face_drift`, and every time came within 0.005 of the record. C-F774
and C-F775 only apply inside dw, so they didn't apply.

## Cost and bounces

| Stage | Bounces | Cost |
|---|---|---|
| #1339 | 1 (needs-info on the hand-off) | Session cost not recorded on the stage (plan estimate $3-5). GPU on lem cuda:0: one OOM attempt, then 143.9 min for the 20 s gate. Download: 57.8 GB (Pro-distill) |

There were also four build sessions and two holds with Don: the download
release, and lem thrashing.

## Deferred and open

- **No onboarding.** There is no `kandinsky-s2v` skill, template or catalog
  entry, and no diffusers floor change for it.
- **Finding for any future S2V-style model:** a model that takes a speech track
  as a waveform needs an `audio` branch in `fetch_media`, which is engine code.
  Nothing else asks for it today.
- **Lite undistilled stays cut.** The plan brought it back only if Pro held.
- **Disk:** the Pro-distill weights (57.8 GB) are still in lem's default HF
  cache (`~/.cache/huggingface/hub`). Deleting them is Don's call.
- **The keyframe-free long talking take is still unsolved.** The face holds to
  85 s only with `standing` or `walk_keyframed`. Other routes are #1312, #1313
  and #1188.
