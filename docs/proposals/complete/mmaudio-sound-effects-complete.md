# MMAudio picture-synced sound effects: `generate_sound_effects` (#1124)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-01). Plan v2 was approved by Don on 2026-11-01. Stage A was a
listening gate run outside dw, and it passed. Stages #1131 (B), #1132 (C)
and #1133 (D) were then built and verified. The plan comment on #1124 is
the full design. This doc records what shipped against it.

## The idea

Sound timed to what happens on screen in a video clip, optionally guided by
a text prompt, plus a text-only mode for one-off effects. It uses MMAudio
`large_44k_v2` ([hkchengrex/MMAudio](https://github.com/hkchengrex/MMAudio),
code MIT, weights CC-BY-NC 4.0). That is a flow-matching network
conditioned on DFN5B CLIP features (8 fps, 384 px), Synchformer features
(25 fps, 224 px) and text. K6, LTX-2.5 and H3 make their own sound, but it
is often weak on action, and restored, upscaled, retimed or animated-still
clips have none. Upstream depends on `open_clip_torch`, `timm`,
`hydra-core`, `torchdiffeq` and others, so the route was to vendor and port
it with no upstream dependencies, the same as #1112.

## Decisions

- **Gate first (stage A).** Don's session read upstream at
  `974010a026c731054592d8f777218bd9d85a6c24` and found nothing blocking:
  every inference-path `torch.load` is `weights_only=True`, and the CUDA
  kernel build is never reached. It then ran the code once on lem GPU 1 in
  a throwaway venv. The inputs were a K6 locomotive shot, the H3 melee
  shot, an LTX-2.5 record lift, a silent LTX2Colorize clip and two
  text-only effects, at seed 1124 with 25 Euler steps and CFG 4.5 in bf16.
  Load took 17.6 s with a 5.84–5.87 GiB peak. Generation took 2–3 s per
  5–8 s clip. The same seed gave bit-identical output. Recordings:
  `lem:~/mmaudio-stageA/`.
- **Verdict (Don, 2026-11-01): a clear win on all three named uses.** The
  record lift and the locomotive were clearly better than their native
  sound, and the melee was better. The silent clip was usable, with the
  prompted run preferred: video only gave bells and no pigeons, and the
  prompt brought the pigeons. The text-only door slam was usable.
- **Q1:** the upstream run was authorized once, for stage A only.
- **Q2:** decode with diffusers' `MMAudioVAE` / `MMAudioVocoder`, with a
  `diffusers>=0.42.0` floor. Stage A found K6's `audio_vae/` and `vocoder/`
  safetensors bit-identical to upstream's `v1-44.pth` + BigVGAN, so there
  is no decoder port and no pickle load for decoding.
- **Q3:** DFN5B's Apple ML Research Model License (research only) is
  accepted. Both licences and the Apple notice are stated wherever a user
  meets the model. dw never writes or ships converted weights.
- **Q4:** warn outside 4–12 s, and refuse over 20 s.
- **Q5:** text-only mode is in scope.
- **The prompt is recommended even with a video.** Video-only is documented
  as the weaker mode, and `prompt` stays optional.

## What was built

### Stage B, #1131: flow network, sampler, CLIP conditioning

- `dw/tasks/mmaudio_model.py` (772 lines) holds the flow network, the
  `large_44k_v2` config, `euler_sample` / `sample_latent` (our own Euler
  loop with CFG, the empty-condition path and the negative-text path, and
  no torchdiffeq), and `load_network`. That load is strict, uses
  `weights_only=True`, and is pinned to `hkchengrex/MMAudio` @ `eb13a1a9`.
- `dw/tasks/mmaudio_clip.py` (399 lines) maps DFN5B's open_clip keys onto
  transformers `CLIPVisionModelWithProjection` / `CLIPTextModel`. The map
  is strict, works in memory only, and reproduces `patch_clip`'s
  normalised last hidden states. The plan's vendored-towers fallback was
  not needed.
- Tests: `tests/test_mmaudio_model.py`, 24 cases. They include an Euler
  closed form, a CLIP forward checked against a plain-torch open_clip
  reference, and pins for no Hub I/O, no forbidden imports and the module
  being inert.
- Deviations: there are two modules, not one, because one module came to
  1,118 lines, over the 1,100 ratchet. ftfy is not carried (the tokenizer
  helper still does upstream's unescape, whitespace and lowercase steps).

### Stage C, #1132: Synchformer, decoder load, frame sampling, centroid

- `dw/tasks/mmaudio_sync_model.py` is the Synchformer visual encoder,
  without timm, einops or omegaconf. Its load is `weights_only=True` and
  strict on `vfeat_extractor*`. Segmentation is 16 frames with a step of 8.
  Its key set was checked against the real checkpoint by reading pickle
  opcodes, never by unpickling it.
- `dw/tasks/mmaudio_decoder.py` builds diffusers `MMAudioVAE` +
  `MMAudioVocoder` from K6 @ `4c487e8b4aeeb9059e5a77905aa2ddb59ec28f81`.
  There is no pickle, key map or fuse.
- `dw/media.py` `sample_frames` (PyAV only) and `dw/dsp.py`
  `spectral_centroid`.
- The `diffusers>=0.42.0` floor in `pyproject.toml`, followed by
  `FALLBACK_RELEASE_FLOOR` in `dw/server/updater.py`.
- `docs/ATTRIBUTIONS.md` rows (Synchformer, BigVGAN, the Synchformer
  checkpoint, the K6 audio tensors) and a `docs/SECURITY.md` paragraph on
  the two pickle loads.
- Deviations:
  - K6's `audio_vae` safetensors carries an embedded `vocoder.` copy
    (667 tensors). The loader filters that prefix out before the strict
    load; this is a filter, not a key map.
  - Only the 0.42 class's own constructor arguments are passed from K6's
    config.
  - The VAE tensors are recorded as CC-BY-NC 4.0 per the plan, though K6's
    card says MIT. ATTRIBUTIONS states both.
  - The K6 pin is newer than the K6 templates' `e056a84f…`.
  - Numerical equivalence with upstream is not tested, because no upstream
    code ran in the build.

### Stage D, #1133: the task, templates, docs, skill lines

- **`generate_sound_effects`** is in `dw/tasks/sound_effects.py` and
  registered in `model_handlers.py` with `consumes_device=True` and
  `task.seed_for`.
  - Arguments: `video`, `prompt`, `negative_prompt`, `duration` (the clip's
    length, or 8 s text-only), `steps` 25 and `guidance_scale` 4.5.
  - Output: a 44.1 kHz mono `AudioTrack`.
  - `generate_sound_effects` is in `_LAZY_FRAME_COMMANDS`, so `video`
    arrives as a file. An in-memory clip is written to a temporary mp4
    first.
- **Refusals, all before any load:** the trust gate runs first. Then:
  neither input given; `duration` over 20 s or longer than the clip; an
  unknown argument; a non-positive `steps`, `guidance_scale` or `duration`.
  The number checks go through `whole_number`/`real_number` +
  `check_arguments`.
- **Warnings** through `emit_warning`: `duration` outside 4–12 s, and a
  noise-like result (flatness ≥ 0.3 and centroid ≥ 8 kHz).
- **Weights:** `required_weights` pins 8 files. They are MMAudio's network
  and Synchformer @ `eb13a1a9…`, K6's `audio_vae/` + `vocoder/` @
  `4c487e8b…`, and DFN5B's weights and tokenizer @ `01b771ed…`. Neither
  `v1-44.pth` nor `bigvgan_generator.pt` is listed. Loads announce
  themselves with phase/log events and download progress through
  `download_watch`.
- **Catalog:** `generate_sound_effects` is in `GENERATIVE_TASKS`. There are
  two templates:
  - `workflows/templates/video-to-sound-effects.json` (shape `audio`,
    declared explicitly; `has-audio`, `needs-input-media`). It outputs the
    stem plus a `pair_audio` mux.
  - `workflows/templates/text-to-sound-effect.json`.
- **Docs:** `docs/TASKS.md` *Sound Effects Generation*. Both licences and
  the Apple notice appear in the template descriptions, `get_task`,
  `TASKS.md` and `ATTRIBUTIONS.md`.
- **Plugin:** skill lines in `series-episodes` (effects in the `rest` bed),
  `ltx-2.5` and `kandinsky-6` (replacing a shot's own sound). Each one
  routes through `match_levels`, because MMAudio runs 7–25 dB hot.
  `minimax-h3` is untouched.
- Test edits outside the plan: `COMPACT_BUDGET` was raised from 12,480 to
  12,650 tokens. Three templates whose quoted `'video'` now reads as a
  variable mention were added to `LEGITIMATE_MENTIONS`. `test_step_cache`
  got rows for both templates.

### Bounce fixes (D)

1. **Architecture review.** `_number` hand-rolled numeric coercion that
   belonged to `dw/task_domains.py`, and `_refuse_remote_code` copied the
   trust gate. The trust-gate copy came from `generate_speech` (#1121).
   Both copies are gone. The new `dw.trust.refuse_unused_task_remote_code`
   delegates to `require_trusted_task_arguments` and is used by both tasks.
2. **Tester (C-F614, C-F617).** `get_task` didn't document the noise
   warning. A cold run fetched about 10.5 GB of weights with no download
   or load event, only `phase_stall` for about 645 s. The summary now
   documents the warning, and a `_cached` helper emits phase and log
   events plus `download_watch` progress for each load.
3. **Tester (C-F620 arm 1).** On a 2.083 s clip, the clip-length default
   rounded down to whole CLIP frames (2.0 s), and the mux padded 0.08 s of
   silence. The default is now the clip's fractional length.

## Verification

Over MCP on lem.
- B: C-F612 passed (no MCP-visible change).
- C: C-F613 passed. The tester flagged at spec time that lem's
  `0.42.0.dev0` would sort below the `>=0.42.0` floor. The stage verified
  on the deployed install.
- D: C-F614 to C-F622, SE-F058 and SE-F059 all pass, at develop
  `e6864b3f`. In the prompted melee run, the stem was 8.0 s, 44.1 kHz
  mono, and non-silent at a mean of −24.2 dB, against the clip's own
  −41.6. The mux was 192 frames.
- C-F617 step 3 (onsets within 0.2 s of visible hits) was **inconclusive**,
  not passed. The melee clip is a continuous scrum, so no single strike
  could be told apart at 320 px, and `measure_sync` wasn't run.
- C-F623 (mps) was conditional on a mini-ai target being in scope.

## Bounces per stage

| Stage | Bounces | Why |
|---|---|---|
| A | — | Not dw code; Don's session. |
| B (#1131) | 0 | |
| C (#1132) | 0 | |
| D (#1133) | 3 | One arch review: numeric coercion and trust-gate copies. Two tester verifies: C-F614 and C-F617, then C-F620 arm 1. All are above. |

Cost per stage: the stage comments carry no `usage:` figures, so cost is
left out. The plan estimated B at $8–12, C at $4–7 and D at $5–8. Stage A
cost no agent spend.

## Not built, and why

- MMAudio checkpoints other than `large_44k_v2`, training, chunking past
  20 s, music and dialogue: non-goals.
- Mixing policy stays with the skills (`match_levels`), not the task.
- No H3 line: `minimax-h3/SKILL.md` had no room, and no reference line was
  needed for the named uses.

## Known edges

- **Picture sync on a clear single hit is unmeasured over MCP** (C-F617
  step 3, above). Stage A's listening verdict is the evidence.
- **The noise threshold has never fired on a dw output.** It is set from
  story-forge's one hiss case (0.45 / 10.7 kHz). All of stage A's outputs
  had flatness ≤ 0.019.
- **Output runs hot.** Verify runs peaked at −0.31 to +0.0 dBFS, and
  `audio_no_headroom` fired. The skills route through `match_levels`, but
  a direct mux of the stem keeps the level.
- **The weights are non-commercial:** CC-BY-NC 4.0 for MMAudio and Apple
  AMLR (research only) for DFN5B. They are dw's first.
- **mps is unverified.** It runs fp32 by default.
- **The K6 audio pin (`4c487e8b`) differs from the K6 templates'
  (`e056a84f`).** That means two snapshots in the cache if both are used.
- **The arch reviewers noted that `docs/ARCHITECTURE.md` has no rows for
  the task-argument remote-code gate, `download_watch` or `model_cache`.**
  This is left for a curator pass.

## Reopen triggers

- A diffusers MMAudio pipeline lands: the vendored modules could shrink to
  a pin.
- A newer MMAudio checkpoint or upstream commit: re-review the code before
  moving `REVISION`.
- A dw output comes back as hiss without the warning firing: re-measure
  the threshold.
