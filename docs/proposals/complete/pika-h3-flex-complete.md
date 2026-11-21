# Pika H3 Flex: a six-step single-clip option beside MiniMax-H3 (#1316)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-17). Plan v1 was approved under diffusers 0.42. Plan v2 (approved by
Don 2026-11-16) followed stage B's first action, which found that Pika's index
loads its own ~216 GB copy of the base. Stages A (#1342), B (#1343) and C
(#1344) all shipped and were verified; the last closed 2026-11-17.

## The idea

diffusers 0.43.0 added Pika H3 Flex: Pika's post-training of MiniMax-H3 into a
six-step video-and-audio generator. It is two rank-128 LoRA adapters,
`fl2va/adapter.safetensors` (t2va and fl2va) and `ref2va/adapter.safetensors`,
over the two H3 transformer partitions, plus a stochastic sampler on a fixed
`denoising_step_list` `[1000, 750, 500, 250, 100, 40, 0]`. dw's fast path was
lightx2v (4-step fl2v, 8-step ref2v), so a vendor-validated 6-step ref2va
looked like a speed-up for reference-to-video.

## Verdict

**Build smaller.** Single clips only (t2va, fl2va, ref2va) in a separate
catalog directory, with the skill routing to Pika only where Don's bench
verdict (Q3) said it wins. Chains, a Pika tier in the existing H3
templates/presets, and in-worker `disable_lora()` reuse were cut.

## What the plan found against the issue

1. **Pika is its own modular pipeline, not an H3 variant.**
   `PikaH3FlexModularPipeline` / `PikaH3FlexBlocks`, its own reference
   dataclasses, no guider (no `guidance_scale`, `negative_prompt` or
   `num_inference_steps`). dw's H3 chain blocks (guides, hold, selflift,
   anchor, mask, windows/capstone, tiles) live in H3's block graph, and
   three of them assume Euler. So the issue's "does the stochastic sampler
   work with the chain machinery" has a structural answer: no.
2. **The step list is a blocks `ConfigSpec` default**
   (`pika_h3_flex/before_denoise.py`), not index config. Every dw
   step-count consumer keyed on `num_inference_steps`, so a Pika job would
   have quoted and reported a wrong default.
3. **(v2) Pika's index names its own base copy.** The repo is 219.9 GB; a
   t2va/fl2va load is ~147 GB and ref2va ~216 GB. Range reads against
   `MiniMaxAI/MiniMax-H3` showed identical tensor name sets, 14 sampled
   tensors byte-identical, the same `audio_vae` sha256, and each adapter's
   manifest names `base.repo_id: MiniMaxAI/MiniMax-H3`. The copy is H3
   re-sharded. **Caveat:** that is a sample, not a proof. If Pika output ever
   looks wrong against Pika's published samples, check the redirect first.

## What was built

### Stage A (#1342): step-list pass counting

`311a9aa8`, merged `c8818141`. `dw/step_passes.py`'s helper returns
`len(denoising_step_list) − 1` when a pipeline config carries one and keeps the
`num_inference_steps` rule otherwise. Plan cost scaling, progress, TST,
MagCache calibration and acceleration warnings all read it. MagCache
calibration and TST step-keyed scheduling warn "not supported on a fixed
step-list sampler" instead of mis-keying. Verified on 0.42 as a no-regression
check; stage B's runs exercised the new path.

### Stage B (#1343): three templates and the bench

Merges `4a1bbefc`, `cf4ff2ef`, `04cf8eda`, `8e1b27f6`, `08a0f929`.

- **`workflows/templates/pika-h3/`**: `text-to-video.json`,
  `first-and-last-frame.json`, `reference-to-video.json`. Each is a
  `ModularPipeline` from `Pika-Labs/Pika-H3-Flex` with the H3 templates' SDNQ
  int8 / group-offload block, one adapter at scale 1.0, no scheduler shift,
  no `num_inference_steps`, cache off, 480p/768p presets, and the
  `num_frames` 120..360 → `17n+5` constraint.
- **Weights redirected** (`8625bce6`, then `c2b4791a`):
  `load_components.pretrained_model_name_or_path` maps `transformer`,
  `transformer_ref`, `text_encoder`, `vae` and `audio_vae` to
  `MiniMaxAI/MiniMax-H3`, pinned at `0919fb68` (H3's own `main`), so Pika
  shares H3's download and its int8 cache. The first pin, `42ed227e` (the
  adapters' recorded base), quantized everything a second time because
  `quantized_cache` keys on the snapshot commit. Pika's scheduler, audio
  scheduler, tokenizer and processor stay Pika's. This is the catalog's first
  per-component repo override.
- **The step list is stated** in `configuration.configs.denoising_step_list`,
  and `step_passes` reads it at validate time (`c870e40c`), so the quote
  counts 6 passes.
- **Download plan by component** (`59db595e`): a modular base whose
  `load_components` moves components elsewhere is planned per component off
  its `modular_model_index.json`, listing only what stays on the base. The
  *Download sources* row of `docs/ARCHITECTURE.md` states the rule
  (`a5d126cd`). First-run download measured within the ≤ 10 GB line
  (C-F946).
- **ref2va VRAM** (`9fdcda01`): `gb_per_reference` 1.0 → 1.2 from this
  template's 768p runs, so a 3-reference 1344x768x124 job warns at validate
  (the #479 int4 calibration under-projected Pika at int8).
- **SE-F066** (`e99583df`): a `loras` `weight_name` must be `.safetensors` in
  every shape, checked in the location walk.
- Guard, budget and structure tests: the few-step no-cache rule covers
  step-list pipelines, `COMPACT_BUDGET` bumped, and structure tests pin the
  override, the step list and adapter partitions.
- A host-OOM in fl2va decode with two int8 loads on lem was its own bug,
  #1407 (fixed `1aaa61bf`), and the stage was parked on it.

### Stage C (#1344): skill and docs

`5a01065b`, `62054a8c`, merged `065d4c41`.
`plugins/dw/skills/minimax-h3/references/pika-flex.md` (single-clip only, no
shift/steps, buckets, seeds not comparable with H3, when to pick it), one
pointer line in SKILL.md, carve-outs from lightx2v's "steps = passes + 1" rule
in the docs and the minimax README, and `test_plugin_skills.py` pins.

## Measured: the Q2 bench and Don's Q3 ruling

lem, RTX 3090, develop `fc971209`, seed 42, 1344x768, 124 frames, int8 on both
sides. Full record: #1316 issuecomment-6962191155; outputs in workspace
`regression-complete`.

| Pair | H3 wall / denoise s | Pika wall / denoise s |
|---|---|---|
| F1 fl2va street, first frame | 238.7 / 98.3 | 328 / 190.1 |
| F2 fl2va portrait, spoken line | 208 / 96.4 | 283 / 168.7 |
| F3 fl2va octopus first+last | 224.5 / 108.2 | 308.5 / 188.9 |
| R1 ref2va aurelio + voice | 431.5 / 290.9 | 625.5 / 536.5 |
| R2 ref2va astronaut + voice | 609 / 533.0 | 671.8 / 543.6 |

- fl2va: Pika takes ~1.3-1.4x H3's wall time (6 passes vs 4, each ~20%
  slower). ref2va steady denoise is about equal (H3 8×36 s ≈ Pika 6×51 s);
  single-run ref2va times carry a one-off ~270 s first-pass warm-up.
- No Pika output looked degraded, so nothing pointed at the redirect.
- **Don's ruling (2026-11-17): Pika is an option, not the default.** R1 a
  draw, R2 split (Pika kept the mirrored visor but showed no speaker), no
  speed win. H3 lightx2v stays the default for fl2va and ref2va. The skill
  names Pika when fidelity to a reference's look matters more than a visible
  speaker, or for closer prompt-following when re-framing the source is
  acceptable (F2/F3). Q6's routing change therefore did not happen.

## Bounces per stage

- **Stage A (#1342):** none; one deploy hiccup (the server did not restart).
- **Stage B (#1343): 4 tester bounces and 1 architecture-review bounce,
  parked twice.**
  1. C-F801/C-F945: the download plan listed the whole Pika repo (204.8 GB).
  2. Architecture review: the *Download sources* map row stated the old
     rule.
  3. C-F805: fl2va killed by the host OOM killer in decode (→ #1407), and
     C-F946: the int8 cache not shared (the `42ed227e` pin).
  4. C-F803: no VRAM warning on 3-reference 768p ref2va, then OOM; and
     SE-F066. The driver parked it at four bounces; Don ruled them four
     real, distinct defects and sent it back.
  - A second park (no progress) because the Q2 bench, a stage B
    deliverable, had not run; Don sent it to the lead to run.
- **Stage C (#1344):** none.
- **Final check (parent, 2026-11-18): 1 bounce, on C-F808.** A plain
  `cache: {type: "mag"}` on a Pika template validated with only the generic
  "run >= 20 steps" short-schedule warning (impossible on a fixed 6-pass
  list), then failed at load in diffusers for missing `mag_ratios` (job
  `dd25e5d56b04`). Stage A's step-list skip had covered only a *calibrating*
  MagCache. A build miss, not a plan gap.
  - **Fix-forward stage #1430:** validate now warns `step_list_unsupported`
    ("MagCache is not supported on a fixed step-list sampler; the run drops
    the cache"), reading the step list through `dw/step_passes.py`, and the
    load drops the cache (`1ec02965`, merged `f1888da2`). Architecture review
    passed; verified on lem against C-F808 with no bounces.

No `usage:` figures were recorded on the stages, so cost is left out.

## Follow-ups and deferred

- **#1427** (from the bench): a ref2va audio reference outside 2-15 s passed
  validate and was refused at run time. Fixed in `4995c855`.
- **Chains on Pika** (chained-segments, repair, regional-edit, face-repair,
  add-sound): its own feature, only if Pika single clips win clearly and Don
  wants chained Pika. The bench did not show a clear win.
- **A Pika tier in the existing H3 templates/presets** (`music-video` etc.):
  cut; Q3 did not make Pika the default.
- **`disable_lora()` reuse in the worker:** dw caches per step config, so a
  Pika and an H3 job are separate entries.
- **Pinning risk:** the templates depend on `MiniMaxAI/MiniMax-H3` @
  `0919fb68` staying Pika's base. A re-pin of the H3 templates should move
  the Pika pin with it, after re-checking the weights.
- MagCache / sparse attention at six steps: cache stays off, and sparse
  attention stays on the H3 default. No A/B isolated either at six steps.
