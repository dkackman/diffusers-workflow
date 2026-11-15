# H3 turbo lead-in: opening evaluations with the turbo LoRA held off (#1326)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-15). Don filed the idea on 2026-11-13 from a reading of
[roadmaus/ComfyUI-Continuity](https://github.com/roadmaus/ComfyUI-Continuity)
(MIT; concept only, no code taken). Plan v2 on #1326 is the full design; this
file records what shipped, what the probe found, and what stays cut.

**Result: negative.** The argument shipped as an opt-in; the probe found it
makes the per-cut step worse on both chain shapes, and Don ruled (Q4,
2026-11-15) that no template or skill uses it.

## The problem

Under the 4-step turbo LoRA, each continued H3 segment brightens and coarsens at
the cut. The source runs the opening steps without the distilled adapter and
reported the per-cut brightness step falling from +2.86 to +0.98 on 8-shot turbo
strips with its own sampler. The question was whether that carries over to dw's
turbo schedules (5 steps = 4 evaluations, and 9 steps = 8).

## Verdict: build smaller

An argument plus a probe, no default change. **Cut:** template defaults and a
SKILL.md rule. *What would have brought them back:* stage B showing the per-cut
lightness step falling by about a third or more at the same seed, and Don's taste
verdict (Q4) finding no loss of detail or motion. Neither held (below), so the
cut stands. A revival would start from a different schedule or sampler, not
from this argument at a different N.

**The plan corrected the issue:**
- H3 runs on diffusers' modular pipeline, which takes no step callback, so the
  switch is a transformer forward pre-hook (modelled on TST), not a
  `callback_on_step_end`.
- H3 is guidance-distilled, so a base-weight evaluation costs the same as a
  turbo one. "Base-weight speed" costs no GPU time; the cost is quality.
- The switch must be `set_adapter_scales` on `turbo` only:
  `attention_kwargs["scale"]` would zero and then reset every adapter,
  wiping anti-burn (#1151) or a non-1 `lora_scale`.

Non-goals that stayed out: a lora-schema field or general per-step LoRA
schedule, a cache-key change, non-H3 pipelines, `narrated-take` (no adapter),
a fractional ramp, and any MCP, REST or template change.

## Plan history

- **v1** (design, 2026-11-13).
- **v2** (decompose, 2026-11-13) folded in the ruling: Q1 [scope] build smaller,
  yes; Q2 [spend] about 3 h of lem, yes; Q3 [scope] the argument's shape
  (`turbo_lead_in`, counts evaluations, adapter fixed to `turbo`), yes. Stage A
  also had to measure the `torch.compile` cost.
- **Q4** [taste], asked on stage B's comparison comment: **ruled no**
  (2026-11-15). The cvc identity hold does not make up for the darkening and the
  saturation climb.

## What was built

### Stage A, #1352: `turbo_lead_in` argument and hook

Merges `c5c77453` (`8e499427` hook and rules, `6220eea4` tests and docs) and
`3a6f3c80` (`50398d35`, the measured compile figure).

- `dw/pipeline_processors/h3_lead_in.py`: a context manager on
  `_call_pipeline`'s ExitStack beside `tst_call`. A forward pre-hook counts
  evaluations by distinct timestep, holds `turbo` at 0 for the first N,
  restores its configured scale after, and restores in a `finally` on success,
  error and cancel. One job event per call:
  `turbo lead-in: evaluations 1..N with adapter 'turbo' at 0 of M`.
- `h3_rules` refuses, at validate and at run: a non-integer or negative value,
  N at or past the call's evaluations, no adapter named `turbo` (listing the
  names present), a non-H3 step, and the `first_block` cache.
- **Compile cost, measured:** the scale-0 graph compiles once per worker,
  +5.5 s on the first lead-in call and nothing after
  (`docs/ACCELERATION.md`, torch.compile).
- Docs: `docs/WORKFLOW_GUIDE.md` (the `turbo_lead_in` section),
  `docs/ACCELERATION.md`, and a ComfyUI-Continuity entry in
  `docs/ATTRIBUTIONS.md`.

**Defect found by stage B, #1365** (fix `07ec5fe5`): on lem, N ≥ 1 never
restored the adapter, so every lead-in arm equalled `lora_scale` 0. The hook
read the timestep as the tensor's max, and a row H3 holds at full noise (audio,
a guide) pins that max, so the count never advanced past one distinct value.
It now reads the target video's row through `timestep_indices`, falling back to
the max only when there are no indices. The unit tests' fake pipeline passed a
single timestep and missed it. After the fix, the
1-segment checks were distinct (lead-in 1 against turbo off: PSNR 26.94).

### Stage B, #1353: lead-in probe on lem (no code)

Workspace `probe-1326`, 8 segments, seed 42, 960×544, `check: true`,
`retries: 0`, no compile. All lead-in arms ran after the #1365 fix
(lem on develop @ 0a60c514).

| arm | workflow | turbo_lead_in | job | GPU time |
|---|---|---|---|---|
| cs-0 | chained-segments (4 eval) | 0 | `5fa5a09b083f` | 14.6 min |
| cs-1 | chained-segments | 1 | `8565bd536f7e` | 14.6 min |
| cs-2 | chained-segments | 2 | `c6844644ac5f` | 14.5 min |
| cvc-0 | chain-video-continuity (8 eval) | 0 | `fecc614dc513` | 62.5 min |
| cvc-2 | chain-video-continuity | 2 | `c663a18db477` | 52.5 min |
| cvc-3 | chain-video-continuity | 3 | `c32ae8c9a6f4` | 58.7 min |

Two earlier lead-in arms (`55a7efee834c`, `f9cda8c398b9`) ran before the fix
and are void.

**Per cut: `analyze_seams` lightness step, seams 1-7**

| arm | lightness_step per seam | mean \|L\| | vs baseline |
|---|---|---|---|
| cs-0 | 0.63, 0.21, 0.19, -0.46, 0.10, -0.67, -1.38 | **0.52** | — |
| cs-1 | 3.12, 1.60, -0.90, -0.69, 0.32, -1.14, -0.40 | **1.17** | +124% |
| cs-2 | 2.28, 1.31, 1.45, -0.19, 0.26, -0.18, -0.24 | **0.84** | +62% |
| cvc-0 | -2.15, -1.28, -2.05, 0.61, -0.76, -1.04, -0.93 | **1.26** | — |
| cvc-2 | -3.84, -2.55, -0.42, -0.14, 0.19, 1.37, -3.41 | **1.70** | +35% |
| cvc-3 | -4.05, -2.30, -0.46, -0.03, -0.99, -0.58, -11.85 | **2.89** | +129% |

**Per seam: chain_check look and jump, and texture drift**

| arm | look_ratio per seam (measure) | jump_ratio (mean) | chain_check fails | detail, seg 1 first → seg 8 last | lightness, start → end |
|---|---|---|---|---|---|
| cs-0 | 1.84c, 1.26, 0.94, 0.46L, 0.34, 0.67L, 1.38L | 0.96 | none | 5.58 → 8.08 (×1.45) | 80 → 80 |
| cs-1 | 5.19c, 2.15c, 0.90L, 0.69L, 0.32L, 1.14L, 0.49c | 1.07 | none | 5.60 → 8.25 (×1.47) | 80.6 → 83.5 |
| cs-2 | 3.14c, 1.61c, 1.45L, 0.20c, 0.26L, 0.35c, 0.47c | 1.10 | none | 5.60 → 8.29 (×1.48) | 80.5 → 81.3 |
| cvc-0 | 2.15L, 1.28L, 2.05L, 1.40s, 1.03s, 1.28s, 2.70s | 10.64 | jump, segs 2-8 | 4.55 → 6.40 (×1.41) | 82.0 → 68.6 |
| cvc-2 | 3.84L, 2.55L, 1.75s, 1.90s, 2.54s, 2.47s, 2.52L | 9.34 | jump, segs 3-8 | 4.30 → 5.21 (×1.21) | 72.0 → 56.5 |
| cvc-3 | 4.05L, 2.55s, 2.87s, 2.38s, 1.92s, 1.82s, 11.85L | 9.44 | jump, segs 2-8; look, seg 8 | 4.28 → 4.80 (×1.12) | 75.4 → 47.0 |

Key: c = contrast, L = lightness, s = saturation (the measure that set
`look_ratio`). Saturation over cvc, segment 1 → 8: cvc-0 17.2 → 23.8,
cvc-2 18.2 → 29.8, cvc-3 22.6 → 39.6.

**Method notes.** `chain_check` keeps only `face_drop` on a passing segment, so
per-seam look and jump values were read back through `analyze_seams`, which
uses the same `seam_measures` code. Texture drift is `segment_looks.detail`;
`compare_outputs(texture_bands=true)` agrees on cs-0 (×1.37). `chain_check` is
not a task of its own (`get_task` answers "Unknown task command").

**Reading.**
- The cut-return bar fails on every arm: mean |lightness step| rises 62-124%
  on chained-segments and 35-129% on chain-video-continuity.
- **chained-segments (keyframe-anchored):** texture drift is unchanged
  (×1.45-1.48); the only change is a larger step at seams 1-2. Frames at
  segment 8 show the same over-sharpened grain on every arm.
- **chain-video-continuity:** a trade, not a win. Texture drift is lower
  (×1.21, ×1.12 against ×1.41), the first seams jump less, and the subject's
  frontal pose and suit detail hold where the baseline turns to a profile. But
  segment 1 itself starts darker and bluer, the chain darkens further
  (end lightness 56.5 and 47.0 against 68.6), saturation climbs about twice as
  fast, and from segment 5 the jump is as bad as the baseline's.

## Cost and bounces

- **Stage A:** verified on the first hand-off (M-F311-M-F315). Its defect was
  found later by stage B and fixed as #1365, a separate implementer issue: one
  bounce in effect.
- **Stage B:** paused once for #1365, then verified on the first hand-off
  (M-F316-M-F318).
- **lem:** about 5.0 GPU-h and 3.0 h wall for the probe, including 1.75 h of
  wall before the fix (the void arms). The approved 3 h held for wall time;
  GPU-hours ran higher because two cards ran in parallel.
- Session spend was not recorded on the stages, so it is left out.

## What stays

- `turbo_lead_in` remains an opt-in H3 call argument. No template references it,
  so removing it later breaks no shipped workflow; it could be accepted as a
  no-op.
- The #1365 lesson: a hook that reads H3's timestep must read the target row's,
  because a call carries one per packed row; fake pipelines in unit tests
  should pass per-row timesteps with `timestep_indices`.
- Other attacks on the same drift: anti-burn on continuation segments (#1151),
  seam restore (#1325), split-unsample (#1313, parked).
