# `refine-in-place`: same-size LTX refine with a strength ladder (#606)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-06). Plan v1 approved by Don 2026-10-05 for stages A and B, with
his answers folded into v2 (scope unchanged). Two stages, #638 and #639,
both verified 2026-10-06. Builds on `ltx2-refine-clip-complete.md` (#543)
and `fit-to-model-restore-complete.md` (#602), whose fit/restore round trip
the template uses.

The concept came from vrgamegirl19/comfyui-vrgamedevgirl
(`LTX25SigmaPreset.py`, `VRGDG_LTXLoopingSampler.py`), whose
source-available license is incompatible with Apache-2.0. Everything here
was re-implemented from the concept. No code, prompt text or presets were
taken, and the ladders were tuned on lem, not copied.

## The report

`refine-clip.json` and `two-stage.json` hard-code `noise_scale` 0.909375 and
`STAGE_2_DISTILLED_SIGMA_VALUES`, and `refine-clip` always upscales 2x. A
user couldn't refine at the source's size or choose how strongly. The idea:
a same-size refine template with one strength variable mapped to a sigma
ladder, `noise_scale = sigmas[0]`. It also floated per-segment strength
decay in `chained-segments`.

## Verdict

**Build smaller.** Stages A and B; per-segment strength in chains
(stage 2) deferred. Demand evidence was thin (no `field-report`; every
`refine-clip` run was in `regression-model-specific`). The case rested on
Don's approval and on #599, whose face-repair stage #624 needs a same-size
refine at a chosen strength.

## What the plan found against the issue

- **Same size needs a new encode path.** `LTX2LatentUpsamplePipeline`
  always runs its upsampler, `LTX2Pipeline.latents` takes latents only, and
  no dw task VAE-encodes video. `LTX2ConditionPipeline` is not a substitute:
  its condition `strength` is a mask that re-blends toward the source at
  every step.
- **No trailing 0 in a ladder.** `FlowMatchEulerDiscreteScheduler` appends
  the terminal 0 itself. Three sigmas per ladder match stage two's count, so
  cost is flat across strengths.
- **The mapping lives in the workflow**, as a `select` by index over ladder
  candidates, never in engine code.

## Decisions (Don, 2026-10-05)

- **Q1:** a community pipeline, `LTX2RefinePipeline`: VAE-encode `video`,
  pass it as `latents`. No ladder or model constant in it.
- **Q2:** 3–5 ladders, as many as tuning can separate.
- **Q3:** an integer knob mapped through `select` by index.
- **Q4:** stage 2 deferred (triggers below).
- **Q5:** the template is `refine-in-place`.
- **Correction 1:** the `ltx-2.5` skill had about 3.8 KB of headroom; no
  cuts needed.
- **Correction 2:** use fit → refine → restore → `pair_audio` if #602
  stage A had landed, otherwise trim. It had (#631), so no trim and no
  comment on #632 was owed.

## What was built

**Stage A, #638: `LTX2RefinePipeline`.**
- `dw/community_pipelines/pipeline_ltx2_refine.py` subclasses
  `LTX2Pipeline` and adds `video`. A given `video` is resized to `width` x
  `height`, encoded with the upsample pipeline's preprocess and
  `vae.encode` (no upsampler), and passed as `latents`. The parent's
  `noise_scale` and `sigmas` do the rest.
- `video` with `latents` is refused, naming both.
- **Community pipelines are visible over MCP.** `list_pipelines` lists the
  shipped `dw/community_pipelines/` classes by dotted path after the
  diffusers names; `get_pipeline_signature` resolves only those modules
  (`dw/introspection.py`).
- Tests: `tests/test_ltx2_refine_pipeline.py` (a real tiny
  `AutoencoderKLLTX2Video`). Docs: a *Community pipelines* row in
  `docs/ARCHITECTURE.md` and a paragraph in `docs/WORKFLOW_GUIDE.md`.

**Stage B, #639: `templates/ltx2/refine-in-place`.**
- Steps: `ladder` (`select`, rule `index`, `index: variable:strength`) →
  `source_audio` (`normalize_audio`, refuses a silent source) → `fitted`
  (`fit_to_model`) → `refine` (`LTX2RefinePipeline`, `intermediate/`) →
  `restored` (`restore_to_source`) → `with_source_audio` (`pair_audio`,
  `final/`). Output is the source's own size and length.
- `strength` is an integer 0–4, default 2. The description says the route
  is unsourced: the vendor's sharpen route is the Refine-Details IC-LoRA.
- `LTX2RefinePipeline.encode_video` accepts a dw `AudioVideo` (taken as its
  frames), since `fit_to_model` returns one.
- **`select`'s index rule** has one owner, `select_index_problem` in
  `dw/task_domains.py`: a whole number, at least 0, and below the candidate
  count when it is known (at run time, or at validate when `candidates` is
  a written list). Both `select_errors` and `select()` call it, and the
  *Task argument domains* map row names it.
- `ltx-2.5` SKILL.md (routing line, "not a knob except refine-in-place's
  `strength`", the ladder rule, cost), a `workflows/templates/ltx2/README.md`
  row, `TestLtxRefineInPlace`, a `TEMPLATE_PIPELINE_KEYS` row, and
  `COMPACT_BUDGET` 9,400 → 9,550.

**The ladders** (tuned on lem, two sources: `ep64-shot-ltx-hal.mp4` and
`ep63-shot-ltx-priya.mp4`; `noise_scale` = first sigma):

| strength | sigmas | look |
|---|---|---|
| 0 | [0.4, 0.25, 0.1] | near the source |
| 1 | [0.55, 0.4, 0.2] | close |
| 2 | [0.7, 0.5, 0.3] | subtle retexture |
| 3 | [0.8, 0.6, 0.35] | subject's surface redrawn |
| 4 | [0.85, 0.65, 0.4] | identity, framing and scene hold; texture, tone and expression re-rendered |

Rejected rungs: σ0 0.975 replaced subject and scene; σ0 0.909375
(refine-clip's value) gave a plastic, deformed face at the same size; σ0
0.3 and 0.5 were indistinguishable from the source. Tuning job ids are in
#639's first hand-off. These are dw's values on diffusers' deterministic
Euler; they may not carry over to upstream LTX-2's Euler ancestral.

**Measured (M-F095):** 512×288×121 at 24 fps in, the same out, with the
source's soundtrack; 85–87 s warm on an RTX 3090, flat across strengths
(catalog cost 1.5 min). Change rises monotonically with strength.

## Bounces per stage

- **#638: none.** Verified on the first hand-off.
- **#639: three.** An architecture bounce (the whole-number index rule was
  a second owner inside `select`; moved to `dw/task_domains.py`); a tester
  bounce on M-F093 (the validate-time message for −1 lacked the range; it
  now names "0 to 4"); an architecture bounce on the map (row "Task
  argument domains" didn't name the moved rule). Verified on the fourth
  hand-off.

No `usage:` figures were recorded on the stages, so cost is left out. The
plan's estimate was $4–6 for A and $5–7 for B, plus about 20 GPU-minutes.

## Deferred and left open

- **Stage 2, per-segment strength decay in `chained-segments`.** It would
  need `chained-segments` moved to `LTX2ConditionPipeline`, a new
  `chain.strengths` list, a continuity-inject mode replacing the index-0
  `LTX2VideoCondition`, a changed `image-conditioned` trait and a new
  step-cache row. **Comes back** on a field report of drift across
  `chained-segments` joins that a lower carry strength would fix, or when
  #599 / #601 lands a multi-frame carry it could reuse, filed then as its
  own idea.
- **No `plugins/dw` version bump** in stage B: the plugin's version is the
  engine's and the release script bumps both.
- **Discovery friction** (from #638's verify): a bare
  `get_pipeline_signature("LTX2RefinePipeline")` gives no hint at the
  dotted path, and the `video` description carries the parent's docstring.
  Hand-authored workflows must keep `reused_components: ["vae"]` (or place
  the VAE on `cuda`); validate doesn't warn when it is dropped.
- **#624** (#599 stage 3, face repair) can now consume the pipeline and
  ladders.
