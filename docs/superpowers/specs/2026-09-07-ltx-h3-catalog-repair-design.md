# LTX-2.5 and MiniMax H3 catalog repair — design

Package A of the Part 4 work in
[agent-catalog-legibility-complete.md](../../proposals/agent-catalog-legibility-complete.md).
It repairs what two vendor-source audits found wrong in the catalog's
model-specific content, so that the skills in Package B
([2026-09-07-dw-plugin-skills-design.md](2026-09-07-dw-plugin-skills-design.md))
describe something true. The audits, with per-claim citations, are
[2026-09-07-ltx-2.5-audit.md](../../proposals/audits/2026-09-07-ltx-2.5-audit.md)
and
[2026-09-07-minimax-h3-audit.md](../../proposals/audits/2026-09-07-minimax-h3-audit.md).

## Why

The catalog's H3 and LTX-2.5 knowledge was written from the vendors' own
material but never cites it, and two things drifted from it: the LTX two-stage
template stops after the latent upsample and skips the refinement pass every
primary source describes, and the LTX prompt library is in a genre the model
was not trained on. The H3 Context-IR builtin is a faithful compression of
MiniMax's two prompt-writing guides with three defects of its own. Everything
here is a data change under the proposal's principle: guide prose, template
JSON, stored prompts. One engine question is named below and is the only
place code might move.

## Scope

In: the eight LTX-2.5 templates' README and the two-stage template, the six
stored LTX prompts, the `h3_context_ir` builtin's system prompt, the MiniMax
templates README, the LTX-2.5 section of `RECIPES_24GB.md`, and the ledger row.

Out, recorded as dated follow-ups in the ledger and not done here: the DFR
pipeline as the newer quality path, temporal upscaling, generated keyframe
slots, the newer H3 turbo LoRAs and their scheduler shifts, the reference
resize policy, fp8/NVFP4, HDR, retake, native multishot. The keyframe
strength question (1.0 versus the guiding range) is also out; the template
works as built and the audit calls it arguable, not wrong.

## 1. The LTX-2.5 two-stage template

`workflows/templates/ltx2/two-stage.json` becomes the three-move flow the
model card, Lightricks' pipeline notes and the diffusers docs all describe:

1. **base**: unchanged settings (768x448, 121 frames, the 8 distilled sigmas,
   guidance off), but `output_type` becomes `"{latent}"` and `save` stays
   false. The step returns the video latents and the audio latents.
2. **upscale**: `LTX2LatentUpsamplePipeline` takes `latents` from the base
   step instead of decoded frames, so nothing round-trips through pixels.
   `output_type` is `"{latent}"`.
3. **refine**: the same `LTX2Pipeline` configuration as the base step, so the
   pipeline cache serves it without a second load. Arguments: the prompt and
   negative prompt, `latents` from the upscale step, `audio_latents` from the
   base step, `sigmas` as
   `constant:diffusers.pipelines.ltx2.utils.STAGE_2_DISTILLED_SIGMA_VALUES`,
   `noise_scale` as the first value of that schedule, width and height at
   twice the base values, the same frame count and frame rate, guidance off
   as before, and `output_type` `"{np}"`. The result is `video/mp4`, muxed
   with the audio the pipeline decodes from the carried audio latents.

The `pair_audio` step goes: stage two produces its own muxed output. The
description says what the three moves are and why the audio is generated in
stage one and only carried through stage two.

**The engine question.** The base step returns a pipeline output object with
two tensors. `previous_result:base` today yields the object's artifacts;
whether `previous_result:base.<field>` can name the video latents and the
audio latents separately depends on the field names `LTX2PipelineOutput`
uses when `output_type` is latent, and on `get_artifact_properties` reaching
them. The plan's first task reads `pipeline_output.py` and tries the
reference in a unit test with a stub output. If the property route works,
no engine change. If it does not, the smallest change that makes it work is
in `dw/result.py` or `dw/previous_results.py`, covered by a test, and named
in the ledger.

`noise_scale` as "the first value of the constant" has no reference syntax
today. The template writes the literal `0.909375` with a comment-bearing
description sentence naming the constant it copies, and
`tests/test_catalog_structure.py` gains a check that the literal equals
`STAGE_2_DISTILLED_SIGMA_VALUES[0]` from the installed library, so a
diffusers change to the schedule fails a test rather than silently
mis-noising.

**Acceptance**: the template validates; a run on lem completes; the refined
output is visibly sharper than the upscale-only output of the previous
template at the same seed (kept side by side in the run's gallery entry);
`cost` is remeasured and written; `RECIPES_24GB.md`'s sharpness sentence is
reinstated only if the comparison supports it.

## 2. The LTX-2.5 prompt library

The six files under `prompts/ltx2/` are rewritten to the trained-caption
format that `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT` in the installed diffusers
specifies, and that the audit summarises as:

- one paragraph, roughly 150 to 220 words, present-progressive verbs;
- opens on the action, never "The scene opens" or "We see";
- for every shot a shot type, a camera motion (stated even when static) and
  a viewpoint, woven into the prose;
- the soundscape interleaved chronologically with the action, not appended;
- observable detail only, plain colour words, no quality tags, no artist
  names, dialogue quoted exactly.

The two prompts that serve image-conditioned templates
(`marmot_robot_overlords`, `polaroid_lighthouse`) follow the I2V variant
instead: describe what changes from the image and do not restate what the
image already shows. `intended_model` becomes `ltx-2.5` on all six, and the
template `summary` fields that say "LTX-2" say "LTX-2.5". The
`enhance-prompt` template's one-line inline idea stays as it is, because
that is the enhancer's input.

Each rewritten prompt keeps its subject and its `description`, so the
gallery and the templates that reference it by name are unaffected.

**Acceptance**: a test in `tests/test_prompts.py` (or a new
`tests/test_ltx_prompt_library.py`) asserts every `prompts/ltx2/*.json` text
is 140 to 240 words, one paragraph, and contains none of a short forbidden
list (`8k`, `ultra-detailed`, `photorealistic`, `vibrant colors`, `The scene
opens`, `We see`). Two of the six are run on lem through the text-to-video
template and looked at.

## 3. The H3 Context-IR builtin

`dw/workflows/h3_context_ir.json`'s `system_prompt` changes in five places,
each traceable to the audit:

1. **`N/A`**: `overall_soundscape` and `non_diegetic_music` are written as
   `N/A` when the brief has no ambient sound or no score, per the base guide.
   Without it the enhancer invents audio for a silent brief.
2. **Reference numbering**: `<Video N>` and `<Audio N>` are numbered
   independently within their own category, and a video reference does not
   by itself produce an `<Audio N>`. The current text says one global order.
3. **Continuity modes relabelled**: the standalone/continuation block stays,
   because dw's chain feature feeds "Continuity: continuation" into the
   enhancer's user message (`resolve_chain_prompts`), but the text says it is
   this engine's convention for writing one segment of a chained take and
   not part of Context-IR.
4. **Dialogue fidelity**: the ref guide's rules that the current text omits:
   `[unclear]` for unintelligible spans, terminal punctuation before `</d>`,
   no `(Sx)` speaker IDs inside `retention_analysis`, and no carrying of
   original dialogue when only a voice's timbre is referenced.
5. **Two unsourced lines removed**: "degrades on anything else" and the
   "no 8k or artist names" rule. The guidance that H3 has no negative prompt
   stays; it is confirmed.

A first line in the system prompt cites the two guides by their paths in the
model card repo, so the next audit starts from a source.

**Acceptance**: a test asserts the system prompt contains `N/A` guidance and
the phrase that relabels continuity as an engine convention; the
`enhance-prompt` template runs on lem with a deliberately silent brief and
the output's two audio fields read `N/A`.

## 4. The two READMEs

`workflows/templates/minimax/README.md` and `workflows/templates/ltx2/README.md`:

- every table link names the file that exists (the templates were renamed to
  kebab-case; the links still name the old ids);
- the LTX README's model link points at the LTX-2.5 repo, not LTX-Video;
- a first paragraph names the vendor sources: for H3 the two prompt-writing
  guides and the `h3-prompt-writing` skill in MiniMax's GitHub repo; for
  LTX-2.5 the model card, Lightricks' `ltx-pipelines` docs, and the
  system-prompt constants in diffusers;
- the H3 README gains one paragraph of model facts the audit found missing:
  the 768-pixel short edge and 32-pixel grid, aspect ratios 1:4 to 4:1, that
  the 5-second floor is diffusers' constraint where the model card says 4,
  and that the 544p turbo LoRA, the 960x544 canvas and nine steps are one
  coupled choice;
- the LTX README's "recommended quality flow" line describes the repaired
  three-move flow and notes that Lightricks' newer DFR pipeline is the
  follow-up.

A catalog-structure test walks both READMEs for relative links and asserts
each resolves, so a future rename fails the suite.

## 5. Ledger and follow-ups

The Part 4 ledger row records what changed, the remeasured costs, and a dated
"missing knowledge" list drawn from both audits so the next model-family pass
starts from it.

## Verification order

1. Unit tests and the catalog-structure suite locally.
2. On lem, on this branch with a server restart (the builtin and any engine
   change need one): two-stage, two rewritten prompts through text-to-video,
   the H3 enhancer with a silent brief.
3. Costs written, ledger updated, PR.
