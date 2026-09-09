# Proposal: onboarding Qwen-Image and Wan2.x into the catalog

Status: draft, 2026-09-08. No implementation.

## Why these two

Neither has a template under `workflows/models/`. Both rank among the most
downloaded, most actively-developed open models in their category as of
2026-09-08 (Hugging Face trending listings, pipeline tags
`text-to-image` / `text-to-video`):

- **Qwen/Qwen-Image** (20B) — 314k downloads, plus `Qwen-Image-2512` (71.8k)
  and the widely-used `Qwen-Image-Edit` sibling. Sits above every FLUX
  variant except `FLUX.1-dev`/`schnell` themselves.
- **Wan-AI/Wan2.x** — the dominant open video base by LoRA-ecosystem size;
  the trending video listing is mostly MiniMax-H3 and Wan2.x
  LoRAs/quantizations (`Wan2.2-T2V-A14B-GGUF`, `Wan2.2-TI2V-5B`,
  `Wan2.1-T2V-1.3B`), which is itself a signal of how much community tooling
  targets it.

That popularity is the whole argument. Nothing about either model is
currently represented in the catalog, so this is greenfield onboarding, not
repair — the `model-family-onboarding` skill's audit step (sources table,
vendor prompt format, reading-order for templates) stands in for the
"contradicted claims" step, since there are no existing claims to check.

## What's already established (from a first pass, not a full audit)

Both fit a 24GB card, via the same quantization/offload primitives the
catalog already uses (`flux2-dev.json`'s BitsAndBytes text-encoder pattern,
`group_offload`, GGUF support in `config_objects.py`) — no engine work
implied:

- **Qwen-Image**: bf16 doesn't fit; 4-bit BitsAndBytes (`nf4`,
  `bnb_4bit_compute_dtype=torch.bfloat16`) brings it to ~17GB with `offload:
  model`, mirroring `flux2-dev.json` rather than needing anything new.
- **Wan2.x 14B**: bf16 doesn't fit; the proven 24GB path is GGUF or FP8
  quantization plus block/group offload, or `sequential` offload at a real
  speed cost (community reports of tens of minutes per 5s clip on aggressive
  CPU offload). Wan2.2 also ships a **5B TI2V** variant that likely fits far
  more comfortably in bf16 or light quantization — a cheaper starting point
  if the 14B's quality isn't the point of a first template.

Neither of these is load-bearing yet — they're first-pass web findings, not
audit citations. The audit step confirms them against the diffusers pipeline
source and the vendor's own hardware notes before anything is written as a
template default.

## Does either need a composition skill?

Undecided by design — that's what the audit is for — but the shape of the
question differs between them:

- **Qwen-Image** (text-to-image) looks like FLUX/Z-Image/Krea: a flat
  text-to-image shape with no chaining or frame arithmetic. On that evidence
  alone it likely needs a template, not a skill. **Qwen-Image-Edit**
  (instruction-based image editing, sometimes multi-image) is the piece that
  would carry real shape decisions and a vendor prompt convention worth
  deferring to — if it's in scope, it's the reason a skill might be
  worthwhile, not the base text-to-image model.
- **Wan2.x** (video) is exactly the class LTX-2.5 and MiniMax H3 already got
  skills for: t2v vs i2v, first/last-frame conditioning, extending a clip,
  frame-count and resolution constraints tied to the vendor's own bucket
  rules. A skill is plausible here on the same grounds as the existing two,
  pending what the audit finds.

## Proposed sequence

Two independent onboardings, run through `model-family-onboarding`
(`.claude/skills/model-family-onboarding/SKILL.md`) one at a time rather than
together, since each produces its own audit file, template(s), and (maybe) a
plugin skill:

1. **Audit** — one research agent per family against the vendor's model
   card(s), GitHub repo, and the diffusers pipeline source
   (`QwenImagePipeline` / `QwenImageEditPipeline`; `WanPipeline` /
   `WanImageToVideoPipeline`). Confirms the VRAM numbers above, the
   resolution/step/guidance defaults the vendor recommends, and — for
   Qwen-Image-Edit and Wan2.x — the prompt conventions and any frame/aspect
   constraints analogous to H3's `17n+5` or LTX's `8k+1`. Saved as
   `docs/proposals/audits/<date>-qwen-image-audit.md` and
   `<date>-wan2.x-audit.md`.
2. **Decide scope** — brainstorm with the user per family: which variant
   (Qwen-Image alone, or with Edit; Wan2.2 5B TI2V, 14B, or both) and which
   quantization path, so the template set doesn't sprawl into every
   checkpoint variant the way the pre-cleanup catalog did.
3. **Template(s)** — under `workflows/models/`, configuring
   `templates/text-to-image` (Qwen-Image) and whichever video template shape
   Wan2.x's t2v/i2v split calls for, following the existing FLUX/H3/LTX
   templates as the pattern.
4. **Skill, if the audit supports one** — `plugins/dw/skills/<family>/SKILL.md`
   per the outline in
   `docs/superpowers/specs/2026-09-07-dw-plugin-skills-design.md`, numeric
   rules pinned to diffusers by `tests/test_plugin_skills.py`, same as H3 and
   LTX-2.5. Skipped for Qwen-Image base text-to-image unless Edit is in
   scope.
5. **Cold drill** — for whichever family gets a skill, the same
   plugin-installed-vs-not comparison the H3/LTX-2.5 drill used.
6. **Ledger** — a row per family in the Part 4 ledger
   (`docs/proposals/agent-catalog-legibility.md`), since this extends that
   proposal's catalog rather than opening a new one.

## What this is not

- Not an engine change. Both fit `offload`, `group_offload`, BitsAndBytes,
  and GGUF as they exist today.
- Not a commitment to Qwen-Image-Edit, a specific Wan variant, or a skill for
  either — all three are audit outputs, not premises.
- Not urgent relative to each other — nothing here requires doing both at
  once; treat as two backlog items that happen to share a rationale.

## Open questions

- Whether Qwen-Image-Edit is in scope for the first pass or a dated
  follow-up — it's the part of the Qwen family that would actually justify a
  skill.
- Which Wan2.x variant to template first: 5B TI2V (cheap, likely bf16-only)
  versus 14B (better quality, needs GGUF/FP8 + offload) — possibly both, as
  FLUX has `flux-dev` and `flux2-dev` at different cost tiers.
- Whether Wan2.x's LoRA ecosystem (camera-motion, acceleration LoRAs seen in
  the trending listing) is worth a template variant the way H3's turbo LoRA
  got one, or is out of scope for a first cut.
