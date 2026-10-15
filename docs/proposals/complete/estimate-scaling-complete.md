# Scale the curated estimate instead of dropping or repeating it (#866)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-15). Plan v4, approved by Don as "build smaller" on 2026-10-14 (v1 → v2:
Q1-Q5 and the fingerprint ruling. v2 → v3: the stage 0 re-plan. v3 → v4: two
fixes to stage A's acceptance). Four stages: #905 (0), #906 (A), #907 (B) and #908 (C),
all verified on lem.

## The report

`POST /api/validate` quoted ~42 min for the ep 6 dialogue-short no matter what
changed. The H3 quality matrix (lem, workspace `h3-quality`, 15 rungs of one
Ref2VA shot, 124 frames, 4 refs) measured the same shot at 3.5 to 44.2 min,
depending on canvas, weights, LoRA, steps, attention and cache. Agents quote
`plan.estimate` before every run, so the quote was off by up to 12x in either
direction. It was the tenth estimate bug in this class (#267, #255, #301,
#589, #590, #593, #796, #821, #851).

The code check (plan v1, at a27ad5aa) found that the flat figure had three
sources, not one:
- **Catalog path.** A move of steps, canvas or frames off the defaults made
  `_own_price` return `basis: unknown`. It was not scaled.
- **Default bucket.** Runs that differed only in weights, LoRA, attention or
  cache landed in it, because those were not drivers.
- **Inherited path.** `_scale_past_template` scaled by pipeline-step count
  and frames only, not by `num_inference_steps` or the canvas.

## What was built

- **Stage 0 (#905): the fingerprint and step cache are keyed only by what
  changes the output.**
  - `cost_drivers`, `vram_estimate` and the other documentation and cost
    metadata no longer reach the plan fingerprint or the step cache keys. The
    exclusion list is pinned by a test.
  - Integral floats are canonicalized in the fingerprint, so `12.0` and `12`
    digest alike. That was the inline vs. by-name mismatch.
  - Seed stays out of the fingerprint (the #85 contract). The step cache keys
    on it.
  - The step cache keeps `subfolder` and `file_base_name`, because a hit
    replays files to their recorded paths.
- **Stage A (#906): the catalog figure scales.**
  - `scale_curated` gives
    `minutes = curated × steps/steps₀ × (w·h)/(w₀·h₀) × frames/frames₀` on the
    four generic driver names (`num_inference_steps`, `width`, `height`,
    `num_frames`). `num_frames` is summed across a list's entries.
  - The result is `basis: "derived"`, and `scaled` names each axis that moved.
  - It multiplies with list-length repricing (`_repriced`). `_tempered` blends
    toward the scaled figure.
  - A non-numeric driver value still gives `unknown`.
  - The rule "a `cost` entry is measured at the workflow's variable defaults"
    is written in the schema and WORKFLOW_GUIDE.
  - dialogue-short's `cost` was re-priced from 42 to 21.2 for its 960x544
    defaults.
  - The 30% accuracy bar is `TestScalingAccuracy`: B4→B1 16.5 vs 15.1, B4→B5
    13.0 vs 11.8, B3→L5 32.3 vs 29.5.
  - The UI's `derived` wording shows `scaled`.
- **Stage B (#907): unscalable drivers.**
  - Any other declared scalar driver that moves off its default keeps the
    (scaled) figure, sets `low_confidence` and is listed in `differs`. This
    holds for every template, with no engine allow-list.
  - The H3 templates add `weights_dtype`, `lora_model_name`,
    `attention_backend` and `cache` to `cost_drivers`, so an int8 or native
    attention run no longer pools with the default bucket. LoRA buckets by its
    exact name.
  - Thanks to stage 0, the new drivers leave H3 fingerprints unchanged.
- **Stage C (#908): the inherited path scales the same way.**
  - `_scale_past_template` reuses `scale_curated`'s steps and canvas ratios,
    so there is one owner and inline and template quotes agree (C-F484:
    within 2%).
  - `inherited_differences` compares attention, cache and LoRA too, through
    `replace_variables` (no second owner of substitution).
  - A reference the definition doesn't declare compares as null.
  - Numbers compare by value: `same_number` moved to `dw/variables.py` and is
    shared with the fingerprint.

## What was cut or deferred, and why

- **Compile as a cost driver.** No H3 template exposes a `compile` variable,
  and a driver must be a declared variable. It comes back if a template
  exposes compile.
- **A learned fixed + passes × s/pass fit.** Deferred by Don. Plain
  proportional scaling over-quotes by about 9-10% on the matrix pairs,
  because it ignores load, encode and decode time. It is also slightly
  under-superlinear in pixels (2.27x measured for 1.98x pixels). Both are
  inside the 30% bar. A miss past 30% is the trigger to build the fit.
- **`passes = steps − 1`.** That is H3 scheduler knowledge, so it can't live
  in the engine. The engine uses a plain step ratio (Q2).
- **Price precedence for a saved copy with its own `cost` and no runs**
  (catalog vs. inherited) is unchanged (v3 review).
- **The step cache across identities.** A renamed or inline copy can't hit
  the cache: it is per card and cleared on an identity switch. That is a
  memory-policy change, filed as #912 (idea).
- **The validate probe ignores the card's warm identity**, so it can report
  `cached_steps` that the run's identity switch then clears. Filed as #911
  (bug).

## Bounces per stage

| Stage | Verify bounces | Architecture bounces | Notes |
|---|---|---|---|
| #905 (0) | 1 | 0 | The bounce showed that v2's "a metadata-only copy hits the step cache" can't hold under the cache's per-identity scope. Re-planned as v3 (in-place edit only), plus the integral-float fix. |
| #906 (A) | 0 | 0 | |
| #907 (B) | 0 | 0 | |
| #908 (C) | 3 | 2 | Architecture: a second owner of the scaling rule, then a second owner of `variable:` substitution. Both were folded into the single owners. Verify: C-F485 `differs: ["LoRA"]` at the defaults, three times. Parked after the 4th hand-off; Don ruled that the plan stands. The causes were an undeclared null-default variable and then `lora_scale` 1.0 vs 1 after a client's JSON round trip. |

Cost per stage: the stage comments record no `usage:` figures, so none is
given here. The plan estimated ~$2 / $5 / $3 / $4.

## Rules this feature added

- A `cost` entry is measured at the workflow's variable defaults.
- The plan fingerprint and the step cache are keyed only by what can change
  the output (pinned in `docs/ARCHITECTURE.md` and its tests).
- A number is its value, not its spelling, in the fingerprint and in the
  inherited-cost comparison.
