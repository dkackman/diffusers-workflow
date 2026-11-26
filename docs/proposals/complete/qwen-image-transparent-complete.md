# Qwen-Image-2.1 transparent (RGBA) template (#1681)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-25). Don filed the idea on 2026-11-25 from a reading of
[janishar/helmstudio](https://github.com/janishar/helmstudio) (Apache-2.0,
alpha; concept only, no code taken). Plan v2 on #1681 is the full design. This
file records what shipped, what was deferred or cut, and why.

**Result:** one template shipped, `workflows/templates/qwen/transparent.json`.
The Turbo catalog entry is deferred with triggers, and the PE prompt enhancer
is cut.

## The problem

Qwen-Image-2.1 is the only catalog model that natively makes RGBA images. It
needs two things: the vendor's prompt wrapper and a PNG result. The 2026-09-20
audit recorded the wrapper (`docs/proposals/audits/2026-09-20-qwen-image-2.1-audit.md`)
but no template taught it. The issue also asked about a
`Qwen/Qwen-Image-2.1-Turbo` entry and the vendor's PE-T2I/PE-I2I enhancers.

## Verdict: build smaller

**The plan corrected the issue:**
- **Licence.** Turbo is under the Qwen Research License (non-commercial, §2(b)),
  the same as our base entry. helmstudio's "Apache" label is wrong.
- **`sample_sigmas`.** diffusers 0.41.0.dev0 honours it. With no `sigmas`
  argument it reads `self.config.sample_sigmas`
  (`pipeline_qwenimage21.py:732-735`), so no engine work is needed.
  `num_inference_steps` has no effect on Turbo.
- **Cost.** Turbo takes 8 steps against the base's 40, but it is a separate
  ~32.5 GB download that shares no files with the base.

Don ruled on 2026-11-25, agreeing with the delegate's recommendation: build
the template (stage B) now, defer Turbo (stage A), and cut the enhancer.

## What shipped

**Stage B, #1709.** Commit `929cbca4`, merge `bf6f8b57` on `develop`.
- `workflows/templates/qwen/transparent.json`, id `QwenImage21Transparent`, on
  the base `Qwen/Qwen-Image-2.1`. It needs no new download.
- A `compose_text` step, `wrap`, builds the prompt the same way
  `dw/workflows/flux3_klein_upsample_edit.json` composes text: `This is an RGBA
  image with transparency. {subject}. The image has alpha channel and the
  background is transparent.` The pipeline's prompt is `previous_result:wrap`.
  No new syntax was added.
- The result is `image/png` (`subfolder: final`). The description says a jpeg
  result flattens the alpha onto white, names the non-commercial licence, and
  warns against a trailing period on `subject`.
- Defaults are 1024×1024 at 40 steps. `width` and `height` are on the 32-px
  grid, so an off-grid value is refused at validate. A `seed` variable is
  declared. There is no `num_images_per_prompt`, so each run makes one cut-out.
  `cost_drivers` are `num_inference_steps`, `width` and `height`. There is no
  `cost` block yet.
- Tests are in `tests/test_qwen_catalog.py`:
  - the template's `wrap` step, run through `Workflow.run`, gives the exact
    wrapped prompt;
  - the result is `image/png`;
  - validation is clean at the defaults and refuses `width=1000`;
  - the listing derives `shape: image`.
- `COMPACT_BUDGET` went from 15_775 to 15_875, with a history line. A row was
  added to `tests/test_step_cache.py` `TEMPLATE_PIPELINE_KEYS`.

**Verification.**
- M-F496–M-F498 passed over MCP on mini-ai (listing, validate, prompt
  composition).
- M-F499 passed on lem (RTX 3090), as Q2 required, on the cached base. The
  seeded run (seed 1709, "a red apple with a green leaf") took about 68 s and
  gave a 1024×1024 `mode: RGBA` PNG. The embedded prompt was exactly the
  wrapped sentence. The output showed the apple alone with unpainted corners,
  plus a very faint residue in two corners.
- Limit: no MCP call reads alpha values, so "transparent" was judged from the
  viewer, not measured.

**Bounces:** none. Stage B verified on its first hand-off.

**Cost:** the stage comments carry no `usage:` figures, so none is recorded.
The plan estimated $2–3.

## Deferred: stage A, the Turbo catalog entry

It was not filed. Any one of these brings it back as a new plan version, with
plan v2's stage A design unchanged:
- #1625 or #1626 asks for a fast Qwen rung;
- a field report finds `loras/qwen-image-2.1/turbo8.json` falls short on
  quality;
- a Qwen-Image-2.1 workload shows up on lem.

**The design kept for the reopen:** `workflows/models/qwen-image-2.1-turbo.json`
is a copy of the base entry with these changes:
- `model_name: Qwen/Qwen-Image-2.1-Turbo` and id `QwenImage21Turbo`;
- no `num_inference_steps` variable or argument, and
  `cost_drivers: [num_images_per_prompt, width, height]`. A steps driver would
  price an ignored value at 5×, in `dw/cost_scaling.py:113-115,161-164` and in
  inline pricing at `dw/server/deps.py:405-417`. Follow Krea2;
- no `cost` block at landing;
- a description naming the fixed 8 steps, the licence, the ~32.5 GB download
  and turbo8 as the alternative;
- a test pinning the no-steps-driver rule.

## Cut: the PE prompt enhancer

It was cut for these reasons:
- A third enhancer preset triggers #696's rule that the menu moves into
  builtin metadata (`dw/server/enhancers.py:9-13`).
- The rewriter is ~20 GB, a Qwen3.5-VL 9B.
- Its JSON output (`rewritten_prompt`, `wh_ratio`) has no consumer in dw.
- The 400-token task path would truncate its 400–500-word prompts.
- It degrades without thinking mode.

**Reopen** on a field report where a Qwen-Image-2.1 prompt underperformed for
lack of the vendor convention, or when #696's metadata menu lands. It would then
be a builtin following `dw/workflows/flux3_klein_upsample*.json`.

## Non-goals that stayed out

The unsloth FP8 files, an edit or multi-reference Turbo entry, a `plugins/dw`
Qwen skill, and any engine, MCP, REST or syntax change.
