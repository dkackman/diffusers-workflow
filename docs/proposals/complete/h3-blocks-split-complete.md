# Split `h3_blocks.py`, and pin the H3 copies of diffusers (#691)

Written by model `claude-opus-5-5` via provider `anthropic`, at close-out
2026-10-08, from plan v1 on #691 and the stage threads (#770, #771). Plan v1
approved by Don 2026-10-08 with every default.

## The ask

The 2026-10-07 develop review (structural item 1) found
`dw/pipeline_processors/h3_blocks.py` (about 1,520 lines) holding four
concerns: the hold/refine/release blocks, the guide layout, pure validation
rules, and `hold_audio_reference` argument handling. `dw/guides.py` imported
the private `hold_audio._not_h3`. It proposed three stages: split the file,
move the guide-chain lengths (`guide_frames` 22/39) into
`chained-segments.json`'s `variable_constraints`, and add drift tests
against the installed diffusers.

Verdict: **build smaller.** Build the drift tests and the split, in that
order. Cut the catalog move.

Don's answers (2026-10-08): Q1, cut the catalog move - **yes, cut**; the
engine keeps `GUIDE_CHAIN_FRAMES` in `h3_rules.py`. Q2, a GPU run for the
split - **none**. Q3, a re-export shim for `h3_blocks.py` - **none, delete it**.

## Why the catalog move was cut

- `variable_constraints` reaches only top-level variables and `for_each`
  entry fields, and has no `one_of`. {22, 39} can be written as
  `modulus 17 / remainder 5 / min 22 / max 39`, but a literal
  `"guide_frames": 39` in any chain block never meets that constraint, and
  neither does any workflow other than `chained-segments`.
- The run-time `ChainConfig` can't see constraints at all.
- So the engine has to keep `guide_chain_problems` and its constant, and a
  template entry would be a second copy of the rule, not a move. The
  issue's own acceptance ("no H3 numeric constant in `dw/` outside a module
  the map names as the H3 adapter") is met by the split: `h3_rules.py` is
  that module.

## What was built

| Stage | Issue | Merge on `develop` | What it did |
|---|---|---|---|
| A drift tests | #770 | `2ff1b0cb` (merge of `b4442cbb`) | `shifted_sigma_grid` factored out of `refine_sigmas` (same operations and order, so the hand-pinned values still hold) and pinned to stock `MiniMaxH3Scheduler.set_timesteps` over N ∈ {2,5,6,8,30,50} and three shifts (`TestSigmaGridDrift`). `GUIDE_FRAMES_PER_CHUNK`/`GUIDE_LATENTS_PER_CHUNK` pinned to `AutoencoderKLMiniMaxH3`'s config defaults (`clip_length`, tokens per clip), plus `guide_latent_frames` against the VAE's encode length (`TestChunkDrift`). `_fill_audio_positions` added to `LAYOUT_ANCHORS`. Each pin was shown to bite by perturbing it. |
| B split | #771 | `f7bbd347` (merge of `d4e0dece`) | `h3_blocks.py` deleted with no shim, replaced by `h3_rules.py` (names and pure rules, torch- and diffusers-free, public `not_h3`, `CHAIN_CONTINUITY_MODES`, the chunk constants), `h3_hold.py` (hold/refine blocks, `encode_audio_span`, `core_denoise_sequences`, `shifted_sigma_grid`, `hold_audio_reference`) and `h3_guides.py` (layout blocks, `LAYOUT_ANCHORS`, `insert_guides`, `guides_refusal`), importing `h3_hold` one way. `guide_chain_problems`' message is built from `GUIDE_CHAIN_FRAMES`, text unchanged. Every caller, test patch and comment updated; new `tests/test_h3_rules.py::test_h3_rules_imports_neither_torch_nor_diffusers`. |

The map row *H3 audio hold, refine and guide layout* in
`docs/ARCHITECTURE.md` names the three modules, the one-way import and the
drift pins; both stages updated it in the same change.

The tester wrote four regression cases, all "nothing moved" checks over
MCP, and all passed on mini-ai (mps): **M-F113** (`get_pipeline_signature`
still lists `hold_audio`, `refine_strength`, `guides`), **M-F114**
(`guide_frames` 22/39 accepted, six other values refused verbatim, no job
queued), **M-F115** (`hold_audio`/`refine_strength` refusals and accepted
values) and **M-F116** (`guides` clip-count and frame refusals). M-F113 and
M-F114 were re-run at stage B. Both architecture reviews passed with no
findings.

### Deviations the builds reported

- **A:** the VAE chunk pin reads the class's config defaults, not a loaded
  model's config; the plan named this as the fallback. An extra test-only
  check of `guide_latent_frames` against the VAE encode length was added.
- **B:** `dw/adapter_compatibility.py` now imports `MEMBER_SEPARATOR` and
  `render_path` from `references` (their owner) instead of the `for_each`
  re-export, because `for_each` pulls in torch through `step_cache`. Same
  objects, no behaviour change.
- **B:** the torch-free probe registers `dw` and `dw.pipeline_processors` as
  bare packages, since `dw/__init__` imports torch on purpose. It checks the
  module's own import graph, and was shown to fail with the
  `adapter_compatibility` fix reverted.
- **B:** four `finally:` resets of `h3_guides._GUIDE_BLOCKS` in
  `test_h3_guides.py` are redundant teardown (monkeypatch restores the
  attribute anyway); they were equally redundant before the split and were
  left as they were.

## Bounces

- **#770: none.**
- **#771: none.**

Cost is left out: the stage comments don't record `usage:` figures. The
plan estimated A at about $2-3 and B at about $3-5.

## Deferred, and why

- **`guide_frames` in `chained-segments.json`** - cut (Q1). The plan's
  fallback, a visibility-only constraint held to the engine by a pin test
  (about $1), was not taken. *Would come back if* `variable_constraints`
  gains a reach into chain blocks, or a `one_of`.
- **`dw/tasks/h3_latent_upscale.py`'s constants** - out of scope; a separate
  task module.
- **Validation import cost.** Validation still imports torch: `dw/__init__`
  does so on purpose, and `dw/guides.py` imports `default_num_frames` from
  `h3_guides`, which loads torch. Only `h3_rules` promises to be torch-free.
- **GPU run of the split** - none (Q2). The unit suite builds every block
  against the installed diffusers.

## Design corrections found against the issue

- "Validation-time imports no longer drag the lazy block factories" bought
  nothing: `dw/__init__.py` and `dw/validation.py` (via `voice_attribution`)
  import torch eagerly. The split is justified by navigation and ownership.
- `refine_sigmas` needs torch, so it went to `h3_hold.py`, not the rules
  module the issue named for pure checks.
- `_fill_audio_positions` was used by the guide layout but missing from
  `LAYOUT_ANCHORS`; stage A added it.
