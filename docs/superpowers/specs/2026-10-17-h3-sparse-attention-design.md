# H3 sparse attention (comfy-kitchen Sol-Attn) - design

Issue: #931. Evidence: the stage-0, prototype, 768p, Ref2VA and skin-recovery comments on #931.

## Intent

Make MiniMax H3 generations 2-4x faster on CUDA by running the generated-video
part of every DiT attention call through comfy-kitchen's `sol_attn` sparse
kernel. Text, reference and audio rows stay exact (dense).

Don's words: sparse is **the normal way H3 runs**, not an opt-in. comfy-kitchen
is installed by default. The UI's attention-backend setting becomes a
dropdown of the backends the server can actually run. The H3 quality ramp
(`plugins/dw/skills/minimax-h3/references/quality-and-speed.md`, *Ramp*) takes
the new backend into account.

Why: H3 is the slowest family dw runs. A 1344x768 Ref2VA final takes ~16 min
dense on lem's 3090s. Sparse cut that to ~7.5 min at a quality Don judged
"really good", indistinguishable from dense apart from skin texture at `tau`
1.0. At `tau` 0.5 he judged the two equal.

## Measured so far (lem, RTX 3090, seed 42)

| shape | dense s/forward | sparse s/forward | wall dense -> sparse |
|---|---|---|---|
| T2VA 960x544, 4-step turbo, tau 1.0 | 24.3 | 10.8 | 150 -> 95 s |
| T2VA 1344x768, 4-step turbo, tau 1.0 | 96.1 | 21.9 | 457 -> 158 s |
| Ref2VA 1344x768, 8-step ref2v turbo, tau 1.0 | 108.4 | 42.1 | 965 -> 430 s |
| same, tau 0.5 | 108.4 | 44.8 | 965 -> 450 s |

Dense here means native SDPA, not `sage_hub`. The ramp's numbers are
`sage_hub`, which is ~30% faster than native per step. The sparse-vs-sage
ratio is not measured yet; the stage 3 measurements take it.

**Peak memory didn't change.** `torch.cuda.max_memory_allocated` was identical
dense and sparse in every run: 6.72 GiB (544p T2VA), 10.81 GiB (768p T2VA)
and 21.16 GiB (768p Ref2VA). The padded tile copies are small next to
group-offloaded weights, so `vram_estimate` ceilings stay as declared. Stage
3 records peak reserved as well, to confirm.

Sparse **moves composition** relative to dense: same subject, story and
cuts, but different framing and set dressing (SSIM 0.75-0.81 against dense).
This is the main way it differs from sage, which the ramp records as "never
moves composition".

## Goals

1. **An engine backend.** A new `attention_backend` value, `sparse`, valid
   wherever `attention_backend` is today: at pipeline level and on a
   component. On a MiniMax-H3 transformer it routes target-video rows through
   `sol_attn` and keeps everything else dense. A `sparse_tau` setting
   (default 0.5) sits next to it.
2. **A dependency.** `comfy-kitchen` becomes a regular dependency in
   `pyproject.toml`, installed by `install.sh` on every platform.
3. **Template defaults.** Every H3 template that takes `attention_backend`
   (19 under `workflows/templates/minimax/`, not the two Music3 ones)
   defaults it to `sparse`.
4. **Graceful fallback.** Where `sparse` can't run, the run uses the
   transformer's normal dense attention and completes. That covers no CUDA
   (MPS, CPU), a card below sm_80, a non-H3 transformer, a dtype or
   head_dim the kernel refuses, or comfy-kitchen failing to import.
   Validate and the run both say so (see *Warnings*).
5. **A UI dropdown.** In the workflow page and editor, a variable that feeds
   an `attention_backend` setting is shown as a dropdown of the backends the
   connected server can run, plus "default (none)". The server reports that
   list.
6. **Ramp.** The H3 skill's ramp is updated:
   - The TV path (L1 draft -> L7 final, plus the L0 sketch) moves to
     `sparse` at both tiers, once stage 3 shows its staging carries.
   - Cinema (B3 -> L5/P1) and close-up (C1) stay on `sage_hub` at both
     tiers until stage 3 is measured.
   - The rule "attention never moves composition" is corrected:
     sage doesn't, sparse does, so draft and final use the same backend.
7. **Measurements (stage 3).** On lem, run the cinema pair B3 -> L5 (base
   model, int8, 1344x768, 12 then 30 steps) with `sparse` against
   `sage_hub`, at seeds 42, 7 and 1234. Record wall time, per-forward time,
   staging carry from draft to final, and Don's skin verdict. Run the TV
   pair L1 -> L7 under `sparse` on the same 3 seeds. Its carry was measured
   under sage, which never moves staging, so it has to be re-shown under
   `sparse`; those runs also give the TV timing. Each path moves to
   `sparse` only on Don's verdict for it.

## Non-goals

- Sparse attention for any family other than H3. LTX-2.5, Kandinsky 6 and
  FLUX 3 have different token layouts. The backend no-ops with a warning
  on them.
- Trained-gate VSA (the FastH3 `to_gate_compress` transplant from the Kablex
  repo) and the `topk` modes. `tau` mode needs no extra weights, and it
  measured well.
- lightx2v Turbo-SLA LoRAs. A separate follow-up issue, filed during this
  work.
- An MPS or CPU sparse path. comfy-kitchen ships only an eager PyTorch
  reference there, which is slower than dense.

## Anti-goals

- **A run must never fail because of `sparse`.** Being the default, a
  `sparse` that can't run must degrade to dense, never refuse at validate
  or at load. A Mac user running an H3 template must get today's behavior.
- **No silent change of staging between a draft and its final.** The ramp
  must never pair a sparse draft with a dense final, or the reverse.
- **No model knowledge in engine code beyond the diffusers class it
  patches.** The backend keys off the transformer class
  (`MiniMaxH3Transformer3DModel`) and the layout tensors diffusers passes
  it. It doesn't key off model names or template ids (CLAUDE.md, *Before
  you edit*).

## Constraints

- **Kernel requirements:** sm_80+ CUDA (or a HIP card with matrix cores),
  bf16/fp16, head_dim 128 (H3's is 128).
  `comfy_kitchen.sol_attn_is_available(device)` is the check.
- **comfy-kitchen 0.2.38** has wheels for Linux x86_64/aarch64 (cp310,
  cp311, cp312 abi3) and Windows, plus a `py3-none-any` wheel elsewhere
  (macOS gets the eager-only build). Apache-2.0. No required dependencies.
- **Signature passthrough.** diffusers' H3 denoiser picks the layout kwargs
  it passes by `inspect.signature(transformer.forward)`. Anything that wraps
  `forward` must keep the signature (`functools.wraps`). Otherwise
  `token_tags`/`position_ids`/indices silently stop arriving
  (prototype-learned).
- **Fractional position ids.** H3's `position_ids` are fractional:
  aspect-normalized space onto [0, 32), and uneven time steps of
  5/3 x (1,4,4,4,...). Tile coordinates come from ranking each axis's
  distinct values.
- **Finding the target video.** Target video is the trailing run of video
  rows: diffusers packs text, references in request order, target audio,
  then target video. A Ref2VA reference image can share the target's
  timestep index, so noise level can't identify it.
- **The process-wide default.** `set_attention_backend` sets a process
  global, and dw restores it today (*Attention backend global leak*,
  `docs/ACCELERATION.md`). `sparse` must not leak onto the next job in the
  persistent worker either: not onto a Music3 job, nor onto an H3 job that
  asked for `native`.
- **CodeQL and security rules** apply as usual. No new file or subprocess
  surface is expected.

## The how, as decided

- **The name is `sparse`** (my call; Don said "sparse"). It's one more
  `attention_backend` value, not a separate key, so the template variable,
  the dropdown, validate and the docs all treat it like `sage_hub`.
- **`tau` defaults to 0.5** (Don's verdict on the Ref2VA skin A/B). It's
  exposed as a template variable `sparse_tau` on the H3 templates (my
  call on the name).
- **The dense prefix:** every row that isn't target video goes first, in
  segment-pure 64-row tiles. Those tiles are `sink_blocks` and `sink_q`:
  always attended, and their queries attend everything. Target video goes
  after, in 4x4x4 tiles. `block_len` carries each tile's live-row count.
  This is the prototype's layout, which passed the full-grid check on T2VA
  and Ref2VA.
- **The fallback is decided once per pipeline call, all or nothing** (my
  call). The backend decides before the first transformer forward whether
  this call runs sparse. If any check fails later in that call, it raises
  rather than dropping that forward to dense: a call that mixes sparse and
  dense forwards would be a hidden staging change inside one generation.
  The prototype's per-call fallback is not ported. The text token refiner
  (text-only rows, no video target) always runs dense; that isn't a
  fallback and isn't reported. Each pipeline call of a chained job decides
  for itself, and the run-time warning names the calls that fell back.
- **The fallback warns instead of refusing.** Being the default, every H3
  validate on mini-ai (MPS) would otherwise carry a warning nobody can act
  on. My call is a distinct, quieter code (see *Warnings*) rather than
  reusing `attention_backend_unavailable`.

## Warnings

- **Validate: `sparse_attention_fallback`.** Validate reports it when
  `sparse` is set and the server's device is not CUDA. It names the device
  and says the run will use dense attention. Every H3 run on a Mac shows it.
  The builder decides whether the catalog/UI surfaces collapse it; it is
  never a refusal.
- **Run time.** The worker's event stream reports once per job when
  `sparse` fell back, with the reason: device, compute capability, dtype,
  head_dim, non-H3 transformer, or import failure. Run-time warnings must
  reach the caller (standing rule from the #82-84 cycle).
- **CUDA list.** `sparse` is **not** added to
  `CUDA_ONLY_ATTENTION_BACKENDS`. It never fails at load, so
  `attention_backend_unavailable` would be the wrong message.

## Server and UI

- **The server reports the backends this device can run.** It lists them on
  an existing system route (`/api/health` or `/api/server`, builder's
  choice): on CUDA the `native` family, the CUDA-only set, and `sparse`
  when `sol_attn_is_available()` holds on at least one card; on MPS/CPU the
  `native` family only. Hub-kernel backends need the `kernels` package and,
  for some publishers, `DIFFUSERS_TRUST_REMOTE_KERNELS`. The builder decides
  whether to filter on those or list them with a note.
- **The UI turns the matching variables into dropdowns.** `VariablesForm`
  gets a dropdown for any variable that feeds an `attention_backend` key in
  the workflow. Whether it detects that by tracing references or by the
  variable's name is the builder's call. The UI has no dropdown for
  variables today.
- **The pipeline-level editor field** (`PipelineOptions.svelte`, a free-text
  `attention_backend` input) uses the same list.
- **Remote servers.** The dropdown reflects the *connected server's*
  device, not the browser's machine (`docs/REMOTE.md`).

## Docs and skill

- **`docs/ACCELERATION.md` *Attention Backends*:** a `sparse` subsection.
  What it does, the measured table, `sparse_tau`, the fallback, and that it
  moves composition relative to dense, unlike sage.
- **`docs/QUANTIZATION.md` / `RECIPES_24GB.md`:** the H3 rows note the
  default backend.
- **`plugins/dw/skills/minimax-h3`:**
  - The ramp and its *Quality ladder* get sparse rows.
  - The *Ramp* path table gets a backend column.
  - The composition rule is corrected.
  - The cinema rows set `attention_backend=sage_hub` explicitly until
    stage 3 lands.
  - Mind the 12 KiB skill byte cap (memory: H3 skill is at its cap). New
    text goes in `references/quality-and-speed.md`, not `SKILL.md`, where
    possible.
- **Template metadata (my call):** stages 1-2 ship with today's
  `cost` minutes. Those are sage/native-era numbers, so they over-quote on
  CUDA, which is the safe direction for the acknowledged-cost gate. Stage 3
  replaces the minutes on the templates its runs cover
  (`reference-to-video`, `dialogue-short`, `video-with-audio-768p`).
  The others keep their minutes until observed cost (`dw/server/observed_cost.py`)
  or a later measurement corrects them; no per-template GPU runs. New
  minutes come from measurements, never from scaling. `attention_backend`
  is already in `cost_drivers` on the ones I read; the builder confirms it
  on all 19.
- **`plugins/dw` versioning and its pinned tests** follow
  `plugins/dw/README.md`.

## Stages

0. **Dependency.** comfy-kitchen in `pyproject.toml`, `install.sh` and
   `docs/DEPENDENCIES.md`. Deploy to lem and mini-ai, and confirm it imports
   on both.
1. **The engine backend.**
   - `sparse` recognized at pipeline and component level, so the value
     never reaches diffusers' `AttentionBackendName`.
   - The H3 processor and tiling, ported from the prototype with its
     checks. The full-grid and injectivity asserts become errors that
     trigger the fallback.
   - The fallback, decided once per pipeline call; the warnings; no global
     leak.
   - Unit tests on CPU: tiling for T2VA and Ref2VA layouts, the
     target-video rule, the fallback on non-CUDA, the signature
     passthrough, and no leak to the next job.
   - Then a lem GPU smoke.
2. **Templates, UI, docs.**
   - The 19 template defaults and the `sparse_tau` variable.
   - The server's backend list and the UI dropdowns.
   - `docs/ACCELERATION.md`.
3. **Measurements (lem), then the ramp.**
   - Cinema B3 -> L5, sparse vs `sage_hub`, 3 seeds.
   - TV L1 -> L7 under `sparse`, the same 3 seeds: staging carry and timing.
   - Peak reserved memory recorded on every run.
   - Don judges skin and staging.
   - The ramp and template `cost` updated from the numbers.

## Left to the builder

- Where the backend code lives (a new `dw/` module is likely). How it
  attaches to the transformer: a processor swap on load, or a
  `set_attention_backend` interception.
- Caching the tile map per layout signature (the prototype keyed on
  shapes and sums), and its memory cost.
- Whether the pipeline-level and component-level `sparse` share one code
  path.
- Test fixtures for H3 layout tensors without loading the model.
- Reconciling the ratchets (modules, `test_dw_patch_targets`, catalog
  budgets) as they move.
