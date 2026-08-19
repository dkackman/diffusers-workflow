# MiniMax-H3 on 24GB: An Assessment

An expert review of how the H3 workflow family is implemented and optimized —
what the levers are, whether they are the right ones, and what is still on the
table. Based on reading the diffusers 0.40.0.dev0 modular pipeline, the SDNQ
0.2.4 and group-offloading code actually installed in this venv, all thirteen
`MiniMaxH3*.json` examples, and dw's pipeline machinery. Three microbenchmarks
were run on the idle RTX 3090 (PCIe transfer modes, SDPA at H3's exact
attention shape); no pipelines were loaded.

**Verdict up front:** the landing spot is fundamentally sound. Every major
lever is the correct one for this model on this card, and several dw-side
inventions (`residency: on_demand`, the modular cache context, per-partition
`workflow=` loading) are exactly what the situation calls for. The remaining
opportunities are second-order: one configuration flag that measurably fights
the hardware (`low_cpu_mem_usage: true`), one structural waste in the text
encoder (13 dead Qwen layers), and a couple of levers not yet pulled
(attention backend, streamed text-encoder offload). Details and numbers below.

---

## 1. The problem being solved

The checkpoint is far bigger than the card on every axis:

| Component | On disk | Under the current config |
| --- | --- | --- |
| `transformer` / `transformer_ref` (each) | 66.3 GB (~31B params, mixed bf16/fp32) | SDNQ int4 → ~17–18 GB, streamed from host RAM |
| `text_encoder` (Qwen3-VL-32B, 64 layers + vision tower) | 66.7 GB | SDNQ int4 (LM only) → ~12–13 GB, leaf-offloaded |
| `vae` | 10.4 GB | SDNQ int8 → ~5 GB, on-demand resident |
| `audio_vae` | 0.6 GB | on-demand resident |
| Hardware | | RTX 3090: 24 GB VRAM, PCIe 4.0 x16, 64 GB host RAM |

No partition fits by itself. Everything below is about which component earns
VRAM residency, and at what precision, and how the rest streams.

## 2. How the modes map onto the machinery

Diffusers ships H3 as one modular pipeline with three block workflows —
`t2va`, `fl2va` (keyframes), `ref2va` (typed references, its own transformer
partition) — auto-selected at call time from the inputs. dw's examples cover
the full matrix and are consistent about which lever varies per mode:

| Example | workflow | steps | cache | LoRA | transformer offload group |
| --- | --- | --- | --- | --- | --- |
| MiniMaxH3 (t2va), EnhancePrompt | t2va | 9 | — | turbo | block_level ×2 |
| I2V, L2V, FL2VA, I2VChained, I2VEnhancePrompt | fl2va | 9 | first_block 0.1 | turbo | block_level ×5 |
| Ref2VA, Ref2VAVideo, Ref2VAChained(+Video) | ref2va | 20 | first_block 0.1 | — | block_level ×1 |
| Ref2VAChainedAligned | ref2va | 9 | first_block 0.1 | turbo (fl2v) | block_level ×1 |
| GeneratedSubject | t2i step → ref2va | 20 | first_block 0.1 | — | block_level ×1 |

The per-mode variation is correct in direction: ref2va carries the most
conditioning rows (reference latents ≈ +1.9 GB per carried segment per
`RECIPES_24GB.md`), so it gets the smallest offload groups; fl2va carries one
or two frames and affords groups of 5; the two-keyframe FL2VA and L2V examples
drop the canvas to 544×544, which is the right lever because sequence length —
not weights — is what grows with conditioning.

Chaining (`chain.py`) composes on top rather than inside the pipeline:
last-frame or last-segment continuity, audio-matched segment planning with
`17n+5` frame snapping (which matches the VAE's chunk contract exactly), and
disk spill so memory stays bounded per segment. This is the right layer for it
— the diffusers pipeline stays stock.

## 3. Lever-by-lever assessment

### Quantization — correct, and correctly surgical

- **SDNQ int4 with `quantization_device: "cuda"`, `return_device: "cpu"`** is
  the right framework choice on Ampere: no fp8 hardware, bitsandbytes NF4
  wouldn't give the int8 triton matmul path, GGUF would fight the modular
  loader. Quantize-on-GPU/return-to-CPU keeps load time sane without VRAM
  spikes (and `loading_device()` correctly materializes on CPU first).
- **`modules_to_not_convert` is a superset of the model's own
  `_keep_in_fp32_modules`** (`proj_in`, `audio_proj_in`, `time_embedder`,
  `proj_out`, `audio_proj_out`) plus the genuinely sensitive extras
  (`token_refiner`, `context_embedder`, `norm_out`, `time_proj`). This is the
  detail most ports get wrong. H3's AdaLN activates the timestep embedding in
  fp32 *because* every block reads the same `temb` and rounding bias
  accumulates coherently across the trajectory — the exclusion list preserves
  that contract.
- **`use_quantized_matmul: true`** on the transformer: int4 is a packed dtype,
  so SDNQ re-quantizes to int8 for the triton matmul — the intended path, and
  the main reason step compute is tolerable (roughly halves the ~13 s of
  bf16-equivalent linear work per step on a 3090).
- **Text encoder: `.model.visual` and `lm_head` excluded.** Keeping the vision
  tower in bf16 is right — it encodes the reference images that identity
  preservation depends on. But see §5.2: `lm_head` is *never executed at all*
  (the encoder path reads `hidden_states[50]`), so excluding it from
  quantization just keeps 1.5 GB of bf16 in host RAM for nothing — it could be
  deleted, not merely spared.

### Offload topology — the right three-way split

The mixed strategy is the correct reading of each component's call pattern:

- **Transformer: `block_level` + `use_stream` + `record_stream`.** Called once
  per step; streaming groups with prefetch is the only viable placement for an
  ~18 GB (quantized) model. `record_stream: true` with streams is the faster
  and correct setting (the repo history shows this was flipped both ways —
  where it landed is right).
- **`text_encoder.model` (dotted path): `leaf_level`.** Offloading the inner
  transformers model rather than the wrapper is necessary — and the diffusers
  encoder block fires the top-level `_hf_hook` by hand precisely because H3
  calls `text_encoder.model(...)` directly. dw's dotted-path
  `get_component()` is what makes this configurable at all.
- **VAEs: `residency: "on_demand"`.** This is dw's own mechanism and it is
  the standout correct call. Group-offloading a VAE is *worse than resident*
  under tiled decode (the whole model restreams once per tile); full residency
  wastes ~5 GB on something called a handful of times. Whole-model
  move-around-the-call, with `functools.wraps` preserving the signature the H3
  denoiser introspects, and `empty_device_cache()` handing the space back — the
  documented result (23.2→18.9 GiB peak, 40→0 allocator retries, ~1% wall
  time) matches what the mechanism should deliver. This is the piece I would
  not change.

The supporting order-of-operations in `Pipeline.load()` is also right and
subtle: quantize → LoRA → cache hooks → *then* group-offload hooks and
placement, so hooks wrap final weights; and the wholesale `pipeline.to(device)`
is suppressed whenever any component offloads (`has_component_group_offload`).

### Cache — correct wiring, one inconsistency

`first_block` at `threshold: 0.1` is the sensible cache for a
guidance-distilled model (no CFG pass to exploit, so FBC's step-similarity
skip is what's available). Two dw-side fixes make it *actually work* here and
both check out against the installed diffusers:

- `stateful_cache_context()` — ModularPipeline is not a DiffusionPipeline and
  never sets the cache context; without this wrapper FBC dies on step one and
  would otherwise leak residuals between REPL runs. Correct, including the
  `_reset_stateful_cache()` on the way out.
- `cache_blocks.json` registers `MiniMaxH3TransformerBlock` with
  `return_encoder_hidden_states_index: null` — verified right: the H3 block is
  single-stream (video+audio+text ride one packed sequence), unlike LTX-2's
  dual-stream blocks which need the argument remap.
- `get_cache_transformer()` knowing about `transformer_ref` is what turns
  caching on for ref2va at all.

The inconsistency: **the two t2va examples don't enable the cache** while every
conditioned example does. If that's deliberate baseline-minimalism it deserves
a comment in the JSON description; if not, it's a free ~1.3–1.8× on the mode
with the fewest other accelerations (t2va also runs at group size 2, so it's
the slowest per-step config in the family).

### Turbo LoRA and schedules — pragmatic, worth an eye

The lightx2v repo ships only `fl2v` distills (8-step bf16, 4-step). Using the
8-step LoRA at 9 steps on fl2va is on-label. Using it on **t2va** (baseline,
EnhancePrompt) and on **ref2va** (ChainedAligned, on `transformer_ref` — a
different checkpoint partition) is off-label; the 20-step no-LoRA ref2va
examples are the honest quality path and it's good both exist. No example
touches `scheduler.shift` (checkpoint: video 12.0, audio 3.0) — right for
20-step runs, and evidently validated at 9 steps with the LoRA, but the
machinery's own docstring notes few-step schedules fight shift 12; if 4-step
generation is ever attempted, the `set_shift` support already built into
`load_and_configure_scheduler` will be the lever.

### What's deliberately absent — also correct

- **`torch.compile`**: right to omit. Streamed group offload swaps tensor
  storages under the graph, SDNQ's matmul already runs through its own triton
  compilation, and cache hooks force graph breaks. The one plausible variant
  (regional compile, no streams) would trade away prefetch overlap.
- **Pipeline-level `offload: "model"`**: meaningless for a modular pipeline
  whose every component exceeds VRAM alone; the per-component `components`
  block is the only correct shape.
- **ComponentsManager auto-offload**: whole-component swapping would thrash
  60 GB-class partitions; dw supports it and the H3 workflows correctly don't
  use it.

## 4. What the microbenchmarks say (measured today, idle 3090)

**PCIe H2D bandwidth** (1 GiB bf16, 4 reps):

| Path | Effective bandwidth |
| --- | --- |
| pageable → device | 8.5 GB/s |
| pinned → device | 23.7 GB/s |
| `pin_memory()` per transfer, then copy — the `low_cpu_mem_usage: true` + `use_stream` path | **4.6 GB/s** |

**SDPA at H3's shape** (1×56 heads×~15.4k rows×128, bf16 — the packed sequence
at 960×544×124f): flash backend **94 ms** per layer → ~4.7 s/step across 50
layers. Attention is therefore roughly a third of step compute, with int8
linears making up most of the rest (~6–7 s). Total per-step compute ≈ 10–13 s,
against ~18 GB of weight traffic per step.

## 5. Recommendations, ranked

### 5.1 Drop `low_cpu_mem_usage: true` on the transformer group offload — A/B it

Every H3 example sets it. Reading the installed
`diffusers/hooks/group_offloading.py`: with `use_stream=True` and
`low_cpu_mem_usage=True`, the CPU-side dict is left *unpinned* and **every
onload re-pins every tensor of the group** (`_pinned_memory_tensors()` →
`tensor.pin_memory()` per tensor per step). That is an extra host-side memcpy
of the entire ~18 GB transformer *every denoising step*, and it lands the
measured 4.6 GB/s effective path — ~4 s/step of transfer work, on the same
Python thread that launches kernels. With `low_cpu_mem_usage: false` the
weights pin once at setup and stream at 23.7 GB/s (~0.8 s/step), comfortably
hidden under 10+ s of compute.

Cost: ~18 GB of host RAM locked for the session (per stored preference, pinned
memory is acceptable on this box; 64 GB total, so it fits alongside the ~13 GB
text encoder with room left, but it is a real bite on a shared machine — hence
A/B rather than blanket-change). Expected win: anywhere from a few percent to
~25%/step depending on how much of the pin-copy currently escapes overlap;
the FBC-skipped steps, whose compute shrinks but whose streaming doesn't,
benefit most.

### 5.2 Truncate the Qwen3-VL stack after layer 50 (and drop `lm_head`)

The conditioning is `hidden_states[50]` of a 64-layer model
(`text_encoder_layer` config; naive truncation to exactly 50 is guarded
against in diffusers because the final norm would apply, but **truncating to
51 layers leaves `hidden_states[50]` bit-identical**). Today, 13 decoder
layers (~20% of the LM) run and stream for nothing on every encode — once per
segment in chained runs — and `lm_head` (1.5 GB bf16, excluded from
quantization) is never executed at all. A dw-side post-load hook (e.g. a
`truncate_layers` entry in the component configuration that slices
`model.language_model.layers` and replaces `lm_head` with an empty stub) would
cut encode time ~20% and host RAM by ~3 GB quantized / ~13 GB if ever run
unquantized. Quality-neutral by construction.

### 5.3 Add `use_stream: true` to the `text_encoder.model` leaf offload

The transformer's offload streams; the text encoder's doesn't (bare
`{"offload_type": "leaf_level"}`), so every leaf transfers synchronously —
and *without* a stream the CPU dict isn't built, so transfers also run from
pageable memory at 8.5 GB/s. `use_stream: true` (leaf-level prefetches
automatically) with default `low_cpu_mem_usage` pins once and overlaps.
Encode happens once per segment, so this is seconds per segment, not
per step — but it is free, and chained/EnhancePrompt runs encode repeatedly.
The LTX-2.5 recipe in this repo already does exactly this; H3 predates it.

### 5.4 Try an attention backend (`pip install kernels`, then `sage_hub`)

At ~4.7 s/step, attention is the largest single compute item FBC doesn't
touch. SDPA is already on its flash path (94 ms measured), so `flash_hub`
will do little — but SageAttention's int8 QK on an sm86 card typically takes
a further ~1.5–2× off attention, i.e. ~2 s/step here. The `kernels` package
is not currently installed, so `attention_backend` at the H3 shapes is an
untested lever on this box; dw already plumbs it per-pipeline and per-
component. Quality check it on a reference workflow before adopting — int8
attention on a guidance-distilled 9-step schedule has less error budget.

### 5.5 Reconsider int4 as the *quality* floor, not the VRAM floor

With streamed offload, transformer precision costs **host RAM and PCIe time,
not VRAM** — the resident set (2 groups) grows trivially. int4→int6 raises
per-step traffic from ~18 GB to ~26 GB (≈1.1 s pinned, still hidden) and host
RAM to ~26 GB, which no longer coexists with 5.1's fully-pinned option on
64 GB — the two recommendations compete for the same RAM budget, and the
faulty-RAM caveat on this box argues against maximizing resident host state.
But as a quality experiment on single (non-chained) runs, a 5- or 6-bit
transformer at unchanged VRAM is a lever the current setup leaves untouched;
the 20-step ref2va path would show the difference most.

### 5.6 Small consistencies

- **Docs drift**: `RECIPES_24GB.md` says `num_blocks_per_group: 1-2`; the
  fl2va examples ship 5. The examples are the truth — update the table, and
  consider noting *why* each mode gets its group size (headroom vs.
  conditioning load), which is currently only inferable.
- **t2va cache omission** (§3) — enable or annotate.
- `"vae": {"enable_tiling": true}` is a near-no-op for H3 — this VAE
  constructs with tiling already on (disabling it would change outputs, per
  the model's own docstring). Harmless, but the JSON implies it's doing work
  it isn't; worth a comment or removal on the H3 examples specifically.
- The EnhancePrompt workflows run Qwen3-4B on **CPU** for the rewrite. If
  that's to keep VRAM virgin before the big load it's defensible (and the
  REPL worker complicates GPU sharing), but a `device: "cuda"` +
  release-before-load ordering would cut a minute-class prompt rewrite to
  seconds; `h3_context_ir.json` already exposes `device` as a variable, so
  callers can opt in today.

## 6. Follow-up: what happened when the recommendations were tested (2026-08-19)

Recommendations 5.1-5.4 and the 5.6 consistency fixes were implemented and
A/B-tested the same day on the live box (three full t2va runs at 960x544x124f,
seed 42: control at HEAD, treatment with everything, validation with the final
config). 5.5 was left alone as directed. The scoreboard:

| # | Lever | Outcome |
| --- | --- | --- |
| 5.1 | `low_cpu_mem_usage: false` (pin once) | **Backed out.** Step time unchanged (15.44 vs 15.47 s/it) - and the run was **OOM-killed** (signal 9) during decode |
| 5.2 | Truncate Qwen3-VL to 51 layers + drop `lm_head` | **Kept.** Implemented as `truncate_layers` / `remove_modules` in dw; 12 unit tests incl. bit-identity of `hidden_states[50]`; validation run clean |
| 5.3 | `use_stream` on the text-encoder leaf offload | **Backed out** with 5.1 - it pins ~10GB of the same host RAM that caused the OOM |
| 5.4 | `sage_hub` attention | **Blocked upstream.** Plan written and Stage 1 executed; see below |
| 5.6 | FBC on t2va, fl2va groups 5->2, drop no-op `vae.enable_tiling` | **Done** across all 13 examples |

**Why 5.1 failed, precisely.** The premise held - the pin-per-onload path really
does run at 4.6 vs 23.7 GB/s - but the conclusion didn't: at int4 on a 3090 the
denoise step is compute-bound at ~15.4 s, and even the slow-path transfer
(~4 s/step) hides completely under it. Pinning bought zero step time. What it
did buy was ~27GB of unswappable host memory on a box whose H3 runs already
peak at 62.7GB RSS of 64GB: the treatment run finished denoising and was then
SIGKILLed by the OOM killer during VAE decode, where host allocations spike.
The revert is unconditional for this machine; on a faster GPU (where compute
per step shrinks toward the transfer time) or a larger-RAM host, both 5.1 and
5.3 are worth retrying - the recipe note in RECIPES_24GB.md says so.

**Validation of what shipped.** The final configuration (truncation + lm_head
removal + consistency fixes) runs clean: 15.49 s/it (parity with control's
15.47), wall 4:48 vs 5:02, peak VRAM 18.3 vs 18.2 GiB, RSS 62.65 vs 62.72 GB,
artifact saved, and frame-60 of the seed-42 output is scene-identical to the
control's (same composition, lighting, subject; minor pose drift consistent
with run-to-run quantized-matmul nondeterminism).

**A finding the A/B surfaced for free:** first_block cache at threshold 0.1
**never skips a step on the 9-step turbo schedule** - control (no cache) and
treatment (cache) step times are identical to within noise. Consecutive steps
of a distillation schedule differ too much for a 0.1 residual threshold. On the
strength of that measurement the cache was then removed from all eight 9-step
turbo workflows (including the two t2va examples the consistency pass had just
added it to - a fix this finding retroactively unfixed) and kept only on the
five 20-step ref2va workflows, where closer-spaced sigmas give it steps it can
actually skip. That expectation is unmeasured so far - if a normal 20-step
ref2va run also shows flat ~15 s steps, it should come out there too. Raising
the threshold on turbo paths was deliberately not pursued - quality risk for a
lever the distilled schedule barely needs.

**5.4 blocker detail.** `kernels` 0.16.0 was installed and the staged plan in
SAGE_HUB_PLAN.md executed to its first gate, which failed upstream twice over:
the `kernels-community/sage-attention` revision diffusers pins ships no build
directory at all, and `kernels-community/flash-attn2` (probed to isolate the
failure) ships only aarch64 builds for torch 2.13/cu130 - nothing for x86_64.
No workflow was touched. The plan documents the retry (one minute, whenever
the hub publishes matching builds) and explicitly rejects the from-source
`sageattention` build as the kind of rabbit hole it is.

**Not investigated, and why:** the OOM kill is confidently attributed to
pinned-memory pressure (unswappable pages + 62.8GB RSS on a 64GB host), but
this box also has a known faulty-RAM history; per standing guidance, no
memory-corruption forensics were attempted. If OOM-adjacent flakiness appears
in normal runs, that investigation is the deferred item to pick up.

## 7. Bottom line

The hard problems — fitting two 66 GB partitions and a 32B VL encoder through
a 24 GB card while keeping audio+video sync, reference identity, and chained
continuity — are solved with the right primitives, and mostly with mechanisms
(on-demand residency, modular cache context, dotted-path offload, workflow
pruning) that had to be built rather than found. Nothing in the current
configuration is *wrong* in kind; the findings above are one flag that fights
the memory subsystem it's meant to serve (5.1), one structural waste inherited
from upstream (5.2), and levers whose absence was reasonable but is now cheap
to test (5.3–5.5). Section 6 records how testing resolved them: the step loop
turned out to be compute-bound at int4 on this card, so the transfer-side
levers (5.1, 5.3) bought nothing and were backed out after one of them
OOM-killed a run on this 64GB host; the structural fix (5.2) and the
consistency cleanups shipped and validate clean; and the attention backend
(5.4) is blocked upstream with a one-minute retry documented. The honest
per-step ceiling on this machine now runs through attention (5.4, when the
hub unblocks) and step count, not through memory plumbing — that part of the
configuration is, demonstrably, already right.
