# Plan: sage_hub attention for MiniMax-H3

Attention is ~4.7 s of every H3 denoise step on this 3090 (measured: 94 ms/layer
x 50 layers at the 960x544x124f packed-sequence shape) and is the largest
compute item the first_block cache does not touch. SageAttention quantizes QK to
int8 and typically runs 1.5-2x faster than SDPA's flash path on sm86. This plan
gets it into an H3 workflow with a fast, reversible test at each stage - written
with prior attention-backend trouble in mind: every stage has a hard pass/fail
gate and stopping at any gate leaves the workflows untouched.

## What is already verified (static, this repo, 2026-08-19)

- The installed diffusers 0.40.0.dev0 registers `sage_hub` in
  `attention_dispatch.py`: kernel repo `kernels-community/sage-attention`,
  entry point `sageattn`, called with `tensor_layout="NHD"`, `is_causal=False`.
- It requires the `kernels` package >= 0.12 - **installed: kernels 0.16.0**.
- `MiniMaxH3AttnProcessor` routes through `dispatch_attention_fn(...,
  backend=self._attention_backend)`, so both of dw's existing wirings work:
  the per-call `"attention_backend"` key in the pipeline configuration (context
  manager) and the per-component pin in a `components` entry
  (`set_attention_backend`). H3 is uncompiled, so the per-call form is the
  right one - it scopes the backend to the step's own pipeline call.
- H3 packs each request into one attention document and **never passes a
  mask** - the model's own docstring calls out that every backend stays
  available. diffusers' sage wrapper rejects `attn_mask`; H3 never hits that.
- Head dim 128, non-causal, batch 1 - inside SageAttention's supported range.
  On sm86 `sageattn` auto-selects its `qk_int8_pv_fp16` CUDA path.

## Stage 1 - kernel loads and runs at H3's shape (no workflow, ~1 min)

Fetch the kernel and run it against SDPA on random tensors at the exact H3
shape (1, 56 heads, ~15.4k rows, 128):

```python
from diffusers.models.attention_dispatch import attention_backend  # noqa - or use dispatch directly
# microbench: sdpa-flash vs sage_hub, same tensors, 5 reps each
```

Gate: kernel downloads, runs without error, and is meaningfully faster.
Failure here (no wheel for this torch/CUDA ABI, dtype rejection) ends the plan
with zero changes - this is where "trouble with attention backends" usually
shows up, and it costs one minute to find out.

**Result (2026-08-19): FAILED at the gate - plan stopped here, nothing changed.**
Two independent failures, both on the hub side rather than this machine's config
(torch 2.13.0+cu130, x86_64, kernels 0.16.0):

- `sage_hub`: `kernels-community/sage-attention` at the revision diffusers pins
  (`version=1` -> `ed09909...`) has **no `build/` directory at all** - the fetch
  404s before ABI selection even starts.
- `flash_hub` (probe to isolate the failure): `kernels-community/flash-attn2`
  resolves, but its build variants for recent torch (`torch210`/`torch212`,
  cu128/cu130) are **aarch64-only** - no x86_64 build matches this system.

So the hub-kernel path is currently unusable on this box for any backend, and
that is not fixable from this repo. What would unblock it, in order of
preference:

1. **Wait and retry** - kernels-community publishes x86_64 builds for
   torch 2.13/cu130 (or diffusers bumps its pinned kernel revisions to ones
   that carry them). Retrying costs one minute: rerun Stage 1.
2. **Native `sage` backend instead of `sage_hub`** - `pip install
   sageattention>=2.1.1`, which is a from-source CUDA build against cu130.
   Deliberately not pursued now: source-building attention kernels is exactly
   the complexity rabbit hole this plan is fenced against, and past attention-
   backend trouble on this machine argues for the prebuilt path or nothing.
3. **Torch downgrade** to a version the hub carries x86_64 builds for -
   rejected outright; the rest of the stack is tuned to this torch.

Stages 2-4 remain valid as written and pick up unchanged whenever Stage 1
passes.

## Stage 2 - one t2va run, timing + sanity (one workflow run)

Add one line to a scratch copy of `MiniMaxH3.json`:

```json
"configuration": { "attention_backend": "sage_hub", ... }
```

Same seed as a reference run. Gate: the run completes, per-step time drops
consistently with Stage 1's projection, and the video is free of the artifacts
int8 attention produces when it goes wrong (blocking, washed-out motion,
desynced or noisy audio - audio rows ride the same attention).

## Stage 3 - quality gate on the sensitive path (one workflow run)

t2va hides identity errors; ref2va does not. Run `MiniMaxH3Ref2VA.json` (20
steps, no LoRA - the quality-first configuration) with and without the
backend, same seed, and compare subject fidelity and lip sync by eye. Gate:
no visible identity or sync degradation. Only after this does the backend go
into the checked-in examples - and then as the pipeline-level key on all of
them, since the transformer partitions share the same attention.

## Stage 4 - adopt or record

- Adopt: add `"attention_backend": "sage_hub"` to the H3 examples,
  note the measured step-time change in `RECIPES_24GB.md`, and note that
  first use per machine downloads the kernel (network required; it caches
  under the Hub cache afterwards).
- Reject: delete the scratch copy; nothing else was touched. Record what
  failed and at which stage in `MINIMAX_H3_ASSESSMENT.md`.

## Known risks, called out in advance

- **ABI mismatch**: `kernels` resolves a prebuilt binary for the running
  torch/CUDA; a missing build errors at first attention call. Fails fast,
  caught in Stage 1.
- **Few-step error budget**: the turbo-LoRA paths run 9 steps - approximation
  error has fewer steps to wash out than at 20. Stage 2 uses the 9-step
  config deliberately so this shows early.
- **Audio**: H3's audio rows attend in the same packed sequence; audio
  artifacts are the most likely subtle failure and the easiest to miss when
  only watching the frames. Listen, not just look, at Stages 2 and 3.
- **first_block cache interaction**: FBC decides skips from first-block
  residuals; a noisier attention slightly perturbs those decisions. If Stage
  2 shows odd step-to-step pacing, rerun once with `cache` removed before
  concluding anything about the backend itself.
