# H3 DMAD step-distillation LoRA for dialogue (#1579, stage #1584)

Does `ZhengmingYu/DMAD`'s MiniMax-H3 student render cleaner fl2va dialogue (lip sync, voice,
motion) than lightx2v turbo or base H3 at few steps?

**Verdict: don't adopt.** The A0 load check failed, which is the plan's stop rule ("If the
load isn't clean, stop and record the verdict 'don't adopt'").
- dw loaded the file and `diffusers` mapped **zero** of its 624 tensors onto the pipeline.
  Then `set_adapters` failed with `Adapter name(s) {'dmad'} not in the list of present
  adapters: set()`.
- The cause is the file's key format, not the workflow (see *Why the load is empty*). The
  plan's precedent applies: FastH3 was rejected for its key format too.
- A1 (base, dmad4, dmad8, turbo), A2 (seed spread) and A3 (non-dialogue pair) **did not
  run**, so they are absent from `list_jobs(workspace="probe-1579")`. Every arm's call is
  **drop**: there is no output to judge.

Run 2026-11-23 (server time) by claude-opus-5-5 (anthropic):
- the main `dw.serve` on lem, deployed from develop @ `bf5d341b` (this stage changes no
  engine code);
- workspace `probe-1579`.

## Setup

- **The workflow:** `bench/workflows/probe-1579-image-to-video.json`, saved in the
  workspace as `probe-1579-image-to-video`. It is `templates/minimax/image-to-video.json`
  with three changes:
  1. `scheduler` and `audio_scheduler` are both rebuilt from the checkpoint's config with
     the full form `{"configuration": {"scheduler_type": "MiniMaxH3Scheduler"},
     "from_config_args": {"stochastic_sampling": "variable:stochastic_sampling"},
     "shift": "variable:video_shift" | "variable:audio_shift"}`.
  2. The LoRA entry gains `"revision": "variable:lora_revision"`.
  3. Defaults are `weights_dtype` `{int8}` and seed 1001.

  Everything else is the template: 960x544, 124 frames, sparse attention, group offload.
- **The DMAD arguments (A0 = A1 dmad4, prompt 1, seed 1001):**
  - `lora_model_name` `ZhengmingYu/DMAD`;
  - `lora_weight_name` `minimax_h3/dmad_minimax_h3_4step_full_critic.safetensors`, a
    string, `.safetensors` only (Q2);
  - `lora_revision` `b08c2bf3be8e67039c363781e6eaacc9713340cd`, pinned (Q2);
  - `lora_adapter_name` `dmad`, not "turbo";
  - 5 steps, shifts 12/2 (the file's own `sampling` metadata), `stochastic_sampling` true.
- **Prompt 1** uses `asset:cast/tomas.jpg` with a 20-word English line in `<d>` (see the
  job's `arguments`). Prompt 2 (`asset:cast/estela.jpg`, 18 words) was written but not
  run.

## A0: the load check

Job `860478669a1e`: run `20261123-004932-e6aa7241`, `cuda:1 RTX 3090`. It **failed after
78 s** in phase `loading`, `LoRA: ZhengmingYu/DMAD`.
- **The scheduler flag took.** `get_job_events` shows both `Loading scheduler:
  MiniMaxH3Scheduler (from_config_args: stochastic_sampling)` and `Loading
  audio_scheduler: MiniMaxH3Scheduler (from_config_args: stochastic_sampling)` before the
  LoRA phase. (Under #1438 this line was only in journalctl. It now reaches the events.)
- **Validate warned** `adapter_unrecognized`: the weight name names neither `ref2v` nor
  `fl2v`, so dw can't check its partition.
- **The error:** `ValueError: Adapter name(s) {'dmad'} not in the list of present
  adapters: set()`, raised by `pipeline.set_adapters` (`dw/pipeline_processors/adapters.py`
  `apply_adapter_settings`) after `load_lora_weights` returned without loading anything.
- **Host peak committed:** 31.7 GB. GPU time spent: about 1.3 min of the 45-60 min Q1
  allowed.

## Why the load is empty

I read the file's safetensors header (78,280 bytes, 624 BF16 tensors) at the pinned
revision with an HTTP range request. Nothing else was downloaded for this.
- **Metadata** (`__metadata__`):
  - model: "DMAD 4-step student of MiniMax-H3 (text-to-audio-video), LoRA on the T2VA
    transformer";
  - sampling: "4 steps, video shift 12, audio shift 2, re-noise step rule, 1344x768, 124
    frames at 24 fps, no CFG";
  - lora_scale 1.0, lora_alpha 128, lora_rank 128;
  - fuse: `W_base += lora_scale * (lora_B @ lora_A)`;
  - key_format: `<module>.lora.down.weight = lora_A [rank, in]; <module>.lora.up.weight =
    lora_B [out, rank]; <module> is the Diffusers MiniMaxH3Transformer3DModel module name`;
  - checkpoint fullcritic, iteration 1600; license MiniMax H3 Community License.
- **No component prefix.** Keys are bare module names: 600 under `transformer_blocks.*`
  and 24 under `token_refiner.*`. None start with `transformer.`, so the pipeline's
  loader can't tell which component they belong to.
- **Two naming styles in one file.**
  - Attention (`to_q`, `to_k`, `to_v`, `to_out.0`) uses `.lora.down.weight` /
    `.lora.up.weight` (208 + 208 tensors).
  - The feed-forward (`ff.net.0.proj`, `ff.net.2`) uses `.lora_A.weight` /
    `.lora_B.weight` (104 + 104 tensors).
- **Partition.** By its metadata it targets the T2VA `transformer`, the one an fl2va step
  denoises against, not `transformer_ref`. So the partition wasn't the problem. The key
  format was.

dw can't fix this from a workflow. `LORA_ENTRY_KEYS` (`dw/trust.py`) has no prefix or
key-map option. Loading the file would need one of these:
- a key conversion in engine code;
- an upstream re-save in the standard diffusers/PEFT format (a `transformer.` prefix and
  one `lora_A`/`lora_B` style);
- a locally re-saved file.

Each of those is outside this bench-only stage and outside Q2's trust ruling, which
covered the published file at a pinned revision.

## Downloads

- `validate_workflow` planned exactly one download: `ZhengmingYu/DMAD`,
  `minimax_h3/dmad_minimax_h3_4step_full_critic.safetensors`, 1.29 GB.
  - The plan lists that entry with `revision: null`, because the download planner doesn't
    carry a LoRA's revision. The loader got the pinned revision from the job's arguments.
- **The 8-step file was never fetched.** A0 stopped the stage before A1's dmad8 arm.
  `list_downloads` is empty afterwards.

## Calls per arm

| Arm | Ran | Call | Why |
|---|---|---|---|
| A0 load check (dmad4, prompt 1, seed 1001) | yes, failed | **don't adopt** | 0 of 624 tensors mapped (key format) |
| A1 base | no | drop | stopped at A0 |
| A1 dmad4 | no (A0 was its first job) | drop | stopped at A0 |
| A1 dmad8 | no | drop | stopped at A0; file not downloaded |
| A1 turbo control | no | drop | no DMAD arm to compare against |
| A2 seed spread | no | drop | stopped at A0 |
| A3 non-dialogue pair | no | drop | stopped at A0 |

## Deviations from the plan

- **`euler_update_form` was left out of `from_config_args`.** The plan's form names only
  `stochastic_sampling`. #1438 showed the checkpoint's value is already the class default,
  `blend`.
- **The 8-step file wasn't downloaded**, and A1-A3 didn't run, both because of the A0
  stop.

## Follow-up

If DMAD is wanted later, it needs its own issue and Don's ruling (`trust`, and `scope` if
the fix is engine code). The options are a diffusers-side H3 key converter, a key-map in
dw's LoRA entry, or asking upstream for a standard-format re-save. Only with a clean load
would the A1-A3 arms here be worth running.
