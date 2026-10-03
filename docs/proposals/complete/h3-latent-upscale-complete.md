# H3 latent upscaler: promote a 544p take to 1344x768 in latent space (#471)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-03). Plan v1 was approved by Don on 2026-09-27 without answers to
Q1-Q4, so each question's stated default stood. Stage 1, #499, shipped in
its final form 2026-10-03 and was verified the same day. Stage 2, #500,
was not built: Don's gate after stage 1 came back "close but soft".

## The idea

[LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
is a 345M-parameter 3D-conv network that resizes H3's 24-channel video
latents in H×W (1x-4x, T kept). The idea was an opt-in path: generate an H3
clip at 960x544, upscale its latents to 1344x768 (the grid goes 34×60 →
48×84), and decode. That skips a native 768p render. The existing
`templates/minimax/*` templates and the `minimax-h3` skill's defaults were
to stay exactly as they were.

## Verdict

**Build smaller.** The demand was Don's own: #484's native 768p Ref2VA
shots turned smeared crowd faces crisp, but took ~31 min each. Nobody had
shown that the upscaler's output was any good without a refine pass, so the
plan built the measurement first (stage 1) and gated the catalog template
(stage 2) on Don comparing it against a native 768p render. The refine pass
(a new denoise-schedule block, ~$8-12) and persisted latents were deferred.

## What the plan found against the issue

1. **dw cannot drop H3's decode block.** The base step decodes at 544p and
   also hands its `latents` on through `output` (`previous_result:base.latents`).
2. **Promotion without regenerating isn't free.** Latents live only in the
   in-process step cache, so promoting an earlier take means re-running its
   544p base with the same seed inside the promoting workflow.
3. **There was no H3 decode step to reuse.** A decode task has to run
   diffusers' `MiniMaxH3VideoDecodeStep` (denormalize, fp16 autocast decode,
   pixel revert), or the colours come out wrong.
4. **The target is given in pixels**, multiples of 16, and divided by 16 in
   the task. The non-uniform 34×60 → 48×84 is allowed.
5. **Normalization: the plan was wrong here, see below.** The plan (and
   the issue) said that pipeline latents are already normalized, so the
   ComfyUI node's `(x-mean)/std → net → x*std+mean` wrapper would normalize
   twice and should be dropped.
6. **`plan.downloads_required` doesn't count a task's weights** (a gap that
   predates this feature; `upscale` and RIFE miss it too). Q3's default kept
   that fix out of this feature.

## What was built

### Stage 1 (#499): `upscale_h3_latents` and `decode_h3_latents`

Final form: `develop` @ `f01ae1fa` (`924a973d`, a revert of the
withdrawal plus the fix), then `8a5136da` (`f24b4406`, the docstring fix
from the architecture review). `c1e8455b` (#584) later gave the guide's
workflow a `num_frames` constraint.

- **`upscale_h3_latents(latents, width, height, model_name?, weight_name?)`**
  in `dw/tasks/h3_latent_upscale.py`. The network is vendored under MIT in
  `dw/tasks/h3_latent_upscaler_model.py`, which asserts the v1 architecture
  instead of inferring it freely.
  - The default weights are the bf16 safetensors (Q4), read from the
    `minimax_h3_latent_upscaler_3d_conv_v1/` subfolder at the pinned
    revision `3f941d5d182014dd5c0a5e16330420ee2d4aa0c6`.
  - It applies the node's normalization wrapper in float32. The node's
    mean/std are checked to equal the H3 VAE's `latents_mean`/`latents_std`.
  - Refusals, each naming the argument: not a 5-D, 24-channel tensor; a
    target that isn't a multiple of 16; a per-axis scale outside 1x-4x; a
    target over the 1344x768 (or 768x1344) canvas. `task_domains` refuses
    `width`/`height` ≤ 0 at validate.
  - `model_name` must be a Hub repo id (paths, URLs, `a/b/c` and `..` are
    refused at validate). `weight_name` must be a bare `.safetensors` file
    name. `/`, `\`, `..`, absolute paths and `.pth`/`.bin`/`.pt`/`.ckpt` are
    refused at validate, and checked again at run time before any download.
    So the trust gate was not widened.
- **`decode_h3_latents(latents, model_name?)`** runs diffusers' own
  `MiniMaxH3VideoDecodeStep` on the H3 VAE.
- **Audio policy:** the base pass's track, re-paired with `pair_audio`.
  Nothing re-denoises.
- **Docs:** `docs/WORKFLOW_GUIDE.md` *Promoting an H3 take to 768p in latent
  space* holds the inline `H3LatentUpscalePreview` workflow (base → up →
  decode → mux). `get_guide` serves it, and `list_tasks`/`get_task`
  describe both tasks.
- **General fixes that landed under this stage:**
  - `3ec965e6`: a `previous_result:` property that no result carries is an
    error, not a step that "succeeds" after zero iterations;
  - `465a61ed`: `pair_audio` unwraps a pipeline's batch of one video;
  - `a6f816e8`: a location refusal no longer echoes a server directory.

### Normalization: what went wrong, and the fix

The first build dropped the node's wrapper, as the plan said. Its output was
garbage: checkerboard, ghosting and magenta blobs, while
`decode_h3_latents` on the *un-upscaled* latents decoded clean. The stage
was withdrawn for release 0.5.0 (#526, `777f2eb2`). Don then asked for a
tensor diff against the ComfyUI reference node before any rebuild:

- The vendored network against the node's own class, with the real
  weights: max |diff| = 0.0. The network was never the bug.
- ComfyUI's H3 VAE `encode` already returns normalized latents, the same
  space a diffusers H3 pipeline returns. The node then normalizes a
  **second** time. So the network was trained on doubly normalized latents,
  and the wrapper has to be applied.
- On a unit-variance input with no wrapper, the output std was 5.28 and the
  mean −0.83. With the wrapper, they were 0.92 and 0.00.

The lesson for a revival or a v2 re-pin: check a vendored model's input
space against the reference implementation's numbers, not against reasoning
about which side normalizes.

### Stage 2 (#500): not built

`templates/minimax/upscale-preview` and its `minimax-h3` skill line were
gated on Don's comparison and closed as not planned on 2026-10-03.

## Measured

**Verification** (lem, RTX 3090, `develop` @ `8a5136da`, 124 frames, job
`baaeda980afa`):
- 1344×768, 124 frames at 24 fps, base audio (32 kHz stereo, 5.17 s). No
  colour cast, no wash-out. The identity decode of `base.latents` matches
  the 544p reference.
- Latency 629.99 s in all: the upscale took 4.3 s, the decode 131.3 s, and
  decode VRAM was 10.93 GB reserved. That is a point reading at step
  boundaries, because the events carry no true peak.
- With `base` served from the step cache, a 1x run took 76 s and a 960×768
  run took 96 s.

**Don's gate** (same T2VA crowd-faces prompt, seed 42, 124 frames):
- upscale preview (job `f90ea84a796e`): 7.8 min;
- native `video-with-audio-768p` (job `074e65929a9d`): 12.7 min.

The native render held distinct faces. The preview's faces were soft and
waxy, with eyes smeared in several. Candle highlights and cobblestones
sharpened well: it is a clean upscale of a soft 544p base, and as the plan
predicted, it can't show detail the base never generated. Saving ~5 min
(~39%) didn't offset that. Caveats: one seed and one prompt, and the two
paths use different LoRAs and shifts. The #484 Ref2VA crowd case itself was
not tested.

## Bounces per stage

- **Stage 1 (#499): 4 tester bounces, then a park, a withdrawal and a
  rebuild, then 1 architecture-review bounce.**
  1. M-F039: the guide section was a `####` heading, which `get_guide`
     can't reach.
  2. M-F041: the mux read `base.sample_rate`, a key the modular result
     doesn't carry, and silently wrote nothing (hence `3ec965e6`).
  3. M-F041: `pair_audio` crashed on the unwrapped video batch (hence
     `465a61ed`).
  4. M-F041: the upscaled output was garbage (normalization, above). The
     loop driver parked the stage after this one, at four bounces.
  - Architecture review of the rebuild: the vendored module's docstring
    still gave the withdrawn reasoning about normalization. Fixed in
    `f24b4406`.
  - The rebuild then verified on its first pass.
- **Stage 2 (#500):** not built.

No `usage:` figures were recorded on the stages, so cost is left out.

## Deferred

- **The refine pass**: a dw-owned (or upstream) replacement for H3's
  set-timesteps block that takes `sigmas`, renoises the upscaled latent to
  σ_start and runs the tail of the schedule with the 768p LoRA at shift 6,
  with audio held or re-paired. The gate's "close but soft" is the result
  plan v1 named as bringing it back. Building it needs a new plan version,
  which is Don's call. #585's H3 LoRA eval is relevant: the fal Realism
  People LoRA on the 768p turbo path gave the best faces at ~zero added
  cost, which raises the bar a refine pass has to clear.
- **Persisted latents** (a `.safetensors` result type reachable by
  `output:`): only worth it if a promotion template lands.
- **`downloads_required` counting task weights** (Q3): a separate issue
  for every task, not done here.
- **Q1**: the tasks stay, since the gate wasn't a "no". They are surface
  for a future refine, and the vendored v1 module stays pinned until a
  deliberate re-pin to upstream's v2 or 2D variant.
