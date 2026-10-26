---
name: flux3-klein
description: Use when a dw MCP server is connected and the user wants an image from FLUX 3 Image Klein - a picture from a short request or a full caption, an edit of their photo, several reference pictures combined into one, or a layout with elements placed by box. Picks the template, states the distilled schedule, the size grid and the reference rules the pipeline does not check for you, quotes cost, and points at Black Forest Labs' upsampler and prompting guide.
---

# FLUX 3 Image Klein on a dw server

FLUX 3 Image Klein (Black Forest Labs) is a 5.6B distilled flow map that
draws text-to-image, edits and multi-reference images in 16 steps. It pairs
the FLUX.2 VAE with a Qwen3-VL-4B text encoder, about 19 GB of bf16 weights.
Klein was trained on long captions written by BFL's prompt upsampler, and BFL
says not to run it without one: a short prompt gives a generic picture.

## Before anything

1. `get_server_info`: the device. The templates place every component on
   `cuda`. Apple Silicon is unmeasured (the same 19 GB of weights in unified
   memory, plus activations); on `mps` say so before spending, and
   on `cpu` stop.
2. `get_diffusers_state`, then `list_classes` for
   `Flux3DistilledModularPipeline`. The modular pipeline is in diffusers
   `main` only (merged 2026-10-15). If it is missing, the server needs
   `update_diffusers` and a restart. That is the user's call; ask first.
3. The weights are gated under the FLUX Kommunity License (non-commercial,
   plus commercial use of outputs below US$5M annual revenue). The user
   accepts it on the model page and the server runs `hf auth login`; a 401 or
   `GatedRepoError` in `get_job` means one of those is missing. The
   upsampler is Apache-2.0 and ungated.
4. `list_workflows(shape="image")` for the text-to-image templates and
   `list_workflows(shape="image-edit")` for the edit ones. Take names,
   `summary` and `cost` from the listing over the names quoted below, then
   `get_workflow` on the one chosen, for its variables.

## Which shape is the request

- **A short request** ("a man walking his dog"):
  `templates/flux3/upsample-then-generate`. The upsampler writes the caption,
  is released, and Klein draws from it. The caption is saved as an
  `intermediate` text file: read it, and reuse or edit it as the prompt of
  text-to-image without paying for the upsampler again.
- **A full caption already written** (or one the upsampler wrote earlier):
  `templates/flux3/text-to-image`.
- **One photo to change**: `templates/flux3/image-edit`, an `instruction`
  and an `image`. The upsampler sees the same picture as the pipeline. With
  no width or height the output is 1 MP at the reference's aspect ratio.
- **Several pictures combined** ("the dog from this one on that mug"):
  `templates/flux3/multi-reference`, an `instruction` and `images` in order.
  The first reference sets the aspect ratio.
- **Elements at given positions**: `templates/flux3/layout-boxes`. The
  caption carries the boxes; the format is in `references/layout-boxes.md`.
  Boxes from a short request (BFL's box upsampler) are not in the catalog
  yet, so the caption is written by hand.
- **The base (non-distilled) checkpoint, guidance, LoRA, fp8 or NVFP4**: not
  in the catalog. Say so rather than improvising.

Input media must be on the server: `upload_asset`, then `asset:` in the
argument, or `output:` for an earlier run's file.

## Hard rules

- 16 steps, no guidance and no negative prompt. The distilled model is a
  flow map; BFL's reference code raises on any guidance other than None
  or 1.0. Leave
  `num_inference_steps` at 16 (trained for 4-16) and add no
  `guidance_scale` or `negative_prompt` to a workflow you write.
- `width` and `height` are multiples of 16 (the VAE's 8x times the 2x2
  latent packing). The pipeline only refuses an off-grid size after the
  weights load, so the templates declare the rule and `validate_workflow`
  catches it first.
- Sides run from 64 to 4096, as BFL states it. Nothing in the templates or
  the pipeline checks that ceiling, and above 1024x1024 a 24 GB card has not
  been measured: everything resident leaves under 2 GB spare at 1024x1024.
- Up to 10 references. That is BFL's training limit, and the pipeline does
  not enforce it: an 11th reference runs, outside what the model learned.
- In a caption written by hand, name references `<ref_image_0>`,
  `<ref_image_1>` and so on, zero-based in the order given. The upsampler's
  own captions say "the first image", "the second image".
- Every component carries an explicit `device`. The modular pipeline runs
  on its first component's device, so one left unplaced keeps the prompt
  and the noise on the CPU (`index is on cpu`, `Cannot generate a cpu
  tensor from a generator of type cuda`). For the same reason the text
  encoder cannot be `residency: on_demand`.
- The upsampler (about 18 GB) and Klein (about 19 GB) do not fit a 24 GB
  card together. Keep `release_models` on the upsample step and load Klein
  after it.

## Prompts

The prompt path is the upsampler. The builtins `flux3_klein_upsample` (a
request to a caption) and `flux3_klein_upsample_edit` (an instruction plus
the references to a caption) run BFL's `flux-3-image-klein-pup` with BFL's
own system prompts, read from the pup repository at a pinned commit. The
templates call them; a workflow you compose names them as a `builtin:` step
before the Klein step.

To write a caption by hand, read BFL's prompting guide:
https://docs.bfl.ml/guides/prompting_summary. It is written for the API
model, which upsamples on the server side, so read it as a description of
the caption Klein expects, not as advice for a short prompt. The stored
`flux3/` prompts (`list_prompts`, `get_prompt`) are BFL's own upsampled
captions, `man_dog` and `balloons_layout`: read one before writing your own.
The upsampler writes English only.

## Run and judge

1. `validate_workflow` first - free, and catches an off-grid size.
2. Quote `plan.estimate` and name any `downloads_required` (Klein, plus the
   upsampler for every template but text-to-image and layout-boxes). Do not
   quote minutes from memory: the estimate is the number. More references
   cost more per step, because each one adds tokens. Get the go-ahead, then
   `run_workflow` with `acknowledged_cost` set to the plan's
   `{fingerprint, minutes, downloads}`.
3. `wait_for_job`, `timeout_seconds` = estimate plus margin. Call again
   while `still_running`. If the call dies client-side first, call again
   with `timeout_seconds` 60 and loop.
4. A saving step carries a `subfolder`: the image shown to the user is
   `final`, the upsampler's caption `intermediate`.
   `list_gallery(subfolder="final")` lists only deliverables, so keep both
   in anything you compose.
5. Judge it yourself: `get_output_image` on the `final` image against the
   ask, and `get_output_text` on the caption when the picture misses - the
   caption is what Klein drew. Then `get_job` for the manifest and its
   warnings, and give the user the gallery `url`.
6. Save a keeper (`get_job_workflow`, `save_workflow`) to rerun it by name.

What goes wrong, and where to look:
- A generic, flat picture: a short prompt went to text-to-image without the
  upsampler. Rerun through upsample-then-generate.
- An edit that changed more than was asked: Klein can alter content outside
  the requested region (BFL's fix for that is API-only). Narrow the
  instruction, or try another seed; compare with `compare_outputs`.
- A small detail of a reference rendered wrong: the upsampler can
  misdescribe fine detail in a reference, and dw sizes references for it
  differently from BFL's own pipeline. Read the caption with
  `get_output_text`. The edit templates always re-upsample: their
  prompt is the upsampler's output and they have no caption variable. Either narrow the
  instruction and rerun, or reuse the corrected caption: `get_workflow` on
  the edit template, and compose a one-step workflow from its Klein step
  with the caption as `prompt` and the same image list. Text-to-image is no
  answer here, since it drops the references.
- A boxed element missing or spilling over: boxes are hints, not masks.
  Enlarge the box or describe the element more, in the prose and the table.

## Sources

The flux-3-image-klein and flux-3-image-klein-pup model cards (Hugging Face),
BFL's flux-3-image-klein GitHub repository (commit bbfe317d), docs.bfl.ml,
and `diffusers.modular_pipelines.flux3`. Read 2026-10-15; audit
`docs/proposals/audits/2026-10-15-flux-3-klein-audit.md`.
