---
name: kandinsky-6
description: Use when a dw MCP server is connected and the user wants Kandinsky 6.0 video - a five-second clip with its own soundtrack from text, or from a picture that becomes the first frame. Picks the template, states the distilled schedule and the size and frame rules the pipeline does not enforce for you, quotes cost, and points at Kandinsky Lab's own prompt format.
---

# Kandinsky 6.0 on a dw server

Kandinsky 6.0 (Kandinsky Lab, MIT) generates video and a 44.1 kHz soundtrack
together, lip-sync included. Every checkpoint is trained on five-second clips:
121 frames at 24 fps, natively 480x864 landscape. The catalog ships the 3B
Lite-distill checkpoint, which fits a 24 GB card.

## Before anything

1. `get_server_info`: the device. On `mps` stop and say so: the video VAE's
   decode keeps memory per tile on Apple Silicon and the worker is killed
   (measured 2026-10-06, torch 2.14.1). On `cpu`, stop.
2. `get_diffusers_state`, then `list_classes` for `Kandinsky6TI2VAPipeline`.
   The pipeline is in diffusers `main` only (merged 2026-10-06), and the
   version string `0.41.0.dev0` does not tell builds apart - check for the
   class. If it is missing, the server needs `update_diffusers` and then a
   server restart. That is the user's call; ask before doing it.
3. `list_workflows(shape="shot")`, and take names, `summary` and `cost`
   from the listing over the names quoted below. `get_workflow` on the one
   chosen, for its variables.

## Which shape is the request

- **A clip from text**: `templates/kandinsky6/text-to-video`.
- **From a picture**: `templates/kandinsky6/image-to-video`. The still is the
  first frame, and the prompt says what happens next in it. The pipeline
  resizes and center-crops the still to `width` x `height`, so for a portrait
  still set `width` 480 and `height` 864, or its sides are cut off.
- **Longer than five seconds, sharper, or a different checkpoint**: not in
  the catalog yet. The super-resolution pipeline needs compiled flex
  attention, and the 29B Pro transformer is 60 GB in bf16. For a longer
  piece, cut several five-second clips together (`list_tasks`, the concat
  and dissolve tasks) rather than raising `num_frames` - every checkpoint
  is trained at 121.

If none fits, compose from `list_tasks` before authoring a new workflow;
read the `workflows` guide's authoring section first.

## Hard rules

- The distilled checkpoint runs 10 steps at `guidance_scale` 1.0, and both
  are part of the checkpoint: its PiFlow scheduler refuses custom sigmas,
  and it cannot be swapped for another scheduler. Guidance above 1.0 adds a
  second pass for nothing. 50 steps at 5.0 belongs to the non-distilled Lite
  and Pro checkpoints, which no template ships.
- The pipeline's own defaults are 512x768 at guidance 5.0, and neither is
  this model's. A workflow you write must set `height`, `width`,
  `num_inference_steps` and `guidance_scale` explicitly.
- `width` and `height` are multiples of 16, or the pipeline refuses them.
- `num_frames` is `4k + 1`. The pipeline floors an off-grid count to a
  shorter clip rather than refusing it, so the templates refuse it. The
  transformer's rotary table caps a clip at 509 frames, but only 121 is
  trained.
- `frame_rate` sets the soundtrack's length, not the picture's. Keep it 24,
  and keep the result's `fps` on the same variable.
- The soundtrack is mono at 44.1 kHz, duplicated to stereo in the mp4.
  `sample_audio` false gives a silent clip and skips the audio decode.

## Prompts

The format is Kandinsky Lab's, from the system prompts of their own prompt
beautifier (read 2026-10-06):
https://github.com/kandinskylab/kandinsky-6/blob/main/comfyui/kandinsky6/beautifier_prompts/t2av_system.txt
and `i2av_system.txt` beside it for a picture. Read the one that fits
before writing a prompt. In short:
- One paragraph of 180-300 words. Give the shot, then exactly one camera
  sentence ("The camera is static." when none is asked for), then
  appearance, then the five seconds of action in order, with light and
  style last.
- Then exactly one audio sentence inside `<AUDCAP>…<ENDAUDCAP>`, with
  speech inside `<S>…<E>` after the speaker and a speech verb.
- For a picture, the image is ground truth: describe what is in it, then
  what happens next.

The `kandinsky6/` stored prompts (`list_prompts`) follow that format. Let the
user's language stand; the model takes it. The pipeline can also expand a
short prompt with its own text encoder (`expand_prompts` true, a shorter
version of the same instructions). No template uses it yet.

## Run and judge

1. `validate_workflow` first - free, and catches bad arguments.
2. Quote `plan.estimate` and name any `downloads_required` (the checkpoint
   is 26.7 GB). Each template is about 3.5 minutes on an RTX 3090, warm or
   cold: model offload moves the 16.6 GB text encoder on and off the card
   every run. Get the go-ahead, then `run_workflow` with `acknowledged_cost`
   set to the plan's `{fingerprint, minutes, downloads}`.
3. `wait_for_job`, `timeout_seconds` = estimate plus margin. Call again
   while `still_running`.
4. A saving step carries a `subfolder`: the one shown to the user is
   `final`, the rest `intermediate`. `list_gallery(subfolder="final")` lists
   only deliverables, so keep both in anything you compose.
5. Judge it yourself:
   - `get_output_frames(count=12)` for the clip's shape, and `at` 0 against
     the still for an image-to-video run, which should match it.
   - `seams=true` at each join of clips you cut together.
   - `get_output_audio` for the soundtrack. The model follows the audio
     sentence loosely: a sound asked for twice may come once.
   - Then `get_job` for the manifest and its warnings, and give the user the
     gallery `url`.
6. Save a keeper (`get_job_workflow`, `save_workflow`) to rerun it by name.

## Sources

The Kandinsky-6.0 model cards (kandinskylab on Hugging Face), the
kandinsky-6 GitHub repository and its beautifier prompts, diffusers PR
#14949 and `diffusers.pipelines.kandinsky6`. Read 2026-10-06; audit
`docs/proposals/audits/2026-10-06-kandinsky-6-audit.md`.
