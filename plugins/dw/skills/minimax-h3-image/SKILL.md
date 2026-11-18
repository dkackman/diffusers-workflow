---
name: minimax-h3-image
description: Use when a dw MCP server is connected and the user wants a MiniMax-H3-Image still - a cast still or keyframe that will become the first (or last) frame of a MiniMax H3 video, a large high-resolution picture from a long description, or an edit or composition of several reference pictures at high resolution. Says when the host can run it and when FLUX 3 Klein is the better choice, states the host-RAM rules, quotes cost, and routes the still into H3 video.
---

# MiniMax-H3-Image on a dw server

MiniMax-H3-Image draws stills on a fixed six-step sampler, with the same
Qwen3-VL text encoder as MiniMax H3. Host RAM is what it costs, not VRAM: it
holds about 64 GB of host RAM and under 6 GB of VRAM. Use it for a still
that feeds H3 video and for large multi-reference edits. Anything quick, a
short request, a layout by boxes, or a host that is too small goes to the
`flux3-klein` skill.

## Before anything

1. `get_server_info`: the device and the workspace. The recipe is CUDA only.
   On `mps` or `cpu`, stop and use the `flux3-klein` skill.
2. Host RAM: the host needs about 70 GB or more, and the warm run holds
   about 64 GB (`host_ram_gb: 64`). `validate_workflow` refuses a run on a
   host whose total RAM is too small, naming `host_ram_gb`. Its
   `plan.host_memory` gives the projected peak. On a refusal, or a host
   under about 70 GB, use the `flux3-klein` skill instead.
3. `list_workflows(shape="image")` and `list_workflows(shape="image-edit")`:
   take names, `summary` and `cost` from the listing over the names quoted
   here, then `get_workflow` on the one you chose.

## Which shape is the request

- **A still from a description**: `templates/minimax-h3-image/text-to-image`.
- **An edit, or several pictures combined** (a cast member placed in a set,
  two portraits in one frame, one photo changed):
  `templates/minimax-h3-image/multi-reference`, `images` a list of one or
  more. A single-image edit is a list of one.
- **A cast still or keyframe for an H3 clip**: either template, then the
  H3 video in a second job (*Into H3 video* below).
- **A short request to expand, a layout by boxes, a quick draft, or a
  storyboard's board sheet**: the `flux3-klein` skill. Klein is faster
  and needs far less host RAM. H3-Image is for the still that will be filmed.

Input media must be on the server: `upload_asset`, then `asset:` in the
argument, or `output:` for an earlier run's file.

## Hard rules

- **About 64 GB of host RAM warm, about 120 GB on a cold first run.** The
  first run on a host without the quantized cache quantizes the weights,
  which peaked at 117 GB measured and took 38 min instead of 2. Warn the
  user before that first run. Changing `weights_dtype`, or any quantizer
  argument, misses the cache and pays that cold pass again.
- **Never beside another host-heavy job.** Two cards share one host's RAM.
  An H3 video or another H3-Image run on the other card can push the host
  over. Let the server's host-RAM gate hold the second job rather than
  routing around it.
- **No step count, guidance or negative prompt.** The checkpoint is
  distilled to a fixed grid of 6 steps with its guidance baked in, and the
  templates declare none. A request to "use more steps" is not a lever here.
- **Sides are multiples of 32**, and validation refuses any other size. `size`
  picks 2048x2048 (the default) or one of the vendor's sizes of about 16 MP.
  The 16 MP sizes have not been timed on a 24 GB card.
- **No speedup below 2048x2048.** The pipeline generates any canvas under
  4,194,304 pixels (2048x2048) at that area and resizes it back. The pass is
  bound by streaming the offloaded weights in any case: 1024x1024 measured
  15.1 s a pass against 15.2 s at 2048. Never offer a smaller canvas as the
  cheap option. Offer Klein.
- **The first download is about 64 GiB.** The text encoder comes from
  MiniMax-H3's repository at the commit the H3 templates pin, so a host that
  runs H3 already holds it.
- **The license is MiniMax's own.** Its territory and commercial terms are
  narrower than an open licence's, so point the user at the model card's
  LICENSE before anything commercial.

## Cost

Each run is priced on the slower of the reference host's two RTX 3090s:
text-to-image 2.3 min, and multi-reference 2.7 min with three references,
both at 2048x2048. The offload is bus-bound, so a card on an x16 link runs
it in about half the time. Quote `plan.estimate` from `validate_workflow`,
not these figures, and add the cold first run when the host has never run
it.

## Prompts

- **Long and literal.** The prompt is a detailed English description of the
  whole picture. It is used verbatim, with no chat template, and is
  truncated at 4096 tokens. Say the task first, then the layout, then the
  details and the style.
- **No prompt enhancer.** MiniMax's expander is a separate 27B model whose
  diffusers blocks run remote code (`trust_remote_code`), and dw does not
  use it. Write the full description yourself.
- **References are one-based**: `<Picture 1>`, `<Picture 2>`, in `images`
  order. Klein's `<ref_image_0>` is zero-based, so a Klein prompt does not
  carry over. Give each reference a role and say what stays as it is. The
  vendor states up to 16 references. Three have been timed here.
- **Camera and look words**: the `cinematography` skill, through its
  MiniMax-H3-Image compile table.
- Stored prompts: `list_prompts`, then `get_prompt` on a project's shot list
  or cast sheet. The template defaults are worked examples of the format.

## Into H3 video

A still becomes an H3 clip in two jobs, not one chained template:

1. Draw the still with the canvas at the video's aspect, twice its size:
   1920x1088 for `templates/minimax/image-to-video` (960x544),
   2688x1536 for `templates/minimax/video-with-audio-768p`'s 1344x768, or
   1088x1088 for `templates/minimax/first-and-last-frame` (544x544). All
   are multiples of 32. A different aspect has its subject cropped.
2. Judge it with `get_output_image` before any video is paid for.
3. Pass it as `output:<path from get_job>` to the H3 template's `image`
   (and `last_image` for first-and-last-frame), or `keep_output` it as a
   cast asset and cite `asset:`. The `minimax-h3` skill picks the H3
   template and its rules.

The cast stays one person when every keyframe of a character is a
multi-reference edit of the same cast still, with the face named as kept.

## Run and judge

`validate_workflow` (free) and quote `plan.estimate`. Then `run_workflow`
with `acknowledged_cost`, `wait_for_job` and `get_job`. Each template saves
its still with `subfolder` `final`. In a workflow you compose around it,
keep that, and mark any scratch step `intermediate`, so
`list_gallery(subfolder="final")` lists only deliverables. Look at the still
with `get_output_image` against the request: the faces of every reference,
and the text if any was asked for. Give the user the gallery `url`.

## What goes wrong

- **The first run takes about 38 minutes and almost all of the host's RAM.**
  That is the one-time quantize, not a hang. Watch `get_job_events` and
  don't cancel it unless the events stop.
- **The job is held or refused for host RAM.** Another host-heavy job is
  running, or the host is too small. Wait for the other job, or use Klein.
- **A face from one reference drifts.** Name it as kept: "keep the face and
  hairstyle in <Picture 1> exactly". Two people in one frame is the hardest
  case, so check both.
- **A wrong size is refused at validate.** That is the 32-pixel rule.
  Choose the nearest multiple of 32 yourself; nothing snaps it.

## Sources

The diffusers modular pipeline (`diffusers.modular_pipelines.minimax_h3_image`),
MiniMax's model card and repository, and `workflows/templates/minimax-h3-image/README.md`
with its audit. Timings and memory are from `bench/records/h3-image.md`.
