---
name: choosing-a-model
description: Use when a dw MCP server is connected and the user asks for a picture without naming a model family, workflow or model ("draw a lighthouse", "make me an image of...", "the best you can do"). Routes the request to the server's preferred model through templates/preferred/text-to-image and its tier, keeps SD 1.5 for "does it work" smoke tests, lets a named family or workflow win, and reads the plan's preference line before spending.
---

# Choosing a model on a dw server

A request that names no model still has to run on one. This skill picks it
from the server's **preference ladder**, not from memory: the server keeps
an ordered list of workflows per task and tier (`draft`, `standard`,
`best`), and resolves a `preference:<key>` path to the first one in the
tier that fits this host. The family skills (`flux3-klein`,
`minimax-h3-image`, `kandinsky-6`, ...) own their models' rules, and win
once a family is named.

## Before anything

This skill is written for dw `0.23.0-alpha.4`; `get_server_info`'s `plugin` says whether that is current.

1. `get_server_info`: the device. A ladder target can be CUDA-only, so the
   same tier can resolve to a different workflow on a Mac than on a CUDA box.
2. Decide what the user named, in this order; the first that applies wins.
3. Before choosing a tier, `get_preferences(key="text_to_image")` shows the
   ladder and this host's pick per tier, with each skip and its reason. It
   is free. Use it to say what each tier would run here; the validate line
   below stays the check before spending.

## Who decides the model

1. **A named family, workflow or model wins over the preference.** "Use
   FLUX 3 Klein", "run `models/z-image`", "on Qwen-Image": run that, through
   its family skill if it has one. Don't route a named model through the
   ladder, and don't swap it for the preferred one.
2. **A "does it work" ask is the smoke test.** "Check the server works",
   "a quick test image", a first run after install: run
   `templates/text-to-image`, the SD 1.5 baseline. It is the smallest,
   fastest, ungated template, there to prove the engine generates. It is not
   the default picture, so don't offer it for a real request.
3. **Anything else uses the preference.** A picture request that names no
   model runs `templates/preferred/text-to-image`, which composes
   `preference:text_to_image`. Use the matching `preference:` key for the
   task; `text_to_image` is the one the catalog holds today, so a request of
   another shape (video, an edit, audio) goes to `list_workflows(shape=...)`
   instead.

## The tier

`tier` is `draft`, `standard` or `best`. Pass the tier the user asked for.
The words map like this:

| The user says | `tier` |
|---|---|
| "quick", "rough", "a draft", "just to see" | `draft` |
| "the best you can", "highest quality", "final" | `best` |
| nothing about quality | **pass none** |

If they asked for none, pass none, so the store's default applies. Today
that default is `standard`. Don't pick `best` to impress: it is the slowest rung and the
biggest download. Don't pass any other value: an unknown tier is refused at
`arguments.tier`.

`draft` resolves the SD 1.5 baseline at 512x512; `standard` and `best` run
the server's chosen models at 1024x1024. An explicit `width` or `height`
wins over the tier's. A run makes one image.

## Before spending

1. `validate_workflow("templates/preferred/text-to-image",
   arguments={"prompt": ..., "tier": ...})`. Its `plan.preferences` holds
   one resolution line:

   `preference:text_to_image (tier standard) -> models/z-image; source: shipped`

   The workflow after `->` is what will run on this host. A `skipped` part
   names every target passed over and why (wrong device, too little memory,
   a gated repo). Tell the user which model that is before you spend, in a
   sentence: "On this server, standard is Z-Image Turbo."
2. Quote `plan.estimate` and get the go-ahead, as for any run.
3. A rung where nothing fits is refused, with each target's reason. Say so,
   and offer the next tier down; don't silently swap the tier.
4. A `declares no variable ... dropped` warning means the chosen target
   doesn't take an argument the template passes (MiniMax-H3-Image makes one
   image per run and takes no `num_images_per_prompt`). It is a warning, not
   a failure: the run still makes one image.

## Run and judge

`run_workflow`, `wait_for_job`, then `get_job`: the picture is the output
whose `subfolder` is `final` (the composed target's own scratch, if any, is
`intermediate`). Judge it with `get_output_image` against the ask. If the user wants
a different look rather than a better take, name the family the
resolution line picked and hand over to its skill: its prompt rules are the
family's, not this skill's. Captions already written to a family's format
are in the prompt library: `list_prompts`, then `get_prompt`.
