# `ltx2/alpha-matte` and `ltx2/restore-colorize`: the remaining LTX-2.5 IC-LoRAs (#1064)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-01). The plan went through ten versions on #1064; v10 was approved
by Don on 2026-10-31 ("approve with defaults"). It was built as two stages:
#1079 (stage A, Alpha-Gen) and #1080 (stage B, Colorization). Stage A was
verified 2026-10-31 on develop @ 6b1dcfcb, stage B 2026-11-01 on develop
@ fc0a40d9, both on lem over MCP.

## The report

The 2026-10-02 LTX-2.5 audit (`audits/2026-10-02-ltx-2.5-audit.md`,
finding 5) listed four IC-LoRAs dw had not onboarded: Alpha-Gen 0.9,
Clean-Plate 1.0, Colorization 0.9 and Day-To-Night 0.9. #1064 asked for at
least Alpha-Gen as a template, with each of the others either onboarded or
recorded as not loadable. SDR-To-HDR was out of scope (engine work).

## Verdict

**Build smaller.** Alpha-Gen ships as one template that writes a grey matte
MP4. Colorization ships as `restore-deblur`'s pipeline wrapped in a fit and
a restore. Clean-Plate and Day-To-Night are recorded as not onboarded.
Demand evidence was thin (no field report asked for a matte or colourising,
and the comparable `restore-*` templates had one run apiece). Alpha-Gen was
worth it because nothing else in dw makes a per-frame soft matte; Don chose
to include Colorization (Q1) since it fits the existing pattern.

## What the plan found against the issue

1. **Alpha-Gen is a plain IC-LoRA.** No `alpha_gen` pipeline is needed: the
   card's recipe is the standard IC command with `--skip-stage-2`, an empty
   prompt, strength 1.0 and `reference_downscale_factor` 1. Card limits:
   up to 1920x1088, 8n+1 frames up to 145. Pad, don't stretch.
2. **dw can't write video with alpha** (`dw/result.py` writes h264/yuv420p
   only). The matte ships as a grey h264 MP4, white = foreground (Q2).
3. **Clean-Plate and Day-To-Night don't fit the distilled IC template**
   (distilled sigmas, guidance 1.0). Day-To-Night wants the dev recipe (30
   steps, CFG 3-4, STG, a negative prompt) and a 24 fps re-encode;
   Clean-Plate steers through the negative prompt, which guidance 1.0
   ignores. Colorization fits (960x544x121 at 24 fps).
4. **The registry's `workflow` field names an H3 transformer partition, not
   a template** (Q5). LTX entries keep `workflow: null`; promotion links to
   the template through an `evidence` item `{issue, note}`.
5. **Nothing carried a source's frame rate past a pipeline step.**
   `fit_to_model` kept the rate on the fitted video but not in its record,
   so a template could not say "the source's rate" in JSON (Q6, Q7).
6. **`restore-deblur` has no fit or restore step**, so following the
   source's rate in Colorization meant `alpha-matte`'s fit → model → final
   shape, not a one-line change.
7. **The Colorization IC-LoRA colours only what the prompt names** (found
   on stage B's first bounce). A generic prompt, or one naming only colour
   categories, left the tester's clip grey (`1521f00bcf5a`,
   `69e6bb12d56b`, both 36.7 dB PSNR against the source); a prompt that
   described the clip and named its colours coloured it (`f80056a98e62`).
   The stored default had coloured the build's own evidence clip only
   because both were street scenes.

## What was built

### Engine: the source's frame rate through a fit (stage A)

- `fit_to_model`'s record gains `source_fps` (the rate the source carries,
  or `null`). Additive: an older record without it still restores.
- `restore_to_source` gains an optional `fps`: omitted keeps today's
  behaviour (the input video's rate); `"source"` uses the record's
  `source_fps`, falling back to the input's rate; a positive number sets
  it. Non-positive or other strings are refused at validate.
- Documented in `docs/TASKS.md` and the task's parameter description. No
  existing template changed in this feature; #1090 later moved
  `upscale-clip`, `refine-clip`, `refine-in-place` and others onto it.

### Stage A (#1079): `ltx2/alpha-matte`

Commits: `71304eb8` (merged `ddb18608`), `9ec2df8a` (`c6a5878e`),
`da0fd435` (`99af9674`), `a99b0886` (`6b1dcfcb`).

- `workflows/templates/ltx2/alpha-matte.json`: `fit` (`fit_to_model`,
  letterbox, downscale 1) → `matte` (`LTX2InContextPipeline` with
  `restore-deblur`'s components and quantization, Alpha-Gen 0.9 at 1.0,
  empty prompt, model and step rate fixed at the card's 24) → `final`
  (`restore_to_source`, `fps: variable:frame_rate`).
- Variables: `frame_rate` defaults to `"source"`; `width`/`height` 32n,
  default 960x544 (Q4); `num_frames` 8n+1, 9-145; `seed`.
- `vae`, `audio_vae` and `vocoder` are `residency: on_demand` (Q8 (a)), so
  the 145-frame cap fits on a warm 24 GB card.
- The description and skill row say a hard cut inside the clip can leak
  source texture into the frames just after it, so matte one shot at a
  time (Q9 (a)).
- No own `vram_estimate`: it inherits deblur's through the same pipeline
  identity. Catalog cost 3.9 on the RTX 3090.
- `loras/ltx-2.5/ic-alpha-gen.json` promoted to `proven`, `workflow: null`,
  evidence job `7219cf6b4cf5`.
- Clean-Plate and Day-To-Night records (below), the `ltx-2.5` skill row,
  family README, `docs/ARCHITECTURE.md` IC count, and test rows
  (`TEMPLATE_PIPELINE_KEYS`, `CARDS`, `COMPACT_BUDGET`).

Verified: 960x544x121 default matte with soft hair edges; 384x544 and
544x544 sources come back at their own size; an 8 fps source gives an 8 fps
matte with the same frame count, `frame_rate=24` overrides it; 145 frames
on a warm card with no OOM (about 11.95 GB allocated after both arms); a
cut-free 145-frame source (`ic-alpha/priya-cutfree-145f.mp4`) is clean on
every checked frame, so the leak on `ep13-episode.mp4` comes from its cut
at frame 141, not the length.

### Stage B (#1080): `ltx2/restore-colorize`

Commits: `416c4068`, `60aef905`, `9c49e16a` (merged `65f26281`);
`9aff5c7d` (merged `3d211b35`).

- `workflows/templates/ltx2/restore-colorize.json`: `fit` → `colorized`
  (deblur's pipeline with `ltx-2.5-22b-ic-lora-colorization-0.9`,
  `lora_scale` default 1.0, model and step rate fixed at 24) → `final`
  (`restore_to_source`, `fps: variable:frame_rate`). The output keeps the
  source's size and frame rate and has no soundtrack (Q11 (i)).
- Stored prompt `ltx2/colorize_dual_panel`, in the card's dual-panel form.
  It is a worked example: the caller writes a prompt that describes their
  own clip and names its colours (Q10 (a)). The template description, the
  prompt's description and the skill row say so, and the skill tells the
  composing agent to look at a frame (`get_output_frames`) first.
- `loras/ltx-2.5/ic-colorization.json` promoted to `proven`, `workflow:
  null`, evidence naming the template and `f80056a98e62`.
- Skill row, README, ARCHITECTURE count, `TEMPLATE_PIPELINE_KEYS`, `CARDS`,
  `_IC_LORA_CONVENTION` and `COMPACT_BUDGET`.

Verified with clip-matched prompts: the black-and-white office fixture at
121 frames comes back in the named colours, same scene, no audio stream
(job `4e475c7b92cb`); an 8 fps source follows its rate and `frame_rate=24`
overrides it with near-identical frames (41.0 dB, SSIM 0.982); a 384x544
portrait comes back at its own size with no letterbox band or edge bleed.

### Records for the two not onboarded

`ic-clean-plate` and `ic-day-to-night` stay `trial`, `workflow: null`, each
with an `evidence` item `{issue: 1064, note}`:
- **Clean-Plate** reopens with a guided (dev/full transformer) IC route in
  dw, or a field report needing subject removal across a clip.
- **Day-To-Night** reopens with the same guided route plus fps resampling
  in `fit_to_model`, or a field report.

## Deferred, and why

- **RGBA / ProRes 4444 / PNG-sequence output and a composite task** (Q2):
  a grey matte is enough for an editor; dw has no alpha writer.
- **Mattes past 145 frames:** the card's limit.
- **Shot-cut detection in `fit_to_model`** (Q9 (d)): the description's
  one-shot-at-a-time rule covers it; a feature of its own if wanted.
- **An engine policy for releasing warm-worker GPU state** (Q8 (c)):
  `on_demand` residency cured the warm OOM without it.
- **An auto-caption step in `restore-colorize`** (Q10 (b)): the composing
  agent can see the frame and write the prompt.
- **The source's soundtrack through `restore-colorize`** (Q11 (ii)).
- **Clean-Plate, Day-To-Night, SDR-To-HDR:** recorded above / out of scope.
- **Moving other templates to the source's rate:** Don's Q7 audit, filed
  separately as #1090.

## Cost and bounces

No stage comment names a `usage:` figure, so cost is left out. The plan
estimated stage B's first build at about $3.5.

- **Stage A (#1079): 3 bounces, 4 builds.**
  1. M-F223: the registry `workflow` field was set to a template name; it
     names an H3 partition (Q5). Re-planned with the frame-rate change (Q6).
  2. M-F225's 145-frame cap OOMed on a warm card (job `376a537cc3e6`).
     Fixed by on-demand decoders (Q8 (a)).
  3. M-F272: a source-texture leak on frame 144, just after a shot cut.
     Fixed by the description (Q9 (a)); the cut-free arm confirmed the
     cause.
- **Stage B (#1080): 1 bounce, 2 builds.** The colour clause failed on all
  three fixtures with the default prompt (finding 7). Re-planned (v9/v10)
  to put the per-clip prompt on the caller. One verify session in between
  did not run (incomplete MCP surface) and is not counted.
- A licence gate on the Colorization LoRA parked stage B once until Don
  accepted it on Hugging Face (2026-10-31).

## Overlap left open

- **#597** (parked): the Restore 1.0 IC-LoRA "includes colourisation". If it
  is onboarded, its description should say when to pick it over
  `restore-colorize`.
