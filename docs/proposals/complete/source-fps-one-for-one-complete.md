# One-for-one templates follow the source's frame rate (#1090)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-01). Plan v1 on #1090 was approved by Don on 2026-10-31 with every
default (Q1-Q4). It was built as three stages, in the order B, A, C:
#1106 (stage B), #1107 (stage A) and #1108 (stage C). All three were
verified on the first attempt on lem over MCP: B on develop @ fc0a40d9,
A @ c69a6cbf, C @ f47d2adb.

## The report

Split out of #1064 by Don's Q7 ruling there: **a template whose output has
one frame per source frame keeps the source's rate unless the caller asks
for another.** Seven LTX-2.5 templates broke it. Each took a clip's frames
one for one and wrote them at a fixed 24 fps, so a 30 fps phone clip came
back 0.8x slow and, in the `pair_audio` routes, its audio was cut to the
wrong length. No field report named it, because most of Don's footage is
generated at 24. The evidence was the class: fps defaults had been fixed
one at a time six times before (#84, #104, #363, #561, #673, #1056).

## Verdict

**Build**, in three stages. Retiming (resampling fps in `fit_to_model`) is
a non-goal: it changes frame counts and breaks the one-for-one rule.

## What the plan found against the issue

1. **`face-repair` already had a carrier.** `paste_face_track` returns the
   clip's own fps, audio and shots. Only the declared `fps` on `pasted`
   made it 24.
2. **`restore-long` already had a carrier.** `window_video` and
   `join_windows` both fall back to the file's rate. Only the declared
   `fps` on the join made it 24. The issue was wrong that this route
   "needs another carrier". `join_windows`' `fps` refused `"source"`.
3. **Every template fed `variable:frame_rate` into the LTX pipeline**, so a
   `"source"` default needed the pipeline's own `frame_rate` made a literal
   `24.0`.
4. **`restore-deblur` and `restore-decompression` read nothing of the
   source's rate**: one pipeline step, the source as a reference only,
   960x544 out. They needed `restore-colorize`'s fit → model → restore
   shape (#1080).
5. **A declared `result.fps` on a `save: false` step does nothing**, and
   `result.fps` refuses a non-number, so `"source"` must never go there.
6. **The writer truncates fractional rates** (`int(fps)`), so a 29.97
   source wrote at 29. Filed and fixed separately as #1109 (Q4).

## The rule, as built

On all seven templates:
- `frame_rate` defaults to `"source"` and sets the **output** rate. A
  number overrides the rate, never the frame count. (On the generating
  templates `frame_rate` still means the model's rate; the descriptions
  say which.)
- The pipeline's `frame_rate` is a literal `24.0` and the model step's
  `result.fps` stays `24.0` (LTX output carries no rate).
- No `result.fps` downstream of the carrier.

## What was built

### Stage B (#1106): `face-repair`, `restore-long`, two task arguments

Commits `4256fddc`, `200a338b` (merged `fc0a40d9`).
- `paste_face_track` gains an optional `fps`: omitted or `"source"` keeps
  the clip's rate; a number above zero (numeric strings parse) sets it.
- `join_windows`' `fps` also accepts `"source"`; its `POSITIVE` domain is
  replaced by the same source-or-positive check.
- One owner for that check: `task_domains.source_fps_problem(fps,
  command)` (renamed from `restore_fps_problem`), called by
  `restore_to_source`, `paste_face_track` and `join_windows` at validate
  and at run time.
- `face-repair`: `pasted` takes `fps: variable:frame_rate`; `fps` dropped
  from `crop` and `pasted`. `restore-long`: `join` takes
  `fps: variable:frame_rate`; `fps` dropped from `join` and `window`.
- `docs/TASKS.md`, both task parameter descriptions, both template
  descriptions and the ltx-2.5 skill rows.

Verified (M-F278 to M-F284): on a 30 fps source, `face-repair` writes 124
frames @30 fps with the source's audio in sync, `frame_rate=24` writes at
24; `restore-long` joins three windows to 282 frames @30 fps with the
source's audio; the 24 fps control is unchanged.

### Stage A (#1107): `upscale-clip`, `refine-clip`, `refine-in-place`

Commit `e748fc2f` (merged `c41fb9f0`).
- `restore_to_source` takes `fps: variable:frame_rate` (#1079's argument).
- `with_source_audio` declares no `result.fps`; the dead `fps` on the
  `save: false` restore and on `refine-clip`'s latent-upsampler step were
  removed (the latter would have resolved to `"source"`).
- Descriptions: "the model runs at 24 and the output is written at the
  source's frame rate; 'frame_rate' overrides that rate, never the frame
  count", replacing "set 'frame_rate' to the source's".
- Tests include a real-path class: an 8 fps source through each
  template's own task steps, pipelines stood in for at 24 fps.

Verified (M-F273 to M-F277): `refine-in-place` and `upscale-clip` on a 30
fps source give 121 frames @30 fps with the audio paired to 4.033 s and no
fps warnings; `frame_rate=24` gives 121 frames @24. `refine-clip` was
checked statically only (same restore-and-pair path).

### Stage C (#1108): `restore-deblur`, `restore-decompression`

Commit `1d1a4eeb` (merged `f47d2adb`). Labelled `breaking-change`.
- Both take `restore-colorize`'s shape: `fit` (`fit_to_model`,
  letterbox, `downscale: 1`) → `restored` (the IC pass on the fitted clip
  at 24) → `final` (`restore_to_source`, `fps: variable:frame_rate`).
- **The output now comes back at the source's size**, not 960x544.
- **The output is silent.** The plan said the model's generated audio
  stays, but `restore_to_source` drops audio and shots, as in colorize
  (#1064 Q11 (i)). The build followed that precedent: both declare
  `"traits": ["needs-input-media"]` (so the catalog no longer derives
  `has-audio`) and the descriptions say "The output has no soundtrack".
- Skill row, `workflows/templates/ltx2/README.md`, test pins.
- One lem run (job `66d0ac46f4e3`, deblur, cold): 111 s wall. The curated
  `cost` stays at 3.4 min.

Verified (M-F285 to M-F288): on a 384x544 30 fps portrait, 97 frames,
both templates come back 384x544, 97 frames @30 fps with no letterbox band
or edge bleed; `frame_rate=24` gives 24 fps with the frame count kept.

## Deferred, and why

- **Retiming in `fit_to_model`**: a non-goal; it returns with Day-To-Night
  or a field report asking for a rate change.
- **`extend-clip`, `dissolve-between-shots`, `assemble-and-score`** (Q1):
  out of scope. `extend-clip` is a generator; the other two join
  generated 24 fps shots and take an explicit `fps`.
- **The source's soundtrack through `restore-deblur` /
  `restore-decompression`** (Q3): no `pair_audio`. The outputs are silent
  (above); a `pair_audio` step is the change if a caller wants the sound.

## Cost and bounces

No stage comment names a `usage:` figure, so cost is left out. The plan
estimated about $9.5 in all ($3 A, $3 B, $3.5 C).

- **Stage B (#1106): 0 bounces, 1 build.**
- **Stage A (#1107): 0 bounces, 1 build.**
- **Stage C (#1108): 0 bounces, 1 build.** The portrait fixture
  `fps-1090/priya-30fps-384x544-97f.mp4` was missing at verify and was
  rebuilt from the suite's recipe.
