# Overlap windowing for long video sources: `window_video`, `join_windows`, `ltx2/restore-long` (#601)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-06). Plan v1 was approved by Don on 2026-10-05, with Q1-Q3 answered
and one change to stage 3 (the count check is owned by the task, not the
template). Plan v3 (2026-10-06) moved the template's price into a stage of
its own after stage 3's build found a cost can't be derived. It was built
as four stages, #628, #629, #630 and #658, all shipped and verified
2026-10-06.

## The idea

`restore-*`, `refine-clip` and `upscale-clip` take at most one LTX bucket
(121 frames, about 5 s), and `refine-clip` trims a longer source. Nothing
cut a long source into overlapping model-sized windows and stitched the
processed windows back to the source's exact length. `dissolve_videos`
crossfades separate clips and shortens the cut by the overlap, so it loses
frames. The concept was observed in a source-available ComfyUI node set
whose licence is incompatible with Apache-2.0, so it was re-implemented
from the concept only.

## Verdict

**Build smaller.** There was no field report yet, so the value was
prospective. The cost was two utility tasks and one template. It needed no
new MCP tool, REST route, schema or syntax, and was highly reversible.

**Cut, with what brings each back:**
- **The H3 17n+5 grid.** The tasks are grid-agnostic. Every long-source
  consumer is LTX. *Back when:* an H3 video-to-video template needs it,
  which takes only a template.
- **Per-window audio blending in the join.** The join re-attaches the
  source's audio (Q2). *Back when:* a windowed template's model generates
  audio worth keeping.
- **Long refine and long upscale templates.** Only `restore-long` was built
  (Q1). *Back when:* a field report asks for a long refine or upscale.

## What the plan found against the idea

1. **"per_entry" is cost pricing, not iteration** (`dw/plan.py`). Iteration
   is `for_each` over a `variable:` list with `item:` and `gather:`.
2. **The split can't return a list of windows.** A list result fans every
   later `previous_result:` out per item, and nothing iterates a list made
   at run time. `window_video` returns one window per call, picked by
   `index`, driven by `for_each` over `variable:windows`.
3. **The grid comes from the existing `num_frames` constraint.** The window
   length is `num_frames` (8n+1 on LTX), and `stride = num_frames − overlap`.
   No frame-snap argument was added.
4. **The join can't read the window plan from its inputs.** Pipeline outputs
   don't carry their input's shots, so `join_windows` takes `source` and
   re-derives the plan from its frame count. That also supplies the audio.
5. **The window count is stated, not derived** (Q3). `for_each` expands
   before validation, so the list is a workflow variable, and a wrong count
   is refused with the needed count named.

## What was built

### Stage 1 (#628): `window_video`

Shipped `9015a5e1`, merged as `a83ad1cc`. The architecture-bounce fix is
`94231fb2` (merge `6b6da603`).

- `window_video(video, index, num_frames, overlap, fps=None)` in the new
  module `dw/tasks/windows.py`, with `assessment=False`. Window `i` spans
  `[i·stride − overlap, i·stride + stride)`. Frames before the source repeat
  frame 0, and frames past its end repeat the last frame. Frames are float32
  in [0,1].
- Audio is the source's samples for the real span, cut on source-frame
  boundaries with `frames_to_samples` (the #401 cumulative rule) through
  `dsp.slice_samples`. Synthetic frames get silence of matching length.
- Refusals: `overlap >= num_frames`, a negative `overlap` or `index`, and a
  window that starts at or past the source's last frame. Argument domains
  are in `TASK_ARGUMENT_DOMAINS`.
- Docs: `TASKS.md` and the `ARCHITECTURE.md` row "Overlap windows of a long
  video".

### Stage 2 (#629): `join_windows` and the shared count rule

Shipped `f9030f69`. The architecture-bounce fix is merge `c4659a2c`.

- `join_windows(videos, source, num_frames, overlap, curve="cosine",
  fps=None)` emits exactly `source_frames` frames. Each later window's first
  `overlap` frames blend over the previous window's last `overlap` at
  `t = (k+1)/(overlap+1)`, with cosine, smoothstep or linear weights. The
  last window's pad is dropped.
- `window_count` / `window_count_problem` in `dw/task_domains.py` hold the
  rule `ceil(source_frames / (num_frames − overlap))`. Both the run-time
  refusal and stage 3's static check call it.
- Refusals: the wrong count (naming the entries to add or drop), a window of
  the wrong length (naming it), mixed frame sizes, and an unknown curve.
- The audio is the source's track. A source with no audio gives none.
- Shots: one per window, with `overlap_frames` on every seam. The sample
  side goes through `dw/shots.py`'s `remeasured_shots`, so `assess_output`
  reads the seams as dissolves and the sync probe passes. The task is in
  `_CUT_TASKS`.

### Stage 3 (#630): the static count check, `ltx2/restore-long`, the skill note

Shipped merge `a6e2acaf`. The bounce fixes are `0663c233` (the default
source) and `48264374` (the path gate on `source`).

- `dw/window_count_errors.py`, wired into `validation_errors` as
  `window_count`. It is owned by the task: it applies to any `join_windows`
  step whose `source` is knowable (`asset:`/`output:` or a readable literal
  path, probed header-only), whose `num_frames`/`overlap` are literal, and
  whose `videos` is a `gather:` of known size. Anything else stays silent
  and is left to the run-time refusal.
- `TASK_MEDIA_ARGUMENTS` in `dw/locations.py` puts `join_windows.source`
  through the same path gate as a media key. Before this, a generic argument
  name escaped the validate-time walk (SE-F042).
- `workflows/templates/ltx2/restore-long.json`: the `window` for_each, a
  `restore` for_each on `restore-deblur`'s pipeline, and `join`. Its
  defaults are `asset:qa-cast/ep13-episode.mp4` (282 frames), windows w0-w2,
  `num_frames` 121 and `overlap` 16. `windows` stays out of `cost_drivers`.
- A note in the `dw:ltx-2.5` skill on choosing `windows`. Docs went into
  `TASKS.md`, `WORKFLOW_GUIDE.md`'s for_each list, the ltx2 README and
  `ARCHITECTURE.md`.
- The GPU run (C-F315, job `5d46a62ec00e`, cuda RTX 3090, cold) restored ep13
  in 3 windows in 12.24 min. The output had the source's length and audio,
  no visible seam and no sync drift.

### Stage 4 (#658): the measured price

Shipped merge `ddffc9d4`.

- `cost: [{device: cuda, name: RTX 3090, vram_gb: 24, minutes: 12.2,
  per_entry: {variable: windows, minutes: 1.56, entries: 3}}]`, from
  C-F315's run. 1.56 is one warm `restore@wN` step (1.60 and 1.48) plus its
  slice. The fallback (Don running it on lem) wasn't needed.

### Stage 5 (#666): fix-forward, the static check on an `output:` source

Shipped merge `53225a9e` (fix `bdb4ae53`).

- **Why:** the feature's final check failed C-F233 step 2. A wrong window
  count with an `output:` source validated clean, then failed at `join` after
  every window had run. Plan v3 names `output:` as knowable, so it was a
  build miss, not a plan gap, and needed no re-plan.
- **Cause:** `dw/server/admission.py` `admit()` activated the request
  workspace's assets for validate-time checks but left the output root at the
  default workspace's. The reference check passed the workspace root
  explicitly, so the reference resolved; the probe (`resolve_probe_path` ->
  `fetch_output`) used the ambient root, missed, and the check stayed silent.
- **Fix:** `admit()` also activates `workspace.outputs` (`activate_output_root`)
  for the same scope, covering validate, submit and rerun. It is at the shared
  scope, so every probe that reads an `output:` source gets it:
  `window_count_errors`, `dissolve_frame_errors`, the slice preflight and the
  warnings. Tests are in `tests/test_admission.py`, in a non-default workspace.
- Verified in the default workspace and in `qa-ep115`: 4 and 6 windows refused
  at `steps[1]` naming 5, 5 clean.

## Deviations from the plan

- **Stage 3 shipped unpriced; stage 4 priced it** (plan v3). A cost is
  measured, never derived, and `restore-deblur` had none to scale.
- **No plugin version bump.** The version is pinned to the engine's and
  moves only at release.
- **The default source is a server asset, `qa-cast/ep13-episode.mp4`.** No
  asset can ship through the repo (`/assets/` is gitignored), so the
  template's original `asset:long-blurry.mp4` default failed validation.
- **Stage 4's estimates read `basis: observed`, not `per_entry`.** The
  observed run's bucket is keyed on `cost_drivers`, which leave out
  `windows`, so it covers every window count and is tempered with the
  `per_entry` line. The figures still scale (10.1 / 11.2 / 12.2 / 13.2 /
  14.3 min for 1-5 windows). That precedence is #154's behaviour. The tester
  noted the label as a question for its owner, not a defect.

## Bounces per stage

The parent's final check bounced once (C-F233 step 2, an `output:` source),
which filed #666.

| Stage | Architecture review | Tester | Notes |
|---|---|---|---|
| #628 | 1 (second owner of frame-aligned slicing: use `dsp.slice_samples`) | 0 | |
| #629 | 1 (second owner of shot sample spans: use `remeasured_shots`) | 0 | |
| #630 | 0 | 2 (C-F314: the default asset didn't exist; SE-F042: `source` not path-gated at validate) | Paused once for the v3 re-plan |
| #658 | 0 | 0 | |
| #666 | 0 | 0 | Fix-forward from the parent's final check (C-F233 step 2) |

The stage comments name no `usage:` figures, so per-stage cost isn't
recorded. The plan estimated about $14.

## Deferred, with triggers

- **H3 grid, per-window audio, long refine/upscale templates:** see the
  verdict's cuts.
- **Content drift between windows.** A cross-window "breathing" was the
  plan's main risk. C-F315 saw no visible seam at `overlap` 16. *Back
  when:* a report shows drift. The fix is conditioning each window on the
  previous window's tail, which is a feature of its own.
- **`assess.py`'s `_sample_span` and `fade_samples` rounding** don't follow
  #401's cumulative rule. They didn't trip this feature's acceptance. A fix
  is the implementer's, as a bug of its own.
- **Suite tidy-up:** C-F234 and C-F235 (plan v2's pricing cases) have
  retire/amend requests open on the harness repo (dkackman/harnest#59, #64).
