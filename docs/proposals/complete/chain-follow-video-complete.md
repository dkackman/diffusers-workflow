# Chain that follows consecutive windows of one long source video (#1061)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-31). Plan v1 approved by Don on 2026-10-31 ("approve defaults",
Q1–Q3) as "build smaller and gated": a probe first, the engine change only
if the probe passed. Three stages: #1074 (probe, verified), #1075
(`match_video` chain key) and #1076 (template and skill line), both closed
not planned when the probe's gate failed. **Nothing was built**: no code,
no template, no deploy. This doc is the record of the probe and of the
design, so a revival starts from both.

## The idea

Let a chain reference consecutive windows of one long source video, so a
long output (past H3's 345-frame cap) follows a driving clip's motion and
camera window by window, not only its own previous tail. Prior art:
einhorn13/mmh3_media's "follow timeline" (MIT, ComfyUI), where each
continuation segment's `<Video 1>` is the next window of the source on the
generation clock, a reroll reads the same window, the paired soundtrack is
sliced with it, and a source that runs out is an error, not padding.

## Decisions

- **No-code composition doesn't work.** A `for_each` over `window_video`
  entries can't do it: those windows start `overlap` frames early (the
  wrong clock), a `for_each` member can take only the whole previous
  render, not its tail, and a `chain` can't vary its reference per segment.
  The smallest working form was a chain key, `match_video: true`, slicing
  one source video reference per segment the way `match_audio` slices
  audio.
- **Correction to the issue's clock:** in dw the carry is a separate
  reference video, not frames inside the clip, so the stride would be
  `num_frames - trim_frames`, not `num_frames - context`.
- **Gate:** build B and C only if a probe showed (a) H3 ref2va's motion and
  camera follow a driving window and (b) a 124-frame shot with two video
  references fits a 24 GB card. Otherwise the verdict is don't build.
- **Q1:** #1045 (long H3 video-to-video via per-window source slicing,
  guides on ref2va) stays parked as the full version, cross-linked; it is
  now parked on this probe's result. **Q2:** segment count derived from the
  source and tail-trimmed, like `match_audio`. **Q3:** run the probe now.
- **Demand:** prospective. None of Don's workspaces restyles or
  motion-transfers a driving clip, and no field report asks for it.

## What the probe found (#1074)

Record: the #1061 comment "Stage A (#1074) probe record". Two ref2va runs on
lem (RTX 3090, 24,124 MB), develop @ b7fbc8e4: int4 sdnq, 768p turbo LoRA,
9 steps, sparse attention, seed 12345, 960x544, 124 frames, two
`MiniMaxH3VideoReference`s (`<Video 1>` the driving window, `<Video 2>` a
48-frame carry). Source `asset:qa-cast/ep49-episode.mp4`, which has a hard
cut at source frame 124.

| | Shot 1 (job `8bd76a6d24f2`) | Shot 2 (job `e0180de3a644`) |
| --- | --- | --- |
| window | `[0,124)`, no cut | `[85,209)`, cut at window frame 39 |
| carry | stand-in, unrelated `ep252` `[200,248)` | shot 1's own tail `[76,124)` |
| time | 1677 s | 1297 s (model cached) |

- **(a) Motion follows: no.** Shot 1, with no cut in its window, followed
  timing, position and framing. Shot 2 dropped the window's first shot
  entirely, rendered only the second from frame 0, ran about 40 frames
  ahead, and invented its last ~20 frames. It ignored the carry too. A
  fixed-stride window over an edited source routinely holds a cut.
- **(b) Fits: pass, thin.** Shot 1's highest sample was 23,364 MB reserved,
  111 MB free; shot 2 showed no OOM (its denoise wasn't sampled). Any
  heavier reference is past the edge, as #479 classifies.
- **(c) Seam: broken.** `get_output_frames(seams=true)` on the joined
  `[0,85)` + shot 2: difference 75.81 at frame 85; `assess_output` warned
  `seam_level_step` 6.08 dB and `shot_level_spread` 6.08 dB.
- **The restyle didn't take** in either run: both came out photoreal
  near-copies of the source, so "follows" meant near-reproduction, not a
  re-render.

The tester verified the record against the jobs (M-F202, M-F203 pass).

## What was not built, and what would reopen it

- **#1075 `match_video` chain key** and **#1076 `chain-follow-video`
  template + minimax-h3 skill line**: closed not planned on the gate.
- **Reopen on either of:**
  - a lighter H3 build with VRAM headroom to test a third reference, a
    longer carry or a stronger `<Video 1>` weighting, shown to follow a cut
    inside a window;
  - #1045's guide path (driving the shot from a guide rather than a ref2va
    video reference).
- **Untested observation, not a reopener:** shot 1, with no cut in its
  window, followed. Windows cut on the source's shot boundaries might
  sidestep the cut failure, but not the dropped carry continuity or the
  restyle not taking.

### The design, kept for a revival

- *Source:* the step's references hold exactly one video reference besides
  the carry; zero or two-plus refused before the first segment, naming the
  count (mirrors `_single_audio_reference` in
  `dw/pipeline_processors/chain.py`).
- *Timeline:* segment i reads source `[start_i, start_i + num_frames)`,
  `start_i` from `plan_segments`; a pure function of the index, so a reroll
  reads the same slice. The slice is a copy of the reference with `frames`
  and its paired `audio` cut to the window, never mutating the shared
  reference, decoded lazily one window at a time.
- *Length:* segment count derived from the source; `segments` and
  `match_audio` are each exclusive with `match_video`; the last segment
  overshoots and the output is tail-trimmed; a source shorter than one
  window is refused.
- *Continuity:* `last_segment` only (`guide` is refused on ref2va anyway).
- *Validation:* the same checks in `validate_workflow`, before GPU time.

Open points the tester raised on the spec, unanswered because B never
started: the shape (`sequence` vs `shot`, where chained H3 templates list
today), what the last short slice gets, whether the output muxes the
source's track or the generated audio, and whether omitted continuity
implies `last_segment`.

## Cost and bounces

| Stage | Outcome | Bounces |
| --- | --- | --- |
| #1074 probe | verified | 0 |
| #1075 `match_video` | not planned (gate) | — |
| #1076 template + skill | not planned (gate) | — |

The probe used about 50 minutes of lem GPU (1677 s + 1297 s). No stage
comment names a `usage:` figure, so session cost is left out.

The acceptance cases written for #1075 and #1076 (M-F204 to M-F213,
`pending:`) have nothing to test and are retired with those stages.
