# H3 multi-frame guides, and guide continuity in chains (#611)

> **Since #691 (2026-10-08):** `h3_blocks.py` named below is split into
> `h3_rules.py`, `h3_hold.py` and `h3_guides.py`; see
> `h3-blocks-split-complete.md`.

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-07). Plan v1 approved by Don 2026-10-05; v2 answered the tester's
spec questions, and v3 (2026-10-06) recorded Don's answers to v2 with no
change of scope. Three stages, #648, #649 and #650, all verified by
2026-10-07. Builds on #598's H3 block seam
(`h3-audio-hold-refine-complete.md`).

The concept came from vrgamegirl19/comfyui-vrgamedevgirl, whose
source-available license is incompatible with Apache-2.0. Everything here
was re-implemented from the concept. No code, prompt text or presets were
taken.

## The report

The issue was a parked idea: latent-space continuation between H3 chunks.
Save each shot's raw latent, slice a token-aligned tail, inject it as a
multi-frame keyframe at frame 0 of the next chunk, hold the matching audio
tail, and mark the next shot stale when one is re-rendered. It was parked
because diffusers' H3 layout (`modular_pipelines/minimax_h3/before_denoise.py`)
takes only single-frame `"first"`/`"last"` keyframes. Don unparked it after
#598, asking for a general guide: any length, at any frame index.

## Verdict

**Build smaller.** Ship the general guide engine and the audio guide, gate
the chain mode on a seam A/B, and cut persisted latents and stale-marking.
The strongest driver was #612: four of its LoRAs (LMS sharpness, style
transfer, VFX edit, head swap) need a full-length aligned guide at frame 0.
Chunk continuation had weaker evidence, since no field report asked for
smoother seams.

## What the plan found against the issue

- **The guide is a replacement layout step, not a denoise block.**
  `prepare_condition_latents` and everything after it already handle
  multi-frame condition rows. Only `prepare_layout` limits a keyframe to one
  frame. So `DwH3GuideLayoutStep` replaces that one step through #598's seam.
- **A guide steers the frames under it; it doesn't replace them.** The
  frames under a guide are still generated, so a chain renders the prefix
  and trims it at assembly. Replacing them in place (Option B) was left as
  a non-goal.
- **The audio tail can't use #598's hold.** Audio latents are packed
  channel-major stereo, so "the first M audio steps" is two row ranges, not
  a prefix. The tail became an audio guide: condition audio rows at the
  target's audio time origin, held at t=1.0 like Ref2VA reference audio.
- **Aligned lengths and indices.** The VAE encodes each 17-frame chunk on
  its own, so a clip encodes to whole latents only at 1, 5 or 17m+5 frames,
  and matches the target grid only at frame 17j. The issue's 1, 5, 22 and
  39 are the first four lengths.
- **Guide continuity can't run on the `last_segment` chains (v2).** Every
  `last_segment` template is `ref2va`, and Q2 keeps guides off `ref2va`. So
  the chain mode landed on `chained-segments` (`fl2va`) only.

## Decisions

- **Q1:** cut persisted latents and stale-marking (triggers under
  *Deferred*).
- **Q2:** guides on `t2va` and `fl2va` only, not `ref2va`. So there is no
  guide continuity on the `ref2va` chain templates either.
- **Q3:** the audio tail is an audio guide, not #598's hold.
- **Q4:** chain continuity is a `continuity: "guide"` option on
  `chained-segments`. It becomes the default only if Don reads the A/B that
  way. The default stays `last_frame`.
- **Q5:** the argument is `guides`, with entries `{video, frame, audio}`.
- **Q6:** `guide_frames` P is 22 or 39 and anything else is refused, not
  snapped; `carry_frames` is refused; `carry_audio` is honoured;
  `trim_frames` is ignored; `crossfade_ms` is ignored when the audio is
  held.
- **Guide limit:** 4, which stage 1 could lower from its VRAM measurement
  without a re-plan. It stayed at 4.
- **Order:** #620 (#598 stage 2) landed before #648, since both edit
  `h3_blocks.py` and the same anchor-pin test.

## What was built

### Stage 1, #648: the `guides` argument

- **`guides`** on H3 `t2va`/`fl2va`: a list of `{video, frame}`.
  `guides: []` is the same as none. `DwH3GuideLayoutStep` in
  `dw/pipeline_processors/h3_blocks.py` wraps the stock layout. Each clip
  is fitted to the canvas, VAE-encoded, and appended as condition rows
  timed from latent index `5*frame//17`. `num_condition_video_rows` is
  widened, so after-denoise slices the rows off.
- **t2va gets two wrapper blocks** (`dw_guide_condition_latents`,
  `dw_guide_latents`). They run the stock fl2va condition-latent step only
  when there are condition latents, because stock t2va has no such step and
  the fl2va one crashes on an empty list.
- **Rules:** `frame` an int, at least 0 and a multiple of 17. A length is
  snapped down to 1, 5 or 17m+5 with a warning. A clip ending past
  `num_frames` is refused, and ending exactly at it is fine. Validation
  (`dw/guides.py`) checks the same rules before the model loads, plus
  probe-based checks that the clip is a video. The 17m+5 grid goes through
  `variable_constraints.aligned`/`aligned_down` with `RENDER_GRID`, after
  the architecture review.
- **Docs:** *H3: holding a clip with `guides`* in `WORKFLOW_GUIDE.md`,
  including that a guide holds look as well as motion, and the
  reproducibility rule. The `minimax-h3` skill has a **Pinned to a clip**
  line.
- **VRAM** at 960×544×124 on an RTX 3090 (int4 SDNQ, turbo LoRA, 9
  steps):

  | guides | peak |
  |---|---|
  | 1 × 22 frames | 18013 MiB |
  | 4 × 22 frames | 19285 MiB |
  | 1 × 124 frames | 19291 MiB |

  `GUIDE_LIMIT` stayed at 4. The limit counts clips, not frames. Four long
  guides were not measured and probably don't fit in 24 GB.

### Stage 2, #649: the audio guide

- **`"audio": true` on a guide entry.** The guide video's audio over its
  span is pre-padded to the audio-VAE hop (keeping the time origin),
  encoded with the posterior mode, and prepended as condition audio rows
  by `DwH3GuideAudioStep` (`dw_guide_audio`), held at t=1.0.
- **Refused** at validate when the guide's video has no audio.
- **Argument realization** (`dw/arguments.py`) now loads a dict's `video`
  with its soundtrack (`load_audio_video`) when the same dict asks for
  audio. Before that fix, the realizer stripped the soundtrack and the run
  refused a guide that validate had accepted.
- **Measured:** a 39-frame guide with speech at frame 0 reproduced the
  guide's words in the opening, by word-timestamped transcription.

### Stage 3, #650: guide continuity on `chained-segments`

- **`continuity: "guide"`** in `chain.py`. Each segment after the first
  takes the previous segment's last P frames as a frame-0 guide, with
  `audio` set from `carry_audio`. On `fl2va` the keyframe is the guide's
  first frame. The first P output frames are trimmed at assembly, so a
  segment adds `num_frames - P` new frames.
- **Refusals** at validate and at run: on `ref2va`, with `references`, or
  on a non-H3 step; `guide_frames` other than 22 or 39; `carry_frames` set.
  The step rule is one function, `_takes_guides_problem`, shared by the
  `guides` check and the chain check, after the architecture review.
- **Template:** `chained-segments` gains the variables `continuity`
  (default `last_frame`) and `guide_frames` (default 22).
- **Skill:** the `minimax-h3` chain advice names `"guide"` on
  `chained-segments` only, and why it is not on the `ref2va` chains.

## The seam A/B (Don's gate on the default)

`chained-segments`, 3 segments, `num_frames` 124, seed 42, the template's
prompt (C-F311):

| arm | continuity | frames | minutes | seams |
|---|---|---|---|---|
| (a) | `last_frame` | 368 | 9.6 | a small pose snap at 124 and 246; a sharpness/lighting change at 246-249; `seam_level_step` warn at seam 1 (4.79 dB) |
| (b) | `guide`, P=22, `carry_audio` | 328 | 13.5 | no jump at 124 or 226; motion runs through; `assess_output` no findings |

The guide holds the seam better on picture and audio level. It costs about
40% more wall time for 40 fewer frames. No re-encode drift was seen, so the
trigger for persisted latents did not fire. **Don decides** whether
`"guide"` becomes the template's default, and whether this is enough
evidence to reopen Q2 (guides on `ref2va`). Until then the default is
`last_frame`.

## Bounces per stage

- **#648: three.**
  - Architecture: a second owner of the 17n+5 grid in `dw/guides.py` and
    `h3_blocks.py`. Moved onto `variable_constraints`.
  - Tester, C-F296 (2): the skill didn't mention `guides`. A **Pinned to a
    clip** line was added.
  - Tester, C-F301 and C-F302.
    - C-F301 expected a full-length guide plus a style prompt to restyle
      the take. It copies the take instead. The skill line saying "restyle
      a take" was wrong and was corrected. Restyling is #612's LoRA.
    - C-F302 expected a no-guides run bit-identical to the pre-#648
      baseline. lem runs with `cudnn_deterministic: false`, so no two runs
      there are bit-identical. A test now pins that the wrapper blocks draw
      nothing from the seed.
- **#649: one.** Tester, C-F304: the realizer stripped the guide's
  soundtrack (above).
- **#650: one.** Architecture: a second owner of the H3-step rule.
  Factored into `_takes_guides_problem`.

Cost per stage is not recorded: the stage comments name no `usage:`
figures. The plan estimated $19-29 over the three stages.

## Deferred and left open

- **Persisted raw latents** (the issue's "Save" step). It would need a
  tensor result type, an `output:` kind and a `.safetensors` load path.
  **Trigger:** visible seam drift caused by the pixel re-encode. The A/B
  showed none.
- **Stale-marking after a re-render.** A chain renders every segment in one
  job, so nothing goes stale. **Trigger:** a per-shot re-render workflow in
  the series skills.
- **Guides on `ref2va`** (Q2), and so guide continuity on the
  reference-carrying chains and a test of the #487 reference leak.
  **Trigger:** Don reading the A/B as clear evidence.
- **`"guide"` as `chained-segments`' default.** This is Don's call from the
  A/B.
- **A total guide-frames cap.** The limit counts clips. Four 124-frame
  guides were not measured. **Trigger:** an OOM from long guides, or a
  workflow that needs more than one long guide.
- **Replacing target frames in place (Option B).** **Trigger:** a guided
  prefix drifting visibly from its guide.
- **#612's aligned-guide LoRAs** can be built on `guides` now. That is
  #612's track.
