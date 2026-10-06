# H3 audio-hold + refine pass after latent upscale (#598)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-06). Plan v1 approved by Don 2026-10-05; v3 (schedule B) approved
2026-10-06 and folded into v4. Four stages, #618, #619, #620 and #621, all
verified by 2026-10-06. Follows #471 (`h3-latent-upscale-complete.md`),
whose deferred refine pass this closes, and #500's "close but soft" gate.

The concept came from vrgamegirl19/comfyui-vrgamedevgirl, whose
source-available license is incompatible with Apache-2.0. Everything here
was re-implemented from the concept. No code, prompt text or presets were
taken.

## The report

Two linked ideas:

1. **Audio-hold.** VAE-encode a fixed track, fit it to the audio latent and
   keep it un-denoised, so H3 generates video *to* that track instead of
   only taking it as a reference.
2. **Refine.** Re-noise an upscaled H3 latent to a low σ and run a few
   steps at 768p, holding the base pass's audio, to close #500's softness.

## Verdict

**Build, with the refine template gated on an A/B.** Hold was the cheap
seam that refine needed. Its own lip-sync value was the stronger claim on
paper, and it did not survive measurement (stage 4). Refine's value was the
weaker evidence, so the template carried a gate that is Don's call.

## What the plan found against the issue

- **diffusers' H3 denoise steps every audio row after the condition rows.**
  So hold widens `num_condition_audio_rows` over the held rows before
  `set_timesteps` and narrows it again before `after_denoise`. No copy of
  the denoise loop is made.
- **Plan v2's refine schedule could not run** (found in #620's build).
  "The grid points below σ₀" at 9 steps, shift 6, strength 0.2 is zero
  evaluations: the shifted grid is `1, .977, .947, .909, .857, .783, .667,
  .462, 0`. Don chose option (B) over a single evaluation (A) and
  ComfyUI-style grid-fraction denoise (C).
- **`set_timesteps(sigmas=...)` applies no shift** and needs a strictly
  decreasing list ending at 0. So dw computes the shift-spaced tail itself,
  reading the shift from the loaded scheduler.
- **The audio scheduler is stepped even when every audio row is held.** It
  gets the same σ list, and the block asserts that no audio row would be
  stepped.

## Decisions

- **D1:** engine H3 blocks in `dw/pipeline_processors/h3_blocks.py`, the
  one place dw modifies a modular pipeline, with an `ARCHITECTURE.md` row.
- **D2:** the blocks are always inserted, as no-ops without their argument.
- **D3:** the call arguments are `hold_audio` and `refine_strength`. No
  MCP tool, REST route or schema change.
- **D4 (superseded):** hold replaces the per-shot audio reference in the
  templates. Reversed by #619's A/B.
- **D5:** build order 1 → 4 → 2 → 3.
- **D6:** refine schedule (B). σ₀ = `refine_strength`, then
  `num_inference_steps` shift-spaced points to 0, which is
  `num_inference_steps − 1` evaluations. At shift 6, 5 points, 0.2:
  `0.2, .157, .109, .057, 0`.
- **D7:** the hold fallback is final. The templates keep the reference and
  `hold_audio` is opt-in.
- **D8:** the template refines at 5 points and strength 0.2.

## What was built

**Stage 1, #618: hold blocks (`hold_audio`).**
- `DwH3HoldAudioStep` and `DwH3ReleaseAudioStep` in `h3_blocks.py`. Hold
  encodes the track through H3's Ref2VA reference-audio encoder, writes it
  over the target audio rows after any reference rows, and crops or pads
  with silence (never stretches). The output carries the original waveform,
  not the VAE round-trip (`output_extraction.py`).
- `insert_audio_hold` places the blocks at component load in the whole
  graph and in the `workflow=`-pruned flat graph. `dw/introspection.py`
  builds the same graph without weights, so `get_pipeline_signature` lists
  a modular pipeline's block inputs, `hold_audio` included.
- Static check in `dw/hold_audio.py`, beside validation. The run-time
  refusal is in `pipeline.py`.
- A path goes through `locations.validate_media_path`. A URL with a
  non-http(s) scheme is refused by the location owner, with
  `security.validate_url` deciding the scheme.

**Stage 4, #619: hold in the templates + lip-sync A/B.**
- The first ship put hold in `music-video`, `chain-matched-to-audio` and
  `chain-matched-and-aligned`, replacing the reference (D4).
- The A/B (seed 42, a 10 s sung clip `asset:h3-hold/sung-10s.wav`, the
  Priya portrait) went against hold: the mouth was open at ~3/6 onsets with
  hold, against 6/6 with the reference, at 19.8 vs 18.3 min.
- Re-shipped on the plan's fallback: the templates keep the reference, and
  the guide, `docs/TASKS.md` and the `minimax-h3` skill call hold opt-in,
  with those numbers. `chain.py` keeps a hold branch for a hand-built
  chain.

**Stage 2, #620: refine schedule (`refine_strength`).**
- `refine_sigmas(strength, num_points, shift)` and
  `DwH3RefineScheduleStep`, inserted before `denoise` in every
  core-denoise sequence. It rebuilds `timesteps`, `audio_timesteps` and
  `row_timestep_plan`. It re-noises only the generated rows, through
  `scheduler.scale_noise`, with the step's generator.
- Refusals (static and at the call, through `refine_problems`): strength
  outside (0, 1) or not a number, no `latents`, no `hold_audio`, a non-H3
  step, and `num_inference_steps` below 2.
- `WORKFLOW_GUIDE.md` *Promoting an H3 take to 768p in latent space* keeps
  the upscale-only example first and adds a *Refining the upscaled
  latents* subsection with the cost note.
- `ARCHITECTURE.md`'s row is now "H3 audio hold and refine", with three
  blocks.

**Stage 3, #621: `templates/minimax/upscale-refine`.**
- Steps: `base` at 960x544 on the 544p turbo LoRA (shift 12, 9 steps,
  latents out), then `up` with `upscale_h3_latents` to 1344x768, then
  `refine` on the 768p turbo LoRA (`video_shift` 6, 5 points, strength
  0.2, `hold_audio` from the base).
- `cost_drivers`: `num_frames`, `num_inference_steps`, `width`, `height`,
  `refine_strength`. `cost`: 12.5 min, RTX 3090. A `minimax-h3` SKILL row
  points to it for keeping a 544p take at 768p. `COMPACT_BUDGET` raised.

## The refine A/B (Don's gate)

M-F086: the crowd-faces prompt from #500, seed 42, 124 frames, on lem.

| arm | route | minutes |
|---|---|---|
| (a) | upscale-only | 7.89 |
| (b) | `upscale-refine` | 11.53 |
| (c) | native `video-with-audio-768p` | 9.59 |
| (d) | #612 LMS upscaler | not landed |

- (b) keeps (a)'s composition and cleans it up: faces intact, cobble
  texture restored, nothing smeared or doubled. It is still a little softer
  than (c).
- (c) is a different take, the sharpest, and **cheaper than (b)**.
- The tester's read: (b) earns its place only when a specific 544p take
  must be kept at 768p. For a fresh 768p clip, (c) is faster and sharper.
  The SKILL row says this.

**Don's ruling (2026-10-06, on #598): no.** Refine spends about 20% more
GPU time than native (11.53 against 9.59 min) to reach the same 1344x768,
and comes out softer. Its 768p decode is native's decode, so it saves no
VRAM either. The template, its SKILL row, its README and checkpoint-table
rows, the guide paragraph naming it and the `COMPACT_BUDGET` raise are
reverted by fix-forward stage #664. The engine stays: `refine_strength` is
opt-in surface, with the guide's *Refining the upscaled latents* section
and its example, as with #499. A cheaper base-plus-upscale route (e.g.
#612's LMS upscaler) that could bring refine in under native would be a new
idea with its own A/B.

## Bounces per stage

- **#618: four, then a park.** Each was a different problem:
  - architecture: the hold opened a path itself, outside `dw/locations.py`;
  - tester: the H3 signature didn't list `hold_audio`;
  - tester: SE-F040, a `file://` location validated clean (reported
    privately);
  - architecture: the URL-scheme rule had a second owner outside
    `security.validate_url`.
  Don handed it back after the park; it verified on the fifth hand-off.
- **#619: one.** The A/B went against hold, and the fallback re-ship
  verified.
- **#620: two stopped builds, then one architecture bounce.** The builds
  stopped on the impossible v2 schedule, which led to plan v3. The review
  found a hand-written re-noise (now `scale_noise`) and a second owner of
  the refine rules (now `refine_problems`). Verified on the next hand-off.
- **#621: none.**

No `usage:` figures were recorded on the stages, so cost is left out.

## Deferred and left open

- **A re-measure of hold's lip sync** (D7): one seed, one sung track. It
  comes back as its own idea only if a field report shows reference lip
  sync failing where hold might help.
- **The 3-pass (2 MP) variant**, LTX-2 changes, persisted latents, and
  task weights in `downloads_required`: non-goals.
- **#612's LMS upscaler arm** of the A/B: not landed. A refine route on it
  is a new idea, not a reopening of this one.
- **M-F077's fixture** `asset:h3-hold/vwa-seed42-baseline.mp4` was never
  seeded. It can only come from a pre-#618 build, so that case was skipped.
- **dkackman/harnest#62** asks to retire M-F084 and M-F082's stale cost
  line, which v2's schedule wrote. M-F096 and M-F098 are authoritative.
