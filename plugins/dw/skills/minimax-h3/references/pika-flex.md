# Pika H3 Flex: a second H3 sampler, single clips only

Part of the `minimax-h3` skill; its *Hard rules* say when to read this.

## What it is

Pika H3 Flex is Pika's distilled adapter over MiniMax-H3's own weights,
run by its own diffusers modular pipeline (`pika_h3_flex`), not by H3's.
Three templates, one per H3 workflow:

- `templates/pika-h3/text-to-video` (t2va, from text);
- `templates/pika-h3/first-and-last-frame` (fl2va: `image`, `last_image`,
  either alone; Pika's blocks centre-crop each to the canvas, so frame 0 can
  be re-framed off the source - H3's fl2va holds it exactly);
- `templates/pika-h3/reference-to-video` (ref2va: `subject`, optional
  `subject_2`/`subject_3`, and `voice`).

The weights are MiniMax-H3's and share its int8 cache; a first run downloads
only the adapter (~2.8 GB) and Pika's small config files.

## Single clip only

Pika is its own block graph, so none of dw's H3 machinery runs on it: no
chain (`chained-segments`, `chain-*`), no `guides`, no `hold_audio`, no
`refine_sigmas`, no `regional-edit`, `replace-performer`, `face-repair` or
`add-sound`, no `turbo_lead_in`, and no `music-video`, `dialogue-short`,
`storyboard` or `shots-batch`. Anything chained, longer than one clip,
repaired or edited stays on the `minimax` templates. One 124-345 frame clip is
the whole of what Pika makes.

## The rules that differ from lightx2v

- **6 passes, no `num_inference_steps`.** The adapter's sampler is a fixed
  7-point step list, `[1000, 750, 500, 250, 100, 40, 0]`: 6 transformer
  passes, and `denoise_total_steps` reports 6. Lightx2v's "steps are the
  LoRA's passes plus one" does not apply: pass no `num_inference_steps`.
- **No shift.** Shift 12/3 is the adapter's own contract, in Pika's
  scheduler: pass no `video_shift`, `audio_shift` or scheduler `shift`, and
  no `lora_alpha` (the adapter runs at scale 1.0).
- **No guidance**: no `guidance_scale`, no negative prompt.
- **Cache off.** `cache` defaults to null. MagCache calibration and TST are
  skipped with a warning on a fixed step-list sampler.
- **Buckets.** `resolution` is `480p` (832x480, the default) or `768p`
  (1344x768). Other ratios at the same tiers, in multiples of 32, are listed
  in each template's description; pass `width` and `height` for one.
- **Frames**: `num_frames` is `17n + 5` from 124 to 345, snapped up (130
  runs 141), as on H3.
- **ref2va at 768p**: two images plus the voice fit 24 GB; three images
  plus the voice do not, and validate refuses it - use 480p or drop one.
  An audio reference must run 2 to 15 seconds; the run, not validate,
  refuses a shorter one, so check its length first.
- **Prompts** use Pika's labels, `@Image1`, `@Audio1`, and `SHOT 1 (0-3s):`
  beats, not H3's `<Picture 1>` and `<d>` blocks. Each template's default
  prompt is the shape.
- **Seeds are not comparable with H3.** The same seed and prompt on a Pika
  and an H3 template are two samples, not a matched pair.

## Measured against H3 (#1316 bench, RTX 3090, 1344x768x124, int8)

- fl2va: Pika takes ~1.3-1.4x the wall time of H3's 4-step lightx2v
  (~310 s against ~225 s): 6 passes against 4, each ~20% slower.
- ref2va: steady denoise is about even (Pika 6 x 51 s against H3's 8-step
  8 x 36 s).
- Pika followed prompted motion more closely and kept a reference's look
  where H3 invented detail; it also re-framed an fl2va keyframe and drifted
  stylistically mid-clip once. H3 held source framing exactly.

## When to pick it

<!-- pending Don's Q3 verdict on #1316 -->
