# `ltx2/upscale-clip`: LTX-2.5 generative 2x upscale of an existing mp4 (#542)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-02). Plan v2 was approved by Don on 2026-09-27. He answered Q1-Q4:
build, the default prompt, `pair_audio` for the source's track, and
`upscaled` kept in `intermediate/`. It was built as one stage, #548, which
shipped 2026-09-28 and was verified 2026-10-03.

## The report

`templates/ltx2/generative-upscale` already did a generative 2x upscale in
its `upscaled` step: `LTX2InContextPipeline` with
`reference_downscale_factor: 2`. Its reference, though, was always a clip
the template had just generated (`previous_result:low_resolution.frames`).
#542 asked for the same upscale on an mp4 from the asset library. #512 (a
general video resize task) was declined, so nothing else gave a way to add
resolution to footage dw did not make.

## Verdict

**Build smaller.** One template, one stage, and no validation that probes
the source. The value was stated as thin: no field report asked for it, and
none of Don's recent jobs ran the nearest siblings. It was worth building
because it is the only route of its kind and fully reversible: a catalog
entry with no engine, MCP, REST or syntax surface.

**Cut:** checking `width`/`height` and `num_frames` against the probed
asset. No pipeline reference is probed at validate today, so it would have
been an engine change. It comes back if a field report shows a run wasted
on a mismatched source.

## What the plan found against the issue

1. **"width/height must be 2x the source" is neither true nor enforceable.**
   The pipeline scales the reference to width/2 × height/2 and center-crops
   it (`pipeline_ltx2_ic_lora.py` `:1403`, `:1424`,
   `resize_mode="crop"`). Any source runs. A source with a different aspect
   ratio is silently cropped, and a larger one is silently downscaled.
2. **"num_frames must match the source" doesn't hold either.** The
   reference is trimmed to `num_frames`. A shorter source isn't padded, so
   the output's tail is generated with no reference.
3. **It isn't `generative-upscale` minus a step.** `upscaled` borrowed its
   transformer and text encoder through `reused_components`. A standalone
   step needs `restore-deblur`'s full SDNQ component blocks.
4. **The upscaler's numbers were pinned nowhere**, including for
   `generative-upscale`.

## What was built

### Stage A (#548): the template, its pins, and its docs

Shipped `develop` @ `aa04263` (`4a826ae` template + tests, `53baa0b`
docs). The bounce fix is `77bb905` (`d8d4970`).

- **`workflows/templates/ltx2/upscale-clip.json`**:
  - `upscaled`: `LTX2InContextPipeline` + `LTX2ReferenceCondition` over
    `variable:source_video` (default `asset:clip.mp4`), with
    `reference_downscale_factor: 2`. The lora is
    `Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler` /
    `ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors` at
    `lora_scale` 1.0, on the restore templates' SDNQ components. Saved to
    `intermediate/`.
  - `with_source_audio`: `pair_audio(video=previous_result:upscaled,
    audio=variable:source_video, fit="video")`, saved to `final/`. A silent
    source fails here with "carries no audio track", after the upscale is
    saved.
  - Defaults are 960×544, 121 frames, 24 fps: a 480×272 source doubled.
    `variable_constraints` are 32n width/height and 8n+1 `num_frames`
    (min 9, no snap). `cost_drivers` are `num_frames`/`width`/`height`, with
    no curated `cost`.
  - The description states the crop/downscale rule, `num_frames` ≤ the
    source's length, reading the source first with `get_gallery_metadata`,
    that it is generative and not a faithful resize, and to run
    `restore-decompression` first on a compressed source.
- **The default prompt (Q2): a literal**, not a stored `prompt:`. The vendor
  card states no caption convention, so there was no trained form to store.
  The literal is "A high-resolution rendering of the scene in the reference
  video, with the same subject, framing and motion, and sharp, fine
  detail.", and the description tells the caller to replace it.
- **Tests:**
  - `tests/test_ltx2_ic_loras.py`: a `CARDS` entry for `upscale-clip`, its
    place in the foreign-clip parametrize, and a new pin for
    `generative-upscale` against the same card.
  - `tests/test_pair_audio.py`: an mp4's track read through `pair_audio`,
    and a silent mp4 named as the fault.
  - `COMPACT_BUDGET` went from 8_850 to 9_000 (measured at 8_910).
- **Docs:**
  - the `ltx-2.5` skill: the description, the 2x line and the "Repairing
    the user's footage" line;
  - the `ltx2` README: a "Quality and scale" row and the restore section's
    wording;
  - `CLAUDE.md` "LTX-2.5 IC-LoRAs".

**Vendor card against `generative-upscale`:** no contradiction. It names the
same weight, factor 2 and strength 1.0, so `generative-upscale` was left
unchanged and is now pinned. The card states no trained bucket, so the
defaults are the family's. The card also scopes the LoRA as a creative
step, not for live-action footage where fidelity matters, and the
description says so.

## Measured in verification

From verify pass 3 on lem, `develop` @ `f01ae1fa`:
- **Happy path**, 480×272/121f/24 fps source with a soundtrack: 220.4 s.
  The final is 960×544, 121 frames, 24 fps. Its audio is the source's
  (every level within 0.03 dB, and 18 ms longer). `assess_output` found no
  sync issue, and the framing is unchanged.
- **16:9 source** (640×360): center-cropped to 960×544 with only a thin
  side trim, as 1.778 → 1.765 predicts. 49.5 s, warm.
- **`num_frames` 161 on a 121-frame source**: 161 frames, with an invented
  tail. `pair_audio` warns that it padded the track with silence. 140.3 s.
- **Silent source**: fails at `with_source_audio` with the upscale kept in
  `intermediate/`.
- **Larger source** (960×544): downscaled first, then runs (200.7 s, pass 2).

## Bounces per stage

- **Stage A (#548): one bounce.** M-F058 failed because the skill's
  "Repairing the user's footage" section didn't name `upscale-clip`. The
  fix had to fit the skill's size cap (`SKILL_SIZE_LIMIT` 12288, now
  exactly 12288), so "restore a compressed source first" was dropped from
  the skill; it stays in the template's description.
- One park, which wasn't a bounce: the `upscale/*` fixtures (480×272 with
  and without audio, 640×360) had to be cut off-box with ffmpeg. Don placed
  them in `regression-model-specific`'s asset library on 2026-10-02. The
  suite recipe's `upload_asset` path was corrected in dkackman/harnest
  `17d65f4`.

No `usage:` figures were recorded on the stage, so cost is left out.

## Deferred

- **Probing the source at validate** for size and length. It comes back on
  a field report of a wasted run.
- **Giving the restore templates the same source-audio treatment** (Q3).
  This needs its own issue.
- **The vendor's Refine-Details IC-LoRA 1.0** (2026-09-27), the vendor's
  own route for sharpening a user's clip, is not in the repo
  (`audits/2026-10-02-ltx-2.5-audit.md`). Neither `upscale-clip` nor
  `refine-clip` is that route.
- #543's `refine-clip` (latent two-stage refine) is the companion route,
  recorded in `complete/ltx2-refine-clip-complete.md`.
