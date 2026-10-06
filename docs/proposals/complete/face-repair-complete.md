# `face-repair`: tracked, distance-gated LTX refine of a far face (#599)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-06). Plan v1 approved by Don 2026-10-05, with his Q1–Q4 answers
matching the plan's defaults (no new version). Three stages, #622, #623 and
#624, all verified 2026-10-06. Stage 3 builds on
`ltx2-refine-in-place-complete.md` (#606), whose `LTX2RefinePipeline` and
`select`-by-index ladder pattern it reuses.

The concept came from vrgamegirl19/comfyui-vrgamedevgirl
(`VRGDG_StandaloneFaceFixNodes.py`, `VRGDG_FaceFix.py`,
`scripts/far_face_repair_backend.py`), whose source-available license is
incompatible with Apache-2.0. Everything here was re-implemented from the
concept. No code, prompt text or presets were taken: the 7%/9% gate and
their sigma ladders were reference only, and ours were tuned on lem.

## The report

`dw/tasks/restore_faces.py` (GFPGAN/CodeFormer) runs one frame at a time
through `_per_frame` with no tracking, so faces flicker on video, and the
faces generators break most are the small ones. The idea: detect (full
frame plus tiles, optional rotation passes, NMS), track one face with EMA
and gap hold, reset at cuts, gate by face size, crop to a padded square,
refine the crop sequence with LTX v2v at low sigmas, and paste back with a
feathered, strength-scaled, colour-matched alpha.

## Verdict

**Build smaller,** with the LTX stage sequenced after #606. Demand evidence
was thin: field report #484 saw smeared crowd faces on H3 at 960×544 and
fixed them by re-rendering at native size. The case rested on Don's
priority:3 approval ("big lift, high benefit").

Cut from v1, each with its trigger:
- **±15°/±30° rotation detection passes.** Back if a lem run shows tilted
  far faces being missed.
- **Landmark-aligned RANSAC affine paste.** Back if the box paste shows
  visible swim on lem.
- **Multi-face tracking.** Deferred by Don. Back after v1 ships and a clip
  needs it.

## What the plan found against the issue

- **It is not one task.** Loading the LTX-2.5 stack inside a task would
  bypass step-level quantization, offload and `release_pipeline`. So it is
  a template: a crop task, an LTX pipeline step, a paste task, with the
  track passed as `previous_result:crop.track`. No syntax or engine change.
- **No same-size low-sigma LTX v2v existed.** `refine-clip` always
  upscales 2×. #606 added the same-size route; stage 3 waited on it (the
  256² crop through `refine-clip` fallback was not needed).
- **The detector.** Neither res10 SSD nor YuNet existed in dw, and the only
  detector (facexlib's RetinaFace in `restore_faces`) fetches its weights
  from GitHub outside dw's Hub-only download rules. YuNet through
  `cv2.FaceDetectorYN`, its ONNX via the existing Hub validators.
- **Shot cuts.** `AudioVideo.shots` already records cuts for generated
  clips, so the tracker resets there first, with a content detector only
  for clips with no recorded shots.
- **Name trap.** The input is `clip`, not `video`: an argument named
  `video` is auto-loaded as bare frames and drops the audio.

## Decisions (Don, 2026-10-05)

- **Q1:** YuNet ONNX via `hf_hub_download` from a repo named by
  `detector_repo`.
- **Q2:** no rotation passes, no landmark-affine paste in v1.
- **Q3:** `gate_full` / `gate_zero` as arguments and template variables,
  chosen on lem.
- **Q4:** stages 1–2 build at priority:3; stage 3 waits on #606.
- Follow-ups filed: #615 (`restore_faces` reuses one `upsample_img` across
  a video's frames) and #616 (facexlib's GitHub weight fetch).

## What was built

**Stage 1, #622: `crop_face_track`** (`dw/tasks/face_track.py`).
- Arguments: `clip`, `crop_size` (512, a multiple of 32), `padding` (0.6),
  `gate_full` (0.06), `gate_zero` (0.12), `min_confidence` (0.6),
  `detector_repo`, `detector_file`, `device`. Domains for every numeric
  argument and the cross-argument rules (`face_track_problems` /
  `face_track_errors`, `check_face_detector_source`) in
  `dw/task_domains.py`.
- Detection: YuNet on the full frame plus 4 overlapping corner tiles
  enlarged 2×, merged with `cv2.dnn.NMSBoxes`.
- Track: one track matched by IoU, confidence and centre distance,
  EMA-smoothed; a miss of up to 6 frames holds the box at strength ×0.7 per
  frame, then resets as `lost`. Resets at recorded `shots` (reason `shot`),
  else at a content cut (reason `cut`, `content_cuts`).
- Strength: face width / frame width, 1 at or below `gate_full`, ramping to
  0 at `gate_zero`.
- Crops: padded square, resized to `crop_size`², mirror-padded to 8n+1
  through `np.pad(..., mode="reflect")` indices.
- Returns `{crops, track}`; `track` (per-frame box, crop, strength, state,
  pad counts, source size, resets, `face_found`) is saved as a JSON record
  (`JsonRecord` in `dw/media_types.py`, `Result.save_artifact`, a new
  *JSON records beside media* map row). No face is a warning, not an error.

**Stage 2, #623: `paste_face_track`** (same module).
- `paste_face_track(clip, repaired, track, feather=0.3, color_match=true)`:
  drops pad frames, resizes each crop back to its square (clipped at frame
  edges), blends with a radial smoothstep feather × strength, and with
  `color_match` shifts the crop's mean colour onto the source inside the
  mask. A strength-0 frame is the source frame itself.
- Carries the source's audio, sample rate, fps and shots
  (`EXPECTED_SITES` gains `paste_face_track: carries`).
- Refuses a track from a clip of another frame count or size, a `repaired`
  shorter than `track.crop_frames`, a non-track record, and `feather` > 1
  (also at validate).

**Stage 3, #624: `templates/ltx2/face-repair`.**
- Steps: `ladder` (`select`, `index: variable:strength`) → `crop`
  (`crop_face_track`) → `refine` (`LTX2RefinePipeline` at `crop_size`,
  `intermediate/`) → `pasted` (`paste_face_track`, `final/`).
- Shape `shot`, traits `needs-input-media` and `has-audio`; cost 2.0 min
  on an RTX 3090 (measured 1.2–2.3). `crop_size` multiple-of-32
  `variable_constraints`.
- Docs: `workflows/templates/ltx2/README.md` row, a line in the `ltx-2.5`
  skill's "Repairing the user's footage" bullet, TASKS.md examples;
  `COMPACT_BUDGET` raised to 10,000 (measured 9,981).

**The tuning** (lem; job ids in #624's hand-off and its commit message):

| strength | sigmas (`noise_scale` = first) |
|---|---|
| 0 | [0.65, 0.45, 0.2] |
| 1 | [0.75, 0.55, 0.3] |
| 2 (default) | [0.85, 0.65, 0.4] |
| 3 | [0.9, 0.72, 0.45] |
| 4 | [0.95, 0.8, 0.5] |

Template defaults: `padding` 1.5, `gate_full` 0.03, `gate_zero` 0.06,
`feather` 0.3. A face crop tolerates higher sigmas than #606's whole-frame
ladders, which top out at 0.85. The tasks' own defaults (0.6, 0.06/0.12)
are unchanged; TASKS.md points at the template's values.

**Verified on lem (C-F202/C-F203):** a ~22 px far face came back clearly
sharper with the same identity and no flicker or seam; a near face and a
no-face clip came back indistinguishable from the source; frames, fps and
soundtrack kept in every case.

## Deviations from the plan

- The refine is `LTX2RefinePipeline` directly, not a `refine-in-place`
  call: `refine-in-place` refuses a silent source, and the crops are silent.
- A small `ladder` select step precedes the plan's three steps.
- `strength` is an integer index 0–4, the same knob as `refine-in-place`.
- Content-cut detection grew a second rule (below).

## Bounces per stage

- **#622: two.** An architecture bounce (hand-written NMS and mirror
  padding where `cv2.dnn.NMSBoxes` and `np.pad` already do it; map rows
  owed for the new domains and for JSON records). A tester bounce on
  C-F195's histogram arm: two letterboxed framings of one portrait had HSV
  correlation 0.999, so no cut was recorded and the old box was held 6
  frames over the new shot. Fixed by `content_cuts`: a cut is also a
  32×18-thumbnail mean difference ≥ 0.06 and ≥ 4× the median of the 8
  frames each side. Verified on the third hand-off. Separately, harnest#54
  rebuilt C-F192's NEAR fixture, whose face was under the gate.
- **#623: none.** Verified on the first hand-off.
- **#624: none.** Verified on the first hand-off.

No `usage:` figures were recorded on the stages, so cost is left out. The
plan's estimate was $5–6, $4 and $5–7, plus lem GPU time for the tuning.

## Deferred and left open

- **Rotation passes, landmark-affine paste, multi-face tracking:** cut
  from v1, triggers above.
- **#615 and #616**, the `restore_faces` follow-ups, are their own issues.
- **Side observation from #624's verify:** `pair_audio` `fit:"video"`
  produced 7.45 s of audio over 2.08 s of video with no trim and no
  `audio_trimmed_to_video` warning. Not a face-repair finding; face-repair
  preserved it exactly.
