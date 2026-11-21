# H3 face repair with a cast reference (#1320)

Design: the plan comment on #1320 (v1). This file records the stage A probe
(#1347).

**Outcome: the gate failed (Don, 2026-11-14).** Stage B, a
`templates/minimax/face-repair-ref` sibling template, is not filed, and #1320
is closed as not planned. See *Don's ruling* below.

## Stage A probe: ref2va 0.8 / 0.9 vs t2va 0.6 (2026-11-14)

_Run by claude-opus-5-5 (anthropic) on lem, workspace `probe-1320`._

**Server:** lem was running a hand-started server on `595cee1f` (0.20.0 code,
reporting 0.20.0-alpha.1) when these jobs ran (see #1334), not the `develop`
deploy.

**Probe workflow:** `probe-1320-ref2va-face` (saved in `probe-1320`). This is
`templates/minimax/face-repair`'s crop, refine and paste chain with:
- `"workflow": "ref2va"`;
- `references: [{from_file: variable:cast_picture}]` with `asset:cast/pat.jpg`
  as `<Picture 1>`;
- the ref2v turbo 8-step LoRA, 9 steps, `video_shift` 12;
- two refine arms in one job: `refine80` / `pasted80` at `refine_strength` 0.8,
  and `refine90` / `pasted90` at 0.9.

Common settings: seed 1092, gate 0.2/0.3, crop 512, padding 1.5, feather 0.3,
int4 weights, sparse attention.

### Takes

| Take | Face width / frame | Source |
|---|---|---|
| Synthetic | ~22 px of 960, **~0.023** | Job `823039a89754`: `reference-to-video` extreme wide shot of Pat in a computer room, 124 frames, 437.7 s. No take of Don's under 0.05 was found, so this was made under the plan's Q5 default. |
| #1097 control | ~0.087 | `KnownIssuesRestorePointHD_tag/20261013-015215-a0997eba/intermediate/KnownIssuesRestorePointHD_tag-shot@tag.0-0.0.mp4` |

### Arms

The `refine_strength` column is the strength each refine pass ran at.

| Take | Arm | Job | Wall | Refine pass (load / encode / denoise / decode, s) | Face SSIM, crop vs crops | Face SSIM, pasted vs source |
|---|---|---|---|---|---|---|
| Synthetic | t2va 0.6 (shipped) | `dc591186acc3` | 118.7 s (job) | 4.5 / 34.3 / 41.4 / 13.5 | 0.882 (26.7 dB) | 0.889 (25.2 dB), 56 px box |
| Synthetic | ref2va 0.8 | `101ec03104c6` | ~160 s (arm) | 4.6 / 38.0 / 103.6 / 13.3 | 0.759 (21.2 dB) | 0.740 (20.6 dB) |
| Synthetic | ref2va 0.9 | `101ec03104c6` | ~154 s (arm) | cached / 36.6 / 104.1 / 12.8 | 0.596 (16.0 dB) | 0.354 (14.8 dB) |
| #1097 | t2va 0.6 (reused) | `38da34a526e1` | 126.9 s (job) | - | 0.813 (23.9 dB) | - |
| #1097 | ref2va 0.8 | `e4e933dac464` | ~226 s (arm, cold load) | 54.4 / 51.4 / 106.1 / 14.5 | 0.670 (19.8 dB) | - |
| #1097 | ref2va 0.9 | `e4e933dac464` | ~157 s (arm) | cached / 37.5 / 105.0 / 14.6 | 0.468 (14.3 dB) | - |

- The ref2va job `101ec03104c6` ran both arms in 339.2 s.
- The ref2va job `e4e933dac464` ran both arms in 420.8 s.
- Per arm, ref2va's denoise is about 2.5x t2va's (9 steps against 5). A warm
  ref2va arm costs about 2.6 min, against about 2 min for the whole t2va job.
- SSIM is for the record only. A lower SSIM is expected when the face is
  actually rebuilt.

Run dirs:
- synthetic t2va: `templates/minimax/face-repair/20261114-153740-a83c147a`;
- synthetic ref2va: `probe-1320-ref2va-face/20261114-153741-cca3acea`;
- #1097 ref2va: `probe-1320-ref2va-face/20261114-120119-5e949930`;
- #1097 t2va: `MiniMaxH3FaceRepair/20261031-212103-f99bfd5b`.

### Face-grid verdicts

Judged with `compare_outputs` on the crops (frames 10/50/90/120) and with
`get_output_frames` on the pasted frames, cropped `[440,160,96,96]` around the
face.

- **Synthetic, t2va 0.6:** keeps the smear. In the pasted frames the face is a
  blur with a dark open-mouth blob. It is not recognisably Pat.
- **Synthetic, ref2va 0.8:** a recognisable Pat: the moustache, the brow, and
  the face shape. The face crop showed no visible seam at the paste. Inside
  the crop, the body is slightly smaller and the cabinets have moved, and the
  probe judged that the feathered paste hid this at the face's scale. **Don's
  full-frame viewing overturned this:** the area around the face is visibly
  distorted (see *Don's ruling*).
- **Synthetic, ref2va 0.9:** also recognisable, but the scene inside the crop is
  restaged (new background and framing). In the pasted frames the face reads
  larger, with a different head pose than the body carries. This is the pop the
  plan's risk named.
- **#1097, t2va 0.6:** faithful, as recorded in #1092.
- **#1097, ref2va 0.8:** close to the source, with mild drift.
- **#1097, ref2va 0.9:** restages the crop: new background and a reframed body.
  It would pop at the paste.

Not checked: mouth sync was not judged by ear against the audio. The frames
show the mouth moving in every arm.

### Proposed gate call (superseded)

The probe session proposed **pass, at `refine_strength` 0.8**: on the
sub-0.05 face, ref2va 0.8 built a recognisable cast face where t2va 0.6 left a
smear, and the face crops showed no distortion at the paste. 0.9 restaged on
both takes. Caveats it named: the sub-0.05 take is synthetic, mouth sync was
judged by eye only, and the runs were on the 595cee1f server. Don's ruling
below replaces this proposal.

### Don's ruling: the gate fails (2026-11-14)

Don watched all five pasted outputs in full frame: the synthetic
take at t2va 0.6, ref2va 0.8 and ref2va 0.9, and the #1097 control at ref2va
0.8 and 0.9.

- **Every ref2va arm, at both strengths and on both takes, shows visible
  distortion in the area around the corrected face.**
- The shipped t2va 0.6 output was not flagged.
- This contradicts the probe's "no seam at the paste" for ref2va 0.8. The
  96 px face crops the grids used do not show the surrounding area, and the
  surround is where the damage is.

The plan's gate requires "no frame distortion", so it fails. Stage B is not
filed, and #1320 closes as not planned with this record.

**Lesson for any future face-pass probe:** judge the paste surround, not only
the face crop. Look at full frames (or a crop several times the face box) of
the pasted output, at the same moments as the face grids. This is the same
lesson as the levelled handoff: the measures and the crops passed, and the
viewing did not.

**Reopen path:** a ref2va face pass that does not disturb the surround, shown
on a full-frame viewing. The per-frame strength work deferred in the plan
stays deferred behind its own trigger.
