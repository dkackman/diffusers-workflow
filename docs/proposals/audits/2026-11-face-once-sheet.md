# Face-once reference sheet vs a portrait for H3 Ref2VA (#1400 stage A, #1406)

Probe date: 2026-11-17. Server `lem` (RTX 3090), `develop` @ 9d875ee4. Agent: claude-opus-5-5
(anthropic). Workspace `probe-1400`. No engine file was changed. `downloads_required` was empty
for both arms before anything ran, and the queue was empty; the four jobs ran one at a time.

## Question

Does a generated reference that shows the face once (a large head panel) beside the body and
clothes (neck-down panels) hold a speaker's identity better in an H3 clip than the single
portrait `templates/minimax/generated-subject-reference` draws today?

## Configuration

Workflow: `bench/workflows/probe-1400-face-once-sheet.json`, an inline copy of
`templates/minimax/generated-subject-reference` that changes only the draw step's prompt and
size (`subject_prompt`, `draw_width`, `draw_height`), and adds a last step,
`analyze_face_drift` (16 samples), that scores the clip against that run's own draw. The H3
step is the template's at its defaults: Ref2VA turbo, 960x544, 124 frames, 9 steps.

- **Arm P**: a 768x768 chest-up portrait of the fisherman (grey close-cropped hair, short grey
  beard, bare-headed, navy cable-knit jumper under an open yellow oilskin) in a net loft.
- **Arm S**: a 1344x768 sheet on an ivory background asking for one head-only close-up in
  columns one and two, and four neck-down body views (front, profile, back, relaxed) in columns
  three to six, "the face appears only once on the sheet". Its `subject_definitions` name
  `<Picture 1>` as a reference sheet whose left panel is the face and right panels the body and
  clothes, and say the layout, ivory background and panels are not reused.

Both arms share one H3 body: the same net-loft setting written in text, the same line ("The tide
never asks permission. You learn to be ready when it turns."), and the same camera and sound.
The S and P prompts differ only in `subject_definitions`. Seeds 1001 and 1002; the seed drives
both the draw and the clip.

## Jobs

| Arm | Seed | Job | Run id | Note |
| --- | --- | --- | --- | --- |
| P | 1001 | `6ed0bcc0bde0` | `20261117-065924-19fcbd82` | #1400 stage A: arm P (768x768 portrait), seed 1001 |
| S | 1001 | `c4730ea6bade` | `20261117-070350-44908187` | #1400 stage A: arm S (1344x768 face-once sheet), seed 1001 |
| P | 1002 | `36fdea114df3` | `20261117-070859-265074ba` | #1400 stage A: arm P (768x768 portrait), seed 1002 |
| S | 1002 | `4aefe1cea6a2` | `20261117-071257-51c48b5a` | #1400 stage A: arm S (1344x768 face-once sheet), seed 1002 |

Files are under `probe-1400-face-once-sheet/<run id>/`: the draw at
`intermediate/BenchProbe1400FaceOnceSheet-draw_subject.0-0.0.jpg`, the clip at
`final/BenchProbe1400FaceOnceSheet-reference_to_video_audio.1-0.0.mp4`, and the score at
`intermediate/face-0.json`.

## The draws

Neither S draw is face-once, so under the plan both are losses for S before any score:

- **S/1001**: two identical head-only close-ups at the top of the sheet (not one), and a head on
  every body view: five figures. `analyze_face_drift` found 6 faces in it.
- **S/1002**: no separate head panel at all. Four full-length figures (front, three-quarter,
  profile, front), each with a head. It found 4 faces.

Z-Image-Turbo ignored both "neck down only" and "the face appears only once". The plan allowed
no prompt tuning, so this was not retried. Both P draws are single chest-up portraits as asked.

## Scores

SFace cosine, the clip's faces against the arm's own draw. With several faces in the
reference, `analyze_face_drift` uses the largest one and says so in `notes` (both S runs did).
Every sampled frame held a face (16/16 in every clip), and no shot boundary was found.

| Seed | P median | P min | S median | S min | S − P (median) |
| --- | --- | --- | --- | --- | --- |
| 1001 | 0.883 | 0.825 | 0.781 | 0.728 | −0.102 |
| 1002 | 0.823 | 0.793 | 0.776 | 0.736 | −0.047 |

## First-frame leak (frames 0 and 12, and 60 for S)

| Clip | Verdict |
| --- | --- |
| P/1001 | No leak: one man, chest-up, in a net loft from frame 0. |
| P/1002 | No leak: one man, chest-up, in a net loft from frame 0. |
| S/1001 | No leak: one man, chest-up, in the dark net loft from frame 0; no panel, ivory background or second figure. |
| S/1002 | No leak: one man, chest-up, in the net loft from frame 0; no panel, ivory background or second figure. |

H3 took the person from the sheet and dropped its layout on both seeds, with the
`subject_definitions` wording above.

## Outcome

**No gain.** The criterion asked for S at least 0.05 above P on both seeds, and no sheet layout
in any S clip. The leak half holds, but S scored below P on both seeds (−0.102 and −0.047), and
neither S draw was face-once, which the plan counts as a loss for S on its own.

## Known limits

- Each arm draws a different person (a different draw on every arm and seed), so a score is
  each clip against its own reference, not one identity measured two ways. The comparison is
  "how well does H3 hold the reference it was given", not "which picture gives the same man".
- #1193 saw the seed alone swing similarity by more than 0.1. With two seeds per arm, the seed
  1002 gap (−0.047) sits inside that swing. The seed 1001 gap (−0.102) is at its edge.
- The S scores are against the sheet's largest face, which is not always frontal. On S/1002
  it may be the profile head, and a frontal clip scores lower against a profile. This counts
  against S, but that is part of what a sheet gives H3 too.
- The sheet asked for was never drawn, so this measured "a multi-figure turnaround sheet vs a
  portrait", not the face-once design. A face-once sheet built another way (e.g. composited
  from a portrait crop and a headless body draw) was not tested.
