# Face-once cast reference sheet vs a portrait: probed, no gain (#1400)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-17). Don filed the idea on 2026-11-16 from KadoBOT/ComfyUI-H3-Continuity-Studio
(MIT, read at a2ddb97), whose `CHARACTER_SHEET` is a 16:9 six-column ivory sheet: one
head-only close-up in columns 1-2 and four neck-down body views (front, 90° profile, back,
relaxed) in columns 3-6, so "the face appears exactly once on the entire sheet". The ask
was to measure whether that layout holds a Ref2VA clip's face better than the single
portrait `templates/minimax/generated-subject-reference` draws today, and to adopt it in
the template and the cast steps of `dw:storyboard` and `dw:script-to-video` only if it did.
The plan comment on #1400 (v3) holds the full sweep. This file records what was measured
and what was withdrawn.

## Verdict: no gain. The portrait stays the cast reference

Stage A gave **no gain** on both halves of the criterion (S at least +0.05 SFace median
over P on both seeds, and no sheet layout in any S clip):

- **The sheet held the face worse.** S scored below P on both seeds: −0.123 and −0.074
  on the median, re-measured from the saved files.
- **Z-Image-Turbo never drew a face-once sheet.** S/1001 drew two head panels and a head
  on every body view (6 faces). S/1002 drew four full figures (4 faces). The plan counted
  that as a loss for S by itself.
- **No layout leak.** All four clips open on one man in the net loft. H3 took the person
  from the sheet and dropped its panels, which is the one encouraging result.

No template, prompt or skill wording changed, so there is nothing to reverse.

## Plan history

- **v1 (design, 2026-11-16).** A two-arm probe (P: today's 768x768 portrait; S: a
  1344x768 face-once sheet), then a stage B to adopt the sheet if it won.
- **v2 (2026-11-16, Don's ruling folded in).** Q1 "no": stage B was not filed up front.
  A's audit had to come back to Don as v3 whatever the numbers showed.
- **v3 (approved 2026-11-17 at its defaults).** Q1 [scope]: close the feature on A's
  record and withdraw B. Q2 [none]: file the in-job vs saved-file face-score gap as a
  separate `idea` (#1418).
- **Correction found by the spec:** the plan said "the same H3 prompt body", but the
  template's prompt also refers to "the setting of `<Picture 1>`" and its lighting. The
  probe wrote the setting in text for both arms, so the arms differ only in
  `subject_definitions`.

## What was built

### Stage A (#1406): face-hold probe (bench only)

Commit 2ab50bde, merged into `develop` as 346ea93f; the bounce fix 0caf73d0, merged as
925f7038. Docs and bench only, no deploy.

- **`bench/workflows/probe-1400-face-once-sheet.json`:** an inline copy of
  `generated-subject-reference` that changes only the draw prompt and size, plus a last
  `analyze_face_drift` step (16 samples) against the run's own draw. H3 at the template's
  defaults: Ref2VA turbo, 960x544, 124 frames, 9 steps.
- **Four H3 clips** in workspace `probe-1400` on lem, run one at a time after
  `downloads_required` came back `[]` for both arms: P/1001 `6ed0bcc0bde0`, S/1001
  `c4730ea6bade`, P/1002 `36fdea114df3`, S/1002 `4aefe1cea6a2`.
- **Record:** `docs/proposals/audits/2026-11-face-once-sheet.md`, and one "Probed and
  not a rule" bullet in `plugins/dw/skills/minimax-h3/references/per-character.md`
  ("No gain … Keep the portrait.").

SFace median / min, each clip against its own draw, from the saved files (re-measure jobs
`efc67c7f386a`, `4bf84a5c7ad9`, `eed1f09af2cb`, `9886f745e555`; the tester's independent
re-measure gave the same eight values):

| Seed | P (portrait) | S (sheet) | S − P (median) |
|---|---|---|---|
| 1001 | 0.889 / 0.847 | 0.766 / 0.708 | −0.123 |
| 1002 | 0.837 / 0.811 | 0.763 / 0.691 | −0.074 |

The scores taken inside the probe jobs (−0.102, −0.047) are superseded. They scored the
frames before the mp4 encode and the draw before its jpg save. The outcome is the same on
both.

**Bounces: 1.** M-F399: the in-job scores couldn't be reproduced from the saved files
within 0.01. The cause was the encode, not the task, and the audit now leads with
saved-file scores. Cases M-F396–M-F402 passed on the re-verify against `develop` @
925f7038.

## Cost and bounces

| Stage | Bounces | Cost |
|---|---|---|
| #1406 | 1 (M-F399, reproducibility) | Session cost not recorded on the stage. GPU: four H3 Ref2VA turbo clips plus four CPU-light re-measures on lem, one card |

## Deferred and open

- **Stage B: withdrawn.** The v2 sketch (a stored cast-sheet prompt, a template default
  change, skill wording) is dropped. A revival starts from the audit.
- **Known limits that keep this "probed, no gain", not "proved worse":** a different
  person per arm; two seeds against #1193's >0.1 seed swing; the S score is against the
  sheet's largest face, not always frontal; and the face-once sheet itself was never
  drawn, so this measured a multi-figure turnaround sheet against a portrait.
- **Proposed, not filed:** a composited face-once sheet (a portrait crop beside a
  headless body draw), the one form the audit says was never tested.
- **Filed:** #1418 (`idea`), the in-job vs saved-file face-score gap of up to 0.045,
  close to the 0.05 margins face-hold probes use.
- Out of scope throughout: wardrobe hold (#1398), location and prop sheets, the drawing
  model (#1317), `dialogue-short`'s cast, engine work.
