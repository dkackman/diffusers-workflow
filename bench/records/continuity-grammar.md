# Continuity grammar probes (#1685 stage 1, #1704) - PRELIMINARY

**PRELIMINARY.** The marker reads below are the lead's own reads of the frames, so they stand only until Don has viewed the outputs. A rule counts as **confirmed** only when both seeds show the effect (plan v1). A read the lead could not settle is recorded here as unsettled and goes to Don as a `taste` question.

Lead: claude-opus-5-5 (anthropic), 2026-11-26. lem was at develop @ ca1ef1e6.

## Setup

- **Question.** Does naming a side, a facing, a light side or a screen direction in the prompt hold that thing across the cards or shots of one scene, where the same prompts without it do not?
- **Rules under test** (plan v1):
  - **R1:** the 180-degree rule. Two people keep their frame sides across a two-shot and their singles.
  - **R2:** matched eyelines. Each single looks toward where the partner sits.
  - **R3:** screen direction. A walker keeps one direction of travel across a cut.
  - **R5:** light side. The key light stays on one side of the face across a wide, a medium and a close-up.
- **Workflows.** All are in `bench/workflows/` and saved in workspace `probe-1685`. Each step's prompt is a variable, and the job's one seed is `variable:seed`.
  - `continuity-side.json` (P-A, R1 and R2) runs FLUX 3 Klein T2I. Its steps are `two` (a two-shot of June and Marcus), `a` (a single of June) and `b` (a single of Marcus).
  - `continuity-light.json` (P-B, R5) runs Klein T2I. Its steps are `wide`, `medium` and `close`, all of June reading in one living room.
  - `continuity-direction.json` (P-C, R3) runs MiniMax-H3 t2va. Its steps are `shot1` (a static wide of Theo walking a street) and `shot2` (a static medium).
- **Settings:**
  - The Klein steps use the catalog's `templates/flux3/text-to-image` settings: 16 steps at 1344x768.
  - The H3 steps use `templates/minimax/video-with-audio-768p` unchanged: the 4-step turbo LoRA v1.2, int4 SDNQ, sparse attention, 5 steps, shift 6/3, 1344x768 and 124 frames.
- **Arms.** The workflow defaults are the **control**: names only, with no side, facing, light side or direction. The **treatment** passes the prompts in `bench/records/continuity-grammar-arms.json`, which differ from the control only as follows:
  - **P-A, two-shot:** adds "June at frame left, facing right; Marcus at frame right, facing left."
  - **P-A, single `a`:** adds "June at frame left, facing right; Marcus, off screen, at frame right, facing left."
  - **P-A, single `b`:** adds "Marcus at frame right, facing left; June, off screen, at frame left, facing right."
  - **P-B:** replaces "Warm practical light." with "Key light from frame left, warm practical."
  - **P-C:** replaces "walks along the street" with "walks from frame left to frame right".
- **Seeds:** 1685 and 1686 for every arm. Each job's `note` names its probe, arm and seed.

## Runs

Every job succeeded on lem's cuda:0 (an RTX 3090). Output paths are relative to workspace `probe-1685`.

| Probe | Arm | Seed | Job | Ran | Output folder |
|---|---|---|---|---|---|
| P-A | control | 1685 | `01bc26f4b8c7` | ~24 s | `continuity-side/20261126-075851-9e7d2e7e/final/` |
| P-A | control | 1686 | `47784abf5969` | ~24 s | `continuity-side/20261126-080108-1e1be8d0/final/` |
| P-A | treatment | 1685 | `f82e48f25759` | ~24 s | `continuity-side/20261126-080132-c27a1dbf/final/` |
| P-A | treatment | 1686 | `32aecd104095` | ~24 s | `continuity-side/20261126-080156-63ddea18/final/` |
| P-B | control | 1685 | `eb69644d79e6` | ~24 s | `continuity-light/20261126-080221-4656f234/final/` |
| P-B | control | 1686 | `e1268b23047e` | ~24 s | `continuity-light/20261126-080245-54a6af65/final/` |
| P-B | treatment | 1685 | `4ae49d950d1b` | ~24 s | `continuity-light/20261126-080309-8c221ef5/final/` |
| P-B | treatment | 1686 | `2619ab70f376` | ~24 s | `continuity-light/20261126-080333-7427c778/final/` |
| P-C | control | 1685 | `7bb47e82a72f` | ~332 s (incl. load) | `continuity-direction/20261126-080359-5fe7dc77/final/` |
| P-C | control | 1686 | `0f412617cacb` | ~260 s | `continuity-direction/20261126-080929-8fcd7a05/final/` |
| P-C | treatment | 1685 | `f636fe850808` | ~264 s | `continuity-direction/20261126-081349-b556e884/final/` |
| P-C | treatment | 1686 | `405ccc8e2fcc` | ~259 s | `continuity-direction/20261126-081812-d59c2ea9/final/` |

The file names inside each folder:
- P-A: `ContinuitySide-two.0-0.0.jpg`, `ContinuitySide-a.1-0.0.jpg` and `ContinuitySide-b.2-0.0.jpg`.
- P-B: `ContinuityLight-wide.0-0.0.jpg`, `ContinuityLight-medium.1-0.0.jpg` and `ContinuityLight-close.2-0.0.jpg`.
- P-C: `ContinuityDirection-shot1.0-0.0.mp4` and `ContinuityDirection-shot2.1-0.0.mp4`.

## Marker reads (PRELIMINARY)

The reads are the lead's. They were taken from `compare_outputs` sheets of each job's cards and from `get_output_frames` four-frame sheets (at 0, 1.7, 3.4 and 5.1 s) of each clip. compare_outputs' PSNR and SSIM between cards (10-16 dB, SSIM 0.25-0.65) only say that the cards are different framings. They carry no reading of side or facing, so the reads below are by eye.

### P-A: sides and eyelines (R1, R2)

| Arm | Seed | Two-shot | Single `a` (June) | Single `b` (Marcus) | Sides hold? | Eyelines match? |
|---|---|---|---|---|---|---|
| control | 1685 | June left, facing right; Marcus right, facing left | June centre-right, looking **frame left**, with an over-the-shoulder edge at left | Marcus centre, looking **frame left**, with an over-the-shoulder edge at left | ✗ June is on the wrong side of the line | ✗ June looks left, away from where Marcus sits |
| control | 1686 | June left, facing right; Marcus right, facing left | June centre, looking off toward frame right | Marcus centre, looking slightly frame right, with a shoulder edge at right | ✗ Marcus looks right, toward his own side | ✗ / unclear |
| treatment | 1685 | June left, facing right; Marcus right, facing left | June frame left, facing right; Marcus over the shoulder at right | June over the shoulder at left; Marcus frame right, facing left | ✓ | ✓ |
| treatment | 1686 | June left, facing right; Marcus right, facing left | June frame left, facing right; Marcus over the shoulder at right | June over the shoulder at left; Marcus frame right, facing left | ✓ | ✓ |

- **R1 is confirmed** for Klein: the treatment holds both seeds, while the control's singles cross the line in both.
- **R2 is confirmed** for Klein, by the same reads.
- The control two-shots already put June at left, because she is named first. The singles are where the control fails.
- In both seeds, naming the off-screen partner's side turned the singles into **over-the-shoulder** shots, with the partner's shoulder or the back of their head in frame, rather than clean singles. That is a framing change the prompt did not ask for. It is still a usable coverage shot, but stage 2's skill text should name it.

### P-B: light side (R5)

The marker is which side of June's face is brighter, plus where the visible practical lamp sits.

| Arm | Seed | Wide | Medium | Close |
|---|---|---|---|---|
| control | 1685 | face fairly even; lamp out of frame | lamp at frame right | lamp at frame right; face even, slightly brighter on frame right |
| control | 1686 | lamp at frame right | lamp at frame left, behind the chair | lamp at frame left in the background; face even |
| treatment | 1685 | face even; no lamp visible | lamp at frame **left** | lamp at frame **left**; face brighter on frame left (forehead, cheek), moderately |
| treatment | 1686 | lamp at frame **right** | lamp at frame **left** | lamp at frame **left**; face brighter on frame left, moderately |

- **R5 is unsettled.** The treatment moves the visible practical to frame left in the medium and the close of both seeds, where the control wanders between left and right.
- But seed 1686's treatment wide still shows a lamp at frame right.
- And the face's key side is only moderately one-sided. At this lighting (warm, low-contrast, "practical") it is a soft gradient, not a clear key.
- Whether the treatment's close-ups *read* as lit from frame left is a perception call: see Q1 below.

### P-C: screen direction (R3)

The marker is Theo's horizontal position at 0, 1.7, 3.4 and 5.1 s in each shot, as a fraction of the frame width.

| Arm | Seed | shot1 (wide) | shot2 (medium) | Direction holds across the cut? |
|---|---|---|---|---|
| control | 1685 | walks **toward the camera** down the street's centre (x ≈ 0.33 → 0.62, mostly growing in size) | walks toward the camera (x ≈ 0.3 → 0.7, growing to a medium close-up) | - no lateral direction in either shot |
| control | 1686 | walks toward the camera along the left pavement | walks toward the camera; the frame grows to a close-up, a push-in despite "The camera does not move" | - no lateral direction in either shot |
| treatment | 1685 | enters at left: x ≈ 0.13 → 0.45 → 0.85, **left to right** | x ≈ 0 → 0.18 → 0.6 → exits right, **left to right** | ✓ |
| treatment | 1686 | x ≈ 0.03 → 0.18 → 0.46 → 0.98, **left to right** | x ≈ 0 → 0.1 → 0.33 → exits right, **left to right** | ✓ |

- **R3 is confirmed** for H3: the treatment gives left-to-right travel in both shots of both seeds, so the direction holds across the cut.
- Without a named direction, H3 chose an axial walk toward the camera in all four control shots. The control does not *break* screen direction; it establishes none, which a cut can't hold either.
- In the treatment, Theo faces three-quarters toward the camera while travelling right, rather than in pure profile. That reads as a natural walk-by.

### Verdicts

| Rule | Verdict | Model | Basis |
|---|---|---|---|
| R1 180-degree rule | **confirmed** | Klein | treatment ✓ on both seeds; control crosses the line on both |
| R2 matched eyelines | **confirmed** | Klein | treatment ✓ on both seeds; control looks away on both |
| R3 screen direction | **confirmed** | H3 | treatment left to right in both shots on both seeds; control has no lateral direction |
| R5 light side | **unsettled** | Klein | the practical moves to frame left in 4 of the treatment's 6 cards; the face key is soft; Q1 below |

## Deviations from plan

1. **The treatment singles name the off-screen partner's side** as well as the subject's own side and facing, because a single's eyeline only has a target if the partner's side is said. A side effect: Klein drew the partner as an over-the-shoulder edge (P-A, above).
2. **The H3 template's turbo LoRA is listed as a download.** `validate_workflow` puts `lightx2v/Minimax-h3-Turbo` (1.29 GB) under `downloads_required` for `continuity-direction.json`, and it does the same for the catalog template `minimax/video-with-audio-768p` itself. That LoRA is the one the template already runs on lem, not a new model, and it was acknowledged in each run's `acknowledged_cost`. Nothing new was downloaded.
3. **The workflows ran from workspace `probe-1685`** (`save_workflow`), not from a deploy. Nothing was deployed.
4. **H3 jobs ran faster than estimated:** ~4.4 min each after the first, not ~5.5 min.
5. **The direction workflow does not validate on an MPS dev machine,** because of a local VRAM ceiling (a projected 19.7 GB against 17.8 GB). It validates clean on lem's 24 GB cards.

## lem time

- Klein: 8 jobs × ~24 s ≈ 3.5 min.
- H3: 4 jobs, from 08:03:59 to 08:22:31 ≈ 18.6 min.
- **Total ≈ 22 min**, inside the approved ~60 min.

## For Don to view

- **Q1 [taste]: does the P-B treatment read as lit from frame left?**
  - Compare the treatment close-ups `continuity-light/20261126-080309-8c221ef5/final/ContinuityLight-close.2-0.0.jpg` (seed 1685) and `continuity-light/20261126-080333-7427c778/final/ContinuityLight-close.2-0.0.jpg` (seed 1686) with the control's `continuity-light/20261126-080221-4656f234/final/ContinuityLight-close.2-0.0.jpg`.
  - **Default if he approves:** R5 stays unsettled. Stage 2's skill text states the light side as a *hint* that moves the practical, not as a rule that holds the key.
  - **If he sees a clear frame-left key:** R5 is confirmed for Klein, and the skill states it as a rule.
- **P-A:** the treatment's singles in `continuity-side/20261126-080132-c27a1dbf/final/` and `continuity-side/20261126-080156-63ddea18/final/`, to confirm the over-the-shoulder framing is acceptable coverage.
- **P-C:** `continuity-direction/20261126-081349-b556e884/final/ContinuityDirection-shot1.0-0.0.mp4` then `continuity-direction/20261126-081349-b556e884/final/ContinuityDirection-shot2.1-0.0.mp4`, played back to back as a cut.
- **Audio:** the run reported near-silent audio (≈ -55 dBFS mean) on every P-C shot. The workflow is the template's t2va, and the sound line was in the prompt. That is outside this probe's question, but it is noted here as observed.
