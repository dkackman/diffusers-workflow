# Place card probe (#1577 stage A, #1581) - PRELIMINARY

**PRELIMINARY.** The marker reads below are the lead's frame reads. They stand as preliminary until Don has viewed the clips (plan v1, Don's ruling).

Lead: claude-opus-5-5 (anthropic), 2026-11-22. lem was at develop @ 7b5906ff.

## Setup

- **Question.** Does a still of the empty set (the *place card* K), given to MiniMax-H3 ref2va after the people, carry the set's markers into a shot that is not framed like the card?
- **Control: #1398's B0** (`fdba91e57259`). It was not re-rendered. B0 lost E's markers:
  - its lamp was white and brass, not red glass;
  - its window was an orange sunset, not blue evening;
  - its coat was purple, not green.
- **E's markers** (from #1398): a red glass lamp at the centre of the desk, a tall wooden bookshelf on the left wall, a tall window on the right with a blue evening sky, and a green coat.
- **K's markers:** lamp, shelf and window. The coat is not in K: the card is an empty set, so the coat is judged against E only.
- **Workflows** (all in workspace `probe-1577`):
  - `bench/workflows/place-card-shot.json`: `subject` then `card`. Runs R1 and R2.
  - `place-card-two.json`: `subject`, `subject2`, `card`. Runs R4.
  - `place-card-insert.json`: `card` only. Runs N1.
- **Settings:** the same as `visual-memory-shot.json`: turbo LoRA, int4 SDNQ, sparse attention, 9 steps, shift 12/3, 1344x768, 124 frames (R4 175), seed 1398. No guides, no anchor and no window in any arm.
- **The card** is the asset `asset:places/1577-study.jpg`, from job `5d47505f9762`.

## Runs

| Arm | Job | Card | Ran | Output |
|---|---|---|---|---|
| K | `5d47505f9762` | - | Klein T2I, seed 42, 1344x768, ~29 s | `templates/flux3/text-to-image/20261122-193111-94b878f8/final/Flux3KleinT2I-text_to_image.0-0.0.jpg` |
| K-night | `e5ba385f1e51` | - | Klein T2I, seed 42, 17 s | `templates/flux3/text-to-image/20261122-201042-f2e9d852/final/Flux3KleinT2I-text_to_image.0-0.0.jpg` |
| K-rev | `15077fa2496a` | - | Klein image-edit of K, ~50 s | `templates/flux3/image-edit/20261122-201057-62b41217/final/Flux3KleinImageEdit-edit.1-0.0.jpg` |
| R0 | `fdba91e57259` (#1398 B0) | none | not re-rendered | `probe-1398` `visual-memory-shot/20261117-044359-71e32755/final/VisualMemoryShot-shot.0-0.0.mp4` |
| R1 | `a80f0b1ec000` | K | cuda:1, denoise 601 s, ~684 s total | `probe-1577/place-card-shot/20261122-193350-abceaf22/final/PlaceCardShot-shot.0-0.0.mp4` |
| R2 | `1b58fcff664b` | K | cuda:1, ~675 s | `probe-1577/place-card-shot/20261122-194512-cf8e5ed7/final/PlaceCardShot-shot.0-0.0.mp4` |
| R3 | - | - | **not run** (see below) | - |
| R4 | `c2af95984127` | K | cuda:0, 175 frames, ~1395 s; the first pass took 509 s | `probe-1577/place-card-two/20261122-195238-46cf6bb6/final/PlaceCardTwo-shot.0-0.0.mp4` |
| N1 | `16f0a5e071f9` | K only | cuda:1, ~694 s, including a 295 s first-pass warm-up | `probe-1577/place-card-insert/20261122-195628-f79483b0/final/PlaceCardInsert-shot.0-0.0.mp4` |
| N0 | `6a555749ffde` | none (t2va) | cuda:1, ~158 s | `templates/minimax/video-with-audio-768p/20261122-200804-199daa3a/final/MiniMaxH3768p-text_to_video_audio.0-0.0.mp4` |

Every arm ran to completion. There was **no OOM**. R4's validate warned of a 23.83 GB projection, and the job fit on the 24 GB card. Each job's `note` names its arm.

## Marker reads (PRELIMINARY)

✓ means the marker matches E (and K); ✗ means it does not; - means the marker is not in the shot.

| Arm | Lamp: red glass | Shelf | Window: blue evening | Coat: green (E only) | Framing |
|---|---|---|---|---|---|
| R0 (B0) | ✗ white/brass | ✓ | ✗ orange sunset | ✗ purple | #1398's reverse |
| R1 | ✓ red, foreground left (not centre) | ✓ background right | ✓ behind her | ✗ purple | a real reverse: it looks toward the window with the lamp in front. Not K's framing |
| R2 | ✓ red, at the left frame edge | ✓ | ✓ behind her, with the lamp reflected in the glass | ✗ purple | a close-up, as asked |
| R4 | ✓ red, centre | ✓ both sides | ✓ | ✗ purple | the two stand side by side facing the camera, not across the desk. Aurelio holds an accordion (a prop from his portrait) in the early frames, then it is gone |
| N1 | ✓ red | ✓ | ✓ | - | the empty desk with an open book. There is a slight push-in despite "Static camera" |
| N0 | ✗ two brass lamps with white shades | ✓ | ✗ white daylight / overcast | - | the same prompt as N1, with no card |
| K-night | ✓ the only light | ✓ | ✓ black night (the look asked for) | - | the same layout as K: works as a look |
| K-rev | - | - | - | - | **failed**: Klein's edit returned K's own view, not a reverse |

Preliminary reading:
- The card carried every set marker (lamp, shelf, window) into every H3 arm. That includes R1's reverse, which is not framed like the card, and R2's close-up.
- N1 against N0 isolates the card: with the same prompt and no card, the lamp and the window both drift.
- The coat, which the card does not hold, stayed purple in every arm, as in B0. So the card fixes the set and nothing else.

## Face similarity (`bench/cut_windows.py pair`)

The clips were fetched over HTTP into a scratch directory and scored locally against #1398's A (`534601548d6d`). The cut is A's end into the arm's start.

| Pair | A's last 1 s | B's first 1 s | Step | B reaches A's end level | SFace across |
|---|---|---|---|---|---|
| A-B0 (re-check) | 376 | 199 | 0.53x (-47%) | never | 0.842 |
| A-R1 | 376 | 140 | 0.37x (-63%) | never | 0.673 |
| A-R2 | 376 | 87 | 0.23x (-77%) | never | 0.804 |
| A-R4 | 376 | 407 | 1.08x (+8%) | at once | 0.378 * |

- **A-B0** reproduces #1398's figure (0.840) to within 0.002.
- **R1** sits about 0.17 below B0. It is a harder reverse angle, so read that with care: one seed per arm.
- **R2** is a close-up, and its low sharpness figure reflects the shallower focus on the face as well as the angle.
- \* **R4 is not a usable score.** `cut_windows` takes the largest face in each frame, and in R4 that is often Aurelio, not Estela. So 0.378 compares A's Estela against a mix of the two people.

<details><summary>Per-second sharpness</summary>

| s | A | B0 | R1 | R2 | R4 |
|---|---|---|---|---|---|
| 0-1 | 376 | 199 | 140 | 87 | 407 |
| 1-2 | 300 | 132 | 139 | 64 | 252 |
| 2-3 | 310 | 116 | 144 | 75 | 270 |
| 3-4 | 365 | 125 | 135 | 80 | 415 |
| 4-5 | 396 | 150 | 220 | 76 | 477 |
| 5-6 | 361 | 174 | 195 | 73 | 462 |
| 6-7 | | | | | 361 |
| 7-8 | | | | | 302 |

</details>

## R3

**R3 was not run.** R3 was to run only if R1 was pulled toward K's framing. R1 was not: it is a real reverse, looking toward the window with the lamp in the foreground, unlike K's wide front view. K-rev, R3's card, failed anyway (see above), so R3 could not have run as planned.

## Audio (mean level)

| Arm | dBFS |
|---|---|
| R1 | -55.7 |
| R2 | -48.4 |
| R4 | -40.0 |
| N1 | -71.9 (near-silent: the server warned) |
| N0 | -55.3 |

## Deviations from plan v1

- **Seed.** The H3 arms use seed 1398, B0's real seed, so that B0 is a true control. The plan said 42. The Klein stills use 42.
- **The prompts name `<Picture N>`.** Like any wording change, that is a confound against B0's prompt, beyond the added card.
- **N0** ran the catalog t2va template on its own schedule (5 steps, shift 6, the fl2v turbo LoRA), not the probe's 9 steps and shift 12.
- **R4's second portrait** is Aurelio (`asset:cast/aurelio.jpg`).
- **The coat is not in the card.** K is an empty set, so the coat is scored against E only.
- **The card asset** is `.jpg`, not `.png`: Klein writes JPEG.
- **K-rev failed.** It is recorded as a failed arm, not dropped.

## lem time

| Model | GPU time |
|---|---|
| H3 | about 59 min (R1 11, R2 11, R4 23, N1 11.5, N0 2.6) |
| Klein | about 2 min |

That is **about 61 GPU-minutes** across the two cards, about 40 min of wall time in parallel. The plan's ~40 min was meant as GPU time, so the probe ran over. Most of the overrun is R4: 175 frames with three references took 23 min.

## For Don to view

The marker table is preliminary until you have viewed these:
- R1 and R2 against B0, for the lamp, window and coat;
- N1 against N0, which isolates the card;
- R4, for the side-by-side staging and the accordion;
- K, K-night and K-rev (stills).
