# The chain levers

Part of the `long-form` skill. It lists every lever on an H3 chain: the
default, the knobs that are not defaults, and the ideas the bench measured
and cut. The verdicts come from `bench/records/counters.md` (*Verdict per
idea*, *Defaults by shot type*, *Re-scored under R35*, *Round 3*). They were
read under the noise floor in `bench/records/noise-floor.md`, *Correction
(2026-11-05, ruling R35)*.

## How the bench reads a lever

- **Scenes.** S1 is static, S2 a moving arc and S3 a dialogue walk.
- **Takes.** 6 segments (26.4 s) at seeds 1166 and 2, and two-minute takes
  for the static case.
- **Measures.** Seam jump, seam look, level step, and texture drift against
  segment 1. A difference inside the floor is a tie. Where the numbers and
  the eye disagree, the eye's reading is recorded.
- **Determinism.** Same-seed runs on a two-card server were not
  bit-identical until `cudnn_benchmark` was turned off
  (`bench/records/determinism.md`). That is why the floor includes a rerun
  gap.

## Defaults (built into `shot`)

| Shot | Build | Record |
| --- | --- | --- |
| `shot: static` | guide, `anchor: segment_1`, `anchor_strength` 0.25 | `counters.md`, *Round 3*: wins on every picture measure at 120 s, by 3-23x the floor; Don's call in `for-don.md` |
| `shot: moving` | guide, no counters | `counters.md`: every guide counter loses seam look on S2 |
| `shot: dialogue` | guide, no counters | `counters.md`: every counter ties or loses on S3 |

`shot` is a preset, so any variable you pass still wins. To turn the anchor
off on a static shot, pass `anchor_strength: null` too, or use `shot: moving`.

**Anchor strength.** Use 0.25, not 0.5. The numbers tie, but by eye 0.5
dissolves the face into a white mask by 62 s. 0.25 keeps the face about 20 s
longer and never loses the eyes (`counters.md`, *Round 3*, *The call*). The
anchor is a static-shot lever only. On the arc it lost seam look at every
strength.

## Knobs, not defaults

Offer these only when the user asks, and say why each is not a default.

- **`continuity: last_frame` with `look_hold`.** It ties its own control on
  texture at both seeds and smooths the hand-over (jump and look), not the
  burn. Plain `last_frame` against guide is mixed at seed 1166 and loses at
  seed 2 (`counters.md`, *Clean last_frame controls*). On a static shot only, last_frame + `look_hold` +
  anchor 0.25 held the face longest of any two-minute run and ran 48 min
  against 69. But that run was confounded, it is worse on mid texture, and
  it had a 4.24 seam jump. It is undecided, and the rerun was dropped (ruling
  R37). Never use it with a camera move.
- **`carry_pick`** (the sharpest of the last frames, on `last_frame`). On
  S1 it ties its control. On the arc it froze the camera. Never use it with
  a camera move.
- **The anti-burn LoRA.** The catalog lists it as continuation-only. In the
  bench, its low texture was blur: the face dissolved by frame 277, the
  statue smeared, and S3 lost on jump and texture. It is a knob, never a
  default.
- **The seam grade** (#1033/#1067) is `look_hold`'s pickup part. It runs
  only on `last_frame`, so it was measured inside that row.

## Measured and cut: do not re-propose

- **keystone as keyframe** (#1187) ties, and with anchor 0.5 makes every
  seam worse.
- **hand-over backoff** (#1189) ties on S1 and loses on S2 and S3.
- **sharpest of the last 4** (#1032, `carry_pick`) ties on S1 and froze S2.
- **guide `look_hold`** (#1191) gains inside the floor on S1 and loses on S2.

## Structural levers

Each of these is *pending measurement* until its record lands.

- **Windowed refine seam pass** (Task 17 Step 4): experimental, behind a
  server flag. Do not offer it to end users. *Pending measurement*
  (`bench/records/structural.md`).
- **Latent carry** between segments (#1150/#1044, Task 17 Step 2):
  experimental and flag-gated. *Pending measurement.*
- **Late capstone** (#1187, the moving-camera candidate): experimental and
  flag-gated. *Pending measurement* (`bench/records/repair.md`, *Step 4*).

## Community reports on drift

These are reports from other users (roycho87/NerdNotes, collecting Banodoco
and r/StableDiffusion threads), not dw measurements. Where the bench has
checked a report, its status is given.

- *Community:* static shots drift worst. **Measured:** S1 burns from 24 s
  without the anchor (`counters.md`).
- *Community:* a hard cut or a new angle resets drift, and people report
  about 30 s of clean chain. **Measured:** the 30 s static cut rule
  (`counters.md`).
- *Community:* speed LoRAs add crunch over a long chain. Unmeasured; the
  speed tiers are *pending measurement*.
- *Community:* colour matching fixes colour but not texture. Unmeasured
  alone. The seam grade above is the nearest measured lever.
