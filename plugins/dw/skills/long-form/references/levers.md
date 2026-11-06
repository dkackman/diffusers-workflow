# The chain levers

Part of the `long-form` skill. It lists every lever on an H3 chain: the
default, the knobs that are not defaults, and the ideas the bench measured
and cut. The verdicts come from `bench/records/counters.md` (*Verdict per
idea*, *Defaults by shot type*, *Re-scored under R35*, *Round 3*), with the
keystone and the structural levers from `bench/records/structural.md` and
`bench/records/repair.md`. They were
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

## Under review: the keystone on a long static take

**`keystone: segment_1`** (#1187; a guide chain key, not a `shot` preset)
makes every later segment's keyframe segment 1's frame 0. At 26 s it ties,
and with anchor 0.5 it makes every seam worse (round 1). At two minutes it is
the only lever that resets the burn (`bench/records/structural.md`,
*Keystone confirmation*):

- it resets on both seeds, with or without the anchor, and costs no wall
  time; no take without it ever comes back clean;
- each reset is a visible pop mid-segment at an unpredictable time
  (39-111 s), where only the backdrop texture snaps clean;
- between resets the crust grows back, so the take is still crusted for
  40-60% of its length. It is a large improvement on the burn, not a clean
  take.

It is the leading candidate for static takes past about 30 s. The default is
undecided pending Don's call on the pop. Until then, cut static takes at about
30 s, and offer the keystone only as a knob with the pop named. Never use it on
a moving shot: it lost the look on S2's arc in round 1.

## Measured and cut: do not re-propose

- **hand-over backoff** (#1189) ties on S1 and loses on S2 and S3.
- **sharpest of the last 4** (#1032, `carry_pick`) ties on S1 and froze S2.
- **guide `look_hold`** (#1191) gains inside the floor on S1 and loses on S2.

## Structural levers

Each is experimental and behind a server flag. Do not offer any of them.

- **Windowed refine seam pass** (Task 17 Step 4): wins as a seam pass on
  every shot type measured (seam jump past the floor on S1, S2 and S3, look
  too on S2, nothing worse), at about 36-40 min per minute of take. It is
  **not a burn fix**: drift against segment 1 is unchanged on every scene
  (`bench/records/structural.md`, *Step 4*, *Verdict*). Promotion waits on
  an eye check and the feature stages.
- **Latent carry** between segments (#1150/#1044, Task 17 Steps 2-3): **not
  a default** and not a burn fix. On the static scene the take still burns;
  on the moving scene the seams get worse on both seeds. Under the keystone
  it ties on numbers; its face hold by eye waits on Don
  (`bench/records/structural.md`, *Steps 2-3*, *Keystone confirmation*).
- **Late capstone** (#1187, the moving-camera candidate): **does not win**.
  On S2 one gate ties every measure and the other is mixed, worse past the
  floor on one seam measure at each seed. Not built
  (`bench/records/repair.md`, *Step 4*).

## Community reports on drift

These are reports from other users (roycho87/NerdNotes,
`20261102/h3_latents_and_degradation.md`, collecting Banodoco and
r/StableDiffusion threads), paraphrased. They moved here from the
`minimax-h3` skill's chain notes. Each one is marked *community*, and where
the bench has checked it, its status follows: a **default** is built into
`shot`, a **knob** is offered only on request, **cut** means measured and
dropped, and **unmeasured** means no bench verdict yet.

- *Community:* static and half-static shots drift worst. On a moving shot the
  background keeps being replaced; on a locked-off shot the drift is mostly
  darkening and colour rather than texture. **Default**, in part. S1 static
  burns from 24 s without the anchor, so `shot: static` carries the anchor
  (`bench/records/counters.md`, *Round 3*). The bench saw texture burn on the
  static shot too, not only colour: the wall and stripes crawl and the face
  hardens by frame 583 (*Eye notes (S1)*). The moving arc drifts as well, as
  a blown-out, invented sky by its last segment (*Eye notes (S2)*). Half-static
  shots were not benched; treat one as moving until #1200 classifies it.
- *Community:* a hard cut or a new angle resets drift. People describe about
  30 s of clean chain with good setups, with drift usually showing around
  segments 3-6. **Default:** cut static takes at about 30 s. The anchor's
  gain holds to 6-8 segments, 25-33 s, and turns into a different failure
  past that (`counters.md`, *Two minutes*; *Round 3* confirms it at anchor
  0.25). A seam where the chain resets on its own shows as a visible jump,
  not a cure (seam 26 of the round-2 two-minute control). The cut itself is `minimax-h3`'s
  `references/cuts.md`.
- *Community:* speed LoRAs add crunch over a long chain. **Unmeasured.** The
  bench ran every arm on the 4-step turbo LoRA, the template default, so it
  has no arm without it. The speed tiers are *pending measurement*
  (`quality-and-speed.md`); the 4-step default is unchanged until they land.
- *Community:* measure drift on a patch the subject never enters, against
  segment 1 at the same frames. Whole-frame averages let changes cancel out.
  **Default** method: the bench measures texture on the scene crop against
  frame 62, and that is the playbook's judge step
  (`compare_outputs(texture_bands=true)` on the patch, and
  `assess_output(probe="analyze_seams")` for each seam's `jump_ratio`).
  Low texture can be blur, so read it with the eye (`counters.md`, the
  anti-burn LoRA and anchor 0.5 rows).
- *Community:* colour matching fixes colour but not texture, and matching to
  a frame with different content makes it worse; match on a masked area or a
  similar frame. **Knob.** The engine version is the seam grade
  (#1033/#1067), `look_hold`'s pickup on `last_frame`. Measured inside the
  last_frame `look_hold` rows, it steadies colour at the seams and leaves the
  burn (`counters.md`, *Verdict per idea*). It is never a default, and never
  on a camera move.
