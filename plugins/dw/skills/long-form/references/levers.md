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
| `shot: static_long` | guide, `keystone: segment_1`, `keystone_slot: last` | `structural.md`, *Keystone slot*: holds the backdrop at 1.1-1.5x segment 1 for 120 s on two seeds, no pop; Don's call (2026-11-06): not a loop |
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

## The keystone on a long static take

**`keystone: segment_1`** (#1187; a guide chain key) pins every later segment
to segment 1's frame 0. **`keystone_slot`** says where: `first` (plain
keystone, the keyframe) or `last` (the fl2va last image). At 26 s plain
keystone ties, and with anchor 0.5 it makes every seam worse (round 1). At
two minutes (`bench/records/structural.md`, *Keystone confirmation*,
*Keystone slot*):

- **Plain keystone** resets the burn on both seeds, with or without the
  anchor, at no wall cost. Each reset is a visible pop mid-segment at an
  unpredictable time (39-111 s), and the crust grows back between resets:
  crusted for 40-60% of the take. A knob: `keystone_slot: first`.
- **Slot `last`** holds the crust at 1.1-1.5x segment 1 for the whole two
  minutes on both seeds, with no pop, and the best two-minute face. Its price
  is the subject easing back to the opening pose at each segment's end
  (every 4.25 s), and about +9 min (+17%) wall. Don's call (2026-11-06): not
  a loop, so it is the static long-take default, as `shot: static_long`.
- The frame slots (`{"frame": 34}`, gated or not) snap the pose or flash
  segment 1's frame: cut.

Static shots only: slot `last` pins every segment's end to segment 1's
pose, and plain keystone lost the look on S2's arc in round 1. Never use
either on a moving shot.

## Narrated take levers

Each verdict is `bench/records/narrated.md` (a narrated take of about 86 s, 12
segments, seed 1 unless said, tier `best`). No arm cleared face, lip sync and
motion together at 86 s, and nothing is measured past 86 s.

- **`shot: dialogue`** (guide continuity, the default): the narration holds
  unbroken, the face is etched by about 30 s and `face_drop` steps up at
  segment 6 (36 s). Good for a take of about 30 s.
- **`shot: standing`** (`keystone: segment_1`, `keystone_slot: last`, no
  keyframes): face and backdrop hold to 85 s, `face_drop` within 0.015 (seed 1 only). She
  stands; each segment eases back to the opening frame. The keyframe-free way
  to a longer take.
- **`shot: walk_keyframed`** (the moving keystone, `keyframes`): the face
  holds to 85 s (`face_drop` at most 0.041 on two seeds) with the cleanest
  seams. It costs one hand-built keyframe per segment. A closed-mouth keyframe
  drops lip sync at the segment's end, so each needs an open, mid-word mouth.
  A keyframe face that differs from the still, such as an edit that ages it, is
  followed and accumulates. A walk did not read at chest-up framing; a slow
  lateral camera arc did, but arc v4 + latent carry sharpens from about
  45 s; the cause is not isolated.
- **`experimental_latent_carry` with `experimental_capstone`**: holds
  saturation flat, does not hold the face past about 60-70 s alone. Stays
  experimental; no key rename.
- **Latent keystone and `anchor_every`**: no measured gain. Stay experimental.
- **`templates/minimax/face-repair`** on a drifted narrated segment: dropped.
  It softens the glaze but has no reference to aim at, and the face comes back
  blurrier and, on one segment, older.
- **Kandinsky S2V 1.0 Lite distilled** (86 s spike): failed on identity, the
  face drifting to a different, darker one by 85 s.
- **Passages joined as cuts**: the released shape for a piece longer than one
  clean take: one narration for the whole script (`templates/narration-long`,
  retimed once), cut into passages with `slice_audio`, one take per passage,
  joined by `concat_videos` at `trim_frames: 0`, `match_levels: "lufs"`
  (`run-and-repair.md`, *A narrated take*). Not yet run end to end.

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
  shots were not benched. They are `moving` by the `class` column of
  `cinematography/references/vocabulary.md` (`push-in`, `handheld`), so they
  take `shot: moving`.
- *Community:* a hard cut or a new angle resets drift. People describe about
  30 s of clean chain with good setups, with drift usually showing around
  segments 3-6. **Default:** past about 30 s a static take is `shot:
  static_long`; cut anything else at about 30 s. The anchor's
  gain holds to 6-8 segments, 25-33 s, and turns into a different failure
  past that (`counters.md`, *Two minutes*; *Round 3* confirms it at anchor
  0.25). A seam where the chain resets on its own shows as a visible jump,
  not a cure (seam 26 of the round-2 two-minute control). The cut itself is `minimax-h3`'s
  `references/cuts.md`.
- *Community:* speed LoRAs add crunch over a long chain. **Unmeasured.** The
  bench ran every arm on the 4-step turbo LoRA, the template default, so it
  has no arm without it. The 8-step 768p file lost to
  it at 1344x768 (`bench/records/tiers.md`); the 4-step default stands.
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
