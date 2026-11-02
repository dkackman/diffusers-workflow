# H3 guide-chain drift counters: latent anchor and guide noise (#1166)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-02). Don filed the idea on 2026-11-02 from a review of
[roycho87/NerdNotes](https://github.com/roycho87/NerdNotes). Plan v2 is the
plan comment on #1166, which Don approved as written with all defaults. It was
built as three stages: #1173, #1174 and #1175. The plan comment is still the
full design; this file records what shipped, what the probe found and what is
still open.

## The problem

On an H3 chain with `continuity: "guide"`, colour and texture drift build up
segment by segment. The held guide rows reach the model almost clean at every
step, so each segment copies its carried frames, drift included. H3MD Director
finds the drift is born as a slow shift in the per-channel latent mean and
spread. The goal was two off-by-default counters, measured on a 10-segment chain
on a static scene and on a moving-camera scene.

## Verdict: build smaller (cheap arms first, then probe)

The plan ranked five arms by cost and built the two that don't touch the denoise
loop, the row plan or the refine schedule:

- **Arm 5, latent statistics anchor** (H3MD concept): `chain.anchor` plus the
  H3 argument `latent_anchor`. Built as stage 1.
- **Arm 2, fixed guide noise** (PrefixStream concept, its paired encode):
  `guide_noise`. Built as stage 2.
- **Arm 3, the in-loop per-step ramp** (NerdNotes/Lumifel): cut. *What would
  bring it back:* arm 2 null and residual texture drift after arm 5. The probe
  found arm 2 not null, so it stays cut. The plan's "Arm 3, for the record"
  keeps the mechanism (v1's per-row re-noise with H3MD's ramp shape) for a
  revival.
- **Arm 4, keystone and capstone** (NerdNotes): cut, replaced by arm 5. The
  keystone alone "did little"; the capstone needed a new late-gated guide
  grammar. *What would bring it back:* arm 5 holds the static scene but fails
  the arc. **That trigger fired** (see the probe), so a sequence-level anchor is
  a live candidate if Don wants to pursue the arc.

Non-goals that stayed out: audio latents, `ref2va`, any default change, MCP or
REST surface, and #1033's seam colour-match.

## Plan history

- **v1.** The in-loop per-step ramp (arm 3) as the main build. Don sent it back:
  Q1 brought the anchors back into view, Q2 asked for a re-plan around all five
  arms.
- **v2 (approved, 2026-11-02).** Arms 5 and 2 first, then a two-scene,
  two-round probe.
  - Q1 [scope]: yes. Arm 3 only on its trigger.
  - Q2 [scope]: yes. Arm 5 replaces arm 4.
  - Q3 [spend]: no keystone arm in round 1.
  - Q4 [spend]: both scenes, about 2¼ h for round 1 plus about 45 min for round 2.
  - Q5 [scope]: syntax as written. If the arc flattens under arm 5, the
    stage 3 reading names `latent_anchor_strength` as the follow-up.
  - Q6 [taste]: **still open** (see "Deferred and open").

## What was built

### Stage 1 (#1173): `chain.anchor` + `latent_anchor`

Final merge `5921be2c` on `develop`, deployed to lem.

- **`latent_anchor`**, an H3 `t2va`/`fl2va` call argument taking video frames.
  `dw/pipeline_processors/h3_anchor.py` holds the per-channel mean/std match and
  the frame prep; `h3_anchor_steps.py` holds `DwH3LatentAnchorStep`, which
  VAE-encodes the anchor with `encode_vae_condition` and rescales the denoised
  latents to its per-channel statistics. Without the argument the latents pass
  through as the same object.
- **Pre-decode step:** `insert_latent_anchor` puts `dw_latent_anchor` right
  after `after_denoise` in every core denoise sequence of `t2va` and `fl2va`.
  `ref2va` gets no step, and `latent_anchor` with `references` is refused at
  validate and at run time.
- **`chain.anchor`:** `"none"` (default) or `"segment_1"`. With `"segment_1"`
  the chain keeps segment 1's decoded frames and passes them as
  `latent_anchor` to every later segment. One rule,
  `h3_rules.chain_anchor_problem`, is shared by validate and `ChainConfig`; any
  other value, of any type, is refused naming both allowed values. The schema
  no longer types the key, so the rule is its only owner.
- **Step rule:** `dw/guides.py` `_takes_guides_problem` is the one step rule for
  `guides`, `latent_anchor` and `chain.anchor: "segment_1"`.
- **Location policy:** `latent_anchor` is a media argument
  (`dw/locations.py` `PIPELINE_MEDIA_ARGUMENTS`), so it meets the same location
  policy as a guide's `video` at validate and submission. The run-time loader
  already enforced it.
- Docs: `WORKFLOW_GUIDE.md` H3 section and guide-chain table, the
  `ARCHITECTURE.md` H3 row, `ATTRIBUTIONS.md` (H3MD Director).

**Bounces: 3.**
1. Architecture review: a copied step rule (`_takes_anchor_problem`) was a
   second owner. It was deleted in favour of `_takes_guides_problem`.
2. C-F671 1c / C-F676: a non-string `chain.anchor` was refused by the schema
   with a bare type error that didn't name the values. Fixed by dropping the
   schema type.
3. SE-F061: a location-policy gap on `latent_anchor`, filed and handled as a
   private advisory. Fixed by `PIPELINE_MEDIA_ARGUMENTS`.

Cases verified: C-F671 to C-F676, SE-F061.

### Stage 2 (#1174): `guide_noise`

Merged as `bdbe4723` on `develop` (commits 3dbf8a04, 9363e97c), deployed to lem.

- **`guide_noise`**, an H3 `t2va`/`fl2va` float in `[0, 1)`. The range rule is
  `h3_rules.guide_noise_problem`, called by validate (`guides.guide_noise_errors`)
  and the run-time refusal (`Pipeline._with_guide_noise`).
- **Cosine taper** per guide frame (`h3_rules.guide_noise_taper`): `guide_noise`
  at the first frame, exactly 0 at the last, the one next to the join. A
  one-frame guide stays clean.
- **Paired encode** in `DwH3GuideLayoutStep._encode_guides`: the clean and the
  noisy pixels are encoded with the same seed, and `E(noisy) − E(clean)` is added
  to the clean latent, so 0 is exactly today's latent. Guide rows are held as
  before, the row plan and refine schedule are unchanged, and guide audio rows
  are untouched.
- A warning when nothing is noised (no `guides` and no guide chain).
- Docs: `WORKFLOW_GUIDE.md`, `ARCHITECTURE.md`, `ATTRIBUTIONS.md` (PrefixStream,
  MIT; Lumifel and NerdNotes for the diagnosis).

**Bounces: 0.** Cases verified: C-F677 to C-F680.

### Stage 3 (#1175): drift probe on lem

No code. Workspace `probe-1166`, workflow `drift-chain` (a copy of
`templates/minimax/chained-segments`: fl2va, turbo-4step-keyframe at 5 steps,
int4, 960×544, seed 1166, `continuity: "guide"`, `guide_frames` 22, 10
segments, 1042 frames). Measured at f62 / f481 / f991 (segments 1 / 5 / 10)
with `compare_outputs(texture_bands=true)` on a subject-free crop, and seams
with `analyze_seams`. Round 2 (B+C) ran because B and C each beat A on some
measure. The two comparison comments on #1166 hold the full tables, job ids and
frames; the arc table in the second comment replaces the first's.

| Arm | Static scene, seg10 fine Y × A / look mean | Arc scene, seg10 / look mean / jump mean |
|---|---|---|
| A control | 1.00 / 1.67 | blown out toward white / 0.98 / 1.13 |
| B `guide_noise` 0.4 | 1.47 (rougher) / 1.11 | most coherent by eye / 1.07 / 1.09 |
| C `chain.anchor: "segment_1"` | 0.40 / 1.27 | crawling sky texture / 3.62 / 2.20 |
| B+C | 0.29 (cleanest) / 0.69 | worst: full-frame posterized texture / 4.64 / 1.88 |

The lead's reading (not the verdict):
- **The anchor is a static-scene tool at full strength.** It is the best arm on
  the static scene, and with `guide_noise` it is better still. On the arc it
  pulls the lighting back to segment 1 at seams (lightness steps of −15 to −27)
  and lays a crawling texture over the sky. The camera keeps moving in every
  arm, so it doesn't freeze the shot.
- **`guide_noise` is not null.** It is the only arm that doesn't hurt the arc,
  but it roughens static texture.
- Arm 4's trigger fired; arm 3's did not.

**Bounces: 1.** C-F681: the first arc arms B, C and B+C didn't share arm A's
segment 1 (about 22 dB). Neither arm changes segment 1 (diagnostics
`33b3c0ddd350`, `326709c7ad8d`, `f6a218511cb4`); that batch went wrong on the
long-lived cuda:0 worker, and the cause wasn't found. The three arms were re-run
with `rerun_job` and matched A exactly. Case verified: C-F681.

## Cost and bounces

| Stage | Bounces | Cost |
|---|---|---|
| #1173 | 3 (architecture, C-F671 1c, SE-F061) | Not recorded on the stage |
| #1174 | 0 | Not recorded on the stage |
| #1175 | 1 (C-F681, arc segment 1) | Not recorded on the stage. Lem time ran over the plan's 3 h by the three arc re-runs and their diagnostics |

## Deferred and open

- **Q6, Don's taste verdict** on the comparison comments: does B or C hold the
  look better than A without softening the seam or flattening the arc's real
  changes? "Yes" brings a template variable and an H3 skill rule as a follow-up
  plan. "No" leaves both arms opt-in and unrecommended. Until he rules, no
  template or skill sets either key.
- **`latent_anchor_strength`** (Q5's named follow-up): the arc shows the full
  match fighting the scene's real lighting change. Not filed yet; it rests on
  Q6.
- **Arm 4 (sequence-level anchor)**: its return trigger fired. It comes back
  only if Don wants the arc solved after Q6.
- **Arm 3 (in-loop ramp)**: stays cut; its trigger did not fire.
- **Unresolved from the build:** what happens when a step's own `latent_anchor`
  and `chain.anchor: "segment_1"` are both set (the tester's spec note). No case
  asserts it.
- **One seed, two scenes.** Read the direction, not `proven`.
- Related: #1033 (seam colour-match, parked), #1150 (latent-tail continuity,
  parked), #1151 (anti-burn LoRA, awaiting its own taste verdict), #1170
  (`chain.seam_source`, sequenced after stage 1 since both edit `ChainConfig`).
