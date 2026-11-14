# LTX long takes (Task 23)

## Verdicts

- **last_frame chain (`ltx2/chained-segments`): stays the default.** Nothing
  in the round beats it on picture at no cost and on both seeds.
- **extend (`ltx2/extend-clip` shape, each segment conditioned on the whole
  previous one): not a default.** It wins seam jump on both seeds and texture
  on seed 1166. But it costs 2.1x the wall time, and on seed 2 the win is an
  empty road: the courier rides out of frame in segment 1, and extend keeps
  him gone for the last 10 s. It continues what is there, faithfully. Offer it
  as a knob for a shot whose subject stays in frame.
- **look_hold: wins on numbers, invisible by eye, safe on a moving camera.**
  On S5 it wins seam look on both seeds (1.72→1.17, 1.68→1.22); seed 2 is
  strictly mixed by 0.001 on fine texture. By eye the 15 s takes are the
  control's shot, same framing at every sampled time. On the LTX orbit it
  ties and does not hold the camera back (below). Warm, it costs no wall
  time. Recommendation: make it the LTX chain default, with Don's ok, since
  the gain is not visible.
- **carry_pick 4: loses.** Mixed on seed 1166 (coarse texture worse), loses
  on seed 2 (jump 1.25→1.81). It also shortens the take by 0.2-0.3 s.
- **check + 1 re-roll: tie, free.** No segment failed on either seed (every
  shot `attempts: 1`), so the take is bit-identical to the control and wall
  time is the same. S5 cannot tell whether the check helps on LTX. It does
  not see the audio dropout below; the check measures picture only.
- **duration_seconds 36 (eight segments): not usable as one take.** On both
  seeds the rider changes by about 30 s (helmet, box, shirt), and pale
  vertical streaks grow in the sky and sea from about 20 s. With look_hold
  the courier shrinks to a dot down the road by 22-27 s, and segment 7
  (29.8 s) replaces him with a different man on a green moped. The look_hold
  seam numbers at 36 s (look 1.41→1.00) are measured on that swap.

## Recommended default

- `ltx2/chained-segments`: last_frame continuity, 3 segments of 121 frames
  (about 15 s), no carry_pick, no check. look_hold on is the one candidate
  change (Don's call, above); until he rules, the template is unchanged.
- **Cap an LTX long take at about 15-20 s (3-4 segments).** Past that, cut
  and start the next shot from a fresh still of the subject. The round gives
  no counter that holds the subject longer.
- `ltx2/extend-clip`: no default change. It is the right template when the
  shot must continue exactly what the previous clip shows, at 2x the cost.
- No template default changes: the round supports none (plan Step 2).

## Orbit check (look_hold on a moving camera)

**look_hold does not hold an LTX orbit back.** The H3 ruling ("never pair
look_hold with last_frame on a moving camera", `counters.md`) is H3's, not
LTX's.

- **Setup.** `bench/workflows/ltx-arc.json`: the S5 chain (last_frame, three
  121-frame segments) on S2's statue still with an orbit caption, measured
  under S2's crop. Arms `bench/arms/ltx-arc.json`, record `ltx-arc.json`.
- **By eye.** On both seeds, with and without look_hold, the camera goes from
  the statue's front (0.5 s) to three-quarter (5 s), side-back (10 s) and the
  statue's back (14.9 s). The look_hold takes match the controls frame for
  frame in framing.
- **Measures.** Every measure is inside the floor on both seeds: a tie.

| arm | seed | job | jump | look | tex fine | tex mid | tex coarse | wall min |
|---|---|---|---|---|---|---|---|---|
| control | 1166 | `a3e99c3df84b` | 0.89 | 0.45 | 0.205 | 0.681 | 0.019 | 11.2 (cold) |
| look_hold | 1166 | `9331cd20fa39` | 0.86 | 0.42 | 0.231 | 0.700 | 0.040 | 8.9 |
| control | 2 | `84328fe3ad81` | 1.31 | 0.80 | 0.776 | 0.857 | 0.652 | 8.8 |
| look_hold | 2 | `64641b316eed` | 1.24 | 0.69 | 0.692 | 0.729 | 0.649 | 8.9 |

- **Cost.** Warm, look_hold costs nothing measurable (8.9 against 8.8 min).
  The 11.4 min of the S5 look_hold runs is a model reload after the extend
  arm, not look_hold.
- **Dropout again.** Seed 2's segment 3 is silent here too (-89 dBFS), with a
  different prompt and picture (below).

## Audio dropout (follow-up candidate)

LTX sometimes renders a near-silent segment, about -89 dBFS against -23 to
-50 dBFS for its neighbours. Segment RMS per take (dBFS):

| arm | seed | seg 1 | seg 2 | seg 3 |
|---|---|---|---|---|
| control | 1166 | -40.5 | -23.0 | -24.2 |
| control | 2 | -42.4 | -46.5 | **-89.2** |
| look_hold | 2 | -42.4 | -46.5 | **-89.2** |
| carry_pick 4 | 2 | -42.4 | -50.3 | **-89.2** |
| check r1 | 2 | -42.4 | -46.5 | **-89.2** |
| extend | 1166 | -49.7 | **-87.8** | -52.9 |
| extend | 2 | -39.0 | -46.7 | **-89.0** |

- On seed 2, segment 3 is silent in every 3-segment arm, though look_hold
  and carry_pick change its start image, and in both orbit takes, which have
  another prompt and still. The silence follows the segment's seed, not the
  prompt or the picture.
- The 36 s seed-2 control's segment 3 is quiet (-57 dBFS) but not silent.
- The lem S5 noise-floor runs (CUDA) had level steps of 2.1-2.9 dB, so no
  dropout there. Every dropout so far is on mini-ai (MPS).
- **CUDA check: no dropout.** The seed-2 control rerun on lem (`645d8600ac92`,
  cuda:0, same workflow and arguments) has 5 s means of -50.0, -51.1 and
  -45.1 dB, so segment 3 is not silent. It is a different draw (CUDA and MPS
  noise differ, so the picture differs too), so this does not rule out the
  seed, but every silent segment so far is MPS. The CUDA take ran 3.2 min
  against 8.9 min warm on mini-ai.
- So on S5 the seam level step measures dropouts, not seams. Knobs here are
  judged on picture, as the S1/S2 ruling does for scenes with no held voice.
- The chain check does not catch it. A segment-level silence check (RMS under
  about -70 dBFS where the prompt asks for sound) would. Worth an issue: is
  it MPS-only, and is it the audio VAE or the vocoder? (#1221; the chain
  check's `audio_dbfs` floor and `chain_segment_silent` warning now catch it.)

### Cause

**The conditioning, not the audio decode** (#1221 stage B2, #1267; mini-ai,
MPS, 2026-11-08). Two runs of the S5 seed-2 control, the realized workflow
of `7fbfc3451e90`, differing only in `torch_dtype`:

| run | job | `torch_dtype` | seg 1 | seg 2 | seg 3 | seg 3, muxed |
|---|---|---|---|---|---|---|
| B, control | `cb3697f4179c` | `"torch.bfloat16"` | -46.3 | -56.8 | **-88.7** | -91.7 |
| A, fp32 decode | `b860e3bc8dd6` | `{"default": "torch.bfloat16", "audio_vae": "torch.float32", "vocoder": "torch.float32"}` | -46.3 | -56.8 | **-103.5** | -128.8 |

Levels in dBFS. The segment columns are the shot's `audio_dbfs`: the RMS of
the generated audio over the kept span, which is what the chain check
measures. "Muxed" is `assess_output`'s shot RMS of the final file.

- **The probe stands.** Run B's segment 3 reads -88.7, within 0.5 dB of the
  -89.2 in the table above, though only just. Segments 1 and 2 don't
  match that table's control row (-42.4, -46.5). This server has rendered
  seed 2 as -46.3 / -56.8 since at least `1deff016d727`, so the take has
  moved since the 2026-11-06 bench runs, and the dropout moved with it.
  Run B is the control for run A because the two share their latents.
  Segments 1 and 2 agree to 0.1 dB, and the decode runs after denoising.
- **The reading.** In run A, segment 3 is still far under -70. It gets
  quieter, close to digital silence (-128.8 muxed), and is not restored.
  The fp32 decode turns these latents into silence. bf16 only added
  rounding noise on top (the -89). So the audio decode doesn't cause the
  dropout. The cause is the **conditioning**: what segment 3 is generated
  from, which is segment 2's last frame plus prompt 3.
- **Re-roll.** This record doesn't settle whether a re-roll can bring sound
  back. A fresh seed draws fresh audio noise against the same conditioning.
  The one re-roll on record, `1deff016d727` (segment 3 re-rolled at seed
  1000005, -88.6), stayed silent. Whether `audio_dbfs` stays re-rollable
  or becomes warn-only is Don's call, asked on #1221.
- **No vocoder-only run.** Neither the vocoder nor the audio VAE was
  tested alone, and the cause doesn't depend on telling them apart. There
  is no vocoder-only run because LTX-2's decode casts the latents to
  `audio_vae.dtype` but passes the mel to the vocoder uncast
  (`pipeline_ltx2_image2video.py:1766-1768`). A dict with only one of them
  in fp32 fails with a dtype mismatch. Both load in fp32 together, or
  neither does.
- **Open by construction: MPS-only?** CUDA and MPS draw different noise
  from the same seed, so the lem take (`645d8600ac92`) is a different
  draw. It can't show whether these conditions would also go silent on
  CUDA.
- **No stage C.** The fp32 decode doesn't remove the dropout, so the LTX
  templates keep their bf16 `audio_vae` and `vocoder`.

## Setup

- **Scene.** S5 (`bench/workflows/s5-ltx-action.json`): a courier cycling a
  coastal road, tracking camera, wind and surf, no speech. Three 121-frame
  segments at 960x544, 24 fps, about 15 s.
- **Arms.** `bench/arms/ltx-long.json` (13 runs, seeds 1166 and 2) and
  `ltx-long-probe.json` (the first control). extend runs
  `bench/workflows/s5-ltx-extend.json`: a 241-frame `LTX2ConditionPipeline`
  pass conditioned on the whole previous 121-frame segment, keeping its new
  120 frames. carry_pick runs `lf-s5-ltx-action.json`.
- **Server.** mini-ai (MPS), bench server port 8766, LTX-2.5 with uint4 SDNQ
  transformer.
- **Records.** `ltx-long.json`, `ltx-long-probe.json`, `ltx-arc.json`.
- **Floor.** jump 0.124, look 0.438, fine 0.081, mid 0.159, coarse 0.118
  (`combined_floor` of the bench's rerun gap and seed floors). On mini-ai the
  S5 control reran bit-identical (`29884b7da674`, `3364de6cb5c1`), so this
  floor is conservative.

## Measures

Each arm is read against the control at its own seed. Cells give the control
value, then the arm. **Bold** is better past the floor; _italic_ is worse past
it. Level step is shown but not counted (dropouts, above).

| arm | seed | job | jump | look | tex fine | tex mid | tex coarse | level step dB | wall min |
|---|---|---|---|---|---|---|---|---|---|
| control | 1166 | `29884b7da674` | 1.29 | 1.72 | 0.500 | 0.560 | 0.628 | 9.3 | 8.9 (11.5 cold) |
| control | 2 | `7fbfc3451e90` | 1.25 | 1.68 | 0.104 | 0.048 | 0.188 | 23.3 | 8.9 |
| extend | 1166 | `516802e1acff` | **0.93** | 2.00 | **0.153** | **0.041** | **0.175** | 36.6 | 18.6 |
| extend | 2 | `f032311a7f24` | **0.77** | **1.15** | 0.145 | _0.224_ | 0.215 | 24.8 | 18.3 |
| look_hold | 1166 | `aaa32cab8384` | **1.13** | **1.17** | 0.437 | 0.426 | **0.455** | 9.0 | 11.4 (reload) |
| look_hold | 2 | `5b471b1428a1` | **0.88** | **1.22** | _0.186_ | 0.147 | **0.004** | 23.3 | 11.4 (reload) |
| carry_pick 4 | 1166 | `d563c0e5d4ab` | **1.12** | 1.34 | **0.344** | 0.434 | _0.961_ | 5.2 | 8.9 |
| carry_pick 4 | 2 | `d0c1b0a56820` | _1.81_ | 1.36 | 0.093 | _0.252_ | 0.173 | 23.3 | 8.9 |
| check r1 | 1166 | `ef9b8abfa067` | 1.29 | 1.72 | 0.500 | 0.560 | 0.628 | 9.3 | 8.9 |
| check r1 | 2 | `61b7fc4d4fd9` | 1.25 | 1.68 | 0.104 | 0.048 | 0.188 | 23.3 | 8.8 |

Strict verdicts (`bench/verdict.py`, all five picture measures): extend wins
at 1166 and is mixed at 2 (mid 0.176 over, floor 0.159); look_hold wins at
1166 and is mixed at 2 (fine +0.082 against a 0.081 floor, a hair); carry_pick
is mixed at 1166 and loses at 2; check ties at both.

36 s takes (eight segments, 864 frames). Seam means span seven seams, so read
them against each other, not against the 15 s rows:

| arm | seed | job | jump | look | tex fine | tex mid | tex coarse | wall min |
|---|---|---|---|---|---|---|---|---|
| control 36 s | 1166 | `cae2ee120f6b` | 1.27 | 1.41 | 0.306 | 0.131 | 0.648 | 23.5 |
| control 36 s | 2 | `f721d3694acb` | 1.34 | 1.38 | 1.058 | 1.560 | 2.157 | 23.5 |
| look_hold 36 s | 1166 | `3bcebd98295e` | **0.80** | 1.00 | 0.371 | _0.561_ | **0.266** | 23.8 |

Seed 2's 36 s texture at the last segment is 3-10x seed 1166's: the streaks.

Cost, wall minutes per output minute on mini-ai (warm): control 35,
look_hold 35 (the 11.4 min rows paid a reload after extend), extend 74,
36 s chain 39.

## By eye

Contact sheets at 0.5, 4.9, 5.2, 9.9, 10.2 and 14.9 s (the seams sit at
5.04 and 10.0 s), and at 1, 10, 20, 30 and 35.9 s for 36 s takes.

- **Seams are clean on every arm.** No arm shows a visible jump at either
  seam; the frames either side of 5.04 and 10.0 s match in framing and light.
- **The camera does not track.** The prompt asks for a tracking shot at the
  rider's speed. Seed 1166 drifts from a side view to behind him by 10 s,
  and he recedes down the road by 15 s. Seed 2 lets him ride out of frame at
  the end of segment 1; last_frame brings a courier back into frame in
  segment 2. extend does not, and shows an empty wall to 15 s.
- **look_hold at 15 s** is the control's shot on both seeds, with the same
  framing at every sampled time; the seam-look gain is not visible at this
  size.
- **36 s.** From about 20 s pale vertical streaks run through the sky and
  sea; by 30 s they cross the whole frame. The rider changes: seed 1166's
  loses the black helmet and the box changes shape; look_hold 36 s swaps him
  for a man in a hat on a green moped at 29.8 s.

## Second reading: S5's own seed floor (2026-11-06)

The verdicts above use the bench-wide floor (S1-S3 seed floors and the lem
rerun gap). The counter round's rule is the scene's own floor,
`combined_floor(noise_floor(control@1166, control@2), rerun_gap)`, picture
measures only. On mini-ai that is, for S5 at 15 s (rerun gap 0, bit-identical
rerun):

| jump | jump max | look | look max | tex fine | tex mid | tex coarse |
|---|---|---|---|---|---|---|
| 0.045 | 0.190 | 0.045 | 0.160 | 0.396 | 0.511 | 0.440 |

and for the 36 s pair: jump 0.066, jump max 0.160, look 0.024, look max 0.200,
fine 0.752, mid 1.429, coarse 1.509.

Under it (primaries seam jump mean and seam look mean, as for S2):

| arm | seed 1166 | seed 2 |
|---|---|---|
| look_hold | wins both, nothing worse | wins both, nothing worse |
| extend | jump mixed, look loses (look mean and max worse) | mixed (look max 1.92 vs 1.72) |
| carry_pick 4 | wins both | jump loses (mean and max) |
| check r1 | tie (bit-identical) | tie (bit-identical) |
| look_hold 36 s | mixed: look max 2.84 vs 1.92, the moped swap seam | - |

So look_hold passes the strict rule on both seeds under the scene floor; the
seed-2 fine-texture "mixed" above comes from S1's 0.081 floor on a crop that
is sky (`noise-floor.md`: S5's fine band is a weak signal). The two-seed
floor is thin (seam look 0.045). Nothing else changes: extend and carry_pick
still fail on picture by eye, and the gain is still not visible at 15 s.

A ready change for Don's yes: `ltx2/chained-segments` gains a `look_hold`
variable (default true) wired to `chain.look_hold`, with two tests in
`tests/test_catalog_structure.py` (`..._holds_the_look_by_default`,
`..._look_hold_false_is_the_plain_chain`). It needs one more edit before it
lands: with `look_hold` declared as a variable anywhere in the catalog,
`test_a_description_names_only_variables_the_workflow_declares` fails on
`minimax/chained-segments`, whose description names `look_hold` without
declaring it. Reword that sentence (or declare the variable there) in the
same commit; the rest of the suite (18,787 tests) passed.
