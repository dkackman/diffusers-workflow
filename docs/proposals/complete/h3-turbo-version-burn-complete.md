# H3 turbo version vs chain burn: benched, no gain (#1399)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-17). Don filed the issue on 2026-11-16 from KadoBOT/ComfyUI-H3-Continuity-Studio
(MIT, read at a2ddb97), whose `guide/continuity/long-chains.md` ("The model matters most")
measured a six-clip chain's texture energy falling from 2.04 (FastH3 omniref) to 1.19 on
MiniMax's REF2VA weights with lightx2v's ref2v turbo LoRA v1.1. The ask was whether
swapping our chains' turbo LoRA would reduce the face-sharpness creep (+105 on the #1325
control `aed61997052a`). The plan comment on #1399 (v2) holds the full sweep; this file
records what was measured.

## Verdict: no gain. The templates keep the turbo files they ship

Neither newer lightx2v file burns less than the file it would replace, each on its own
partition and against its own control:

- **fl2v 4step v1.3 (A1) vs v1.2 (A0), on `chained-segments`:** creep no lower on either
  stand-in region (whole frame 1.04x A0's, centre box 1.43x), the same detail climb
  (x1.47 vs x1.45), seams within tolerance (mean |L| 0.43 vs 0.52), and it darkens 6.6 L
  where A0 holds.
- **ref2v 8step v1.1 (B1) vs v1.0 (B0), on `chain-video-continuity`:** fewer `chain_check`
  fails (2 vs 7) and a smaller detail climb (x1.32 vs x1.41), but a mean seam |L| of
  **4.94 against 1.26**, which fails the bar's "seams no worse" outright.

No catalog entry was added and no template default moved, so there is nothing to reverse.

## What the design sweep settled

- **The swap as the issue wrote it is refused, correctly.** `MiniMaxAI/MiniMax-H3` has two
  transformer partitions: t2va/fl2va denoise on `transformer/`, ref2va on
  `transformer_ref/`. A ref2v LoRA on an fl2va step is a cross-partition error at validate
  (`dw/adapter_compatibility.py`). Continuity Studio's "REF2VA weights" are our
  `transformer_ref/`, not a checkpoint we lack.
- **Their result changes two things at once.** Continuity Studio carries continuity with
  guides on the REF2VA weights; dw's ref2va chain carries the tail as references, and
  guides need the fl2va layout step. That setup was cut (Q3) and returns only as its own
  feature, which B1's result does not call for.
- **#1326 had already measured the v1.0 pair** (`complete/h3-turbo-lead-in-complete.md`):
  the two chains burned about equally. The case for running was that v1.1 is the file
  Continuity Studio measured, and we had never run it.

## Plan history

- **v1 (design, 2026-11-16).** Verdict "build smaller": a two-arm bench (A1 fl2v v1.3, B1
  ref2v v1.1), an arm C (B1 + #1151 anti-burn at 0.5) behind a gate, #1326's runs reused
  as controls, no code.
- **Approved at defaults 2026-11-16.** Q1 [spend] ~75 min lem plus ~60 for C and two
  downloads; Q2 [none] reuse controls `5fa5a09b083f`/`fecc614dc513`; Q3 [scope] cut guides
  on `transformer_ref/`; Q4 [scope] include A1. Don added run conditions: one H3 job at a
  time, quote `get_diffusers_state`, each arm against its own control only.
- **v2 (2026-11-16).** Folded the ruling in. No scope change.
- **Spec note left open:** arm C's gate compared B1 with A0 (across chains), against the
  "own control only" rule. It became moot: the gate could not be read (below).

## What was built

### Stage A (#1403): bench and record (bench only)

Commit dc7f01ab, merged into `develop` as 9d875ee4. Bench files and docs only, no deploy.

- **Arms** in workspace `probe-1399` on lem, seed 42, #1326's prompts, 960x544, 8
  segments, only `lora_weight_name` varying (`bench/workflows/probe-1399-*.json`):
  A1 `372b4229b0a1` (fl2v v1.3, 5 steps, shift 6/3, 15.2 min) and B1 `8f7d30d7434e`
  (ref2v v1.1, 9 steps, shift 12/3, 53.8 min). ModelTC publishes no shift for either
  file, so each kept its predecessor's.
- **Record:** `bench/records/narrated.md`, *#1399 turbo version vs burn (#1403)*, with the
  full measurement table, the quoted diffusers state and a stated tolerance (creep ±15%,
  seam L ±0.3, mean |L| ±0.1).
- **`bench/region_creep.py` (new):** gray Laplacian creep per 1 s window on a fixed centre
  box and on the whole frame, the stand-in for the face measure.

Deviations, all accepted by Don on 2026-11-17 (record ruling on #1403):

1. **No face creep or SFace.** The #1326 subject is a toy astronaut with a mirrored visor;
   YuNet found a "face" in 2-38 of 140-164 samples and never in both the first and last
   window of any arm. The region stand-in replaced it, and it misses chroma noise (B0's
   background speckle reads as burn while its gray Laplacian falls).
2. **Arm C did not run.** Gate condition 2 held (B1 2 fails vs B0 7); condition 1 could not
   be read (whole frame 0.52x opens it, centre box 0.92x shuts it), and an unreadable gate
   was treated as not met. B1's 4.94 seams rule C out either way.
3. **Different diffusers builds.** The arms ran on 0.44.0.dev0 `1dabc292`; the controls very
   likely on `43e30028` (inferred from that week's records, not held by the jobs). The
   tolerance above is the margin below which a difference is not attributed to the file.

## Cost and bounces

| Stage | Bounces | Cost |
|---|---|---|
| #1403 | 0 (went to Don on a `record` question first; verified on the first hand-off, cases C-F956–C-F960) | Session cost not recorded on the stage. GPU: two H3 chains on lem, about 69 min (A1 15.2, B1 53.8), one job at a time |

## Deferred and open

- **Arm C (anti-burn on ref2v v1.1):** not run; B1's seams rule it out.
- **Guides on `transformer_ref/`** (Continuity Studio's actual setup): not filed. Its
  trigger was B1 meeting the bar, which it did not.
- **fl2v v1.3 as a template default:** not filed. Its trigger was A1 winning.
- **Known limits that keep this "no gain", not "proved worse":** one seed per arm,
  different diffusers builds for arms and controls, and a stand-in burn measure.
- **For any future turbo-version bench:** use a subject with a visible face, so face creep
  can be the measure.
- **Tester note:** the record's per-seam and detail figures live in a repo file a tester
  cannot read over MCP; the verify compared mean |L| only.
