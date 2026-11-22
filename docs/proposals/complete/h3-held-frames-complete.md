# Held frames in H3 takes: `analyze_held_frames` and `interpolate_frames` `refill: "held"` (#1527)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-22). Three stages, all verified: #1551 (bench probe), #1552
(`analyze_held_frames`), #1553 (`refill: "held"`). Plan v1 was approved by Don
on 2026-11-21 with every question at its default. v2 (Q5) and v3 (Q6) were
approved at their defaults on 2026-11-22 by the plan-review delegate under
Don's 2026-11-21 delegation. The probe record is
`bench/records/held-frames.md` (with `held-frames.json`). This doc is the
design record around it.

## The idea

vrgamegirl19/comfyui-vrgamedevgirl (pack 10.0.9, read at 12adc26) reported
that their H3 takes sometimes repeat a frame, so motion reads as choppy
(12 fps shown as 24). A sheet of stills can't show it. They saw it in 17 of
30 takes, on the Singularity V2 ref2va finetune, and it followed the shot,
not the seed. They measured it as the neighbour difference of small grey
frames and repaired it with keep-every-2nd plus RIFE, restoring the real
frame at cuts. Their licence is source-available with a paid commercial
tier, so only the concepts were used. No code was taken.

dw had RIFE (`interpolate_frames`) and cut detection (`scene_cuts`), but no
measure, and `assess_output` and `get_output_frames` both look at stills. An
unattended run would accept a choppy take. The 2026-10 AISkills record
(`complete/h3-aiskills-probes-complete.md`, P1) had deferred a cadence probe
until choppy motion showed up in a report.

## Decisions

- **Verdict (v1): build, gated on a probe.** Demand came only from outside
  dw. There were no field reports, and Don had not reported it. So stage A
  measured Don's own H3 workspaces first, and B and C were built only if A
  found holds.
- **Q1:** a fourth `assess_output` probe, run in the default all-probes
  call.
- **Q2:** the repair is an option on `interpolate_frames`, not a skill
  recipe. A recipe would have needed a new decimate task, a length fix-up, a
  trim per cut, `replace_cuts` and `pair_audio`: five or more steps of
  bookkeeping.
- **Q3 (taste, default taken):** the threshold is the lowest count among the
  takes A flags with a cluster. That is **13**. Don did not watch the takes.
- **Q4:** the bench reads Don's workspaces read-only on lem.
- **Q5 (v2):** the repair refills only the held runs, each between its
  anchors, and passes every other frame through byte-equal. This replaced
  v1's keep-every-2nd, because A measured a different pattern (below).
- **Q6 (v3, loosen, default taken):** a refilled run whose anchors differ by
  less than (k+1)·ε can still count in `held_after`. The docs and the H3
  skill say to reroll when `held_after` stays at 13 or more. The rejected
  alternative was to widen the window into real frames, which gives up
  byte-equal passthrough.

## What the probe found (stage A)

On 649 demand H3 takes in 12 workspaces (CPU only, lem, 2026-11-22):
- 510 hold nothing at ε 1.0, and the median is 0 in every mode and schedule.
  79 have a cluster. 10 that are not near-static hold 7 or more. The highest
  is a `soviet-anthem` fanfare take, with 35 held frames in 7 clusters.
- Holds concentrate in 5-step takes, plus a few 9- and 12-step ones.
- **The pattern is one frame shown twice, recurring every 4 frames.** Of 149
  bounded runs, 139 have length 1 and 3 have length 3. 68 spacings are 4
  frames, against 4 at 2 frames. It is not a 12 fps cadence. The cause (VAE
  decode or sampler, latent stride) is not settled.
- Chain segments slow almost to a stop in pairs about every 44 to 50 frames.
  These are near-stops (d ≥ ε), not holds, and were left alone.
- ε = 1.0 separated duplicates from motion under h264 noise.
- **The finetune risk didn't materialize:** base H3 on our pipeline holds.

## What was built

**Stage A, #1551** (bench only). Merged at develop `4ec3720a`, with the
record re-handed off at `19d1b4ff`.
- `bench/held_frames.py` (`scan`, `report`, `file`) reuses
  `dw.tasks.assess.read_media` and `content_cuts`, with unit tests on
  synthetic stacks.
- The record says how to rerun it. A third scan gave a byte-identical JSON.

**Stage B, #1552** (server and plugin). Develop `60e2d098`, commits
`18eec51e` (engine) and `45dd251d` (tests, docs, skills).
- `analyze_held_frames` in `dw/tasks/assess.py`. A frame is held when its d
  is under 1.0 and a neighbouring d is at least 2.0. Frame 0 and frames
  within 1 of a `content_cuts` cut don't count. A cluster is 2 or more holds
  with gaps of 4 frames or fewer.
- It reports `held_frames`, `held_ratio`, `frames` and `clusters[]` (seconds).
  `detail=true` adds a downsampled `series`.
- `RULES` row `held_frames` (`>= 13`, warn). There is one finding per
  cluster, whose `at` is the cluster's start and `range` is
  [start, end].
- `_not_applicable` skips stills and audio-only files. A silent video is
  still assessed.
- It is in `PROBES`, the MCP docstring (inside `SURFACE_BUDGET`), TASKS,
  WORKFLOW_GUIDE, MCP, SERVER and ARCHITECTURE. The `minimax-h3`
  (`references/watching-a-run.md` *Held frames*), `script-to-video` and
  `long-form` check steps run it on every take.
- Cut detection decodes a second time, and only when the first pass finds a
  hold.

**Stage C, #1553** (server and plugin).
- Develop `4bd201a1` (`cd96b673`, `b3eba66b`, `14b789ce`), the bounce-1 fix
  at `2e872551` (`9c7a3863`), and the v3 docs at `429c0798` (`0c14dec4`).
- `interpolate_frames` `refill: "held"`, exclusive with `multiplier`. It is
  refused at validate and at run time.
- Each run grows from B's flagged frames over the adjacent still frames, then
  is refilled at t = (j+1)/(k+1) between its moving anchors. Every other
  frame passes through byte-equal. A run cut off by a cut or the clip's end
  is left held.
- Frame count, fps, shots and audio are the source's. A file `video` input
  (`output:`/`asset:`) loads with its soundtrack when `refill` is set.
- The RIFE inference callable takes a `timestep`, default 0.5, so
  `multiplier` output is unchanged.
- `held_before`, `held_after`, `refilled_frames`, `refilled_runs` and
  `left_held_frames` appear in the step's `log` event, not in `get_job`
  outputs.
- TASKS *Refilling held frames* and the WORKFLOW_GUIDE remedy row cover it,
  and so does the H3 skill's repair line, including v3's slow-run sentence.
- Measured over MCP: the fanfare take went 35 → 7, under the mark. A
  synthetic every-other-frame fixture (H4) went 72 → 0. The k = 3 G4 fixture
  went 2 → 2 with genuinely new frames: SSIM 0.81–0.96, against a
  re-encode baseline of about 0.985.

## Bounces per stage

| Stage | Bounces | Cause |
| --- | --- | --- |
| A, #1551 | 1 | Two of three cases couldn't be run from the first hand-off: the record lacked the rerun and the decimated/static sanity check. Fixed by adding both to the record. Nothing was re-measured. |
| B, #1552 | 0 | |
| C, #1553 | 2 (+1 re-plan before build) | Its first build session stopped without building. A's record showed keep-every-2nd was the wrong stride, which led to plan v2. Bounce 1: with `refill` set, a file `video` input skipped `asset:`/`output:` resolution, fixed in `9c7a3863`. Bounce 2: C-F1270 (G4 held 2 → 2) broke v2's implied promise that every refilled run clears. The cause was the plan, not the code, which led to plan v3 / Q6 and a docs-only change. |

No stage comment names `usage:` figures, so per-stage session cost is not
recorded. The plan estimated about $13 ($3 + $5 + $5), and v3 added about
$0.5. No GPU time beyond the verifies: A was CPU only.

## Not built, and why

- **Widening a slow run's window (Q6's alternative):** it gives up
  byte-equal passthrough, and the slow run is rare: 3 four-frame runs among
  149. It *comes back if* takes routinely stay over the mark after a refill.
- **Touching chain segments' near-stops:** they aren't holds by the measure.
  It comes back on a field report of chain stutter.
- **A fix inside H3 sampling, and automatic rerolls:** non-goals. The holds
  follow the shot, not the seed.
- **#1327's "measured motion" gate:** not woken. B's per-frame d series is
  the measure it would gate on, through its own design.
- **The cause of the 4-frame stride:** not investigated.

## Still unverified

- **C-F1283** (a four-frame run fast enough to reach `held_after` 0) was
  skipped. No qa-cast stretch has d ≥ 5.0. The unit tests cover it.
- **C-F1268's "RIFE not loaded"** half: `get_job_events` names no RIFE load,
  so it is unobserved over MCP. The unit tests cover it.
- Retiring the superseded C cases is filed as dkackman/harnest#279
  (C-F1198, C-F1200, C-F1202, C-F1203's figure) and #284 (C-F1270).

## Standing cost

- One more probe in every default `assess_output` call. A clean take pays
  one decode.
- One more option on `interpolate_frames`.
- The threshold (13) rests on the Q3 default, not on a viewing by Don.
  Re-set it if a field report disagrees.
- The cut guard knows only the cuts `content_cuts` detects. Holds near the
  clip's ends in a synthetic pure-twos clip are undercounted. H3's real
  pattern isn't affected.
