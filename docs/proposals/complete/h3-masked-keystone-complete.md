# MiniMax-H3 masked handoff + keystone (+ keystone attention) probe (#1505)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-20). Don filed the idea on 2026-11-19 from roycho87's comment on #1391
(issuecomment-7013595305) and NerdNotes test 21 (CC BY-SA 4.0, concepts and data only;
geiru_context_suite is GPL-3.0 and none of its code was read). It is the combination
#1322's close-out named as its reopen trigger: the masked handoff with a keystone, one
seed, against the same-seed control. Plan v1 is the plan comment on #1505. The bench
record is `bench/records/narrated.md`, *Masked keystone: early gate (#1505 stage A,
#1506)*.

## Verdict: build smaller, a two-step seg01-02 gate only

- **The claim tested.** A keystone grafted into a masked carried latent (rather than
  sitting beside a guide) pulls the new frames, and keystone attention on test 21's
  saturation blocks (`30-32,34-37,39,42,44`) then holds saturation across the seam
  without losing #1322's sharpness win.
- **Bench only.** No engine, MCP, REST or workflow-syntax change. A full 12-segment run
  was never in scope: it needed a new ruling from Don on the gate's numbers.

## Plan history

- **v1 (approved 2026-11-20, "Approve with defaults").**
  - Q1 [scope]: Don's #1394 "no more lem time on this" covered the guides-only arms,
    not this mechanism.
  - Q2 [spend]: up to ~1.75 h of lem, step 2 only if step 1 passes.
  - Q3 [none]: the sharpness block set stays out.
  - Q4 [none]: reuse the #1394 control `2817a04202ae` if the build is still `e84bda1c`.
- The design sweep corrected the issue in two places: the processor installs on a
  masked regional-edit call (no refusal or row conflict), and #1322's probe had no
  guides, so every arm here carries the #1394 tail guide.

## What was built

### Stage A (#1506): masked keystone probe, early gate (bench only)

Merges on `develop`: 029e96ca (record 9b624780) and, after bounce 1, 71f4f074 (record
274635df, plus `bench/records/img/masked-keystone-early-face-crops.png`).

- `bench/workflows/masked-keystone-probe.json` (step 1, bright keystone),
  `masked-keystone-control-early.json` (M) and `masked-keystone-attention-early.json`
  (MA, adds only `experimental_keystone_attention {"balance": 20, "blocks":
  "30-32,34-37,39,42,44"}`).
- `tests/test_bench_masked_keystone.py` (15 tests): source frame order, the mask's
  latent-cell boundary, the single-argument arm diff, the join dropping held frames.
- The `docs/ATTRIBUTIONS.md` entry for roycho87's comment and NerdNotes test 21.

**Deviations from the plan** (all recorded on #1506):
1. A masked `source_video` must be num_frames long (first try `da262a1f03dc` failed on
   it), so the source is the keystone, the tail, then seg01's frame 191 held to 192
   frames, all padding under the regenerated mask.
2. The mask is built in-workflow (#1322's `mask-22.mp4` isn't on lem): 22 black + 170
   white frames. 18 frames are actually held (latent-cell rounding), so the join drops
   18 and the head is 187 frames.
3. SFace is re-measured for all three columns against one 0.5 s reference, so the
   baseline reads .967 / .943 rather than #1394's figures.

## What was measured

Jobs on lem, cuda:0, one at a time, diffusers `0.44.0.dev0` @ `e84bda1c`: step 1
`71ec47d79e63`, M `158ff0b20f0c`, MA `9eedaee96617`; baseline `2817a04202ae` reused.
Per-pass times 86.0 / 86.2 / 88.5 s.

- **Step 1 passed on its letter, weakly.** Luma moved +1.10% toward the bright
  keystone on frames 22+ (bar +0.5%; #1394's guides-only figure was +0.13%). But the
  held frames' luma dips and jumps back at the mask edge, which is #1322's seam lift
  and can explain the whole move.
- **Step 2 failed.**

| Gate measure | Baseline `2817a04202ae` | M `158ff0b20f0c` | MA `9eedaee96617` | MA's bar |
|---|---|---|---|---|
| saturation rise | +1.79 | +3.05 | +3.77 | ≤ +1.34 and ≤ M: fails |
| face sharpness distance | 16.9 | 19.9 | 10.6 | ≤ 16.9: passes |
| luma rise | +1.58 | +4.46 | +3.92 | ≤ +1.58: fails |
| SFace, seg02 last 1 s | .943 | .942 | .950 | ≥ .923: passes |
| seam-1 `jump_ratio` | 1.09 | 0.40 | 0.40 | < 1.64: passes |

The seg02-only rise rows (saturation 2.72 / 4.52 / 5.24, luma 1.14 / 3.05 / 2.50) give
the same verdict. M fails saturation, luma and sharpness distance, so "the masked
keystone helps, the attention doesn't" doesn't apply either. The masked carry keeps
#1322's seam win (jump_ratio 0.40) and repeats its loss: brightness and saturation
climb.

## Cost and bounces

- Stage A: one build session plus a bounce-1 record fix; about 1.5 h of lem (three
  jobs at ~30 min each, plus the failed first try). No `usage:` figures were posted.
- **Bounces:** stage A, one. The first hand-off left out the diffusers build, whether
  the baseline was reused, and M's and the baseline's columns (M-F449, M-F453,
  M-F454). The runs and verdict didn't change; the second verify passed.

## Cut, and what brings each back

- **The full 12-segment run.** Only on a gate pass, which didn't happen.
- **The sharpness block arm (`32-35,37,39-40,42,44`).** Only if a masked carry first
  holds saturation and luma; sharpness is the measure it already wins.
- **An engine masked-handoff chain mode.** Still declined (#1322).
- **More seeds.** Only on a single-seed pass.

## What stays

Nothing new in the engine. The bench workflows and their test stay as the
construction a revival starts from. roycho87's guides-vs-carried-latent explanation is
not supported at our settings (20 steps, a narrated face): the keystone's pull under
a masked carry is confounded with the seam lift, and the attention re-split doesn't
stop the saturation climb. The narrated drift remains open.
