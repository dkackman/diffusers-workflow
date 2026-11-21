# Performer replacement seam: an eroded-mask arm (#1382)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-16). #1382 was filed from #1374 (stage B of #1194), whose record named two
tighter edits to stop `replace-performer`'s regenerated band redrawing nearby set
geometry, such as the couch-back step on S4 `deflect`: `mask_grow` 0, benched in
#1374, and a mask eroded by part of a patch cell before it is pooled. Plan v1
deferred the eroded arm until `mask_grow` 0 had been shown to leave the couch step.
That trigger fired on 2026-11-16, and Don unparked it. Plan v2, the plan comment on
#1382, was approved at its defaults the same day and built as one bench-only stage,
#1390. This file records what was measured.

## Verdict: build, bench only

#1374's record showed `mask_grow` 0 narrowing the change from about 80 px to about
32 px. But its edge was harder (`step` 9.81 / 11.29 against 4.37 / 4.45 at
`mask_grow` 1), and the couch back was still redrawn. Eroding the mask was the one
untested lever. The plan kept it out of the engine: no negative `mask_grow`, no
morphology task, no template or default change. If erosion had won, the engine form
would have needed its own plan.

## Plan history

- **v1:** defer, with the trigger "`mask_grow` 0 still leaves the couch step".
  - Q1 [scope]: Don deferred.
  - Q2 [none]: bench only.
- **v2 (approved 2026-11-16, at defaults):**
  - Q1 [none]: N = 16 px (half a 32 px cell), with one retry at 24 px if 16 changed no
    cells.
  - Q2 [none]: `seam_bands.py --inner K`, byte-for-byte unchanged without the flag.
  - Q3 [none]: the stage queued behind #1379 and #1385 for lem time, one H3 job at a
    time.
  - Don added the ghost test: judge the shoulder crops for a ghost of the old
    silhouette, not only `step`.
- **Correction v2 made to v1:** v1's "report band 0" couldn't show an edge inside the
  silhouette, since band 0 is the mean over the whole person. Hence `--inner`.
- **Arms set their `asset:` inputs explicitly,** because the workflow's `output:`
  defaults pointed at the old bench server, which is gone.

## What was built

### Stage A (#1390): eroded-mask arm (bench only)

Merge `5192dfdc` on `develop` (commits `36de9047` for the scripts, arms and tests,
`1702afa9` for the record).

- **`bench/erode_mask.py MASK.mp4 OUT.mp4 --px N`:** thresholds at 128 as
  `h3_mask_cells.binarized` does, erodes each frame with a (2N+1) square minimum
  filter, writes a 0/255 mask with the same size, frame count and fps, and prints the
  fraction removed. The frame border does not erode, so a person cut off by the frame
  edge stays masked up to it. Tests: `tests/test_bench_erode_mask.py`.
- **`bench/seam_bands.py --inner K`:** bands -1..-K inside the union mask by distance
  to its outside, band 0 the rest of the inside, and a `jump` (band -1 minus band 1)
  printed only with the flag. `step` is unchanged. Output without the flag was checked
  byte-for-byte against the old script. Tests: `tests/test_bench_seam_bands.py`.
- **Arms** `t16-swap-e16-s1194` / `-s2002` in `bench/arms/t16-performer-replacement.json`:
  the g0 arms with `mask_video` `asset:bench/t16/deflect-mask-e16.mp4`. Jobs
  `38bc1fead1fe` (4.1 min) and `b75eda2bd8a8` (4.0 min, warm) on lem cuda:0.
- **Record:** `bench/records/performer-replacement.md`, section "The seam: an eroded
  mask (#1382)".

**Result: erosion fails the rule, so `mask_grow` 1 stays and no engine erode is
proposed.**

- Cells regenerated (of 18870): g0 7491, e16 5959. 16 px was enough, so the N = 24
  retry was not run (offline it would have given 5291).
- `step`: 2.04 / 2.16, under g1's 4.37 / 4.45. **Passes.**
- `jump` between bands -1 and 1: 28.50 / 29.37 against g1's step of 4.37 / 4.45.
  **Fails.**
- Ghost: a pale streak of the source silhouette, shoulder to magazine, on both seeds.
  **Fails.**
- Couch back: still redrawn beside the shoulder on s2002. **Fails.**

A cleaner edge outside the silhouette is paid for with a visible ghost inside it. An
engine erode would ship the same trade.

**A note on the rule's jump test.** g1's own jump is 35.87 / 44.18, so no arm that
regenerates the person could pass "jump no larger than g1's band 1-2 step". The test
compares a measure across the mask edge, where the person is replaced on purpose, with
one outside it. The verdict doesn't rest on it: the ghost and the couch test fail by
eye. A future seam rule should compare a jump against g1's jump, not g1's step.

## Cost and bounces

| Stage | Bounces | Cost |
|---|---|---|
| #1390 | 0 | Session cost not recorded on the stage; the plan estimated $3-5. GPU: 8.1 min for two arms on lem cuda:0 |

Cases verified: C-F934 to C-F939, over MCP, against lem `develop @ 5192dfdc`. C-F938's
unit tests and table are a docs-review item.

## Deferred and open

- **No engine erode, negative `mask_grow` or morphology task.** Nothing in this record
  argues for one. It comes back only with a new measure showing an erosion that leaves
  no ghost.
- **The couch-back step stays open** at `mask_grow` 1. Other levers are how the mask is
  made (#1375, track_people), not how much of it is kept.
- **`upload_asset` with a name files under `uploads/`,** so a bench asset meant to sit
  beside its siblings was copied into the workspace's assets directory by hand, as
  #1374 did.
