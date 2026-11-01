# H3 long takes: odd shots, then bridges, A/B against a chain (#1094)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-31). Plan v2 approved by Don on 2026-10-31 as "build smaller": a
no-code probe on lem, then one skill reference edit either way, no template.
Two stages, #1102 (probe) and #1103 (skill text), both verified. The probe
record is the stage A results comment on #1094; this doc is the design
record around it.

## The idea

From assessing r34vtraining/Longshot_Studio (a ComfyUI studio for long
MiniMax H3 films; no license, so only the concept was taken). Its tip: to
stop drift building over a long continuous take, render the odd shots on
their own, then fill each even shot as an FL2VA bridge pinned to its
rendered neighbours (first frame = shot i-1's last, last frame = shot i+1's
first). No shot is more than one hop from a fresh render, and the odd shots
can run in parallel across cards.

## Decisions

- **Plan v2 verdict: build smaller.** Probe first, then land only what the
  probe confirms (the repo's practice since #985, #1059, #1031). The issue's
  order (skill text first) was reversed.
- **Corrections to the issue** (accepted by Don):
  - The take needs an odd shot count; a 6-shot take's shot 6 has one
    neighbour. The probe used 5 shots (odd 1, 3, 5; bridges 2, 4).
  - A fair A/B needs a shared identity anchor: both arms used the same
    reference, and shot 1 was one render shared by both arms.
  - The "at least 5 s" bridge is already `first-and-last-frame`'s floor:
    124 frames (5.17 s). Validate refuses 107 and below; 108-123 is
    rounded up to 124 with a warning (tester's note on the spec).
- **Q1: probe at all?** Yes: it settles #1031's take-drift reopen and
  #985's end-frame snap claim on its own.
- **Q2: arm A's baseline.** The `last_segment` Ref2VA chain, the strongest
  one the skill prefers.
- **Q3: where the result lands.** `minimax-h3/references/cuts.md`, plus
  one `script-to-video` sentence on a win; a one-line cross-link with
  #1093's `joins.md` from whichever landed second.
- **Win rule** (fixed before the run): arm B wins only if its take drift is
  smaller on lightness and saturation, it holds identity at least as well,
  its worst seam is no worse, and no bridge shows a tail snap.

## What the probe found

lem, develop cb49329c, workspace `qa-1094-bridge`, draft tier (1344x768,
124 frames, 5 steps, int4, sparse, seed 12345). 9 renders, no re-rolls.

| Win-rule condition | Arm A (chain) | Arm B (odd-then-bridge) | Holds for B? |
|---|---|---|---|
| Take drift, lightness / saturation | -14.09 / +5.09 | +0.16 / +1.05 | yes |
| Identity | one subject; prop doubled, framing wanders | subject duplicated in shot 3 and both bridges | **no** |
| Worst seam (`frame_delta` / `jump_ratio`) | 19.57 / 7.29 | 1.85 / 0.81 | yes |
| Visible bridge tail snap | n/a | none | yes |

**Verdict: doesn't win, on identity.** Arm B's independent shot 3 rendered
two astronauts against a one-subject prompt, and both bridges faithfully
interpolated into and out of it. Every other condition went to arm B by a
wide margin: every arm B seam was under the clip's own frame-to-frame
motion, every arm A seam a visible jump.

Also measured:
- **Bridge audio:** FL2VA bridges carry a different soundscape from the
  Ref2VA shots around them (3.6-6.7 dB level steps, a `seam_level_step`
  warning on all four arm B seams; bridge 2 near-silent). A bridged take
  would need `match_levels` or a bed under the whole take.
- **#1031 take-drift reopen: not met.** Arm A's end differs, but its seams
  don't score low; arm B's seams score low, but its end holds.
- **#985 end-frame snap: not reproduced.** Neither bridge jumps onto its
  pinned last frame (last-12-frame per-frame delta at most 1.03 against a
  body typical of 2.3-6.7), so no tail discard is needed at this tier.

Jobs: `71a3c4ee9eba` (shared shot 1), `e1d37983f04a` (arm A chain, shots
2-5), `d4be72601012` / `1ae6a04cb793` (arm B shots 3, 5), `932c2add5238` /
`305f70b310b9` (bridges 2, 4), `8a8adfbf3c30` (joins and measurements).

### Deviations from the protocol (accepted by Don, Q4)

- The reference was a shared HF-hosted URL, not an `upload_asset` asset
  (the upload was refused; later traced to TokenEconomizer deny rules).
- Arm A's chain was hand-built: a 48-frame carry of each previous segment
  passed as a video reference, which is what `LastSegmentContinuity`
  injects. A real `chain` block can't continue from a separately rendered
  shot 1.
- Take drift used 4 vs 4 frames, not 8: `look_stats` is not an MCP task and
  `analyze_seams`' look window is fixed at 4.
- Ref2VA jobs serialized on cuda:0 because of host RAM (~121 GB each); only
  the bridges ran on both cards at once.

The tester bounced stage A on these (see below). Plan v3, which would have
re-run the probe to a protocol the code can express, was written but never
posted: Don ruled (Q4) to accept the first run, since none of the
deviations touches the identity failure that decides the verdict, and no
re-probe with a clean shot 3 (it would make the verdict post hoc).

## What was built

**Stage A, #1102** (no code): the probe above. No commit or deploy; the
jobs (each with a `set_note`) and the results comment on #1094 are the
record.

**Stage B, #1103** (plugin only): `5880731e`, develop `7dd927b1`. The
"doesn't win" branch: a *Probed, not a rule: odd shots, then bridges*
section at the end of `plugins/dw/skills/minimax-h3/references/cuts.md`, in
`dialogue.md`'s format, with the claim, the numbers, the job ids, the
deviations and #1094. `script-to-video/SKILL.md` and `minimax-h3/SKILL.md`
are unchanged. #1093 had landed, so `script-to-video/references/joins.md`
cross-links the section. No version bump (the plugin's version is the
engine's).

## Bounces

- **#1102:** one bounce. The tester failed M-F264 (carry, drift window) and
  M-F265 (`asset:`) against the recorded deviations; resolved by Don's Q4
  ruling and amended cases, then verified against the existing jobs with no
  new renders. Before that, two verify sessions ran with only
  `get_server_info` exposed (TokenEconomizer's tool pruning had denied the
  rest), and the driver parked it once; not a fault of the stage.
- **#1103:** none.

Cost: the stage comments name no `usage:` figures, so none are recorded.
GPU time was about 9 renders: ~5 min per Ref2VA shot, ~6.5 min per bridge.

## Deferred, and what brings it back

- **A template** (two `for_each` passes, odd then bridges). `for_each` can't
  express "bridge k reads odd shots k and k+1" (a group reference resolves
  to the same-key member only, a cross-list one raises `ForEachError`), and
  one template runs serially on one card. *Comes back if* a re-probe wins
  **and** sessions hand-build odd-then-bridge more than once.
- **A re-probe with a clean shot 3.** The seam and drift numbers suggest it
  might flip the verdict. Don declined it for #1094. *Comes back on* a
  request for a long continuous take that a cut can't serve, or a field
  report of drift over a chained take. Run it to plan v3's protocol: arm A
  as one real `chain` job with shot 1 as its segment 1 (arm B takes it with
  `trim_video`), references via `upload_asset`, drift as the mean of two
  4-vs-4 `analyze_seams` steps, and re-rolls (up to 2) for a wrong subject
  count on either arm, fixed before the run.
- **A take-drift field in `analyze_seams`.** #1031's reopen condition was
  not met.
