# Narrated passages joined as cuts, run end to end by a cold agent

_Complete, 2026-11-14 (#1311, stages #1336, #1337, #1338, #1362). Design record by
`claude-opus-5-5` (anthropic), lead close-out session, from plan v2 on #1311. It is the
follow-on to `narrated-long-take-complete.md`, which shipped this shape as text only._

## Intent

A narrated piece longer than one clean H3 take (`dialogue` about 30 s, `standing` about
85 s) is split into passages: one take per passage, joined as cuts. The shape shipped
untested. This feature ran it end to end with a cold Sonnet agent that had only the dw
MCP and the plugin, and fixed every gap the agent hit.

**Verdict: build smaller.** One cold drill whose output is also the piece, instead of a
lead-run piece plus a separate drill. Known gaps were closed before any GPU was spent.

## The recipe as shipped

Text lives in `plugins/dw/skills/long-form/references/run-and-repair.md` (*Longer than one
take*), with pointers in long-form `SKILL.md`, `references/levers.md`,
`references/audio.md`, `docs/WORKFLOW_GUIDE.md` (*A narrated long take*) and the
`templates/minimax/narrated-take` description.

1. **One narration for the whole script.** Make a 5-10 s voice reference with
   `templates/generate-speech`, run `templates/narration-long` from it, and retime it
   once with `templates/narration-retime` (`get_workflow` names both; both declare
   `shape: "audio"`).
2. **Passages.** Split at paragraph breaks so each fits its preset: `standing` for a
   monologue to camera, `dialogue` only for a passage under 30 s. Cut the retimed track
   with `slice_audio`.
3. **Takes.** One `narrated-take` per passage, with its slice, its script and a different
   still or angle (`templates/image-edit` from the first still).
4. **Run the first passage and judge it** before queueing the rest. Cost: about 2.6
   GPU-minutes per second of `standing` video, since `validate_workflow` gives
   `narrated-take` no estimate.
5. **Join** with a one-step `concat_videos` workflow: `trim_frames: 0`,
   `match_levels: "lufs"`. Then check each cut with `assess_output(probe="analyze_seams")`,
   `get_output_frames(seams=...)` and `get_output_audio`.

`chained-segments` with an invented voice for each passage is not this recipe (W3).

## What was built, by stage

| Stage | What | Bounces |
| --- | --- | --- |
| #1336, stage 1 | The recipe in `run-and-repair.md` with pointers, and `boundary_slack_seconds` exposed on `narration-retime` (`COMPACT_BUDGET` 14,450). C-F763 to C-F768. | 0 |
| #1337, stage 2 | Drill attempt 1 and its record (`bench/records/drill.md`, *Cold drill: the Gettysburg Address as passages joined as cuts*). C-F852, C-F853. | 1, a plan gap: under v1 the stage could not pass, so a re-plan to v2 (ruled 2026-11-14) |
| #1338, stage 3 | W1-W7 fixes in skill, guide and template text; `shape: "audio"` declared on `narration-retime` and `narration-long` (Q6). C-F771, C-F854, C-F855. | 1: C-F854, the declaration was missing on `narration-long` because the catalog test refuses a declaration equal to the derived shape. Fixed by a `PINNED_DECLARATIONS` exception in `tests/test_catalog_structure.py` |
| #1362, stage 4 | Drill attempt 2 on deployed develop (`get_health` `0.21.0-alpha.1` at `3a6f3c80`) and its record (*Cold drill attempt 2*); "not yet run end to end" replaced with the measured result; V2's probe line and the measured cost added. C-F769, C-F770, C-F772, C-F856, C-F857. | 0 |

No stage comment gave a `usage:` figure, so session cost is left out. The two drill
agents cost $1.567 (attempt 1, 4 turns) and $1.432 (attempt 2, 3 turns).

## The two attempts

| | Attempt 1 (#1337) | Attempt 2 (#1362) |
| --- | --- | --- |
| Shape | 7 `dialogue` passages on `chained-segments`, H3's own voice, no narration track, one still | 2 `standing` passages on `narrated-take`, one `narration-long` + retime, `slice_audio`, a second angle from `image-edit` |
| Server | stale hand-started `595cee1f` (`0.20.0-alpha.1`) | deployed develop `3a6f3c80` |
| Result | 2:35.17, 6 cuts, holes of 1.75-2.75 s at 5 of 6 | 1:46.58, 1 cut at 41.0 s, 0.75 s below -50 dBFS there, length off the narration by 4 ms |
| GPU | about 42 min on both cards | 4.6 h, on cuda:0 (inside Q1's cap) |
| Root causes | W1 (couldn't find `narration-retime`), W2 (stalled on a missing voice clip) | V1 (null estimate, a tool gap), V2 (skipped its own probe; no cost this time), V3 (wrong arg name, self-corrected) |

Attempt 2 was the last under Q1. Keeper: `asset:gettysburg/gettysburg-address.mp4` in
workspace `drill-1311-2`.

## Open questions, answered

- **Voice across cuts:** one voice by construction, since every passage is a slice of one
  track. **Room tone:** level step 0.41 dB, spectral shift 0.04 at the cut. Whether it
  sounds continuous is Don's listening verdict.
- **`standing` on a second seed:** it held. Seed 1362, 65.6 s, `face_drop` at most 0.027,
  every segment passed on its first attempt.
- **Is the skill text enough for the agent to choose unaided?** After stage 3, yes: the
  agent chose `narrated-take` and `standing`, and split at the paragraph break without
  guessing. Before stage 3 it was not (attempt 1).

## Deferred, and why

- **Don's eye-and-ear verdict on the piece** (the three constitution-walk-2 defects, room
  tone and lip sync across the cut at 41.0 s). The plan reserves it to Don. The tester
  passed the measured cases, and the stage hand-off asks for the verdict without claiming
  it. It is not recorded yet.
- **`validate_workflow`'s null `estimate.minutes`** for `narrated-take` (and for
  `image-edit`, `narration-retime` and the join): an engine/catalog gap, outside this
  feature's text-only scope. The recipe carries the measured cost until it's fixed. No
  issue for it is cited in the stage record.
- **The `has-audio` trait on `narration-retime`.** W1's
  `traits="has-audio,needs-input-media"` call still misses it. The plan didn't promise a
  fix, and `shape="audio"` finds it now.
- **A narration-length validate warning.** This was a non-goal, to be reconsidered only if
  attempt 2 oversized a passage. It didn't.
- **A join template** (Q3): not built. One-step `concat_videos` does the job.

## Harness notes

- Attempt 1's runner let the agent end turns on background `sleep`s. Attempt 2 ran with
  `--disallowedTools Bash`, so it waited with `wait_for_job`.
- The runner can't pin the agent to one card. In attempt 2 the host RAM gate serialized
  the second passage behind the first anyway.
- Transcripts are on Don's Mac at `~/testing/drill-1311-runner/` and
  `~/testing/drill-1311-2-runner/`, not in the repo.
