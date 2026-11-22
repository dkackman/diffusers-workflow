# H3 Ref2VA environment swap that keeps the motion (#1518)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-22). Don filed the idea in November 2026 from a Reddit post (an anecdote, not
evidence). The post showed H3 Ref2VA swapping a clip's environment and weather while
keeping its motion and camera, using a structured prompt: `subject_definitions`, an
`Edit <Video 1>` summary, `retention_analysis` lines, CHANGE/PRESERVE blocks and
"preserve the original paired soundtrack". Plan v1 is the plan comment on #1518. This
file records what was built, what the probe found and what stays open. The data is in
`bench/records/env-swap-ref2va.md`.

## Verdict: build smaller (a gated bench probe, one stage)

The mechanism argued against it: H3 reads a video reference at 2 fps through the text
encoder, and nothing had measured motion carried by a reference. So the plan ran a
two-run gate first (arm A, the structured prompt, against arm B, a plain paragraph, at
seed 1001), and the rest of the matrix only on a pass. Don approved plan v1 on
2026-11-21 with every question at its default:

- **Q1:** gate first, then arm C (A without `retention_analysis`) and seed 2002 only
  if A passed.
- **Q2:** the soundtrack is judged by muxing the source audio back on, not as a
  generation outcome. A reference conditions the output and never copies its audio.
- **Q3:** no regional-edit arm.
- **Q4:** the missing `video_reference_exponent` on `composable-references` became its
  own issue, #1537, filed by Don.

## What was built

- `bench/workflows/env-swap-{a,b,c}.json`: ref2va, the source clip as the only
  reference (one `MiniMaxH3VideoReference`, no image, guides, `source_video` or mask),
  124 frames, 960x544, 9 steps, turbo ref2v LoRA, int4. The arms differ only in
  `prompt`.
- `tests/test_bench_env_swap.py`: each arm validates, has exactly one video reference,
  the arms differ only in prompt, and C is A minus its `retention_analysis` block.
- `bench/records/env-swap-ref2va.md`: the source, the verbatim prompts, six jobs, two
  mux jobs, per-frame tables, timings and verdicts.
- `plugins/dw/skills/minimax-h3/SKILL.md` § Which shape is the request: one bullet,
  "New setting, same action: the clip as sole video reference, a CHANGE/PRESERVE
  prompt; confirmed", citing the record. It carries no `retention_analysis:`.
- `plugins/dw/skills/minimax-h3/references/intents.md` row 1 went from `untested` to
  `confirmed`, citing the record (the #1519 index).

## What the probe found

A claim holds only when both seeds agree.

| Claim | Verdict |
|---|---|
| A swaps the environment and keeps motion, camera and identity | **Confirmed** (5/5 frames at both seeds) |
| Snow in every frame | **Lying snow only**: falling snow isn't legible in any arm |
| `retention_analysis` is what does the work (A against C) | **Not shown**: C kept a restyled fountain at 1001, and A and C were near-identical at 2002 |
| The structured prompt holds motion tighter than a plain paragraph | **Confirmed, narrowly**: B leaves the walk on one end frame at each seed |
| A plain paragraph also swaps the environment | **Confirmed** |
| The source soundtrack survives | **Not by generation**: the outputs make their own quiet ambience; `pair_audio` puts the source track back length-exact |

This overturns the plan's expectation (a refutation was the likely result). The record
names the probe's limits: the source's audio is flat room tone, so sync was judged by
timing only, and motion was judged by eye at five frames (no pose measure exists).

**Cost.** Per pass 74-95 s, the whole-clip video-reference class, against 22 s for an
image-only reference. Wall 700-1136 s per arm. The first job on a cold card paid about
5 min more. lem time in all: about 88 min over six arms, against the plan's ≤ ~2 h.

## Deviations found in the build

- The arms ran as inline workflows identical to the committed files, because lem's
  checkout didn't carry the branch. The seed-2002 arms are `rerun_job` with
  `{"seed": 2002}`.
- The post's exact text wasn't available, so the prompts follow its described
  structure. The record cites the source and says so.

## Stages

| Stage | What | Bounces |
|---|---|---|
| #1545 | arms, pinning test, the full matrix on lem, record, skill line and intents row | 1 |

The bounce: the hand-off didn't quote the record's facts (job ids, prompts, per-frame
verdicts, timings), which the specced cases judge from the hand-off. The tester's
plugin cache was also stale, so the skill line looked absent. The fix quoted the
record and added the mux for the four arms that had none (`a77cff54e534`). The stage
comments give no `usage:` figures, so dollar cost isn't recorded; the plan estimated
$4-6.

## Not built, and what reopens it

- **A regional-edit arm** (background mask, `hold_audio`): reopens if a Ref2VA swap
  is needed where the person must stay pixel-exact or the sound must be kept by
  generation.
- **A sync test with audible events:** reopens with a source clip that has footsteps
  or other transients to land.
- **Falling snow:** reopens if a piece needs visible falling weather; the probe never
  showed it.
- **A motion or pose measure:** out of scope; reopens if a second motion-keeping probe
  needs it.
