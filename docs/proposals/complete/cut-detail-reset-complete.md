# A cut's face-detail reset, closed with a late-frame still

_Complete, 2026-11-16 (#1385, stages #1388, #1389). Design record by `claude-opus-5-5`
(anthropic), lead close-out session, from plan v1 on #1385. It follows
`narrated-passages-complete.md`, whose keeper this feature probed._

## Intent

Inside a `narrated-take` the face starts as smooth as its still and grows crunchier as
H3 runs. Don's verdict on the #1311 keeper (`asset:gettysburg/gettysburg-address.mp4`,
workspace `drill-1311-2`) was that the ripened face looks more realistic, and that the
problem is the reset at the 41.0 s cut, where passage B opens on a fresh, smooth still.

**Verdict: build smaller.** A bench probe on the keeper's cut first, with a control and
one cheap arm per open option, then the recipe text for the arm Don picked by eye. No
engine, MCP, REST or workflow-syntax surface.

**Non-goals:** stopping or slowing the in-take climb (#1325 and #1322 tried that at seams
and were declined), any change to the render or the chain, a new join option.

## The recipe as shipped

`plugins/dw/skills/long-form/references/run-and-repair.md`, *Longer than one take*, step
3 (*Takes*): for each later passage, take the sharpest face in the previous take's last
2 s with a `get_frame` task, run `templates/flux3/image-edit` with the new angle as
`instruction` and the take's seed, and pass the result as the next take's `image`.
Only B's opening segment was measured (Q3 = no), so the text tells the agent to judge
the passage's last segment with `get_output_frames`.

Pointers: long-form `SKILL.md` (a cut "drops the face to about a quarter of its
sharpness"), `references/levers.md` (one bullet after *Passages joined as cuts*),
`docs/WORKFLOW_GUIDE.md` *A narrated long take* step 3, and, at close-out,
`script-to-video/references/shot-document.md` (it still said a cut "costs nothing at the
seam"). The `narrated-take` description is unchanged.

## The probe (#1388)

On the keeper, A's last 1 s face sharpness is 1141 and B's first 1 s is 276 (0.24x).
B stays at about 200-280 through 17 s. Each arm is a 16 s clip, the cut at 8.0 s, in lem's
workspace `bench`. Record: `bench/records/narrated.md`, *Cut probe at the #1311 keeper's
41.0 s cut*. Tools: `bench/cut_windows.py`, `bench/workflows/cut-probe-{cj,stills,sl}.json`.

| Arm | What | Step at cut | SFace | B's first 8 s |
| --- | --- | --- | --- | --- |
| C | the keeper's cut | 0.23x | 0.902 | stays soft |
| J | B's head in four 2 s chunks, `sharpen` 1.1 / 0.75 / 0.45 / 0.2, rejoined | 0.99x | 0.921 | matched by construction |
| S | `sharpen` (0.36) on B's still, B's segment 1 re-rendered | 0.39x | 0.913 | partly, about 300-450 |
| L | A's frame 980, flux3 image-edit to angle B, B's segment 1 re-rendered | 0.86x | 0.942 | yes, about 700-950 |

L's gate (re-smoothing by FLUX) passed: the edited still read 2085 against a threshold of
799. **Don's pick (Q2, by eye): L.** J reads as etched, a filter and not texture, so the
ramped-sharpen option stays cut. S and C are the weaker arms. The measures agreed with his
eye this time.

**Record corrected:** `bench/records/drill.md`'s "Resets at the cut: none" (whole-frame
measures) carries a correction block.

## What was built, by stage

| Stage | What | Bounces |
| --- | --- | --- |
| #1388, stage A | The cut probe, its four arms and the record. Bench only, lem. C-F925 to C-F930. | 0 |
| #1389, stage B | The L-branch text in `run-and-repair.md`, long-form `SKILL.md`, `levers.md` and the guide; pins in a new test, `test_the_cut_detail_reset_names_the_late_frame_route`. Server + plugin, mini-ai. C-F931 to C-F933. | 0. Its first build session found Q2 not yet ruled and built nothing; Don ruled L the same day |

GPU: the S and L renders took 13.7 and 12.3 min on lem, inside Q1. No stage comment gave
a `usage:` figure, so session cost is left out.

Deviations worth keeping: the probe workflows don't match `tests/test_bench_workflows.py`'s
`s*.json` glob, so that test doesn't validate them (each validated clean by hand). To fit
the new line, long-form `SKILL.md` (at its 12 KiB cap) lost two redundant lines.

## Deferred, and why

- **B's late segments under a late-frame still** (Q3): not re-rendered (about 171 min).
  Whether a sharper start etches B's end over 65 s is unmeasured; the recipe says so.
- **A ramped `sharpen` or a `concat_videos` picture-match option:** cut. Its trigger was
  Don picking J, and he found J etched.
- **A face-sharpness field on `analyze_seams`:** its whole-frame `detail_ratio` read 0.86
  at this cut and missed what Don saw. The face measure tracked his eye here, which is the
  plan's trigger for this. It would be new MCP output, so it needs its own issue.
- **A head-of-take H3 refine:** not tried, for the reasons `levers.md` gives for
  `face-repair` on narrated segments and #1325's decline.
