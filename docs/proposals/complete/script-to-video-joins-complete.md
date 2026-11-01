# Cutaway and lead-in joins in `script-to-video` (#1093)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-31). Plan v2 approved under Don's delegation on 2026-10-31 (Q1
composition-only, Q2 the default re-entry plus one line on the A-tail video
reference, Q3 the tester runs both paid cases). One stage, #1100, verified
2026-10-31.

Source of the idea: r34vtraining/Longshot_Studio (a ComfyUI studio for long
MiniMax H3 films, assessed 2026-10-31). The repo has no licence, so only the
concept was taken. Its join chip has Cut, Context, Bridge, Continuous,
Lead In and Cutaway.

## The question

How can a shot list say "skip over this one" (a B-roll insert that the shot
after it doesn't continue from) and "land on the next shot" (a shot whose
last frame is pinned to its successor's first)? Today an insert in the
middle of a dialogue run breaks continuity. The shot after it either
continues from the insert or starts cold.

## What the design sweep found

- **The chain block can't say it.** `dw/workflow_schema.json`'s chain
  `$def` has one uniform `continuity` and a string `prompts` array.
  `run_chain` (`dw/chain.py`) overwrites one `carry` per segment and renders
  strictly in order. A cutaway needs a non-adjacent carry. A lead-in needs
  a forward dependency.
- **`for_each` can't say it either.** There is no relative indexing, and
  groups expand whole, so no per-entry frame grab can sit between entries.
  A literal `from_previous_result: "shot@<key>"` cross-entry reference
  slipped through then, with no test pinning it. It was filed as #1101,
  and validation now refuses it (`complete/for-each-cross-entry-complete.md`).
- **Separate runs can say it.** `get_first_frame` / `get_last_frame` read
  `output:` files. The renderers both joins need already exist:
  `templates/minimax/image-to-video`, `first-and-last-frame`,
  `last-frame-only` and `dialogue-short`.
- **Seam bookkeeping needs no change.** For a cutaway, both seams in the
  cut (A→B, B→C) are real picture cuts, so `concat_videos`' `hard_cut` is
  honest. The A→C continuity is checked by comparing frames, not as a seam.
  For a lead-in, the B→C join is pinned by construction.
- **Corrections to the issue.** "Context (`h3_context_ir`)" is a prompt
  convention plus `last_segment`/`guide` continuity, not an engine join.
  "Bridge" exists only as a Music3 lyric tag, and the two-keyframe pin is
  `first-and-last-frame`. The `shots-batch` / `storyboard` entries render
  independently, so the join belongs in the skill's shot list, not in a
  template field.

## Verdict

**Build smaller**: composition layer only (skill text). No engine, schema,
template or task change. The evidence was thin: no field report asked for
either join, and Don's film workspaces already build episodes as separate
shots joined by `concat_videos`, which is where both joins work as separate
runs.

## What was built

### Stage A, #1100: the joins in the `script-to-video` skill

- **`plugins/dw/skills/script-to-video/references/joins.md`** (new):
  - **Cutaway A, B, C.** C's source is A. `get_last_frame` of `output:<A>`
    goes into `image-to-video` for a silent C, or `dialogue-short` with the
    same cast references for a talking C. One line names A's tail as a
    video reference (`chain-video-continuity`'s pattern) for a C that picks
    up motion mid-action, with its slot and VRAM cost. Concat is in story
    order. The re-entry check compares A's last frame with C's first.
  - **Lead-in B → C.** Render the target first. `get_first_frame` of
    `output:<C>` becomes `last_image` in `first-and-last-frame`, or
    `last-frame-only` is used when B has no fixed opening. Keep lead-ins
    short.
  - **Render order and re-rolls.** Order follows sources, not story.
    Re-rolling C invalidates B, and both are quoted. Re-rolling A
    invalidates any C or cutaway taken off it. A forced re-render counts
    against the dependent shot's three-attempt budget.
  - One line says an LTX-2.5 re-entry works the same way through
    `templates/ltx2/image-to-video`.
  - **Taking a frame off a finished shot** (not in the plan): a one-step
    `get_last_frame` / `get_first_frame` workflow reading
    `video: "output:<shot file>"`. Its still is passed on as
    `{"location": "output:<frame file>"}` or kept as an `asset:`. The plan
    assumed this step without writing it down.
- **SKILL.md step 2.** A shot's `opening_state` is now the `ending_state`
  of its **source**. A per-shot `join` field (`continue` (the default),
  `cut`, `cutaway`, `lead_in`) links `references/joins.md`. Step 5 gets one
  line on source order and re-roll quoting. SKILL.md is 10,115 bytes, under
  the 12 KiB cap.
- **Tests** (`tests/test_plugin_skills.py`): `get_first_frame` and
  `get_last_frame` are in `SKILL_TASKS["script-to-video"]`. The task check
  also accepts `list_tasks()["video_processors"]`, which is the only list
  that holds the frame commands. `TestScriptToVideoJoins` pins the four
  joins, the source rule, "Render the target first", "Re-rolling C
  invalidates B", and that the frame-grab step validates.
- **No README change and no version bump.** `plugins/dw/README.md` lists
  skills, not their references. The plugin's version is the engine's, set
  by the release script.

### Verification (lem, H3 draft tier)

- **Cutaway (M-F261).** A café two-shot, then a rainy-street B-roll cut,
  then C from A's last frame. A's last frame and C's first are the same
  picture, and C is visibly not B's scene. The concat has two `hard_cut`
  seams. `assess_output` raised only audio level steps between unrelated
  shots, which `match_levels` handles.
- **Lead-in (M-F262).** C was rendered before B. B's last frame equals C's
  first, and concat frames 121–126 show no jump. Mid-B the subject drifts
  and turns back, which is the long-FL2VA drift the plan's risks name. It
  happens even at the 124-frame floor, and the shot stays usable.
- **Refusal (M-F263).** A frame grab or a `first-and-last-frame` request
  against a nonexistent `output:` is refused at validate and at
  `run_workflow`, naming the reference, and nothing is queued. A re-roll
  quote of C lists C and B with a sum. `first-and-last-frame` at 960×544 and
  `last-frame-only` are both unpriced (`basis: unknown`). The tester filed
  the missing prices as a server issue and the case's estimate clause as a
  suite amendment on harnest.

## Cost and bounces

| Stage | Bounces | Cost |
|---|---|---|
| A, #1100 | 0 | not recorded on the stage |

## What was deferred, and what would reopen it

- **Engine chain joins** (a per-segment `joins` array in the chain `$def`,
  carries kept by index, dependency render order in `run_chain`, a
  `last_image` continuity class in lockstep with `CHAIN_CONTINUITY_MODES`,
  and a cut-vs-continuation seam attribute through `shots.py`,
  `concat_videos` and `analyze_seams`). Estimated at 4-5 stages, $15-25.
  **Reopen when** a field report shows a chained take (not a shot list)
  broken by an insert, or a session builds the triple by hand across jobs
  more than once and the multi-job quote and re-roll become the complaint.
  Don't start it while #1030 (latent carry) and #1033 (colour-match chained
  segments) are parked undecided: they touch the same loop.
- **A non-adjacent continuity probe.** The A→C match is judged by eye, and
  the engine route would add one.
- **The `shot@<key>` cross-entry reference**: #1101. Don ruled to refuse
  it, and it shipped 2026-10-31 (`complete/for-each-cross-entry-complete.md`).
- **Related improvements:** #1032 (sharpest-of-N last frame) would improve a
  cutaway's re-entry frame. #1067 (colour-match a region to its source)
  could match C's colour to A's tail.
