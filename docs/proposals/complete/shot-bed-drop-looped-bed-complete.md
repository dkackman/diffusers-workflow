# Bed drop detection and a looped bed in `assemble-and-score` (#1620)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-26). Plan v4 on #1620 is the full design. This file records what
shipped, what was left out, and why.

**Result:** `assess_output` now warns `shot_bed_drop` when a voiced shot's
noise floor sits more than 20 dB below its louder neighbour's: the hole that
passed in "Whitefish Point". `assemble-and-score` lays a looped bed, sliced
from the cut's own sound, with three variables. That replaces the hand-built
`slice_audio` → `loop_audio` → `mix_audio` chain. `slice_audio` gains
`past_end: "excerpt"` for a window cut out of something longer.

## The problem

From #1523's field report ("Whitefish Point", lem, item 9): H3 Ref2VA dialogue
shots in a storm set dropped to about -52 dBFS between lines, while the bed
elsewhere sat near -19 LUFS. `assess_output` and `check-script` passed the cut,
because `shot_dead_air` fires at an absolute floor
(`DEAD_AIR_FLOOR_DBFS = -65.0`). The user caught it by ear and fixed it with a
hand-built loop bed. Earlier beds (#465, #218, #544/#545, #1118) were built by
hand the same way, and #1622 showed hand-copies of `assemble-and-score`
spreading.

## What shipped

### Stage A, #1718: `shot_bed_drop` (server)

- **The field.** `analyze_shots` emits per shot `floor_below_neighbours_db`:
  the louder adjacent shot's `noise_floor_dbfs` minus this shot's, not clamped
  at 0. It is null for a cut of one shot, and where the floors it needs are
  null.
- **The rule.** `shot_bed_drop` in `dw/assessment_rules.py` fires above
  20 dB, at severity warn. It takes `shot_dead_air`'s quiet guard (skipped at
  shot RMS ≤ -30 dBFS) and is skipped in a `silent_by_design` shot. Its `says`
  names the remedy and tells the agent to leave a deliberate scene change
  alone.
- **Unchanged:** `shot_dead_air` and `DEAD_AIR_FLOOR_DBFS`, which the
  hallucination guard, `script_take` and `script_check` share.
- **Docs:** the rule tables in `docs/TASKS.md` and `docs/WORKFLOW_GUIDE.md`,
  and `minimax-h3/references/watching-a-run.md`.

### Stage B, #1719: the bed variables (server + plugin)

- **Template.** `assemble-and-score` gains `bed_start_seconds` 0.0,
  `bed_seconds` 1.0 and `bed_gain` 0.0 (float literals, per Don's ruling).
  Three steps go before `mixed`:
  - `bed_slice`: `slice_audio` of `previous_result:edit`, `past_end: "excerpt"`;
  - `bed_resampled`: to `sample_rate`;
  - `bed`: `loop_audio` to `total_frames` at `fps`, 250 ms crossfade.

  `mixed` gets a third entry, `previous_result:bed` at `variable:bed_gain`. At
  gain 0 the film's audio is bit-identical to the old two-track mix (same
  `pcm_sha256`). Cost stays 0.2.
- **`slice_audio` `past_end: "excerpt"`** (Q5). It cuts at the source's end
  with no padding, logs it, and refuses a start at or past the end. It raises
  neither `slice_dropped_loud_tail` nor `slice_trimmed_tail`. `"stop"` is a
  section kept whole, and `"excerpt"` is a window out of something longer.
  Validate's static slice checks (`reference_limits.py`,
  `slice_preflight.py`) treat it like `"stop"` through a shared
  `SLICE_STOPS_AT_END`.
- **Failure names the step.** From C-F1403's bounce: a run failure raised
  inside a step is labelled with the step's name once, by `run_step`
  (`dw/workflow_run.py` `note_failed_step`/`failed_step_label`, composed in
  `dw/worker.py`). A validate refusal of a task argument opens with
  `Step '<name>': ` and ends with `(set by variable '<name>')`. The
  hand-written `Step '{self.name}': ` prefixes in `pipeline_h3.py`,
  `pipeline.py` and `previous_results.py` were removed: they named the model
  id, not the step, and would have doubled the label. `docs/ARCHITECTURE.md`'s
  *Failed-run reporting* and *Task argument domains* rows name the rule.
- **`shot_bed_drop`'s `says`** now names the `bed_*` variables, copied from a
  `find_loop_bed` pick.
- **Skills** point at the variables instead of the hand-built chain:
  `series-episodes/SKILL.md`, `series-episodes/references/dialogue-mix.md`,
  `long-form/references/audio.md` and `minimax-h3/references/cuts.md`.
- **Tests:** `tests/test_slice_excerpt_bed.py`, the step chain in
  `test_stills_to_video.py`, the catalog budgets (raised with a dated comment),
  and the plugin-skill pins.

No new MCP tool, REST route, workflow syntax or dependency. The new surface is
one rule name, one `analyze_shots` field, three template variables and one
`past_end` value.

## Left out, and what would bring it back

- **A within-shot relative gap check.** By level alone, a storm hole (lines at
  -19, gaps at -52) looks like a dry room's pauses (lines at -20, room tone at
  -55); only the neighbours tell them apart. Reopen it if a field report shows
  a hole in a cut where every shot is a dialogue shot.
- **A bed from an outside ambience asset** (Q2). A `bed` asset can't default to
  "none" without a new engine concept (a null input that silently does
  nothing). Reopen it if a field report needs ambience that is in no shot.
- **#1578 check 4** (level jump between neighbours). It is already covered by
  `seam_level_step` and `shot_level_spread`. Per Q3 a comment on #1578 says so,
  and #1578 stays parked for Don.

## Rulings

Don approved v2 on 2026-11-25 (Q1-Q4 at their defaults, with a checkpoint
before B), the build of B on 2026-11-26 (float defaults), and v3 on
2026-11-26 (Q5 at its default), each "approve with delegate recommendations".

## Known limits

- The 20 dB threshold comes from one field case, and a deliberate scene change
  (a storm exterior cut to a quiet interior) can trip it. It is a warn.
- The bed's three steps run on every run of the template, doing nothing at
  gain 0.
- A bed window holding speech brings back #1118's problem. Choosing it with
  `find_loop_bed`, which rejects tonal windows, is the guard, and the skills
  say so.

## Stages, bounces and cost

| Stage | Bounces | Notes |
|---|---|---|
| #1718 `shot_bed_drop` | 1 (tester) | C-F1400 call 3: the WORKFLOW_GUIDE rule row. Fixed in docs (`3ff38741`); re-verified at `5ca7a7ec`. |
| #1719 bed variables | 1 tester, 2 architecture review; 1 re-plan before building | The first build session stopped: v2's `past_end: "stop"` would warn `slice_dropped_loud_tail` on nearly every run, so plan v3 added `"excerpt"` (Q5). The tester bounced C-F1403: an impossible bed window failed without naming the bed's step or variable. The fix labelled every step failure, and the two review bounces removed the hand-written labels it duplicated (`pipeline_h3.py`/`pipeline.py`, then `previous_results.py`). Verified at `acb3c237`. |

The stage comments don't name `usage:` figures, so cost isn't recorded. The
plan estimated $3-5 for A and $4-7 for B.

Verification (lem, workspace `regression-complete`): C-F1394-C-F1400 (A) and
C-F1401-C-F1405, C-F1446-C-F1453 (B) pass.
