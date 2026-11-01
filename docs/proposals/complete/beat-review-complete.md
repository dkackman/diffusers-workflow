# Beat review: beat authoring and the beat review in the story skills (#1122)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-01). Plan v2 approved by Don on 2026-11-01 as "build smaller"; one
stage, #1125 (A), verified. The plan comment on #1122 is the full design;
this doc records what shipped against it, and keeps the deferred stage B
design so it can be approved later without a new sweep.

## The idea

A clip-level beat gate, from story-forge (MIT,
`nicedreamzapp/story-forge`: `pipeline-tools/beat_gate.py`, `LESSONS.json`;
assessed 2026-11-01). A vision-language judge describes a shot's frames
blind, then gives a verdict against the shot's script beat, with a majority
over sampled frames, a distinct rejection reason and a correction for the
next attempt. dw confirmed each line is *heard* (`check_script`); nothing
confirmed each beat is *seen*. story-forge's founding failure: an episode
that passed 86 of 91 automated checks while the bear poked a wall with a
stick instead of shouldering the door open.

## Decisions

- **Plan verdict: build smaller.** In story-forge a script reviewed its own
  output, so a local VLM was the only eye. In dw the driving agent is
  already a strong vision model reading frames with `get_output_frames` in
  "Run and judge". What carries over is **how** to judge and **how to write
  a beat that can be judged**, which is skill text. The engine task (stage
  B) was deferred until evidence shows the agent's own review misses beats.
- **Q1:** build smaller, stage A now, stage B deferred with its return
  triggers. **Q2:** the shot-sheet fields are `beat` and `must_not`.
  **Q3:** a pointer only in `series-episodes`, no restatement.
- **Size-cap amendment (Don):** `script-to-video/SKILL.md` was at 10,115 of
  its 12,288-byte `SKILL_SIZE_LIMIT`, so the rules, the protocol and the
  audit paragraph went in a new `references/beat-review.md`; `SKILL.md`
  carries only the two fields and a pointer, plus a size-margin pin.
- **Correction to the proposal:** the issue proposed a VLM task first; the
  plan made skill guidance first. SmolVLM-256M, dw's default vision model,
  is too weak for the verdict in any case.

## What was built

**A, #1125: beat authoring and the beat review (plugin-only).** Develop
957a7c53, merged 84e8edde (branch `feat/1122-A-beat-review`). No engine,
MCP, REST or syntax surface changed; no `lem` deploy.

- `plugins/dw/skills/script-to-video/SKILL.md` §2: shot-sheet fields
  `beat` (one visible state a single frame can show) and `must_not`
  (physically or story-impossible things to rule out), after
  `opening_state` / `ending_state`. "Run and judge" points to
  `references/beat-review.md` before the keep and still names
  `get_output_frames`. 10,504 of 12,288 bytes after the change.
- `plugins/dw/skills/script-to-video/references/beat-review.md` (new,
  4,147 bytes):
  - *Writing a beat:* who, doing what, to what, where, in a state one frame
    proves. Four banned kinds, each with a rewrite: motion over time,
    intent, off-screen facts, emotion words.
  - *Before anything renders:* every edit-list shot has a beat and every
    beat a shot; nothing renders while one is missing.
  - *The review:* 3 evenly spaced frames per clip, 5 only on a split
    verdict, the image itself for a still; describe blind before the
    verdict (a fresh subagent without the beat where the client has one);
    one verdict per frame against `beat` and `must_not`; pass on a
    majority; exactly one reason per rejected frame from `beat_not_shown`,
    `must_not_present`, `identity_drift`, `text_in_frame`,
    `reference_leak`; one concrete correction into the next prompt, within
    the three-attempt budget.
  - *Audit the auditor:* a human overrule is appended to the workspace note
    with `set_note("workspace:<name>", ...)`.
- `plugins/dw/skills/series-episodes/SKILL.md` "Run and judge": a pointer
  to `dw:script-to-video`'s beat review per episode shot.
- Pins: `tests/test_plugin_skills.py::TestScriptToVideoBeatReview` (fields,
  pointer, majority, the five reasons, the four banned kinds, 3/5 frames,
  describe-before-verdict, budget, no-render-without-a-beat, audit
  paragraph, the series pointer with no restated reasons) and
  `SCRIPT_SKILL_MARGIN = 1024` (SKILL.md stays at least 1 KB under the cap).

Deviations from plan v2, accepted at verify:
- **No plugin version bump.** `plugin.json`'s version is pinned to
  `pyproject.toml`'s and only the release script moves both.
- **The series pointer is written `script-to-video/references/beat-review.md`**,
  since the link test resolves a bare `references/...` against the
  linking skill's own directory.

## Bounces per stage

| Stage | Bounces | Cause |
|---|---|---|
| A, #1125 | 0 | Verified by reading the served plugin text against the acceptance list (no GPU run, as the plan approved). |

No stage comment names `usage:` figures, so per-stage session cost is not
recorded. The plan estimated about $2-3 for stage A.

## Not built, and why

### Stage B: a `beat_gate` task (deferred, Q1)

Designed so it can be approved as is:
- A new command, `returns="json"`:
  `{verdict, pass_fraction, votes[{frame, description, pass, reason}], correction}`.
- Two `generate_text` calls on one cached load: describe with no beat, then
  a verdict from the description plus the beat.
- Frames from `evenly_spaced_indices`, shot windows from
  `assess.resolve_shots` / `sample_span`.
- Beats from a new `beats: [{shot, text, must_not?}]` argument, not `lines`
  (`dw/script_lines.py` refuses unknown keys there).
- Default model `Qwen/Qwen3-VL-4B-Instruct` (templates already load it),
  declared via `required_weights`, `release_models` before the generator.
- `pass_fraction` feeds `select first_above 0.5` in a beat-gated variant of
  `best-of-n-to-video`.
- Opens with a no-code lem probe: score a calibration set of Don's kept and
  rejected shots with Qwen3-VL-4B through `text_generation`; if 4B can't
  match the human verdicts, the task isn't worth building at that size.
- Its cost: a task and docs, a `beats` argument shape, a template against
  the compact catalog budget, VLM memory no task-level VRAM mechanism can
  declare (`dw/vram_inheritance.py` projects pipeline steps only), and a
  calibration set to keep true. Estimated $8-12.

*Comes back if:* a field report shows the agent's own beat review passed a
shot that missed its beat; or Don asks for unattended best-of-n judged on
beats; or a run is long enough that frame review by the agent costs more
context than a local judge would.

### Also out of scope

- A beat linter in code: the skill states the rules and the agent applies
  them.
- In-workflow branching or retry (no `when`/`retry`, by design), a new MCP
  tool (surface at 15,052 of 16,000 tokens), and any change to `judge`'s
  scalar contract.

## Known edges

- The review relies on the agent's discipline; nothing enforces it, as with
  the rest of "Run and judge".
- "Blind" isn't truly blind without a subagent: the agent wrote the beat.
- Frames add context cost over a long episode, hence 3 by default.

## Overlap left as it was

#1123 (LTX-2.5 and prompt lessons, same source, open) owns model-specific
prompt wording. This feature owns the beat-as-visible-state rule and the
review protocol; #1123 should point to `references/beat-review.md` rather
than restate them.
