# Task-only workflows priced from declared task rates (#1619)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-26). Plan v1 on #1619 is the full design. This file records what
shipped, what was left out, and why.

**Result:** a task-only workflow with no history now gets a price. Each
non-model media command declares a rate on `register_command`, and
`validate_workflow` scales it by the input's length, probed at validate time.
The answer is `basis: "derived"` with `low_confidence: true`, where it used to
be `minutes: null`, `basis: "unknown"`. Observed history still wins once the
workflow has a successful run.

## The problem

From #1523's field report ("Whitefish Point", item 7): `validate_workflow` on
`fitz-light/cut` (only task steps: assemble and score, `apply_lut`,
`film_grain`, `export_timeline`) answered `minutes: null`, `basis: "unknown"`.
Its first run had failed (#1613), so there was no history, and the agent had
to ack `{"minutes": null}`. It couldn't tell the user whether the cut would
take seconds or an hour. #948 had already added `unpriced_reason`, which
explains the gap but doesn't fill it.

The issue offered three options: a per-task floor, a rate scaled by input
length, or a failed run's minutes as weak history.

## What shipped (stage 1, #1698)

- **`register_command(cost=...)`** (`dw/tasks/registry.py`). The value is
  `{"fixed_seconds": F, "per_input_second": R}`: exactly those two keys,
  non-negative numbers. It is read as `RegistryTable("cost")` and refused on a
  `consumes_device=True` command. Rates are about 2× the slowest rate measured
  on lem's fitz-light-cut runs (jobs 80d8a9044533 and 3d049d5dbe6a):

  | Command | F (s) | R (s per input-s) |
  |---|---|---|
  | `apply_lut`, `film_grain` | 2 | 1.5 |
  | `export_timeline` | 2 | 0 |
  | `concat_videos` | 2 | 0.5 |
  | `pair_audio` | 2 | 0.35 |
  | `normalize_audio` | 1 | 0.1 |
  | `fade_audio`, `slice_audio`, `resample_audio`, `mix_audio` | 1 | 0.05 |

  Device-consuming commands (`transcribe_audio`, `upscale`, `separate_stems`,
  judges) declare nothing.
- **Input length at validate time** (`dw/plan_pricing.py` `task_rate_price`).
  Each media argument is probed with `probe_metadata`. Media is decided by
  `dw/locations.py` `is_media_argument`, the one rule the location check uses
  too. A list argument is summed. A `previous_result:` input takes the longest
  length any step's inputs resolved to. A step with nothing resolved is priced
  at F alone. Either fallback names the step in `differs`.
- **Own-price slot only** (`dw/plan.py` `estimate`): used when every step is a
  task with a declared cost, and the workflow has no `cost` block and no
  inherited figure. Elided steps aren't priced, because they won't run.
- **The answer.** `minutes` is the sum, rounded up to 0.1. `scaled` reads
  `task rates: <cmd> F s + R s/input-s over N s; ...`. There is no
  `unpriced_reason`.
- **Still unknown** when any command declares no rate. `unpriced_reason` then
  starts `no declared rate: X, Y` and names only those commands.
- **Surfaces.** No new MCP tool, parameter, REST route, workflow syntax or
  `basis` value. Only REST description text changed (`unpriced_reason`,
  `scaled`, `differs`), with OpenAPI and `api-schema.ts` regenerated. Docs:
  `docs/SERVER.md`, `docs/TASKS.md`, `docs/ARCHITECTURE.md` (the estimate,
  adding a task, and locations rows), `docs/proposals/1355-task-only-lane.md`.
- **Tests:** `tests/test_task_rates.py`, and `tests/test_inherited_cost.py`
  updated for the new reason text.

## Left out, and what would bring it back

- **(C) Failed runs as weak history.** This would mean loosening
  `finished_runs`' SUCCEEDED filter for every workflow, and a failed run
  undercounts the work (#1613 failed at export). Reopen it if a command keeps
  failing before a rate can be set for it.
- **(A) Per-command rates learned from any workflow's step times.** Step times
  aren't stored, so this needs a new store and at least one more stage. Reopen
  it if field reports show the declared rates more than 2× off on a host.
- **Task steps inside a pipeline workflow**, model-backed tasks, and a
  first-run term are non-goals.

Don's delegate approved plan v1 as written on 2026-11-25 (Q1–Q4 at their
defaults).

## Known limits

- The rates come from one host (lem). A slower CPU quotes low until its first
  success, and `low_confidence` says so.
- On mps the declared rate stays the quote until #1686 lands (observed history
  never matches there).
- The estimate keeps its own per-path probe cache. It can't reach the one
  validation uses.
- `plan_pricing._task_steps` is a bit stricter than
  `observed_cost._is_task_only`: it also needs each `task` to be a dict.
  Merging the two would change how history is bucketed.
- Inline workflows (`workflow_name: null`) record no history, so they keep the
  derived quote after a run. Only a saved workflow moves to `observed`.
- Cosmetic: on a tempered estimate after one run, the task-rate figure is
  labelled `curated_minutes`, and `scaled` keeps the task-rate text.

## Stages, bounces and cost

| Stage | Bounces | Notes |
|---|---|---|
| #1698 declared task rates | 1 (architecture review) | The review found a second copy of the media-argument rule in `plan_pricing` (it dropped `PIPELINE_MEDIA_ARGUMENTS`), and an architecture map that hadn't been updated. Fixed by `locations.is_media_argument` and map updates. Tester verified on the first pass. |

The stage comments don't name `usage:` figures, so cost isn't recorded. The
plan estimated $4–6.

Verification (develop @ 065d183c, lem, workspace `regression-complete`):
C-F1302, C-F1303, C-F1305, C-F1306, C-F1307 and SE-F083 pass. C-F1304's
wording, which validates an inline workflow again, didn't match the plan's
item 3. The saved-workflow form passed (`observed`, 0.5 min). The case
amendment is dkackman/harnest#302.
