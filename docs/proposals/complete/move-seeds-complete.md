# All 13 camera moves measured on two seeds (#1260)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-16). Plan v2 approved by Don on 2026-11-15 at the defaults; two
stages, #1379 (H3) and #1380 (LTX-2.5), both verified on 2026-11-16.

## The idea

Stage B of #1200's plan, which Don declined to fund under #1200 (no lem
time) and filed as its own parked idea. The `cinematography` skill hands 13
move ids to driving sessions, but only the six #1148 measured had a
verdict, and those rested on one seed. The other seven (`truck-right`,
`crane-down`, `tracking`, `whip-pan`, `push-in`, `handheld`, `crash-zoom`)
read `not measured`. Don unparked it on 2026-11-15.

## What the design found

- **#1148's protocol carries over** (`audits/2026-11-higgsfield/move-trial.md:14-44`):
  the telephone-box subject at `wide`, each family's template at its
  defaults with only `prompt` and `seed`, the move sentence verbatim from
  the compile table.
- **`tracking` and `handheld` need a subject that moves.** Both sentences
  follow `<subject>`, and the box doesn't move, so those two rows use a man
  in a dark coat walking past the box.
- **Placeholders filled for a box:** `whip-pan` pans left onto the stone
  fountain (clear of the whip-pan "right" ban), `push-in` onto the glass
  door, `crash-zoom` on the box; LTX viewpoint "a front-facing angle".
- **Seed 42 on the six measured moves is #1148's jobs, reused**, not re-run.
- **The issue's 1.5 GPU-h overstated it.** #1148's own timings put a clip
  at about 1.6 min (H3) and 1.1 min (LTX) warm; the plan expected ~70 min.

## Decisions (Don, 2026-11-15)

- **Q1 [spend]:** up to 1.5 lem GPU-hours, at most 48 clips, no downloads,
  stage A held until #1351 closed. Yes.
- **Q2 [none]:** the walking man for `tracking`/`handheld`. As written.
- **Q3 [none]:** cell format `<seed 42>, <seed 7>`, column
  `measured (#1148, #1260)`. As written.
- **Scheduling:** no overlap with #1325's probe or #1340's revert window;
  H3 never alongside H3. On 2026-11-16 Don ruled stage A run on diffusers
  0.44.0.dev0 @ `43e30028`, and #1340 then closed with no revert.

## What was built

The record is `docs/proposals/audits/2026-11-move-seeds.md`: the sentences
as run, a job id per cell, frame notes for every new clip, reruns, limits,
and `get_diffusers_state` before and after each stage.

**Stage A, #1379 (H3).** Commit `07cbb4cc`, merged as `1b891f98`. 20 new
clips, 26.9 GPU-min, run one at a time. `compile-h3.md`'s column is
`measured (#1148, #1260)` with two verdicts per cell.

| move | H3 seed 42, seed 7 |
|---|---|
| `static`, `dolly-in`, `dolly-out`, `crane-up`, `crane-down`, `orbit`, `push-in`, `crash-zoom` | honoured, honoured |
| `truck-left`, `truck-right`, `whip-pan` | partial, partial |
| `tracking` | partial, honoured |
| `handheld` | ignored, partial |

No H3 move was ignored on both seeds, so no H3 rerun.

**Stage B, #1380 (LTX-2.5 and the skill).** Commit `cefa1e96`, merged as
`35332374`. 20 new clips plus one rerun, 21.0 GPU-min. `compile-ltx-2.5.md`
gets the same column; `cinematography/SKILL.md`'s table grows to all 13
moves × H3/LTX/K6 and its prose speaks of two seeds; `RELEASING.md`,
`move-class-complete.md` and `todo.md` updated.

| move | LTX seed 42, seed 7 |
|---|---|
| `static`, `dolly-in`, `dolly-out`, `truck-left`, `crane-up`, `orbit`, `push-in` | honoured, honoured |
| `truck-right` | partial, honoured |
| `crane-down` | ignored, honoured |
| `tracking` | ignored, partial |
| `whip-pan`, `crash-zoom` | partial, partial |
| `handheld` | ignored, ignored: hint only, rarely honoured |

The `handheld` rerun (`2305175384ca`, a revised "the shot is handheld, the
camera shaking with small jolts…" sentence) was also ignored, so the
original sentence stays and the move stays in the vocabulary as a hint.

**Total:** 40 new clips plus one rerun, 47.9 GPU-min against the 90 cap.
No downloads, no test changes, no engine, MCP, REST or syntax change. K6
cells unchanged. Plugin-only; nothing deployed.

## Verification and bounces

| Stage | Bounces | Verified by |
|---|---|---|
| A, #1379 | 1 | C-F899 to C-F905. The bounce: the hand-off didn't quote the per-cell table (job ids, frame notes); re-handed off with the table, no code change. |
| B, #1380 | 1 | C-F906 to C-F913. The same bounce, fixed the same way. |

Both bounces were the record, not the build: the tester can't read source,
so a measurement stage's hand-off has to carry its full table.

Session cost isn't recorded here: the stage comments name no `usage:`
figures.

## Deferred

- **Rates.** Two seeds are two samples; no cell claims a rate.
- **LTX `handheld`** gave no shake on three clips. A better phrasing, or a
  shake applied outside the model, is left open.
- **The walker leaves frame** on LTX `tracking`/`handheld`; a subject
  confound noted in the audit's *Limits*, not separately probed.
- **Benching counters on `push-in`/`handheld`** stays with #1225.
- **Kandinsky 6** cells are still #1148's one seed (non-goal).
