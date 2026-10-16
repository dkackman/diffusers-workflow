# Sweeps and experiment tracking: a written method, not a server feature (#872)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-10-14). The idea came from the H3 quality matrix on lem (workspace
`h3-quality`, 31 rungs, 83 notes). It was run entirely outside dw, with two
local scripts: `tools/ladder.py` built and submitted one workflow per rung and
polled for an idle server, and `tools/push_notes.py` wrote the notes back.
Plan v1 was designed on #872. #890's verify bounce made v2, which Don
approved. It was built as one docs stage, #890. The plan comment on #872
stays the full design; this is the record of what shipped.

This supersedes `sweeps-and-comparison.md` (the U6 scoping notes behind #379,
closed not planned), which is deleted. Its comparison half shipped as
`compare_outputs` (#871, `complete/compare-outputs-complete.md`). Its sweep
half is answered here by a method. Its batch rerun is Q1 below, not taken.
Its server-side `sweep` field design stays in git history if it is revived.

## The problem

"Render this shot under N configurations, one factor at a time, and keep the
results together" is how every quality or speed question gets answered. dw
had no written method for it, so every session rebuilt ladder.py.

## Verdict: build smaller

Every step of the scripts maps onto a tool dw already has. What was missing
was the method, so the feature is one guide section. Cut, with what would
bring each back:
- **A server sweep** (`sweep=` on `run_workflow`, a batch object, a
  multi-job wait). The two things the script fought belong to other issues:
  waiting for an idle server is #869 (dispatch is not host-RAM-aware), and a
  workflow per rung is #865 (H3's accelerator knobs aren't variables).
  *Trigger:* a field report, after #865 and #869 land, that one call per
  variant cost real context or time.
- **Structured notes** (JSON or score fields on `set_note`, filter and sort
  by score). A note holds 4,000 characters, and a first-line convention makes
  a score readable in a listing. The server change is wide: `dw/notes.py`,
  the sqlite `note TEXT` column, eight `api_models.py` fields, the UI note
  editor and pages, eight test files, `list_page`'s read-after-limit, and
  `"note"` already meaning a hint string in six MCP responses. *Trigger:* a
  second experiment needs server-side ranking by score, or #871's grids need
  to order by score.
- **An experiment object.** A workspace already is one.

Non-goals stayed out: server-side batching, typed notes, score filter or
sort, an experiment object, comparison grids.

## Plan history

- **v1:** item 5 said same-seed reruns of an unchanged step are cache hits,
  and the acceptance repeated an *earlier* variant.
- **v2 (approved):** #890's verify showed `dw/step_cache.py` keeps one entry
  per `(workflow, step)`: the latest run only. Item 5 became "only the latest
  run of a step is cached", and the acceptance repeat is of the latest
  variant. Item 2 now says a saved copy keeps the template's `cost` block, so
  `plan.estimate` still has a basis. A step cache keyed per input
  fingerprint was not pursued: it is engine work, and it weighs against
  #368's byte cap.
- **Questions:** Q1 (`arguments` + `note` on `rerun_job`): no, since
  `save_workflow` plus `run_workflow` covers it. Q2: the note convention is
  a first line `score N/5 · judged_by <who>`, then one `key: value` line per
  facet. Q3: the section lives in `WORKSPACES.md`.

## What was built

### Stage A (#890): `## Running a sweep` in `docs/WORKSPACES.md`

Commits 83ba2bd2 (v1 section and test) and 2aecdb5d (v2: latest-run-only
caching, keep the cost block), merged to `develop` at c243b994.
`get_guide("workspaces", section="Running a sweep")` serves it. It covers:
1. one workspace per experiment, with the question and factors in its
   workspace note;
2. a base workflow with every factor as a variable: the template if it
   exposes the factor, else a `save_workflow` copy that keeps the `cost`
   block (and the `{}` note for dtype values);
3. a pinned seed in `arguments`, one factor per variant, and each variant's
   `plan.estimate` quoted for one go-ahead;
4. one `run_workflow` per variant with a `note` whose first line is the
   overrides, and `wait_for_job` before the next (#869: drop that sentence
   when #869 ships);
5. only the latest run of a step is cached, each card's worker has its own
   cache, and a restart empties it, so a re-render is not a failed variant;
6. `list_jobs(workspace=…)` and `get_job` for the results table;
7. `set_note` assessments in the Q2 form;
8. `keep_output` of the winner, and the conclusion in the workspace note.

`tests/test_server_guides.py` asserts the section resolves and names the
tools it tells an agent to call. No MCP or REST surface changed, and
`SURFACE_BUDGET` is untouched.

## Verification and bounces

| Stage | Bounces | What bounced |
|---|---|---|
| #890 | 1 | Verify on mini-ai: C-F469. Going back to an earlier variant (steps=8 after 4/8/12) re-rendered instead of being `reused`, against v1 item 5. The plan was wrong, not the build, so it was re-planned as v2 (2aecdb5d). |

- #890 was verified on mps (mini-ai, develop @ c243b994) against C-F467 to
  C-F470. All four pass. Nothing in the stage is CUDA-specific.
- Cost is not recorded: the stage comments name no `usage:` figures.

## Found along the way

- **A reused workspace name shows old jobs.** `list_jobs(workspace=)` in a
  recreated `regression-sweep` returned six `historical: true` jobs from the
  deleted workspace of the same name. The tester proposed on the harness repo
  that the cases count non-historical jobs only.
- **Note limits, as verified:** 4,000 characters accepted and read back in
  full by `get_gallery_metadata`; 4,001 refused; `list_gallery` clips at
  about 200 with `note_truncated: true`.

## Follow-up

#861, #875 and #862 are each a sweep, and can follow the section instead of
a local script. Pair it with `compare_outputs` (#871) for judging. When #869
ships, remove step 4's wait sentence.
