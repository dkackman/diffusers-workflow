# Requests: a result whose plan or prompt changed after its take reads revised (#1531)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-22). Don filed the issue on 2026-11-21 from his notes verification on lem
(request `wind-turbine-dusk`, workspace `notes`). The agent rewrote board-01's saved
prompt after Don's note ("make it a wheat field"), but the ledger still read board-01
**done** with the grass take, and a dependent shot read **ready** against it. The ledger
had no notion of a stale result. Plan v1 is the plan comment on #1531, and it still
holds the full sweep (file:line). This file records what was built, what was cut and
what stays.

## Verdict: build (three stages), with the cascade cut

- **The case.** The request page and every agent's "what is ready" loop read item status
  off the ledger. A done item stayed done after its plan changed, so dependents of a
  result the user had rejected went ahead, and with video that means a wasted take.
  The evidence was one direct observation; no field report covered it, because the
  ledger was a week old.
- **Don's ruling (2026-11-21), every question at its default:**
  - Q1: `revised` is a new status value, not a flag on `done`.
  - Q2: no cascade to done dependents.
  - Q3: the server measures it (plan edits, then saved-prompt text digests). The
    agent never marks an item.
  - Q4: a revised item whose dependencies are done is `ready`.
- **Cheaper alternatives rejected:**
  - An agent-written `revised: true` field (about $3). It trusts the agent on every edit
    and can't see a prompt edited from the Library page.
  - Stage 1 alone. It misses this issue's own incident, a saved prompt rewritten with
    no ledger write.
  - Doing nothing. That was the behaviour the incident caught.

**Corrections to the proposal** (found by the plan's sweep):
1. "A job records the resolved prompt text" did not hold. The history row keeps
   `arguments` with `prompt:` refs unresolved; the text was resolved only in the worker.
   Stage 2 added the record.
2. A take need not match its item. Variations and reruns with overrides keep the link,
   so "the take's arguments differ from the item's" alone would flag every variation.
   The rule needs a difference **and** a change after the take.

## The rule (derived at read time, never stored)

An item that would read `done` reads **`revised`** when no live succeeded take is
*current* and no acceptance has been recorded since the change. A take T is current when
both hold:
1. **Plan:** the item's `changed_at` is not later than T's `created_at`, or the item's
   `workflow` and `arguments` equal T's.
2. **Prompts:** every `prompt:` reference in the item's current arguments resolves to
   text whose digest equals the one T recorded. A take with no recorded digest (any take
   from before stage 2), or a prompt that no longer resolves, counts as equal.

A revised item keeps its `result` and `attempts`, derives `revised_because` (such as
`["arguments", "prompt:board-01"]`), counts in `remaining_minutes`, and is `ready` when
its dependencies are done. A dependent of a revised item waits, because
`revised ≠ done`. A write that sets `accepted` while the item reads revised records an
acceptance basis (time and current prompt digests), and the item reads done until either
moves again. The basis judges only the accepted take; every other take is judged by its
own recorded digests.

## What was built

### Stage 1: `revised` from plan edits (#1557)

Merged to `develop` as f600834e (650b47fd, 7db35bc3).
- `dw/ledger.py`: the server stamps `changed_at` when an item is created or its
  `workflow` or `arguments` change. A GET→PUT round trip keeps it, and a client-written
  value is ignored. It also stamps the acceptance basis.
- `dw/ledger_views.py`: `is_revised` and `_current` (rule 1), `revised_because`, and
  `remaining_minutes`. The review hold from #1521 derives for `revised` as it does
  for `planned`. #1521 closed, so the hold line shipped.
- `LedgerItem.status` gained `revised`, and `status_counts` gained the key. The
  `api-schema.ts` was regenerated, and the UI got the chip word and tone.
- The `get_request`/`set_request` descriptions, `docs/WORKSPACES.md`, `docs/SERVER.md`,
  `docs/MCP.md`, `docs/ARCHITECTURE.md` and the script-to-video request reference were
  updated.

### Stage 2: `revised` from a saved prompt's text (#1558)

Merged to `develop` at 069cf0ae (975477ff, 0095eb59), then e430b54f (map row) and
4e858f81 (fix).
- The worker records `{prompt name: sha256 of the resolved text}` at inline time
  (`dw/realize.py` `_inline_prompt`), into the run's `manifest.json` under
  `workflow.prompt_digests` (`dw/workflow_run.py`). It hashes the text actually run,
  even when the prompt was edited while the job was queued.
- `request_jobs` hands the views each take's workflow, arguments and digests. The
  history row can't carry the manifest's `workflow` block without a job-history schema
  change, so the digests are read lazily per job behind a bounded cache. A prompt's
  digest is memoized per view by path and mtime.
- Rule 2 is in `dw/ledger_views.py` (`_moved_prompts`, `_recorded_digests`).
- A prompt-driven revision is no ledger write. `save_prompt` adds no revision or history
  entry and wakes no notes-channel event, so an agent sees it on its next `get_request`.

### Stage 3: the request page offers the new take (#1559)

Merged to `develop` as fc893f8f (36691bb7).
- A revised item shows its kept result with the **Revised** chip. A line built from
  `revised_because` says why (for example, "The prompt board-01 changed since take 1."),
  through `revisedLine` and `shownTake` in `itemState.ts`, shown in
  `RequestItem.svelte`.
- **Run take N** sits beside the result and opens the run dialog with the item's current
  plan. **Keep this take** writes `accepted` naming the shown take.
- **Run what is ready** and the attention band include revised items. The UI's own
  dependency checks stay `status === 'done'`, so dependents wait unchanged.

## Cost

The stage comments carry no `usage:` figures, so no measured cost is recorded here. The
plan estimated $13–15 for the three stages.

## Bounces per stage

- **Stage 1 (#1557): none.** Architecture review passed, and the tester verified
  C-F1205–C-F1213 on mini-ai.
- **Stage 2 (#1558): two.**
  - Architecture review bounced on the "Requests ledger" map row. It didn't describe the
    lazy per-job manifest read. The row was rewritten (e430b54f) and the code was kept.
  - The tester bounced C-F1217 step 3 and one clause of C-F1221. After a re-accept, a
    new take on the current text still read `revised`, because `_moved_prompts` applied
    the acceptance basis's digests to every take, not only the accepted one. Fixed in
    4e858f81 with `test_basis_judges_only_the_accepted_take`. The WORKSPACES.md clause on
    the basis and the notes channel was added. Then C-F1214–C-F1221 passed.
- **Stage 3 (#1559): none.** C-F1222–C-F1225 passed through `ui-check` with mocked
  page data.

## Cut, and what brings each back

- **Cascading staleness to done dependents** (Q2). A dependent would have to know which
  take of its dependency it consumed, and no take records that reliably. Reopen it if a
  take starts recording its inputs' job ids (e.g. through `output:` references), or if
  a field report shows a stale dependent shipped.
- **A saved workflow's file changing under the same name.** Catalog and builtin files
  change mtime on every deploy, so this would flag everything. Workspace-saved workflows
  could follow if asked for.
- **The agent-marked flag** (Q3 alternative). It isn't needed while the server measures.

## Known limits

- **The board-02 half of the incident is not covered.** Board-02's own prompt never
  changed; the agent didn't carry Don's note over to it. No measurement can see that,
  so it stays the agent's job.
- Takes from before stage 2 carry no digest and are never revised by a prompt edit.
- Deleting a referenced prompt makes no claim. The item stays as it was.
- An external client that switches exhaustively on item status sees a seventh value.
  None exists in the repo outside `ui/`.
