# Rerun a restart's tombstones as one batch (#1266)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-15). The idea came out of #1220, whose plan v1 cut automatic resume after
a restart. #1245 shipped restart tombstones instead (`6db8d963`): each admitted
job gets a row in `jobs.sqlite`, and at the next startup the jobs a dead server
held are closed as `cancelled`, with an error that starts `server restart:`.
Don unparked #1266 on 2026-11-15 and asked for the explicit bulk action. Plan v2
is the plan comment on #1266 and still holds the full sweep and design. This file
records what shipped and what stayed cut.

## The problem

After an unclean death you couldn't find a dead server's tombstones by listing:
`JobSummary` carries neither `error` nor `server_instance`, and listings stop at
20 rows (MCP) or 200 (server). The dead instance's id was only in a log line.
Each tombstone had to be rerun by hand, and nothing stopped a retry from queueing
a job twice. No field report asked for this, and the issue's revisit trigger
never fired. It was built on Don's ruling, as insurance for overnight and bench
queues.

## Verdict: build smaller

- **Built:** a batch named by any one of its tombstones, over REST and as a
  `rerun_job` parameter. It previews without consent, reruns in `created_at`
  order through the rerun path, and skips members that were already rerun.
- **Cut: any UI.** The Jobs page can't tell a tombstone from a user cancel,
  because `JobSummary` has no `error`. *What would bring it back:* a report
  from someone using the browser.
- **Cut (kept cut from #1220):** resume at startup, a `--resume-queue` flag, and
  restoring the `move_job` order. Dispatch reorders starts anyway
  (`dw/server/dispatch.py`).
- **Not covered:** rows from before #1245 (`server_instance` NULL) are refused
  as not a tombstone.

## Plan history

- **v1 (2026-11-15).** Q1 [scope] a new route plus a `restart_batch` parameter,
  with the batch named by a tombstone; Q2 [default] each member reruns in its own
  workspace; Q3 [default] without `acknowledged_cost` the call answers a preview;
  Q4 [loosen] positive-path cases left pending until a real unclean death.
- **v2 (approved 2026-11-15).** Q1-Q3 ruled as designed. Q4 ruled otherwise: a
  staged `kill -9` drill on mini-ai at stage B's hand-off makes real tombstones,
  so the positive path is real acceptance. The same drill is #1245's verify.

## Correction to the issue's sketch

"Rerun the jobs from server instance X" couldn't work as written. No route
exposes an instance id. Instead, **any tombstone names its batch**: every row
with the same `server_instance` and a `server restart:` error. No instance id
became input surface. The batch reply does echo `server_instance`, and that is
the only place an instance id is visible over MCP.

## What was built

### Stage A (#1370): the batch over REST

Commit `e14ad822`, merged to `develop` as `dc38380f`.

- `JobHistory.restart_batch(job_id)` (`dw/server/job_history.py`) returns `None`
  unless the id is a tombstone (`cancelled`, a `server restart:` error, and a
  non-NULL `server_instance`). Otherwise it returns every such row of that
  instance, `ORDER BY created_at, id`, with no row cap. `JobHistory.rerun_ids`
  plus a `jobs_rerun_of` index answer "has this been rerun".
- `JobManager.rerun_restart_batch` (`dw/server/jobs.py`) runs under its own
  `_batch_lock`. It takes members oldest first and skips three kinds: one already
  rerun (`already rerun as <id>`, checked against both history and live jobs),
  one whose workspace was deleted, and one the admission check refuses (with that
  check's message). It queues the rest through `rerun()`, so each carries
  `rerun_of` and has its dispatch needs computed fresh. One member failing never
  stops the rest. The single `/rerun` route's admission moved into a shared
  `_admit_rerun` helper, and its behaviour didn't change.
- `POST /api/jobs/{job_id}/rerun-restart-batch`, body `{acknowledged_cost?}`,
  answers as follows:
  - an unknown id gets 404;
  - a non-tombstone, a NULL-instance row or a live job gets 400 "… is not a
    restart tombstone";
  - a bound acknowledgement gets 400, and that check runs before the tombstone
    check;
  - any other call answers 200 with `{server_instance, queued, skipped}`.

  The preview adds `would_queue: [{id, workflow, workspace, estimate}]`, with
  `queued: []`. Models are in `api_models.py`, the OpenAPI document and
  `api-schema.ts` were regenerated, and the route isn't in `UI_READ_ROUTES`.
- Tests are in `tests/test_restart_tombstones.py`. They cover:
  - members come oldest first and each rerun carries `rerun_of`;
  - a repeat call skips everything, and so does one after another restart;
  - two concurrent calls queue each member once;
  - a member in a deleted workspace or refused at admission is skipped, and the
    rest are queued;
  - the 400s and the 404;
  - the preview queues nothing;
  - a live sibling's rows are never members.
- Docs: the Jobs paragraph in `docs/SERVER.md`, and the Job history row in
  `docs/ARCHITECTURE.md`.

### Stage B (#1371): `rerun_job(restart_batch=true)`, and the drill

Commit `57c2335c` merged as `2f8bd3e3`, then the map fix `ef048df7` merged as
`002fc3ad`.

- `rerun_job(..., restart_batch: bool = False)` (`dw_mcp/tools_jobs.py`,
  `dw_mcp/diagnose.py`) posts to the batch route. With `restart_batch`,
  `new_seed` and `arguments` are refused before any request is sent
  (`RESTART_BATCH_OVERRIDE_REFUSAL`); `arguments={}` counts as no override.
  Without `acknowledged_cost`, the batch form passes through and returns the
  server's preview. The single form still refuses on the client. No new tool, and
  the surface budget holds.
- Docs: a `### After a restart` section in `docs/SERVER.md` under *Jobs API*, as
  the description's pointer target (`get_guide("server", section="After a
  restart")`; no guide is named "jobs"). `docs/MCP.md` gets the `rerun_job`
  row, the cost-gate row and the troubleshooting row.
- **Architecture-review bounce (1).** The *Spending needs consent* row in
  `docs/ARCHITECTURE.md` still said every `acknowledged_cost` tool refuses until
  the flag is set. For this form the gate is the server's preview now, and the
  amended row says so.
- **The drill (Q4).** In the scratch workspace `drill-1371`, the session queued
  four noted `templates/text-to-image` jobs. It then ran `kill -9` on `dw.serve`
  (pid 62245, owned by `don`), and launchd respawned it as pid 62906 within 3 s.
  No sudo was needed. The kill left one running tombstone (`8d8cc7ba497f`,
  "running when the server stopped, before it opened a run") and three queued ones
  (`31531e28c813`, `34c47920676f`, `6196d06350c5`, "queued, never started"). The
  workspace and tombstones are kept as #1245's evidence.

Cases verified on mini-ai (MPS) at `002fc3ad`, together with #1245 per Don's
ruling:
- C-F859 to C-F861 (stage A, reached through stage B);
- C-F862 to C-F868 (stage B).

The preview gave the same four members in order, whichever tombstone named the
batch. With consent it queued four reruns in order, each carrying `rerun_of` and
landing in `drill-1371` while another workspace was pinned, and all four
succeeded. A repeat queued nothing and listed each member as "already rerun as
<id>".

## Cost and bounces

| Stage | Bounces | Cost |
|---|---|---|
| A (#1370) | 0. Its verify was blocked, not failed: the route is reachable only through B, so Don flipped the blocker link and A verified with B | Not recorded on the stage. The plan estimated ≈ $5 |
| B (#1371) | 1 architecture-review bounce (the consent row in the map), fixed docs-only. 0 tester bounces | Not recorded on the stage. The plan estimated ≈ $3 |

## Deferred and open

- **UI for tombstones:** cut, with the trigger above.
- **A stage only reachable through a later one** (stage A here) has to verify
  with that stage. Decomposition ordered it A → B and set a blocker link that
  deadlocked verification. Don flipped it. A future plan whose first stage has no
  MCP path should say up front that the stage verifies with the later one.
- **Tool count.** C-F863 hard-coded 65 tools, and the client listed 63. Stage B
  added no tool. The tester filed a suite amendment on harnest to drop the
  hard-coded count.
- **Repeated work.** A rerun of a job that was running redoes its finished steps,
  because the step cache doesn't survive a restart (#281/#283).
- **No cap on N.** A large overnight batch is guarded only by the preview's
  estimate.
