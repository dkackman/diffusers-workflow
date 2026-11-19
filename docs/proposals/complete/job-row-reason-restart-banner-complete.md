# Job-list row reasons and the restart-batch banner (#1449)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-19). Don filed the idea on 2026-11-18 from a UI support assessment of
v0.19.0..develop. `POST /api/jobs/{id}/rerun-restart-batch` (#1266, #1370) had no UI
caller. In the jobs list a restart tombstone (#1245) was just a `cancelled` chip, and a
failed row didn't say why until it was opened. Plan v2 is the plan comment on #1449,
and it still holds the full sweep. This file records what was built, the deviations,
and what stays open.

## Verdict: build (two stages, one server and one UI)

- **Correction to the idea:** it was not UI-only. `JobSummary` carried no `error` and
  no `server_instance`. `recent_summaries` left both out of its SELECT, and live
  `Job.summary()` had no error. So the client couldn't tell from the list which rows
  were tombstones or which batch they belonged to. Stage A adds the fields.
- **Correction to the idea:** "not yet rerun" can't be read exactly from a list that
  is capped and filtered. The banner's count is a client estimate, and the preview is
  the authority.
- **The row reason is the real win**, for the browser and for MCP `list_jobs` alike:
  triage no longer takes one `get_job` per row. The banner is thinner (no field report
  asked for it), but it is the only UI caller of a verified route, and it cost about
  half a stage.
- **Cheaper alternatives rejected:** UI-only with one `getJob` per closed row (N
  fetches per load, and nothing for MCP), the row reason without the banner, and doing
  nothing.

## Plan history

- **v1 (2026-11-18).** Two stages, A (server) and B (UI). The advisor approved it on
  Don's behalf at 2026-11-19T04:49Z, taking every question's default. Adjustments:
  - no fresh kill -9 tombstone drill, because the existing `drill-1371` tombstones
    suffice;
  - preview before any confirmed batch rerun, and don't confirm if anything H3-family
    is in `would_queue`;
  - neither stage blocks on #1245, #1448 or #1458;
  - the reason line is its own element, so a later #1458 request link is a sibling.
- **v2 (2026-11-19).** Folded in the ruling. No scope change.

**Rulings (advisor, all defaults):**
- Q1 [scope]: server fields (stage A), not N detail fetches from the UI.
- Q2 [none]: the names `reason` and `restart_batch`; first line only, clipped at 160
  with `…`.
- Q3 [none]: no server-side "already rerun" marker.
- Q4 [none]: declare `cancel_requested` on `JobSummary` (a strict-mode fix).
- Q5 [none]: no look gate. Don judges the page when he next opens it.

## What was built

### Stage A (#1472): `reason` and `restart_batch` on job list rows (server)

Merged to `develop` at `92348019` (commit `cff463a8`). The architecture review passed.

- **One helper, `row_reason_fields(status, error, server_instance)`** in
  `dw/server/job_record.py`. Live `Job.summary()`, history's `recent_summaries` and
  history's `_to_detail` all use it.
  - `reason` is the stripped first line of `error`, clipped to 160 characters plus
    `…`. It appears only on failed or cancelled rows that have a non-blank error.
  - `restart_batch` is the row's `server_instance`. It appears only on a cancelled row
    whose error starts with `RESTART_PREFIX` and whose instance is non-null.
  - Each field is absent, never null, when it doesn't apply.
- `recent_summaries` now selects `error` and `server_instance`.
- `JobSummary` declares `reason`, `restart_batch` and `cancel_requested` via
  `sometimes()`.
- The MCP `list_jobs` description gained the plan's clause. The surface budget holds.
- `openapi.json` and the generated UI types were regenerated.
- **Tests:**
  - `tests/test_job_row_reason.py`: the helper; a REST row compared with its detail's
    error; a queued cancel and a running row carrying neither field; a strict-mode list
    with `cancel_requested`;
  - a failed-job case in `test_job_history_parity.py`;
  - in `test_restart_tombstones.py`, a check that a swept batch's `restart_batch`
    equals the preview's `server_instance`.
- **Docs:** `SERVER.md` ("After a restart" step 1, and the `GET /api/jobs` row),
  `ARCHITECTURE.md` job-history row, `RELEASING.md` Unreleased.

**Deviations** (accepted at verify):
1. `get_job` detail also carries `reason`, and `restart_batch` on a tombstone, next to
   the unchanged full `error`. `JobDetail` inherits `JobSummary` and live `detail()`
   spreads `summary()`, so history's `_to_detail` fills the same fields to keep
   live/history parity.
2. `RESTART_PREFIX` moved from `job_history.py` to `job_record.py`, because history
   imports record and not the reverse. `job_history.RESTART_PREFIX` still resolves.

**Bounces: 0.** The first verify on mini-ai passed C-F1064 to C-F1067.

### Stage B (#1473): jobs-list reason line and restart-batch banner (UI)

Merged to `develop` at `be9a0097` (commit `b56f3fc2`). UI only, with no server restart.

- **Row reason:** a `.reason` span, sibling of `.name` inside a new `.namecol` wrapper
  in `JobsPage.svelte`. It is muted (`--muted`, AA in both themes), one line,
  CSS-ellipsized, with the full text in `title`.
- **Banner:** `restartBanners()` in `ui/src/lib/restartBatches.ts`.
  - It groups cancelled rows by `restart_batch` and drops any tombstone that a loaded
    row's `rerun_of` names.
  - There is one banner per batch, newest first, reading "N jobs were cancelled by a
    server restart" with a **Rerun…** button.
  - Banners are hidden under a status filter other than `cancelled`.
- **Click:** `rerunRestartBatchWithConfirm(id)` posts `{}` and opens a dialog. It
  lists the would-queue jobs, each with its estimate, then a total, then the skipped
  jobs with their reasons.
  - Confirm re-posts `{acknowledged_cost: true}` and shows the toast "Queued N; skipped
    K: …". Cancel sends nothing.
  - An empty preview says "Nothing is left to rerun from this server restart." and
    offers **Close** only. The banner for that batch then stays hidden for the rest of
    the session; this is kept in memory, not persisted.
  - The poll never calls the route.
- **Wiring:** `api.rerunRestartBatch`; `RestartBatchRerun` in `types.ts`; the route
  in `UI_READ_ROUTES`; `describeEstimate` and `describeEstimateTotal` exported from
  `plan.ts`.
- **Tests:** vitest in `JobsPage.test.ts` and `plan.test.ts`. A Playwright spec,
  `ui/e2e/restart-banner.spec.ts`, runs against two tombstones that
  `ui/e2e/serve_fixture.py` seeds, sharing `server_instance` `e2e-dead-server`.
- **Docs:** `SERVER.md` Jobs bullet, `ARCHITECTURE.md` "Spending needs consent" row
  (the UI is now the route's second caller), `RELEASING.md` Unreleased.

**Deviations** (accepted at verify):
1. `confirmDialog` gained an `alert: true` option that hides Cancel, for the
   Close-only dialog. Existing callers are unchanged.
2. Every confirm dialog's message now scrolls (`max-height: 60vh`). Only long messages
   are affected.
3. The banner's N ignores the workflow-name text filter, because one batch can span
   workflows.
4. The click sends whichever tombstone of the batch is listed first. Any member
   identifies the batch.

**Bounces: 1 hold, no code failure.**
- **Verify 1 (develop @ `be9a0097`), partial.** C-F1068, C-F1069 and C-F1072's
  unconfirmed path passed. The dialog cases couldn't be exercised because the tester's
  `ui-check` runner refused any POST to the batch route. The advisor handed it back
  once `ui-check` @ `d86a1a6` could stub that POST.
- **Verify 2 (develop @ `37f8ee1e`), passed** C-F1070 and C-F1071.
- **No real non-empty batch was confirmed.** `drill-1371`'s batch was already fully
  rerun, so the real route served only the empty preview. The confirm path was checked
  against a stubbed POST, including its request bodies. A real confirm needs a fresh
  tombstone, which requires a kill -9 of a live `dw.serve` (Don's hands).

Stage costs aren't recorded, because the stage comments name no `usage:` figures. The
plan estimated about $7. Neither stage used GPU time beyond reading existing jobs.

## Deferred, and what brings each back

- **A server-side "already rerun" marker on tombstones** (Q3). Until one exists, the
  banner can show a batch that is already done: one past the 200-row cap, or one whose
  rerun is in a deleted workspace. The click's preview corrects it. Revisit if that
  confuses in practice.
- **A real end-to-end confirm** from the browser, the next time a server restart leaves
  tombstones on a test bed.
- **Overrides on the jobs list** (#1448's cut stage B). If it is revived, it rebases
  onto this stage A's `JobSummary`/`recent_summaries`/parity edits.

## Known limits

- **#1483** (#1458 stage 3, the Requests UI) was blocked on this issue. It edits the
  same `JobsPage.svelte` and puts its request link as a sibling of `.reason` in
  `.namecol`.
- #1245 (tombstones) was still with Don at close-out. If it is reversed, `reason` stays
  and the banner simply never shows.
- The plan's timings weren't measured on mini-ai, and CUDA hasn't seen either stage.
  Nothing in them is accelerator-specific.
