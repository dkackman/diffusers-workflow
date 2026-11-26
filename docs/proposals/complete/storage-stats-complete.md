# Read-only storage stats: job history and logs (#1457, stage A)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-26). Plan v1 (design) deferred the feature; Don unparked it on
2026-11-26 on the deferral's third trigger ("Don wanting the read-only stats
line") and ruled **stage A only**. Plan v2 records that ruling. One stage,
#1766, built, verified on mini-ai (mps) and closed.

## The idea

The 2026-11-18 UI support assessment found the old maintenance-screen
proposal (`docs/proposals/maintenance-screen.md`, since pruned in 1958f831)
still open past its phase 0 (WAL mode): no view of the job-history
database's size and row count, no pruning, no output-directory totals, no
log line, and a model-cache summary.

## What the sweep found (plan v1)

- No field report asked for it, and no issue in the class needed one-off
  fixes. The disk-usage regressions (#177, #259, #313, #849) were a stale
  usage cache, since fixed.
- Real use is small: tens of MB of SQLite for a year of heavy use; the log
  caps itself at about 400 MB (`dw/log_setup.py`, `backupCount=7`).
- Much of the proposal had already shipped elsewhere: WAL mode
  (`tests/test_job_history_wal.py`), per-workspace disk usage
  (`GET /api/workspaces`), the model-cache summary (`ModelsPage.svelte`), dw's
  caches (`CachesPanel.svelte`, #828), and orphaned run directories (#374,
  shipped under #170 as `list_gallery(only_orphans=true)` plus bulk delete).
- The proposal's prune SQL would have deleted open rows: every admitted job
  has a row since #1220.
- A pruned row silently erases observed cost, host-RAM gate learning, rerun
  lineage and device affinity, gallery provenance and the event log, against
  the #944 keep-rows ruling.
- Side finding: `GET /api/jobs` `total` stopped at 200. Filed and fixed as
  #1466.

## Decisions

- **Q1 [scope]:** defer (2026-11-19, delegate), then **stage A only** (Don,
  2026-11-26). No prune, no delete or VACUUM surface.
- **Q2 [none]:** file the 200-cap bug: #1466.

## What was built

### Stage A, #1766: storage stats (cfb76411, merged 910c943d; size fix c2033e38, merged b4fdeecd)

- `GET /api/server` / `ServerInfo` gains `storage`:
  - `job_history {path, size_bytes, rows, oldest, newest}` from
    `JobHistory.storage_stats()` (`dw/server/job_history.py`). `size_bytes`
    includes the `-wal`/`-shm` sidecars; `rows` is a real `COUNT(*)`;
    `oldest`/`newest` are `MIN`/`MAX(created_at)` as epoch-second floats.
  - `logs {path, size_bytes, backups, total_bytes}` from
    `log_storage_stats()` (`dw/log_setup.py`), read off the `dw` logger's
    file handler; `backups` counts `path.1`..`path.7` on disk.
  - Each block is read in its own guard: an unreadable figure or block is
    `null`, never an error. `logs` is `null` when no log file is configured.
- MCP `get_server_info` relays the block unchanged; its docstring names
  `storage` and was trimmed to stay inside `SURFACE_BUDGET`.
- Server › Status shows a read-only "Storage" panel before `CachesPanel`:
  "N jobs · size · first day – last day" and "size · N backups · total in
  all"; null figures render as "unreadable" / "no log file configured".
- The architecture review's size ceiling bounce moved `WorkerInfo`,
  `HealthInfo` and the `ServerInfo` family into
  `dw/server/api_models_system.py`, re-exported from `api_models`.
- Tests: `tests/test_server_storage.py` (one per figure, plus a real job
  moving `rows` 0 → 1), `serverinfo.test.ts`, `ServerPage.test.ts`. OpenAPI
  and `api-schema.ts` regenerated; `docs/SERVER.md` and `docs/MCP.md`
  updated; `UI_READ_ROUTES` unchanged (no new route).
- Deviation: the UI says "jobs" where the API field is `rows`.

Acceptance cases C-F1539 to C-F1542 (`regression-suite-complete.md`) all
passed on mini-ai at b4fdeecd.

## Deferred

- **Stage B (prune):** `POST /api/jobs/prune`, any delete or VACUUM surface.
  Not filed. It needs a separate ruling from Don, because it erases learned
  history (#944). v1's triggers for revisiting still apply: a field report
  about job-history size or a slow jobs listing, or `jobs.sqlite` above
  200 MB (it was 23 MB on mini-ai at verification).
- Output-directory totals and the model-cache summary: already served by
  `GET /api/workspaces`, the Models page and `CachesPanel`.

## Cost and bounces

Planned at about $3 of agent work and one short GPU job for verification.
The stage comments name no `usage:` figures, so actual cost is not recorded.

- **#1766: one bounce**, from the architecture review, not the tester:
  `dw/server/api_models.py` was over the 1,100-line `SIZE_CEILING`. Fixed by
  a pure move (now 1,028 lines). The tester's verify passed first time.
