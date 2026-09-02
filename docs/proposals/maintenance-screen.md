# Proposal: a Maintenance screen

Status: draft / scoping only — no implementation.

## Problem

`dw/server/jobs.py`'s `JobHistory` writes one row per finished job to
`~/.diffusers_helper/jobs.sqlite` forever. No pruning, no WAL mode. At normal
scale (a few jobs a day) this is a non-issue; a year of heavy daily use could
put this into the tens-of-thousands-of-rows range, plus each row carries up
to `MAX_PERSISTED_EVENTS` (200) JSON-serialized progress events and the full
`manifest`/`spec`/`warnings` blobs. Slow, unbounded disk growth, and no way
for a user to see it happening or do anything about it — no UI, no CLI.

More generally, the server accumulates other persistent state (HF hub
cache, generated output files, log files) with uneven visibility: the hub
cache already has a dedicated page; outputs and logs have none.

This doc scopes a **Maintenance** area — UI page(s) plus matching REST
routes — to surface and manage all of it. No code changes are proposed
here; this is the design to review before opening an implementation issue.

## What state exists, and what's already covered

| State | Location | Currently surfaced? |
|---|---|---|
| Job history | `~/.diffusers_helper/jobs.sqlite` (`JobHistory` in `dw/server/jobs.py`) | Jobs page shows the list/detail; **no size/count/prune UI or route** |
| HF hub model cache | `constants.HF_HUB_CACHE`, inventoried by `dw/hub_cache.py`'s `scan_models()` | **Yes** — Models page (`ui/src/lib/pages/ModelsPage.svelte`) via `GET /api/models`, with per-repo delete (`DELETE /api/models`) and download management. Out of scope here; cross-reference only. |
| Generated output files | `--output-dir` (default `./outputs`), read live by `GET /api/gallery` | Gallery page lists/deletes files **individually**; no aggregate size, no bulk/age-based cleanup |
| Log file | `resolve_path(settings.log_filename)`, default `~/.diffusers_helper/log/dw.log` | **Already self-managing**: `dw/log_setup.py` uses `ConcurrentRotatingFileHandler` with `maxBytes=50MB`, `backupCount=7` — worst case ~400MB, capped, no action needed. Worth a read-only "current size" line for completeness, but not a pruning control. |
| Settings file | `~/.diffusers_helper/settings.json` | Not job-history-like (single small file); out of scope. |

So the hub cache is already handled — the new screen should link to the
existing Models page rather than duplicate its scan/delete logic. The two
real gaps are **job history** (no visibility, no pruning) and **output
directory size** (visible per-file in Gallery, but no aggregate/bulk view).
Logs are already bounded and need no interaction, just a status line.

## Proposed shape

### New UI page: "Maintenance"

Fits the existing nav pattern in `ui/src/App.svelte` (`route.parts[0]`
dispatch, one entry in the `<nav>` icon row) alongside Workflows / Prompts /
Jobs / Editor / Gallery / Models / Schema. A new `MaintenancePage.svelte` in
`ui/src/lib/pages/`, following the same fetch-on-mount + card layout the
Models and Gallery pages already use.

Sections on the page:

1. **Job history** — DB file size, row count, oldest/newest `created_at`.
   Two prune actions: "keep only the last N jobs" and "delete jobs older
   than N days" (N as a number input, sane default e.g. 500 / 90 days), each
   behind a confirm dialog showing how many rows/bytes it would remove.
2. **Output directory** — aggregate size and file count of `--output-dir`
   (the same directory Gallery reads), computed from the same listing logic
   Gallery already does. Point at the Gallery page for per-file deletion
   rather than re-implementing it here; maybe a single "older than N days"
   bulk action if wanted, but the individual delete already exists so this
   is low priority.
3. **Log file** — current size and rotated-backup count, read-only. Links
   to the file path. No action needed since rotation is already automatic.
4. **Model cache** — one line ("N repos, X GB") linking to the Models page.
   No duplicated controls.

### New REST routes (`dw/server/app.py`)

Following the existing route conventions (plain functions, `manager`/
`history` closures already in scope in `create_app`):

- `GET /api/maintenance/jobs` — `{db_path, size_bytes, row_count, oldest_created_at, newest_created_at}`.
- `POST /api/maintenance/jobs/prune` — body `{older_than_days}` or
  `{keep_last}` (one of the two); returns `{deleted, freed_bytes}`. Refuse
  (400) if both or neither are given.
- `GET /api/maintenance/outputs` — `{output_dir, size_bytes, file_count}`
  (thin wrapper: sum of the same `os.listdir`/`os.stat` walk `GET
  /api/gallery` already does, minus the per-file payload).
- `GET /api/maintenance/logs` — `{log_path, size_bytes, backup_count,
  total_size_bytes}` (stat the configured file plus its `.1`..`.7`
  rotated siblings).

`/api/models` (hub cache) is intentionally not duplicated — the page links
out to it.

### Server-side code needed

- `dw/server/jobs.py`: `JobHistory` needs two new methods —
  a stats method (`SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM
  jobs`, plus `os.path.getsize(self.db_path)`) and a `prune(older_than_days=
  None, keep_last=None)` method (`DELETE FROM jobs WHERE created_at < ?`, or
  `DELETE FROM jobs WHERE id NOT IN (SELECT id FROM jobs ORDER BY
  created_at DESC LIMIT ?)`), followed by `VACUUM` or at least `PRAGMA
  incremental_vacuum` so freed space is actually reclaimed on disk, not just
  freed within the file. This is the one piece of real new logic; everything
  else is thin read-only aggregation of data the app already computes
  (`scan_models`, the `os.listdir`/`os.stat` loop in the gallery route, log
  path from `settings.log_filename`).
- No new module needed — a `maintenance.py` under `dw/server/` is
  reasonable if the routes grow beyond a few lines each, but the stats/prune
  logic belongs on `JobHistory` itself since it already owns the connection
  and lock.

### WAL mode — separate quick win

Turning on `PRAGMA journal_mode=WAL` in `JobHistory._connect()` (or once at
init) is low-risk and orthogonal to the pruning UI: it doesn't need a
screen, a route, or a design decision from the user, and it directly
addresses the "no WAL mode" half of the audit finding. Recommend doing it
as an immediate one-line fix independent of this proposal, not blocked on
the Maintenance screen shipping. (Note: `sqlite3.connect(..., timeout=5)`
already exists for lock contention; WAL reduces writer/reader blocking
further, which matters more once SSE polling and prune both touch the DB.)

## Effort and risk

Low-to-medium. No new state, no schema migration beyond what `JobHistory`
already does defensively (`ALTER TABLE ... ADD COLUMN` pattern already
exists as precedent for future schema changes). Main risks:

- Prune-while-running races: pruning should skip/exclude the small
  in-memory `TERMINAL_JOBS_KEPT` window and never touch rows for jobs the
  `JobManager` still holds live — worth an explicit look at the boundary
  between in-memory `Job` objects and persisted rows before implementing.
- `VACUUM` on a large DB briefly locks the file; should run it off the
  request thread or accept the request blocking a couple seconds for a
  DB that's still only tens of MB even at "a year of heavy use."
- UI is new but small — four read cards and two destructive actions with
  confirms, no new client-side state management beyond what Models/Gallery
  pages already establish as pattern.

Rough sizing: WAL quick-win, <1 hour. Phase 1 below, roughly a day
(server: stats/prune methods + 3 routes + tests; UI: one page, two
confirm-guarded actions). Phase 2, a few hours if wanted.

## Suggested phasing

**Phase 0 (do now, independent of this doc):** enable WAL mode on the jobs
DB connection. One line, immediately reduces write contention, no UI
required.

**Phase 1 (minimal Maintenance screen):**
- `GET /api/maintenance/jobs` + `POST /api/maintenance/jobs/prune`
  (`keep_last` only — simpler than supporting both `keep_last` and
  `older_than_days` on day one).
- One `MaintenancePage.svelte` section: job history stats + one prune
  button with a confirm dialog.
- Nav entry.

This alone closes the audit finding: pruning becomes possible and visible.

**Phase 2 (fill out the page):**
- `older_than_days` prune variant.
- Output directory aggregate stats + optional bulk age-based delete.
- Log file stats card (read-only).
- Model cache summary card linking to Models page.

**Not proposed:** scheduled/automatic pruning (e.g. a cron-like "prune on
startup if over N rows"). Worth considering once the manual controls exist
and usage patterns are observed, but automatic deletion of history without
an explicit user action is a bigger trust decision than this finding calls
for.
