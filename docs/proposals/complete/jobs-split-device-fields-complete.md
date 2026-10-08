# `dw/server/jobs.py` split into pool, results and memory; device ordinal and card as stored fields (#693)

Written by model `claude-opus-5-5` via provider `anthropic`, at close-out
2026-10-08, from plan v1 on #693 and the stage threads (#776, #777).
Plan v1 approved by Don 2026-10-08 with every default (Q1–Q4). Built and
verified on mini-ai (mps) the same day.

## The ask

The 2026-10-07 develop review (structural item 3) found `dw/server/jobs.py`
at about 1,450 lines holding every server-side job concern: the dispatcher,
the slot, VRAM-fit and affinity policy, worker-result consumption, memory
and cache probing, the history readers, and `slots[0]` compatibility shims
(`_current_job_id`, `_worker_lock`, `last_memory`, `last_memory_at`) left
from before the multi-GPU pool. Device identity was stored only as the
display label `"cuda:1 NVIDIA GeForce RTX 3090"` and parsed back with
`split(" ", 1)` (`devices.card_of`, `devices.ordinal_of`) for rerun
affinity, `workers()` and observed-cost card matching.

## Verdict

**Build smaller, in two stages (A → B).** The value is to the agent
sessions that edit the server: no field report came from the label parse,
which was correct because an ordinal never holds a space. The cheaper
alternatives were doing nothing, or stage A only (keeping the parse). Don
chose both stages (Q4).

Cut from the issue: the "dispatcher stays ~300 lines" target (it needs the
history readers moved, a non-goal), and the new fields on REST/MCP (Q1: they
stay internal; `device` stays the only API field).

## Design corrections found against the issue

1. "`tests/test_worker_pool.py` must pass unchanged" contradicted deleting
   the shims and parsers: one line asserted `manager.last_memory` and one
   test unit-tested `ordinal_of`/`card_of`. The contract became: every
   lock, dispatch and affinity test passes unchanged, and only those two
   places move (Q2).
2. Two modules don't reach ~300 lines. The memory/cache code was a third
   clean seam and moved too (Q3).
3. `route` takes `JobManager._lock` and is monkeypatched by the admission
   test, so it stays a manager method. Only the lock-free policy moved.
4. Callers reach the moved functions through the manager
   (`routes/jobs.py`, `routes/library.py`, `admission.py`, tests), so the
   manager keeps thin delegators under the old names.
5. Old history rows held only the label. Without a backfill, adding columns
   would have dropped pre-migration runs from observed cost and rerun
   affinity. The migration backfills in SQL (`INSTR`/`SUBSTR`), so no
   Python label parse remains.

## What was built

### Stage A (#776): split `jobs.py`; delete the `slots[0]` shims

- `dw/server/pool.py`: `WorkerSlot` and the lock-free policy (`slot_for`,
  `choose_slot`, `check_fits`, `dispatch_need`, `largest_ceiling(_gb)`,
  `unfit_message`). #685's refusal wording is unchanged.
- `dw/server/job_results.py`: `consume_results` and the output-name,
  manifest, progress and memory record helpers. `output_dir`, the slot and
  the memory recorder are explicit arguments; there is no back-reference to
  the manager. They still run under `slot.lock`, because `_run_on` calls
  them.
- `dw/server/worker_memory.py`: `WorkerBusy`, `memory_status`,
  `clear_memory`, `probe_cache` and `record_memory`, taking the manager's
  lock as an argument. Slot locks keep their bounded 2 s acquire and are
  never taken under that lock.
- `jobs.py` keeps `JobManager`, the dispatcher, submit, rerun, the history
  readers, cancel and move, with delegators for every name a route or test
  calls. `JobManager`, `WorkerBusy` and `TERMINAL_JOBS_KEPT` stay importable
  from `dw.server.jobs`; `worker_manager` is still `slots[0].manager`.
- The four shims are deleted. `JobManager.running_job_id()` returns the
  oldest running job, and `routes/system.py` uses it.
- `jobs.py` went from 1,476 to 935 lines (927 after stage B).
- Tests: the shim reads in `test_server.py`, `test_server_jobs.py` and one
  line of `test_worker_pool.py` go to `slots[0]`; a new
  `test_server.py::test_health_names_the_running_job`.
- Docs: the `ARCHITECTURE.md` rows *Job queue*, *Worker pool dispatch* and
  *Card affinity* name the new modules, plus a new *Per-card memory and
  cache* row; the `jobs.py` docstring.
- **Deviations (no behavior change):** the route served by
  `running_job_id()` is `/api/health` (`current_job`), not `/api/system`
  as the plan said; two extra delegators remain (`_record_memory`, which a
  test calls, and `largest_ceiling`, which `admission.py` calls); 935
  lines rather than the plan's ~650, the difference being the history
  readers the plan kept in place.

### Stage B (#777): device ordinal and card as stored fields

- `Job.device_ordinal` and `Job.device_card` are set at dispatch from
  `slot.device_fields()`. `job.device` is a read-only property joining them
  with `devices.label_of`.
- `job_history.py` adds two columns with the PRAGMA-guarded ALTER pattern
  and a one-time SQL backfill from `device`: split at the first space; a
  label with none (`cpu`, `mps`) becomes the ordinal with a NULL card.
- Readers on the fields: rerun affinity (`_original_device`, via the new
  internal `JobHistory.device_ordinal(job_id)` for a finished job),
  `workers()`'s `name`, and `observed_cost._on_this_card` (matches on
  `device_card`; a row with no ordinal counts only on the default card).
- `card_of`, `ordinal_of` and the now-unused `devices.device_label` are
  deleted. No `split(" ", 1)` on a device label remains in `dw/`.
- History still writes the `device` label column, so REST and MCP `device`
  are byte-identical; `JobSummary`, `openapi.json` and `api-schema.ts` are
  untouched.
- Tests: `test_job_history_device.py::TestFieldBackfill` (a fixture DB with
  `cuda:1 NVIDIA…`, `cpu`, `mps`, `mps <name>` and NULL rows; the backfill
  runs once; a DB from before any device column gets all three columns),
  two field-level tests in `test_worker_pool.py` replacing the parser test
  (Q2), a pre-migration rerun-affinity test, and `test_observed_cost.py`
  `TestByCard` on backfilled rows.
- Docs: the `dw/devices.py` docstring, `job_history.py`'s schema note, and
  the *Worker card pinning* map row.
- **Deviation:** the card is cached once per worker by
  `WorkerManager.device_fields()` (same `card_name` source) rather than
  read per job at dispatch. It also lets test fakes name a card on a box
  with no CUDA.

## Verification

Both stages passed architecture review and the tester's MCP cases on
mini-ai (mps) first time: C-F363–C-F366 for stage A, C-F367–C-F370 for
stage B. C-F367 pinned pre-migration jobs labelled
`mps Apple M5 Pro (MPS)`, a card name with spaces and parentheses, and
C-F369 confirmed observed-cost run counts kept the pre-migration runs and
grew by exactly one per new run.

## Bounces

| Stage | Bounces | What |
|---|---|---|
| A (#776) | 0 | Architecture review pass; C-F363–C-F366 pass. |
| B (#777) | 0 | Architecture review pass; C-F367–C-F370 pass. |

Cost per stage is not recorded: the stage comments name no `usage:` figures.

## Deferred, and why

- **The history readers stay in `jobs.py`** (`get`, `definition`,
  `rerun_spec` …). They are the manager's public API to the routes; moving
  them would only add delegators. *Comes back if* a later review shows them
  still costing readers of the dispatcher.
- **`device_ordinal`/`device_card` on REST/MCP** (Q1). *Comes back if* a
  consumer needs to filter or group jobs by card.
- **Multi-card paths on hardware.** mini-ai has one mps worker, so rerun
  affinity to another card and a CUDA card name were checked only by unit
  tests. lem (CUDA, multi-card) has not run these stages.

## Notes for later

- The UI header's running job (`/api/health` `current_job`) was checked
  over MCP through `get_health`, not by hand in the browser.
- C-F369 noted `get_workflow`'s `observed.name` is null and undocumented;
  not filed.
