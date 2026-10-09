# One job per GPU: a worker per card and a shared dispatcher (#462)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-07). Plan v1 was written 2026-09-26, held by Don until the
#598-#603 batch landed, reviewed against `develop` on 2026-10-07, and
approved by Don without answers to its questions, so each question's
default stood. It was built as four stages, #675, #676, #677 and #678,
all shipped and verified 2026-10-07/08, plus one fix-forward stage, #682,
after the feature's final check. No design doc existed before this
one: the plan lived only on the issue.

## The idea

lem gained a second RTX 3090. The server ran one job at a time on one
device: one `_run_loop` thread, one `WorkerManager`, a lock held for the
whole job and a scalar current job. Don's MCP sessions queue long,
independent renders back to back (on 2026-09-26, 13 `LaVela*` shots of
25-110 min each, some waiting 1-3 h), so a second card is throughput.

**Non-goals:** splitting one job across cards (no model or tensor
parallelism); a per-job host-RAM admission gate; sharing the step or
pipeline cache between workers; multi-GPU for `dw.run`/`dw.repl`.

## Verdict

**Build**, in four stages. The cheaper alternative, two `dw.serve`
instances pinned one per card, needed only stage A but gave up the single
queue: every MCP session would pick a server and judge VRAM fit by hand.
It remains the fallback setup and works on stage A alone.

## The design as built

1. **One worker process per configured card** (`dw/devices.py`).
   `dw.serve --devices cuda:0,cuda:1` (or `"devices"` in settings) builds
   one `WorkerSlot`/`WorkerManager` per entry. `WorkerManager.ensure_worker`
   sets `CUDA_VISIBLE_DEVICES=<index>` and `DW_DEVICE=cuda` in the
   parent's environment around `Process.start()`, then restores them, so
   inside the worker the card is index 0 and the index-0 readers
   (`device_memory_stats`, the peak-stats reset, the auto-offload
   `mem_get_info`) are correct without change. With no `--devices`
   nothing is pinned and the server is as before. Startup refuses a card
   the machine lacks (naming the cards present), a card named twice, and
   a bare `cuda` in a list of several.
2. **One queue and a dispatcher** (`dw/server/jobs.py`). A dispatcher
   thread walks the queue in order and gives each job a free card that
   fits it. A job that fits no free card keeps its place while a smaller
   one behind it takes the card (backfill, Q2). `move_job` still means
   start order. Cancel reaches the worker running the job, and a worker
   crash fails only its own job.
3. **Fit** (`dw/server/admission.py`, `dw/vram_estimate.py`). A declared
   `vram_estimate` is a hard need, held to the card's GiB rounded to one
   decimal (23.6 on a 3090), the same ceiling admission already used. A
   job above every card is refused at submit (also rerun and the enhance
   submit) with the largest card named. A need from catalog `cost` alone
   is soft: compared to the rounded-up size (24), and never refused.
   Admission computes the need once, over the expanded workflow; dispatch,
   routing and the estimate all read that one figure.
4. **Affinity** (`JobManager._choose_slot`). Among the free cards a job
   fits: a rerun's original card, then the card that last ran the same
   workflow identity, then a card with no worker running, then
   `--devices` order. A rerun takes another card rather than wait for a
   busy original. `probe_cache` asks the card the job would route to.
5. **Records.** Jobs carry `device` (`"cuda:1 NVIDIA GeForce RTX 3090"`,
   NULL for older rows). Observed cost buckets by card name, not index,
   and `plan.estimate` quotes the routed card's history and names it in
   `priced_for`.
6. **Host RAM** (Q3: no gate). The later-started worker gets
   `oom_score_adj` +100 over the others so a host-RAM squeeze kills it
   first, and each worker reports `host_memory_rss_mb`.

**Surfaces.**
- REST: `GET /api/health` gains `workers: [{device, name, vram_gb,
  current_job, alive, host_memory_rss_mb}]`; `current_job` (longest
  running) and `worker_alive` (any alive) stay. Job objects gain
  `device`. `GET /api/memory?device=` and `POST /api/memory/clear?device=`;
  without `device`, get adds `workers` and keeps the first card's reading
  at the top level, and clear clears every idle card, refusing (409) only
  when no card asked about is idle. An unknown device is a 400.
- MCP: `get_health`, `get_job`, `list_jobs` pass the fields through;
  `get_memory`/`clear_memory` take `device`; the instructions and
  descriptions say "one job per GPU". The surface budget passed at
  14,248 of 14,265 with the ceiling unchanged.
- UI: `WorkerList.svelte` in the status popover and server page (one row
  per card, with its job linked); `device` on the jobs list and job page.
- Docs: SERVER.md (*Choosing the card*, the two ceilings, the entry
  shape), MCP.md, WORKSPACES.md, ARCHITECTURE.md rows *Worker card
  pinning*, *Worker pool dispatch* and *Card affinity*.

Every shape change is additive: none is a `breaking-change`.

## What the 2026-10-07 review changed, and what the build found

- **lem's two cards are identical 3090s** (Q5). The plan assumed one card
  over 24 GB. The VRAM fit, backfill and the submit-time 400 were built
  anyway, per the approved plan, but on lem they only matter for a job
  too big for either card. The bigger-card and backfill arms are proven
  by `tests/test_worker_pool.py`'s scripted managers, not on hardware.
- **The run-version/run-directory race was already fixed** (`open_run`'s
  FileLock and exclusive `os.mkdir`). Stage A added only the
  cross-process test, `tests/test_runs.py::TestRunRaceAcrossProcesses`.
- **The index-0 fixes needed no code**: pinning makes index 0 right. The
  one server-side reader, `ObservedCosts.device()`, now reads by ordinal.
- **"One job at a time" had fewer sites** than the plan counted; B and C
  rewrote most of them before D.

## Deferred, and what brings each back

- **Host-RAM admission gate** (Q3). *Back when:* a field report of a
  cross-job OOM kill.
- **An `overlapped` flag on jobs, excluded from observed cost** (Q4).
  *Back when:* the contention skew in cold medians is measured.
- **`list_workflows`' cost listing per card.** It still quotes the
  default card; only `plan.estimate` names its card. *Back when:* lem
  gets cards that differ.
- **`get_server_info` card names.** Not added; card name and total come
  from `get_memory` and `get_health.workers`.
- **The cancel window** between dispatch and the worker starting a job is
  unchanged from before.

## Stages

Cost: the stage comments carry no `usage:` figures, so none is recorded.

| Stage | What | Shipped | Bounces |
|---|---|---|---|
| #675 A | worker pinning, `--devices` (one entry), job `device`, cost by card name, run-race test | `a7354733`, map fix `0402df95` | 1 architecture (no map row for `dw/devices.py`) |
| #676 B | worker pool, dispatcher and backfill, fit check and 400, `workers` on health, `oom_score_adj` | `d890661c`, `0d8c71d0`, `19844829` | 1 architecture (rationale in `dw/server/CLAUDE.md`, moved to the `jobs.py` docstring); 1 verify (the `workers` entry shape, the fit check's 24 vs admission's 23.6 at the boundary, and lem still on one card until Don set `devices`) |
| #677 C | identity and rerun affinity, routed probe, per-card memory, `priced_for` | `26f063b9`, `bffb2641` | 1 architecture (the VRAM need computed a second way in `routes/jobs.py`, and three related second-owner/map findings) |
| #678 D | MCP text, docs, UI | `7163f069` | none |
| #682 fix-forward | `get_job`/`list_jobs` descriptions name `device` | `22ebd1a7` | none |

## The final check, and the fix-forward

The feature's first final check failed 2 of 14 cases:

- **C-F350, a build miss.** Stage D changed the instructions and the
  `get_health`/`get_memory` descriptions but left `get_job` and
  `list_jobs` without a word on a job's `device`. Fix-forward #682 added
  it to both, within `SURFACE_BUDGET`, and C-F350 passed as written.
- **C-F344, a case error, not a server one.** The case declared the VRAM
  need through a workflow's `cost[].vram_gb`. That figure records a
  measured run, so stage B made it a soft need on purpose: it orders
  dispatch and refuses nothing. Only a declared `vram_estimate` gates
  submit, and that gate worked. Plan v1 named both sources loosely; the
  build settled it. The case was amended in dkackman/harnest#81 (Don
  approved; the soft-need arm was added) rather than the server changed.

Acceptance cases C-F337-C-F350 (`regression-suite-complete.md`) passed
on lem with both cards, C-F350 after #682. The UI is Don's to check by
eye.
