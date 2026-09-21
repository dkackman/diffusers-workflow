# Live `get_memory` during a run (#269, gap 2)

Split from #269 on 2026-09-21. Gap 1 of that issue (a failed job's `progress`
going null instead of keeping the phase it died in) is fixed directly
(`dw/server/jobs.py`, `Job.progress()`) and does not need this decision. Gap 2
does: `get_memory` refusing with `reason: "job_running"` for the whole
duration of a run, which is exactly when a caller wants it most - to tell a
merely slow run from one that is thrashing toward an OOM, in time to act
(`cancel_job`) rather than reading the answer in a post-mortem after #265/#266
had already failed.

## Why this isn't a mechanical fix

`memory_status()` (`dw/server/jobs.py`) answers live only when the worker is
idle; while a job runs it returns `_cached_memory("job_running")` without
asking the worker anything. That refusal isn't a missing branch - it reflects
two real constraints:

1. `_worker_lock` is held by the runner thread for a job's entire duration.
   `memory_status()` cannot take it to send an ordinary command and block on
   the reply without serializing behind the run it's trying to inspect.
2. `_consume_results()` is the sole reader of `result_queue` while a job is
   in flight. It expects a fixed sequence of messages for *that* job
   (`progress`, `error`/`success`, etc.) and has no mechanism today to
   recognize and route an out-of-band reply to a concurrent caller. The
   closest existing pattern, `probe_cache`'s `probe_id` correlation, exists
   for exactly this kind of request/response-while-something-else-runs
   problem - but `probe_cache` itself explicitly refuses when a job is
   active (`self._current_job_id is not None`), so it's a model to extend,
   not a mechanism already fit for this.

Either fix means widening what can happen while the worker is mid-run:
new command handling in `dw/worker.py`'s `_watch_commands()` (already
mid-run-only, currently limited to `cancel`/`ping`/`shutdown`), and new
correlation state in `JobManager` to route a reply back to the right caller
instead of the runner thread. That's new engine surface, not a one-line
guard change - hence the escalation.

## Two designs

**(a) Proactive: emit `memory_info` at phase boundaries.**
The worker already emits one `memory_info` message, once, right after a run
finishes. Extend that to fire at each phase transition (`loading` ->
`generating` -> `decoding` -> `saving`) instead of only at the end.
`JobManager` folds each into the job's cached memory reading the same way it
already does for the post-run one (`_record_memory`), so `get_memory` while
a job is running is instantly answerable - it is not live in the sense of
"queried right now," but it's fresh as of the last phase boundary, which is
already enough resolution to catch "still climbing" vs. "flat" over the
handful of phases a run has.

- Pro: no new correlation machinery, no new command type, no contention with
  `_worker_lock` - it rides the same one-directional message flow that
  progress events already use.
  Con: resolution is coarse (phase boundaries, not on-demand); a phase that
  runs long (a slow `generating` on a big denoise loop) still leaves a stale
  reading for its whole duration, which is close to today's failure mode for
  exactly the case #265/#266 cared about most.

**(b) On-demand: a correlated request/response over the worker's command
queue, mid-run.**
Add a `memory_status` command that `_watch_commands()` answers even while a
job is active (it can - `_get_memory_info()` is pure stat reads and already
thread-safe against the run loop), tagged with a request id. `JobManager`
sends it without taking `_worker_lock` (a new, PID/queue-safe path,  not
the same lock the runner holds), and `_consume_results()` recognizes a
tagged `memory_info` reply and hands it to the waiting caller instead of
folding it into the job's own event stream - the same shape as
`probe_cache`'s `probe_id`, extended to run concurrently with an active job
rather than refusing when one exists.

- Pro: genuinely live, matches what `get_memory`'s docstring already
  promises for the idle case.
  Con: new correlation state in `JobManager` shared between the runner
  thread and whatever thread services `get_memory` calls; needs care that a
  slow or wedged worker (already mid-OOM) doesn't leave a `get_memory` call
  hanging on a queue nobody is servicing - probably wants its own short
  timeout, distinct from a job's.

## Recommendation

(a) first: it is the smaller change, reuses machinery this codebase already
trusts (the existing post-run `memory_info` message, `_record_memory`), and
directly serves the motivating case (distinguishing "slow" from "climbing")
without touching `_worker_lock` or `_consume_results`'s message routing. If
phase-boundary resolution turns out to be too coarse in practice - a single
long `generating` phase hiding a mid-phase spike - (b) is the fallback, and
(a)'s phase-boundary readings remain useful as a baseline even if (b) is
later added on top.

Filed as its own issue against #269 rather than answered here because it
adds new mid-run worker protocol either way and the tradeoff above is a
product decision (how fresh does "live" need to be), not an implementation
detail.

Model: sonnet. Provider: anthropic.
