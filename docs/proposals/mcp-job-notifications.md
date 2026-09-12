# Proposal: gapless job-completion signaling across the MCP bridge

Status: **not started** - design only, written after tracing the current
`wait_for_job` / event-log path end to end. No code changes yet.

## The problem, as reported

An agent driving `dw` over MCP (a Claude Code session, interactively or via a
`ScheduleWakeup`-paced loop) submits a job and then polls for it to finish.
In practice that active loop sometimes fails to notice the terminal
transition - a completed run, or a worker OOM-kill - promptly. The job
*does* reach a terminal state server-side; the agent just doesn't reliably
find out from the next poll.

## Why the current contract allows that gap

`wait_for_job` (`dw_mcp/diagnose.py:147-229`) polls `GET /api/jobs/{id}`
every `WAIT_POLL_SECONDS` (1.0s) and returns as soon as `status` is one of
`succeeded`/`failed`/`cancelled`, capped at `MAX_WAIT_SECONDS` (55s) per
call - the comment there explains the cap exists because no MCP client
holds a tool call open longer than roughly one generation's runtime. A
caller whose job outlives one call gets `still_running: true` back and is
expected to call again.

That contract only tracks a *snapshot* of `status`. It never tells the
caller what it has and hasn't seen, so nothing about it prevents this
sequence: call N times out at `still_running: true`; the caller's own
bookkeeping (a `ScheduleWakeup` cadence, a retry decision made by the model)
drifts, skips a beat, or gets reset; call N+1 starts a fresh 55s window with
no memory of call N. If the terminal transition happened in the seam
between "the loop should have re-invoked" and "the loop actually did," there
is no cursor tying the two calls together that would let the caller prove
to itself it never missed anything - it's trusting its own retry discipline,
which is exactly the thing that reportedly drifts.

The server side, by contrast, already has what's missing. `dw/server/jobs.py`
keeps a per-job append-only event log; `GET /api/jobs/{id}/events` (SSE,
`dw/server/app.py:1033-1069`) and its polling twin `GET
/api/jobs/{id}/event-log` both already support resuming from a sequence
number (`Last-Event-ID` / an offset param), and `ui/src/lib/api.ts`'s
`streamJobEvents()` already resumes this way on reconnect. `wait_for_job`
just doesn't use any of it - it was built against the coarse `GET
/api/jobs/{id}` status field, not the event log the SSE route already reads.

A worker OOM-kill is, separately, already detected reliably:
`WorkerManager.crash_details()` (`dw/repl_worker.py`) inspects the dead
worker's exit code and identifies a negative code consistent with SIGKILL
within one liveness-poll interval (~1s) of the process dying; `_consume_results`
(`dw/server/jobs.py:1019-1082`) converts that into a `failed` job with a
human-readable reason string. So the job never gets stuck in `running` -
the gap is purely that this distinction (OOM vs. an ordinary pipeline
exception vs. cancellation) is flattened into one free-text `error` string
by the time a caller sees it, so an agent can't branch on it without parsing
prose.

## Design

### 1. Cursor-based `wait_for_job`

Change `wait_for_job` to read the same event log the SSE/event-log routes
already read, keyed by sequence number, instead of the bare status field:

- New optional parameter `since_seq` (default `0`).
- Each call advances through `job.events_after(since_seq)` until either a
  terminal event appears or the 55s budget expires.
- Response gains `last_seq` (the highest sequence number the call actually
  observed) alongside `status`/`still_running`.
- The documented contract for a caller: **always pass the `last_seq` you
  were given into your next call.** Doing so makes it structurally
  impossible to skip a terminal event - the next call resumes exactly where
  the last one left off, rather than re-asking "what's the status *now*"
  and trusting that "now" didn't skip past "finished" in between. This
  turns a timing bug into a logic property: correctness no longer depends
  on how promptly the caller re-invokes, only on whether it ever stops
  calling at all (which is a visible, reportable "I gave up," not a silent
  miss).
- `get_job_events` (the existing paged event reader) takes the same
  `since_seq` / `last_seq` pairing for consistency, replacing its current
  offset-based paging.

### 2. Explicit `failure_kind`

Add a `failure_kind` field, populated at the same point
`_consume_results` already classifies a dead worker or a `worker_crashed`
message:

- `"oom"` - negative exit code consistent with SIGKILL (today's
  `crash_details()` heuristic)
- `"crash"` - worker died some other way (different signal, non-zero exit)
- `"error"` - the pipeline raised inside a live worker
- `"cancelled"` - explicit cancellation
- absent/`null` on success

Additive to the existing `status`/`error`/`traceback` fields - a new
nullable column in `jobs.sqlite`, no migration of existing rows needed
beyond a default of `null`. This lets a caller branch on `failure_kind`
(e.g. suggest a smaller batch or lower resolution on `"oom"`) instead of
pattern-matching the free-text reason string.

### 3. Document the recommended loop

Once the cursor exists, the failure mode this proposal is about becomes a
caller-discipline problem with a mechanical fix: always thread `last_seq`
forward, whether the next call happens immediately (blocking retry) or
after a `ScheduleWakeup` delay. Document this explicitly in
`dw_mcp/CLAUDE.md` and the `Authoring a workflow from an agent` section of
`docs/WORKFLOW_GUIDE.md` (or wherever job-waiting guidance already lives)
rather than leaving each session to reconstruct a polling loop from first
principles: submit → `wait_for_job(since_seq=0)` → if `still_running`,
re-call with the returned `last_seq` → on terminal status, branch on
`failure_kind`.

### Rejected: server-initiated push (webhook/callback)

`dw.serve` calling out to the agent when a job finishes was considered and
dropped. There is no MCP-native or Claude-Code-native inbound channel that
lets an MCP server wake a specific idle session; building one would mean
standing up a listener process outside both `dw` and the MCP protocol, which
just relocates today's polling problem onto a different piece of
infrastructure rather than removing it. The cursor fix above removes the
actual defect (a possible-but-unprovable miss) without adding a process,
a port, or a new failure mode of its own.

## Scope / non-goals

- No change to job semantics, `TERMINAL_STATES`, or the worker crash-detection
  heuristic itself - this only exposes what's already computed, more
  precisely and with a resumable cursor.
- No change to the SSE endpoint's own framing; it already resumes by
  sequence number. This brings `wait_for_job`/`get_job_events` up to the
  same standard, it doesn't change the standard.
- Not a UI change. `ui/src/lib/api.ts`'s `streamJobEvents()` already
  resumes by `seq`; this keeps the MCP surface consistent with it rather
  than introducing a second convention.

## Testing

- Server: a test that a SIGKILL'd worker's job record carries
  `failure_kind: "oom"`, alongside the existing crash-detection tests.
- MCP: a test that two sequential `wait_for_job` calls, with the cursor
  passed forward, never re-report an event already seen and never skip a
  terminal one - including the case where the terminal event lands in the
  gap between the two calls (assert on the boundary, not just the happy
  path).

## Open question for whoever picks this up

Whether `since_seq`/`last_seq` should be *required* (no default of `0`) so
that a caller cannot silently opt back into the old snapshot behavior by
omitting it. Leaning towards optional-with-a-default for backward
compatibility with any existing caller that only reads `status`, but a
required cursor is the stronger guarantee if there's an appetite for a
breaking MCP-surface change here.
