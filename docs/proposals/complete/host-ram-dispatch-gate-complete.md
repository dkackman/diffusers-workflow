# Hold a queued job for host RAM at dispatch (#869)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-15). Plan v2, approved by Don as "build smaller" on 2026-10-14 (v1 →
v2: Don's answers folded in as decisions D1-D3, no scope change). Two stages:
#909 (A) and #910 (B), both verified on lem. Part 2 of the report (the wrong
job died) was split out and shipped as #881.

## The report

On 2026-10-13, lem (2x RTX 3090, 125 GB host RAM) dispatched two int8 H3
Ref2VA jobs to its two cards at once. Each holds ~62-63 GB RSS after load, and
two don't fit. The earlier job was SIGKILLed by the OOM killer and the later
one died with SIGTERM. The dispatcher (#462) routed on VRAM need only, and
`host_memory_projection` produced validate warnings only. Three places told
agents to work around it by hand: the workspaces guide ("wait between
variants"), the `minimax-h3` quality reference ("One int8 job at a time") and
`tools/ladder.py` waiting for lem to be idle.

The code sweep (plan v1) found:
- **History had no (pipeline, weights dtype) key.** The dtype is usually a
  template variable feeding a `quantization_config`, and history grouped by
  workflow name. So the plan added a *memory key* computed from the admitted
  definition.
- **`host_memory_projection` projects list runs only.** On its own it would
  never have figured an H3 shot.
- **"Free" can't be MemAvailable alone.** A job that is still loading hasn't
  reached its peak, so each running job's remaining growth is charged.
- **A `warning` event sticks to `job.warnings` for good.** A queue wait is a
  moment, not a fact about the result, so it gets its own event and field.
- **The dispatcher only woke on submit, finish, hold release or shutdown.** A
  held job needs a timed re-check.

## What was built

- **Stage A (#909): memory key and projected peak, nothing gated.**
  - A `memory_key TEXT` history column (migrated with `ALTER TABLE`; older
    rows are NULL and count as no data), written by `record()`. The key is the
    sorted `pipeline_identity` of each pipeline step plus every
    `quantization_config`, `offload` and `group_offload` value, with the
    caller's arguments substituted through the single owner of `variable:`
    substitution. Renamed and inline copies of one template share a key;
    `weights_dtype` int4 and int8 don't.
  - `dw/server/host_ram_need.py: projected_peak` returns `observed` (the max
    `host_memory_job_peak_rss_mb` over successful same-key runs, D2), else
    `fit` (`host_memory_projection.projected_mb`, a list run), else None.
  - `host_memory_projection` now shares one `_projection` between
    `projected_mb` and `host_memory_warnings`. The warning text and its 0.9 ×
    total-RAM ceiling are unchanged.
  - `Admission.host_ram_need`, threaded through submit, rerun and enhance.
  - Validate's `plan.host_memory: {projected_mb, basis, runs}`, absent with
    no figure.
- **Stage B (#910): the gate.**
  - `_host_needs` beside `_needs` in `JobManager`. The rule lives in
    `dw/server/pool.py: host_ram_wait`: `available = MemAvailable − Σ running
    max(0, projected_peak − worker RSS now)`, and a job is held while its
    peak exceeds `available` plus the RSS of the idle worker it would reuse.
  - Backfill: a held job keeps its place and smaller jobs behind it may start.
  - Escapes: never held with no other job running, never held without a
    figure, and no gate when `host_memory_stats()` can't read MemAvailable.
  - While any job is held, the dispatcher re-checks every 10 s
    (`HOST_RAM_RECHECK_SECONDS`), so memory freed outside dw is seen.
  - A `queue_wait` event (`reason: "host_ram"`, "waiting for host RAM: needs
    ~N GB, ~M GB free"), once per hold, never a `warning` (D1).
  - `waiting_for` on the job summary and detail while queued and held: REST,
    regenerated `openapi.json` and `api-schema.ts`, MCP `get_job`,
    `list_jobs` and `wait_for_job`, and the UI's `JobHeader.svelte`.
  - Cancel clears it; `move` wakes the dispatcher.
  - Docs: SERVER.md states the rule and its escapes; WORKSPACES.md and the
    `minimax-h3` quality reference drop the wait-by-hand advice; MCP.md and
    the ARCHITECTURE.md dispatch row name the owners.

On lem the verify held a second int8 H3 job ("needs ~87 GB, ~57 GB free")
with cuda:1 idle and started it the instant the first finished, while two
int4 jobs (~47 GB observed each) started at once.

## What was cut or deferred, and why

- **Refusal at submit.** Never: host RAM is this machine's state, not the
  caller's choice, and #243's warn-not-refuse rule stands.
- **A "would wait now" prediction in validate's plan.** It guesses at a
  queue that changes before submit. *Back when:* a field report of agents
  submitting blind and being surprised by `waiting_for`.
- **A new job status.** Don's answer: the job stays `queued`, and
  `waiting_for` says why.
- **`num_frames`, resolution and reference count in the memory key** (D3).
  *Back when:* under-holding shows up from a key collision across sizes.
  #881's OOM ranking is the backstop.
- **Pipelines inside a composed child workflow** are not in the parent's
  key (stage A deviation).
- **No plugin version bump.** The plugin version is pinned to the engine's
  and the release script bumps both.

## Bounces per stage

| Stage | Verify bounces | Architecture bounces | Notes |
|---|---|---|---|
| #909 (A) | 1 | 1 | Architecture (two findings in one bounce): a second owner of `variable:` substitution in `host_ram_need`, folded into the single owner, and the new module had no ARCHITECTURE.md row. Verify: C-F488 expected a projection warning that #243's 0.9 ceiling never emits at ~61.6 GB on lem. No code change; Don approved the case amendment. |
| #910 (B) | 0 | 1 | Architecture: the ARCHITECTURE.md dispatch row didn't name `pool.py: host_ram_wait` as the owner of the hold rule. |

C-F495 (single worker or MPS, dispatch unchanged) was not run on lem, which
has two workers, so it stays `pending: #910`.

Cost per stage: the stage comments record no `usage:` figures, so none is
given here. The plan estimated $4-6 (A) and $5-6 (B).

## Found on the way

- `validate_workflow` accepted `weights_dtype: "int8"` unbraced, and the run
  failed at load. Fixed as #926 (validate warns on an unbraced type name
  under a `*_dtype` key).

## Rules this feature added

- Dispatch may delay a job for host RAM but never refuses one; a job with no
  other job running is never held.
- A queue wait is a `queue_wait` event and a live `waiting_for`, never a
  `warning`.
- The observed host-RAM figure is the max over same-memory-key successful
  runs on this machine.
