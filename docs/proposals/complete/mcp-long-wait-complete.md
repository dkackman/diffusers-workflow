# One `wait_for_job` call that covers a long render (#377)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-10-03). Plan v1 was approved by Don on 2026-09-27 with no answers to
Q1-Q4, so the defaults stood: a cap of 1800 on `lem`, stage 2 conditional
on a measured cut, no background-shell wait in the skills, and the old
proposal moved to `declined/`. Stage 1 (#546) shipped 2026-09-28 and was
verified 2026-10-03. Stage 2 (#547) was not built.

## The report

`wait_for_job` capped each call at 55 s. On 2026-09-25 a 41-minute H3
render (5 ref2va shots, job `51441504a8b6`) needed about 45 calls, or the
workaround Don used: two blind background `sleep`s with polls between them.
At ~$0.08 a poll at ~130k context (#248), that wait cost about $3.60 in
polling alone. The proposal on file, `mcp-job-notifications.md`, asked for
cursor-based gapless signaling plus a `failure_kind`.

## Verdict

**Build smaller.** Build only a single call that lasts as long as the
render. The 55 s cap was a deployment default, not a client limit, and the
means to raise it (`DW_MCP_MAX_WAIT_SECONDS`) had shipped in #248 but was
never turned on or measured.

**Cut:** both halves of the proposal. The reasons and reopen triggers are
at the top of `../declined/mcp-job-notifications.md`. In short, the cursor
fixes a gap that does not exist (terminal status is sticky), and the
proposed `failure_kind` would have labelled most GPU OOMs `error`.

## What the plan found against the proposal

1. **No seam gap.** Terminal status is sticky, so any later snapshot sees
   it. The one real miss is a server restart dropping the job, which is
   #300's and is outside a cursor's reach.
2. **OOM detection was in the wrong place.** CUDA OOMs arrive as ordinary
   errors from a live worker. Only the worker can classify them.
3. **The real risk was the transport, which the proposal never
   considered.** The HTTP mount answers with `json_response=True`, so a long
   wait is one silent request. Any idle timeout on the path would cut it.
   Stage 1 measured this, and stage 2 (a progress heartbeat over SSE) was
   held back until a cut was seen.

## What was built

### Stage 1 (#546): the cap on lem, the measurement, the docs

Shipped `develop` @ `e5bdbed` (`305305a`). The bounce fix is `5d48399c`
(`2666c6fc`).

- **Deploy.** `scripts/dw-serve.service` carries
  `Environment=DW_MCP_MAX_WAIT_SECONDS=1800`. `scripts/deploy.sh` passes it
  (default 1800) on the screen-path launch for a box with no unit, which
  covers mini-ai. #248's "no unit manages the process" was out of date:
  lem runs the systemd user unit `dw-serve`, whose *installed* copy didn't
  carry the variable. Don replaced it with the repo's unit on 2026-10-01
  (the old one is kept as `dw-serve.service.bak-2026-10-01`). The code
  default stays 55, so an unknown deployment is unchanged.
- **Docs.** WORKFLOW_GUIDE "The loop" step 5 states the rule once: ask for
  `plan.estimate` plus a margin as `timeout_seconds`; the reply's
  `timeout_applied_seconds`/`timeout_capped` say what you got; call again
  while `still_running`; the default is 20 s. It quotes no cap.
  `docs/MCP.md` (the `wait_for_job` and `run_workflow` rows, loop step 3)
  names the default of 55, the env var and the client-timeout caveat,
  because it is written for operators. `dw_mcp/CLAUDE.md` names the env var.
- **Skills.** `ltx-2.5`, `minimax-h3`, `minimax-music3` and
  `script-to-video` state the same rule. `series-episodes` and the guide's
  transcribe example now say `wait_seconds=60`: a request sized to a short
  job, not a cap. The three family skills sat at the 12,288-byte
  `SKILL_SIZE_LIMIT`, so each was tightened elsewhere (about 150 bytes,
  no fact removed).
- **No code change.** No engine, REST or MCP surface change, and no
  description text: the tool description already interpolates the live
  cap ("at most 1800.0 seconds").
- **Client side.** Claude Code's `MCP_TOOL_TIMEOUT` defaults to about 28 h,
  but its idle limit for an HTTP MCP server,
  `CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT`, defaults to 5 min and would cut a
  silent long wait first. Harness sessions export both at 1900000 ms
  (dkackman/harnest `b7e7e47`). An interactive session needs the same.

### Stage 2 (#547): progress heartbeat. Not built

It was conditional on stage 1 recording a long call cut by the transport.
Neither measurement was cut, so it closed `not planned` on 2026-10-03.
`mcp_mount` keeps `json_response=True`, and `wait_for_job` stays silent
while it blocks.

## Measured in verification

On lem, `templates/minimax/shots-batch`, seed 377, shots 1-3:
- **First verify** (`develop` @ `f01ae1fa`, job `865715efb4d8`, estimate
  12.4 min): one `wait_for_job(timeout_seconds=1116)` held 902 s of wall
  clock and returned `succeeded`, `waited_seconds: 900.0`, not capped.
- **Second verify** (`develop` @ `3926d430`, job `eeae7f9f25d9`, estimate
  13.6 min): one `wait_for_job(timeout_seconds=1224)` returned `succeeded`,
  `waited_seconds: 706.4`, not capped.
- **The clamp** on a finished job: 5000 and 1801 apply 1800 capped, 1800
  applies 1800 uncapped, and no timeout applies 20.
- **An unknown id** with `timeout_seconds=1800` fails at once with
  "Unknown job" (an empty id gets "Not Found"). It never blocks for the
  budget, which was the edge owed to #300.

## Bounces per stage

- **Stage 1 (#546): one bounce.** C-F173 failed because the `ltx-2.5` skill
  never named `timeout_applied_seconds`/`timeout_capped`, though the build
  comment said it did. The one-line fix had to fit 3 bytes under the size
  cap, so three other phrases in the skill were trimmed.
- One park that wasn't a bounce: the cap wasn't live on lem until Don
  replaced the installed systemd unit.
- **Stage 2 (#547):** not built.

No `usage:` figures were recorded on the stages, so cost is left out.

## Deferred

- **The heartbeat (stage 2).** Reopen it if a long wait is cut by a
  transport or idle timeout: Don's own ≥10-minute interactive wait (the
  plan's last acceptance line, his to check), or a client that cannot raise
  its idle limit. The design is #547's body.
- **Don's interactive line.** His own Claude Code session holding a
  ≥10-minute `wait_for_job` was his check, not a suite case. It depends on
  his session exporting `CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT` above the wait.
- **C-F171's "names the id" clause.** The error says "Unknown job" without
  the id. The plan didn't promise it, and the case amendment is
  dkackman/harnest#48.
- **The cursor and `failure_kind`:** cut, see
  `../declined/mcp-job-notifications.md`.
- **A job lost on restart** stays #300's.
