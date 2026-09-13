# Agent Loop

Some of the Issues in this repo are worked by an automated implementer/tester
agent loop, not a human. It lives in a separate repo
([`dkackman/iterate`](https://github.com/dkackman/iterate) — private) that
drives two Claude Code agents in strict alternation against this repo's
Issues. Nothing about running the loop lives here; this page exists so that
anyone who opens or comments on one of its Issues — a human, or another
agent joining in — knows how to act on it correctly.

## The two roles

- **Implementer** — has this repo checked out and SSH access to the box
  running the MCP server. Reproduces, fixes, deploys, and hands the ticket
  back. Never verifies its own fix.
- **Tester** — talks to the MCP server only as a protocol client (MCP tool
  calls), no source checkout, no shell/SSH access to the server box. Its job
  is independent verification: it re-runs the original repro over MCP and
  either confirms the fix or bounces the ticket back.

That asymmetry — the only role that can mark an issue verified is the one
with no ability to patch around a bug — is the entire point of the loop. A
human or third agent joining in should preserve it: don't fix and verify the
same ticket yourself.

A third, standalone **regression agent** periodically runs a growing suite
of scripted MCP calls (`regression-suite-*.md` in the `iterate` repo) against
the live server and files/comments on Issues for anything that regresses. It
doesn't participate in the implementer/tester handoff.

## Reading a ticket

Tickets use the **MCP agent-loop ticket** issue template. Two label
families carry all the state — check both before acting:

- **`owner:*`** — exactly one of `owner:implementer` / `owner:tester` /
  `owner:don` at a time: whoever is expected to act on it next. If you're
  not that owner, leave the issue alone beyond reading it (or the specific
  handoff comment/label a role prompt allows).
- **`status:*`** — where the ticket is in its lifecycle:
  - *(no status label)* — open, ready for the implementer to reproduce.
  - `status:fixed-pending-verify` — implementer has fixed and deployed;
    waiting on the tester to re-run the repro over MCP.
  - `status:verified` — tester confirmed the fix over MCP; issue is closed
    as `completed`.
  - `status:needs-info` — a question bounce; whoever is asked needs to
    answer before work continues.
  - `status:needs-approval` (+ `owner:don`) — parked for the human. The
    implementer uses this for anything beyond a rename-level change
    (engine behavior, breaking syntax). Neither agent touches a parked
    issue.

Two built-in GitHub labels close a ticket without a fix:

- **`wontfix`** — the implementer's call, with a reason in a comment; issue
  closed as `not planned`. The tester may reopen once with new evidence; a
  second `wontfix` is final.
- **`duplicate`** — closed as `not planned`, with a comment naming the
  issue it duplicates (`duplicate of #NN`). A closed issue is still
  canonical for duplicate detection — the implementer checks
  `gh issue list --state all` before starting work, not just open issues.

`breaking-change` marks an Issue whose fix changed the MCP interface, so the
tester adjusts its calls instead of filing the change as a new bug.

## If you want to participate

Whether you're a human or another agent:

- Only touch an issue that's currently owned by you (or, for a human, one
  parked with `owner:don`).
- When you hand a ticket to the next owner, swap the `owner:*` label and
  say what you did in a comment — the next actor has no memory of this
  session, only the issue thread.
- Keep implementer and tester roles separate. If you're fixing code, don't
  also close the issue as verified — that requires an independent MCP call
  from someone who didn't write the patch.
- Reference the issue number in any commit that fixes it
  (`fix(mcp): #42 - ...`), and work on a branch merged to `develop`, never
  `master`.
- Only the tester (or an equivalently independent verifier) closes an issue
  as `completed`/`status:verified`, and only after a real MCP call
  reproduces the fix — not by reading the diff.

See the `iterate` repo's `CLAUDE.md` for the full protocol this is
summarized from, including how the automated loop itself is run.
