# Agent Loop

Some Issues in this repo are worked by an automated agent loop, not a human.
The harness is [`dkackman/harnest`](https://github.com/dkackman/harnest); its
[ticket protocol](https://github.com/dkackman/harnest/blob/main/docs/harness/ticket-protocol.md)
is the full rule set this page summarizes, and `lib/classify.jq` there is the
authoritative map from labels to the session that acts next. Nothing about
running the loop lives here. This page is for anyone, human or agent, who
opens or comments on one of its Issues.

## The two roles

- **Implementer**: has this repo checked out and SSH access to the box running
  the MCP server. Triages, reproduces, fixes, deploys and hands the ticket
  back. Never verifies its own fix.
- **Tester**: talks to the MCP server only as a protocol client, with no
  source checkout and no shell on the server box. It re-runs the original
  repro over MCP and either confirms the fix or bounces the ticket.

The only role that can mark an issue verified is the one that cannot patch
around a bug. Preserve that: do not fix and verify the same ticket yourself.

Other sessions in the harness:

- **Lead** (`owner:lead`): designs a feature with Don, turns an `idea` into a
  feature or a single fix, or closes it.
- **Docs reviewer**: read-only; closes a `docs-review` ticket.
- **Regression agent**: runs the scripted MCP suites (`regression-suite-*.md`
  in harnest) against the live server and files `regression` and
  `performance` issues. It takes no part in the implementer/tester hand-off.

## Outside filings

The loop runs as the repo owner's GitHub login and must not act unattended on
text anyone else filed in this public repo. An issue from another login,
including a `field-report` from someone else's session, is relabelled
`owner:don` + `status:needs-approval` before any role works it, and reaches
the loop only if the maintainer hands it back.

**Security findings are never public issues.** A boundary that did not hold
(a gate, a path, a URL, a disclosure, a destructive call without
acknowledgement) goes to a draft GitHub security advisory. No agent may add
the `security` label.

## Reading a ticket

Tickets use the **MCP agent-loop ticket** issue template (ideas use **MCP
agent-loop idea**). Two label families carry the state; check both before
acting.

**`owner:*`** is exactly one of `owner:implementer`, `owner:tester`,
`owner:lead` or `owner:don`: whoever acts next. If it is not you, leave the
issue alone beyond reading it. The owner label is the only signal: a status
addressed to Don is stale once he swaps the owner back.

**`status:*`** is where the ticket is:

| Label | Meaning |
| --- | --- |
| *(none)* | Open, ready for the implementer to reproduce |
| `status:fixed-pending-verify` | Fixed and deployed; the tester re-runs the repro over MCP |
| `status:verified` | The tester confirmed the fix over MCP; closed as `completed` |
| `status:reviewed` | A `docs-review` fix closed by the docs reviewer (never `status:verified`) |
| `status:needs-info` | A question bounce; whoever is asked answers before work continues |
| `status:needs-approval` | With `owner:don`: parked for the human, after a written proposal. Required for engine or syntax changes and anything breaking beyond a rename. Neither agent touches it |
| `status:plan-review` | A feature plan is posted and waiting for Don |
| `status:plan-approved` | Don approved the plan. Only Don adds it |
| `status:needs-spec` | The tester writes acceptance cases for the plan's stages |

Labels that close a ticket without a fix:

- **`wontfix`**: the implementer's call, with a reason in a comment; closed as
  `not planned`. The tester may reopen once with new evidence; a second
  `wontfix` is final.
- **`duplicate`**: closed as `not planned` with `duplicate of #NN` in a
  comment. Closed issues count for duplicate detection
  (`gh issue list --state all`). A verified issue that reproduces again is not
  a duplicate: it becomes a new `regression` issue (`regression of #NN`), and
  nobody reopens the old one.

Other labels:

| Label | Meaning |
| --- | --- |
| `docs-review` | A fix to text neither the server nor the plugin serves; the docs reviewer verifies it instead of the tester |
| `breaking-change` | The fix changed the MCP interface; the tester adjusts its calls rather than filing a new bug |
| `field-report` | Filed from Don's own dw sessions, not the harness |
| `regression`, `performance` | Filed by the regression agent |
| `idea` | An idea for the lead to assess |
| `feature`, `stage` | Work designed with Don and built in stages; a `stage` is a child issue of one feature |
| `lane:batch` | A plan review with no question of a kind Don decides, approved in a batch |
| `parked` | Deferred: on `owner:lead`, until every blocking issue closes; on `owner:don`, Don's |
| `release`, `release-blocker` | While an open `release` issue exists, only `release-blocker` issues move |
| `stabilization`, `arch-approved` | dw's stabilization freeze (in force while `docs/stabilization/FREEZE` exists on `develop`): work parked for it, and new surface Don approved during it |
| `arch-review` | A hand-off waiting on the architecture reviewer, who checks it against `docs/ARCHITECTURE.md` and passes or bounces it |
| `consolidation` | Filed weekly from `scripts/arch_report.py`; Don plans it (to `owner:lead`) or closes it |
| `backend:shared`, `backend:cuda`, `backend:mps` | What the bug is about |
| `target:lem`, `target:mini-ai`, `target:minid`, `target:local` | Which server's loop claims it; a non-lem target is for Don to triage |
| `verified-on:mps` | Verified on the Mac only; CUDA has not re-verified the fix |
| `priority:2`, `priority:3` | Backlog rank |

## If you want to participate

- Only touch an issue owned by you (or, for a human, one parked with
  `owner:don`).
- To hand a ticket on, swap the `owner:*` label and say what you did in a
  comment. The next actor has no memory of your session, only the thread.
- Keep implementer and tester separate. Only the tester (or an equally
  independent verifier) closes an issue as `completed` with
  `status:verified`, and only after a real MCP call shows the fix, not by
  reading the diff. The one exception is `docs-review`.
- Reference the issue in the commit that fixes it, with a scope
  (`fix(mcp): #42 - ...`), on a branch merged to `develop`, never `master`.
