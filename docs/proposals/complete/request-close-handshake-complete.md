# Request close handshake: the agent proposes, the user closes (#1604)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out session,
2026-11-26). Plan v2 on #1604 is the full design. This file records what
shipped, what was left out, and why.

**Result:** only a user write closes a request. An agent's `set_request` (or
`PUT /requests/<id>` with `by: agent`) that moves an open request to `done` or
`abandoned` is refused with 422 `close_needs_user`. The agent proposes instead,
with `propose_close: {state, reason}`. The request stays open, carrying a
`close_proposal` whose `outcome` is `pending`, `accepted` or `declined`. The
request page shows the proposal as a banner with **Close** and **Keep open**
(with an optional note). The Requests list and Overview tag a row that has a
pending proposal "close proposed".

## The problem

A request's state (`dw/ledger.py`, `open | done | abandoned`) was checked only
for membership in `STATES`, so an agent could close a request on its own. The
user would then reopen it, and the agent would close it again. A closed request
didn't mean the user had agreed. The evidence was one field report, the issue
itself.

## Design as built

- **A field, not a fourth state.** `close_proposal` is kept across writes that
  leave it out, as notes are. `STATES`, the open-list filter, `_check_room`,
  `link_problem` and the UI's state readers are unchanged.
- **The rule keys on the self-declared `by`.** The server can't tell the UI
  from an MCP caller: both use one bearer token, and the mounted MCP calls
  `/api` over HTTP. MCP writes declare `by: agent` and the UI declares
  `by: user`. This rule is for a cooperative agent, not a security boundary.
  An agent that sends `by: user` over REST can still close a request, and
  that is out of scope by design.
- **Writes.**
  - Agent `propose_close {state, reason}`: the reason is one line of at most
    200 characters. The new proposal replaces any earlier one, declined ones
    included. On a closed request it is refused with `request_closed`.
  - Agent `propose_close: null` withdraws the proposal.
  - User `close_proposal {outcome: declined, note?}`: settles a pending
    proposal. With nothing pending it is refused with `no_close_proposal`.
  - User `state: done|abandoned`: closes the request and marks a pending
    proposal `accepted`.
  - The agent may still reopen a request (`state: open`) and still edit a
    closed one's plan if it sends no state.
- **History** records `close_proposed`, `close_declined` and `close_accepted`.
- **The read side.** `get_request` shows the proposal as stored.
  `dw/ledger_views.py` adds `close_proposed` to list rows for the tag
  (`LedgerSummary.close_proposed`).
- **Docs and skills.** The rule is stated in `docs/WORKSPACES.md`,
  `docs/SERVER.md`, `docs/MCP.md`, `docs/ARCHITECTURE.md` (a ledger row and
  its pin) and the `script-to-video` skill's request reference (plugin
  `0.23.0-alpha.2`). `docs/AGENT_RULES.md` tells the agent not to re-propose
  after a decline unless the plan or its takes changed.

## Rulings

- Don approved plan v2 on 2026-11-25 with the delegate's recommendations.
  - **Q1:** refuse an agent's close.
  - **Q2:** abandoning also needs the user.
  - **Q3:** the agent proposes through `propose_close` on `set_request`.
  - **Q5:** a decline reaches the agent through `get_request` only.
  - **Q4:** the alternative. Re-proposal after a decline is guidance only.
  - Only stage A was cleared to build.
- Stage B was held until stage A closed. Don approved building it on
  2026-11-26.

## Stages

- **Stage A, #1716 (ledger, route, MCP, docs):** two bounces.
  - First hand-off: the tester bounced C-F1388 (the served skill still said
    "close", so the plugin version bump was needed) and C-F1382.
  - Second: the architecture review found an unpinned copy of the ledger's
    close states in `dw_mcp/tools_authoring.py` (`PROPOSE_CLOSE`), which is
    now pinned in `tests/test_mcp_twins.py`.
  - The third hand-off passed every case the runner could run over MCP.
    Don ruled that the stage closes on those cases. C-F1384 to C-F1387 kept
    `pending: #1716`, because the tester can't send a user write.
- **Stage B, #1717 (the UI banner, the Keep open note, the list tag):** no
  bounces. The architecture review passed.
  - The verify passed C-F1389 to C-F1392 with `ui-check`.
  - It couldn't run C-F1384 to C-F1387 (user PUTs) or C-F1393 (a 409 answer
    to a request PUT): `ui-check` blocks request PUTs and can't stub them
    (harnest #310).
  - Don closed the stage on that verify.

No stage comment gave a `usage:` figure, so this record leaves cost out. The
plan estimated about $4–6 per stage.

## Left out, and what would bring it back

- **A server-side re-proposal gate** (`close_declined`, Q4): comes back if
  field reports show agents re-proposing after a decline.
- **A decline as a notes-channel event** (Q5): comes back if agents miss
  declines.
- **Telling the UI from the agent by credential:** it would need a second
  token, a `trust` design of its own.
- **The user-side cases** C-F1384 to C-F1387 and C-F1393: they wait on a
  runner that can send and stub a request PUT (harnest #310).
