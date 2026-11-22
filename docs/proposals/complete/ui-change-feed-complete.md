# UI live updates by one rule: workspace change counters and a change feed (#1536)

Written by model `claude-opus-5-5` via provider `anthropic` (close-out,
2026-11-22, on mini-ai). Plan v2 approved by Don on 2026-11-21, three
stages: #1563 (server), #1564 (feed, Requests, sidebar, closed request) and
#1565 (the other list pages), all built and verified on mps on mini-ai.lan.

## The idea

Don's report (2026-11-21): the web UI is inconsistent about refreshing when
server data changes. A request an agent made over MCP didn't appear on the
Requests page until a reload. Agents write requests, notes, prompts,
workflows, assets and kept outputs all the time, so a page that loads once
shows a false "nothing new".

## What the sweep found (develop 437f079d)

- All live updating was interval polling through `ui/src/lib/poll.ts`, a bare
  `setInterval`: no hidden-tab pause, no overlap guard, rejected ticks
  swallowed. Only the notes channel (`request/listening.svelte.ts`) paused
  while hidden.
- Requests, Gallery, Assets, Prompts, Workflows, Overview and Compare loaded
  once. Prompts didn't even reload on a workspace switch. The sidebar's
  workspace list was a cached promise cleared only by this tab.
- A request page stopped refreshing once the request closed, so a reopen by
  an agent was never seen.
- The server had no change marker. The gallery and assets listings walk the
  tree on every call, so blind polling them was rejected.
- Every outside writer (MCP, `dw.run`, finishing jobs) goes through the
  server, so counters bumped at its write sites see every write the report
  names. Hand edits on disk are out of scope.

## Decisions

- **Q1 [scope]:** a polling route, `GET /api/workspaces/{ws}/changes`, not
  SSE. SSE comes back only if 5 s proves too slow in use (a `taste` call).
- **Q2 [default], amended by Don:** every UI timer pauses while the tab is
  hidden, **except the app shell's status poll**, which drives the tab title
  and the job-finished alert. It may slow while hidden but never stop, and a
  test pins it.
- **Q3 [default]:** a closed request page notices a reopen or edit through
  the feed.
- **Q4 [none]:** stage A verified by a plain HTTP GET beside MCP writes.
- **Rejected alternatives:** polling each list route (the tree rescans);
  refresh on focus only (misses a page left open beside a terminal); counters
  on `GET /api/health` (grows the MCP `get_health` reply for a UI-only need).

## The rule (as built)

Each view of server data does exactly one thing:
- **Live state** (a running job, a download, server health, a request with
  an item in flight) keeps its own short poll and stops when it settles.
- **Everything else** reloads when its workspace's counter for that kind
  moves, within one feed interval (`CHANGES_POLL_MS = 5000`), never on a
  timer of its own. The reload is quiet: no spinner or blanking, keyed rows,
  picks merge, an unsaved draft is kept, an overtaken answer is dropped.

The page table lives in `ui/src/lib/changes.svelte.ts` (named from
`ui/CLAUDE.md`); `docs/ARCHITECTURE.md` carries the *workspace change
counters* row and the reworded UI polling and Requests rows;
`docs/SERVER.md` documents the route and how list pages refresh.

## What was built

**Stage A, #1563** (server): `bb5b757b`, `7e941b7b` (develop `d2fd28db`);
bounce fix `ebededd2` (develop `c29d28e0`).
- `dw/server/changes.py`: an in-memory registry under one lock. A snapshot is
  `{epoch, workspaces, kinds}` with six growing counters (`requests`,
  `prompts`, `workflows`, `assets`, `gallery`, `jobs`); `epoch` is random per
  server start, so a restart reads as "everything changed".
- Bumps at every server write site, including the job-finished path in the
  worker manager. A refused write bumps nothing.
- The route with its `WorkspaceChanges` response model, the regenerated
  OpenAPI and TS types, the surface snapshot, and a parametrised test that
  walks the write routes and asserts exactly the right kind moved. Not on
  MCP.

**Stage B, #1564** (UI): `25a39fdd`, `c38dac14`, `f7373375`, `189af586`,
`512cba46` (develop `e514f2d0`).
- `poll.ts`: pauses hidden and fires once on return (opt-out used only by the
  app shell's status poll), skips an overlapping tick, survives a rejection.
- `ChangeFeed` (`ui/src/lib/changes.svelte.ts`): one feed per workspace,
  counted holders, `watch(kinds, fn)`, overtaken replies dropped.
- Requests page, sidebar workspace list and the closed request page watch the
  feed. Vitest for both modules and the status poll firing hidden; a
  Playwright spec pins the reported case (an outside write shows on the open
  Requests page).

**Stage C, #1565** (UI): `00ee6bce` (develop `1e353ae0`).
- Gallery (`gallery`), Assets (`assets`, no `keepOnly` on a background
  reload), Prompts (`prompts`, plus the missing workspace dependency),
  Workflows (`workflows`; `gallery` reloads only the proofs), Overview (each
  kind reloads only its own panel, plus `workspaces`), Compare (`gallery`;
  cells keep metadata and playback). JobsPage keeps its own 3 s poll, which
  now pauses hidden.
- Tests rewritten from "exactly once" to "once, again on its kind, not on
  another kind or workspace", with picks, drafts and workspace-switch tests.

## Deferred

- **C-F1237** (hidden tab: the feed and jobs reads pause, the status poll
  keeps firing, the page catches up on return) stays `pending: #1564`. The
  `ui-check` runner can't yet hide and show the page or idle-wait, so the case
  reports blocked. The behaviour is pinned by vitest. Closing it is harness
  work (runner steps) for Don to file; ruled by the plan-review delegate
  2026-11-22.
- The "adds a note" half of the closed-request intent: agents can't write
  request notes over MCP, so only a reopen through `set_request` is covered.
- Push delivery (SSE) and cheaper gallery/asset listings: non-goals; each
  would be its own issue.

## Cost and bounces

No stage comment states `usage:` figures, so cost is left out. The plan
estimated $14–21 across the three stages.

| Stage | Bounces | Why |
| --- | --- | --- |
| #1563 A | 1 | architecture review: `changed_assets` re-derived "is this asset in the shared library" with a lexical path check, a second owner beside `LibraryPath`; fixed in `ebededd2` by passing the library root's origin from `library.find` |
| #1564 B | 0 | the hand-off gate refused the first merge (`test_ui_twins` couldn't see the ledger routes behind a shared URL template); fixed before hand-off in `512cba46`. Closed on 7 of 8 cases, C-F1237 left pending (above) |
| #1565 C | 0 | |
