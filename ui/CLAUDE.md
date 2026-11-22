# ui

Guidance for the web UI single-page app. Build and dev setup are in ui/README.md; the look is in the header comment of `src/app.css`; the rules the UI shares with the engine - its response contract among them - are rows in `docs/ARCHITECTURE.md`.

Checks, from `ui/`: `npm run check`, `npm run lint`, `npm test`, and `npx playwright test` (it starts its own server - do not have `dw.serve` on the same port). Server response types come from `src/lib/generated/` - `python scripts/dump_openapi.py` then `npm run gen:api` - never written by hand.

## Rules for new UI

- A workspace's views are `#/ws/<name>/<section>`, listed in `WS_SECTIONS` (`src/lib/routes.ts`), routed in `src/App.svelte` and named in `src/lib/Sidebar.svelte`: overview, workflows, jobs, requests, gallery, compare, assets and edit. A new section goes in all three. **requests** is the workspace's requests ledger: `requests/<id>` is one request, edited only through intents (`src/lib/request/intents.ts`) that `planStore.svelte.ts` applies to the plan as last read when it writes - a plan read earlier is never sent again; what a row shows is derived in `src/lib/request/itemState.ts`, not in a component (`docs/ARCHITECTURE.md`, _Requests in the UI_).
- Show the proof: a list of things that produce images shows the images.
- Every token pair should pass WCAG AA in both themes.
- `--live` marks machine state and `--select` the user's place (focus, the open workspace, the active section), each nothing else; where they may appear is held by `scripts/design-rules.test.ts`.

## Keeping views fresh

What the server holds changes under an open page - an agent writes a request, a job finishes, another tab saves. The rule:

- **Live state keeps its own short poll**: a job in flight, a download, the server's status. It starts through `poll` (`src/lib/poll.ts`), never a bare `setInterval`.
- **Everything else reloads from the change feed**: `changes(workspace).watch(kinds, fn)` (`src/lib/changes.svelte.ts`) calls `fn` when one of the named kinds (`requests`, `prompts`, `workflows`, `assets`, `gallery`, `jobs`, or the server-wide `workspaces`) or the epoch moves. One read of `GET /api/workspaces/{name}/changes` every `CHANGES_POLL_MS` per workspace serves every watcher on screen; return the stop from an `$effect`. A page of server data never gets a timer of its own.
- **Every timer pauses while the tab is hidden and fires once on its return**, skips a tick while the last is pending, and survives a rejected one - `poll` does all three. The one exception is the app shell's status poll (`src/lib/statusPoll.ts`), which keeps going hidden: it drives the tab title and the job-finished alert, which exist for the hidden tab.
- **A reload from the feed is quiet**: no skeleton, no error over a list it already holds, and it keeps edits, focus and selection.

| View                                        | Live poll                                                                          | Feed watch                                                                                                     |
| ------------------------------------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| App shell (`App.svelte`)                    | status, 5 s, also while hidden (`pollStatus`)                                      | -                                                                                                              |
| Sidebar workspace list                      | -                                                                                  | `workspaces`: clears the cached list and reads it again                                                        |
| Requests (`RequestsPage`)                   | -                                                                                  | `requests`: quiet reload                                                                                       |
| One request (`RequestPage`)                 | 5 s while a job of it is queued or running, 20 s while open (`request/refresh.ts`) | `requests`, only while closed: a reopen shows within one feed read; reads keep the higher revision (`newerOf`) |
| Linked notes (`request/LinkedNotes.svelte`) | 20 s                                                                               | -                                                                                                              |
| Jobs (`JobsPage`)                           | 3 s                                                                                | -                                                                                                              |
| Models (`ModelsPage`)                       | 2 s while a download or a diffusers update runs                                    | -                                                                                                              |
| Server status (`ServerPage`)                | health and memory, 5 s                                                             | -                                                                                                              |
