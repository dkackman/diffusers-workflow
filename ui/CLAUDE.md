# ui

Guidance for the web UI single-page app.

`npm run build` outputs `ui/dist`, which the server serves — the SPA is not
served from source, so a UI change is invisible until it is rebuilt.

Front-end checks, run from `ui/`: `npm run check`, `npm run lint`, `npm test`,
and `npx playwright test` (e2e — it starts its own server, so do not have
`dw.serve` running on the same port).

Every request is scoped to the selected workspace in one place: `scoped()` in
`lib/api.ts` reads `workspace.current` and appends `?workspace=` to request
paths, download URLs and the `/outputs` URLs `outputUrl` builds; a `url` the
server returns (an upload's preview, an asset entry) is used verbatim and the
server is responsible for scoping it, so no page threads a workspace through
its calls. The exception is a file that belongs to a *job*: `outputUrl` and
`fetchOutputText` take an explicit workspace, which wins over the picker, so a
job's files load from where they were written.
`lib/workspace.svelte.ts` holds the selection (restored from localStorage in
`main.ts` before the first request); a page refetches on a switch by reading
`workspace.current` inside its load effect.

See docs/SERVER.md.
