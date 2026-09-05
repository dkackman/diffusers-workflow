# ui

Guidance for the web UI single-page app.

`npm run build` outputs `ui/dist`, which the server serves — the SPA is not
served from source, so a UI change is invisible until it is rebuilt.

Front-end checks, run from `ui/`: `npm run check`, `npm run lint`, `npm test`,
and `npx playwright test` (e2e — it starts its own server, so do not have
`dw.serve` running on the same port).

Every request is scoped to the selected workspace in one place: `setApiWorkspace`
in `lib/api.ts` appends `?workspace=` to requests and to the `/outputs`,
`/inputs` and download URLs, so no page threads a workspace through its calls.
`lib/workspace.svelte.ts` holds the selection (restored from localStorage in
`main.ts` before the first request); a page refetches on a switch by reading
`workspace.current` inside its load effect.

See docs/SERVER.md.
