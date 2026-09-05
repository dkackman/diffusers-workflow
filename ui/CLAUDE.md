# ui

Guidance for the web UI single-page app.

`npm run build` outputs `ui/dist`, which the server serves — the SPA is not
served from source, so a UI change is invisible until it is rebuilt.

Front-end checks, run from `ui/`: `npm run check`, `npm run lint`, `npm test`,
and `npx playwright test` (e2e — it starts its own server, so do not have
`dw.serve` running on the same port).

See docs/SERVER.md.
