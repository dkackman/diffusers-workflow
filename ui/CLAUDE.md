# ui

Guidance for the web UI single-page app. Build and dev setup are in ui/README.md; the look is in the header comment of `src/app.css`; the rules the UI shares with the engine - its response contract among them - are rows in `docs/ARCHITECTURE.md`.

Checks, from `ui/`: `npm run check`, `npm run lint`, `npm test`, and `npx playwright test` (it starts its own server - do not have `dw.serve` on the same port). Server response types come from `src/lib/generated/` - `python scripts/dump_openapi.py` then `npm run gen:api` - never written by hand.

## Rules for new UI

- Show the proof: a list of things that produce images shows the images.
- Every token pair should pass WCAG AA in both themes.
- `--live` marks machine state and `--select` the user's place (focus, the open workspace, the active section), each nothing else; where they may appear is held by `scripts/design-rules.test.ts`.
