# ui

Guidance for the web UI single-page app. Build and dev setup are in ui/README.md; the look is encoded in the header comment of `src/app.css`, and the rules the UI shares with the engine are in `docs/ARCHITECTURE.md` and docs/SERVER.md.

Front-end checks, run from `ui/`: `npm run check`, `npm run lint`, `npm test`, and `npx playwright test` (e2e - it starts its own server, so do not have `dw.serve` running on the same port).

## Rules for new UI

- The UI reads engine-derived fields (`version`, `subfolder`, `plan`) and never computes or orders them itself.
- Show the proof: a list of things that produce images shows the images.
- Every token pair passes WCAG AA in both themes.
- `--live` appears in these places and nowhere else: the header's running link and VRAM pressure, a running job's row and status chip, the job page's progress bar and current step, the flow view's active step, a model download in flight, and the focus ring. Resting meters stay `--muted`; selection is the user's state, not the machine's, so it reads as a heavier ink edge.
