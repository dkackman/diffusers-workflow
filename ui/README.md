# diffusers-workflow UI

Svelte 5 + Vite single-page app over the dw server's JSON API. Needs Node >= 24.20.0 (`ui/.nvmrc`). Rules for new UI are in `ui/CLAUDE.md`.

## Build (what `python -m dw.serve` serves at `/`)

```bash
cd ui
npm install
npm run build        # -> ui/dist, auto-detected by the server
```

The server serves `ui/dist` in a checkout, else the copy packaged into the wheel (`dw/server/ui`).

## Develop

```bash
python -m dw.serve                  # API on :8765
npm run dev                         # Vite dev server on :5173, proxies /api and /outputs to :8765
```

## Checks

```bash
npm run check                       # svelte-check + tsc
npm run lint                        # eslint
npm run format                      # prettier --write
npm test                            # vitest
npx playwright test                 # e2e; starts its own server on :8971
npm run preflight                   # all of the above, plus the ratchet and build
```

Server response types in `src/lib/generated/` are generated, never written by hand: `python scripts/dump_openapi.py`, then `npm run gen:api`.
