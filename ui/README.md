# diffusers-workflow UI

Svelte 5 + Vite single-page app over the dw server's JSON API.

## Build (what `python -m dw.serve` serves at `/`)

```bash
export PATH=$HOME/.local/node/bin:$PATH   # user-local Node 24.20.0 LTS (no system install)
cd ui
npm install
npm run build        # -> ui/dist, auto-detected by the server
```

## Develop

```bash
python -m dw.serve                  # API on :8765
npm run dev                         # Vite dev server on :5173, proxies /api and /outputs
npm run check                       # svelte-check + tsc
```
