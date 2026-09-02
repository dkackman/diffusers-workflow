# Split Front End and Back End — Scoping Doc

> Status: scoped against the code on 2026-09-02. Not designed and not
> started. Written while finishing the desktop installer work
> ([design](../specs/2026-09-01-desktop-installers-design.md)), to
> capture why HTTP MCP transport was deliberately *not* included there.

## Goal

Run the UI on one machine and the engine on another: a laptop driving a
GPU box on the same network, rather than everything on one host.

## What already works

Three things make this less work than it looks.

**The SPA uses relative URLs.** `ui/src/lib/api.ts` calls `fetch(path)`,
`/outputs/<name>`, and `EventSource('/api/jobs/<id>/events')` — never an
absolute base. So the natural architecture is that the GPU box serves the
SPA it already serves, and the laptop points a browser or the desktop
shell's webview at it. **No front-end change is required at all.** Any
design that instead bundles the SPA on the client and calls a remote API
cross-origin gives up that property and buys CORS, CSP and an absolute
API base in exchange for nothing.

**`dw.serve` already takes `--host`.** Binding a LAN address is an
argument, not a code change.

**MCP already reaches a remote engine.** `dw_mcp` is an HTTP client of
`dw.serve`, and `dw-mcp --url http://gpu-box:8765` works today. This is
the reason HTTP MCP transport is not on this path — see below.

## What blocks it

**There is no authentication of any kind.** No `CORSMiddleware`, no
dependency guard, no token anywhere in `dw/server/app.py`. That is
entirely correct for a server bound to `127.0.0.1`, which is why the
desktop shell hardcodes `--host 127.0.0.1` in `server.rs`. It is also
the whole problem: the moment the engine binds a LAN address, anyone who
can reach the port can

- queue jobs on the GPU,
- read and delete generated output,
- delete models from the Hugging Face cache (`dw/hub_cache.py`),
- write and delete stored workflows and prompts,
- and trigger `POST /api/models/update-diffusers`, which runs
  `pip install` as the server user (`dw/server/updater.py`).

That last one is remote code execution by design — it is a perfectly
reasonable thing for a localhost server to offer and an unacceptable
thing for an unauthenticated network service.

**The desktop shell assumes it owns the engine.** `Shell` holds a
`Supervisor`, and startup either provisions or starts a local venv.
Pointing at a remote server is a third startup path, not a variation of
those two.

**Two local-only mechanisms stay local-only.** The `server.json`
handshake (`ports.rs` → `dw_mcp/client.py`, `dw/repl.py`) is a
filesystem contract and is meaningless across machines; a remote
configuration has to carry an explicit URL. And the REPL's
already-running warning is about contending for one GPU, so it should
keep probing the *local* server, not a configured remote one.

## Sequence

Deliberately ordered — each step is unsafe before the one above it.

1. **Authentication.** Decide the mechanism (a bearer token in settings,
   or an explicit "terminate TLS and auth at a reverse proxy" contract)
   and, more importantly, decide which endpoints are privileged.
   `update-diffusers`, model deletion, and workflow/prompt writes are not
   the same risk class as listing workflows.
2. **Bind address.** Plumb `--host` through the shell, and refuse a
   non-localhost bind unless auth is configured. Failing loudly is the
   point: the dangerous outcome is a LAN-exposed engine that nobody
   realized was open.
3. **Shell remote mode.** A startup path that skips provisioning entirely
   and points the webview at a configured URL. Contained, but it does
   reshape `Shell` and the provisioning screen.
4. **HTTP MCP transport.** Only now, and only if wanted. `dw_mcp/server.py`
   currently calls `run(transport="stdio")`; the SDK supports streamable
   HTTP. It is genuinely nicer than stdio for a remote box — no local
   process to spawn, no absolute paths in client config.

## Why HTTP MCP is last, not first

It is tempting to treat HTTP MCP as *the* remote feature, because it is
the one with "HTTP" in the name. It is not on the critical path:

- It changes how an MCP **client** reaches `dw-mcp`. It does not change
  how `dw-mcp` reaches the engine, which is already HTTP and already
  remote-capable via `--url`.
- Added before step 1, it puts a **second** unauthenticated entry point
  on a box that is about to be exposed — one that wraps the same
  privileged endpoints, including the pip-invoking one.

So it is a convenience on top of a secured remote engine, not the thing
that makes a remote engine possible.

## Non-goals

- Multi-user or multi-tenant operation. The job queue is FIFO and
  single-worker by design; this is one person's GPU, reached from
  elsewhere.
- Exposure to the public internet. LAN or a private network, with TLS
  and auth handled at a proxy if it ever leaves one.
- Any change to `ui/`.
