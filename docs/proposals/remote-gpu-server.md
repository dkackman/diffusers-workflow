# Proposal: running dw entirely on a dedicated GPU server

Status: **not committed** - scoping notes written against a real ask (running
`dw.serve` on a separate machine with the GPU, connecting to it from a
browser and from Claude Code's MCP client on a different machine day to
day). Nothing here should be implemented without a fresh look at whether the
manual workaround (SSH tunnel, documented below) is good enough in practice.

## What already works, unmodified

This is closer to "already supported, thinly" than "not supported."
`dw.serve --host 0.0.0.0 --token "..."` plus an SSH tunnel or LAN reach
already gets a browser talking to a remote GPU box today - see
[SERVER.md](../SERVER.md)'s Authentication section. What's missing is doing
the same for the MCP client, and tightening the network-facing edges that
currently assume "your own GPU to your own browser" (`dw/serve.py:6`)
rather than "a GPU box you deliberately expose to one or two trusted
clients."

- **Bind address is already a flag.** `--host`/`--port` (`dw/serve.py:23-27`)
  need no change to listen on `0.0.0.0` or a specific LAN interface.
- **Token auth already exists and covers the REST/SSE/UI surface.**
  `DW_API_TOKEN`/`--token` (`dw/serve.py:44-49`, `docs/SERVER.md:220-249`) -
  constant-time bearer check on `/api/*`, with the three download/thumbnail/
  SSE routes accepting `?token=` for the cases a browser can't attach a
  header. This is a single shared static secret, not a login system - true
  today and still true after anything proposed below.
- **`Origin`/`Host` checks are already DNS-rebinding-aware for a wildcard
  bind** (`docs/SERVER.md:186-200`) - nothing to add there.
- **The web UI already has a token field** (`docs/SERVER.md:246-249`) -
  storing the token in `localStorage`, attached to every call. Pointing a
  browser at a remote box today is: open the token field, paste it, done.

## Gaps

### 1. `dw-mcp` is explicitly told not to do this

[MCP.md](../MCP.md#security) is direct about it:

> Because it adds no authentication, `dw-mcp` must not be pointed at a
> `dw.serve` reachable beyond localhost. Treat `--url`/`DW_MCP_URL` the same
> way you would treat opening the web UI to the network: don't.

That statement predates `--token` being usable end-to-end for this case, but
re-reading the code, it's *only partially* stale: `DwClient` (`dw_mcp/
client.py:70-76`) already accepts a `token` and threads it as `Authorization:
Bearer` on the `httpx.Client` it builds, and `dw-mcp --token`/`DW_API_TOKEN`
(`dw_mcp/__main__.py:22-26`) already wires that through. So a token-
authenticated `dw-mcp --url http://<box>:8765 --token ...` pointed at a
LAN/tunnel-reachable `dw.serve` is not actually unauthenticated today - the
docs warning is broader than the current code strictly requires. What's
still true and still a real gap:

- The warning itself needs rewriting once (if) this is scoped, so it says
  "don't do this without a token" rather than "don't do this."
- `dw-mcp` has no equivalent of the UI's "the page won't load without a
  token" self-check - point it at a token-protected remote server with no
  `--token` given, and every tool call just 401s one at a time rather than
  failing fast at startup with a clear message.
- Nothing about `dw-mcp`'s stdio transport changes here. It still runs next
  to Claude Code (locally, wherever Claude Code spawns it) and reaches out
  to the remote `dw.serve` over HTTP - this section is about making that
  HTTP leg safe to point at a non-loopback host, not about relocating
  `dw-mcp` itself. (Running `dw-mcp` *on* the GPU box as a standalone HTTP-
  transport service is a materially different, larger change - see
  "Related, out of scope" below.)

### 2. No TLS story

Both the token and every job's arguments/results travel in plaintext once
the bind address is non-loopback. An SSH tunnel (the setup already in use
informally) sidesteps this by never putting plaintext on a real network
segment - the tunnel's local end is still `127.0.0.1` to `dw.serve`. Binding
`--host 0.0.0.0` directly and reaching it by LAN IP or a tunnel-free network
path has no equivalent protection: the bearer token in the `Authorization`
header (and in the three `?token=` URLs) is readable to anything that can
see the traffic. `dw.serve` has no built-in TLS termination, and adding one
(cert management, `uvicorn`'s `--ssl-keyfile`/`--ssl-certfile`, or
documenting a reverse-proxy pattern instead) is a real, non-trivial decision
- see Proposed scope.

### 3. No "is this server remote" signal anywhere in the UI or MCP tools

Once a browser or `dw-mcp` is pointed at a remote box, nothing in the
response surface says so. `get_health` (MCP) / `GET /api/health` (REST)
report worker liveness but not host identity; there's no equivalent of
`whoami`/`hostname` a client could use to confirm "yes, this is the 3090
box" versus a stale tunnel pointed at nothing. Minor, but worth a line in
the health payload if this gets built out, since a remote setup is exactly
the situation where "which box am I actually talking to" stops being
obvious from context.

### 4. Long-running server lifecycle on a headless box

Today `dw.serve` is a foreground process a developer starts by hand in a
terminal (`docs/SERVER.md`'s examples are all bare `python -m dw.serve`).
Run "entirely on a dedicated GPU server" implies the process survives an SSH
disconnect, restarts after the box reboots, and its logs are somewhere a
person can find them without a live terminal attached - none of which is
dw's concern to build (this is systemd-unit/launchd-plist territory, not
application code), but worth naming as part of "the setup" even though it's
zero lines of code in this repo.

## Proposed scope

Two small code changes, everything else is documentation/ops:

### A. Fail fast in `dw-mcp` when talking to a non-loopback host with no token

At `DwClient` construction (`dw_mcp/client.py:70-76`) or in `dw_mcp/
__main__.py:main()`, check: if `resolve_base_url(...)`'s host is not
loopback and no token was resolved, print a clear stderr warning (or refuse
to start, mirroring `dw.serve`'s own non-loopback-without-token startup
warning at `docs/SERVER.md:189`) instead of leaving every subsequent tool
call to fail as an unexplained 401. Small, self-contained, `dw_mcp/client.py`
+ `dw_mcp/__main__.py` only.

### B. Rewrite the MCP.md security warning

Replace the blanket "don't" with the actual constraint: a token-
authenticated remote `dw.serve` is supported the same way the web UI
supports it, plaintext transport risk and all - point `dw-mcp` at a remote
host only over a channel you already trust (SSH tunnel, VPN, or a LAN you
control), the same guidance `docs/SERVER.md`'s Authentication section
already gives for the browser case. Documentation-only, `docs/MCP.md`.

### Explicitly out of scope for phase 1

- **TLS termination inside `dw.serve`.** The tunnel already solves transport
  security for the primary case (SSH access to the box); a from-scratch TLS
  story is a bigger, separate decision (self-signed vs. real certs, cert
  rotation, whether to punt to a reverse proxy like Caddy/nginx in front of
  uvicorn instead of adding cert flags to `dw/serve.py` itself). Worth its
  own proposal if the LAN-without-tunnel case becomes a real requirement
  rather than an occasional convenience.
- **Multi-client / multi-user auth.** The single shared static token is a
  deliberate, already-documented limitation (`docs/SERVER.md:254-257`).
  Nothing here changes that model - "connect from browser and agent
  terminal" is two clients sharing one token, which the existing model
  already handles.
- **A process supervisor / systemd unit shipped in this repo.** Lifecycle
  management (section 4 above) is host-specific and belongs in the user's
  own dotfiles/ops setup, not in `dw/` or `pyproject.toml`.
- **Relocating `dw-mcp` itself onto the GPU box as a standalone HTTP-
  transport MCP service** (as opposed to running it locally and pointing it
  at a remote `dw.serve`). That's a different mechanism - `dw_mcp/server.py`
  only wires up `mcp.server.stdio` today (`dw_mcp/__main__.py:38`) - and a
  materially larger change (new transport, its own auth story, a persistent-
  service lifecycle for `dw-mcp` itself) than what this document scopes.
  Worth a separate proposal if a subprocess-per-Claude-Code-session model
  stops being good enough.

## Recommended setup today, no code changes

Since (A) and (B) above are small and this is genuinely mostly-already-
supported, the practical setup right now, before any of this is built:

```bash
# On the GPU box
python -m dw.serve --token "$(openssl rand -hex 32)"   # stays on 127.0.0.1

# On the client machine
ssh -L 8765:127.0.0.1:8765 user@gpu-box -N -f
export DW_API_TOKEN="<same token>"
dw-mcp                      # defaults to http://127.0.0.1:8765, now tunneled
# browser: http://127.0.0.1:8765, paste the token once in the UI's token field
```

This needs the token even though the tunnel already authenticates by SSH key
- the token is what stops anything else on the GPU box's loopback interface
(other users, other processes) from reaching the API through the tunnel's
local end. Everything in "Proposed scope" above is about making the direct
(`--host 0.0.0.0`, no tunnel) variant of this equally safe and equally
well-signposted, not about enabling something that doesn't work today.

## Open questions

- Is the SSH-tunnel setup (works today, zero code changes) actually
  insufficient for the day-to-day flow, or does this proposal's real value
  turn out to be just (A) and (B) - two small papercut fixes - rather than
  anything bigger?
- If a from-scratch TLS story ever gets scoped, does it belong in
  `dw/serve.py` (uvicorn's own `--ssl-keyfile`/`--ssl-certfile`) or is the
  right answer "document a reverse-proxy pattern and explicitly decline to
  add cert handling to the application"?
- Does the per-plugin-trust-list work in
  [plugin-extensibility.md](plugin-extensibility.md) intersect with this at
  all (a remote, less-trusted box being a reason to want per-plugin trust
  rather than the single `--trust-workflows` switch)? Worth a cross-check
  once/if both move forward, not blocking either individually.

Citations are file:line references at the time of writing and may drift.
