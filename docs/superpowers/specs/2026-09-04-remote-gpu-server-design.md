# Remote GPU server: design

Date: 2026-09-04. Status: approved for planning. Supersedes the scoping notes
in [docs/proposals/remote-gpu-server.md](../../proposals/remote-gpu-server.md),
which is updated alongside this spec to record the decisions.

## Goal

Run `dw.serve` as a standalone, long-lived service on the machine that has
the GPU, and use it from other machines on the local network with **nothing
installed on the client**: a browser for the web UI, and an agent terminal
(Claude Code) connecting to MCP over HTTP. The same shape must later be
exposable beyond the LAN (NAT) without redesign - by fronting it with a
TLS-terminating reverse proxy or an overlay network, not by adding TLS to
the application.

## Decisions

| Question | Decision |
| --- | --- |
| Where does MCP run? | Both. The stdio `dw-mcp` stays for a client with a local install; `dw.serve --mcp` additionally mounts a Streamable-HTTP MCP endpoint at `/mcp` on the same port, so a remote agent needs no install. |
| Transport security | Plain HTTP + static bearer token on the LAN. Beyond the LAN: a documented reverse-proxy (Caddy) or Tailscale pattern. No TLS in `dw.serve`. |
| Lifecycle | `pip install` from a clone + a shipped systemd unit template and env file under `contrib/systemd/`. |
| Install path | Unchanged: `git clone` + `install.sh` on the GPU box. |
| Out of scope | TLS in uvicorn, multi-user auth, Docker, install-without-clone, per-plugin trust. |

## What is wrong today

1. **Direct-LAN browser access is broken, not "thinly supported".** The
   `reject_foreign_origins` middleware (`dw/server/app.py`) 403s any request
   whose `Origin` host is not loopback. A browser at
   `http://192.168.1.50:8765` sends `Origin: http://192.168.1.50:8765` on
   every `fetch` and gets 403. Only the SSH-tunnel path works, because the
   tunnel's Origin is `127.0.0.1`.
2. `dw-mcp` pointed at a non-loopback server with no token fails one 401 at
   a time instead of at startup.
3. Nothing in the response surface identifies which machine answered.
4. There is no remote MCP transport; MCP requires a local Python install.
5. No documented, restart-safe way to keep the server running on a headless
   box.

## Design

### 1. Server: non-loopback binds work

**Origin rule.** An `Origin` header is accepted when its host is any of:

- a loopback name (`localhost`, `127.0.0.1`, `::1`);
- the configured `--host` (lower-cased), when that is not a wildcard;
- **the hostname of the request's own `Host` header** (same-origin).

The third clause is the one that makes LAN and reverse-proxy access work.
It is still safe against DNS rebinding: a rebinding attack yields a request
whose `Origin` is the attacker's domain while `Host` is whatever resolved;
they differ, so it is refused. A drive-by page on another origin is
refused for the same reason.

Ports are ignored in the comparison, matching
the existing `Host` check; scheme is ignored too (a proxy terminating TLS
forwards `Host` but the browser's Origin is `https://…`).

**Health identity.** `GET /api/health` gains:

```json
{ "hostname": "<socket.gethostname()>", "device": "<cuda|mps|cpu>", "mcp": false }
```

`device` comes from `dw.get_device_type()` on the resolved device (the same
value the worker will use); `mcp` reports whether `/mcp` is mounted. The MCP
`get_health` tool passes these through unchanged.

**Tests** (`tests/test_server_security.py` or wherever Origin/Host tests
live today): LAN-IP Origin with matching Host → 200; hostname Origin with
matching Host → 200; Origin host ≠ Host host → 403 (rebinding); loopback
Origin still accepted for any Host; `--host 192.168.1.50` accepts that
Origin regardless of Host. Health payload fields present.

### 2. `dw-mcp` (stdio) remote safety

In `dw_mcp/__main__.py`, before building the server:

- **Refuse** to start when the resolved base URL's host is not loopback and
  no token resolved. Message names both `--token` and `DW_API_TOKEN` and
  says why (an unauthenticated remote `dw.serve` would let anyone on the
  network run workflows). Exit status 2.
- **Probe** `GET /api/health` once. On 401: "server requires a token" and
  exit 2. On a connection error: name the URL tried and exit 2. On success,
  write one line to stderr: `dw-mcp: connected to <hostname> (dw <version>,
  <device>) at <url>`. `--no-probe` skips this (tests, servers still
  starting).
- Loopback detection: `urlparse(url).hostname` in `{"localhost",
  "127.0.0.1", "::1"}`. Reuse or mirror the constant; `dw_mcp` must not
  import `dw.server.app` (that pulls in `dw/__init__` and torch - the
  boundary test guards this), so the set is duplicated in `dw_mcp/client.py`
  with a comment pointing at its twin.

**Tests** (`tests/test_mcp_client.py` / a new `tests/test_mcp_main.py`):
`main([...])` with `httpx.MockTransport` injected - non-loopback + no token
exits 2 without any request; 401 exits 2 with the token message; success
prints the identity line; `--no-probe` sends nothing.

### 3. Remote MCP inside `dw.serve`

**Flag.** `dw.serve --mcp` (default off). Refused with a clear error when
the bind is non-loopback and no token is configured - unlike the REST-only
warning, this is a hard error, because MCP can author and run workflows and
there is no UI-style "paste the token" gate in front of it.

**Wiring** (`dw/server/mcp_mount.py`, new, so `app.py` does not grow):

```python
def mount_mcp(app, *, port, token):
    from dw_mcp.client import DwClient          # dw_mcp is a pure HTTP client
    from dw_mcp.server import build_server      # only module importing the SDK
    from mcp.server.transport_security import TransportSecuritySettings
    client = DwClient(base_url=f"http://127.0.0.1:{port}", token=token)
    server = build_server(client)
    asgi = server.streamable_http_app(
        streamable_http_path="/",   # the SDK app routes at "/" ...
        stateless_http=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False),   # app.py's checks own this
    )
    app.mount("/mcp", asgi)         # ... so the mount makes it "/mcp" (verified against mcp 2.1.1)
    return client                   # closed in the app's lifespan
```

The import of `mcp` is inside the function; a missing package produces
`SystemExit("--mcp needs the mcp extra: pip install 'diffusers-workflow[mcp]'")`.
`stateless_http=True` so a server restart or a dropped connection never
strands a session. The handlers reach the REST API over loopback HTTP with
the same token - a few milliseconds per call, and it keeps `dw_mcp` a pure
HTTP client. The client is closed in the existing `lifespan`.

**Token gate.** `require_bearer_token` gates `/mcp` (and `/mcp/…`) exactly
like `/api/`: header only, no query-parameter allowance.

**`download_output` on a remote MCP.** The tool writes to the filesystem of
the process running MCP - on the GPU box, not the agent's machine. The tool
description gains a sentence saying so; docs say to use `get_output_image`
/ `get_output_text` (inline content) remotely.

**Health.** `mcp: true` when mounted.

**Tests** (`tests/test_server_mcp.py`, new; `pytest.importorskip("mcp")`):
`create_app(..., mcp=True)` mounts `/mcp`; without a token it is reachable;
with a token, no header → 401, wrong → 401; a real MCP client session over
`httpx.ASGITransport` (the SDK's `streamablehttp_client` takes an httpx
client factory) lists tools equal to `EXPECTED_TOOLS`; `--mcp` with
`--host 0.0.0.0` and no token exits non-zero in `dw.serve.main`.
`create_app` gains `mcp=False` and `port` parameters so tests can build it
without `main()`.

### 4. Docs and ops

- `contrib/systemd/dw-serve.service` (`Type=simple`, `EnvironmentFile=`,
  `Restart=on-failure`, `WorkingDirectory=` the clone, `ExecStart=<venv>/bin/dw-serve --host 0.0.0.0 --mcp`)
  and `contrib/systemd/dw-serve.env.example` (`DW_API_TOKEN`, `DW_PROMPT_DIR`,
  `HF_HOME`, `DW_DEVICE`). A short README in the folder: copy, edit, `systemctl enable --now`.
- New `docs/REMOTE.md`: the LAN recipe end to end (generate a token with
  `openssl rand -hex 32`, unit install, firewall note, browser token field,
  `claude mcp add --transport http dw http://<box>:8765/mcp --header "Authorization: Bearer <token>"`,
  and the stdio alternative), what to expect in `/api/health`, then a
  "Beyond the LAN" section: Caddy reverse proxy with automatic TLS
  (`reverse_proxy 127.0.0.1:8765`, bind `dw.serve` to loopback in that
  setup), Tailscale as the no-certs alternative, and an explicit "never
  port-forward plain HTTP".
- `docs/MCP.md` Security: replace the blanket "don't" with the real
  constraint (token required; transport trust is the network's job), add
  the `/mcp` transport and the `download_output` caveat.
- `docs/SERVER.md`: Origin rule, health fields, `--mcp`.
- `CLAUDE.md`: one line each for `--mcp` and `contrib/systemd`.
- `docs/proposals/remote-gpu-server.md`: status → accepted, record the
  Origin finding and the decisions table, point here.

## Threat model, briefly

Same as today with one boundary moved: the token is now the *only* thing
between "on the LAN" and "can run arbitrary workflows". Therefore:
`--trust-workflows` stays off on any remote server (documented, unchanged);
the token must be long and random (docs show the command); plaintext HTTP is
acceptable only on a network you control, and the NAT section exists so
nobody port-forwards 8765.

## Sequencing

1 (server Origin + health) → 3 (MCP mount; depends on health fields and the
token gate) → 2 (dw-mcp probe; depends on health identity) → 4 (docs, unit
files). 2 and 3 are independent of each other and can run in parallel after
1.
