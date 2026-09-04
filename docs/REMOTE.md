# Using dw from another machine

`dw.serve` runs on the machine with the GPU; a browser and Claude Code on
your laptop use it over the network. Nothing is installed on the laptop.

## On the GPU box

1. Install as usual: `git clone …`, `bash ./install.sh`.
2. Make a token: `openssl rand -hex 32`. This one string is the only thing
   between "on your network" and "can run workflows on your GPU" - keep it
   long and random.
3. Run it bound to the network, with the token, with MCP:

       DW_API_TOKEN=<token> dw-serve --host 0.0.0.0 --mcp

   `--mcp` is refused outright (exit 2) on a non-loopback bind with no
   token: an MCP endpoint can author and run workflows and has no
   token-entry page in front of it the way the web UI does.

   To keep it running across logouts and reboots, install the systemd unit
   in [contrib/systemd](../contrib/systemd/README.md) instead.
4. Open port 8765 in the box's firewall for your LAN only (for example
   `sudo ufw allow from 192.168.1.0/24 to any port 8765`).

Never pass `--trust-workflows` on a server other machines can reach: it
lets any workflow - including one an agent authored - execute arbitrary
Python. See [SECURITY.md](SECURITY.md#trust-model).

Check it from the laptop:

    curl http://<box>:8765/api/health -H "Authorization: Bearer <token>"
    {"status":"ok","version":"…","hostname":"gpu-box","device":"cuda","mcp":true,…}

`hostname` and `device` are there so you can tell which machine answered.

## Browser

Open `http://<box>:8765`. Click the key icon next to the theme toggle,
paste the token once; it is kept in the browser's `localStorage` and sent
with every request.

## Claude Code, no local install

    claude mcp add --transport http dw http://<box>:8765/mcp \
      --header "Authorization: Bearer <token>"

Start a new Claude Code session; `/mcp` should list `dw` as connected.
Pick the scope with `-s user` to have it in every project.

`http://<box>:8765/mcp` and `http://<box>:8765/mcp/` are both answered, and
both require the token in an `Authorization: Bearer` header - the
`?token=...` allowance a few browser GET routes have does not extend to
`/mcp`.

Two things differ from the local stdio setup:

- `download_output` writes on the GPU box (where the MCP server runs), not
  on your laptop. Use `get_output_image` / `get_output_text` to see a
  result, or open `http://<box>:8765/outputs/<name>` in the browser.
- The connection is a plain HTTP call per tool invocation; there is no
  subprocess to restart.

## Claude Code with a local install (stdio)

If the laptop also has `dw` installed, the stdio server works against a
remote box too:

    claude mcp add dw -- /path/to/venv/bin/dw-mcp \
      --url http://<box>:8765 --token <token>

`dw-mcp` refuses to start (exit 2) against a non-loopback URL without a
token, and makes one `GET /api/health` at startup so a wrong URL or token
is reported once, with a message, instead of as a 401 on every tool call.
Against a remote URL a failed probe is fatal - it is a misconfiguration you
have to fix. Against a loopback URL it only warns and serves anyway, since
there it usually means "dw.serve is not up yet", which the next tool call
reports for itself. `--no-probe` skips the check entirely.

## Beyond your LAN

Everything above is plaintext HTTP: the token and every prompt and result
are readable by anything on the network path. That is acceptable on a
network you control and nowhere else. **Do not port-forward 8765 on your
router.** Two ways to reach the box from outside:

**Tailscale (or another WireGuard overlay).** Install it on the box and the
laptop; use the box's Tailscale IP or MagicDNS name in every URL above.
Traffic is encrypted end to end, no certificates to manage, and `dw-serve`
can stay bound to the Tailscale interface (`--host 100.x.y.z`) rather than
`0.0.0.0`. This is the recommended option.

**A TLS-terminating reverse proxy.** Bind `dw-serve` back to loopback and
put Caddy in front of it with a real hostname:

    # /etc/caddy/Caddyfile
    dw.example.com {
        reverse_proxy 127.0.0.1:8765
    }

Caddy obtains and renews a certificate automatically. Use
`https://dw.example.com` (no port) in every URL above. `dw-serve`'s Origin
and Host checks work unchanged behind the proxy because Caddy forwards the
`Host` header as-is. nginx works the same way with `proxy_pass` and
`proxy_set_header Host $host;` plus your own certificate.

The token is still the only authentication in either setup; a proxy or a
VPN protects the transport, not the door.

## What is not here

- TLS inside `dw-serve` itself: a reverse proxy does it better.
- More than one token, or users: one shared secret, deliberately
  ([SERVER.md](SERVER.md#authentication)).
- Docker: `install.sh` on the box is the supported install.
