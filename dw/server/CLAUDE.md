# dw/server

Guidance for the HTTP server package.

`dw.serve --mcp` additionally serves the MCP tool surface at `/mcp`
(`mcp_mount.py`, Streamable HTTP, same bearer token) so an agent on another
machine needs no local install; it is refused on a non-loopback bind without a
token. `contrib/systemd/` has a unit file and docs/REMOTE.md the LAN/NAT setup.
The `Origin` check accepts the request's own `Host` hostname, which is what
makes a non-loopback bind usable from a browser.

`guides.py` serves the prose guides (`GET /api/guides`) an agent reads before
choosing a capability. The `GUIDES` table there is the closed set of names; a
guide file resolves to the checkout's `docs/` first, else the packaged
`dw/docs/` copy `scripts/build_dist.sh` makes, the same rule `default_ui_dir`
uses for the SPA. `dw_mcp/guides.py` is a proxy of these routes.

`exports.py` gathers one finished job into a standalone directory -
`workflow.json` (the realized copy when the run wrote one), `manifest.json`,
`job.json`, a README, and copies of the assets, earlier-run inputs and outputs
it referenced - under `<workspace>/exports/<job id>/`. `EXPORTS_SUBDIR` lives
in `dw/workspace.py` rather than here, since `RESERVED_WORKSPACE_NAMES` needs
it and `dw/workspace.py` must not import from `dw.server`; this module
re-imports it. `app.py`'s `POST /api/jobs/{id}/export` calls it and returns
the summary plus a `zip_url`; `GET /exports/{id}.zip` builds the archive on
request from the same directory rather than keeping a second copy.

See docs/SERVER.md.
