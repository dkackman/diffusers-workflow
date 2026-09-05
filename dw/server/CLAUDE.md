# dw/server

Guidance for the HTTP server package.

`dw.serve --mcp` additionally serves the MCP tool surface at `/mcp`
(`mcp_mount.py`, Streamable HTTP, same bearer token) so an agent on another
machine needs no local install; it is refused on a non-loopback bind without a
token. `contrib/systemd/` has a unit file and docs/REMOTE.md the LAN/NAT setup.
The `Origin` check accepts the request's own `Host` hostname, which is what
makes a non-loopback bind usable from a browser.

See docs/SERVER.md.
