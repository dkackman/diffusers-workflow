"""Mount the MCP tool surface inside the HTTP server.

`dw_mcp` is a pure HTTP client of dw.serve; mounting it here does not
change that - the tools reach the REST API over the server's own bind
address with the same token the API requires, a few milliseconds per
call. This is the only module under dw/ that imports the mcp SDK, and it
does so lazily: the package is an optional extra.
"""

import ipaddress


def client_base_url(host, port):
    """The URL the mounted tools use to reach this same server.

    A loopback or wildcard bind is reached at 127.0.0.1; any other bind
    address has to be used verbatim, since uvicorn is then listening on
    that address alone and a loopback connection would be refused - which
    is exactly the `--host 100.x.y.z` Tailscale setup docs/REMOTE.md
    recommends. An IPv6 literal is bracketed for the URL's authority.
    """
    # imported here rather than at module scope: dw.server.app imports this
    # module (lazily, inside create_app), so an import back at import time
    # would be a cycle
    from .app import LOOPBACK_HOSTS, WILDCARD_HOSTS

    host = (host or "").lower()
    if host in WILDCARD_HOSTS | LOOPBACK_HOSTS:
        host = "127.0.0.1"
    try:
        if ipaddress.ip_address(host).version == 6:
            host = f"[{host}]"
    except ValueError:
        # a hostname, not an IP literal - nothing to bracket
        pass
    return f"http://{host}:{port}"


class _SingleRouteApp:
    """Send every path the parent routed here to the sub-app's one route.

    `dw.server.app` routes `/mcp` and anything under it to this wrapper,
    so the SDK's Starlette app - which has a single route at `/` - sees
    `/` whatever spelling the client used. Without it a bare `POST /mcp`
    either 404s or takes a redirect to `/mcp/`, and `http://box:8765/mcp`
    is the URL the docs tell people to configure.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") in ("http", "websocket"):
            scope = dict(scope, path="/", raw_path=b"/")
        await self.app(scope, receive, send)


def build_mcp_app(*, host, port, token):
    """The ASGI app to serve at /mcp, the MCPServer behind it, and the
    HTTP client the tools use to reach this same server (client_base_url).

    The SDK's Starlette app carries its own lifespan (the session
    manager), which the parent app does not run for a sub-app it routes
    to - `create_app`'s lifespan enters `server.session_manager.run()`
    itself.
    """
    try:
        from mcp.server.transport_security import TransportSecuritySettings
    except ImportError:
        raise SystemExit(
            "--mcp needs the mcp extra: pip install 'diffusers-workflow[mcp]'"
        )

    from dw_mcp.client import DwClient
    from dw_mcp.server import build_server

    client = DwClient(base_url=client_base_url(host, port), token=token)
    server = build_server(client)
    asgi = server.streamable_http_app(
        # the SDK app routes at "/"; the parent routes /mcp here and
        # _SingleRouteApp rewrites the path to "/" on the way in
        streamable_http_path="/",
        # one JSON reply per request, no sessions to strand on a restart
        json_response=True,
        stateless_http=True,
        # dw.server.app's own Origin/Host middleware runs first and owns this
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        ),
    )
    return _SingleRouteApp(asgi), server, client
