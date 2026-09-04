"""Mount the MCP tool surface inside the HTTP server.

`dw_mcp` is a pure HTTP client of dw.serve; mounting it here does not
change that - the tools reach the REST API over loopback with the same
token the API requires, a few milliseconds per call. This is the only
module under dw/ that imports the mcp SDK, and it does so lazily: the
package is an optional extra.
"""


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


def build_mcp_app(*, port, token):
    """The ASGI app to serve at /mcp, the MCPServer behind it, and the
    loopback client the tools use.

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

    client = DwClient(base_url=f"http://127.0.0.1:{port}", token=token)
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
