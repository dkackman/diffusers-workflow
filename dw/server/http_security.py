"""The HTTP security layer: the four middlewares and the query-token marker.

`install_middleware` reads the bind and token values from `app.state` at
request time (`api_token`, `wildcard_bind`, `allowed_hosts`), so the layer has
no closure to share with the factory that builds the app.
"""

import secrets
from urllib.parse import urlparse

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.routing import Match, Route

# Types a browser renders as a document, where script runs: /outputs and
# /inputs serve these under a CSP sandbox (_sandbox_active_content)
ACTIVE_DOCUMENT_TYPES = frozenset(
    {
        "text/html",
        "application/xhtml+xml",
        "text/xml",
        "application/xml",
        "image/svg+xml",
    }
)


def query_token_ok(fn):
    """Mark a GET endpoint as one a browser loads without being able to set
    headers (EventSource, an <img> tag, an <a download> navigation) - only
    routes carrying this marker accept the bearer token as a ?token= query
    param. Matched by the actual route at request time, not by a path
    suffix, so a resource that merely happens to be named "download" or
    "thumbnail" does not inherit the allowance."""
    fn.query_token_ok = True
    return fn


def _matched_route(request: Request):
    """Resolve the Route (if any) that will handle this request. Runs in
    middleware, before routing has attached anything to request.scope, so
    routes are matched by hand against request.app.router.routes. Skips
    non-Route entries (the SPA static Mount) and routes with no endpoint.

    A HEAD request path-matches a GET-only route as Match.PARTIAL (method
    mismatch) rather than Match.FULL, since this route is declared with
    methods=["GET"] and nothing here adds HEAD to it - but a HEAD request
    is still the same header-less browser load a GET would be, so it is
    treated the same for the query-token allowance."""
    method = request.scope.get("method")
    for route in request.app.router.routes:
        if not isinstance(route, Route) or route.endpoint is None:
            continue
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return route
        if (
            match == Match.PARTIAL
            and method == "HEAD"
            and route.methods
            and "GET" in route.methods
        ):
            return route
    return None


def install_middleware(app):
    """Add the four HTTP middlewares, in the order that makes the stack:
    Starlette puts the last one added outermost."""

    @app.middleware("http")
    async def reject_foreign_origins(request, call_next):
        """Refuse browser cross-origin requests - a drive-by web page must
        not be able to queue jobs on this server. Requests without an
        Origin header (curl, scripts, same-origin GETs) pass.

        An Origin is accepted when its hostname is a loopback name, the
        configured bind host, or the hostname the request itself was
        addressed to (same-origin). The last clause is what lets a browser
        on another machine use a `--host 0.0.0.0` server by its LAN IP or
        hostname - and it stays safe against DNS rebinding, where the
        attacker's page carries its own Origin while Host is whatever
        resolved: the two differ, so the request is refused. Scheme and
        port are ignored, matching the Host check: a TLS-terminating proxy
        forwards Host unchanged while the browser's Origin is https."""
        origin = request.headers.get("origin")
        if origin:
            try:
                origin_host = (urlparse(origin).hostname or "").lower()
            except ValueError:
                # urlparse raises on a bracketed host that is not IPv6
                # ('http://[::1].evil.example'): refused like any other
                # foreign Origin rather than escaping as a 500
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-origin requests are not allowed"},
                )
            request_host = (request.url.hostname or "").lower()
            # origin_host must be non-empty for the same-origin clause:
            # `Origin: null` (a sandboxed iframe, a file:// page) parses to
            # no hostname and would otherwise match a request whose Host
            # carries none either
            if origin_host not in request.app.state.allowed_hosts and not (
                origin_host and origin_host == request_host
            ):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-origin requests are not allowed"},
                )
        return await call_next(request)

    # Defense-in-depth for requests that carry no Origin at all (curl,
    # scripts, the MCP client) and so skip the check above entirely: a
    # request that arrived on this port but claims to be addressed to some
    # unrelated public domain is rejected. This does not stop DNS rebinding
    # by itself (the Origin check already does, since a browser's Origin
    # header reflects the real requesting origin regardless of DNS) - it
    # only closes the gap for non-browser clients that never send Origin.
    # A wildcard bind is reached by whatever address the machine has - a LAN
    # IP, a hostname - never by the bind string itself, so there is no
    # allowlist to build; the Host check is skipped for it.
    @app.middleware("http")
    async def reject_foreign_hosts(request, call_next):
        hostname = request.url.hostname
        if (
            not request.app.state.wildcard_bind
            and hostname is not None
            and hostname.lower() not in request.app.state.allowed_hosts
        ):
            return JSONResponse(
                status_code=400,
                content={"detail": "Unrecognized Host header"},
            )
        return await call_next(request)

    @app.middleware("http")
    async def require_bearer_token(request: Request, call_next):
        """Static bearer-token auth (opt-in via --token / DW_API_TOKEN).
        Only /api/* is gated - the UI's own static files and /outputs (an
        <img>/<script> tag cannot attach an Authorization header anyway)
        stay reachable so the page can load far enough to let a user enter
        the token in the first place. EventSource cannot set custom headers
        either, and neither can the <img> tags the gallery grid loads its
        thumbnails through nor the <a download> navigations the download
        buttons make, so those GET routes additionally accept the token as a
        `token` query parameter - a documented trade-off, not a header-auth
        peer."""
        token = request.app.state.api_token
        if not token:
            return await call_next(request)
        path = request.url.path
        if not (path.startswith("/api/") or path == "/mcp" or path.startswith("/mcp/")):
            return await call_next(request)
        provided = None
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            provided = auth[len("bearer ") :].strip()
        # GET/HEAD only, and only on a route explicitly marked
        # query_token_ok - matched against the real route (see
        # _matched_route), not by a path suffix, so a resource that
        # happens to be named "download" or "thumbnail" does not inherit
        # the allowance meant for the real routes.
        if provided is None and request.method in ("GET", "HEAD"):
            route = _matched_route(request)
            if route is not None and getattr(route.endpoint, "query_token_ok", False):
                provided = request.query_params.get("token")
        # compared as bytes: compare_digest refuses non-ASCII str
        if provided is None or not secrets.compare_digest(
            provided.encode("utf-8"), token.encode("utf-8")
        ):
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing or invalid bearer token"},
            )
        return await call_next(request)

    # Added last, so it is the outermost middleware and its headers land on
    # every response - including the 400/401/403 answers the checks above
    # return without reaching a route. nosniff stops a browser reading an
    # output as a type other than the one it was served as; DENY stops any
    # other site framing the UI to click its buttons (#407)
    @app.middleware("http")
    async def browser_headers(request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        return response
