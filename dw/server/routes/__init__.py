"""The routers, in registration order.

Order is part of the surface: a greedy `{name:path}` GET must be registered
after its `/download`, `/variables` and `/metadata`-style siblings, and Starlette
takes the first route that matches. `create_app` registers these in this order,
then adds the `/mcp` routes, the files router and the UI mount.
"""

from . import jobs, library, system

ROUTERS = (jobs.router, system.router, library.router)


def include_routers(app):
    """Register every router's routes on `app`, in `ROUTERS` order.

    Not `app.include_router`: from fastapi 0.141 that leaves one lazy
    `_IncludedRouter` node per router in `app.router.routes`, and
    `http_security._matched_route` (which decides whether a request may carry
    `?token=`) walks that list looking for `Route` entries carrying
    `endpoint.query_token_ok`. Appending the routes themselves keeps the table
    flat, as it was when the handlers were registered on the app directly. No
    router here carries a prefix, dependency or response override for
    `include_router` to apply, and a route holds no per-app state - handlers
    read `request.app.state` - so two apps can share the route objects.
    """
    for router in ROUTERS:
        app.router.routes.extend(router.routes)
