"""FastAPI application exposing the workflow engine.

`create_app` is the factory: it builds the JobManager (and the MCP app), puts
what the handlers read on `app.state`, installs the middleware and registers
the routers in `dw/server/routes/`. All state lives in the JobManager; the
routers are routing, validation and SSE framing. Everything path-shaped goes
through dw.security validators. Interactive API docs are served at /docs
(OpenAPI at /openapi.json).
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.routing import Route

from ..hub_cache import DownloadManager
from ..workspace import LORAS_SUBDIR, ConfiguredWorkspace, Workspace
from . import api_models
from .http_security import install_middleware
from .jobs import JobManager
from .netinfo import LOOPBACK_HOSTS, WILDCARD_HOSTS
from .observed_cost import ObservedCosts
from .outputs import MEDIA_KINDS, RAW_MEDIA_EXTENSIONS
from .routes import include_file_routes, include_routers
from .updater import DiffusersUpdater


def default_ui_dir():
    """Where the built SPA lives: ui/dist in a checkout (the copy npm just
    built), else the copy packaged into the wheel at dw/server/ui, else None."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(here)), "ui", "dist"),
        os.path.join(here, "ui"),
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "index.html")):
            return candidate
    return None


def _lifespan(manager, mcp_server, mcp_client):
    @asynccontextmanager
    async def lifespan(app):
        if mcp_server is None:
            yield
        else:
            # the SDK's session manager is the mounted app's own lifespan,
            # which Starlette does not run for a sub-app
            try:
                async with mcp_server.session_manager.run():
                    yield
            finally:
                # the client owns a connection pool; a session manager that
                # fails to start must not leak it
                mcp_client.close()
        manager.shutdown()

    return lifespan


def _store_directories(
    state, manager, workflow_dir, prompt_dir, asset_dir, examples_dirs, workspace
):
    """The directories every handler resolves against, on `app.state`."""
    state.job_manager = manager
    # This box's own job history as a cost, recomputed when the jobs table
    # moves rather than when a file does - a job landing changes every
    # figure and changes no workflow file (#93)
    state.observed_costs = ObservedCosts(getattr(manager, "history", None))
    state.workflow_dir = workflow_dir
    state.prompt_dir = prompt_dir
    # Where uploads land and 'asset:' references resolve. None when the
    # caller configured no asset library: uploads, keep and delete then
    # answer 409, since there is nowhere for an asset to be
    state.asset_dir = os.path.abspath(asset_dir) if asset_dir else None
    # The workspace the three directories above default to folders of, for a
    # client that wants to name the root rather than reason about the parts.
    # None when the caller resolved no workspace (a test building an app
    # around three explicit directories)
    state.workspace = os.path.abspath(workspace) if workspace else None
    # The LoRA catalog this server writes to: the root's loras/, shared by
    # every workspace like the prompt library. None when there is no root (a
    # server configured from loose directories) - the shipped catalog is
    # still read, and a save answers 409
    state.lora_dir = (
        os.path.join(state.workspace, LORAS_SUBDIR) if state.workspace else None
    )
    # The root that holds named workspaces. Its own folders are the default
    # workspace - which is what the three directories above already point at,
    # so a server given individual directory overrides simply has one
    # workspace and no others
    state.workspace_root = (
        Workspace(state.workspace, "flag") if state.workspace else None
    )
    # The default workspace itself, as a Workspace: its four folders are the
    # configured directories above, not '<root>/workflows' and friends - a
    # caller can override any one of them individually (--workflow-dir,
    # etc), so they cannot be derived from a root the way a named
    # workspace's folders are
    state.default_workspace = ConfiguredWorkspace(
        workflows=state.workflow_dir,
        assets=state.asset_dir,
        outputs=manager.output_dir,
        prompts=state.prompt_dir,
        root=state.workspace,
    )
    # A StaticFiles instance per output/asset root, built lazily and reused -
    # a mount is bound to one directory at startup, but a named workspace's
    # root does not exist yet then. Keeping the instance around (rather than
    # building one per request) is what makes /outputs and /inputs answer
    # ETag/If-None-Match with 304 and Range with 206 the way a real mount
    # does, instead of the plain FileResponse this replaced always resending
    # the whole file
    state.static_files_by_root = {}
    # The two sets zip_download's compression policy reads, exposed so a test
    # can assert the relationship without reaching into the module
    state.media_kinds = MEDIA_KINDS
    state.raw_media_extensions = RAW_MEDIA_EXTENSIONS


def _store_serving(
    state, host, port, token, examples_dirs, download_manager, diffusers_updater
):
    """What the middlewares and the system routes read at request time."""
    wildcard_bind = host in WILDCARD_HOSTS
    allowed_hosts = set(LOOPBACK_HOSTS)
    if host and not wildcard_bind:
        allowed_hosts.add(host.lower())
    state.api_token = token
    state.bind_host = host
    state.bind_port = port
    state.wildcard_bind = wildcard_bind
    state.allowed_hosts = allowed_hosts
    state.examples_dirs = examples_dirs
    state.downloads = download_manager or DownloadManager()
    state.updater = diffusers_updater or DiffusersUpdater()
    # One index per distinct listing, per app (see deps.ceiling_index)
    state.ceiling_indexes = {}


def _mount_mcp(app, mcp_asgi):
    """The two `/mcp` routes. One route pair rather than
    app.mount("/mcp", ...): Starlette's Mount only matches paths *under* its
    prefix, so a bare POST /mcp - the URL clients are configured with - would
    fall through to the SPA catch-all and come back 405. Exactly the two
    spellings require_bearer_token gates - a single "/mcp{path:path}" route
    would also answer /mcpfoo, which the gate does not cover.
    build_mcp_app's wrapper normalizes the path for the SDK app's single
    route."""
    app.router.routes.append(Route("/mcp", endpoint=mcp_asgi, name="mcp"))
    app.router.routes.append(
        Route("/mcp/{sub_path:path}", endpoint=mcp_asgi, name="mcp_sub")
    )


def create_app(
    workflow_dir="./workflows",
    output_dir="./outputs",
    log_level="INFO",
    job_manager=None,
    ui_dir=None,
    download_manager=None,
    diffusers_updater=None,
    prompt_dir="./prompts",
    asset_dir=None,
    examples_dirs=None,
    workspace=None,
    host="127.0.0.1",
    token=None,
    mcp=False,
    port=8765,
):
    """Build the application. A caller (tests) can inject a JobManager.

    `host` is the address the server is bound to (informational here - it
    is added to the Host-header allowlist alongside the loopback names, so
    a deployment bound to one specific non-loopback address still accepts
    its own requests). `token`, if given, is a static bearer token required
    on every /api/* request - see require_bearer_token in http_security.

    Registration order is the surface: the routers in `ROUTERS` order, then
    `/mcp`, then the files routes, then the UI mount last.
    """
    manager = job_manager or JobManager(
        output_dir, log_level=log_level, workflow_dir=workflow_dir
    )
    # An injected manager must confine jobs to the same workflow_dir the
    # routes do, or /api/validate and /api/jobs would enforce different
    # boundaries
    if manager.workflow_dir is None:
        manager.workflow_dir = workflow_dir
    elif manager.workflow_dir != workflow_dir:
        raise ValueError(
            "job_manager.workflow_dir must match the app's workflow_dir: "
            f"{manager.workflow_dir!r} != {workflow_dir!r}"
        )

    mcp_asgi = mcp_server = mcp_client = None
    if mcp:
        from .mcp_mount import build_mcp_app

        mcp_asgi, mcp_server, mcp_client = build_mcp_app(
            host=host, port=port, token=token
        )

    app = FastAPI(
        title="diffusers-workflow",
        description="Declarative diffusers workflows over HTTP: queue a job, "
        "stream its progress, fetch what it saved.",
        lifespan=_lifespan(manager, mcp_server, mcp_client),
    )
    if not api_models.STRICT:
        api_models.send_rejected_responses(app)
    _store_directories(
        app.state,
        manager,
        workflow_dir,
        prompt_dir,
        asset_dir,
        examples_dirs,
        workspace,
    )
    app.state.mcp_mounted = mcp_asgi is not None
    _store_serving(
        app.state,
        host,
        port,
        token,
        examples_dirs,
        download_manager,
        diffusers_updater,
    )
    install_middleware(app)
    include_routers(app)
    if mcp_asgi is not None:
        _mount_mcp(app, mcp_asgi)
    include_file_routes(app)

    resolved_ui = ui_dir or default_ui_dir()
    if resolved_ui:
        # Mounted last so /api and /outputs keep precedence; html=True serves
        # index.html at /, and the SPA routes by hash so no fallback is needed
        app.mount("/", StaticFiles(directory=resolved_ui, html=True), name="ui")

    return app
