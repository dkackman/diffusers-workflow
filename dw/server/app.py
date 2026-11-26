"""FastAPI application exposing the workflow engine.

`create_app` is the factory: it builds the JobManager (and the MCP app), puts
what the handlers read on `app.state`, installs the middleware and registers
the routers in `dw/server/routes/`. All state lives in the JobManager; routers
are routing, validation and SSE framing; every path goes through dw.security
validators. API docs are at /docs (OpenAPI at /openapi.json).
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Route

from ..hub_cache import DownloadManager
from ..workspace import (
    LORAS_SUBDIR,
    ConfiguredWorkspace,
    Workspace,
    _holds_a_workspace,
)
from . import api_models
from .http_security import install_middleware
from .jobs import JobManager
from .changes import Changes
from .listening import Listening
from .netinfo import LOOPBACK_HOSTS, WILDCARD_HOSTS
from .observed_cost import ObservedCosts
from .outputs import MEDIA_KINDS, RAW_MEDIA_EXTENSIONS
from .routes import include_file_routes, include_routers
from .updater import DiffusersUpdater

logger = logging.getLogger("dw")


def default_ui_dir():
    """The built SPA: ui/dist in a checkout, else the wheel's dw/server/ui, or None."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(here)), "ui", "dist"),
        os.path.join(here, "ui"),
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "index.html")):
            return candidate
    return None


class UiFiles(StaticFiles):
    """The built UI. /assets/ names are content-hashed, so cache them for a
    year; index.html names those hashes and revalidates, or a returning
    browser runs an old build. StaticFiles alone resolves paths."""

    def file_response(self, full_path, stat_result, scope, status_code=200):
        response = super().file_response(full_path, stat_result, scope, status_code)
        hashed = scope["path"].startswith("/assets/")
        immutable = "public, max-age=31536000, immutable"
        response.headers["Cache-Control"] = immutable if hashed else "no-cache"
        return response


def _compress(app_):
    """Gzip all but /mcp and /mcp/* (that client asked for none) and `skip`'s types."""
    # Starlette's default list, passed so an upgrade cannot change what is gzipped
    skip = "application/grpc application/gzip application/x-gzip application/zip audio/* font/woff font/woff2 image/avif image/gif image/jpeg image/png image/webp text/event-stream video/*"
    gzip = GZipMiddleware(
        app_, minimum_size=1000, compresslevel=6, exclude_content_types=skip.split()
    )

    async def route(scope, receive, send):
        path = scope.get("path", "")
        mcp = path == "/mcp" or path.startswith("/mcp/")
        await (app_ if mcp else gzip)(scope, receive, send)

    return route


def _lifespan(manager, mcp_server, mcp_client):
    @asynccontextmanager
    async def lifespan(app):
        if mcp_server is None:
            yield
        else:
            # the SDK session manager is the sub-app's lifespan, which Starlette skips
            try:
                async with mcp_server.session_manager.run():
                    yield
            finally:
                # the client's pool must not leak if the session manager fails to start
                mcp_client.close()
        manager.shutdown()

    return lifespan


def _store_directories(
    state, manager, workflow_dir, prompt_dir, asset_dir, examples_dirs, workspace
):
    """The directories every handler resolves against, on `app.state`."""
    state.job_manager = manager
    # Job history as a cost, recomputed when the jobs table moves, not a file: a
    # job landing changes no workflow file (#93)
    state.observed_costs = ObservedCosts(getattr(manager, "history", None))
    state.workflow_dir = workflow_dir
    state.prompt_dir = prompt_dir
    # Where uploads land and 'asset:' resolves; None: uploads/keep/delete 409
    state.asset_dir = os.path.abspath(asset_dir) if asset_dir else None
    # The directories above default under it; None for a test's explicit three
    state.workspace = os.path.abspath(workspace) if workspace else None
    # The LoRA catalog written (root's loras/), shared by all like the prompt library;
    # None without a root: the shipped catalog is still read, and a save 409s
    state.lora_dir = (
        os.path.join(state.workspace, LORAS_SUBDIR) if state.workspace else None
    )
    if state.lora_dir and _holds_a_workspace(state.lora_dir):
        logger.warning(
            f"{state.lora_dir} holds a workspace named '{LORAS_SUBDIR}', but "
            f"'{LORAS_SUBDIR}' is now reserved (the LoRA catalog): the workspace "
            "is no longer listed and LoRA saves land beside its folders. "
            "Rename the directory to keep using it."
        )
    # The root holding named workspaces; its folders are the default one, so a
    # server given directory overrides has that one workspace and no others
    state.workspace_root = (
        Workspace(state.workspace, "flag") if state.workspace else None
    )
    # The default workspace: its folders are the configured directories above
    # (each overridable, e.g. --workflow-dir), not derived from the root
    state.default_workspace = ConfiguredWorkspace(
        workflows=state.workflow_dir,
        assets=state.asset_dir,
        outputs=manager.output_dir,
        prompts=state.prompt_dir,
        root=state.workspace,
    )

    # A workspace's requests.json folder, for a rerun; None (no raise) once gone
    def ledger_root_for(name):
        from .deps import workspace_for

        try:
            return workspace_for(state, name).root or None
        except Exception:
            return None

    manager.ledger_root_for = ledger_root_for
    # A StaticFiles per output/asset root, built lazily (a named root may not exist
    # at startup) and reused, so /outputs and /inputs answer 304 and 206
    state.static_files_by_root = {}
    # The two sets zip_download's policy reads, exposed for tests
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
    # One index per distinct listing (deps.ceiling_index)
    state.ceiling_indexes = {}
    # The LoRA catalog's partitions per listing (deps.lora_index)
    state.lora_indexes = None
    state.catalog_listings = {}
    # Which workspace workflows name each asset (deps.asset_users)
    state.asset_users = {}


def _mount_mcp(app, mcp_asgi):
    """The two `/mcp` routes, not app.mount("/mcp"): a Mount only matches *under*
    its prefix, so a bare POST /mcp (the configured URL) would fall to the SPA
    catch-all and 405. Exactly the two spellings require_bearer_token gates; one
    "/mcp{path:path}" would also answer /mcpfoo. build_mcp_app's wrapper
    normalizes the path for the SDK."""
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
    devices=None,
    managed=False,
):
    """Build the application. A caller (tests) can inject a JobManager.

    `devices` is `--devices` as resolved: one worker per entry (#462).

    `host` is the bound address, added to the Host-header allowlist beside loopback so a
    bind to one non-loopback address accepts its own requests. `token`, if given, is the
    bearer token every /api/* request needs (http_security's require_bearer_token).

    Registration order is the surface: `ROUTERS`, `/mcp`, the files routes, the UI.
    """
    manager = job_manager or JobManager(
        output_dir, log_level=log_level, workflow_dir=workflow_dir, devices=devices
    )
    # An injected manager must share the routes' workflow_dir, or validate
    # and jobs would enforce different boundaries
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
    # started by dw setup's service, which restarts it
    app.state.managed = managed
    app.state.mcp_mounted = mcp_asgi is not None
    app.state.listening = Listening()
    # One registry the routes and job manager both bump (changes.py)
    app.state.changes = Changes()
    manager.changes = app.state.changes
    _store_serving(
        app.state,
        host,
        port,
        token,
        examples_dirs,
        download_manager,
        diffusers_updater,
    )
    app.state.downloads.changes = app.state.changes
    app.state.updater.changes = app.state.changes
    # Before install_middleware: outside its streams, GZip ignores minimum_size
    app.add_middleware(_compress)
    install_middleware(app)
    include_routers(app)
    if mcp_asgi is not None:
        _mount_mcp(app, mcp_asgi)
    include_file_routes(app)

    resolved_ui = ui_dir or default_ui_dir()
    if resolved_ui:
        # Last, so /api and /outputs win; html=True serves index.html at / (the
        # SPA routes by hash, no fallback needed)
        app.mount("/", UiFiles(directory=resolved_ui, html=True), name="ui")

    return app
