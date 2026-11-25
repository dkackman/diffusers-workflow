"""Workspace change counters: how the UI notices a write it did not make.

Every server write that changes what a page lists bumps one counter for
its workspace and its kind of data (`KINDS`), after the write succeeded; a
write that fails bumps nothing. `GET /api/workspaces/{name}/changes` hands
the UI one workspace's counters, and a page reloads when the one it shows
moves. Every outside writer reaches the disk through this server (the MCP
server is an HTTP client of it, `dw.run` queues on it, jobs finish in its
job manager), so the counters see every write but a hand edit on disk.

- Counters only grow, and live in memory only. `epoch` is a random id
  made when a workspace's counters are, so a server restart, or a
  workspace deleted and created again under the same name (`forget`),
  reads to a client as "everything changed" rather than as counters that
  went backwards.
- `workspaces` is one server-wide counter (create, delete, a workspace's
  note), carried in every snapshot so the sidebar needs no second read.
- A library every workspace shares (the prompt library, the shared asset
  library, and the server's models and LoRAs) is bumped with `bump_all`: a
  server-wide count added to each workspace's own, so every workspace's page
  of that kind reloads.

A route never bumps by hand. Every write route declares what it changes,
`@writes(kinds...)` under its `@router.<method>(...)`, or `@NO_CHANGE(reason)`
when it writes nothing a page lists; `by=` names who bumps instead (the job
manager, for a queued job). Every router is `APIRouter(route_class=ChangeRoute)`,
whose handler bumps the declared kinds after the response is built, and only
for a 2xx: a raised exception, or a 4xx or 5xx, bumps nothing. The target is
the request's workspace, which `deps.selected_workspace` records on
`request.state`; a handler whose target is conditional takes
`change: Change = Depends(pending_change)` and narrows it (`only`,
`in_workspace`, `everywhere`, `in_library`, `nothing`). `WORKSPACES` is the
declarable name of the server-wide counter.

The rule "every server write bumps its kind" is a row of
docs/ARCHITECTURE.md, enforced by tests/test_changes.py, which calls each
write route and checks exactly the right kind moved, and checks that every
write route declares. In memory under one lock, the `dw/server/listening.py`
pattern.
"""

import threading
import uuid
from typing import NamedTuple, Optional

from fastapi import Request
from fastapi.routing import APIRoute

from ..library import COMMON_ORIGIN
from ..workspace import DEFAULT_WORKSPACE_NAME

REQUESTS = "requests"
PROMPTS = "prompts"
WORKFLOWS = "workflows"
ASSETS = "assets"
GALLERY = "gallery"
JOBS = "jobs"
MODELS = "models"
LORAS = "loras"
KINDS = (REQUESTS, PROMPTS, WORKFLOWS, ASSETS, GALLERY, JOBS, MODELS, LORAS)
# Kinds with no workspace: the hub cache and the LoRA library belong to the
# server, so a write to one is always applied with `bump_all`
SERVER_WIDE = (MODELS, LORAS)
# The server-wide counter, declarable on a route; not one of `KINDS`
WORKSPACES = "workspaces"


def _space():
    return {"epoch": uuid.uuid4().hex, "kinds": dict.fromkeys(KINDS, 0)}


class Changes:
    def __init__(self):
        self._lock = threading.Lock()
        self._spaces = {}
        self._shared = dict.fromkeys(KINDS, 0)
        self._workspaces = 0

    @staticmethod
    def _key(workspace):
        return workspace or DEFAULT_WORKSPACE_NAME

    def _counters(self, workspace):
        """Held under the lock: the workspace's counters, made on first use."""
        return self._spaces.setdefault(self._key(workspace), _space())

    def bump(self, workspace, *kinds) -> None:
        """One write in `workspace` changed each of `kinds`."""
        _check(kinds)
        with self._lock:
            counters = self._counters(workspace)["kinds"]
            for kind in kinds:
                counters[kind] += 1

    def bump_all(self, *kinds) -> None:
        """One write to a library every workspace shares."""
        _check(kinds)
        with self._lock:
            for kind in kinds:
                self._shared[kind] += 1

    def bump_workspaces(self) -> None:
        """A workspace was created or deleted, or its note changed."""
        with self._lock:
            self._workspaces += 1

    def forget(self, workspace) -> None:
        """A workspace was deleted or created: its counters start again
        under a new epoch, so a client holding the old ones reloads."""
        with self._lock:
            self._spaces.pop(self._key(workspace), None)

    def snapshot(self, workspace) -> dict:
        with self._lock:
            counters = self._counters(workspace)
            return {
                "epoch": counters["epoch"],
                "workspaces": self._workspaces,
                "kinds": {
                    kind: counters["kinds"][kind] + self._shared[kind] for kind in KINDS
                },
            }


def _check(kinds):
    unknown = [kind for kind in kinds if kind not in KINDS]
    if unknown or not kinds:
        raise ValueError(f"Unknown change kind(s): {unknown or 'none named'}")


class Declared(NamedTuple):
    """What a route said it changes (`writes`, `NO_CHANGE`)."""

    kinds: tuple
    shared: bool
    by: Optional[str]
    reason: Optional[str]


def writes(*kinds, shared=False, by=None):
    """Declare the kinds a write route changes. `shared` moves every
    workspace (a library they all share); `by` names who bumps instead of
    the route ("the job manager"), so `ChangeRoute` bumps nothing."""
    unknown = [kind for kind in kinds if kind not in KINDS + (WORKSPACES,)]
    if unknown or not kinds:
        raise ValueError(f"Unknown change kind(s): {unknown or 'none named'}")

    def declare(fn):
        fn.writes = Declared(tuple(kinds), shared, by, None)
        return fn

    return declare


def NO_CHANGE(reason):
    """Declare a write route that changes nothing a page lists, and why."""
    if not reason:
        raise ValueError("NO_CHANGE needs a reason")

    def declare(fn):
        fn.writes = Declared((), False, None, reason)
        return fn

    return declare


class Change:
    """One request's pending bump: the route's declaration, narrowed by
    its handler. `ChangeRoute` applies it after a 2xx."""

    def __init__(self, declared):
        self.declared = declared
        self.kinds = declared.kinds
        self.shared = declared.shared
        self.named = False
        self.workspace = None
        self.skip = False

    def only(self, *kinds):
        outside = [kind for kind in kinds if kind not in self.declared.kinds]
        if outside or not kinds:
            raise ValueError(f"Not declared by this route: {outside or 'none named'}")
        self.kinds = tuple(kinds)
        return self

    def in_workspace(self, name):
        self.named, self.workspace = True, name
        return self

    def everywhere(self):
        self.shared = True
        return self

    def in_library(self, ws, root):
        """An asset in library `root` (the `LibraryRoot` that
        `AssetLibrary.find` or `writable_root` answered): the shared
        library's moves every workspace, any other only `ws`. Which library
        holds a file is `LibraryPath`'s answer (`origin`), never re-derived
        from the path."""
        if root.origin == COMMON_ORIGIN:
            return self.everywhere()
        return self.in_workspace(ws.name)

    def nothing(self):
        self.skip = True
        return self

    def apply(self, changes, recorded, route):
        """Bump what is pending. `recorded` is the workspace the request
        selected (`_UNSET` when it selected none)."""
        if self.skip or self.declared.by or not self.kinds:
            return
        if WORKSPACES in self.kinds:
            changes.bump_workspaces()
        kinds = tuple(kind for kind in self.kinds if kind != WORKSPACES)
        if not kinds:
            return
        wide = kinds if self.shared else tuple(k for k in kinds if k in SERVER_WIDE)
        if wide:
            changes.bump_all(*wide)
        kinds = tuple(kind for kind in kinds if kind not in wide)
        if not kinds:
            return
        target = self.workspace if self.named else recorded
        if target is _UNSET:
            raise RuntimeError(f"{route} declares {kinds} but named no workspace")
        changes.bump(target, *kinds)


_UNSET = object()


def pending_change(request: Request) -> Change:
    """FastAPI dependency: this request's pending `Change`, to narrow."""
    return request.state.dw_change


class ChangeRoute(APIRoute):
    """A route that bumps what its endpoint declared, after a 2xx."""

    def get_route_handler(self):
        handler = super().get_route_handler()
        declared = getattr(self.endpoint, "writes", None)
        if declared is None:
            return handler
        name = self.name

        async def bump_after(request):
            change = request.state.dw_change = Change(declared)
            response = await handler(request)
            if 200 <= response.status_code < 300:
                recorded = getattr(request.state, "dw_workspace", _UNSET)
                change.apply(request.app.state.changes, recorded, name)
            return response

        return bump_after
