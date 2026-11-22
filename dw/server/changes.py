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
  library) is bumped with `bump_all`: a server-wide count added to each
  workspace's own, so every workspace's page of that kind reloads.

The rule "every server write bumps its kind" is a row of
docs/ARCHITECTURE.md, enforced by tests/test_changes.py, which calls each
write route and checks exactly the right kind moved. In memory under one
lock, the `dw/server/listening.py` pattern.
"""

import threading
import uuid

from ..library import COMMON_ORIGIN
from ..workspace import DEFAULT_WORKSPACE_NAME

REQUESTS = "requests"
PROMPTS = "prompts"
WORKFLOWS = "workflows"
ASSETS = "assets"
GALLERY = "gallery"
JOBS = "jobs"
KINDS = (REQUESTS, PROMPTS, WORKFLOWS, ASSETS, GALLERY, JOBS)


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


def changed(request, workspace, *kinds) -> None:
    """Route form: bump `kinds` in `workspace` on this request's app."""
    request.app.state.changes.bump(workspace, *kinds)


def changed_everywhere(request, *kinds) -> None:
    """Route form of `bump_all`."""
    request.app.state.changes.bump_all(*kinds)


def changed_assets(request, ws, root) -> None:
    """An asset in library `root` (the `LibraryRoot` that `AssetLibrary.find`
    or `writable_root` answered) changed: the shared library's moves every
    workspace's `assets`, any other only `ws`'s. Which library holds a file
    is `LibraryPath`'s answer (`origin`), never re-derived from the path."""
    if root.origin == COMMON_ORIGIN:
        changed_everywhere(request, ASSETS)
    else:
        changed(request, ws.name, ASSETS)
