"""dw's own disk caches: what derived data its runs leave on disk, how big it
is, and clearing it (#828).

Each cache is a `DiskCache` in `CACHES`, named by the REST routes
(`GET /api/caches`, `DELETE /api/caches/{name}`) and summarized in the MCP
`get_server_info`'s `caches`:

- `inductor`: torch.compile's kernels, under `TORCHINDUCTOR_CACHE_DIR`, which
  `dw/__init__.py` always sets.
- `triton`: Triton's compiled and autotuned kernels (SDNQ's `triton.jit` /
  `triton.autotune` matmul and attention, most of the compile disk use).
  Registered where Triton itself puts it - `TRITON_CACHE_DIR`, else
  `$TRITON_HOME/.triton/cache`, else `~/.triton/cache` - resolved here
  without importing triton. dw sets no `TRITON_*` variable and never creates
  that directory.

**Clearing never trusts the root.** A root comes from an env var, and one
pointed at `~` by mistake must not be emptied. Only a root carrying the
`.dw-cache` marker is clearable. `mark_roots()`, run once at `dw.serve`
start, writes it into a root that is empty or missing (a missing `triton`
root is left missing), or into a populated root equal to the member's
default exactly - the cache dw (or Triton, by default) has been filling all
along. A populated root anywhere else stays unmarked and its clear is
refused, until someone marks it by hand.

**Clear empties the root and keeps it and its marker:** every top-level child
is renamed into one `<root>/.clearing-<pid>-<rand>/` directory, which is then
deleted. A writer racing the clear finds an empty cache and rebuilds; it never
reads a half-deleted tree. The route refuses while a job is running or queued.
"""

import importlib.util
import logging
import os
import secrets
import shutil
import threading
from dataclasses import dataclass
from typing import Callable

from . import _inductor_cache_dir_default
from .security import SecurityError, validate_path

logger = logging.getLogger("dw")

MARKER = ".dw-cache"
_CLEARING_PREFIX = ".clearing-"
_MARKER_TEXT = (
    "This directory is a cache dw may empty from its server page or "
    "DELETE /api/caches. Delete this file to stop that.\n"
)

# One clear at a time: a second one would find the first's staging directory
# among the root's children and move it out from under its rmtree
_clear_lock = threading.Lock()


class UnknownCache(ValueError):
    pass


class NotClearable(Exception):
    pass


@dataclass(frozen=True)
class DiskCache:
    name: str
    # The absolute root, from the environment as it stands
    root: Callable[[], str]
    # The root with the member's env var unset - only for grandfathering
    default_root: Callable[[], str]
    enabled: Callable[[], bool]
    cost_of_clearing: str
    # Whether the cache lists entries of its own (none do yet)
    has_entries: bool = False
    # Whether mark_roots may create a missing root
    create_root: bool = True


def _inductor_root():
    return os.path.abspath(
        os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        or _inductor_cache_dir_default(os.environ)
    )


def _inductor_default():
    return os.path.abspath(_inductor_cache_dir_default(os.environ))


def _triton_default():
    return os.path.abspath(os.path.join(os.path.expanduser("~"), ".triton", "cache"))


def _triton_root():
    """Where Triton caches, as triton.knobs resolves it - without importing
    triton, which the server process never loads and macOS doesn't have."""
    explicit = os.environ.get("TRITON_CACHE_DIR", "").strip()
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    home = os.environ.get("TRITON_HOME", "").strip()
    if home:
        return os.path.abspath(
            os.path.join(os.path.expanduser(home), ".triton", "cache")
        )
    return _triton_default()


def _triton_installed():
    # find_spec on a top-level name looks for it without importing it. The
    # workers run in this interpreter's environment, so this is theirs too
    return importlib.util.find_spec("triton") is not None


CACHES = {
    cache.name: cache
    for cache in (
        DiskCache(
            name="inductor",
            root=_inductor_root,
            default_root=_inductor_default,
            enabled=lambda: True,
            cost_of_clearing=(
                "The next first step of each torch.compile'd shape compiles "
                "its kernels again, up to a few minutes per shape."
            ),
        ),
        DiskCache(
            name="triton",
            root=_triton_root,
            default_root=_triton_default,
            enabled=_triton_installed,
            cost_of_clearing=(
                "The next first step of each SDNQ or compiled shape "
                "re-autotunes its Triton kernels (100-300 s on MiniMax-H3). "
                "This directory is shared with any other Triton user on "
                "this machine, whose kernels are cleared too."
            ),
            # A box without Triton gets no stray directory
            create_root=False,
        ),
    )
}


def get(name):
    try:
        return CACHES[name]
    except (KeyError, TypeError):
        raise UnknownCache(
            f"Unknown cache {name!r} - the caches are: {', '.join(CACHES)}"
        ) from None


def _children(root):
    """The root's entries other than the marker."""
    with os.scandir(root) as entries:
        return [entry for entry in entries if entry.name != MARKER]


def _has_marker(root):
    return os.path.isfile(os.path.join(root, MARKER)) and not os.path.islink(
        os.path.join(root, MARKER)
    )


def _write_marker(root):
    marker = validate_path(os.path.join(root, MARKER), base_dir=root)
    with open(marker, "w") as f:
        f.write(_MARKER_TEXT)


def mark_root(cache):
    """Mark one cache's root as dw's to clear, if the rule allows it.
    Returns whether the root carries the marker afterwards."""
    root = cache.root()
    if os.path.islink(root) or (os.path.exists(root) and not os.path.isdir(root)):
        return False
    if not os.path.exists(root):
        if not cache.create_root:
            return False
        os.makedirs(root, exist_ok=True)
    if _has_marker(root):
        return True
    if _children(root) and root != cache.default_root():
        logger.info(
            f"The {cache.name} cache at {root} was not created by dw and "
            f"has no {MARKER} marker, so it can't be cleared from the server"
        )
        return False
    _write_marker(root)
    return True


def mark_roots():
    """Write the ownership marker into each cache root the rule allows. Run
    once at server start; a root it can't write is left unclearable."""
    for cache in CACHES.values():
        try:
            mark_root(cache)
        except (OSError, SecurityError) as e:
            logger.warning(f"Could not mark the {cache.name} cache: {e}")


def _walk(top):
    """(bytes, files, newest mtime) of the regular files under `top`,
    symlinks neither followed nor counted, the marker not counted."""
    total, files, newest = 0, 0, None
    stack = [top]
    while stack:
        directory = stack.pop()
        try:
            entries = os.scandir(directory)
        except OSError:
            continue
        with entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if entry.name == MARKER and directory == top:
                        continue
                    stat = entry.stat(follow_symlinks=False)
                except OSError:
                    # A file deleted mid-walk (a racing compile's temp file)
                    continue
                total += stat.st_size
                files += 1
                if newest is None or stat.st_mtime > newest:
                    newest = stat.st_mtime
    return total, files, newest


def _volume(path):
    """shutil.disk_usage of the volume `path` is (or would be) on: a missing
    root is measured at its nearest existing ancestor."""
    while path and not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent
    try:
        return shutil.disk_usage(path)
    except OSError:
        return None


def _not_clearable_reason(root):
    if not os.path.isdir(root) or os.path.islink(root):
        return "root does not exist (nothing to clear)"
    if not _has_marker(root):
        return f"root not created by dw (no {MARKER} marker)"
    return None


def scan(cache):
    """One cache as plain data."""
    root = cache.root()
    present = os.path.isdir(root) and not os.path.islink(root)
    total, files, newest = _walk(root) if present else (0, 0, None)
    usage = _volume(root)
    reason = _not_clearable_reason(root)
    info = {
        "name": cache.name,
        "root": root,
        "enabled": cache.enabled(),
        "bytes": total,
        "files": files,
        "last_used": newest,
        "clearable": reason is None,
        "cost_of_clearing": cache.cost_of_clearing,
        "disk_free": usage.free if usage else None,
        "disk_total": usage.total if usage else None,
    }
    if reason:
        info["not_clearable_reason"] = reason
    return info


def scan_all():
    return [scan(cache) for cache in CACHES.values()]


def clear(name):
    """Empty a marked cache root, keeping the root and its marker. Returns
    `{freed, files}` for what was removed.

    Raises UnknownCache for a name not in the registry and NotClearable for a
    root without the marker - before anything is moved."""
    cache = get(name)
    with _clear_lock:
        root = cache.root()
        reason = _not_clearable_reason(root)
        if reason:
            raise NotClearable(f"The {name} cache at {root} can't be cleared: {reason}")
        root = validate_path(root)
        staging = validate_path(
            os.path.join(
                root, f"{_CLEARING_PREFIX}{os.getpid()}-{secrets.token_hex(4)}"
            ),
            base_dir=root,
        )
        os.mkdir(staging)
        try:
            for entry in _children(root):
                if entry.path == staging or entry.is_symlink():
                    # A link is left alone: validate_path resolves it, so
                    # the rename below would move its target, not the link
                    continue
                try:
                    source = validate_path(entry.path, base_dir=root)
                    target = validate_path(
                        os.path.join(staging, entry.name), base_dir=root
                    )
                except SecurityError:
                    # A name validate_path refuses (one holding "..") stays
                    continue
                try:
                    os.rename(source, target)
                except FileNotFoundError:
                    continue
            freed, files, _ = _walk(staging)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    logger.info(f"Cleared the {name} cache at {root} ({freed} bytes, {files} files)")
    return {"freed": freed, "files": files}
