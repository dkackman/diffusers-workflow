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
- `quantized`: SDNQ components quantized once and saved (#815,
  `dw/quantized_cache.py`), under `DW_QUANTIZED_CACHE_DIR`, enabled by
  `DW_QUANTIZED_CACHE=1`. Its entries - one `<64-hex key>/` directory each,
  described by its `entry.json` - are listed and can be deleted one at a time
  (`DELETE /api/caches/quantized/entries/{entry}`). A cache that is off still
  lists, and still clears, what is on disk.

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
reads a half-deleted tree. An entry delete moves the one entry the same way. The
routes refuse while a job is running or queued, and hold the job queue still
(`JobManager.hold_dispatch`) while they work, so a job submitted meanwhile waits
instead of starting into a root being emptied. A staging directory a killed
server left behind is removed at `mark_roots()` and at the next clear, and is
never counted by a scan. `scan_all()` reuses its answer for a few seconds.

**An entry id must have a cache key's shape** (`quantized_cache.is_key`, 64
lowercase hex) before any path is built from it.
"""

import importlib.util
import logging
import os
import secrets
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Callable

from . import _inductor_cache_dir_default, quantized_cache
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


# How long scan_all's answer is reused. A scan stat()s every file of every
# cache, and get_server_info asks on each call (#840)
_SCAN_TTL = 5.0
_scan_memo = None  # (monotonic time, scan_all's result)


class UnknownCache(ValueError):
    pass


class NotClearable(Exception):
    pass


class NoEntries(ValueError):
    """An entry operation on a cache that has no entries, or an entry id
    that isn't one (a 400)."""


class UnknownEntry(ValueError):
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
    # Whether the cache lists entries of its own: `<root>/<64-hex id>/`
    # directories, each described by an `entry.json`
    has_entries: bool = False
    # Whether mark_roots may create a missing root
    create_root: bool = True
    # The free space under which the cache's volume reads `low_disk`, from
    # the volume's total - for a cache that refuses to write below a floor
    low_disk_floor: Callable[[int], int] | None = None


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
        DiskCache(
            name="quantized",
            root=quantized_cache.cache_root,
            default_root=quantized_cache.default_root,
            enabled=quantized_cache.enabled,
            cost_of_clearing=(
                "The next load of each SDNQ-quantized component reads its "
                "bf16 weights and quantizes them again (minutes per component "
                "on MiniMax-H3), and saves a new entry while the cache is on."
            ),
            has_entries=True,
            # configure_at_startup creates it when the cache is on; a server
            # with it off gets no stray directory
            create_root=False,
            low_disk_floor=quantized_cache.floor_bytes,
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


def _sweep_staging(root):
    """Delete the `.clearing-*` directories a clear that died mid-delete left
    in `root`. Only called where no clear can be running: server start, or
    inside `_clear_lock`."""
    try:
        leftovers = [
            entry.path
            for entry in _children(root)
            if entry.name.startswith(_CLEARING_PREFIX)
            and entry.is_dir(follow_symlinks=False)
        ]
    except OSError:
        return
    for path in leftovers:
        shutil.rmtree(path, ignore_errors=True)
        logger.info(f"Removed the leftover {path} from an interrupted clear")


def mark_root(cache, quiet=False):
    """Mark one cache's root as dw's to clear, if the rule allows it.
    Returns whether the root carries the marker afterwards. `quiet` logs the
    refusal at DEBUG: only the startup pass says it at INFO (#841)."""
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
        logger.log(
            logging.DEBUG if quiet else logging.INFO,
            f"The {cache.name} cache at {root} was not created by dw and "
            f"has no {MARKER} marker, so it can't be cleared from the server",
        )
        return False
    _write_marker(root)
    return True


def mark_roots():
    """Write the ownership marker into each cache root the rule allows. Run
    once at server start; a root it can't write is left unclearable."""
    for cache in CACHES.values():
        try:
            if mark_root(cache):
                _sweep_staging(cache.root())
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
                if directory == top and entry.name.startswith(_CLEARING_PREFIX):
                    # A clear's staging directory is on its way out
                    continue
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


def _reason(cache):
    """Why `cache` can't be cleared now, or None. A root that came into being
    after server start (Triton's, made by the first compile) is marked here
    by the same rule mark_roots applies at startup."""
    root = cache.root()
    if os.path.isdir(root) and not _has_marker(root):
        try:
            mark_root(cache, quiet=True)
        except (OSError, SecurityError) as e:
            logger.warning(f"Could not mark the {cache.name} cache: {e}")
    return _not_clearable_reason(root)


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
    reason = _reason(cache)
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
    if cache.low_disk_floor and usage:
        info["low_disk"] = usage.free < cache.low_disk_floor(usage.total)
    if cache.has_entries:
        info["entries"] = scan_entries(root) if present else []
    return info


def _entry_fields(path):
    """What an entry's `entry.json` says of it, or None when it can't be
    read. `component` falls back to the subfolder - the component's name in
    a diffusers repo - since entry.json doesn't record it."""
    try:
        entry = quantized_cache.read_entry(path)
    except (OSError, ValueError):
        return None
    return {
        "model": entry.get("model_name"),
        "subfolder": entry.get("subfolder"),
        "component": entry.get("component") or entry.get("subfolder") or None,
        "created": entry.get("created"),
        "last_hit": entry.get("last_hit"),
    }


def scan_entries(root):
    """Each `<64-hex id>/` directory under a cache root, newest first. An
    entry whose `entry.json` is missing or unreadable is listed with
    `unreadable: true`, so it can still be deleted."""
    entries = []
    with os.scandir(root) as children:
        for child in children:
            if not quantized_cache.is_key(child.name) or not child.is_dir(
                follow_symlinks=False
            ):
                continue
            total, files, newest = _walk(child.path)
            fields = _entry_fields(child.path)
            entry = {"id": child.name, "bytes": total, "files": files}
            if fields is None:
                entry.update(
                    unreadable=True,
                    model=None,
                    subfolder=None,
                    component=None,
                    created=newest,
                    last_hit=None,
                )
            else:
                entry.update(unreadable=False, **fields)
            entries.append(entry)
    entries.sort(key=lambda e: e["created"] or 0, reverse=True)
    return entries


def scan_all():
    """Every cache as plain data, reused for `_SCAN_TTL` seconds; a clear or
    entry delete drops the reused answer."""
    global _scan_memo
    memo = _scan_memo
    if memo is not None and time.monotonic() - memo[0] < _SCAN_TTL:
        return memo[1]
    result = [scan(cache) for cache in CACHES.values()]
    _scan_memo = (time.monotonic(), result)
    return result


def _forget_scans():
    global _scan_memo
    _scan_memo = None


def entry_cache(name, entry):
    """The cache an entry route names, after checking the id's shape - so
    nothing builds a path from a malformed one. Raises UnknownCache, or
    NoEntries for a cache without entries or an id that isn't 64 hex."""
    cache = get(name)
    if not cache.has_entries:
        raise NoEntries(f"The {name} cache has no entries - clear it whole")
    if not quantized_cache.is_key(entry):
        raise NoEntries(f"Not an entry id: {entry!r} (64 lowercase hex characters)")
    return cache


def _stage_out(root, sources):
    """Move `sources` (paths directly under `root`) into one fresh
    `.clearing-*` directory and delete it. Returns (bytes, files) removed."""
    staging = validate_path(
        os.path.join(root, f"{_CLEARING_PREFIX}{os.getpid()}-{secrets.token_hex(4)}"),
        base_dir=root,
    )
    os.mkdir(staging)
    try:
        for path in sources:
            try:
                source = validate_path(path, base_dir=root)
                target = validate_path(
                    os.path.join(staging, os.path.basename(path)), base_dir=root
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
    return freed, files


def delete_entry(name, entry):
    """Delete one entry of a marked cache root. Returns `{freed, files}`.

    Raises UnknownCache, NoEntries (no entries here, or a malformed id),
    NotClearable for a root without the marker, or UnknownEntry - each
    before anything is moved."""
    cache = entry_cache(name, entry)
    with _clear_lock:
        root = cache.root()
        reason = _reason(cache)
        if reason:
            raise NotClearable(f"The {name} cache at {root} can't be changed: {reason}")
        root = validate_path(root)
        _sweep_staging(root)
        path = quantized_cache.entry_path(root, entry)
        if os.path.islink(path) or not os.path.isdir(path):
            raise UnknownEntry(f"The {name} cache has no entry {entry}")
        freed, files = _stage_out(root, [path])
        _forget_scans()
    logger.info(f"Deleted {name} cache entry {entry} ({freed} bytes, {files} files)")
    return {"freed": freed, "files": files}


def clear(name):
    """Empty a marked cache root, keeping the root and its marker. Returns
    `{freed, files}` for what was removed.

    Raises UnknownCache for a name not in the registry and NotClearable for a
    root without the marker - before anything is moved."""
    cache = get(name)
    with _clear_lock:
        root = cache.root()
        reason = _reason(cache)
        if reason:
            raise NotClearable(f"The {name} cache at {root} can't be cleared: {reason}")
        root = validate_path(root)
        _sweep_staging(root)
        # A link is left alone: validate_path resolves it, so the rename
        # would move its target, not the link
        freed, files = _stage_out(
            root, [entry.path for entry in _children(root) if not entry.is_symlink()]
        )
        _forget_scans()
    logger.info(f"Cleared the {name} cache at {root} ({freed} bytes, {files} files)")
    return {"freed": freed, "files": files}
