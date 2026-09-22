"""Makes a `from_pretrained` download visible on the run's event stream (#343).

`from_pretrained` downloads with no hook of its own, and a pull that runs
long looks identical to a hang: the phase-stall watchdog (dw/events.py) has
nothing but silence to report. Rather than intercepting the download - which
would mean second-guessing what `from_pretrained` fetches, and getting it
wrong for a variant or an alternate weight file it would not have pulled -
this only watches the Hugging Face cache directory the download writes into
while a `loading` phase is in progress, the same directory `hub_cache.py`
scans for `list_downloads`. Byte growth there is real progress whoever
triggered it, and is reported as such.
"""

import logging
import os
import threading
import time

from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.file_download import repo_folder_name
from huggingface_hub.utils import HFValidationError, validate_repo_id

logger = logging.getLogger("dw")

# How often the watcher re-measures the cache directory, and the minimum gap
# between emitted events - well under the phase-stall threshold (30s) so a
# real, ongoing download never trips it, but not so tight that a run emits a
# progress event on every tick of a fast-growing directory.
CHECK_INTERVAL_SECONDS = 1.0
EMIT_INTERVAL_SECONDS = 5.0


def is_watchable_repo_id(name):
    """Whether name is shaped like a Hugging Face repo id - a local path or
    checkout is not something a cache directory watch means anything for."""
    try:
        validate_repo_id(name)
        return True
    except HFValidationError:
        return False


def _blob_dir_size(blob_dir):
    total = 0
    try:
        with os.scandir(blob_dir) as entries:
            for entry in entries:
                try:
                    if entry.is_file(follow_symlinks=False):
                        total += entry.stat(follow_symlinks=False).st_size
                except OSError:
                    # A blob renamed or removed mid-scan (a completed
                    # .incomplete file, for instance) is not a fault
                    continue
    except FileNotFoundError:
        return 0
    return total


class DownloadWatch:
    """Context manager that watches one repo's cache directory for growth
    for as long as it is open, emitting a throttled `download_progress`
    event on the active run context while bytes are arriving.

    Not a download itself - `from_pretrained` runs unmodified inside the
    `with` block. Growth with nothing watching it (a cache miss the caller
    already had, or a repo id from_pretrained resolves differently than
    expected) simply emits nothing, same as no watch at all.
    """

    def __init__(self, repo_id, context, cache_dir=None, repo_type="model"):
        self._repo_id = repo_id
        self._context = context
        resolved_cache_dir = cache_dir or HF_HUB_CACHE
        self._blob_dir = os.path.join(
            resolved_cache_dir,
            repo_folder_name(repo_id=repo_id, repo_type=repo_type),
            "blobs",
        )
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="dw-download-watch"
        )
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=CHECK_INTERVAL_SECONDS * 2)
        return False

    def _run(self):
        baseline = _blob_dir_size(self._blob_dir)
        last_size = baseline
        last_sample_at = time.monotonic()
        last_emit_at = 0.0
        while not self._stop.wait(CHECK_INTERVAL_SECONDS):
            try:
                size = _blob_dir_size(self._blob_dir)
                now = time.monotonic()
                if size <= last_size:
                    last_size = size
                    last_sample_at = now
                    continue
                elapsed = now - last_sample_at
                rate = (size - last_size) / elapsed if elapsed > 0 else None
                last_size = size
                last_sample_at = now
                if now - last_emit_at < EMIT_INTERVAL_SECONDS:
                    continue
                last_emit_at = now
                self._context.emit(
                    "download_progress",
                    repo_id=self._repo_id,
                    downloaded_bytes=max(0, size - baseline),
                    bytes_per_second=rate,
                )
            except Exception as e:
                # A watch that breaks must not take the download down with it
                logger.debug(f"Download watch for '{self._repo_id}' failed: {e}")
                return


def _format_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n:.0f} {unit}"
        n /= 1024


def format_progress(repo_id, downloaded_bytes, bytes_per_second):
    """The `phase_detail` text a job's progress reports while this repo is
    downloading - `downloading <repo>: 12.3 GB, 38 MB/s`, per #343."""
    text = f"downloading {repo_id}: {_format_bytes(downloaded_bytes)}"
    if bytes_per_second:
        text += f", {_format_bytes(bytes_per_second)}/s"
    return text


def watch(repo_id, cache_dir=None, repo_type="model"):
    """A DownloadWatch on the active run context, or a no-op context manager
    when repo_id is not shaped like a hub repo (a local path, for instance)."""
    from .events import get_context

    if not is_watchable_repo_id(repo_id):
        return _NULL_WATCH
    return DownloadWatch(repo_id, get_context(), cache_dir=cache_dir, repo_type=repo_type)


class _NullWatch:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


_NULL_WATCH = _NullWatch()
