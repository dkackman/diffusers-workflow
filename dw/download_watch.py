"""Makes a `from_pretrained` download visible on the run's event stream (#343).

`from_pretrained` downloads with no hook of its own, and a pull that runs
long looks identical to a hang: the phase-stall watchdog (dw/events.py) has
nothing but silence to report. This does not intercept or pre-fetch
anything - `from_pretrained` fetches exactly what it would have - it only
listens to the byte counts the download already reports while a `loading`
phase is in progress.

Two signals, and the larger is reported:

- The hub's own xet progress report (`XetDownloadProgressReporter`), which
  carries the bytes *received from the network* as well as the bytes written
  to disk. The network count is the one that matters: hf_xet buffers and
  writes a file in order, and on lem held a 2.8 GB file at 67 MB on disk
  while 700 MB had arrived, then wrote the rest at the very end. No location
  on disk tells that apart from a hang; the transfer count does. Hooked
  whether or not the hub's progress bars are displayed.
- The repo's cache directory, for a plain HTTP download (no hf_xet), which
  writes its `.incomplete` file as bytes arrive. A published blob is a
  symlink into the hub's shared store (`<cache>/blobs/xx/<sha>`), so the
  size is taken through the link - measured beside it, the total fell by the
  size of every file that finished (#343, a negative rate).

Both only ever count up, so `downloaded_bytes` never goes backwards.

A `download_progress` event fires about every EMIT_INTERVAL_SECONDS while a
download is underway. One reporting growth is progress. One reporting none
is not: it still fires, so `phase_detail` says `no bytes for 45s` instead of
repeating the last healthy rate, but the stall watchdog counts it as silence
and says `phase_stall` once bytes have been flat for its threshold - a
download and a hang no longer read the same.

A cancel during a download aborts it: the xet session is aborted (the hub's
own KeyboardInterrupt path, `abort_xet_session`) and a plain HTTP download is
stopped at its next chunk, and the load surfaces as `WorkflowCancelled`
rather than running to the end of a multi-gigabyte file first.
"""

import logging
import os
import threading
import time

from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.file_download import repo_folder_name
from huggingface_hub.utils import HFValidationError, validate_repo_id

from .events import WorkflowCancelled, get_context

logger = logging.getLogger("dw")

# How often the watcher re-measures, and the gap between emitted events -
# well under the phase-stall threshold (30s) so a real, ongoing download
# never trips it, but not so tight that a run emits an event per chunk.
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
    """Bytes under a repo's blobs/ directory, counting a published blob
    through its symlink into the shared store."""
    total = 0
    try:
        with os.scandir(blob_dir) as entries:
            for entry in entries:
                try:
                    if entry.is_file():
                        total += entry.stat().st_size
                except OSError:
                    # A blob renamed or published mid-scan is not a fault
                    continue
    except FileNotFoundError:
        return 0
    return total


# The watches currently open and the hub hooks they share. One job runs at a
# time in the worker, but a hook is process-wide, so it is installed on the
# first open watch and removed with the last.
_lock = threading.Lock()
_active = []
_originals = {}


def _note_hub_bytes(transferred, written):
    with _lock:
        watches = list(_active)
    for watch_ in watches:
        watch_._add_hub_bytes(transferred, written)


def _any_cancelled():
    with _lock:
        return any(w._context.cancelled for w in _active)


def _install_hooks():
    try:
        from huggingface_hub.utils import _xet_progress_reporting as xet_reporting

        reporter = xet_reporting.XetDownloadProgressReporter
        original = reporter.update_progress
        last_seen = {}

        def update_progress(self, group_report, *args, **kwargs):
            # Runs on hf_xet's callback thread, which prints and swallows
            # anything raised here - so nothing may be raised, and the
            # hub's own bar update is guarded along with the counting
            try:
                key = id(self)
                previous = last_seen.get(key, (0, 0))
                transferred = group_report.total_transfer_bytes_completed
                written = group_report.total_bytes_completed
                last_seen[key] = (
                    max(previous[0], transferred),
                    max(previous[1], written),
                )
                _note_hub_bytes(
                    max(0, transferred - previous[0]), max(0, written - previous[1])
                )
            except Exception as e:
                logger.debug(f"Download watch could not read xet progress: {e}")
            try:
                return original(self, group_report, *args, **kwargs)
            except Exception as e:
                logger.debug(f"xet progress update raised: {e}")

        _patch(reporter, "update_progress", update_progress)
    except (ImportError, AttributeError) as e:
        logger.debug(f"No xet progress reporter to hook: {e}")

    try:
        from importlib import import_module

        hub_tqdm = import_module("huggingface_hub.utils.tqdm").tqdm
        original_update = hub_tqdm.update

        def update(self, n=1):
            # A plain HTTP download calls this once per chunk on the thread
            # running from_pretrained - the one place that can stop it
            if _any_cancelled():
                raise WorkflowCancelled("Workflow run was cancelled")
            return original_update(self, n)

        _patch(hub_tqdm, "update", update)
    except (ImportError, AttributeError) as e:
        logger.debug(f"No hub progress bar to hook: {e}")


def _patch(owner, name, replacement):
    # Remembers whether the class defined the method itself or inherited it,
    # so removing the hook restores exactly what was there
    _originals[(owner, name)] = vars(owner).get(name)
    setattr(owner, name, replacement)


def _remove_hooks():
    for (owner, name), original in _originals.items():
        if original is None:
            delattr(owner, name)
        else:
            setattr(owner, name, original)
    _originals.clear()


def _abort_xet_downloads():
    try:
        from huggingface_hub.utils._xet import abort_xet_session

        abort_xet_session()
    except Exception as e:
        logger.debug(f"Could not abort the xet session: {e}")


class DownloadWatch:
    """Context manager that reports one repo's download on the active run
    context for as long as it is open.

    Not a download itself - `from_pretrained` runs unmodified inside the
    `with` block. A load that downloads nothing (every file cached) emits
    nothing, same as no watch at all.
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
        self._counts_lock = threading.Lock()
        self._transferred = 0
        self._written = 0
        self._aborted = False

    def _add_hub_bytes(self, transferred, written):
        with self._counts_lock:
            self._transferred += transferred
            self._written += written

    def _hub_bytes(self):
        with self._counts_lock:
            return max(self._transferred, self._written)

    def __enter__(self):
        with _lock:
            if not _active:
                _install_hooks()
            _active.append(self)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="dw-download-watch"
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=CHECK_INTERVAL_SECONDS * 2)
        with _lock:
            if self in _active:
                _active.remove(self)
            if not _active:
                _remove_hooks()
        if (
            exc is not None
            and not isinstance(exc, WorkflowCancelled)
            and self._context.cancelled
        ):
            # The abort surfaces as whatever the download library raises
            # (hf_xet: "RuntimeError: Operation cancelled") - it is the
            # cancel, not a load failure, and must read as one
            raise WorkflowCancelled("Workflow run was cancelled") from exc
        return False

    def _run(self):
        baseline = _blob_dir_size(self._blob_dir)
        disk_high_water = 0
        downloaded = 0
        emitted = 0
        last_emit_at = time.monotonic()
        last_change_at = last_emit_at
        while not self._stop.wait(CHECK_INTERVAL_SECONDS):
            try:
                if self._context.cancelled and not self._aborted:
                    self._aborted = True
                    _abort_xet_downloads()
                disk_high_water = max(
                    disk_high_water, _blob_dir_size(self._blob_dir) - baseline
                )
                now = time.monotonic()
                current = max(downloaded, disk_high_water, self._hub_bytes())
                if current > downloaded:
                    downloaded = current
                    last_change_at = now
                elapsed = now - last_emit_at
                if downloaded == 0 or elapsed < EMIT_INTERVAL_SECONDS:
                    continue
                growth = downloaded - emitted
                emitted = downloaded
                last_emit_at = now
                # Flat bytes still report, so phase_detail stops showing the
                # last healthy rate - but do not count as progress, so the
                # stall watchdog sees the silence a hang is
                self._context.emit(
                    "download_progress",
                    counts_as_progress=growth > 0,
                    repo_id=self._repo_id,
                    downloaded_bytes=downloaded,
                    bytes_per_second=growth / elapsed,
                    seconds_since_bytes_changed=round(now - last_change_at, 1),
                )
            except Exception as e:
                # A watch that breaks must not take the download down with it
                logger.debug(f"Download watch for '{self._repo_id}' failed: {e}")
                return


def _format_bytes(n):
    # Decimal units, as the labels say and as the hub itself reports sizes -
    # 1,879,623,333 bytes is 1.9 GB (it is 1.75 GiB)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n:.0f} {unit}"
        n /= 1000


def format_progress(
    repo_id, downloaded_bytes, bytes_per_second, seconds_since_bytes_changed=None
):
    """The `phase_detail` text a job's progress reports while this repo is
    downloading - `downloading <repo>: 12.3 GB, 38.0 MB/s` per #343, or
    `downloading <repo>: 12.3 GB, no bytes for 45s` once bytes stop."""
    text = f"downloading {repo_id}: {_format_bytes(downloaded_bytes or 0)}"
    if bytes_per_second:
        text += f", {_format_bytes(bytes_per_second)}/s"
    elif seconds_since_bytes_changed:
        text += f", no bytes for {seconds_since_bytes_changed:.0f}s"
    return text


def watch(repo_id, cache_dir=None, repo_type="model"):
    """A DownloadWatch on the active run context, or a no-op context manager
    when repo_id is not shaped like a hub repo (a local path, for instance)."""
    if not is_watchable_repo_id(repo_id):
        return _NULL_WATCH
    return DownloadWatch(
        repo_id, get_context(), cache_dir=cache_dir, repo_type=repo_type
    )


class _NullWatch:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


_NULL_WATCH = _NullWatch()
