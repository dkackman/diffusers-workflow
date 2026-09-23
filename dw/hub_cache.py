"""Inventory and deletion for the Hugging Face hub cache.

Every from_pretrained download lands in the hub cache and stays there until
something deletes it - a few video models is a few hundred GB. This wraps
huggingface_hub's own cache scanner so the server can show what is on disk
and free it, without inventing any path handling of its own: deletion goes
through scan_cache_dir's delete_revisions strategy, which only ever removes
revisions it found inside the cache directory.
"""

import json
import logging
import os
import shutil
import threading
import time
import uuid

from huggingface_hub import constants, scan_cache_dir
from huggingface_hub.utils import CacheNotFound

try:
    # Xet-backed downloads aggregate into two bars built from our tracker
    # class: reconstruction (file bytes written) and transfer (network
    # bytes). Only reconstruction matches the file-size total the manager
    # reports against - counting both would double the progress - and the
    # transfer bar is recognizable by the bar_format it is created with
    from huggingface_hub.utils._xet_progress_reporting import XET_TRANSFER_BAR_FORMAT
except ImportError:  # older huggingface_hub - no xet aggregation
    XET_TRANSFER_BAR_FORMAT = None

logger = logging.getLogger("dw")


def _resolved_cache_dir(cache_dir):
    return str(cache_dir) if cache_dir else constants.HF_HUB_CACHE


def scan_models(cache_dir=None):
    """The cache's contents as plain data: repos sorted largest-first,
    with per-revision detail, plus disk totals for the volume it lives on."""
    resolved = _resolved_cache_dir(cache_dir)
    try:
        scan = scan_cache_dir(resolved)
    except CacheNotFound:
        # No cache directory yet - nothing downloaded is a state, not an error
        return {
            "cache_dir": resolved,
            "size_on_disk": 0,
            "repos": [],
            "warnings": [],
            "disk_free": None,
            "disk_total": None,
        }

    repos = []
    for repo in scan.repos:
        revisions = sorted(
            repo.revisions, key=lambda rev: rev.last_modified or 0, reverse=True
        )
        repos.append(
            {
                "repo_id": repo.repo_id,
                "repo_type": repo.repo_type,
                "size_on_disk": repo.size_on_disk,
                "nb_files": repo.nb_files,
                "last_accessed": repo.last_accessed,
                "last_modified": repo.last_modified,
                "revisions": [
                    {
                        "commit_hash": revision.commit_hash,
                        "size_on_disk": revision.size_on_disk,
                        "refs": sorted(revision.refs),
                        "last_modified": revision.last_modified,
                    }
                    for revision in revisions
                ],
            }
        )
    repos.sort(key=lambda entry: entry["size_on_disk"], reverse=True)

    usage = shutil.disk_usage(resolved)
    return {
        "cache_dir": resolved,
        "size_on_disk": scan.size_on_disk,
        "repos": repos,
        "warnings": [str(warning) for warning in scan.warnings],
        "disk_free": usage.free,
        "disk_total": usage.total,
    }


_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".msgpack", ".onnx", ".pt")


def _repo_folder_name(repo_id, repo_type="model"):
    return f"{repo_type}s--{repo_id.replace('/', '--')}"


def _snapshot_dir(repo_dir):
    """The snapshot directory `refs/main` currently points at, or None -
    a repo dir with no readable ref or no matching snapshot folder is
    itself a sign the pull never finished."""
    try:
        with open(os.path.join(repo_dir, "refs", "main")) as f:
            commit = f.read().strip()
    except OSError:
        return None
    snapshot = os.path.join(repo_dir, "snapshots", commit)
    return snapshot if os.path.isdir(snapshot) else None


def _component_incomplete(folder, variant):
    """Whether a diffusers pipeline component's folder is missing files a
    load would need. A component with no weight files at all (a scheduler
    or tokenizer config) is complete once its folder exists; one with
    weight files is checked against the requested `variant` only when that
    variant is actually in play, since an unvarianted component (most
    schedulers, safety checkers) never carries a tagged file."""
    if not os.path.isdir(folder):
        return True
    try:
        entries = os.listdir(folder)
    except OSError:
        return True
    if not entries:
        return True
    if not variant:
        return False
    weight_files = [e for e in entries if e.endswith(_WEIGHT_SUFFIXES)]
    if not weight_files:
        return False
    return not any(f".{variant}." in name for name in weight_files)


def repo_download_incomplete(repo_id, cache_dir=None, variant=None):
    """Whether repo_id, though listed by `scan_models`, is left over from an
    interrupted pull rather than fully fetched (#382): a cancelled
    `snapshot_download` leaves the revision folder in place - the repo still
    shows up in `scan_cache_dir` - but with an in-progress blob or with whole
    components never fetched.

    Two checks, in order of how cheap they are to rule out: any
    `*.incomplete` blob (huggingface_hub's own marker for a file mid-transfer)
    anywhere in the repo's `blobs/` makes the repo incomplete outright. Past
    that, a diffusers pipeline repo (one with a `model_index.json` in its
    current snapshot) is checked component by component, for the requested
    `variant`; a plain checkpoint/LoRA repo (no `model_index.json`) has
    nothing further to check beyond its blobs.  This is not a full hub
    file-list diff - it does not fetch the repo's file list from the hub, so
    a file that was never attempted at all on an otherwise-untouched repo
    (nothing downloaded, no blobs, no snapshot) is caught by the "no snapshot"
    case, not enumerated.
    """
    resolved = _resolved_cache_dir(cache_dir)
    repo_dir = os.path.join(resolved, _repo_folder_name(repo_id))
    if not os.path.isdir(repo_dir):
        return True

    blobs_dir = os.path.join(repo_dir, "blobs")
    if os.path.isdir(blobs_dir):
        try:
            if any(name.endswith(".incomplete") for name in os.listdir(blobs_dir)):
                return True
        except OSError:
            return True

    snapshot = _snapshot_dir(repo_dir)
    if snapshot is None:
        return True

    index_path = os.path.join(snapshot, "model_index.json")
    if not os.path.isfile(index_path):
        # Not a diffusers pipeline repo - blob completeness is all there is
        return False
    try:
        with open(index_path) as f:
            index = json.load(f)
    except (OSError, ValueError):
        return False

    for key, value in index.items():
        if key.startswith("_") or not isinstance(value, list):
            continue
        if _component_incomplete(os.path.join(snapshot, key), variant):
            return True
    return False


def delete_model(repo_id, cache_dir=None):
    """Delete every cached revision of repo_id. Returns the bytes freed.

    Raises ValueError when the repo is not in the cache - the caller typed
    or raced something, and nothing was deleted.
    """
    scan = scan_cache_dir(_resolved_cache_dir(cache_dir))
    repo = next((r for r in scan.repos if r.repo_id == repo_id), None)
    if repo is None:
        raise ValueError(f"'{repo_id}' is not in the hub cache")

    strategy = scan.delete_revisions(
        *[revision.commit_hash for revision in repo.revisions]
    )
    freed = strategy.expected_freed_size
    strategy.execute()
    return freed


# ------------------------------------------------------------------ downloads


class DownloadCancelled(Exception):
    """Raised inside a download's progress callback to abort it."""


class DownloadManager:
    """Background snapshot downloads into the hub cache, with progress.

    One thread per download; progress is fed by a tqdm-compatible tracker
    that snapshot_download instantiates per file, so the counters aggregate
    across the file pool. Cancellation raises out of the next progress tick;
    huggingface_hub's partial files remain resumable, so a cancelled or
    failed download picks up where it stopped when retried.
    """

    KEEP_FINISHED = 20

    def __init__(self, download_fn=None, info_fn=None):
        # Injectable for tests - the defaults reach the network
        if download_fn is None or info_fn is None:
            from huggingface_hub import HfApi, snapshot_download

            download_fn = download_fn or snapshot_download
            info_fn = info_fn or (
                lambda repo_id: HfApi().repo_info(repo_id, files_metadata=True)
            )
        self._download_fn = download_fn
        self._info_fn = info_fn
        self._lock = threading.Lock()
        self._downloads = {}

    def start(self, repo_id):
        """Begin downloading repo_id; returns the download's status dict.
        Raises ValueError for an invalid repo id or one already in flight."""
        from huggingface_hub.utils import HFValidationError, validate_repo_id

        try:
            validate_repo_id(repo_id)
        except HFValidationError as e:
            raise ValueError(str(e))

        cancel_event = threading.Event()
        with self._lock:
            for entry in self._downloads.values():
                if entry["repo_id"] == repo_id and entry["status"] == "downloading":
                    raise ValueError(f"'{repo_id}' is already downloading")
            entry = {
                "id": uuid.uuid4().hex[:12],
                "repo_id": repo_id,
                "status": "downloading",
                "downloaded": 0,
                "total": None,
                "error": None,
                "started_at": time.time(),
                "finished_at": None,
                # Set before the entry is published: cancel() reaches for this
                # the moment the id is visible, and assigning it after the
                # lock left a window where cancelling raised KeyError
                "_cancel": cancel_event,
            }
            self._downloads[entry["id"]] = entry
            self._prune()

        thread = threading.Thread(
            target=self._run, args=(entry, cancel_event), daemon=True
        )
        thread.start()
        return self.status(entry["id"])

    def _run(self, entry, cancel_event):
        try:
            try:
                info = self._info_fn(entry["repo_id"])
                total = sum(sibling.size or 0 for sibling in (info.siblings or []))
                with self._lock:
                    entry["total"] = total or None
            except Exception as e:
                # Size is cosmetic; the download itself decides success
                logger.debug(f"No size metadata for {entry['repo_id']}: {e}")

            self._download_fn(
                entry["repo_id"], tqdm_class=_tracker_class(self, entry, cancel_event)
            )
            self._finish(entry, "completed")
        except DownloadCancelled:
            self._finish(entry, "cancelled")
        except Exception as e:
            self._finish(entry, "failed", str(e))

    def _finish(self, entry, status, error=None):
        with self._lock:
            entry["status"] = status
            entry["error"] = error
            entry["finished_at"] = time.time()

    def _add_progress(self, entry, n):
        with self._lock:
            entry["downloaded"] += n

    def cancel(self, download_id):
        """Request cancellation; returns the status dict or None if unknown.
        Takes effect at the download's next progress tick."""
        with self._lock:
            entry = self._downloads.get(download_id)
            if entry is None:
                return None
            if entry["status"] == "downloading":
                entry["_cancel"].set()
        return self.status(download_id)

    def status(self, download_id):
        with self._lock:
            entry = self._downloads.get(download_id)
            return _public(entry) if entry else None

    def status_list(self):
        """Every tracked download, newest first."""
        with self._lock:
            entries = sorted(
                self._downloads.values(),
                key=lambda e: e["started_at"],
                reverse=True,
            )
            return [_public(entry) for entry in entries]

    def is_active(self):
        with self._lock:
            return any(
                entry["status"] == "downloading" for entry in self._downloads.values()
            )

    def _prune(self):
        # Called with the lock held: drop the oldest finished entries
        finished = sorted(
            (e for e in self._downloads.values() if e["status"] != "downloading"),
            key=lambda e: e["started_at"],
        )
        excess = len(finished) - self.KEEP_FINISHED
        for entry in finished[: max(0, excess)]:
            del self._downloads[entry["id"]]


def _public(entry):
    return {key: value for key, value in entry.items() if not key.startswith("_")}


def _tracker_class(manager, entry, cancel_event):
    """A tqdm stand-in snapshot_download instantiates per file; every update
    feeds the shared counters and honours cancellation."""

    class Tracker:
        def __init__(self, *args, **kwargs):
            self.n = 0
            self.total = kwargs.get("total")
            # The xet transfer bar still ticks (cancellation works through
            # it) but its network bytes stay out of the shared counters
            self._counted = (
                XET_TRANSFER_BAR_FORMAT is None
                or kwargs.get("bar_format") != XET_TRANSFER_BAR_FORMAT
            )

        def update(self, n=1):
            if cancel_event.is_set():
                raise DownloadCancelled()
            if n:
                self.n += n
                if self._counted:
                    manager._add_progress(entry, n)
            return True

        def close(self):
            pass

        def refresh(self):
            pass

        def set_description(self, *args, **kwargs):
            pass

        def set_postfix(self, *args, **kwargs):
            pass

        def set_postfix_str(self, *args, **kwargs):
            pass

        @property
        def format_dict(self):
            # What tqdm exposes for rate rendering; huggingface_hub reads
            # 'rate' from it when composing the xet speed postfix
            return {"n": self.n, "total": self.total, "elapsed": 0, "rate": None}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

        @staticmethod
        def get_lock():
            return threading.RLock()

        @staticmethod
        def set_lock(lock):
            pass

    return Tracker
