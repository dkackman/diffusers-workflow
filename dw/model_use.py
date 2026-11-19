"""The model-use index: when a dw job last used each Hub repo (#1445).

The hub cache's own `last_accessed` is the file atime - any process that
opens a blob moves it, a `noatime` mount never does, and macOS updates it
lazily. This index is the record dw keeps itself: when a job's run starts
(the worker's `run_start` event, `dw/server/jobs.py` `_run_on`), every repo
admission found the job names (`dw.plan_downloads.repos_named`) is stamped
with that time. `GET /api/models` serves it per repo as `last_job_use`;
deleting a cached repo forgets its entry.

One JSON file, `<settings dir>/model-use.json`:
`{"version": 1, "repos": {"<repo id>": <epoch seconds>}}`. A write merges
under a file lock (`model-use.json.lock`, so two servers sharing a settings
dir keep each other's stamps), keeps the newer stamp per repo, and replaces
the file atomically. The index is capped at `MAX_REPOS` entries: the repos
just stamped always stay, the rest of the room goes to the newest.

Everything here is best-effort. A missing, corrupt or unwritable file reads
as empty and costs the stamp, never the job: an error is logged at debug and
swallowed."""

import json
import logging
import os

from filelock import FileLock
from huggingface_hub.utils import HFValidationError, validate_repo_id

from .settings import get_settings_dir

logger = logging.getLogger("dw")

FILE_NAME = "model-use.json"
VERSION = 1
MAX_REPOS = 1000
LOCK_TIMEOUT = 10


def _path():
    return get_settings_dir() / FILE_NAME


def _is_repo_id(name):
    if not isinstance(name, str) or not name:
        return False
    try:
        validate_repo_id(name)
    except (HFValidationError, ValueError):
        return False
    return True


def _load(path):
    """The `repos` map in the file at `path`; `{}` for a missing or corrupt
    file, and only well-formed entries of a readable one."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as error:
        logger.debug("model-use index %s unreadable: %s", path, error)
        return {}
    repos = data.get("repos") if isinstance(data, dict) else None
    if not isinstance(repos, dict):
        return {}
    return {
        repo: stamp
        for repo, stamp in repos.items()
        if _is_repo_id(repo)
        and isinstance(stamp, (int, float))
        and not isinstance(stamp, bool)
    }


def read():
    """Every recorded repo and the epoch seconds a job last used it."""
    try:
        return _load(_path())
    except Exception as error:  # best-effort: never fail a listing
        logger.debug("model-use index read failed: %s", error)
        return {}


def _write(path, repos):
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump({"version": VERSION, "repos": repos}, handle)
    os.replace(tmp, path)


def _cap(repos, current):
    """At most `MAX_REPOS` entries: every repo in `current` kept, the rest of
    the room filled by the newest stamps."""
    if len(repos) <= MAX_REPOS:
        return repos
    kept = {repo: repos[repo] for repo in current if repo in repos}
    rest = sorted(
        (item for item in repos.items() if item[0] not in kept),
        key=lambda item: item[1],
        reverse=True,
    )
    for repo, stamp in rest[: max(MAX_REPOS - len(kept), 0)]:
        kept[repo] = stamp
    return kept


def record(repos, when):
    """Stamp each repo id in `repos` as used at `when` (epoch seconds). A name
    that is not a hub id is skipped; an older stamp never replaces a newer
    one."""
    try:
        current = []
        for repo in repos or []:
            if _is_repo_id(repo) and repo not in current:
                current.append(repo)
        if not current:
            return
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(path) + ".lock", timeout=LOCK_TIMEOUT, thread_local=False):
            merged = _load(path)
            for repo in current:
                previous = merged.get(repo)
                merged[repo] = when if previous is None else max(previous, when)
            _write(path, _cap(merged, current))
    except Exception as error:  # best-effort: never fail a run
        logger.debug("model-use index record failed: %s", error)


def forget(repo):
    """Drop `repo`'s entry, if the index holds one."""
    try:
        path = _path()
        if not path.exists():
            return
        with FileLock(str(path) + ".lock", timeout=LOCK_TIMEOUT, thread_local=False):
            merged = _load(path)
            if merged.pop(repo, None) is None:
                return
            _write(path, merged)
    except Exception as error:  # best-effort: the delete already succeeded
        logger.debug("model-use index forget failed: %s", error)
