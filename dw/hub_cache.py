"""Inventory and deletion for the Hugging Face hub cache.

Every from_pretrained download lands in the hub cache and stays there until
something deletes it - a few video models is a few hundred GB. This wraps
huggingface_hub's own cache scanner so the server can show what is on disk
and free it, without inventing any path handling of its own: deletion goes
through scan_cache_dir's delete_revisions strategy, which only ever removes
revisions it found inside the cache directory.
"""

import shutil

from huggingface_hub import constants, scan_cache_dir
from huggingface_hub.utils import CacheNotFound


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
