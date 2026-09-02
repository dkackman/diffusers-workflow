"""Upgrade diffusers to GitHub HEAD (or a pinned commit) from inside the
running server, or revert to the last known-good published release.

The newest model pipelines often land in diffusers git before a release,
so the Models page offers the same install install.sh performs. pip runs
in a background thread against this venv with a fixed argument list built
from validated inputs - no request data reaches the command line unchecked.
The persistent worker keeps the old import until it restarts, so on success
the caller shuts an idle worker down and the next job starts on the new
version.
"""

import json
import logging
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..security import sanitize_command_args, validate_commit_hash

logger = logging.getLogger("dw")

DIFFUSERS_GIT_URL = "git+https://github.com/huggingface/diffusers"
PYPI_PACKAGE = "diffusers"
PIP_TIMEOUT_SECONDS = 900
LOG_TAIL_LINES = 30

# Used only if pyproject.toml can't be read/parsed at runtime (e.g. running
# from an installed wheel without the source tree alongside it).
FALLBACK_RELEASE_FLOOR = "0.40.0"

# dw/server/updater.py -> dw/server -> dw -> repo root
_PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"
_RELEASE_FLOOR_PATTERN = re.compile(r'"diffusers>=([0-9][0-9.]*[0-9]|[0-9])"')


def release_floor():
    """The diffusers version floor pinned in pyproject.toml - the oldest
    release this project is declared to work against, and what "revert to
    a known-good release" pins to when the git HEAD install is broken."""
    try:
        text = _PYPROJECT_PATH.read_text()
        match = _RELEASE_FLOOR_PATTERN.search(text)
        if match:
            return match.group(1)
    except OSError:
        pass
    return FALLBACK_RELEASE_FLOOR


def diffusers_install_info():
    """The installed diffusers version and, for a git install, its commit
    (from the PEP 610 direct_url.json pip records)."""
    from importlib.metadata import distribution, PackageNotFoundError

    info = {"version": None, "commit": None}
    try:
        dist = distribution("diffusers")
    except PackageNotFoundError:
        return info
    info["version"] = dist.version
    try:
        direct_url = dist.read_text("direct_url.json")
        if direct_url:
            info["commit"] = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    except Exception:
        pass
    return info


def build_pip_args(commit=None, revert=False):
    """The pip argument list for one of three installs: HEAD, a pinned
    commit, or a revert to the pyproject.toml release floor. `commit` is
    validated by the caller before it reaches here - this only assembles
    the already-validated pieces, never a raw request value.
    """
    if revert:
        return [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"{PYPI_PACKAGE}=={release_floor()}",
        ]
    target = DIFFUSERS_GIT_URL if not commit else f"{DIFFUSERS_GIT_URL}@{commit}"
    return [sys.executable, "-m", "pip", "install", "--upgrade", target]


class DiffusersUpdater:
    """One upgrade at a time, in a background thread, with a status dict
    the UI polls."""

    def __init__(self, run_fn=None):
        # Injectable for tests - the default runs pip against this venv
        self._run_fn = run_fn or self._run_pip
        self._lock = threading.Lock()
        self._state = {
            "status": "idle",
            "error": None,
            "log": None,
            "started_at": None,
            "finished_at": None,
            "requested_commit": None,
            "revert": False,
            "before": None,
        }

    @staticmethod
    def _run_pip(commit=None, revert=False):
        args = sanitize_command_args(build_pip_args(commit=commit, revert=revert))
        return subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=PIP_TIMEOUT_SECONDS,
        )

    def start(self, on_success=None, commit=None, revert=False):
        """Begin the upgrade; returns the status dict. Raises ValueError if
        one is already running or the arguments conflict. on_success runs
        in the worker thread after pip exits cleanly.

        commit: a pre-validated git commit hash (see
        dw.security.validate_commit_hash) to pin the git install to,
        instead of tracking HEAD. Mutually exclusive with revert.
        revert: pin back to the diffusers release floor from pyproject.toml
        instead of installing from git - the way back when a git HEAD (or
        a pinned commit) turns out to be broken.
        """
        if commit and revert:
            raise ValueError("commit and revert are mutually exclusive")
        with self._lock:
            if self._state["status"] == "running":
                raise ValueError("A diffusers update is already running")
            before = diffusers_install_info()
            self._state = {
                "status": "running",
                "error": None,
                "log": None,
                "started_at": time.time(),
                "finished_at": None,
                "requested_commit": commit,
                "revert": revert,
                "before": before,
            }
        thread = threading.Thread(
            target=self._run, args=(on_success, commit, revert, before), daemon=True
        )
        thread.start()
        return self.status()

    def _run(self, on_success, commit, revert, before):
        try:
            completed = self._run_fn(commit=commit, revert=revert)
            ok = completed.returncode == 0
            output = (completed.stdout or "") + "\n" + (completed.stderr or "")
            tail = "\n".join(output.strip().splitlines()[-LOG_TAIL_LINES:])
            with self._lock:
                self._state["status"] = "succeeded" if ok else "failed"
                self._state["error"] = (
                    None if ok else f"pip exited with code {completed.returncode}"
                )
                self._state["log"] = tail
                self._state["finished_at"] = time.time()
            if ok:
                after = diffusers_install_info()
                what = (
                    f"reverted to release {after.get('version')}"
                    if revert
                    else f"upgraded from git ({commit or 'HEAD'})"
                )
                logger.info(f"diffusers {what}: before={before} after={after}")
                if on_success:
                    on_success()
            else:
                logger.error(f"diffusers update failed: {tail}")
        except Exception as e:  # pip timeout, missing interpreter, ...
            with self._lock:
                self._state["status"] = "failed"
                self._state["error"] = str(e)
                self._state["finished_at"] = time.time()
            logger.error(f"diffusers update failed: {e}")

    def status(self):
        """The update's state plus what is installed right now."""
        with self._lock:
            state = dict(self._state)
        state.update(diffusers_install_info())
        return state
