"""Upgrade diffusers to GitHub HEAD from inside the running server.

The newest model pipelines often land in diffusers git before a release,
so the Models page offers the same install install.sh performs. pip runs
in a background thread against this venv with a fixed argument list - no
request data reaches the command line. The persistent worker keeps the old
import until it restarts, so on success the caller shuts an idle worker
down and the next job starts on the new version.
"""

import json
import logging
import subprocess
import sys
import threading
import time

logger = logging.getLogger("dw")

DIFFUSERS_GIT_URL = "git+https://github.com/huggingface/diffusers"
PIP_TIMEOUT_SECONDS = 900
LOG_TAIL_LINES = 30


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
        }

    @staticmethod
    def _run_pip():
        return subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", DIFFUSERS_GIT_URL],
            capture_output=True,
            text=True,
            timeout=PIP_TIMEOUT_SECONDS,
        )

    def start(self, on_success=None):
        """Begin the upgrade; returns the status dict. Raises ValueError if
        one is already running. on_success runs in the worker thread after
        pip exits cleanly."""
        with self._lock:
            if self._state["status"] == "running":
                raise ValueError("A diffusers update is already running")
            self._state = {
                "status": "running",
                "error": None,
                "log": None,
                "started_at": time.time(),
                "finished_at": None,
            }
        thread = threading.Thread(target=self._run, args=(on_success,), daemon=True)
        thread.start()
        return self.status()

    def _run(self, on_success):
        try:
            completed = self._run_fn()
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
                logger.info("diffusers upgraded from git HEAD")
                if on_success:
                    on_success()
            else:
                logger.error(f"diffusers upgrade failed: {tail}")
        except Exception as e:  # pip timeout, missing interpreter, ...
            with self._lock:
                self._state["status"] = "failed"
                self._state["error"] = str(e)
                self._state["finished_at"] = time.time()
            logger.error(f"diffusers upgrade failed: {e}")

    def status(self):
        """The update's state plus what is installed right now."""
        with self._lock:
            state = dict(self._state)
        state.update(diffusers_install_info())
        return state
