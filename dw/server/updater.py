"""Upgrade diffusers to GitHub HEAD (or a pinned commit) from inside the
running server, or revert to the last known-good published release.

The newest model pipelines often land in diffusers git before a release,
so the System page offers the same install install.sh performs. pip runs
in a background thread against this venv with a fixed argument list built
from validated inputs - no request data reaches the command line unchecked.
The persistent worker keeps the old import until it restarts, so on success
the caller shuts an idle worker down and the next job starts on the new
version.
"""

import json
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from ..security import sanitize_command_args, validate_url

logger = logging.getLogger("dw")

DIFFUSERS_GIT_URL = "git+https://github.com/huggingface/diffusers"
DIFFUSERS_ARCHIVE_URL = "https://github.com/huggingface/diffusers/archive/{sha}.zip"
DIFFUSERS_MAIN_COMMIT_URL = (
    "https://api.github.com/repos/huggingface/diffusers/commits/main"
)
MAIN_COMMIT_TIMEOUT_SECONDS = 10
_ARCHIVE_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_ARCHIVE_URL_PATTERN = re.compile(
    r"^https://github\.com/huggingface/diffusers/archive/([0-9a-f]{40})\.zip$"
)
PYPI_PACKAGE = "diffusers"
PIP_TIMEOUT_SECONDS = 900
LOG_TAIL_LINES = 30

# Used only if pyproject.toml can't be read/parsed at runtime (e.g. running
# from an installed wheel without the source tree alongside it).
FALLBACK_RELEASE_FLOOR = "0.43.0"

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
            data = json.loads(direct_url)
            info["commit"] = data.get("vcs_info", {}).get("commit_id")
            if not info["commit"]:
                match = _ARCHIVE_URL_PATTERN.match(str(data.get("url", "")))
                if match:
                    info["commit"] = match.group(1)
    except Exception:
        pass
    return info


def main_commit(fetch=None):
    """The commit sha at the tip of diffusers main, from the GitHub API.
    `fetch(url, timeout)` returns the parsed JSON; tests inject it. Raises
    RuntimeError when it cannot be read, which the updater reports as the
    failed update's error."""
    url = validate_url(DIFFUSERS_MAIN_COMMIT_URL)
    try:
        if fetch is None:
            import httpx

            response = httpx.get(
                url,
                timeout=MAIN_COMMIT_TIMEOUT_SECONDS,
                headers={"Accept": "application/vnd.github+json"},
                follow_redirects=True,
            )
            response.raise_for_status()
            data = response.json()
        else:
            data = fetch(url, MAIN_COMMIT_TIMEOUT_SECONDS)
        sha = data["sha"]
    except Exception as e:
        raise RuntimeError(
            f"Could not read the diffusers main commit from GitHub: {e}"
        ) from e
    if not isinstance(sha, str) or not _ARCHIVE_SHA_PATTERN.match(sha):
        raise RuntimeError(
            f"GitHub returned an unusable commit for diffusers main: {sha!r}"
        )
    return sha


def _probe(args):
    """Whether `args` runs and exits 0, quietly and briefly."""
    try:
        result = subprocess.run(
            sanitize_command_args(args), capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return False
    return result.returncode == 0


def git_usable(platform=None):
    """Whether pip can clone with git here.

    An installed copy (one with an install record) always uses the archive:
    what it installs must not depend on whether the person happens to have
    git. Otherwise git counts only when `git --version` works. On macOS
    /usr/bin/git is the xcrun stub on every Mac: without the Command Line
    Tools it fails, and running it pops the Tools installer, so the stub is
    tried only once `xcode-select -p` reports the Tools are there.
    """
    from ..setup_state import read_install_record

    try:
        if read_install_record() is not None:
            return False
    except Exception:
        return False
    git = shutil.which("git")
    if git is None:
        return False
    if (platform or sys.platform) == "darwin" and git == "/usr/bin/git":
        if not _probe(["xcode-select", "-p"]):
            return False
    return _probe([git, "--version"])


def run_commands(commands, timeout=PIP_TIMEOUT_SECONDS):
    """Run each command without a shell, stopping at the first failure; the
    result carries the last command's return code and every command's
    output."""
    stdout, stderr = [], []
    completed = None
    for command in commands:
        try:
            completed = subprocess.run(
                sanitize_command_args(command),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except Exception as e:  # a timeout, a missing interpreter
            # How many finished before this one: the caller may need to know
            # that an earlier step landed
            e.ran = len(stdout)
            raise
        stdout.append(completed.stdout or "")
        stderr.append(completed.stderr or "")
        if completed.returncode != 0:
            break
    return SimpleNamespace(
        returncode=completed.returncode if completed is not None else 0,
        stdout="\n".join(stdout),
        stderr="\n".join(stderr),
        # How many commands started: more than one means earlier ones landed
        ran=len(stdout),
    )


def build_pip_commands(commit=None, revert=False, python=None, git=None):
    """The pip commands, run in order, for one of three installs: HEAD, a
    pinned commit, or a revert to the pyproject.toml release floor.
    `commit` is validated by the caller before it reaches here - this only
    assembles the already-validated pieces, never a raw request value.
    `python` is the interpreter whose environment they install into - this
    one by default; the container entrypoint (docker/entrypoint.py) passes
    its diffusers overlay venv's.

    A git install is two commands. diffusers main keeps one dev version
    string for a whole release cycle, so `install --upgrade <git url>` saw a
    newer commit as already satisfied and kept the old one (#663); the git
    build is forced in instead, without its dependencies (forcing those
    would reinstall torch). pip builds the wheel before it uninstalls, so a
    failed build leaves the old install in place. The second command, a
    plain install of the package name, then adds any dependency the new
    commit requires - no URL, so no second clone, and no --upgrade, so the
    git build stays.

    Without git (`git_usable`, or `git=False`) the target is the GitHub
    archive of one full commit; with no `commit` that branch reads main's
    sha from the GitHub API, so building the commands can reach the network.
    """
    pip = [python or sys.executable, "-m", "pip", "install"]
    if revert:
        return [pip + [f"{PYPI_PACKAGE}=={release_floor()}"]]
    if git is None:
        git = git_usable()
    if git:
        target = DIFFUSERS_GIT_URL if not commit else f"{DIFFUSERS_GIT_URL}@{commit}"
    else:
        # No git on this machine (an installed copy): pip installs the
        # GitHub archive of one full commit instead of cloning.
        sha = commit or main_commit()
        if not _ARCHIVE_SHA_PATTERN.match(sha):
            raise ValueError("Without git the commit must be a full 40-character hash")
        target = DIFFUSERS_ARCHIVE_URL.format(sha=sha)
    return [
        pip + ["--force-reinstall", "--no-deps", target],
        pip + [PYPI_PACKAGE],
    ]


class DiffusersUpdater:
    """One upgrade at a time, in a background thread, with a status dict
    the UI polls."""

    def __init__(self, run_fn=None):
        # Injectable for tests - the default runs pip against this venv
        self._run_fn = run_fn or self._run_pip
        # The server's `Changes`, set by create_app; a finished update bumps it
        self.changes = None
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
        """Run the pip commands in turn (`run_commands`)."""
        return run_commands(build_pip_commands(commit=commit, revert=revert))

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
        self._changed()

    def _changed(self):
        """The update finished, either way: the install status moved. Never
        raises: a counter is a hint."""
        if self.changes is None:
            return
        try:
            self.changes.bump_all("models")
        except Exception as e:
            logger.warning(f"Could not bump change counters for an update: {e}")

    def status(self):
        """The update's state plus what is installed right now."""
        with self._lock:
            state = dict(self._state)
        state.update(diffusers_install_info())
        return state
