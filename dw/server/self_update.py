"""Update dw itself from the Models page, on an installed copy.

An install made by the bootstrap scripts (INSTALL.md) runs `dw-serve --managed`
under a service manager that restarts it. Only that server may update itself:
pip upgrades `diffusers-workflow[server,mcp]` in this venv with the installed
torch pinned by a constraints file (an unpinned upgrade could swap a CUDA build
for another), diffusers moves to main as the Update from GitHub button does,
`dw setup` refreshes the registrations (best effort: the packages are in
place whether or not it succeeds), and the server then exits with
RESTART_EXIT_CODE through its normal graceful shutdown so the workers stop as
they do on Ctrl-C. launchd KeepAlive and systemd Restart=always restart on any
exit; Windows Task Scheduler restarts only on a non-zero one, hence 75.

An unmanaged server (a checkout, a container) is only shown INSTALL_COMMANDS'
one-liner: it has nobody to restart it, and pip into a developer's venv is not
this button's business.
"""

import logging
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from ..security import validate_url
from ..setup_state import beside_python, read_install_record
from .updater import LOG_TAIL_LINES, build_pip_commands, run_commands

logger = logging.getLogger("dw")

RESTART_EXIT_CODE = 75
# Long enough for the HTTP response that reports success to be written first
EXIT_DELAY_SECONDS = 1.0

DW_PACKAGE = "diffusers-workflow[server,mcp]"
PYPI_JSON_URL = "https://pypi.org/pypi/diffusers-workflow/json"
LATEST_TIMEOUT_SECONDS = 3
LATEST_CACHE_SECONDS = 600

# The one place the bootstrap one-liners live (docs/INSTALL.md repeats them)
BOOTSTRAP_SH_URL = "https://github.com/dkackman/diffusers-workflow/releases/latest/download/bootstrap.sh"
BOOTSTRAP_PS1_URL = "https://github.com/dkackman/diffusers-workflow/releases/latest/download/bootstrap.ps1"
INSTALL_COMMAND_POSIX = f"curl -LsSf {BOOTSTRAP_SH_URL} | bash"
INSTALL_COMMAND_WINDOWS = (
    f'powershell -ExecutionPolicy ByPass -c "irm {BOOTSTRAP_PS1_URL} | iex"'
)


def install_command(platform=None):
    """The one-liner that installs or updates dw on this OS."""
    if (platform or sys.platform) == "win32":
        return INSTALL_COMMAND_WINDOWS
    return INSTALL_COMMAND_POSIX


_latest_cache = {"at": None, "value": None}
_latest_lock = threading.Lock()


def _fetch_pypi(url, timeout):
    """GET `url` and return its parsed JSON (the default `fetch`)."""
    import httpx

    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return response.json()


def latest_version(fetch=None, clock=time.monotonic):
    """The newest dw release on PyPI, or None when it can't be read. Cached
    for ten minutes, failures included, so an offline server isn't asked on
    every poll. `fetch(url, timeout)` returns the parsed JSON; tests inject
    it, or patch `_fetch_pypi`, and nothing here reaches the network
    otherwise.

    The lock guards the cache read and write only, never the fetch: a read
    that finds the cache warm answers at once even while another read waits
    on a slow PyPI (#1743). Two cold reads may both fetch; the later write
    wins, which is harmless."""
    with _latest_lock:
        at = _latest_cache["at"]
        if at is not None and clock() - at < LATEST_CACHE_SECONDS:
            return _latest_cache["value"]
    value = None
    try:
        url = validate_url(PYPI_JSON_URL)
        data = (fetch or _fetch_pypi)(url, LATEST_TIMEOUT_SECONDS)
        version = data["info"]["version"]
        if isinstance(version, str) and version:
            value = version
    except Exception as e:
        logger.debug(f"Could not read the latest dw release: {e}")
    with _latest_lock:
        _latest_cache.update(at=clock(), value=value)
    return value


def forget_latest():
    """Drop the cached release (tests)."""
    with _latest_lock:
        _latest_cache.update(at=None, value=None)


def installed_torch_version():
    """The installed torch's version, local part included, or None."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("torch")
    except PackageNotFoundError:
        return None


def torch_index_from_record():
    """The torch index the install used, or None for PyPI or no record."""
    try:
        record = read_install_record()
    except Exception:
        return None
    index = (record or {}).get("torch_index")
    if not index or index == "pypi":
        return None
    return index


def build_upgrade_commands(
    python, torch_version, torch_index, constraints_dir=None, git=None
):
    """The commands, run in order, that update this install: dw with torch
    held where it is, diffusers to main, then the registrations refreshed.
    `torch_index` is the extra index torch came from (None or "pypi" for
    PyPI). The constraints file lands in `constraints_dir` (a fresh temp
    directory by default)."""
    command = [python, "-m", "pip", "install", "-U", DW_PACKAGE]
    if torch_version:
        directory = Path(constraints_dir or tempfile.mkdtemp(prefix="dw-update-"))
        constraints = directory / "constraints.txt"
        constraints.write_text(f"torch=={torch_version}\n", encoding="utf-8")
        command += ["-c", str(constraints)]
    if torch_index and torch_index != "pypi":
        command += ["--extra-index-url", torch_index]
    return [
        command,
        *build_pip_commands(python=python, git=git),
        [str(beside_python("dw")), "setup", "--yes", "--no-open", "--no-service"],
    ]


def _run_commands():
    """Run the pip steps, stopping at the first failure, then the
    registration refresh. The refresh is best effort: once pip has put the
    new packages in place the server has to restart to run them, so a
    failed `dw setup` is logged (its output joins the log) and the run
    still succeeds."""
    # The constraints file only has to outlive the pip steps
    with tempfile.TemporaryDirectory(prefix="dw-update-") as constraints_dir:
        *installs, refresh = build_upgrade_commands(
            sys.executable,
            installed_torch_version(),
            torch_index_from_record(),
            constraints_dir,
        )
        result = run_commands(installs)
    if result.returncode != 0:
        return result
    try:
        refreshed = run_commands([refresh])
        failure = (
            f"exited with code {refreshed.returncode}" if refreshed.returncode else None
        )
    except Exception as e:  # a missing dw script, a timeout
        refreshed, failure = SimpleNamespace(stdout="", stderr=""), str(e)
    stderr = [result.stderr, refreshed.stderr]
    if failure:
        message = (
            f"the registration refresh (`dw setup`) {failure}; the update is "
            "installed - run `dw setup` again after the restart"
        )
        logger.warning(f"dw update: {message}")
        stderr.append(message)
    return SimpleNamespace(
        returncode=0,
        stdout="\n".join([result.stdout, refreshed.stdout]),
        stderr="\n".join(stderr),
    )


_restart = {"code": None}


def restart_requested():
    """The exit code a requested restart asked for, or None."""
    return _restart["code"]


def request_restart(code=RESTART_EXIT_CODE):
    """Ask this process to shut down gracefully and exit with `code`: the
    same SIGINT path as Ctrl-C, so uvicorn runs its lifespan and the workers
    are stopped. dw.serve.run turns the KeyboardInterrupt that ends
    uvicorn.run into the exit code."""
    _restart["code"] = code
    signal.raise_signal(signal.SIGINT)


class DwUpdater:
    """One self-update at a time, in a background thread, with a status
    dict the UI polls."""

    def __init__(
        self, run_fn=None, exit_fn=None, delay=EXIT_DELAY_SECONDS, is_busy=None
    ):
        # Injectable for tests: the default runs pip, the default exit
        # restarts this process
        self._run_fn = run_fn or _run_commands
        self._exit_fn = exit_fn or request_restart
        self._delay = delay
        # True while a job runs or is queued; the app wires the job manager's
        self.is_busy = is_busy or (lambda: False)
        self._lock = threading.Lock()
        self._thread = None
        self._state = {"status": "idle", "error": None, "log": None}

    def start(self, on_success=None):
        """Begin the update; returns the status. Raises ValueError if one is
        already running. on_success runs in the thread after the commands
        exit cleanly, just before the exit is requested."""
        with self._lock:
            if self._state["status"] == "running":
                raise ValueError("A dw update is already running")
            self._state = {"status": "running", "error": None, "log": None}
            self._thread = threading.Thread(
                target=self._run, args=(on_success,), daemon=True
            )
            self._thread.start()
        return self.status()

    def join(self, timeout=None):
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _run(self, on_success):
        ok = False
        try:
            completed = self._run_fn()
            # A run function that has nothing to report succeeded
            ok = completed is None or completed.returncode == 0
            output = ""
            if completed is not None:
                output = (completed.stdout or "") + "\n" + (completed.stderr or "")
            tail = "\n".join(output.strip().splitlines()[-LOG_TAIL_LINES:])
            with self._lock:
                self._state = {
                    "status": "succeeded" if ok else "failed",
                    "error": None
                    if ok
                    else f"the update exited with code {completed.returncode}",
                    "log": tail,
                }
            if not ok:
                logger.error(f"dw update failed: {tail}")
        except Exception as e:  # pip timeout, missing interpreter, ...
            with self._lock:
                self._state = {"status": "failed", "error": str(e), "log": None}
            logger.error(f"dw update failed: {e}")
        if ok:
            if on_success:
                on_success()
            # After the response that reports success has been written
            time.sleep(self._delay)
            if self.is_busy():
                # The packages are installed; only the restart is withheld,
                # so it cannot kill the job
                error = "a job is running; restart dw-serve when it finishes"
                with self._lock:
                    self._state = {**self._state, "status": "failed", "error": error}
                logger.error(f"dw update: {error}")
                return
            logger.info("dw updated; exiting so the service manager restarts it")
            self._exit_fn(RESTART_EXIT_CODE)

    def status(self):
        with self._lock:
            return dict(self._state)
