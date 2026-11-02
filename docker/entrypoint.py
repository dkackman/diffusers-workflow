"""The container's entrypoint: the diffusers overlay, the API token, then dw.serve.

The image is immutable and carries torch, the dw wheel and every pinned
dependency, plus a *baked* diffusers - a git commit chosen when the image was
built. diffusers is the one package that tracks GitHub main, so the one that
changes lives on a volume: `DW_ENV_DIR/venv`, a venv made with
`--system-site-packages` over the image's interpreter. It sees every image
package, and its own site-packages comes first, so a diffusers installed there
shadows the baked one. pip run from it leaves the image's copy alone and adds
only the dependencies a new commit needs that the image does not already
satisfy. The server runs under the overlay's interpreter, so the Models page's
updater (`dw/server/updater.py`, which pips into `sys.executable`) lands its
installs on the volume too.

`DW_ENV_DIR/overlay.json` records which image the overlay was built against
(the image's build id, a hash of its `pip freeze`) and which diffusers it holds.
A restart never moves diffusers. A new image rebuilds the overlay - an old one
can hold dependency versions that would now shadow newer ones the image ships -
and reinstalls the diffusers the old overlay held, not HEAD. When the overlay
does not import (a broken HEAD, no network on the first start), the server runs
on the image's interpreter and the baked diffusers instead, so it still comes
up and the Models page can pin a working commit.

`DW_DIFFUSERS_REF` picks what a new overlay installs: `head` (the default),
a commit hash (which the overlay is then held to), or `image` (no overlay).

Subcommands: `serve` (the default; anything after it is passed to dw.serve),
`update-diffusers [--commit SHA | --revert]`, `healthcheck`, `record-image`
(run once at image build) and `probe` (run under the overlay's interpreter).
See docs/DOCKER.md.
"""

import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from dw.security import InvalidInputError, sanitize_command_args, validate_commit_hash
from dw.server.updater import build_pip_commands

BASE_PYTHON = sys.executable
IMAGE_INFO = Path(os.environ.get("DW_IMAGE_INFO", "/opt/dw/image.json"))
DEFAULT_ENV_DIR = "/data/env"
EXAMPLES_DIR = "/opt/dw/examples/workflows"
CONTAINER_PORT = 8765
TOKEN_FILE_NAME = "api-token"
# pip builds diffusers from a git clone; a cold one is a few minutes
INSTALL_TIMEOUT_SECONDS = 1800
PROBE_TIMEOUT_SECONDS = 300


def log(message):
    print(f"[dw-container] {message}", file=sys.stderr, flush=True)


def env_dir(environ):
    return Path(environ.get("DW_ENV_DIR") or DEFAULT_ENV_DIR)


def overlay_python(environ):
    return env_dir(environ) / "venv" / "bin" / "python"


def marker_path(environ):
    return env_dir(environ) / "overlay.json"


def parse_ref(value):
    """`DW_DIFFUSERS_REF` as ("head", None), ("image", None) or ("commit", sha)."""
    value = (value or "head").strip().lower()
    if value in ("head", "image"):
        return value, None
    try:
        return "commit", validate_commit_hash(value)
    except InvalidInputError:
        raise SystemExit(
            f"DW_DIFFUSERS_REF={value!r}: expected 'head', 'image' or a commit hash"
        )


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def run_command(command, run=subprocess.run, timeout=INSTALL_TIMEOUT_SECONDS):
    """Run one fixed argument list; True when it exits 0. Output tails are
    logged on failure, not on success - pip's progress is noise in `docker logs`."""
    try:
        completed = run(
            sanitize_command_args(command),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f"{command[0]} failed to run: {e}")
        return False
    if completed.returncode != 0:
        output = ((completed.stdout or "") + (completed.stderr or "")).strip()
        log(f"exit {completed.returncode}: {' '.join(command[:6])}")
        for line in output.splitlines()[-20:]:
            log(f"  {line}")
        return False
    return True


def probe(python, run=subprocess.run):
    """What diffusers `python` imports ({"version", "commit"}), or None when
    dw's server or diffusers fails to import under it."""
    try:
        completed = run(
            sanitize_command_args(
                [str(python), str(Path(__file__).resolve()), "probe"]
            ),
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f"overlay probe failed to run: {e}")
        return None
    if completed.returncode != 0:
        for line in (completed.stderr or "").strip().splitlines()[-10:]:
            log(f"  {line}")
        return None
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def install_target(kind, commit, marker, previous):
    """What a (re)built overlay installs, as build_pip_commands' keywords.

    A pinned ref wins. Otherwise the diffusers the old overlay held, so an image
    upgrade never moves it: its commit, or a revert when it was a release (the
    Models page's revert). Nothing recorded means a new overlay, so HEAD."""
    if kind == "commit":
        return {"commit": commit}
    held = previous or marker or {}
    held_commit = held.get("commit") or held.get("diffusers_commit")
    if held_commit:
        return {"commit": held_commit}
    held_version = held.get("version") or held.get("diffusers_version")
    if held_version and "dev" not in held_version:
        return {"revert": True}
    return {}


def build_overlay(environ, target, run=subprocess.run):
    """A fresh overlay venv with the diffusers `target` names. True on success."""
    venv = env_dir(environ) / "venv"
    if venv.exists():
        shutil.rmtree(venv)
    if not run_command(
        [BASE_PYTHON, "-m", "venv", "--system-site-packages", str(venv)], run
    ):
        return False
    what = "release floor" if target.get("revert") else target.get("commit") or "HEAD"
    log(f"installing diffusers ({what}) into the overlay at {venv}")
    for command in build_pip_commands(python=str(overlay_python(environ)), **target):
        if not run_command(command, run):
            return False
    return True


def prepare_overlay(environ, image, run=subprocess.run):
    """The interpreter dw.serve runs under: the overlay's, or the image's when
    `DW_DIFFUSERS_REF=image` or the overlay will not import."""
    kind, commit = parse_ref(environ.get("DW_DIFFUSERS_REF"))
    if kind == "image":
        log(
            f"DW_DIFFUSERS_REF=image: using the baked diffusers {image.get('diffusers')}"
        )
        return BASE_PYTHON

    env_dir(environ).mkdir(parents=True, exist_ok=True)
    marker = read_json(marker_path(environ)) or {}
    python = overlay_python(environ)
    build_id = image.get("build_id")

    current = None
    if python.exists() and marker.get("build_id") == build_id:
        current = probe(python, run)
        if (
            current
            and kind == "commit"
            and not (current.get("commit") or "").startswith(commit)
        ):
            log(
                f"overlay holds {current.get('commit')}, DW_DIFFUSERS_REF pins {commit}"
            )
            current = None

    if current is None:
        # The old overlay's own answer beats the marker: the Models page may have
        # moved it since the marker was written. It may not run at all once the
        # image's Python changed, so the marker is the fallback.
        previous = probe(python, run) if python.exists() else None
        target = install_target(kind, commit, marker.get("diffusers"), previous)
        reason = (
            "new"
            if not marker
            else "image changed"
            if marker.get("build_id") != build_id
            else "rebuild"
        )
        log(f"building the diffusers overlay ({reason})")
        ok = build_overlay(environ, target, run)
        current = probe(python, run) if ok else None
        if current is None:
            # Keep what it was meant to hold, so the next start retries it
            write_json(
                marker_path(environ),
                {"build_id": None, "diffusers": marker.get("diffusers") or previous},
            )
            log(
                "the overlay did not install or import - serving the baked diffusers "
                f"{image.get('diffusers')}; an update from the Models page now lands in the "
                "container, not the volume, and is lost when it is re-created"
            )
            return BASE_PYTHON

    write_json(
        marker_path(environ),
        {"build_id": build_id, "python": image.get("python"), "diffusers": current},
    )
    log(
        f"diffusers {current.get('version')} ({current.get('commit') or 'release'}) from the overlay"
    )
    return str(python)


def token_path(environ):
    """Beside settings.json, in the directory the settings mount shares."""
    settings = environ.get("DIFFUSERS_HELPER_ROOT") or Path.home() / ".diffusers_helper"
    return Path(settings) / TOKEN_FILE_NAME


def saved_token(environ):
    try:
        return token_path(environ).read_text().strip()
    except OSError:
        return ""


def resolve_token(environ):
    """The API token: DW_API_TOKEN, else the one saved in the settings
    directory, else a new one saved there. None only with DW_INSECURE_NO_TOKEN=1.

    The server binds every interface inside the container and cannot see
    whether the host published its port on localhost only, so it gets a token
    unless told otherwise - Jupyter's pattern, but kept across restarts so an
    MCP client configured with it keeps working."""
    token = (environ.get("DW_API_TOKEN") or "").strip()
    if token:
        return token
    if environ.get("DW_INSECURE_NO_TOKEN") == "1":
        log("DW_INSECURE_NO_TOKEN=1: no API token - publish the port on localhost only")
        return None
    path = token_path(environ)
    token = saved_token(environ)
    if not token:
        path.parent.mkdir(parents=True, exist_ok=True)
        token = secrets.token_hex(32)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(token + "\n")
        log(f"generated an API token and saved it to {path}")
    return token


def serve_command(python, args):
    """dw.serve's argument list: the container's defaults unless `args` names them."""
    command = [python, "-m", "dw.serve"]
    if "--host" not in args:
        command += ["--host", "0.0.0.0"]
    if "--port" not in args:
        command += ["--port", str(CONTAINER_PORT)]
    if "--examples-dir" not in args and Path(EXAMPLES_DIR).is_dir():
        command += ["--examples-dir", EXAMPLES_DIR]
    return command + list(args)


def serve(args, environ=None, run=subprocess.run, execve=os.execve):
    environ = dict(os.environ if environ is None else environ)
    image = read_json(IMAGE_INFO) or {}
    home = Path(environ.get("HOME") or env_dir(environ) / "home")
    home.mkdir(parents=True, exist_ok=True)

    token = resolve_token(environ)
    python = prepare_overlay(environ, image, run)
    if token:
        # The environment, not --token: argv shows in `ps` on the host
        environ["DW_API_TOKEN"] = token
        port = environ.get("DW_HOST_PORT") or CONTAINER_PORT
        log(f"open http://localhost:{port}/?token={token}")
    command = serve_command(python, args)
    log(" ".join(command))
    execve(python, sanitize_command_args(command), environ)


def update_diffusers(args, environ=None, run=subprocess.run):
    """The Models page's update, from `docker exec`: HEAD, a commit, or a revert."""
    parser = argparse.ArgumentParser(prog="update-diffusers")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--commit", help="a commit hash to pin (default: main HEAD)")
    group.add_argument(
        "--revert", action="store_true", help="the pyproject release floor"
    )
    options = parser.parse_args(args)
    environ = dict(os.environ if environ is None else environ)
    python = overlay_python(environ)
    if not python.exists():
        log(f"no overlay at {python} - the server is on the baked diffusers")
        return 1
    commit = validate_commit_hash(options.commit) if options.commit else None
    for command in build_pip_commands(
        commit=commit, revert=options.revert, python=str(python)
    ):
        if not run_command(command, run):
            return 1
    current = probe(python, run)
    if current is None:
        log("the update installed but does not import - pin a commit or --revert")
        return 1
    marker = read_json(marker_path(environ)) or {}
    marker["diffusers"] = current
    write_json(marker_path(environ), marker)
    log(
        f"diffusers {current.get('version')} ({current.get('commit') or 'release'}) installed"
    )
    log("a running job keeps the old import; restart the container to load it")
    return 0


def healthcheck(environ=None):
    environ = os.environ if environ is None else environ
    request = urllib.request.Request(f"http://127.0.0.1:{CONTAINER_PORT}/api/health")
    # Never resolve_token: a healthcheck must not mint one
    token = (environ.get("DW_API_TOKEN") or "").strip() or saved_token(environ)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return 0 if response.status == 200 else 1
    except OSError:
        return 1


def record_image():
    """Write IMAGE_INFO at build time: the build id the overlay compares
    against, and the diffusers baked in."""
    from dw.server.updater import diffusers_install_info

    freeze = subprocess.run(
        sanitize_command_args([BASE_PYTHON, "-m", "pip", "freeze", "--all"]),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    python = ".".join(str(part) for part in sys.version_info[:3])
    build_id = hashlib.sha256((python + "\n" + freeze).encode()).hexdigest()[:16]
    IMAGE_INFO.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        IMAGE_INFO,
        {"build_id": build_id, "python": python, "diffusers": diffusers_install_info()},
    )
    print(IMAGE_INFO.read_text())
    return 0


def probe_main():
    """Run under the overlay's interpreter: import what dw.serve needs, print
    the diffusers it got."""
    import diffusers  # noqa: F401
    import dw.server  # noqa: F401
    from dw.server.updater import diffusers_install_info

    print(json.dumps(diffusers_install_info()))
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    command, rest = (
        (argv[0], argv[1:]) if argv and not argv[0].startswith("-") else ("serve", argv)
    )
    if command == "serve":
        return serve(rest)
    if command == "update-diffusers":
        return update_diffusers(rest)
    if command == "healthcheck":
        return healthcheck()
    if command == "record-image":
        return record_image()
    if command == "probe":
        return probe_main()
    log(f"unknown command {command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
