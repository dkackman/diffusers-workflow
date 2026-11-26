"""`dw setup` and `dw agent`: the command that ties the installer together.

`dw setup` reads the install record, leaves an already-running server alone,
installs the per-user service, registers Claude Desktop and Claude Code, and can
report (`--status`) or undo (`--uninstall`) what it did. It changes only what
its manifest records. `dw agent` starts Claude Code with the dw-notes channel.
Every OS and Claude call goes through a collaborator module, so a test fakes
them; prompts read stdin only on a terminal and never under `--yes`.
"""

import argparse
import getpass
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import webbrowser

import dw
from dw import setup_clients, setup_service
from dw.setup_state import Manifest, SetupError, install_root, read_install_record

DEFAULT_PORT = 8765
_PROMPT_DONE = "Start a new Claude session to pick up the registration."
_NO_CLAUDE = "Claude Code not found"


def _platform() -> str:
    return sys.platform


def _interactive() -> bool:
    return sys.stdin.isatty()


def _input(prompt: str) -> str:
    return input(prompt)


def _open_browser(url: str) -> None:
    webbrowser.open(url)


def _hf_token() -> str | None:
    from huggingface_hub import get_token

    return get_token()


def _hf_login() -> bool:
    return subprocess.run(["hf", "auth", "login"], check=False).returncode == 0


def _url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def server_answering(port: int) -> bool:
    """True when something answers `GET /api/health` on loopback within 2 seconds."""
    try:
        with urllib.request.urlopen(f"{_url(port)}/api/health", timeout=2) as response:  # noqa: S310
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError, ValueError):
        return False


def server_busy(port: int) -> bool | None:
    """Whether the server is running or holding a job (`current_job` or
    `queued` in `GET /api/health`); None when that cannot be read."""
    try:
        with urllib.request.urlopen(f"{_url(port)}/api/health", timeout=2) as response:  # noqa: S310
            health = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, ValueError):
        return None
    if not isinstance(health, dict):
        return None
    return bool(health.get("current_job")) or bool(health.get("queued"))


def _our_service_installed(manifest: Manifest, runner, platform: str) -> bool:
    """Whether the answering server is setup's own service: the manifest
    records one and the service manager reports it installed. A re-run of
    the bootstrap (the upgrade) then rewrites it and restarts it."""
    if not manifest.entries("service"):
        return False
    try:
        return bool(setup_service.service_status(runner, platform)["installed"])
    except SetupError:
        return False


def _restart_allowed(args) -> bool:
    if not server_busy(args.port):
        return True
    print(
        "Warning: the dw server is running a job or has jobs queued; "
        "restarting the service stops them."
    )
    return _confirm("Restart it now?", args.yes, default=False)


def agent_command(extra: list[str]) -> list[str]:
    return [
        "claude",
        "--dangerously-load-development-channels",
        "server:dw-notes",
        *extra,
    ]


def _confirm(question: str, yes: bool, default: bool) -> bool:
    if yes:
        return True
    if not _interactive():
        return default
    answer = _input(f"{question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    return default if not answer else answer.startswith("y")


def _plugin_version(runner) -> str | None:
    out = runner(["claude", "plugin", "list"]).stdout or ""
    block = re.search(
        re.escape(setup_clients.PLUGIN) + r"(.*?)(?:\n\s*[❯>]\s|\Z)", out, re.DOTALL
    )
    found = re.search(r"[Vv]ersion:?\s*v?(\S+)", block.group(1)) if block else None
    return found.group(1) if found else None


def _desktop_state(manifest: Manifest, platform: str) -> dict:
    registered = False
    path = setup_clients.desktop_config_path(platform)
    if path is not None and path.exists():
        try:
            servers = json.loads(path.read_text(encoding="utf-8")).get("mcpServers")
            registered = isinstance(servers, dict) and "dw" in servers
        except (OSError, ValueError, AttributeError):
            registered = False
    return {
        "present": setup_clients.desktop_present(platform),
        "registered": registered,
        "ours": manifest.has("desktop", "dw"),
    }


def _code_state(runner, manifest: Manifest) -> dict:
    cli = setup_clients.claude_cli()
    state = {
        "cli": cli,
        "plugin": False,
        "plugin_version": None,
        "notes_channel": False,
        "duplicates": [],
    }
    if cli is None:
        return state
    listing = runner(["claude", "plugin", "list"])
    state["plugin"] = setup_clients.PLUGIN in (listing.stdout or "")
    if state["plugin"]:
        state["plugin_version"] = _plugin_version(runner)
    servers = setup_clients._mcp_servers(runner) or {}
    state["notes_channel"] = "dw-notes" in servers
    ours = {e["key"] for e in manifest.entries("code-mcp")}
    state["duplicates"] = [
        n for n in setup_clients.code_duplicates(runner) if n not in ours
    ]
    return state


def status_report(runner, platform: str, port: int = DEFAULT_PORT) -> dict:
    manifest = Manifest.load()
    try:
        service = setup_service.service_status(runner, platform)
    except SetupError:
        service = {"installed": False, "running": False, "stops_at_logout": None}
    answering = server_answering(port)
    desktop = _desktop_state(manifest, platform)
    code = _code_state(runner, manifest)
    record = read_install_record()
    version = dw.__version__
    missing = []
    if record is None:
        missing.append("install record")
    if not answering:
        missing.append("server not answering")
    if not service["installed"]:
        missing.append("service")
    if desktop["present"] and not desktop["registered"]:
        missing.append("Claude Desktop registration")
    if code["cli"] is not None:
        if not code["plugin"]:
            missing.append("Claude Code plugin")
        if not code["notes_channel"]:
            missing.append("dw-notes channel")
    return {
        "version": version,
        "install_root": str(install_root()),
        "install_record": record,
        "server": {
            "url": _url(port),
            "answering": answering,
            "managed": answering and bool(service["running"]),
        },
        "service": service,
        "desktop": desktop,
        "code": code,
        "plugin_version_mismatch": code["plugin_version"] is not None
        and setup_clients.versions_differ(code["plugin_version"], version),
        "missing": missing,
    }


def _print_status(report: dict) -> None:
    server, desktop, code = report["server"], report["desktop"], report["code"]
    print(f"dw {report['version']} installed at {report['install_root']}")
    print(f"install record: {report['install_record'] or 'none'}")
    print(
        f"server {server['url']}: "
        f"{'answering' if server['answering'] else 'not answering'}"
        f"{', managed by the service' if server['managed'] else ''}"
    )
    service = report["service"]
    print(
        f"service: {'installed' if service['installed'] else 'not installed'}, "
        f"{'running' if service['running'] else 'not running'}"
    )
    print(
        f"Claude Desktop: {'present' if desktop['present'] else 'absent'}, "
        f"{'registered' if desktop['registered'] else 'not registered'}"
        f"{' (by setup)' if desktop['ours'] else ''}"
    )
    print(
        f"Claude Code: {code['cli'] or _NO_CLAUDE}; plugin "
        f"{code['plugin_version'] or ('installed' if code['plugin'] else 'missing')}; "
        f"notes channel {'present' if code['notes_channel'] else 'missing'}"
    )
    if code["duplicates"]:
        print(f"duplicate dw servers in Claude Code: {', '.join(code['duplicates'])}")
    if report["plugin_version_mismatch"]:
        print(f"the plugin version differs from dw {report['version']}")
    print(f"missing: {', '.join(report['missing']) or 'nothing'}")


def _step(label: str, failures: list[str], action) -> None:
    try:
        print(f"{label}: {action()}")
    except SetupError as e:
        failures.append(label)
        print(f"{label}: failed: {e}")


def _install(args, runner, platform: str) -> int:
    manifest = Manifest.load()
    failures: list[str] = []
    record = read_install_record()
    print(f"dw {dw.__version__}; install record: {record or 'none'}")
    _service(args, manifest, runner, platform, failures)
    if args.no_desktop:
        print("Claude Desktop: skipped (--no-desktop)")
    elif platform == "linux":
        print("Claude Desktop: skipped, no Claude Desktop on Linux")
    else:
        _step(
            "Claude Desktop",
            failures,
            lambda: setup_clients.register_desktop(manifest, platform),
        )
        print(setup_clients.desktop_plugin_steps())
    if args.no_code and args.no_desktop:
        print("dw-mcp shim: skipped (--no-code --no-desktop)")
    else:
        # Claude Code and Cowork (Claude Desktop) both start the plugin's dw
        # server through this path, so it does not wait on the claude CLI
        _step("dw-mcp shim", failures, lambda: setup_clients.write_mcp_shim(manifest))
    if args.no_code:
        print("Claude Code: skipped (--no-code)")
    elif setup_clients.claude_cli() is None:
        print(f"Claude Code: skipped, {_NO_CLAUDE}")
    else:
        _step(
            "Claude Code",
            failures,
            lambda: setup_clients.register_code(manifest, runner, dw.__version__),
        )
        ours = {e["key"] for e in manifest.entries("code-mcp")}
        for name in setup_clients.code_duplicates(runner):
            if name not in ours:
                print(
                    f"Warning: Claude Code already has an MCP server named {name} "
                    "that duplicates the plugin's dw tools; remove it with "
                    f"`claude mcp remove {name}`."
                )
    _hf(args)
    if not args.no_open:
        _open_browser(_url(args.port))
    print(_PROMPT_DONE)
    return 1 if failures else 0


def _service(args, manifest: Manifest, runner, platform: str, failures) -> None:
    """Install the service, or rewrite and restart setup's own one; leave a
    server setup's service did not start alone."""
    answering = server_answering(args.port)
    ours = (
        answering
        and not args.no_service
        and _our_service_installed(manifest, runner, platform)
    )
    if answering and not ours:
        print(
            f"A dw server is already running at {_url(args.port)} that setup's "
            "service did not start; no service installed."
        )
    if args.no_service:
        print("service: skipped (--no-service)")
    elif answering and not ours:
        return
    elif platform not in ("darwin", "linux", "win32"):
        print(f"service: skipped, no service manager for {platform}")
    elif ours and not _restart_allowed(args):
        print(
            "service: not restarted; the running server keeps the old version "
            "until it restarts - run dw setup again when its jobs finish"
        )
    else:
        if ours:
            print(f"The server at {_url(args.port)} is setup's service; restarting it.")
        _step(
            "service",
            failures,
            lambda: setup_service.install_service(manifest, runner, platform),
        )
        if platform == "linux" and "service" not in failures:
            _linger(args, manifest, runner)


def _linger(args, manifest: Manifest, runner) -> None:
    try:
        if setup_service.linger_enabled(runner):
            return
    except SetupError:
        return
    print("The service stops at logout unless lingering is enabled for your user.")
    if not _confirm("Enable lingering now (needs sudo)?", args.yes, default=False):
        print(
            "linger: not enabled; run `sudo loginctl enable-linger $USER` to keep dw running"
        )
        return
    try:
        enabled = setup_service.enable_linger(runner, use_sudo=True)
    except SetupError as e:
        print(f"linger: failed: {e}")
        return
    if enabled:
        manifest.record("linger", getpass.getuser(), {})
        manifest.save()
        print("linger: enabled")
    else:
        print("linger: sudo refused; the service stops at logout")


def _hf(args) -> None:
    if args.yes or not _interactive() or _hf_token():
        return
    if _confirm("No Hugging Face login found. Run `hf auth login` now?", False, True):
        if not _hf_login():
            print("Hugging Face login did not complete; run `hf auth login` later.")


def _uninstall(args, runner, platform: str) -> int:
    manifest = Manifest.load()
    failures: list[str] = []
    if not manifest.entries():
        print("Nothing to remove: setup has recorded no changes.")
        return 0
    if not _confirm("Remove everything dw setup added?", args.yes, default=False):
        print("Nothing removed.")
        return 0
    if any(
        manifest.entries(k) for k in ("code-plugin", "code-marketplace", "code-mcp")
    ):
        if setup_clients.claude_cli() is None:
            print(f"Claude Code: skipped, {_NO_CLAUDE}; its entries stay recorded")
        else:
            _step(
                "Claude Code",
                failures,
                lambda: setup_clients.unregister_code(manifest, runner),
            )
    if manifest.entries("shim"):
        _step("dw-mcp shim", failures, lambda: setup_clients.remove_mcp_shim(manifest))
    if manifest.entries("desktop"):
        _step(
            "Claude Desktop",
            failures,
            lambda: setup_clients.unregister_desktop(manifest, platform),
        )
    if manifest.entries("service"):
        _step(
            "service",
            failures,
            lambda: setup_service.uninstall_service(manifest, runner, platform),
        )
    for entry in manifest.entries("linger"):
        manifest.forget("linger", entry["key"])
        manifest.save()
        print("linger: left enabled; run `sudo loginctl disable-linger $USER` to undo")
    return 1 if failures else 0


def _setup(args) -> int:
    runner = setup_service.default_runner
    platform = _platform()
    try:
        if args.status:
            report = status_report(runner, platform, args.port)
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                _print_status(report)
            return 0
        if args.uninstall:
            return _uninstall(args, runner, platform)
        return _install(args, runner, platform)
    except SetupError as e:
        print(f"dw setup: {e}", file=sys.stderr)
        return 1


def _agent(extra: list[str]) -> int:
    claude = setup_clients.claude_cli()
    if claude is None:
        print(
            "Claude Code was not found on PATH. Install it from "
            "https://claude.com/claude-code, then run `dw agent` again.",
            file=sys.stderr,
        )
        return 1
    command = [claude, *agent_command(extra)[1:]]
    if sys.platform == "win32":
        return subprocess.run(command, check=False).returncode
    os.execvp(command[0], command)
    return 0  # pragma: no cover


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dw")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="install the service and register Claude")
    setup.add_argument("--yes", action="store_true", help="take every default")
    setup.add_argument("--no-service", action="store_true")
    setup.add_argument("--no-desktop", action="store_true")
    setup.add_argument("--no-code", action="store_true")
    setup.add_argument("--no-open", action="store_true", help="do not open the web UI")
    setup.add_argument("--status", action="store_true", help="report, change nothing")
    setup.add_argument("--json", action="store_true", help="with --status, print JSON")
    setup.add_argument("--uninstall", action="store_true")
    setup.add_argument("--port", type=int, default=DEFAULT_PORT)
    sub.add_parser("agent", help="start Claude Code with the dw-notes channel")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["agent"]:  # everything after `agent` belongs to claude
        return _agent(argv[1:])
    return _setup(_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
