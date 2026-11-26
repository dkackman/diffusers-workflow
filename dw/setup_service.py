"""The per-user `dw-serve` service `dw setup` installs: launchd, systemd or Task Scheduler.

The service starts `dw-serve` at login on loopback with the MCP endpoint and
`--managed`, and the service manager restarts it when it exits (an update exits
with code 75 to ask for that). Every command goes through a `Runner`, so the
default one passes its arguments through `sanitize_command_args()` and never
uses a shell, and a test hands in a recording fake. Setup removes only a service
its manifest records, and refuses to overwrite a definition it did not write.
"""

import getpass
import os
import plistlib
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from xml.sax.saxutils import escape

from dw.security import sanitize_command_args
from dw.setup_state import Manifest, SetupError, venv_bin

Runner = Callable[[list[str]], subprocess.CompletedProcess]

SERVICE_LABEL = "com.github.dkackman.dw-serve"  # launchd
_UNIT_NAME = "dw-serve"  # systemd unit and Windows task name
_TIMEOUT = 120


def default_runner(args: list[str]) -> subprocess.CompletedProcess:
    """Run `args` without a shell, capturing output; a missing program is a SetupError."""
    try:
        return subprocess.run(
            sanitize_command_args(args),
            capture_output=True,
            text=True,
            check=False,
            timeout=_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SetupError(f"could not run {args[0]}: {e}") from e


def _name(platform: str) -> str:
    return SERVICE_LABEL if platform == "darwin" else _UNIT_NAME


def serve_command(port: int = 8765) -> list[str]:
    return [
        str(venv_bin("dw-serve")),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--mcp",
        "--managed",
    ]


def _uid() -> int:
    return os.getuid() if hasattr(os, "getuid") else 0


def _launch_agents_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def _plist_path() -> Path:
    return _launch_agents_dir() / f"{SERVICE_LABEL}.plist"


def _log_dir() -> Path:
    return Path.home() / "Library" / "Logs" / "diffusers-workflow"


def _unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / f"{_UNIT_NAME}.service"


def launchd_plist(command: list[str], log_dir: Path) -> bytes:
    return plistlib.dumps(
        {
            "Label": SERVICE_LABEL,
            "ProgramArguments": list(command),
            "RunAtLoad": True,
            "KeepAlive": True,
            "StandardOutPath": str(log_dir / "dw-serve.log"),
            "StandardErrorPath": str(log_dir / "dw-serve.err.log"),
        }
    )


def _systemd_word(word: str) -> str:
    word = word.replace("%", "%%")
    if any(c.isspace() for c in word) or '"' in word:
        return '"' + word.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return word


def systemd_unit(command: list[str]) -> str:
    exec_start = " ".join(_systemd_word(w) for w in command)
    return (
        "[Unit]\n"
        "Description=diffusers-workflow server (dw-serve)\n"
        "After=network.target\n"
        "\n"
        "[Service]\n"
        f"ExecStart={exec_start}\n"
        "Restart=always\n"
        "RestartSec=5\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def windows_task_xml(command: list[str]) -> str:
    user = os.environ.get("USERNAME") or getpass.getuser()
    domain = os.environ.get("USERDOMAIN")
    account = escape(f"{domain}\\{user}" if domain else user)
    arguments = escape(subprocess.list2cmdline(command[1:]))
    # RestartOnFailure treats a non-zero exit (75 is the update's restart request)
    # as a failure, so the task is started again up to 3 times, a minute apart
    return (
        '<?xml version="1.0" encoding="UTF-16"?>\n'
        '<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">\n'
        "  <Triggers>\n"
        f"    <LogonTrigger><Enabled>true</Enabled><UserId>{account}</UserId></LogonTrigger>\n"
        "  </Triggers>\n"
        "  <Principals>\n"
        f'    <Principal id="Author"><UserId>{account}</UserId>'
        "<LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal>\n"
        "  </Principals>\n"
        "  <Settings>\n"
        "    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>\n"
        "    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>\n"
        "    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>\n"
        "    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>\n"
        "    <RestartOnFailure><Interval>PT1M</Interval><Count>3</Count></RestartOnFailure>\n"
        "  </Settings>\n"
        '  <Actions Context="Author">\n'
        f"    <Exec><Command>{escape(command[0])}</Command>"
        f"<Arguments>{arguments}</Arguments></Exec>\n"
        "  </Actions>\n"
        "</Task>\n"
    )


def _check_platform(platform: str) -> None:
    if platform not in ("darwin", "linux", "win32"):
        raise SetupError(f"no per-user service support for platform {platform!r}")


def _run(runner: Runner, args: list[str], what: str) -> subprocess.CompletedProcess:
    result = runner(args)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise SetupError(f"{what} failed (exit {result.returncode}): {detail}")
    return result


def _write_ours(path: Path, content: bytes, recorded: bool) -> None:
    """Write our definition; refuse to replace one setup did not write."""
    if path.exists() and not recorded and path.read_bytes() != content:
        raise SetupError(
            f"{path} already exists and setup did not create it; "
            "remove it or rename it, then run setup again"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


BOOTSTRAP_ATTEMPTS = 5
BOOTSTRAP_DELAY = 1.0


def install_service(
    manifest: Manifest,
    runner: Runner,
    platform: str = sys.platform,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    _check_platform(platform)
    name = _name(platform)
    recorded = manifest.has("service", name)
    command = serve_command()
    if platform == "darwin":
        path = _plist_path()
        _log_dir().mkdir(parents=True, exist_ok=True)
        _write_ours(path, launchd_plist(command, _log_dir()), recorded)
        domain = f"gui/{_uid()}"
        runner(
            ["launchctl", "bootout", f"{domain}/{SERVICE_LABEL}"]
        )  # may not be loaded
        # bootout is asynchronous: bootstrap can fail (exit 5) while the old
        # job is still tearing down, so retry a few times.
        for attempt in range(BOOTSTRAP_ATTEMPTS):
            try:
                _run(
                    runner,
                    ["launchctl", "bootstrap", domain, str(path)],
                    "launchctl bootstrap",
                )
                break
            except SetupError:
                if attempt == BOOTSTRAP_ATTEMPTS - 1:
                    raise
                sleep(BOOTSTRAP_DELAY)
        manifest.record("service", name, {"platform": platform, "path": str(path)})
        manifest.save()
        return f"started the launchd agent {SERVICE_LABEL} ({path})"
    if platform == "linux":
        path = _unit_path()
        _write_ours(path, systemd_unit(command).encode("utf-8"), recorded)
        manifest.record("service", name, {"platform": platform, "path": str(path)})
        manifest.save()
        _run(
            runner, ["systemctl", "--user", "daemon-reload"], "systemctl daemon-reload"
        )
        # decide from systemd, not the manifest: a failed enable leaves the
        # service recorded but never enabled
        enabled = runner(["systemctl", "--user", "is-enabled", name]).returncode == 0
        if enabled:
            _run(runner, ["systemctl", "--user", "restart", name], "systemctl restart")
            return f"restarted the systemd user unit {name} ({path})"
        _run(
            runner, ["systemctl", "--user", "enable", "--now", name], "systemctl enable"
        )
        return f"enabled and started the systemd user unit {name} ({path})"
    if not recorded and runner(["schtasks", "/Query", "/TN", name]).returncode == 0:
        raise SetupError(
            f"a scheduled task named {name} already exists and setup did not create it"
        )
    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "dw-serve-task.xml"
        xml.write_bytes(windows_task_xml(command).encode("utf-16"))
        _run(
            runner,
            ["schtasks", "/Create", "/TN", name, "/XML", str(xml), "/F"],
            "schtasks /Create",
        )
    manifest.record("service", name, {"platform": platform})
    manifest.save()
    _run(runner, ["schtasks", "/Run", "/TN", name], "schtasks /Run")
    return f"created and started the scheduled task {name}"


def uninstall_service(
    manifest: Manifest, runner: Runner, platform: str = sys.platform
) -> str:
    _check_platform(platform)
    name = _name(platform)
    if not manifest.has("service", name):
        return "nothing to remove: setup did not install a service"
    if platform == "darwin":
        runner(["launchctl", "bootout", f"gui/{_uid()}/{SERVICE_LABEL}"])
        _plist_path().unlink(missing_ok=True)
        message = f"removed the launchd agent {SERVICE_LABEL}"
    elif platform == "linux":
        runner(["systemctl", "--user", "disable", "--now", name])
        _unit_path().unlink(missing_ok=True)
        runner(["systemctl", "--user", "daemon-reload"])
        message = f"removed the systemd user unit {name}"
    else:
        runner(["schtasks", "/End", "/TN", name])
        runner(["schtasks", "/Delete", "/TN", name, "/F"])
        message = f"removed the scheduled task {name}"
    manifest.forget("service", name)
    manifest.save()
    return message


def linger_enabled(runner: Runner) -> bool:
    result = runner(["loginctl", "show-user", getpass.getuser(), "-p", "Linger"])
    return result.returncode == 0 and "Linger=yes" in (result.stdout or "")


def enable_linger(runner: Runner, use_sudo: bool) -> bool:
    command = ["loginctl", "enable-linger", getpass.getuser()]
    return runner((["sudo"] if use_sudo else []) + command).returncode == 0


def service_status(runner: Runner, platform: str = sys.platform) -> dict:
    """`stops_at_logout` is None when it cannot be told (loginctl missing)."""
    _check_platform(platform)
    name = _name(platform)
    if platform == "darwin":
        result = runner(["launchctl", "print", f"gui/{_uid()}/{SERVICE_LABEL}"])
        installed = result.returncode == 0
        running = installed and "state = running" in (result.stdout or "")
        return {"installed": installed, "running": running, "stops_at_logout": True}
    if platform == "linux":
        enabled = runner(["systemctl", "--user", "is-enabled", name])
        installed = enabled.returncode == 0 or _unit_path().exists()
        running = runner(["systemctl", "--user", "is-active", name]).returncode == 0
        try:
            stops: bool | None = not linger_enabled(runner)
        except SetupError:
            stops = None
        return {"installed": installed, "running": running, "stops_at_logout": stops}
    result = runner(["schtasks", "/Query", "/TN", name, "/FO", "LIST", "/V"])
    installed = result.returncode == 0
    running = installed and "Running" in (result.stdout or "")
    return {"installed": installed, "running": running, "stops_at_logout": True}


def restart_service(runner: Runner, platform: str = sys.platform) -> None:
    _check_platform(platform)
    name = _name(platform)
    if platform == "darwin":
        _run(
            runner,
            ["launchctl", "kickstart", "-k", f"gui/{_uid()}/{SERVICE_LABEL}"],
            "launchctl kickstart",
        )
    elif platform == "linux":
        _run(runner, ["systemctl", "--user", "restart", name], "systemctl restart")
    else:
        runner(["schtasks", "/End", "/TN", name])
        _run(runner, ["schtasks", "/Run", "/TN", name], "schtasks /Run")
