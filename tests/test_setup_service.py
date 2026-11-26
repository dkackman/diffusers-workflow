"""The per-user service: definitions, install/uninstall through a fake runner."""

import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

from dw import setup_service
from dw.setup_state import Manifest, SetupError


class FakeRunner:
    def __init__(self, returncode=0, stdout=""):
        self.calls = []
        self.returncode = returncode
        self.stdout = stdout
        self.by_command = {}

    def __call__(self, args):
        self.calls.append(list(args))
        rc, out = self.by_command.get(tuple(args[:2]), (self.returncode, self.stdout))
        return subprocess.CompletedProcess(args, rc, out, "boom" if rc else "")


class SystemdRunner(FakeRunner):
    """Answers is-enabled by whether an enable has succeeded."""

    def __init__(self, enable_rc=0):
        super().__init__()
        self.enabled = False
        self.enable_rc = enable_rc

    def __call__(self, args):
        if args[:3] == ["systemctl", "--user", "is-enabled"]:
            self.calls.append(list(args))
            rc = 0 if self.enabled else 1
            return subprocess.CompletedProcess(args, rc, "", "")
        if args[:3] == ["systemctl", "--user", "enable"]:
            self.calls.append(list(args))
            self.enabled = self.enable_rc == 0
            return subprocess.CompletedProcess(args, self.enable_rc, "", "boom")
        return super().__call__(args)


@pytest.fixture
def fake_runner():
    return FakeRunner()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("DW_INSTALL_ROOT", str(tmp_path / "root"))
    return tmp_path


@pytest.fixture
def manifest(home):
    return Manifest.load(home / "manifest.json")


def test_the_plist_keeps_the_server_alive():
    plist = plistlib.loads(
        setup_service.launchd_plist(["/v/bin/dw-serve", "--mcp"], Path("/logs"))
    )
    assert plist["KeepAlive"] is True and plist["RunAtLoad"] is True
    assert plist["ProgramArguments"] == ["/v/bin/dw-serve", "--mcp"]
    assert plist["StandardOutPath"].startswith("/logs")


def test_the_unit_restarts_always():
    unit = setup_service.systemd_unit(["/v/bin/dw-serve", "--mcp"])
    assert "Restart=always" in unit and "ExecStart=/v/bin/dw-serve --mcp" in unit
    assert "RestartSec=5" in unit and "WantedBy=default.target" in unit


def test_the_unit_quotes_a_path_with_spaces():
    unit = setup_service.systemd_unit(["/my dir/dw-serve", "--mcp"])
    assert 'ExecStart="/my dir/dw-serve" --mcp' in unit


def test_the_task_xml_restarts_on_failure_without_a_time_limit():
    xml = setup_service.windows_task_xml([r"C:\v\Scripts\dw-serve.exe", "--mcp"])
    assert "<LogonTrigger>" in xml and "<ExecutionTimeLimit>PT0S<" in xml
    assert "<Interval>PT1M</Interval><Count>3</Count>" in xml
    assert r"<Command>C:\v\Scripts\dw-serve.exe</Command>" in xml


def test_serve_command_binds_loopback_with_mcp_and_managed(home):
    cmd = setup_service.serve_command(platform="linux")
    assert cmd[1:] == ["--host", "127.0.0.1", "--port", "8765", "--mcp", "--managed"]
    assert Path(cmd[0]) == Path(sys.executable).parent / "dw-serve"


def test_the_windows_task_is_told_the_install_root(home):
    cmd = setup_service.serve_command(platform="win32")
    assert cmd[0].endswith("dw-serve.exe")
    assert cmd[-2:] == ["--install-root", str(home / "root")]


def test_the_service_carries_the_install_root_and_works_there(home):
    env = setup_service.service_environment()
    assert env == {"DW_INSTALL_ROOT": str(home / "root")}
    plist = plistlib.loads(
        setup_service.launchd_plist(["/x"], Path("/logs"), home / "root", env)
    )
    assert plist["EnvironmentVariables"] == env
    assert plist["WorkingDirectory"] == str(home / "root")
    unit = setup_service.systemd_unit(["/x"], Path("/my root"), {"A": "b c"})
    assert "WorkingDirectory=/my root\n" in unit and 'Environment="A=b c"' in unit
    xml = setup_service.windows_task_xml(["C:\\x.exe"], Path("C:\\root"))
    assert "<WorkingDirectory>C:\\root</WorkingDirectory>" in xml


def test_without_an_override_the_service_sets_no_environment(home, monkeypatch):
    monkeypatch.delenv("DW_INSTALL_ROOT")
    assert setup_service.service_environment() == {}
    assert "--install-root" not in setup_service.serve_command(platform="win32")
    plist = plistlib.loads(setup_service.launchd_plist(["/x"], Path("/l"), None, {}))
    assert "EnvironmentVariables" not in plist


def test_install_makes_the_working_directory(home, fake_runner, manifest):
    setup_service.install_service(manifest, fake_runner, platform="linux")
    unit = (home / ".config" / "systemd" / "user" / "dw-serve.service").read_text()
    assert f"WorkingDirectory={home / 'root'}" in unit and (home / "root").is_dir()


def test_installing_twice_on_linux_enables_once_and_records_once(home, manifest):
    fake_runner = SystemdRunner()
    setup_service.install_service(manifest, fake_runner, platform="linux")
    setup_service.install_service(manifest, fake_runner, platform="linux")
    enable = ["systemctl", "--user", "enable", "--now", "dw-serve"]
    assert fake_runner.calls.count(enable) == 1
    assert ["systemctl", "--user", "restart", "dw-serve"] in fake_runner.calls
    assert len(manifest.entries("service")) == 1
    unit = home / ".config" / "systemd" / "user" / "dw-serve.service"
    assert "--managed" in unit.read_text()


def test_a_failed_enable_is_retried_not_restarted(home, manifest):
    runner = SystemdRunner(enable_rc=1)
    with pytest.raises(SetupError, match="enable"):
        setup_service.install_service(manifest, runner, platform="linux")
    runner.enable_rc = 0
    setup_service.install_service(manifest, runner, platform="linux")
    enable = ["systemctl", "--user", "enable", "--now", "dw-serve"]
    assert runner.calls.count(enable) == 2
    assert ["systemctl", "--user", "restart", "dw-serve"] not in runner.calls
    assert runner.enabled


def test_a_foreign_unit_is_not_overwritten(home, fake_runner, manifest):
    unit = home / ".config" / "systemd" / "user" / "dw-serve.service"
    unit.parent.mkdir(parents=True)
    unit.write_text("[Service]\nExecStart=/mine\n")
    with pytest.raises(SetupError, match="already exists"):
        setup_service.install_service(manifest, fake_runner, platform="linux")
    assert unit.read_text() == "[Service]\nExecStart=/mine\n"
    assert manifest.entries("service") == []


def test_install_on_macos_writes_the_plist_and_bootstraps(home, fake_runner, manifest):
    setup_service.install_service(manifest, fake_runner, platform="darwin")
    plist = home / "Library" / "LaunchAgents" / "com.github.dkackman.dw-serve.plist"
    assert plistlib.loads(plist.read_bytes())["KeepAlive"] is True
    assert (home / "Library" / "Logs" / "diffusers-workflow").is_dir()
    assert fake_runner.calls[-1][:2] == ["launchctl", "bootstrap"]
    assert manifest.has("service", "com.github.dkackman.dw-serve")


def test_a_failed_bootstrap_is_a_setup_error(home, manifest):
    runner = FakeRunner()
    runner.by_command[("launchctl", "bootstrap")] = (5, "")
    with pytest.raises(SetupError, match="bootstrap"):
        setup_service.install_service(
            manifest, runner, platform="darwin", sleep=lambda _s: None
        )
    # recorded before bootstrap, so a later run may rewrite the plist it left
    assert manifest.has("service", "com.github.dkackman.dw-serve")
    runner.by_command[("launchctl", "bootstrap")] = (0, "")
    setup_service.install_service(manifest, runner, platform="darwin")


def test_bootstrap_is_retried_after_an_async_bootout(home, manifest):
    class Flaky(FakeRunner):
        def __call__(self, args):
            if args[:2] == ["launchctl", "bootstrap"]:
                n = sum(1 for c in self.calls if c[:2] == ["launchctl", "bootstrap"])
                self.by_command[("launchctl", "bootstrap")] = (5 if n == 0 else 0, "")
            return super().__call__(args)

    runner = Flaky()
    sleeps = []
    setup_service.install_service(
        manifest, runner, platform="darwin", sleep=sleeps.append
    )
    boots = [c for c in runner.calls if c[:2] == ["launchctl", "bootstrap"]]
    assert len(boots) == 2
    assert len(sleeps) == 1
    assert manifest.has("service", "com.github.dkackman.dw-serve")


def test_install_on_windows_creates_and_runs_the_task(home, fake_runner, manifest):
    fake_runner.by_command[("schtasks", "/Query")] = (1, "")
    setup_service.install_service(manifest, fake_runner, platform="win32")
    verbs = [c[1] for c in fake_runner.calls if c[0] == "schtasks"]
    assert verbs == ["/Query", "/Create", "/End", "/Run"]
    assert manifest.has("service", "dw-serve")


def test_a_foreign_windows_task_is_not_replaced(home, fake_runner, manifest):
    with pytest.raises(SetupError, match="already exists"):
        setup_service.install_service(manifest, fake_runner, platform="win32")
    assert not any(c[1] == "/Create" for c in fake_runner.calls)


def test_uninstall_removes_only_a_recorded_service(fake_runner, manifest):
    assert "nothing" in setup_service.uninstall_service(
        manifest, fake_runner, platform="darwin"
    )
    assert fake_runner.calls == []


def test_uninstall_removes_the_unit_and_the_record(home, fake_runner, manifest):
    setup_service.install_service(manifest, fake_runner, platform="linux")
    setup_service.uninstall_service(manifest, fake_runner, platform="linux")
    assert not (home / ".config" / "systemd" / "user" / "dw-serve.service").exists()
    assert manifest.entries("service") == []
    assert ["systemctl", "--user", "disable", "--now", "dw-serve"] in fake_runner.calls


def test_linger_uses_sudo_only_when_asked(fake_runner):
    setup_service.enable_linger(fake_runner, use_sudo=False)
    assert fake_runner.calls[-1][:2] == ["loginctl", "enable-linger"]
    setup_service.enable_linger(fake_runner, use_sudo=True)
    assert fake_runner.calls[-1][:3] == ["sudo", "loginctl", "enable-linger"]


def test_linger_enabled_reads_loginctl():
    assert setup_service.linger_enabled(FakeRunner(stdout="Linger=yes\n")) is True
    assert setup_service.linger_enabled(FakeRunner(stdout="Linger=no\n")) is False


def test_status_on_linux_reports_logout_behaviour(home):
    runner = FakeRunner(stdout="Linger=no\n")
    status = setup_service.service_status(runner, platform="linux")
    assert status == {"installed": True, "running": True, "stops_at_logout": True}


def test_status_on_macos_reads_launchctl_print():
    runner = FakeRunner(stdout="state = running\n")
    status = setup_service.service_status(runner, platform="darwin")
    assert status["installed"] and status["running"]
    assert not setup_service.service_status(FakeRunner(returncode=113), "darwin")[
        "installed"
    ]


def test_restart_runs_the_platform_command(fake_runner):
    setup_service.restart_service(fake_runner, platform="linux")
    assert fake_runner.calls[-1] == ["systemctl", "--user", "restart", "dw-serve"]
    setup_service.restart_service(fake_runner, platform="darwin")
    assert fake_runner.calls[-1][:3] == ["launchctl", "kickstart", "-k"]


def test_an_unknown_platform_is_refused(fake_runner, manifest):
    with pytest.raises(SetupError):
        setup_service.install_service(manifest, fake_runner, platform="plan9")


def test_the_default_runner_sanitizes_and_uses_no_shell(monkeypatch):
    seen = {}

    def fake_run(args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    setup_service.default_runner(["echo", "hi"])
    assert seen["kwargs"].get("shell", False) is False
    with pytest.raises(SetupError, match="dangerous characters"):
        setup_service.default_runner(["echo", "a;b"])


def test_the_default_runner_runs_claude_by_its_full_path(monkeypatch):
    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        setup_service, "claude_executable", lambda: "/home/u/.local/bin/claude"
    )
    setup_service.default_runner(["claude", "mcp", "list"])
    assert seen["args"] == ["/home/u/.local/bin/claude", "mcp", "list"]


def test_a_localized_schtasks_still_reads_running():
    runner = FakeRunner()
    runner.by_command[("schtasks", "/Query")] = (0, "Status: Wird ausgeführt\n")
    runner.by_command[("powershell", "-NoProfile")] = (0, "Running\r\n")
    status = setup_service.service_status(runner, "win32")
    assert status["installed"] and status["running"]


def test_running_is_the_task_state_not_the_query_text():
    runner = FakeRunner()
    runner.by_command[("schtasks", "/Query")] = (0, "Status: Running\n")
    runner.by_command[("powershell", "-NoProfile")] = (0, "Ready\n")
    status = setup_service.service_status(runner, "win32")
    assert status["installed"] and not status["running"]


def test_an_absent_task_is_not_asked_for_its_state():
    runner = FakeRunner()
    runner.by_command[("schtasks", "/Query")] = (1, "")
    status = setup_service.service_status(runner, "win32")
    assert not status["installed"] and not status["running"]
    assert not any(c[0] == "powershell" for c in runner.calls)
