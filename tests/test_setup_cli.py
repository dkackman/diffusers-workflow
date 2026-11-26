"""dw setup / dw agent, with every OS and Claude call faked."""

import json
import subprocess

import pytest

from dw import setup_cli, setup_clients, setup_service


class Fakes:
    def __init__(self):
        self.server_answering = False
        self.claude_cli = "/usr/bin/claude"
        self.platform = "darwin"
        self.interactive = False
        self.answers = []
        self.linger_on = False
        self.service_installs = []
        self.linger_calls = []
        self.marketplace_adds = 0
        self.removed = []
        self.opened = []
        self.out = ""
        self.manifest = None


@pytest.fixture
def fakes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", str(tmp_path / "settings"))
    monkeypatch.setenv("DW_INSTALL_ROOT", str(tmp_path / "root"))
    f = Fakes()
    from dw.setup_state import Manifest

    f.manifest = Manifest.load()
    runner_calls = []

    def runner(args):
        runner_calls.append(args)
        return subprocess.CompletedProcess(args, 1, "", "")

    def install_service(manifest, run, platform, **_):
        f.service_installs.append(platform)
        manifest.record("service", "svc", {})
        manifest.save()
        return "installed"

    def uninstall_service(manifest, run, platform):
        f.removed.append("service")
        manifest.forget("service", "svc")
        manifest.save()
        return "removed"

    def register_desktop(manifest, platform):
        manifest.record("desktop", "dw", {})
        manifest.save()
        return "registered"

    def unregister_desktop(manifest, platform):
        f.removed.append("desktop")
        manifest.forget("desktop", "dw")
        manifest.save()
        return "removed"

    def register_code(manifest, run, version):
        if not manifest.has("code-marketplace", "m"):
            f.marketplace_adds += 1
        manifest.record("code-marketplace", "m", {})
        manifest.save()
        return "Claude Code: added"

    def unregister_code(manifest, run):
        f.removed.append("code")
        manifest.forget("code-marketplace", "m")
        manifest.save()
        return "removed"

    def enable_linger(run, use_sudo):
        f.linger_calls.append(use_sudo)
        return True

    def ask(prompt):
        return f.answers.pop(0)

    patches = {
        (setup_cli, "server_answering"): lambda port: f.server_answering,
        (setup_cli, "_platform"): lambda: f.platform,
        (setup_cli, "_interactive"): lambda: f.interactive,
        (setup_cli, "_input"): ask,
        (setup_cli, "_open_browser"): f.opened.append,
        (setup_cli, "_hf_token"): lambda: "token",
        (setup_service, "default_runner"): runner,
        (setup_service, "install_service"): install_service,
        (setup_service, "uninstall_service"): uninstall_service,
        (setup_service, "linger_enabled"): lambda run: f.linger_on,
        (setup_service, "enable_linger"): enable_linger,
        (setup_service, "service_status"): lambda run, platform: {
            "installed": False,
            "running": False,
            "stops_at_logout": None,
        },
        (setup_clients, "claude_cli"): lambda: f.claude_cli,
        (setup_clients, "register_desktop"): register_desktop,
        (setup_clients, "unregister_desktop"): unregister_desktop,
        (setup_clients, "desktop_present"): lambda platform: platform != "linux",
        (setup_clients, "register_code"): register_code,
        (setup_clients, "unregister_code"): unregister_code,
        (setup_clients, "code_duplicates"): lambda run: [],
        (setup_clients, "_mcp_servers"): lambda run: {},
    }
    for (mod, name), value in patches.items():
        monkeypatch.setattr(mod, name, value)

    def run_main(argv):
        code = setup_cli.main(argv)
        f.out += capsys.readouterr().out
        f.manifest = Manifest.load()
        return code

    f.run = run_main
    return f


def test_an_answering_server_gets_no_service(fakes):
    fakes.server_answering = True
    assert fakes.run(["setup", "--yes", "--no-open"]) == 0
    assert fakes.service_installs == [] and "already running" in fakes.out


def test_setup_is_safe_to_rerun(fakes):
    fakes.run(["setup", "--yes", "--no-open"])
    fakes.run(["setup", "--yes", "--no-open"])
    assert len(fakes.manifest.entries("service")) == 1
    assert len(fakes.manifest.entries("desktop")) == 1
    assert fakes.marketplace_adds == 1


def test_status_json_has_the_contract_keys(fakes):
    assert fakes.run(["setup", "--status", "--json"]) == 0
    report = json.loads(fakes.out)
    assert set(report) == {
        "version", "install_root", "install_record", "server", "service",
        "desktop", "code", "plugin_version_mismatch", "missing",
    }  # fmt: skip
    assert set(report["server"]) == {"url", "answering", "managed"}
    assert set(report["desktop"]) == {"present", "registered", "ours"}
    assert fakes.manifest.entries() == []


def test_status_text_names_what_is_missing(fakes):
    fakes.run(["setup", "--status"])
    assert "missing:" in fakes.out and "service" in fakes.out


def test_uninstall_removes_only_manifest_entries(fakes):
    fakes.run(["setup", "--yes", "--no-open"])
    assert fakes.run(["setup", "--uninstall", "--yes"]) == 0
    assert fakes.manifest.entries() == []
    assert sorted(fakes.removed) == ["code", "desktop", "service"]


def test_uninstall_with_nothing_recorded_does_nothing(fakes):
    fakes.run(["setup", "--uninstall", "--yes"])
    assert fakes.removed == [] and "Nothing to remove" in fakes.out


def test_skipped_targets_say_why(fakes):
    fakes.claude_cli = None
    fakes.platform = "linux"
    fakes.run(["setup", "--yes", "--no-open"])
    assert "Claude Code not found" in fakes.out
    assert "no Claude Desktop on Linux" in fakes.out


def test_flags_skip_steps(fakes):
    fakes.run(
        ["setup", "--yes", "--no-open", "--no-service", "--no-desktop", "--no-code"]
    )
    assert fakes.service_installs == [] and fakes.manifest.entries() == []
    assert fakes.opened == []


def test_a_failing_step_exits_nonzero_and_continues(fakes, monkeypatch):
    def boom(manifest, run, platform):
        raise setup_cli.SetupError("no permission")

    monkeypatch.setattr(setup_service, "install_service", boom)
    assert fakes.run(["setup", "--yes", "--no-open"]) == 1
    assert "no permission" in fakes.out and fakes.manifest.has("desktop", "dw")


def test_agent_loads_the_notes_channel():
    assert setup_cli.agent_command(["-c"]) == [
        "claude", "--dangerously-load-development-channels", "server:dw-notes", "-c",
    ]  # fmt: skip


def test_agent_without_claude_says_how_to_install(monkeypatch, capsys):
    monkeypatch.setattr(setup_cli.shutil, "which", lambda name: None)
    assert setup_cli.main(["agent"]) == 1
    assert "Install it" in capsys.readouterr().err


def test_agent_execs_claude(monkeypatch):
    monkeypatch.setattr(setup_cli.shutil, "which", lambda name: "/bin/claude")
    monkeypatch.setattr(setup_cli.sys, "platform", "linux")
    seen = []
    monkeypatch.setattr(setup_cli.os, "execvp", lambda prog, argv: seen.append(argv))
    setup_cli.main(["agent", "-c"])
    assert seen == [setup_cli.agent_command(["-c"])]


def test_linux_asks_before_lingering(fakes):
    fakes.platform = "linux"
    fakes.interactive = True
    fakes.answers = ["n"]
    fakes.run(["setup", "--no-open"])
    assert fakes.linger_calls == [] and "stops at logout" in fakes.out


def test_linux_yes_enables_lingering(fakes):
    fakes.platform = "linux"
    fakes.run(["setup", "--yes", "--no-open"])
    assert fakes.linger_calls == [True]
    assert len(fakes.manifest.entries("linger")) == 1


def test_linux_without_a_terminal_declines_lingering_and_says_so(fakes):
    fakes.platform = "linux"
    fakes.run(["setup", "--no-open"])
    assert fakes.linger_calls == [] and "not enabled" in fakes.out


def test_setup_opens_the_web_ui(fakes):
    fakes.run(["setup", "--yes"])
    assert fakes.opened == ["http://127.0.0.1:8765"]
    assert "Start a new Claude session" in fakes.out


def test_own_mcp_server_is_not_reported_as_a_duplicate(fakes, monkeypatch):
    fakes.manifest.record("code-mcp", "dw", {})
    fakes.manifest.save()
    monkeypatch.setattr(setup_clients, "code_duplicates", lambda run: ["dw", "other"])
    fakes.run(["setup", "--yes", "--no-open"])
    assert "named other" in fakes.out and "named dw" not in fakes.out
