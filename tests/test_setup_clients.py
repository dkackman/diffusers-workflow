"""Claude Desktop and Claude Code registration; HOME and APPDATA point at tmp_path."""

import json
import subprocess
from pathlib import Path

import pytest

from dw import setup_clients, setup_state


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    monkeypatch.setenv("DW_INSTALL_ROOT", str(tmp_path / "root"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)


@pytest.fixture
def desktop_dir(tmp_path):
    d = tmp_path / "Library" / "Application Support" / "Claude"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def manifest(tmp_path):
    return setup_state.Manifest.load(tmp_path / "manifest.json")


class FakeRunner:
    def __init__(self, outputs=None, fail=()):
        self.outputs = outputs or {}
        self.fail = fail
        self.calls = []

    def __call__(self, args):
        self.calls.append(args)
        key = " ".join(args)
        code = 1 if any(key.startswith(f) for f in self.fail) else 0
        return subprocess.CompletedProcess(
            args, code, stdout=self.outputs.get(key, ""), stderr="boom"
        )


PLUGINS = "Installed plugins:\n\n  ❯ dw@diffusers-workflow\n    Version: 1\n"
MARKETS = (
    "Configured marketplaces:\n\n  ❯ diffusers-workflow\n"
    "    Source: GitHub (dkackman/diffusers-workflow@v0.22.0)\n"
)


def test_desktop_merge_keeps_other_servers(desktop_dir, manifest):
    cfg = desktop_dir / "claude_desktop_config.json"
    cfg.write_text('{"mcpServers": {"fs": {"command": "npx"}}, "theme": "dark"}')
    setup_clients.register_desktop(manifest, platform="darwin")
    data = json.loads(cfg.read_text())
    assert data["theme"] == "dark" and set(data["mcpServers"]) == {"fs", "dw"}
    assert Path(data["mcpServers"]["dw"]["command"]).is_absolute()
    assert manifest.has("desktop", "dw")


def test_register_desktop_is_idempotent(desktop_dir, manifest):
    setup_clients.register_desktop(manifest, platform="darwin")
    msg = setup_clients.register_desktop(manifest, platform="darwin")
    assert "already registered" in msg


def test_unregister_removes_only_our_entry(desktop_dir, manifest):
    cfg = desktop_dir / "claude_desktop_config.json"
    cfg.write_text('{"mcpServers": {"fs": {"command": "npx"}}}')
    setup_clients.register_desktop(manifest, platform="darwin")
    setup_clients.unregister_desktop(manifest, platform="darwin")
    assert json.loads(cfg.read_text())["mcpServers"] == {"fs": {"command": "npx"}}
    assert not manifest.has("desktop", "dw")


def test_a_changed_entry_is_not_removed(desktop_dir, manifest):
    cfg = desktop_dir / "claude_desktop_config.json"
    setup_clients.register_desktop(manifest, platform="darwin")
    cfg.write_text('{"mcpServers": {"dw": {"command": "/mine"}}}')
    msg = setup_clients.unregister_desktop(manifest, platform="darwin")
    assert "changed" in msg
    assert json.loads(cfg.read_text())["mcpServers"]["dw"]["command"] == "/mine"


def test_a_hand_written_dw_entry_is_left_and_reported(desktop_dir, manifest):
    cfg = desktop_dir / "claude_desktop_config.json"
    cfg.write_text('{"mcpServers": {"dw": {"command": "/mine/dw-mcp"}}}')
    msg = setup_clients.register_desktop(manifest, platform="darwin")
    assert "already" in msg
    assert json.loads(cfg.read_text())["mcpServers"]["dw"]["command"] == "/mine/dw-mcp"
    setup_clients.unregister_desktop(manifest, platform="darwin")
    assert "dw" in json.loads(cfg.read_text())["mcpServers"]


def test_desktop_absent_on_linux(manifest):
    assert setup_clients.desktop_config_path(platform="linux") is None
    assert not setup_clients.desktop_present(platform="linux")
    assert "nothing to register" in setup_clients.register_desktop(
        manifest, platform="linux"
    )


def test_desktop_not_installed_is_skipped(manifest, tmp_path):
    assert "not installed" in setup_clients.register_desktop(
        manifest, platform="darwin"
    )
    assert not (tmp_path / "Library").exists()


def test_a_broken_desktop_config_is_not_overwritten(desktop_dir, manifest):
    cfg = desktop_dir / "claude_desktop_config.json"
    cfg.write_text("{x")
    with pytest.raises(setup_state.SetupError):
        setup_clients.register_desktop(manifest, platform="darwin")
    assert cfg.read_text() == "{x"


def test_windows_prefers_the_store_layout_when_it_holds_the_config(tmp_path):
    store = (
        tmp_path / "Local" / "Packages" / "Claude_abc" / "LocalCache" / "Roaming"
    ) / "Claude"
    store.mkdir(parents=True)
    (store / "claude_desktop_config.json").write_text("{}")
    (tmp_path / "Roaming" / "Claude").mkdir(parents=True)
    assert setup_clients.desktop_config_path("win32") == (
        store / "claude_desktop_config.json"
    )


def test_windows_defaults_to_appdata(tmp_path):
    (tmp_path / "Roaming" / "Claude").mkdir(parents=True)
    assert setup_clients.desktop_config_path("win32") == (
        tmp_path / "Roaming" / "Claude" / "claude_desktop_config.json"
    )
    assert setup_clients.desktop_present("win32")


def test_code_commands_register_the_notes_channel():
    cmds = setup_clients.code_commands("0.23.0", ref_syntax=None)
    notes = next(c for c in cmds if "dw-notes" in c)
    assert notes[:5] == ["claude", "mcp", "add", "--scope", "user"]
    assert notes[-1] == "--notes-channel" and Path(notes[-2]).is_absolute()
    assert cmds[0] == [
        "claude",
        "plugin",
        "marketplace",
        "add",
        "dkackman/diffusers-workflow",
    ]


def test_code_commands_pin_the_ref():
    cmds = setup_clients.code_commands("0.22.0", ref_syntax="@")
    assert cmds[0][-1] == "dkackman/diffusers-workflow@v0.22.0"


def test_register_code_adds_and_records(manifest):
    runner = FakeRunner()
    setup_clients.register_code(manifest, runner, "0.23.0")
    assert ["claude", "plugin", "install", "dw@diffusers-workflow"] in runner.calls
    assert manifest.has("code-marketplace", "diffusers-workflow")
    assert manifest.has("code-plugin", "dw@diffusers-workflow")
    assert manifest.has("code-mcp", "dw-notes")


def test_a_present_plugin_is_not_installed_twice(manifest):
    runner = FakeRunner(
        {"claude plugin list": PLUGINS, "claude plugin marketplace list": MARKETS}
    )
    setup_clients.register_code(manifest, runner, "0.23.0")
    assert not any(c[:3] == ["claude", "plugin", "install"] for c in runner.calls)
    assert not any(
        c[:4] == ["claude", "plugin", "marketplace", "add"] for c in runner.calls
    )
    assert not manifest.has("code-plugin", "dw@diffusers-workflow")


def test_a_failed_command_raises(manifest):
    with pytest.raises(setup_state.SetupError, match="boom"):
        setup_clients.register_code(
            manifest, FakeRunner(fail=("claude plugin install",)), "0.23.0"
        )


def test_unregister_code_removes_only_recorded(manifest):
    manifest.record("code-plugin", "dw@diffusers-workflow", {})
    runner = FakeRunner()
    setup_clients.unregister_code(manifest, runner)
    assert runner.calls == [["claude", "plugin", "uninstall", "dw@diffusers-workflow"]]
    assert not manifest.entries("code-plugin")


def test_unregister_code_keeps_the_record_when_removal_fails(manifest):
    manifest.record("code-mcp", "dw-notes", {})
    msg = setup_clients.unregister_code(
        manifest, FakeRunner(fail=("claude mcp remove",))
    )
    assert "could not remove" in msg and manifest.has("code-mcp", "dw-notes")


def test_a_hand_added_dw_mcp_is_a_duplicate():
    listing = (
        "Checking MCP server health…\n\n"
        "dw: /usr/local/bin/dw-mcp  - ✔ Connected\n"
        "dw-notes: /x/venv/bin/dw-mcp --notes-channel - ✔ Connected\n"
        "other: npx foo - ✔ Connected\n"
    )
    assert setup_clients.code_duplicates(FakeRunner({"claude mcp list": listing})) == [
        "dw"
    ]


def test_unparseable_listing_means_no_duplicates():
    assert setup_clients.code_duplicates(FakeRunner({"claude mcp list": "???"})) == []
    assert setup_clients.code_duplicates(FakeRunner(fail=("claude",))) == []


def test_desktop_plugin_steps_name_the_marketplace():
    assert "dkackman/diffusers-workflow" in setup_clients.desktop_plugin_steps()


REAL_MARKETS = """Configured marketplaces:

  ❯ diffusers-workflow
    Source: Folder (/Users/x/diffusers-workflow)

From claude.ai:

  ❯ claudeai-my-uploads (listed as "My Uploads") — hosted on claude.ai, your uploads · not added
    Add: claude plugin marketplace add --claudeai claudeai-my-uploads

  ❯ (no CLI name) · listed as "Anthropic Directory" (browse on claude.ai)
"""
REAL_PLUGINS = """Installed plugins:

  ❯ code-review@claude-plugins-official
    Version: 1.0.1
    Scope: user
    Status: ✔ enabled

  ❯ dw@diffusers-workflow
    Version: 0.23.0-alpha.1
    Read from: /Users/x/diffusers-workflow/plugins/dw
    Scope: project
    Status: ✔ enabled

  ❯ dw@diffusers-workflow
    Version: 0.23.0-alpha.1
    Scope: user
    Status: ✔ enabled
"""
REAL_MCP = """Checking MCP server health…

claude.ai Claude Docs: https://api.anthropic.com/v1/pages/mcp - ✔ Connected
claude.ai Google Drive: https://drivemcp.googleapis.com/mcp/v1 - ⊘ Disabled for this project (re-enable via /mcp)
dw: http://lem:8765/mcp (HTTP) - ✔ Connected
"""


def test_real_listings_parse():
    assert (
        setup_clients._listed(subprocess.CompletedProcess([], 0, stdout=REAL_MARKETS))
        == {"diffusers-workflow", "claudeai-my-uploads"} - {"claudeai-my-uploads"}
        | set()
    )
    assert setup_clients._listed(
        subprocess.CompletedProcess([], 0, stdout=REAL_PLUGINS)
    ) == {"code-review@claude-plugins-official", "dw@diffusers-workflow"}
    servers = setup_clients._mcp_servers(FakeRunner({"claude mcp list": REAL_MCP}))
    assert servers == {"dw": "http://lem:8765/mcp (HTTP)"}


def test_a_real_http_dw_server_is_a_duplicate():
    runner = FakeRunner({"claude mcp list": REAL_MCP})
    assert setup_clients.code_duplicates(runner) == ["dw"]


def test_real_listings_skip_present_marketplace_and_plugin(manifest):
    runner = FakeRunner(
        {
            "claude plugin list": REAL_PLUGINS,
            "claude plugin marketplace list": REAL_MARKETS,
        }
    )
    setup_clients.register_code(manifest, runner, "0.23.0")
    assert not any(c[1:3] == ["plugin", "install"] for c in runner.calls)


def test_a_dw_entry_without_a_command_is_never_adopted(desktop_dir, manifest):
    cfg = desktop_dir / "claude_desktop_config.json"
    body = '{"mcpServers": {"dw": {"url": "http://x"}}}'
    cfg.write_text(body)
    assert "already" in setup_clients.register_desktop(manifest, platform="darwin")
    assert json.loads(cfg.read_text()) == json.loads(body)
    manifest.record("desktop", "dw", {"path": str(cfg)})
    setup_clients.unregister_desktop(manifest, platform="darwin")
    assert json.loads(cfg.read_text()) == json.loads(body)
