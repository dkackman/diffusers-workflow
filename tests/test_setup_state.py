"""The installer's state: the install record, the setup manifest and the safe JSON merge."""

import json
import stat
import sys
from pathlib import Path

import pytest

from dw import setup_state


def test_install_root_honours_the_override(tmp_path, monkeypatch):
    monkeypatch.setenv("DW_INSTALL_ROOT", str(tmp_path))
    assert setup_state.install_root() == tmp_path


def test_install_root_default_is_per_platform(monkeypatch):
    monkeypatch.delenv("DW_INSTALL_ROOT", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert (
        setup_state.install_root()
        == Path.home() / ".local" / "share" / "diffusers-workflow"
    )
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", "C:\\Users\\x\\AppData\\Local")
    assert setup_state.install_root() == Path("C:\\Users\\x\\AppData\\Local") / (
        "diffusers-workflow"
    )


def test_beside_python_names_the_running_venv(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "executable", str(tmp_path / "bin" / "python"))
    assert setup_state.beside_python("dw", platform="linux") == tmp_path / "bin" / "dw"
    assert setup_state.beside_python("dw", platform="win32") == (
        tmp_path / "bin" / "dw.exe"
    )


def test_claude_is_found_off_path(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_state.shutil, "which", lambda name: None)
    monkeypatch.setattr(setup_state, "CLAUDE_SYSTEM_DIRS", (str(tmp_path / "brew"),))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert setup_state.claude_executable(platform="darwin") is None
    brew = tmp_path / "brew" / "claude"
    brew.parent.mkdir()
    brew.write_text("")
    brew.chmod(0o755)
    assert setup_state.claude_executable(platform="darwin") == str(brew)
    local = tmp_path / ".local" / "bin" / "claude"
    local.parent.mkdir(parents=True)
    local.write_text("")
    local.chmod(0o755)
    assert setup_state.claude_executable(platform="linux") == str(local)


def test_claude_on_windows_finds_the_npm_cmd(tmp_path, monkeypatch):
    monkeypatch.setattr(setup_state.shutil, "which", lambda name: None)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    npm = tmp_path / "Roaming" / "npm" / "claude.cmd"
    npm.parent.mkdir(parents=True)
    npm.write_text("")
    npm.chmod(0o755)
    assert setup_state.claude_executable(platform="win32") == str(npm)


def test_claude_on_path_wins(monkeypatch):
    monkeypatch.setattr(setup_state.shutil, "which", lambda name: "/p/claude")
    assert setup_state.claude_executable() == "/p/claude"


def test_shim_dir_is_the_same_everywhere():
    assert (
        setup_state.shim_dir()
        == Path.home() / ".local" / "share" / "diffusers-workflow" / "bin"
    )


def test_a_missing_record_is_none(tmp_path):
    assert setup_state.read_install_record(tmp_path) is None


def test_a_record_is_read(tmp_path):
    (tmp_path / "install.json").write_text('{"torch_index": "pypi"}')
    assert setup_state.read_install_record(tmp_path) == {"torch_index": "pypi"}


def test_an_unparseable_record_names_the_file(tmp_path):
    (tmp_path / "install.json").write_text("{nope")
    with pytest.raises(setup_state.SetupError, match="install.json"):
        setup_state.read_install_record(tmp_path)


def test_manifest_round_trips_and_forgets(tmp_path):
    m = setup_state.Manifest.load(tmp_path / "m.json")
    m.record("desktop", "dw", {"path": "/x"})
    m.save()
    again = setup_state.Manifest.load(tmp_path / "m.json")
    assert again.has("desktop", "dw")
    assert again.entries("desktop")[0]["detail"] == {"path": "/x"}
    again.forget("desktop", "dw")
    again.save()
    assert not setup_state.Manifest.load(tmp_path / "m.json").has("desktop", "dw")


def test_manifest_rejects_an_unknown_kind(tmp_path):
    m = setup_state.Manifest.load(tmp_path / "m.json")
    with pytest.raises(setup_state.SetupError):
        m.record("bogus", "k", {})


def test_manifest_record_replaces_the_same_key(tmp_path):
    m = setup_state.Manifest.load(tmp_path / "m.json")
    m.record("service", "dw", {"a": 1})
    m.record("service", "dw", {"a": 2})
    assert len(m.entries()) == 1
    assert m.entries("service")[0]["detail"] == {"a": 2}


def test_merge_keeps_other_keys_and_backs_up_once(tmp_path):
    f = tmp_path / "c.json"
    f.write_text('{"mcpServers": {"other": {"command": "x"}}}')
    changed = setup_state.merge_json_file(
        f,
        lambda d: {
            **d,
            "mcpServers": {**d["mcpServers"], "dw": {"command": "/dw"}},
        },
    )
    assert changed and json.loads(f.read_text())["mcpServers"].keys() == {
        "other",
        "dw",
    }
    backup = tmp_path / "c.json.dw-backup"
    original = {"mcpServers": {"other": {"command": "x"}}}
    assert json.loads(backup.read_text()) == original
    # a second change keeps the first backup
    setup_state.merge_json_file(f, lambda d: {**d, "z": 1})
    assert json.loads(backup.read_text()) == original


def test_merge_noop_writes_nothing(tmp_path):
    f = tmp_path / "c.json"
    f.write_text('{"a": 1}')
    assert setup_state.merge_json_file(f, lambda d: dict(d)) is False
    assert not (tmp_path / "c.json.dw-backup").exists()


def test_merge_refuses_an_unparseable_file_and_leaves_it(tmp_path):
    f = tmp_path / "c.json"
    f.write_text("{broken")
    with pytest.raises(setup_state.SetupError, match="c.json"):
        setup_state.merge_json_file(f, lambda d: d)
    assert f.read_text() == "{broken"
    assert not (tmp_path / "c.json.dw-backup").exists()


def test_merge_treats_an_empty_file_as_empty(tmp_path):
    f = tmp_path / "c.json"
    f.write_text("")
    assert setup_state.merge_json_file(f, lambda d: {**d, "a": 1})
    assert json.loads(f.read_text()) == {"a": 1}


def test_merge_keeps_the_permission_bits(tmp_path):
    f = tmp_path / "c.json"
    f.write_text('{"a": 1}')
    f.chmod(0o644)
    setup_state.merge_json_file(f, lambda d: {**d, "b": 2})
    assert stat.S_IMODE(f.stat().st_mode) == 0o644


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges")
def test_merge_writes_through_a_symlink(tmp_path):
    real = tmp_path / "real.json"
    real.write_text('{"a": 1}')
    link = tmp_path / "link.json"
    link.symlink_to(real)
    setup_state.merge_json_file(link, lambda d: {**d, "b": 2})
    assert link.is_symlink()
    assert json.loads(real.read_text()) == {"a": 1, "b": 2}


def test_a_malformed_manifest_entry_names_the_file(tmp_path):
    f = tmp_path / "m.json"
    f.write_text('{"entries": [{"kind": "desktop"}]}')
    with pytest.raises(setup_state.SetupError, match="m.json"):
        setup_state.Manifest.load(f)


def test_merge_creates_a_missing_file(tmp_path):
    f = tmp_path / "sub" / "c.json"
    assert setup_state.merge_json_file(f, lambda d: {**d, "a": 1})
    assert json.loads(f.read_text()) == {"a": 1}
    assert not (tmp_path / "sub" / "c.json.dw-backup").exists()


def test_an_install_root_with_traversal_is_refused(monkeypatch):
    monkeypatch.setenv("DW_INSTALL_ROOT", "/tmp/../etc")
    with pytest.raises(setup_state.SetupError, match="DW_INSTALL_ROOT"):
        setup_state.install_root()
