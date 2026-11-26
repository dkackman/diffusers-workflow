# Installer and agent hookup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One pasted line installs dw, starts it at login, connects it to Claude Desktop and Claude Code, and opens the web UI; an "Update dw" button and re-running the line upgrade it; every release ships a `.mcpb` bundle and an MCP Registry entry, and the `dw` plugin carries the tool server.

**Architecture:** Bootstrap shell scripts get uv, Python, the right torch and dw into a fixed venv, then hand off to `dw setup` - a Python command split into state, service, clients and CLI modules, every OS and Claude call behind an injected runner so it is unit-tested with fakes. The server-side upgrade reuses the diffusers updater's thread-and-pip pattern and exits so the service manager restarts it. The `.mcpb` is built from `dw_mcp` alone by a script the release job runs.

**Tech Stack:** Python 3.12 (stdlib `plistlib`, `json`, `subprocess`, `argparse`), bash and PowerShell, uv, FastAPI route + Svelte 5 UI, GitHub Actions, `@anthropic-ai/mcpb` CLI, `mcp-publisher`.

**Spec:** `docs/superpowers/specs/2026-11-25-installer-design.md`

## Global Constraints

- Install target: `diffusers-workflow[server,mcp]`, Python >= 3.12.
- Install root: `~/.local/share/diffusers-workflow` (macOS, Linux), `%LOCALAPPDATA%\diffusers-workflow` (Windows); venv at `<root>/venv`, made with `uv venv --seed` (pip present, so `dw/server/updater.py` works). Env override `DW_INSTALL_ROOT`.
- Install record: `<root>/install.json` = `{"torch_index": "<url or 'pypi'>", "version": "<dw version>", "diffusers_commit": "<sha or null>"}`, written by the bootstrap, read by Python.
- Setup manifest: `get_settings_dir() / "setup-manifest.json"` (`dw/settings.py`), the only record of what setup added.
- Server: loopback `127.0.0.1:8765`, started by setup's service as `dw-serve --host 127.0.0.1 --port 8765 --mcp --managed`. `--managed` (new, Task 3) is how the server knows setup's service started it and the service manager will restart it; it is exposed to routes as `app.state.managed`.
- Torch choice exactly as `install.sh`: driver CUDA >= 13 (driver 580+) -> `https://download.pytorch.org/whl/cu130`; older driver -> PyPI; no driver -> `https://download.pytorch.org/whl/cpu`; macOS -> PyPI plus `fp4-fp8-for-torch-mps`. torchaudio from the same index.
- diffusers: on a machine without git, install `https://github.com/huggingface/diffusers/archive/<sha>.zip` for `main`'s current sha (`--force-reinstall --no-deps`, then a plain `diffusers` install), as `install.sh` does for the git URL.
- Claude Desktop config: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS), `%APPDATA%\Claude\claude_desktop_config.json` (Windows); no Desktop on Linux. Every write merges and backs up to `<file>.dw-backup` first.
- Security (`CLAUDE.md`): paths through `dw/security.py` validators, URLs through `validate_url()`, subprocess args through `sanitize_command_args()`, never `shell=True`.
- Architecture ratchet: `docs/stabilization/baseline.json` `modules` is 282. This plan adds 5 modules (`dw/setup_state.py`, `dw/setup_service.py`, `dw/setup_clients.py`, `dw/setup_cli.py`, `dw/server/self_update.py`) -> 287. Raising it needs Don's approval, asked once before Task 2 is committed; the raise commit says `(modules: 287)`.
- Tests run locally: `venv/bin/python -m pytest <files> -n0 -q`; ruff (`.venv/bin/ruff check` and `ruff format --check`) clean before every commit. UI: `cd ui && npm run check && npm test`.
- Never `git add -A ui` (Don edits `ui/package.json`); add named files.
- Commit headlines use `refs #N`, never `fix: #N` (closes the parent).

## Review Focus

1. **A machine that already runs dw** (a checkout started by hand, lem's systemd unit, mini-ai's LaunchDaemon, Docker): setup must find the server on 8765, install no service, and register against it. Test in Task 5.
2. **A person's own Claude config with other servers, or an existing `dw` entry they wrote by hand**: merged, backed up, never removed by `--uninstall`; a hand-written user-scope `dw` in Claude Code beside the plugin's server is reported as a duplicate, not deleted. Tests in Tasks 4 and 5.
3. **Malformed or missing `claude_desktop_config.json` / `install.json` / manifest** (empty file, invalid JSON, Desktop installed but never opened): a clear message and no write over a file that did not parse. Tests in Tasks 2 and 4.
4. **Re-running setup or the bootstrap** (the upgrade path): no duplicate service, no second marketplace, torch build unchanged, paths refreshed. Tests in Tasks 3, 5 and 7.
5. **The update button on a server setup did not start** (no `--managed`): it runs nothing and returns the command; a running or queued job refuses the update as the diffusers update does. Tests in Task 8.

---

### Task 1: Verify the hosts (spike, no product code)

Throwaway checks that decide Task 10's shape and one detail of Task 4. Steps marked **(Don)** need the Claude Desktop app.

**Files:**
- Modify: `docs/superpowers/specs/2026-11-25-installer-design.md` (append a `## Verification record` section)

- [ ] **Step 1:** Build a minimal uv-type `.mcpb` by hand from `dw_mcp/` (manifest `server.type: "uv"`, a `pyproject.toml` with dw_mcp's four dependencies) and pack it with `npx -y @anthropic-ai/mcpb pack`.
- [ ] **Step 2:** In a scratch plugin directory, add `.mcp.json` that launches that bundle; `claude plugin marketplace add <scratch dir>`, install, start `claude`, run `/mcp`. Record: does Claude Code start a `.mcpb` from a plugin, and does `get_server_info` answer against a local `dw-serve`?
- [ ] **Step 3:** Same scratch plugin with `.mcp.json` launching an absolute `dw-mcp` through `${HOME}` expansion. Record whether `${HOME}` expands.
- [ ] **Step 4:** Run `claude plugin marketplace add --help` and try `dkackman/diffusers-workflow@v0.22.0` and `#v0.22.0`. Record which ref syntax, if any, is accepted.
- [ ] **Step 5 (Don):** In Claude Desktop, add the scratch marketplace, install the plugin, open Cowork, check the server's tools are listed; double-click the bare `.mcpb` and check it installs in Desktop chat.
- [ ] **Step 6:** Write the verification record. It states one decision: **Plugin server = bundle** (Steps 2 and 5 both worked) or **Plugin server = stable path** (otherwise), and the marketplace ref syntax or "no ref".
- [ ] **Step 7: Commit** `docs(spec): installer host verification record`.

---

### Task 2: Install record, setup manifest, safe JSON merge

**Files:**
- Create: `dw/setup_state.py`
- Modify: `docs/stabilization/baseline.json` (`modules` 282 -> 287, after Don's approval)
- Test: `tests/test_setup_state.py`

**Interfaces:**
- Produces:
  - `install_root() -> Path` (honours `DW_INSTALL_ROOT`)
  - `venv_bin(name: str, root: Path | None = None) -> Path` (`bin/<name>` or `Scripts\<name>.exe`)
  - `read_install_record(root: Path | None = None) -> dict | None` (None when absent; raises `SetupError` when unparseable)
  - `class SetupError(Exception)`
  - `class Manifest` with `load(path: Path | None = None) -> Manifest`, `record(kind: str, key: str, detail: dict) -> None`, `forget(kind: str, key: str) -> None`, `entries(kind: str | None = None) -> list[dict]`, `has(kind: str, key: str) -> bool`, `save() -> None`. Kinds: `"service"`, `"desktop"`, `"code-plugin"`, `"code-marketplace"`, `"code-mcp"`, `"linger"`.
  - `merge_json_file(path: Path, update: Callable[[dict], dict]) -> bool` - parses (missing or empty file = `{}`), writes `<path>.dw-backup` before the first change, writes atomically; returns whether it changed; raises `SetupError` naming the file when it does not parse, and writes nothing.

- [ ] **Step 1: Write the failing tests**

```python
def test_install_root_honours_the_override(tmp_path, monkeypatch):
    monkeypatch.setenv("DW_INSTALL_ROOT", str(tmp_path))
    assert setup_state.install_root() == tmp_path

def test_a_missing_record_is_none(tmp_path):
    assert setup_state.read_install_record(tmp_path) is None

def test_an_unparseable_record_names_the_file(tmp_path):
    (tmp_path / "install.json").write_text("{nope")
    with pytest.raises(setup_state.SetupError, match="install.json"):
        setup_state.read_install_record(tmp_path)

def test_manifest_round_trips_and_forgets(tmp_path):
    m = setup_state.Manifest.load(tmp_path / "m.json")
    m.record("desktop", "dw", {"path": "/x"}); m.save()
    again = setup_state.Manifest.load(tmp_path / "m.json")
    assert again.has("desktop", "dw")
    again.forget("desktop", "dw"); again.save()
    assert not setup_state.Manifest.load(tmp_path / "m.json").has("desktop", "dw")

def test_merge_keeps_other_keys_and_backs_up_once(tmp_path):
    f = tmp_path / "c.json"; f.write_text('{"mcpServers": {"other": {"command": "x"}}}')
    changed = setup_state.merge_json_file(f, lambda d: {**d, "mcpServers": {**d["mcpServers"], "dw": {"command": "/dw"}}})
    assert changed and json.loads(f.read_text())["mcpServers"].keys() == {"other", "dw"}
    assert json.loads((tmp_path / "c.json.dw-backup").read_text()) == {"mcpServers": {"other": {"command": "x"}}}

def test_merge_refuses_an_unparseable_file_and_leaves_it(tmp_path):
    f = tmp_path / "c.json"; f.write_text("{broken")
    with pytest.raises(setup_state.SetupError, match="c.json"):
        setup_state.merge_json_file(f, lambda d: d)
    assert f.read_text() == "{broken"

def test_merge_treats_an_empty_file_as_empty(tmp_path):
    f = tmp_path / "c.json"; f.write_text("")
    assert setup_state.merge_json_file(f, lambda d: {**d, "a": 1})
```

- [ ] **Step 2:** Run `venv/bin/python -m pytest tests/test_setup_state.py -n0 -q`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement `dw/setup_state.py` to the interfaces above. Atomic write = temp file in the same directory + `os.replace`. A no-op update (equal dict) writes nothing and returns False.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Ask Don to approve the modules rise 282 -> 287 for the five modules named in Global Constraints. On approval, set `modules` to 287 in `docs/stabilization/baseline.json` as its own commit `chore(arch): raise modules baseline 282 -> 287 (modules: 287) for the installer modules`.
- [ ] **Step 6: Commit** `feat(setup): install record, setup manifest and safe JSON merge`.

---

### Task 3: Per-user service

**Files:**
- Create: `dw/setup_service.py`
- Modify: `dw/serve.py` (a `--managed` flag, stored on `app.state.managed`; default False)
- Test: `tests/test_setup_service.py`, `tests/test_serve.py` (or the file that tests `dw/serve.py`'s arguments)

**Interfaces:**
- Consumes: `Manifest`, `venv_bin`, `SetupError` (Task 2).
- Produces:
  - `Runner = Callable[[list[str]], subprocess.CompletedProcess]` (args are sanitized with `sanitize_command_args()` inside the default runner)
  - `SERVICE_LABEL = "com.github.dkackman.dw-serve"` (launchd), `"dw-serve"` (systemd unit and Windows task name)
  - `serve_command(port: int = 8765) -> list[str]` = `[<venv_bin("dw-serve")>, "--host", "127.0.0.1", "--port", "8765", "--mcp", "--managed"]`
  - `launchd_plist(command: list[str], log_dir: Path) -> bytes` (`KeepAlive` true, `RunAtLoad` true, stdout/stderr in `log_dir`)
  - `systemd_unit(command: list[str]) -> str` (`Restart=always`, `RestartSec=5`, `WantedBy=default.target`)
  - `windows_task_xml(command: list[str]) -> str` (logon trigger, current user, no execution time limit, restart on failure 3 times at 1 minute - exit code 75 counts as a failure)
  - `install_service(manifest: Manifest, runner: Runner, platform: str = sys.platform) -> str` (returns what it did, for the log; replaces an existing definition of ours in place)
  - `uninstall_service(manifest: Manifest, runner: Runner, platform: str = sys.platform) -> str`
  - `service_status(runner: Runner, platform: str = sys.platform) -> dict` = `{"installed": bool, "running": bool, "stops_at_logout": bool | None}`
  - `restart_service(runner: Runner, platform: str = sys.platform) -> None`
  - `linger_enabled(runner: Runner) -> bool`, `enable_linger(runner: Runner, use_sudo: bool) -> bool`

- [ ] **Step 1: Write the failing tests** (a recording fake runner; `platform` passed explicitly; `HOME` pointed at `tmp_path`)

```python
def test_the_plist_keeps_the_server_alive():
    plist = plistlib.loads(setup_service.launchd_plist(["/v/bin/dw-serve", "--mcp"], Path("/logs")))
    assert plist["KeepAlive"] is True and plist["RunAtLoad"] is True
    assert plist["ProgramArguments"] == ["/v/bin/dw-serve", "--mcp"]

def test_the_unit_restarts_always():
    unit = setup_service.systemd_unit(["/v/bin/dw-serve", "--mcp"])
    assert "Restart=always" in unit and "ExecStart=/v/bin/dw-serve --mcp" in unit

def test_serve_command_binds_loopback_with_mcp_and_managed():
    cmd = setup_service.serve_command()
    assert cmd[1:] == ["--host", "127.0.0.1", "--port", "8765", "--mcp", "--managed"]

def test_serve_records_managed_on_the_app():  # in the dw/serve.py test file
    assert build_app_from_args(["--managed"]).state.managed is True
    assert build_app_from_args([]).state.managed is False

def test_installing_twice_on_linux_enables_once_and_records_once(tmp_path, fake_runner, manifest):
    setup_service.install_service(manifest, fake_runner, platform="linux")
    setup_service.install_service(manifest, fake_runner, platform="linux")
    assert fake_runner.calls.count(["systemctl", "--user", "enable", "--now", "dw-serve"]) == 1
    assert len(manifest.entries("service")) == 1

def test_uninstall_removes_only_a_recorded_service(fake_runner, manifest):
    assert "nothing" in setup_service.uninstall_service(manifest, fake_runner, platform="darwin")
    assert fake_runner.calls == []

def test_linger_uses_sudo_only_when_asked(fake_runner):
    setup_service.enable_linger(fake_runner, use_sudo=False)
    assert fake_runner.calls[-1][:2] == ["loginctl", "enable-linger"]
    setup_service.enable_linger(fake_runner, use_sudo=True)
    assert fake_runner.calls[-1][:3] == ["sudo", "loginctl", "enable-linger"]
```

- [ ] **Step 2:** Run `venv/bin/python -m pytest tests/test_setup_service.py -n0 -q`. Expected: FAIL.
- [ ] **Step 3:** Implement. macOS: write `~/Library/LaunchAgents/<label>.plist`, `launchctl bootout gui/<uid>/<label>` (ignore failure) then `launchctl bootstrap gui/<uid> <plist>`; logs in `~/Library/Logs/diffusers-workflow/`. Linux: write `~/.config/systemd/user/dw-serve.service`, `systemctl --user daemon-reload`, `enable --now` only when not already enabled (else `restart`). Windows: `schtasks /Create /TN dw-serve /XML <file> /F`, `/Run`. Status reads `launchctl print`, `systemctl --user is-active`/`is-enabled` plus `loginctl show-user <user> -p Linger`, `schtasks /Query`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat(setup): per-user dw-serve service for launchd, systemd and Task Scheduler`.

---

### Task 4: Claude Desktop and Claude Code registration

**Files:**
- Create: `dw/setup_clients.py`
- Test: `tests/test_setup_clients.py`

**Interfaces:**
- Consumes: `Manifest`, `merge_json_file`, `venv_bin`, `SetupError` (Task 2); `Runner` (Task 3).
- Produces:
  - `desktop_config_path(platform: str = sys.platform) -> Path | None` (None on Linux; also checks the Microsoft Store layout `%LOCALAPPDATA%\Packages\Claude_*\LocalCache\Roaming\Claude` on Windows and prefers whichever exists)
  - `desktop_present(platform: str = sys.platform) -> bool` (the config's directory exists)
  - `register_desktop(manifest: Manifest, platform: str = sys.platform) -> str` - sets `mcpServers.dw = {"command": <abs dw-mcp>, "args": []}`; an existing `dw` entry setup did not write is left alone and reported
  - `unregister_desktop(manifest: Manifest, platform: str = sys.platform) -> str` - removes `dw` only when the manifest says setup wrote it and the entry still has the command it wrote
  - `claude_cli() -> str | None` (`shutil.which("claude")`)
  - `MARKETPLACE_SOURCE = "dkackman/diffusers-workflow"`, `PLUGIN = "dw@diffusers-workflow"`
  - `code_commands(version: str, ref_syntax: str | None) -> list[list[str]]` - marketplace add (with the ref from Task 1, or none), plugin install, `mcp add --scope user dw-notes -- <abs dw-mcp> --notes-channel`. Before Task 10 it also returns `mcp add --scope user dw -- <abs dw-mcp>`; Task 10 removes that line.
  - `register_code(manifest: Manifest, runner: Runner, version: str) -> str` - skips a marketplace or plugin already present (`claude plugin marketplace list`, `claude plugin list`), including one synced from the person's claude.ai account
  - `unregister_code(manifest: Manifest, runner: Runner) -> str`
  - `code_duplicates(runner: Runner) -> list[str]` - user-scope MCP servers other than ours whose command ends in `dw-mcp`
  - `desktop_plugin_steps() -> str` - the printed steps for adding the marketplace and installing `dw` in Claude Desktop

- [ ] **Step 1: Write the failing tests** (`HOME`/`APPDATA` pointed at `tmp_path`; fake runner answering `list` calls)

```python
def test_desktop_merge_keeps_other_servers(desktop_dir, manifest):
    cfg = desktop_dir / "claude_desktop_config.json"
    cfg.write_text('{"mcpServers": {"fs": {"command": "npx"}}, "theme": "dark"}')
    setup_clients.register_desktop(manifest, platform="darwin")
    data = json.loads(cfg.read_text())
    assert data["theme"] == "dark" and set(data["mcpServers"]) == {"fs", "dw"}
    assert Path(data["mcpServers"]["dw"]["command"]).is_absolute()

def test_a_hand_written_dw_entry_is_left_and_reported(desktop_dir, manifest):
    cfg = desktop_dir / "claude_desktop_config.json"
    cfg.write_text('{"mcpServers": {"dw": {"command": "/mine/dw-mcp"}}}')
    msg = setup_clients.register_desktop(manifest, platform="darwin")
    assert "already" in msg and json.loads(cfg.read_text())["mcpServers"]["dw"]["command"] == "/mine/dw-mcp"
    setup_clients.unregister_desktop(manifest, platform="darwin")
    assert "dw" in json.loads(cfg.read_text())["mcpServers"]

def test_desktop_absent_on_linux():
    assert setup_clients.desktop_config_path(platform="linux") is None

def test_a_broken_desktop_config_is_not_overwritten(desktop_dir, manifest):
    cfg = desktop_dir / "claude_desktop_config.json"; cfg.write_text("{x")
    with pytest.raises(setup_state.SetupError):
        setup_clients.register_desktop(manifest, platform="darwin")
    assert cfg.read_text() == "{x"

def test_code_commands_register_the_notes_channel():
    cmds = setup_clients.code_commands("0.23.0", ref_syntax=None)
    notes = next(c for c in cmds if "dw-notes" in c)
    assert notes[:5] == ["claude", "mcp", "add", "--scope", "user"] and notes[-1] == "--notes-channel"

def test_a_present_plugin_is_not_installed_twice(fake_runner_with_plugin, manifest):
    setup_clients.register_code(manifest, fake_runner_with_plugin, "0.23.0")
    assert not any(c[:3] == ["claude", "plugin", "install"] for c in fake_runner_with_plugin.calls)

def test_a_hand_added_dw_mcp_is_a_duplicate(fake_runner_listing_dw):
    assert setup_clients.code_duplicates(fake_runner_listing_dw) == ["dw"]
```

- [ ] **Step 2:** Run `venv/bin/python -m pytest tests/test_setup_clients.py -n0 -q`. Expected: FAIL.
- [ ] **Step 3:** Implement to the interfaces. Parse `claude mcp list` / `claude plugin list` output defensively (a parse failure means "unknown", which skips the duplicate report, never a removal).
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat(setup): register dw with Claude Desktop and Claude Code`.

---

### Task 5: `dw setup` and `dw agent`

**Files:**
- Create: `dw/setup_cli.py`
- Modify: `pyproject.toml` (`[project.scripts]` add `dw = "dw.setup_cli:main"`)
- Test: `tests/test_setup_cli.py`

**Interfaces:**
- Consumes: Tasks 2-4.
- Produces:
  - `main(argv: list[str] | None = None) -> int` - subcommands `setup`, `agent`
  - `dw setup` flags: `--yes`, `--no-service`, `--no-desktop`, `--no-code`, `--no-open`, `--status`, `--json` (with `--status`), `--uninstall`, `--port` (default 8765)
  - `status_report(runner, platform) -> dict` - the `--status --json` contract, keys exactly: `version`, `install_root`, `install_record`, `server` (`{"url", "answering": bool, "managed": bool}`), `service` (Task 3's dict), `desktop` (`{"present", "registered", "ours"}`), `code` (`{"cli", "plugin", "plugin_version", "notes_channel", "duplicates"}`), `plugin_version_mismatch: bool`, `missing: list[str]`
  - `server_answering(port: int) -> bool` - `GET http://127.0.0.1:<port>/api/health`, 2 s timeout
  - `agent_command(extra: list[str]) -> list[str]` = `["claude", "--dangerously-load-development-channels", "server:dw-notes", *extra]`

- [ ] **Step 1: Write the failing tests** (every OS and Claude call faked; `server_answering` monkeypatched)

```python
def test_an_answering_server_gets_no_service(fakes):
    fakes.server_answering = True
    assert setup_cli.main(["setup", "--yes", "--no-open"]) == 0
    assert fakes.service_installs == [] and "already running" in fakes.out

def test_setup_is_safe_to_rerun(fakes):
    setup_cli.main(["setup", "--yes", "--no-open"]); setup_cli.main(["setup", "--yes", "--no-open"])
    assert len(fakes.manifest.entries("service")) == 1 and len(fakes.manifest.entries("desktop")) == 1
    assert fakes.marketplace_adds == 1

def test_status_json_has_the_contract_keys(fakes, capsys):
    setup_cli.main(["setup", "--status", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"version", "install_root", "install_record", "server", "service", "desktop", "code", "plugin_version_mismatch", "missing"}

def test_uninstall_removes_only_manifest_entries(fakes):
    setup_cli.main(["setup", "--yes", "--no-open"])
    setup_cli.main(["setup", "--uninstall", "--yes"])
    assert fakes.manifest.entries() == [] and fakes.removed_only_recorded

def test_skipped_targets_say_why(fakes):
    fakes.claude_cli = None; fakes.platform = "linux"
    setup_cli.main(["setup", "--yes", "--no-open"])
    assert "Claude Code not found" in fakes.out and "no Claude Desktop on Linux" in fakes.out

def test_agent_loads_the_notes_channel():
    assert setup_cli.agent_command(["-c"]) == ["claude", "--dangerously-load-development-channels", "server:dw-notes", "-c"]

def test_linux_asks_before_lingering(fakes):
    fakes.platform = "linux"; fakes.answers = ["n"]
    setup_cli.main(["setup", "--no-open"])
    assert fakes.linger_calls == [] and "stops at logout" in fakes.out
```

- [ ] **Step 2:** Run `venv/bin/python -m pytest tests/test_setup_cli.py -n0 -q`. Expected: FAIL.
- [ ] **Step 3:** Implement. Order: read install record -> server check -> service (unless `--no-service` or answering) -> Linux linger prompt (`--yes` = yes; a sudo refusal is reported, not fatal) -> Desktop (registration + `desktop_plugin_steps()` printed) -> Claude Code -> duplicate report -> when interactive and no HF token (`huggingface_hub.get_token()`), offer `hf auth login` -> open `http://127.0.0.1:<port>` (`webbrowser`) unless `--no-open` -> final line "Start a new Claude session to pick up the registration." `dw agent` uses `os.execvp` on POSIX, `subprocess.run` + exit code on Windows; with no `claude` on `PATH` it prints how to install Claude Code and returns 1. Exit 0 when every requested step succeeded or was skipped with a reason, 1 otherwise.
- [ ] **Step 4:** Run the tests. Expected: PASS. Then `venv/bin/pip install -e . --no-deps` and `venv/bin/dw setup --status` prints a report.
- [ ] **Step 5: Commit** `feat(setup): dw setup and dw agent commands`.

---

### Task 6: diffusers updater without git

**Files:**
- Modify: `dw/server/updater.py` (`diffusers_install_info`, `build_pip_commands`)
- Test: `tests/test_updater.py`

**Interfaces:**
- Produces: `DIFFUSERS_ARCHIVE_URL = "https://github.com/huggingface/diffusers/archive/{sha}.zip"`; `main_commit(fetch=None) -> str` (GitHub `commits/main`, `validate_url()`, injectable fetch); `build_pip_commands(commit=None, revert=False, python=None, git=None)` - `git` defaults to `shutil.which("git") is not None`; without git the HEAD target is the archive of `commit or main_commit()`. `diffusers_install_info()` also reads the commit from `direct_url.json`'s `url` when it matches the archive pattern.

- [ ] **Step 1: Write the failing tests**

```python
def test_without_git_head_installs_the_main_archive():
    cmds = updater.build_pip_commands(git=False, commit="a" * 40)
    assert cmds[0][-1] == f"https://github.com/huggingface/diffusers/archive/{'a' * 40}.zip"

def test_with_git_the_target_is_unchanged():
    assert updater.build_pip_commands(git=True)[0][-1] == updater.DIFFUSERS_GIT_URL

def test_the_commit_is_read_from_an_archive_install(fake_dist):
    fake_dist.direct_url = {"url": f"https://github.com/huggingface/diffusers/archive/{'b' * 40}.zip", "archive_info": {}}
    assert updater.diffusers_install_info()["commit"] == "b" * 40
```

- [ ] **Step 2:** Run `venv/bin/python -m pytest tests/test_updater.py -n0 -q`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement. The sha is checked against `^[0-9a-f]{40}$` before it enters a URL.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat(updater): install and read diffusers from a GitHub archive when git is absent`.

---

### Task 7: Bootstrap scripts

**Files:**
- Create: `dw/install_scripts/bootstrap.sh`, `dw/install_scripts/bootstrap.ps1`
- Modify: `pyproject.toml` (`[tool.setuptools.package-data]` `dw` add `"install_scripts/*"`)
- Test: `tests/test_bootstrap.py`

**Interfaces:**
- Produces: `bootstrap.sh [--torch-index URL|pypi|cpu] [--version X] [dw setup flags...]`, same for `.ps1`. Steps exactly as the spec's *Decided shape*: uv if missing (`https://astral.sh/uv/install.sh` / `install.ps1`), torch choice (re-used from `install.json` when present, else detected; `--torch-index` overrides), `uv venv --seed --python 3.12` at `<root>/venv` when absent, torch/torchvision/torchaudio from the index, `diffusers-workflow[server,mcp]` (`==<version>` or latest; on an existing venv `-U` with `torch==<installed>` as a constraint), diffusers archive of `main` (Global Constraints), macOS `fp4-fp8-for-torch-mps`, write `install.json`, then `exec <venv>/bin/dw setup "$@"`. `DW_BOOTSTRAP_DRY_RUN=1` prints each command instead of running it (the test hook). `DW_BOOTSTRAP_WHEEL=<path>` installs that wheel in place of the PyPI release (for Task 12's pre-release runs).

- [ ] **Step 1: Write the failing tests** (run under bash with `DW_BOOTSTRAP_DRY_RUN=1`, fake `nvidia-smi`, `uv` and `uname` on a temp `PATH`, as `tests/test_install_docs.py` runs `install.sh` with a fake npm)

```python
@pytest.mark.parametrize("driver_cuda,index", [("13.0", "cu130"), ("12.4", "pypi"), (None, "cpu")])
def test_torch_index_follows_the_driver(driver_cuda, index, run_bootstrap):
    out = run_bootstrap(driver_cuda=driver_cuda)
    assert index in out.torch_line

def test_a_rerun_keeps_the_recorded_torch_index(run_bootstrap, root_with_record_cpu):
    out = run_bootstrap(driver_cuda="13.0", root=root_with_record_cpu)
    assert "whl/cpu" in out.torch_line and "uv venv" not in out.text

def test_it_installs_the_server_and_mcp_extras_and_hands_off(run_bootstrap):
    out = run_bootstrap(driver_cuda=None, args=["--no-open"])
    assert "diffusers-workflow[server,mcp]" in out.text and out.last_line.endswith("dw setup --no-open")

def test_diffusers_comes_from_an_archive_not_git(run_bootstrap):
    assert "git+" not in run_bootstrap(driver_cuda=None).text

def test_the_bootstrap_ships_in_the_wheel():
    assert "install_scripts/*" in tomllib.loads(PYPROJECT.read_text())["tool"]["setuptools"]["package-data"]["dw"]
```

A PowerShell twin of the index test runs when `pwsh` is on `PATH` (skip otherwise).

- [ ] **Step 2:** Run `venv/bin/python -m pytest tests/test_bootstrap.py -n0 -q`. Expected: FAIL.
- [ ] **Step 3:** Write both scripts. The driver detection is a copy of `install.sh` / `install.ps1`'s, with a comment naming the source so the two are changed together.
- [ ] **Step 4:** Run the tests and `shellcheck dw/install_scripts/bootstrap.sh`. Expected: PASS, no shellcheck errors.
- [ ] **Step 5: Commit** `feat(install): one-line bootstrap scripts for macOS, Linux and Windows`.

---

### Task 8: "Update dw" button

**Files:**
- Create: `dw/server/self_update.py`
- Modify: `dw/server/routes/system.py` (two routes beside the diffusers update), `ui/src/lib/generated/openapi.json` (`venv/bin/python scripts/dump_openapi.py`), `ui/src/lib/generated/api-schema.ts` (`npm run gen:api`), `ui/src/lib/api.ts`, `ui/src/lib/pages/ModelsPage.svelte`
- Test: `tests/test_self_update.py`, `tests/test_server_system.py` (or the file that tests the diffusers routes), `ui/src/lib/pages/ModelsPage.test.ts`

**Interfaces:**
- Consumes: `read_install_record`, `venv_bin` (Task 2); `build_pip_commands` (Task 6).
- Produces:
  - `GET /api/system/dw` -> `DwUpdateState` = `{"version": str, "latest": str | None, "managed": bool, "command": str, "status": "idle"|"running"|"succeeded"|"failed", "error": str | None, "log": str | None}`. `latest` from `https://pypi.org/pypi/diffusers-workflow/json` (cached 10 min; None when unreachable). `command` is the one-liner from `INSTALL.md`.
  - `POST /api/system/dw/update` -> `DwUpdateState`; 409 when a job is running or queued (same wording pattern as the diffusers update), 409 when `managed` is false (body says to run `command`).
  - `class DwUpdater` in `self_update.py` with `start(on_success)` / `status()`, the `DiffusersUpdater` pattern: a thread runs `build_upgrade_commands(python, torch_version, torch_index)` = `[python, "-m", "pip", "install", "-U", "diffusers-workflow[server,mcp]", "-c", <constraints file pinning torch==<installed>>, (+ "--extra-index-url", index when not pypi)]`, then Task 6's diffusers commands, then `[venv_bin("dw"), "setup", "--yes", "--no-open", "--no-service"]`; on success it schedules a process exit with code 75 after the response so the service manager restarts the server.
  - `api.dwStatus()`, `api.updateDw()` in `ui/src/lib/api.ts`.

- [ ] **Step 1: Write the failing tests**

```python
def test_an_unmanaged_server_refuses_and_returns_the_command(client):  # app.state.managed False
    r = client.post("/api/system/dw/update")
    assert r.status_code == 409
    assert "bootstrap" in r.json()["detail"]

def test_a_queued_job_refuses_the_update(managed_client_with_queued_job):
    assert managed_client_with_queued_job.post("/api/system/dw/update").status_code == 409

def test_upgrade_pins_the_installed_torch(tmp_path):
    cmds = self_update.build_upgrade_commands("/v/bin/python", "2.14.1+cu130", "https://download.pytorch.org/whl/cu130", constraints_dir=tmp_path)
    assert (tmp_path / "constraints.txt").read_text().strip() == "torch==2.14.1+cu130"
    assert "diffusers-workflow[server,mcp]" in cmds[0] and cmds[-1][-4:] == ["setup", "--yes", "--no-open", "--no-service"]

def test_success_exits_for_a_restart(fake_exit):
    updater = self_update.DwUpdater(run_fn=lambda: None, exit_fn=fake_exit)
    updater.start(on_success=None); updater.join()
    assert fake_exit.code == 75
```

UI test (`ModelsPage.test.ts`, the shared `apiMock.ts`): with `managed: false` the page shows the command and no button; with `latest` newer than `version` and `managed: true`, "Update dw" calls `api.updateDw()`; `status: "running"` shows progress and keeps the 2 s poll the diffusers update uses.

- [ ] **Step 2:** Run `venv/bin/python -m pytest tests/test_self_update.py -n0 -q` and `cd ui && npx vitest run src/lib/pages/ModelsPage.test.ts`. Expected: FAIL.
- [ ] **Step 3:** Implement the module and routes (`dw/server/CLAUDE.md` conventions; response models strict), regenerate `openapi.json` and `api-schema.ts`, add the API calls and the Models page row beside the diffusers row, matching its markup and copy style.
- [ ] **Step 4:** Run the Python tests, `venv/bin/python scripts/dump_openapi.py --check`, `cd ui && npm run check && npm test`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(server,ui): Update dw button for an installed copy` (named files only under `ui/`).

---

### Task 9: The `.mcpb` bundle and the release job

**Files:**
- Create: `packaging/mcpb/manifest.json.in`, `packaging/mcpb/server.json.in`, `scripts/build_mcpb.py`
- Modify: `.github/workflows/ci.yml` (`wheel` job builds the bundle; `release` job attaches it, both bootstrap scripts and `SHA256SUMS`; new `mcp-registry` job), `tests/test_mcp_main.py` or new `tests/test_mcpb.py`
- Test: `tests/test_mcpb.py`

**Interfaces:**
- Produces:
  - `scripts/build_mcpb.py --out dist/` stages `dw_mcp/` + rendered `manifest.json` (`server.type: "uv"`, `name: "dw"`, `version` = pyproject version, `user_config`: `url` string default `http://127.0.0.1:8765`, `token` string `sensitive: true`, `required: false`; passed to the server as `DW_MCP_URL` / `DW_API_TOKEN`) + a `pyproject.toml` whose dependencies are `mcp`, `httpx`, `anyio`, `packaging` with the floors copied from the root `pyproject.toml`, then runs `npx -y @anthropic-ai/mcpb pack <stage> dist/dw-mcp-<version>.mcpb`. `--stage-only` stops before packing (the test hook).
  - `server.json` for the registry: `name: "io.github.dkackman/dw"`, `packages: [{"registryType": "mcpb", "identifier": "https://github.com/dkackman/diffusers-workflow/releases/download/v<version>/dw-mcp-<version>.mcpb", "fileSha256": <sha>}]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_dw_mcp_imports_nothing_from_dw():
    for path in (REPO / "dw_mcp").glob("*.py"):
        tree = ast.parse(path.read_text())
        names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        names += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 0]
        assert not any(m == "dw" or m.startswith("dw.") for m in names), path.name

def test_the_stage_carries_the_engine_version_and_the_floors(tmp_path):
    subprocess.run([sys.executable, "scripts/build_mcpb.py", "--stage-only", "--out", str(tmp_path)], check=True, cwd=REPO)
    manifest = json.loads((tmp_path / "stage" / "manifest.json").read_text())
    assert manifest["version"] == PYPROJECT_VERSION and manifest["server"]["type"] == "uv"
    deps = tomllib.loads((tmp_path / "stage" / "pyproject.toml").read_text())["project"]["dependencies"]
    assert "mcp>=2.3.0" in deps

def test_the_identifier_contains_mcp_and_points_at_the_release():
    rendered = render_server_json("0.23.0", "f" * 64)
    pkg = rendered["packages"][0]
    assert "mcp" in pkg["identifier"] and pkg["identifier"].endswith("/v0.23.0/dw-mcp-0.23.0.mcpb")
```

- [ ] **Step 2:** Run `venv/bin/python -m pytest tests/test_mcpb.py -n0 -q`. Expected: FAIL.
- [ ] **Step 3:** Write the templates and script. In `ci.yml`: `wheel` adds `actions/setup-node`, runs `scripts/build_mcpb.py --out dist/`, and a smoke step that unpacks the bundle and runs its server with `uv run` for `--help`; `release` copies `dw/install_scripts/bootstrap.*` into `dist/`, writes `SHA256SUMS` over `dist/*`, and attaches all; new `mcp-registry` job (`needs: release`, `permissions: id-token: write`, `continue-on-error: true`) renders `server.json` with the bundle's sha, installs `mcp-publisher`, `mcp-publisher login github-oidc`, `mcp-publisher publish`.
- [ ] **Step 4:** Run the tests and `npx -y @anthropic-ai/mcpb pack` locally through the script (full build). Expected: PASS and a `dist/dw-mcp-<version>.mcpb`.
- [ ] **Step 5: Commit** `feat(release): dw-mcp .mcpb bundle, checksums and MCP Registry entry`.

---

### Task 10: The plugin carries the tool server

Shape set by Task 1's verification record.

**Files:**
- Create: `plugins/dw/.mcp.json`
- Modify: `dw/setup_clients.py` (`code_commands` drops `mcp add dw`), `scripts/release.sh` and `scripts/update_plugin.sh` if the bundle variant puts a file in the plugin, `tests/test_plugin_skills.py`, `tests/test_setup_clients.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_the_plugin_declares_the_dw_server():
    servers = json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"]
    assert set(servers) == {"dw"}

def test_setup_no_longer_adds_a_separate_dw_server():
    cmds = setup_clients.code_commands("0.23.0", ref_syntax=None)
    assert not any(c[:3] == ["claude", "mcp", "add"] and "dw" in c and "--notes-channel" not in c for c in cmds)
```

Plus, by variant: **bundle** - the bundle the entry names is the one `scripts/build_mcpb.py` writes for the plugin version, and `release.sh` refreshes it; **stable path** - the entry's command is the `${HOME}`-relative `venv_bin("dw-mcp")` path of the default install root (Windows equivalent per the verification record).

- [ ] **Step 2:** Run `venv/bin/python -m pytest tests/test_plugin_skills.py tests/test_setup_clients.py -n0 -q`. Expected: FAIL.
- [ ] **Step 3:** Implement the variant the record chose.
- [ ] **Step 4:** Run the tests; `claude plugin validate plugins/dw`. Expected: PASS, valid.
- [ ] **Step 5: Commit** `feat(plugin): dw plugin carries the dw tool server`.

---

### Task 11: Documentation

Every item in the spec's *Documentation* section. Facts come from Tasks 2-10 as built (flags, paths, labels, JSON keys), not from this plan.

**Files:**
- Create: `docs/INSTALL.md`
- Modify: `README.md`, `docs/DEPENDENCIES.md`, `docs/MCP.md`, `docs/SERVER.md`, `docs/REMOTE.md`, `docs/WORKSPACES.md`, `plugins/dw/README.md`, `docs/RELEASING.md`, `docs/ARCHITECTURE.md`, `CLAUDE.md`, `dw/server/guides.py` (`GUIDES["install"] = ("INSTALL.md", "<one-line description>")`)
- Test: `tests/test_install_docs.py`, `tests/test_server_guides.py`, `tests/test_docs_links.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_readme_leads_with_the_one_liner():
    text = README.read_text()
    assert text.index("releases/latest/download/bootstrap.sh") < text.index(CLONE)

def test_install_md_documents_every_setup_flag():
    text = (REPO / "docs" / "INSTALL.md").read_text()
    for flag in ("--yes", "--no-service", "--no-desktop", "--no-code", "--no-open", "--status", "--json", "--uninstall"):
        assert flag in text

def test_install_md_lists_the_status_contract_keys():
    text = (REPO / "docs" / "INSTALL.md").read_text()
    for key in ("install_root", "plugin_version_mismatch", "missing"):
        assert f"`{key}`" in text

def test_install_is_a_served_guide():
    assert guides.GUIDES["install"][0] == "INSTALL.md"
```

The existing clone-URL pins in `tests/test_install_docs.py` stay: the checkout route keeps `git clone`.

- [ ] **Step 2:** Run `venv/bin/python -m pytest tests/test_install_docs.py tests/test_server_guides.py tests/test_docs_links.py -n0 -q`. Expected: the new tests FAIL.
- [ ] **Step 3:** Write the docs. `ARCHITECTURE.md` row: concept "Installer setup", modules `dw/setup_state.py`, `dw/setup_service.py`, `dw/setup_clients.py`, `dw/setup_cli.py`, rule "setup removes only what `setup-manifest.json` records; every config write merges and backs up first; an unparseable config is never written", tests `tests/test_setup_state.py::test_merge_refuses_an_unparseable_file_and_leaves_it`, `tests/test_setup_cli.py::test_uninstall_removes_only_manifest_entries`. `RELEASING.md` *Unreleased* entry and the release-job steps, including the one-time MCP Registry setup (the `io.github.dkackman` namespace comes from the GitHub OIDC login; nothing to store).
- [ ] **Step 4:** Run the doc tests, then the full suite `venv/bin/python -m pytest -q` and `venv/bin/python scripts/arch_metrics.py --check docs/stabilization/baseline.json`. Expected: PASS; modules 287.
- [ ] **Step 5: Commit** `docs: installer, dw setup, the bundle and the plugin's server`.

---

### Task 12: Clean-machine runs (before release)

- [ ] **Step 1:** Build the wheel (`scripts/build_dist.sh`) and serve `dist/` plus the bootstrap locally; on a fresh macOS user account run the bootstrap with `DW_BOOTSTRAP_WHEEL` pointing at the local wheel (Task 7).
- [ ] **Step 2 (Don):** same on a fresh Windows user and on lem in a scratch user (lingering prompt; existing `dw-serve` on 8765 must be detected - run with the system unit stopped, then started).
- [ ] **Step 3:** Record results in the PR description: install time, `dw setup --status --json` output, a note reaching a `dw agent` session live.
