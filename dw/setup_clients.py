"""Registering dw with Claude Desktop and Claude Code, and undoing exactly that.

Claude Desktop is registered by merging `mcpServers.dw` into its
`claude_desktop_config.json` (backed up first, other servers untouched). Claude Code
is registered through the `claude` CLI: the dw marketplace, the `dw` plugin and the
`dw-notes` channel server. Setup records what it added in the manifest and removes only
that; an entry the person made themselves is reported and left alone. Output of
`claude ... list` is parsed defensively - anything unrecognised means "unknown", which
never removes anything and never reports a duplicate.
"""

import glob
import os
import re
import shutil
import sys
from pathlib import Path

from dw.setup_service import Runner
from dw.setup_state import Manifest, SetupError, merge_json_file, shim_dir, venv_bin

MARKETPLACE_SOURCE = "dkackman/diffusers-workflow"
MARKETPLACE_NAME = "diffusers-workflow"
PLUGIN = "dw@diffusers-workflow"
_SERVER = "dw"
_NOTES = "dw-notes"
_CONFIG = "claude_desktop_config.json"


def desktop_config_path(platform: str = sys.platform) -> Path | None:
    """Claude Desktop's config file (it may not exist yet); None on Linux."""
    if platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / _CONFIG
    if platform != "win32":
        return None
    roaming = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    candidates = [Path(roaming) / "Claude"]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        for store in sorted(
            glob.glob(
                os.path.join(
                    glob.escape(local),
                    "Packages",
                    "Claude_*",
                    "LocalCache",
                    "Roaming",
                    "Claude",
                )
            )
        ):
            candidates.append(Path(store))
    for directory in candidates:
        if (directory / _CONFIG).exists():
            return directory / _CONFIG
    for directory in candidates:
        if directory.exists():
            return directory / _CONFIG
    return candidates[0] / _CONFIG


def desktop_present(platform: str = sys.platform) -> bool:
    path = desktop_config_path(platform)
    return path is not None and path.parent.exists()


def _dw_mcp() -> str:
    return str(venv_bin("dw-mcp"))


def register_desktop(manifest: Manifest, platform: str = sys.platform) -> str:
    path = desktop_config_path(platform)
    if path is None:
        return "Claude Desktop has no Linux build; nothing to register"
    if not path.parent.exists():
        return "Claude Desktop is not installed; skipped"
    command = _dw_mcp()
    recorded = next(iter(manifest.entries("desktop")), None)
    ours = ((recorded or {}).get("detail") or {}).get("command")
    outcome: dict = {}

    def update(data: dict) -> dict:
        servers = data.get("mcpServers", {})
        if not isinstance(servers, dict):
            raise SetupError(f"{path} has an mcpServers value that is not an object")
        existing = servers.get(_SERVER)
        if existing is not None and not (
            ours is not None
            and isinstance(existing, dict)
            and existing.get("command") == ours
        ):
            outcome["foreign"] = True
            return data
        return {
            **data,
            "mcpServers": {**servers, _SERVER: {"command": command, "args": []}},
        }

    changed = merge_json_file(path, update)
    if outcome.get("foreign"):
        return (
            f"Claude Desktop already has a dw entry in {path} that setup did not "
            "write; left as it is"
        )
    manifest.record("desktop", _SERVER, {"path": str(path), "command": command})
    manifest.save()
    verb = "registered dw in" if changed else "dw is already registered in"
    return f"{verb} Claude Desktop ({path})"


def unregister_desktop(manifest: Manifest, platform: str = sys.platform) -> str:
    entry = next(iter(manifest.entries("desktop")), None)
    if entry is None:
        return "nothing to remove: setup did not register Claude Desktop"
    detail = entry.get("detail") or {}
    path = Path(detail["path"]) if detail.get("path") else desktop_config_path(platform)
    wrote = detail.get("command")
    removed: dict = {}

    def update(data: dict) -> dict:
        servers = data.get("mcpServers")
        existing = servers.get(_SERVER) if isinstance(servers, dict) else None
        if (
            wrote is not None
            and isinstance(existing, dict)
            and existing.get("command") == wrote
        ):
            removed["yes"] = True
            return {
                **data,
                "mcpServers": {k: v for k, v in servers.items() if k != _SERVER},
            }
        return data

    if path is not None and path.exists():
        merge_json_file(path, update)
    manifest.forget("desktop", entry["key"])
    manifest.save()
    if removed:
        return f"removed dw from Claude Desktop ({path})"
    return "the dw entry in Claude Desktop was changed since setup; left as it is"


def claude_cli() -> str | None:
    return shutil.which("claude")


def code_commands(version: str, ref_syntax: str | None) -> list[list[str]]:
    """The `claude` commands that register dw with Claude Code, in order."""
    source = MARKETPLACE_SOURCE
    if ref_syntax:
        source = f"{source}{ref_syntax}v{version}"
    mcp = _dw_mcp()
    return [
        ["claude", "plugin", "marketplace", "add", source],
        ["claude", "plugin", "install", PLUGIN],
        [
            "claude",
            "mcp",
            "add",
            "--scope",
            "user",
            _NOTES,
            "--",
            mcp,
            "--notes-channel",
        ],
    ]


def _shim_path(platform: str) -> Path:
    return shim_dir() / ("dw-mcp.exe" if platform == "win32" else "dw-mcp")


def write_mcp_shim(manifest: Manifest, platform: str = sys.platform) -> str:
    """Make `shim_dir()/dw-mcp` reach the `dw-mcp` beside the running interpreter.

    The plugin's `.mcp.json` names that stable path, so one plugin install carries the
    tool server wherever dw is installed. POSIX gets a symlink, Windows a copy. A file
    or link setup did not record is never replaced.
    """
    exe = "dw-mcp.exe" if platform == "win32" else "dw-mcp"
    target = Path(sys.executable).parent / exe
    if not target.exists():
        raise SetupError(f"cannot link the dw-mcp shim: {target} does not exist")
    shim = _shim_path(platform)
    ours = manifest.has("shim", "dw-mcp")
    if os.path.lexists(shim):
        if not ours:
            return f"{shim} already exists and setup did not make it; left as it is"
        if (
            platform != "win32"
            and shim.is_symlink()
            and shim.resolve() == target.resolve()
        ):
            return f"dw-mcp shim already in place ({shim})"
        shim.unlink()
    shim.parent.mkdir(parents=True, exist_ok=True)
    if platform == "win32":
        shutil.copy2(target, shim)
    else:
        shim.symlink_to(target)
    manifest.record("shim", "dw-mcp", {"path": str(shim), "target": str(target)})
    manifest.save()
    return f"added dw-mcp shim ({shim})"


def remove_mcp_shim(manifest: Manifest, platform: str = sys.platform) -> str:
    """Remove the shim setup recorded; nothing else."""
    if not manifest.has("shim", "dw-mcp"):
        return "no dw-mcp shim recorded"
    shim = _shim_path(platform)
    if platform == "win32":
        ok = shim.is_file()
    else:
        ok = shim.is_symlink()
    if os.path.lexists(shim) and not ok:
        return f"{shim} is not the shim setup made; left as it is"
    shim.unlink(missing_ok=True)
    manifest.forget("shim", "dw-mcp")
    manifest.save()
    return f"removed dw-mcp shim ({shim})"


def _listed(result) -> set[str] | None:
    """The `name`s a `claude plugin ... list` printed (lines like `❯ name`), or None."""
    if result.returncode != 0:
        return None
    out = result.stdout or ""
    names = {
        m.group(1)
        for m in re.finditer(r"^\s*[❯>]\s+(\S+)\s*$", out, flags=re.MULTILINE)
    }
    if names or not out.strip() or re.search(r"\bno\b", out, flags=re.IGNORECASE):
        return names
    return None


def _mcp_servers(runner: Runner) -> dict[str, str] | None:
    """User-visible MCP servers from `claude mcp list` as {name: command line}, or None."""
    result = runner(["claude", "mcp", "list"])
    if result.returncode != 0:
        return None
    servers: dict[str, str] = {}
    for line in (result.stdout or "").splitlines():
        m = re.match(r"^(\S[^:]*?):\s+(.*?)\s+-\s+\S", line)
        if m and not m.group(1).startswith("claude.ai "):
            servers[m.group(1)] = m.group(2)
    return servers


def _try(runner: Runner, args: list[str]) -> bool:
    return runner(args).returncode == 0


def register_code(manifest: Manifest, runner: Runner, version: str) -> str:
    marketplaces = _listed(runner(["claude", "plugin", "marketplace", "list"]))
    plugins = _listed(runner(["claude", "plugin", "list"]))
    servers = _mcp_servers(runner)
    notes: list[str] = [write_mcp_shim(manifest)]
    for cmd in code_commands(version, "@"):
        if cmd[1:4] == ["plugin", "marketplace", "add"]:
            kind, key, present = (
                "code-marketplace",
                MARKETPLACE_NAME,
                marketplaces is not None and MARKETPLACE_NAME in marketplaces,
            )
        elif cmd[1:3] == ["plugin", "install"]:
            kind, key, present = (
                "code-plugin",
                PLUGIN,
                plugins is not None and PLUGIN in plugins,
            )
        else:
            name = cmd[5]
            kind, key, present = (
                "code-mcp",
                name,
                servers is not None and name in servers,
            )
        if present:
            notes.append(f"{key} already present")
            continue
        result = runner(cmd)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise SetupError(f"`{' '.join(cmd[:4])}` failed: {detail}")
        manifest.record(kind, key, {})
        manifest.save()
        notes.append(f"added {key}")
    return "Claude Code: " + "; ".join(notes)


def unregister_code(manifest: Manifest, runner: Runner) -> str:
    removals = [
        ("code-plugin", ["claude", "plugin", "uninstall"]),
        ("code-marketplace", ["claude", "plugin", "marketplace", "remove"]),
        ("code-mcp", ["claude", "mcp", "remove", "--scope", "user"]),
    ]
    notes: list[str] = []
    for kind, prefix in removals:
        for entry in manifest.entries(kind):
            key = entry["key"]
            if _try(runner, [*prefix, key]):
                manifest.forget(kind, key)
                manifest.save()
                notes.append(f"removed {key}")
            else:
                notes.append(f"could not remove {key}; run `{' '.join(prefix)} {key}`")
    if manifest.has("shim", "dw-mcp"):
        notes.append(remove_mcp_shim(manifest))
    return "Claude Code: " + ("; ".join(notes) if notes else "nothing to remove")


def code_duplicates(runner: Runner) -> list[str]:
    """MCP servers, not our notes channel, that would collide with the plugin's dw tools.

    Any server named `dw` (even an HTTP one) or whose command is `dw-mcp`.
    """
    servers = _mcp_servers(runner) or {}
    found = []
    for name, line in servers.items():
        words = line.split()
        if name == _NOTES or "--notes-channel" in words:
            continue
        if name == _SERVER or (words and re.search(r"dw-mcp(\.exe)?$", words[0])):
            found.append(name)
    return found


def desktop_plugin_steps() -> str:
    return (
        "To add the dw plugin in Claude Desktop:\n"
        "  1. Open Customize > Plugins (or Settings > Plugins) and choose Add marketplace.\n"
        f"  2. Enter {MARKETPLACE_SOURCE}.\n"
        "  3. Install the dw plugin from that marketplace."
    )
