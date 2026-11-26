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

from packaging.version import InvalidVersion, Version

from dw.setup_service import Runner
from dw.setup_state import (
    Manifest,
    SetupError,
    beside_python,
    claude_executable,
    merge_json_file,
    shim_dir,
)

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
    return str(beside_python("dw-mcp"))


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
    return claude_executable()


_PRE_LABELS = {"a": "alpha", "b": "beta", "rc": "rc"}


def release_tag(version: str) -> str:
    """The repo's git tag for a dw version.

    Tags are `v` + the version as pyproject.toml spells it (`v0.23.0-alpha.1`),
    but a wheel's metadata normalizes it (`0.23.0a1`): map the PEP 440
    pre-release back. A dev or local part has no tag of its own and is dropped.
    """
    try:
        parsed = Version(version)
    except InvalidVersion:
        return f"v{version}"
    tag = "v" + ".".join(str(part) for part in parsed.release)
    if parsed.pre:
        label, number = parsed.pre
        tag += f"-{_PRE_LABELS[label]}.{number}"
    return tag


def versions_differ(a: str, b: str) -> bool:
    """Whether two version strings name different versions (PEP 440)."""
    try:
        return Version(a) != Version(b)
    except InvalidVersion:
        return a != b


def code_commands(version: str, ref_syntax: str | None) -> list[list[str]]:
    """The `claude` commands that register dw with Claude Code, in order."""
    source = MARKETPLACE_SOURCE
    if ref_syntax:
        source = f"{source}{ref_syntax}{release_tag(version)}"
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


def _locked(shim: Path, e: OSError) -> SetupError:
    return SetupError(
        f"cannot replace {shim}: {e}. Close Claude Code and Claude Desktop "
        "(they run dw-mcp) and run dw setup again"
    )


def _points_to(shim: Path) -> str | None:
    try:
        return os.readlink(shim)
    except OSError:
        return None


def _our_shim(shim: Path, detail: dict, platform: str) -> bool:
    """Whether the thing at `shim` is still the shim setup made: on Windows a
    file, elsewhere a symlink that points where setup pointed it (any symlink,
    for a record that predates keeping the target)."""
    if platform == "win32":
        return shim.is_file() and not shim.is_symlink()
    if not shim.is_symlink():
        return False
    wrote = detail.get("target")
    return wrote is None or _points_to(shim) == wrote


def _same_bytes(a: Path, b: Path) -> bool:
    try:
        return a.stat().st_size == b.stat().st_size and a.read_bytes() == b.read_bytes()
    except OSError:
        return False


def write_mcp_shim(manifest: Manifest, platform: str = sys.platform) -> str:
    """Make `shim_dir()/dw-mcp` reach the `dw-mcp` beside the running interpreter.

    The plugin's `.mcp.json` names that stable path, so one plugin install carries the
    tool server wherever dw is installed - Claude Code and Cowork both start it. POSIX
    gets a symlink, Windows a copy (skipped when the bytes already match, since a
    running dw-mcp.exe cannot be replaced). Anything at that path setup did not
    record, or that has changed since, is never replaced.
    """
    target = beside_python("dw-mcp", platform)
    if not target.exists():
        raise SetupError(f"cannot link the dw-mcp shim: {target} does not exist")
    shim = _shim_path(platform)
    recorded = next((e for e in manifest.entries("shim") if e["key"] == "dw-mcp"), None)
    detail = {"path": str(shim), "target": str(target)}
    if os.path.lexists(shim):
        if recorded is None:
            where = _points_to(shim)
            if where is not None and Path(where) == target:
                return f"dw-mcp shim already in place ({shim})"
            what = (
                f"it points to {where}" if where is not None else "not a link to dw-mcp"
            )
            return (
                f"warning: {shim} already exists and setup did not make it "
                f"({what}); left as it is, and the plugin's dw server runs it"
            )
        if not _our_shim(shim, recorded.get("detail") or {}, platform):
            return f"{shim} is not the shim setup made; left as it is"
        if platform == "win32" and _same_bytes(shim, target):
            manifest.record("shim", "dw-mcp", detail)
            manifest.save()
            return f"dw-mcp shim already in place ({shim})"
        if platform != "win32" and _points_to(shim) == str(target):
            return f"dw-mcp shim already in place ({shim})"
        try:
            shim.unlink()
        except OSError as e:
            raise _locked(shim, e) from e
    shim.parent.mkdir(parents=True, exist_ok=True)
    try:
        if platform == "win32":
            shutil.copy2(target, shim)
        else:
            shim.symlink_to(target)
    except OSError as e:
        raise _locked(shim, e) from e
    manifest.record("shim", "dw-mcp", detail)
    manifest.save()
    return f"added dw-mcp shim ({shim})"


def remove_mcp_shim(manifest: Manifest, platform: str = sys.platform) -> str:
    """Remove the shim setup recorded, if it is still the one setup made."""
    recorded = next((e for e in manifest.entries("shim") if e["key"] == "dw-mcp"), None)
    if recorded is None:
        return "no dw-mcp shim recorded"
    shim = _shim_path(platform)
    if os.path.lexists(shim):
        if not _our_shim(shim, recorded.get("detail") or {}, platform):
            manifest.forget("shim", "dw-mcp")
            manifest.save()
            return f"{shim} is not the shim setup made; left as it is"
        try:
            shim.unlink()
        except OSError as e:
            raise _locked(shim, e) from e
    manifest.forget("shim", "dw-mcp")
    manifest.save()
    return f"removed dw-mcp shim ({shim})"


def _first_section(out: str) -> list[str]:
    """The lines of a listing's first section: its first non-blank line and the
    indented lines under it, up to the next unindented one (`From claude.ai:`
    after `Configured marketplaces:`)."""
    lines = out.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    section = lines[:1]
    for line in lines[1:]:
        if line.strip() and not line[0].isspace():
            break
        section.append(line)
    return section


def _listed(result) -> set[str] | None:
    """The `name`s a `claude plugin ... list` printed in its first section
    (lines like `❯ name`), or None. A section with no names is empty only when
    it says so on a line of its own (`No plugins installed`)."""
    if result.returncode != 0:
        return None
    out = result.stdout or ""
    section = _first_section(out)
    names = set()
    for line in section:
        m = re.match(r"^\s*[❯>]\s+(\S+)\s*$", line)
        if m:
            names.add(m.group(1))
    if names or not out.strip():
        return names
    if any(re.match(r"^\s*no\s", line, flags=re.IGNORECASE) for line in section):
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


def _marketplace_source(result) -> str | None:
    """The source `claude plugin marketplace list` shows for dw's marketplace
    (`Source: GitHub (owner/repo@ref)`), or None."""
    if result.returncode != 0:
        return None
    blocks = re.split(r"^\s*[❯>]\s+", result.stdout or "", flags=re.MULTILINE)
    for block in blocks:
        lines = block.splitlines()
        if lines and lines[0].strip() == MARKETPLACE_NAME:
            found = re.search(r"Source:\s*\S+\s*\(([^)]+)\)", block)
            return found.group(1) if found else None
    return None


def _failure(cmd: list[str], result) -> SetupError:
    detail = (result.stderr or result.stdout or "").strip()
    return SetupError(f"`{' '.join(cmd[:4])}` failed: {detail}")


def _add_marketplace(runner: Runner, target: str, notes: list[str]) -> str:
    """Add the marketplace at `target`, else at the default branch (a version
    with no tag yet: --status then reports the plugin version mismatch)."""
    cmd = ["claude", "plugin", "marketplace", "add", target]
    result = runner(cmd)
    if result.returncode == 0:
        return target
    fallback = ["claude", "plugin", "marketplace", "add", MARKETPLACE_SOURCE]
    if runner(fallback).returncode != 0:
        raise _failure(cmd, result)
    notes.append(
        f"{target} could not be added, so the marketplace follows the default "
        "branch; dw setup --status reports the plugin version"
    )
    return MARKETPLACE_SOURCE


def _register_marketplace(
    manifest: Manifest, runner: Runner, target: str, notes: list[str]
) -> bool:
    """Add the dw marketplace pinned at `target`, or move setup's own pin to
    it; returns whether an existing marketplace was moved."""
    listing = runner(["claude", "plugin", "marketplace", "list"])
    names = _listed(listing)
    present = names is not None and MARKETPLACE_NAME in names
    recorded = next(iter(manifest.entries("code-marketplace")), None)
    moved = False
    if present:
        current = _marketplace_source(listing)
        if recorded is None:
            pinned = f" (pinned at {current})" if current and current != target else ""
            notes.append(f"{MARKETPLACE_NAME} already present{pinned}")
            return False
        detail = recorded.get("detail") or {}
        current = current or detail.get("source")
        if current is None or target in (current, detail.get("wanted")):
            notes.append(f"{MARKETPLACE_NAME} already present")
            return False
        remove = ["claude", "plugin", "marketplace", "remove", MARKETPLACE_NAME]
        result = runner(remove)
        if result.returncode != 0:
            raise _failure(remove, result)
        notes.append(f"moved {MARKETPLACE_NAME} from {current}")
        moved = True
    source = _add_marketplace(runner, target, notes)
    manifest.record(
        "code-marketplace", MARKETPLACE_NAME, {"source": source, "wanted": target}
    )
    manifest.save()
    if not moved:
        notes.append(f"added {MARKETPLACE_NAME}")
    return moved


def _register_plugin(
    manifest: Manifest, runner: Runner, moved: bool, notes: list[str]
) -> None:
    """Install the plugin; reinstall it when its marketplace moved."""
    plugins = _listed(runner(["claude", "plugin", "list"]))
    present = plugins is not None and PLUGIN in plugins
    if present and not moved:
        notes.append(f"{PLUGIN} already present")
        return
    if present:
        runner(["claude", "plugin", "uninstall", PLUGIN])  # may be gone already
    cmd = ["claude", "plugin", "install", PLUGIN]
    result = runner(cmd)
    if result.returncode != 0:
        raise _failure(cmd, result)
    if not present or manifest.has("code-plugin", PLUGIN):
        manifest.record("code-plugin", PLUGIN, {})
        manifest.save()
    notes.append(f"{'reinstalled' if present else 'added'} {PLUGIN}")


def register_code(
    manifest: Manifest,
    runner: Runner,
    version: str,
    servers: dict[str, str] | None = None,
) -> str:
    """Register the dw marketplace (pinned at this version's tag), the dw
    plugin and the dw-notes channel. A re-run at a new version moves setup's
    own marketplace pin and reinstalls the plugin from it. `servers` is the
    caller's `claude mcp list` (`_mcp_servers`), so it is read once; None
    reads it here."""
    commands = code_commands(version, "@")
    notes: list[str] = []
    moved = _register_marketplace(manifest, runner, commands[0][-1], notes)
    _register_plugin(manifest, runner, moved, notes)
    if servers is None:
        servers = _mcp_servers(runner)
    if servers is not None and _NOTES in servers:
        notes.append(f"{_NOTES} already present")
    else:
        result = runner(commands[2])
        if result.returncode != 0:
            raise _failure(commands[2], result)
        manifest.record("code-mcp", _NOTES, {})
        manifest.save()
        notes.append(f"added {_NOTES}")
    return "Claude Code: " + "; ".join(notes)


_GONE = re.compile(r"not (?:found|installed)|does not exist|no such", re.I)


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
            result = runner([*prefix, key])
            output = f"{result.stdout or ''} {result.stderr or ''}"
            if result.returncode == 0 or _GONE.search(output):
                # removed by the person already: nothing left to undo
                manifest.forget(kind, key)
                manifest.save()
                notes.append(
                    f"removed {key}"
                    if result.returncode == 0
                    else f"{key} was already removed"
                )
            else:
                notes.append(f"could not remove {key}; run `{' '.join(prefix)} {key}`")
    return "Claude Code: " + ("; ".join(notes) if notes else "nothing to remove")


def code_duplicates(servers: dict[str, str] | None) -> list[str]:
    """MCP servers, not our notes channel, that would collide with the plugin's dw tools.

    Any server named `dw` (even an HTTP one) or whose command is `dw-mcp`.
    `servers` is `_mcp_servers()`'s listing (None when it could not be read).
    A command path may hold spaces, so the executable is the line up to the
    first argument that starts with `-`, or the whole line.
    """
    found = []
    for name, line in (servers or {}).items():
        words = line.split()
        if name == _NOTES or "--notes-channel" in words:
            continue
        executable = re.split(r"\s+-", line, maxsplit=1)[0].strip()
        if name == _SERVER or re.search(r"(?:^|[/\\])dw-mcp(\.exe)?$", executable):
            found.append(name)
    return found


def desktop_plugin_steps() -> str:
    return (
        "To add the dw plugin in Claude Desktop:\n"
        "  1. Open Customize > Plugins (or Settings > Plugins) and choose Add marketplace.\n"
        f"  2. Enter {MARKETPLACE_SOURCE}.\n"
        "  3. Install the dw plugin from that marketplace."
    )
