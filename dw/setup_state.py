"""What `dw setup` knows and has changed: the install record, the setup manifest and a safe JSON merge.

The install record (`<root>/install.json`) says how the venv was built - the torch
index, the dw version and the diffusers commit - so an update rebuilds it the same way.
The setup manifest (`setup-manifest.json` in the settings directory) lists every
outside change setup made (a service, a Claude registration, linger), so `dw setup
--uninstall` undoes exactly those and nothing else. `merge_json_file` is how setup
edits a config file it does not own: it merges, backs the original up to
`<file>.dw-backup` before the first change, and refuses a file it cannot parse rather
than overwrite something the user wrote.
"""

import json
import os
import stat
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from dw.settings import get_settings_dir

_APP = "diffusers-workflow"
_KINDS = (
    "service",
    "desktop",
    "code-plugin",
    "code-marketplace",
    "code-mcp",
    "shim",
    "linger",
)


class SetupError(Exception):
    """Setup cannot continue; the message names the file or step at fault."""


def install_root() -> Path:
    """Where dw is installed: `DW_INSTALL_ROOT`, else the per-OS user data directory."""
    override = os.environ.get("DW_INSTALL_ROOT")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / _APP
    return Path.home() / ".local" / "share" / _APP


def shim_dir() -> Path:
    """The directory of launcher shims. The same path on every OS, because a Claude
    plugin's `.mcp.json` names it through `${HOME}`."""
    return Path.home() / ".local" / "share" / _APP / "bin"


def venv_bin(name: str, root: Path | None = None, platform: str | None = None) -> Path:
    """The executable `name` inside the install's venv."""
    venv = (root if root is not None else install_root()) / "venv"
    if (platform or sys.platform) == "win32":
        return venv / "Scripts" / f"{name}.exe"
    return venv / "bin" / name


def beside_python(name: str, platform: str | None = None) -> Path:
    """The executable `name` beside the interpreter running now.

    Everything setup registers (the service's dw-serve, Claude's dw-mcp, the
    Update button's `dw setup`) names this, not the install root's venv: a
    checkout running `venv/bin/dw setup` then registers its own venv, and an
    installed copy's interpreter is the install root's venv anyway.
    """
    suffix = ".exe" if (platform or sys.platform) == "win32" else ""
    return Path(sys.executable).parent / f"{name}{suffix}"


def _read_json(path: Path, what: str) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as e:
        raise SetupError(f"cannot read {what} {path}: {e}") from e
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise SetupError(f"{what} {path} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise SetupError(f"{what} {path} is not a JSON object")
    return data


def read_install_record(root: Path | None = None) -> dict | None:
    """The install record, or None when there is none."""
    path = (root if root is not None else install_root()) / "install.json"
    if not path.exists():
        return None
    return _read_json(path, "install record")


class Manifest:
    """The changes setup made outside the install root, by kind and key."""

    def __init__(self, path: Path, entries: list[dict]):
        self._path = path
        self._entries = entries

    @classmethod
    def load(cls, path: Path | None = None) -> "Manifest":
        path = path if path is not None else get_settings_dir() / "setup-manifest.json"
        data = _read_json(path, "setup manifest")
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            raise SetupError(f"setup manifest {path} has no entries list")
        for entry in entries:
            if not (
                isinstance(entry, dict)
                and isinstance(entry.get("kind"), str)
                and isinstance(entry.get("key"), str)
            ):
                raise SetupError(
                    f"setup manifest {path} has a malformed entry: {entry!r}"
                )
        return cls(path, entries)

    def record(self, kind: str, key: str, detail: dict) -> None:
        if kind not in _KINDS:
            raise SetupError(f"unknown manifest kind {kind!r}")
        self.forget(kind, key)
        self._entries.append({"kind": kind, "key": key, "detail": detail})

    def forget(self, kind: str, key: str) -> None:
        self._entries = [
            e for e in self._entries if not (e["kind"] == kind and e["key"] == key)
        ]

    def entries(self, kind: str | None = None) -> list[dict]:
        return [e for e in self._entries if kind is None or e["kind"] == kind]

    def has(self, kind: str, key: str) -> bool:
        return any(e["kind"] == kind and e["key"] == key for e in self._entries)

    def save(self) -> None:
        _write_atomic(self._path, {"entries": self._entries})


def _write_atomic(path: Path, data: dict) -> None:
    """Write beside the target and `os.replace` it, so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        if path.exists():  # keep the target's permission bits (mkstemp makes 0600)
            os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def merge_json_file(path: Path, update: Callable[[dict], dict]) -> bool:
    """Apply `update` to the JSON object in `path`; return whether the file changed.

    A missing or empty file is `{}`. The original is copied to `<path>.dw-backup`
    before the first change and the backup is never overwritten. A file that does
    not parse raises `SetupError` and is left as it was.
    """
    path = Path(os.path.realpath(path))  # write through a symlink, keep it a link
    current = _read_json(path, "config file")
    updated = update(json.loads(json.dumps(current)))
    if updated == current:
        return False
    backup = path.with_name(path.name + ".dw-backup")
    if (
        path.exists()
        and path.read_text(encoding="utf-8").strip()
        and not backup.exists()
    ):
        backup.write_bytes(path.read_bytes())
    _write_atomic(path, updated)
    return True
