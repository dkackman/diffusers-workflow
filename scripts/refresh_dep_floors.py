"""Rewrite pyproject.toml's version floors from the active environment.

The policy the floors follow: each dependency's minimum is the version the
project is actually developed and tested against. Run this from the venv
after upgrading packages, review the diff, commit.

Floors only move forward - a floor newer than the installed version is
left alone (the venv is behind, not the pyproject). Specifiers other than
>= (markers, extras, exact pins) keep everything but the floor.

    python scripts/refresh_dep_floors.py
"""

import re
import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version, InvalidVersion

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def refreshed(requirement_text):
    """The requirement with its >= floor raised to the installed version,
    or None for no change."""
    requirement = Requirement(requirement_text)
    floors = [s for s in requirement.specifier if s.operator == ">="]
    if len(floors) != 1:
        return None
    try:
        installed = Version(version(requirement.name))
    except (PackageNotFoundError, InvalidVersion):
        return None
    if installed.is_prerelease or installed <= Version(floors[0].version):
        # A git/dev install is not a floor; neither is a venv that's behind
        return None
    return requirement_text.replace(
        f">={floors[0].version}", f">={installed.public}", 1
    )


def main():
    text = PYPROJECT.read_text(encoding="utf-8")
    changed = []

    def replace(match):
        old = match.group(1)
        new = refreshed(old)
        if new is None:
            return match.group(0)
        changed.append(f"  {old}  ->  {new}")
        return f'"{new}"'

    # Every quoted requirement that carries a >= floor, wherever it sits
    # (dependencies or an extra)
    updated = re.sub(r'"([A-Za-z0-9_.\[\],-]+>=[^"]+)"', replace, text)

    if not changed:
        print("All floors already match the environment")
        return
    PYPROJECT.write_text(updated, encoding="utf-8")
    print("\n".join(changed))
    print(f"\n{len(changed)} floor(s) raised - review with: git diff pyproject.toml")


if __name__ == "__main__":
    sys.exit(main())
