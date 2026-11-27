#!/usr/bin/env python3
"""Stamp a release's version into copies of the bootstrap scripts.

    python scripts/stamp_bootstrap.py 0.40.0 dist/bootstrap.sh dist/bootstrap.ps1

The release job runs this on the copies it attaches, so a script fetched from
`releases/download/v<X>/` installs dw <X> rather than the newest release.
`--version` / `-Version` and DW_BOOTSTRAP_WHEEL still override the stamp.

Each script's placeholder line must appear exactly once, and its stamped form
exactly once afterwards; anything else fails, so a renamed placeholder cannot
ship an unstamped script.
"""

import re
import sys
from pathlib import Path

# The default-version line of each script, as the repo copy holds it, and the
# stamped form; keyed by the script's suffix
PLACEHOLDERS = {
    ".sh": ('  VERSION=""', '  VERSION="{version}"'),
    ".ps1": ('    [string]$Version = "",', '    [string]$Version = "{version}",'),
}
VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$")


def stamp_text(text, suffix, version):
    """Return text with its placeholder line stamped; ValueError unless exactly one."""
    if not VERSION_PATTERN.match(version):
        raise ValueError(f"{version!r} is not a semver version")
    placeholder, stamped = PLACEHOLDERS[suffix]
    stamped = stamped.format(version=version)
    lines = text.split("\n")
    found = lines.count(placeholder)
    if found != 1:
        raise ValueError(f"expected 1 line {placeholder!r}, found {found}")
    lines = [stamped if line == placeholder else line for line in lines]
    if lines.count(stamped) != 1:
        raise ValueError(f"expected 1 stamped line {stamped!r} after stamping")
    return "\n".join(lines)


def stamp_file(path, version):
    path = Path(path)
    if path.suffix not in PLACEHOLDERS:
        raise ValueError(f"{path}: not a bootstrap script (.sh or .ps1)")
    path.write_text(stamp_text(path.read_text(), path.suffix, version))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    version, *paths = argv
    suffixes = sorted(Path(p).suffix for p in paths)
    if suffixes != sorted(PLACEHOLDERS):
        print("name exactly one bootstrap.sh and one bootstrap.ps1", file=sys.stderr)
        return 2
    try:
        for path in paths:
            stamp_file(path, version)
    except ValueError as e:
        print(f"stamp_bootstrap: {e}", file=sys.stderr)
        return 1
    for path in paths:
        print(f"stamped {path} with {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
