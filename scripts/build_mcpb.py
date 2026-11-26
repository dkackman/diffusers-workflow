#!/usr/bin/env python3
"""Build the dw-mcp `.mcpb` bundle (dw_mcp alone, run by uv) and its registry entry.

    python scripts/build_mcpb.py --out dist/           # stage, pack
    python scripts/build_mcpb.py --out dist/ --stage-only
    python scripts/build_mcpb.py --unpack dist/dw-mcp-<v>.mcpb <dir>
    python scripts/build_mcpb.py --server-json dist/dw-mcp-<v>.mcpb   # print server.json

The bundle's dependency floors are read from the root pyproject.toml, so they
cannot drift from the engine's.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = REPO / "packaging" / "mcpb"
# dw_mcp imports anyio directly; the root pyproject only gets it through mcp
ANYIO_FLOOR = "anyio>=4.0"
# The one owner of the packer's version: pack and unpack both use it
MCPB_PACKAGE = "@anthropic-ai/mcpb@2.1.2"
BUNDLE_DEPENDENCIES = ("httpx", "anyio", "packaging")


def _pyproject():
    return tomllib.loads((REPO / "pyproject.toml").read_text())["project"]


def _name(requirement):
    return requirement.split(";")[0].strip().split(">")[0].split("<")[0].split("=")[0]


def bundle_dependencies(project=None):
    project = project or _pyproject()
    pool = list(project["dependencies"]) + list(project["optional-dependencies"]["mcp"])
    found = {_name(r): r for r in pool}
    found.setdefault("anyio", ANYIO_FLOOR)
    return [found["mcp"]] + [found[n] for n in BUNDLE_DEPENDENCIES]


def _fill(template, **values):
    text = (TEMPLATES / template).read_text()
    for key, value in values.items():
        text = text.replace(f"@{key.upper()}@", value)
    return json.loads(text)


def render_manifest(version):
    return _fill("manifest.json.in", version=version)


def render_server_json(version, sha):
    return _fill("server.json.in", version=version, sha=sha)


def mcpb_command(*args):
    return ["npx", "-y", MCPB_PACKAGE, *map(str, args)]


def stage(out, project=None):
    project = project or _pyproject()
    version = project["version"]
    root = Path(out) / "stage"
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(
        REPO / "dw_mcp", root / "dw_mcp", ignore=shutil.ignore_patterns("__pycache__")
    )
    (root / "manifest.json").write_text(
        json.dumps(render_manifest(version), indent=2) + "\n"
    )
    deps = "".join(f'    "{d}",\n' for d in bundle_dependencies(project))
    (root / "pyproject.toml").write_text(
        "[project]\n"
        'name = "dw-mcp"\n'
        f'version = "{version}"\n'
        f'requires-python = "{project["requires-python"]}"\n'
        f"dependencies = [\n{deps}]\n"
    )
    shutil.copy(REPO / "LICENSE", root / "LICENSE")
    # Ship the resolution: the launch is `uv run --frozen`, so what runs is
    # what was locked here, not whatever the floors resolve to later
    subprocess.run(["uv", "lock"], cwd=root, check=True)
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="dist", help="output directory")
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument(
        "--unpack", nargs=2, metavar=("BUNDLE", "DIR"), help="unpack a bundle"
    )
    parser.add_argument(
        "--server-json", metavar="BUNDLE", help="print server.json for a built bundle"
    )
    args = parser.parse_args(argv)
    version = _pyproject()["version"]
    if args.server_json:
        sha = hashlib.sha256(Path(args.server_json).read_bytes()).hexdigest()
        print(json.dumps(render_server_json(version, sha), indent=2))
        return 0
    if args.unpack:
        subprocess.run(mcpb_command("unpack", *args.unpack), check=True)
        return 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = stage(out)
    if args.stage_only:
        return 0
    bundle = out / f"dw-mcp-{version}.mcpb"
    subprocess.run(
        mcpb_command("pack", root, bundle),
        check=True,
    )
    shutil.rmtree(root)
    print(bundle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
