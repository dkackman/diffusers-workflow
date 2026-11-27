"""The .mcpb bundle: dw_mcp alone, versioned and floored from the root pyproject."""

import ast
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PYPROJECT_VERSION = tomllib.loads((REPO / "pyproject.toml").read_text())["project"][
    "version"
]


def _mcp_floor():
    """The mcp requirement as the root pyproject's mcp extra declares it."""
    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    (floor,) = [r for r in extras["mcp"] if r.startswith("mcp")]
    return floor


def test_name_is_the_package_of_a_full_requirement():
    script = _script()
    full = "Foo_Bar[extra1,extra2]>=1.0,<2; python_version >= '3.10'"
    assert script._name(full) == "Foo_Bar"
    assert script._name("uvicorn[standard]>=0.54.0") == "uvicorn"
    assert script._name("mcp ~= 2.3") == "mcp"


def _script():
    spec = importlib.util.spec_from_file_location(
        "build_mcpb", REPO / "scripts" / "build_mcpb.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dw_mcp_imports_nothing_from_dw():
    for path in (REPO / "dw_mcp").glob("*.py"):
        tree = ast.parse(path.read_text())
        names = [
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
        ]
        names += [
            n.module or ""
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.level == 0
        ]
        assert not any(m == "dw" or m.startswith("dw.") for m in names), path.name


def test_the_stage_carries_the_engine_version_and_the_floors(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "scripts/build_mcpb.py",
            "--stage-only",
            "--no-lock",
            "--out",
            str(tmp_path),
        ],
        check=True,
        cwd=REPO,
    )
    stage = tmp_path / "stage"
    manifest = json.loads((stage / "manifest.json").read_text())
    assert manifest["version"] == PYPROJECT_VERSION
    assert manifest["server"]["type"] == "uv"
    env = manifest["server"]["mcp_config"]["env"]
    assert env == {
        "DW_MCP_URL": "${user_config.url}",
        "DW_API_TOKEN": "${user_config.token}",
    }
    assert manifest["user_config"]["token"]["sensitive"] is True
    deps = tomllib.loads((stage / "pyproject.toml").read_text())["project"][
        "dependencies"
    ]
    assert _mcp_floor() in deps
    assert {d.split(">")[0] for d in deps} == {"mcp", "httpx", "anyio", "packaging"}
    assert (stage / "dw_mcp" / "__main__.py").is_file()
    assert "--frozen" in manifest["server"]["mcp_config"]["args"]


def test_the_stage_locks_with_uv_in_the_stage(tmp_path, monkeypatch):
    script = _script()
    calls = []
    monkeypatch.setattr(script.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    root = script.stage(tmp_path)
    assert calls == [((["uv", "lock"],), {"cwd": root, "check": True})]


def test_the_stage_without_lock_runs_nothing(tmp_path, monkeypatch):
    script = _script()
    calls = []
    monkeypatch.setattr(script.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    script.stage(tmp_path, lock=False)
    assert calls == []


def test_pack_and_unpack_name_the_pinned_packer(tmp_path, monkeypatch):
    """The root package.json pins mcpb exactly, its lock agrees, and pack and
    unpack run the binary `npm ci` installs: no `npx -y` resolve at build time."""
    package = json.loads((REPO / "package.json").read_text())
    assert package["private"] is True
    assert "scripts" not in package
    pin = package["devDependencies"]["@anthropic-ai/mcpb"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", pin), "an exact version, no range"
    lock = json.loads((REPO / "package-lock.json").read_text())
    assert lock["packages"][""]["devDependencies"] == package["devDependencies"]
    locked = lock["packages"]["node_modules/@anthropic-ai/mcpb"]
    assert locked["version"] == pin
    assert locked["integrity"].startswith("sha512-")
    # Every transitive package is locked to a resolved tarball and hash
    for name, entry in lock["packages"].items():
        if name:
            assert entry.get("integrity"), name

    script = _script()
    binary = tmp_path / "node_modules" / ".bin" / "mcpb"
    binary.parent.mkdir(parents=True)
    binary.write_text("")
    monkeypatch.setattr(script, "MCPB_BIN", binary)
    for verb in ("pack", "unpack"):
        assert script.mcpb_command(verb, "a", "b") == [str(binary), verb, "a", "b"]
    assert _script().MCPB_BIN == REPO / "node_modules" / ".bin" / "mcpb"
    source = (REPO / "scripts" / "build_mcpb.py").read_text()
    assert "npx" not in source


def test_a_missing_packer_says_to_run_npm_ci(tmp_path, monkeypatch):
    script = _script()
    monkeypatch.setattr(script, "MCPB_BIN", tmp_path / "node_modules" / ".bin" / "mcpb")
    monkeypatch.setattr(script, "REPO", tmp_path)
    with pytest.raises(SystemExit, match="run `npm ci` at the repo root"):
        script.mcpb_command("pack", "a", "b")


def test_the_wheel_job_installs_the_root_lock():
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    wheel = ci[ci.index("\n  wheel:") : ci.index("\n  release:")]
    assert re.search(
        r"cache-dependency-path: \|\n\s+ui/package-lock.json\n\s+package-lock.json\n",
        wheel,
    )
    assert re.search(r"\n      - run: npm ci\n", wheel)
    assert wheel.index("- run: npm ci") < wheel.index("scripts/build_mcpb.py --out")


def test_ci_pins_the_publisher_and_verifies_it():
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    assert "releases/latest" not in ci
    assert re.search(r"tag=v\d+\.\d+\.\d+", ci)
    assert re.search(r"sha=[0-9a-f]{64}", ci)
    assert "sha256sum -c" in ci
    assert "npx -y @anthropic-ai/mcpb" not in ci


def test_ci_pins_the_uv_version():
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    blocks = re.findall(
        r"uses: astral-sh/setup-uv@\S+\n(?:\s+with:\n((?:\s+\S.*\n)*))?", ci
    )
    assert blocks
    for with_block in blocks:
        assert re.search(r'version: "?\d+\.\d+\.\d+"?', with_block or "")


def test_the_identifier_contains_mcp_and_points_at_the_release():
    rendered = _script().render_server_json("0.23.0", "f" * 64)
    pkg = rendered["packages"][0]
    assert "mcp" in pkg["identifier"]
    assert pkg["identifier"].endswith("/v0.23.0/dw-mcp-0.23.0.mcpb")
    assert pkg["fileSha256"] == "f" * 64
    assert rendered["name"] == "io.github.dkackman/dw"
