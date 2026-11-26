"""The .mcpb bundle: dw_mcp alone, versioned and floored from the root pyproject."""

import ast
import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYPROJECT_VERSION = tomllib.loads((REPO / "pyproject.toml").read_text())["project"][
    "version"
]


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
    assert "mcp>=2.3.0" in deps
    assert {d.split(">")[0] for d in deps} == {"mcp", "httpx", "anyio", "packaging"}
    assert (stage / "dw_mcp" / "__main__.py").is_file()


def test_the_identifier_contains_mcp_and_points_at_the_release():
    rendered = _script().render_server_json("0.23.0", "f" * 64)
    pkg = rendered["packages"][0]
    assert "mcp" in pkg["identifier"]
    assert pkg["identifier"].endswith("/v0.23.0/dw-mcp-0.23.0.mcpb")
    assert pkg["fileSha256"] == "f" * 64
    assert rendered["name"] == "io.github.dkackman/dw"
