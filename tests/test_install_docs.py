"""Pins for the cold-install path (#1525 stage A): the README and the Docker and
remote guides carry the clone URL, install.sh builds the web UI that dw-serve
serves from ui/dist, and the README's Python range is the one install.sh checks
and pyproject.toml declares.

The UI-build tests run install.sh's own web UI block under bash with a fake
npm, so they test the script's logic rather than a grep of its text."""

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
INSTALL_SH = REPO / "install.sh"
DOCKER_MD = REPO / "docs" / "DOCKER.md"
REMOTE_MD = REPO / "docs" / "REMOTE.md"
PYPROJECT = REPO / "pyproject.toml"

CLONE = "git clone https://github.com/dkackman/diffusers-workflow.git"
BASH = shutil.which("bash")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("doc", [README, DOCKER_MD, REMOTE_MD], ids=lambda p: p.name)
def test_doc_carries_the_clone_url(doc):
    text = _text(doc)
    assert CLONE in text, f"{doc.name} has no `{CLONE}` step"
    assert "git clone …" not in text


@pytest.mark.parametrize("doc", [README, DOCKER_MD], ids=lambda p: p.name)
def test_clone_is_followed_by_cd_into_the_checkout(doc):
    text = _text(doc)
    after = text[text.index(CLONE) + len(CLONE) :]
    assert after.lstrip().startswith("cd diffusers-workflow")


def test_docker_guide_clones_before_init_env():
    text = _text(DOCKER_MD)
    assert text.index(CLONE) < text.index("bash docker/init-env.sh")


def test_docker_guide_has_a_same_host_mcp_line_with_a_placeholder_token():
    text = _text(DOCKER_MD)
    assert re.search(
        r"claude mcp add [^\n]*--transport http dw http://localhost:8765/mcp \\\n"
        r'\s+--header "Authorization: Bearer <token>"',
        text,
    )
    assert "login-url" in text[text.index("Connect Claude Code on the same host") :]


def test_readme_registers_dw_mcp_for_every_directory():
    text = _text(README)
    assert 'claude mcp add --scope user dw -- "$(pwd)/venv/bin/dw-mcp"' in text
    assert "inside Claude Code" in text


def _ui_block() -> str:
    """install.sh's web UI step: from its comment to the end of its if/else."""
    text = _text(INSTALL_SH)
    match = re.search(r"^# Web UI.*?^fi$", text, re.M | re.S)
    assert match, "install.sh has no '# Web UI' block"
    return match.group(0)


def _run_ui_block(tmp_path: Path, with_npm: bool) -> subprocess.CompletedProcess:
    (tmp_path / "ui").mkdir()
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    if with_npm:
        npm = fakebin / "npm"
        npm.write_text(
            # PATH holds only this directory, so the script uses builtins only
            '#!/bin/sh\necho "${PWD##*/} npm $*" >> "$NPM_LOG"\n',
            encoding="utf-8",
        )
        npm.chmod(0o755)
    script = tmp_path / "ui-block.sh"
    script.write_text("set -o errexit\n" + _ui_block() + "\n", encoding="utf-8")
    return subprocess.run(
        [BASH, str(script)],
        cwd=tmp_path,
        env={"PATH": str(fakebin), "NPM_LOG": str(tmp_path / "npm.log")},
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_install_sh_builds_the_ui_when_npm_is_present(tmp_path):
    _run_ui_block(tmp_path, with_npm=True)
    calls = _text(tmp_path / "npm.log").splitlines()
    assert calls == ["ui npm ci", "ui npm run build"]


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_install_sh_prints_the_build_commands_without_npm(tmp_path):
    result = _run_ui_block(tmp_path, with_npm=False)
    assert "cd ui && npm ci && npm run build" in result.stdout
    assert not (tmp_path / "npm.log").exists()
    # The old message said `npm install` builds the UI, which it does not.
    assert "npm install" not in result.stdout


def _range(versions) -> tuple[int, int]:
    minors = sorted({int(v) for v in versions})
    assert minors == list(range(minors[0], minors[-1] + 1)), minors
    return minors[0], minors[-1]


def test_python_range_agrees_across_readme_install_sh_and_pyproject():
    classifiers = tomllib.loads(_text(PYPROJECT))["project"]["classifiers"]
    declared = _range(
        m.group(1)
        for c in classifiers
        if (m := re.fullmatch(r"Programming Language :: Python :: 3\.(\d+)", c))
    )

    install = _text(INSTALL_SH)
    loop = re.search(r"for V in ([^;]+); do", install).group(1).split()
    searched = _range(v.removeprefix("3.").removeprefix("3") for v in loop)
    low = int(re.search(r'PYTHON_MINOR_VER" -lt "(\d+)"', install).group(1))
    above = int(re.search(r'PYTHON_MINOR_VER" -ge "(\d+)"', install).group(1))
    checked = (low, above - 1)

    readme_ranges = {
        (int(a), int(b))
        for a, b in re.findall(r"Python 3\.(\d+)-3\.(\d+)", _text(README))
    }

    assert searched == checked == declared
    assert readme_ranges == {declared}
