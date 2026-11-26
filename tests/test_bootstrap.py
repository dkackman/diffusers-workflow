"""Pins for the one-line bootstrap scripts (dw/install_scripts/).

bootstrap.sh runs under bash with DW_BOOTSTRAP_DRY_RUN=1 and fake `uname`,
`nvidia-smi`, `uv` and `curl` first on PATH, so the tests read the commands it
would run and nothing is downloaded or installed. HOME and DW_INSTALL_ROOT point
into tmp_path."""

import json
import os
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "dw" / "install_scripts"
PYPROJECT = REPO / "pyproject.toml"
BASH = shutil.which("bash")
PWSH = shutil.which("pwsh")
SHA = "0123456789abcdef0123456789abcdef01234567"
CPU = "https://download.pytorch.org/whl/cpu"
CU130 = "https://download.pytorch.org/whl/cu130"


@dataclass
class Out:
    text: str

    @property
    def lines(self):
        return [line for line in self.text.splitlines() if line.strip()]

    @property
    def torch_line(self):
        return next(line for line in self.lines if " torchvision " in line)

    @property
    def last_line(self):
        return self.lines[-1]


def _fake(path: Path, name: str, body: str) -> None:
    exe = path / name
    exe.write_text("#!/bin/sh\n" + body + "\n")
    exe.chmod(0o755)


def _fakes(tmp_path, driver_cuda, system, sha):
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    _fake(bin_dir, "uname", f'echo "{system}"')
    _fake(bin_dir, "uv", "exit 0")
    if driver_cuda is None:
        _fake(bin_dir, "nvidia-smi", "exit 9")
    else:
        _fake(
            bin_dir,
            "nvidia-smi",
            f'echo "| NVIDIA-SMI 1  Driver Version: 1  CUDA Version: {driver_cuda}  |"',
        )
    if sha is None:
        _fake(bin_dir, "curl", "exit 22")
    else:
        _fake(bin_dir, "curl", f'echo \'{{"sha": "{sha}", "node_id": "x"}}\'')
    return bin_dir


@pytest.fixture
def run_bootstrap(tmp_path):
    root = tmp_path / "root"

    def run(
        driver_cuda=None, args=(), system="Linux", sha=SHA, root_dir=root, env=None
    ):
        bin_dir = _fakes(tmp_path, driver_cuda, system, sha)
        home = tmp_path / "home"
        home.mkdir(exist_ok=True)
        full_env = {
            "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
            "HOME": str(home),
            "DW_INSTALL_ROOT": str(root_dir),
            "DW_BOOTSTRAP_DRY_RUN": "1",
            "DW_BOOTSTRAP_TTY": "0",
            **(env or {}),
        }
        result = subprocess.run(
            [BASH, str(SCRIPTS / "bootstrap.sh"), *args],
            env=full_env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        return Out(result.stdout)

    run.root = root
    return run


def _record(root: Path, index: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "install.json").write_text(
        json.dumps({"torch_index": index, "version": "0", "diffusers_commit": None})
    )


def _existing_venv(root: Path) -> None:
    py = root / "venv" / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("#!/bin/sh\n")
    py.chmod(0o755)


needs_bash = pytest.mark.skipif(BASH is None, reason="bash not available")


@needs_bash
@pytest.mark.parametrize(
    "driver_cuda,expected",
    [("13.0", CU130), ("12.4", "pypi"), (None, CPU)],
)
def test_torch_index_follows_the_driver(driver_cuda, expected, run_bootstrap):
    out = run_bootstrap(driver_cuda=driver_cuda)
    if expected == "pypi":
        assert "--index-url" not in out.torch_line
    else:
        assert f"--index-url {expected}" in out.torch_line
    assert f"torch index: {expected}" in out.text


@needs_bash
def test_macos_uses_pypi_with_the_pin_and_the_mps_extra(run_bootstrap):
    out = run_bootstrap(driver_cuda="13.0", system="Darwin")
    assert "--index-url" not in out.torch_line and "torch<2.15" in out.torch_line
    assert "fp4-fp8-for-torch-mps" in out.text


@needs_bash
def test_a_rerun_keeps_the_recorded_torch_index(run_bootstrap):
    _record(run_bootstrap.root, CPU)
    _existing_venv(run_bootstrap.root)
    out = run_bootstrap(driver_cuda="13.0")
    assert "whl/cpu" in out.torch_line and "uv venv" not in out.text
    assert " -U " in out.text


@needs_bash
def test_the_torch_index_flag_overrides_the_record_and_is_not_passed_on(run_bootstrap):
    _record(run_bootstrap.root, CPU)
    out = run_bootstrap(driver_cuda=None, args=["--torch-index", "cu130", "--no-open"])
    assert "whl/cu130" in out.torch_line
    assert out.last_line.endswith("dw setup --no-open")


@needs_bash
def test_a_fresh_install_makes_the_venv_and_pins_torch(run_bootstrap):
    out = run_bootstrap(driver_cuda=None)
    assert "+ uv venv --seed --python 3.12 " in out.text
    assert " -U " not in out.text
    assert "-c " in out.text and "torch-constraints.txt" in out.text


@needs_bash
def test_it_installs_the_server_and_mcp_extras_and_hands_off(run_bootstrap):
    out = run_bootstrap(driver_cuda=None, args=["--no-open"])
    assert "diffusers-workflow[server,mcp]" in out.text
    assert out.last_line.endswith("dw setup --no-open")


@needs_bash
def test_version_and_wheel_choose_the_requirement(run_bootstrap):
    assert (
        "diffusers-workflow[server,mcp]==1.2.3"
        in run_bootstrap(args=["--version", "1.2.3"]).text
    )
    wheel = run_bootstrap(
        env={"DW_BOOTSTRAP_WHEEL": "/tmp/dw-1.0-py3-none-any.whl"}
    ).text
    assert (
        "/tmp/dw-1.0-py3-none-any.whl[server,mcp]" in wheel
        and "diffusers-workflow[" not in wheel
    )


@needs_bash
def test_diffusers_comes_from_an_archive_not_git(run_bootstrap):
    out = run_bootstrap(driver_cuda=None)
    assert "git+" not in out.text
    assert (
        f"--reinstall --no-deps https://github.com/huggingface/diffusers/archive/{SHA}.zip"
        in out.text
    )
    assert f'"diffusers_commit": "{SHA}"' in out.text


@needs_bash
@pytest.mark.parametrize("sha", [None, "not-a-sha", SHA.upper()])
def test_a_failed_sha_fetch_falls_back_to_released_diffusers(sha, run_bootstrap):
    out = run_bootstrap(sha=sha)
    assert "archive" not in out.text and '"diffusers_commit": null' in out.text
    assert "uv pip install" in out.text and out.text.rstrip().endswith("dw setup")


@needs_bash
def test_dry_run_installs_nothing(run_bootstrap, tmp_path):
    root = tmp_path / "fresh" / "root"
    run_bootstrap(driver_cuda=None, root_dir=root)
    assert not root.exists() and not root.parent.exists()


@needs_bash
def test_the_sha_override_replaces_the_github_lookup(run_bootstrap):
    other = "f" * 40
    out = run_bootstrap(sha=None, env={"DW_BOOTSTRAP_DIFFUSERS_SHA": other})
    assert f"archive/{other}.zip" in out.text
    assert f'"diffusers_commit": "{other}"' in out.text
    bad = run_bootstrap(sha=SHA, env={"DW_BOOTSTRAP_DIFFUSERS_SHA": "nope"})
    assert "archive" not in bad.text and '"diffusers_commit": null' in bad.text


@needs_bash
def test_the_body_runs_from_main_called_on_the_last_line():
    lines = [
        line
        for line in (SCRIPTS / "bootstrap.sh").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert lines[-1] == 'main "$@"'


@needs_bash
def test_yes_keeps_the_terminal_redirect_off_the_handoff(run_bootstrap):
    out = run_bootstrap(args=("--yes",))
    assert out.last_line.endswith("dw setup --yes")
    assert "/dev/tty" not in out.text


@needs_bash
def test_an_available_tty_is_redirected_onto_the_handoff(run_bootstrap):
    out = run_bootstrap(env={"DW_BOOTSTRAP_TTY": "1"})
    assert out.last_line.endswith("</dev/tty")


@needs_bash
def test_yes_suppresses_the_redirect_even_with_a_tty(run_bootstrap):
    out = run_bootstrap(args=("--yes",), env={"DW_BOOTSTRAP_TTY": "1"})
    assert out.last_line.endswith("dw setup --yes")
    assert "/dev/tty" not in out.text


def test_the_bootstrap_ships_in_the_wheel():
    package_data = tomllib.loads(PYPROJECT.read_text())["tool"]["setuptools"][
        "package-data"
    ]
    assert "install_scripts/*" in package_data["dw"]
    assert (SCRIPTS / "bootstrap.sh").is_file() and (
        SCRIPTS / "bootstrap.ps1"
    ).is_file()


@pytest.mark.skipif(PWSH is None, reason="pwsh not available")
@pytest.mark.parametrize(
    "driver_cuda,expected",
    [("13.0", CU130), ("12.4", "pypi"), (None, CPU)],
)
def test_the_powershell_twin_follows_the_driver(driver_cuda, expected, tmp_path):
    bin_dir = _fakes(tmp_path, driver_cuda, "Linux", None)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path / "home"),
        "DW_INSTALL_ROOT": str(tmp_path / "root"),
        "DW_BOOTSTRAP_DRY_RUN": "1",
        "DW_BOOTSTRAP_DIFFUSERS_SHA": SHA,
    }
    result = subprocess.run(
        [
            PWSH,
            "-NoProfile",
            "-File",
            str(SCRIPTS / "bootstrap.ps1"),
            "-TorchIndex",
            "",
            "--no-open",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert f"torch index: {expected}" in result.stdout
    assert "diffusers-workflow[server,mcp]" in result.stdout
    assert f"archive/{SHA}.zip" in result.stdout


def _pwsh_dry_run(tmp_path, helper):
    bin_dir = _fakes(tmp_path, None, "Linux", None)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path / "home"),
        "DW_INSTALL_ROOT": str(tmp_path / "root"),
        "DIFFUSERS_HELPER_ROOT": str(helper),
        "DW_BOOTSTRAP_DRY_RUN": "1",
        "DW_BOOTSTRAP_DIFFUSERS_SHA": SHA,
    }
    result = subprocess.run(
        [PWSH, "-NoProfile", "-File", str(SCRIPTS / "bootstrap.ps1"), "--no-open"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.skipif(PWSH is None, reason="pwsh not available")
def test_the_powershell_upgrade_ends_setups_task_before_pip(tmp_path):
    helper = tmp_path / "helper"
    helper.mkdir()
    (helper / "setup-manifest.json").write_text(
        json.dumps({"entries": [{"kind": "service", "key": "dw-serve"}]})
    )
    lines = _pwsh_dry_run(tmp_path, helper).splitlines()
    end = lines.index("+ schtasks /End /TN dw-serve")
    first_pip = next(i for i, line in enumerate(lines) if " pip install " in line)
    assert end < first_pip


@pytest.mark.skipif(PWSH is None, reason="pwsh not available")
def test_the_powershell_install_ends_no_task_setup_did_not_record(tmp_path):
    assert "schtasks" not in _pwsh_dry_run(tmp_path, tmp_path / "no-helper")
