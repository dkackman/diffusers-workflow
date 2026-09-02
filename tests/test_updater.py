"""Unit tests for the diffusers updater's pip command construction and
release-floor lookup. The HTTP-facing lifecycle (start/status, busy/refused
states) is covered in tests/test_server.py::TestDiffusersUpdate; these tests
stay below the FastAPI layer and never spawn a real pip subprocess."""

import sys

from dw.server.updater import (
    DIFFUSERS_GIT_URL,
    PYPI_PACKAGE,
    build_pip_args,
    release_floor,
)


class TestBuildPipArgs:
    def test_default_tracks_git_head(self):
        args = build_pip_args()
        assert args == [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            DIFFUSERS_GIT_URL,
        ]

    def test_commit_pins_the_git_install(self):
        args = build_pip_args(commit="abc1234")
        assert args == [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            f"{DIFFUSERS_GIT_URL}@abc1234",
        ]

    def test_revert_installs_the_pinned_release_without_git(self):
        args = build_pip_args(revert=True)
        assert args[:4] == [sys.executable, "-m", "pip", "install"]
        target = args[4]
        assert target.startswith(f"{PYPI_PACKAGE}==")
        assert "git+" not in target
        assert target == f"{PYPI_PACKAGE}=={release_floor()}"

    def test_revert_ignores_commit(self):
        """revert wins if both were somehow passed through - the route
        itself rejects this combination before start() is ever called, but
        build_pip_args stays defensively unambiguous."""
        args = build_pip_args(commit="abc1234", revert=True)
        assert "git+" not in args[-1]
        assert args[-1] == f"{PYPI_PACKAGE}=={release_floor()}"


class TestReleaseFloor:
    def test_matches_the_pin_in_pyproject_toml(self):
        import re
        from pathlib import Path

        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        text = pyproject.read_text()
        match = re.search(r'"diffusers>=([0-9][0-9.]*[0-9]|[0-9])"', text)
        assert match, "pyproject.toml should pin a diffusers floor"
        assert release_floor() == match.group(1)

    def test_falls_back_when_pyproject_is_unreadable(self, monkeypatch):
        from dw.server import updater as updater_module

        monkeypatch.setattr(
            updater_module,
            "_PYPROJECT_PATH",
            updater_module._PYPROJECT_PATH.parent / "does-not-exist.toml",
        )
        assert release_floor() == updater_module.FALLBACK_RELEASE_FLOOR


class TestDiffusersUpdaterRunFn:
    """The default _run_fn sanitizes and runs the built argument list -
    verified here with subprocess.run mocked out, never actually invoked."""

    def test_run_pip_invokes_subprocess_with_built_args(self, monkeypatch):
        from types import SimpleNamespace
        from dw.server.updater import DiffusersUpdater

        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr("dw.server.updater.subprocess.run", fake_run)

        result = DiffusersUpdater._run_pip(commit="deadbee")
        assert result.returncode == 0
        assert captured["args"] == [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            f"{DIFFUSERS_GIT_URL}@deadbee",
        ]
        # shell=True is never passed - subprocess.run defaults to shell=False,
        # and the call here never overrides it
        assert captured["kwargs"].get("shell", False) is False
        assert captured["kwargs"]["capture_output"] is True
        assert captured["kwargs"]["text"] is True

    def test_run_pip_revert_invokes_subprocess_with_release_pin(self, monkeypatch):
        from types import SimpleNamespace

        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr("dw.server.updater.subprocess.run", fake_run)

        from dw.server.updater import DiffusersUpdater

        DiffusersUpdater._run_pip(revert=True)
        assert captured["args"][-1] == f"{PYPI_PACKAGE}=={release_floor()}"
        assert "git+" not in captured["args"][-1]
