"""Unit tests for the diffusers updater's pip command construction and
release-floor lookup. The HTTP-facing lifecycle (start/status, busy/refused
states) is covered in tests/test_server.py::TestDiffusersUpdate; these tests
stay below the FastAPI layer and never spawn a real pip subprocess."""

import sys

from dw.server.updater import (
    DIFFUSERS_GIT_URL,
    PYPI_PACKAGE,
    build_pip_commands,
    release_floor,
)


class TestBuildPipCommands:
    def test_default_tracks_git_head(self):
        commands = build_pip_commands()
        assert commands == [
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                DIFFUSERS_GIT_URL,
            ],
            [sys.executable, "-m", "pip", "install", PYPI_PACKAGE],
        ]

    def test_commit_pins_the_git_install(self):
        commands = build_pip_commands(commit="abc1234")
        assert commands[0][-1] == f"{DIFFUSERS_GIT_URL}@abc1234"

    def test_a_git_install_replaces_an_install_of_the_same_version(self):
        """diffusers main keeps one dev version string for a whole release
        cycle, so `--upgrade` saw the newer commit as already satisfied and
        left the old one installed while reporting success (#663). The git
        install is forced, and without its dependencies - forcing those
        would reinstall torch."""
        for commands in (build_pip_commands(), build_pip_commands(commit="abc1234")):
            install = commands[0]
            assert "--force-reinstall" in install
            assert "--no-deps" in install
            assert "--upgrade" not in install

    def test_a_git_install_then_fills_in_new_dependencies(self):
        """The forced install skipped dependencies; a plain install of the
        package name (no URL, so no second clone, and no --upgrade, so the
        git build stays) adds whatever the new commit requires."""
        follow_up = build_pip_commands()[1]
        assert follow_up == [sys.executable, "-m", "pip", "install", PYPI_PACKAGE]

    def test_revert_installs_the_pinned_release_without_git(self):
        commands = build_pip_commands(revert=True)
        assert len(commands) == 1
        args = commands[0]
        assert args[:4] == [sys.executable, "-m", "pip", "install"]
        target = args[4]
        assert target.startswith(f"{PYPI_PACKAGE}==")
        assert "git+" not in target
        assert target == f"{PYPI_PACKAGE}=={release_floor()}"

    def test_revert_ignores_commit(self):
        """revert wins if both were somehow passed through - the route
        itself rejects this combination before start() is ever called, but
        build_pip_commands stays defensively unambiguous."""
        commands = build_pip_commands(commit="abc1234", revert=True)
        assert len(commands) == 1
        assert "git+" not in commands[0][-1]
        assert commands[0][-1] == f"{PYPI_PACKAGE}=={release_floor()}"


class TestReleaseFloor:
    def test_matches_the_pin_in_pyproject_toml(self):
        import re
        from pathlib import Path

        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        text = pyproject.read_text()
        match = re.search(r'"diffusers>=([0-9][0-9.]*[0-9]|[0-9])"', text)
        assert match, "pyproject.toml should pin a diffusers floor"
        assert release_floor() == match.group(1)

    def test_transformers_floor_excludes_5170(self):
        import re
        from pathlib import Path

        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        text = pyproject.read_text()
        match = re.search(r'"transformers>=([0-9][0-9.]*[0-9]|[0-9]),!=5\.17\.0"', text)
        assert match, "pyproject.toml should exclude the broken 5.17.0 release (#232)"

    def test_falls_back_when_pyproject_is_unreadable(self, monkeypatch):
        from dw.server import updater as updater_module

        monkeypatch.setattr(
            updater_module,
            "_PYPROJECT_PATH",
            updater_module._PYPROJECT_PATH.parent / "does-not-exist.toml",
        )
        assert release_floor() == updater_module.FALLBACK_RELEASE_FLOOR


class TestDiffusersUpdaterRunFn:
    """The default _run_fn sanitizes and runs the built commands in order -
    verified here with subprocess.run mocked out, never actually invoked."""

    def _fake_run(self, monkeypatch, returncodes=None):
        from types import SimpleNamespace

        calls = []
        codes = list(returncodes or [])

        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            code = codes.pop(0) if codes else 0
            return SimpleNamespace(
                returncode=code, stdout=f"out{len(calls)}", stderr=f"err{len(calls)}"
            )

        monkeypatch.setattr("dw.server.updater.subprocess.run", fake_run)
        return calls

    def test_run_pip_invokes_subprocess_with_built_commands(self, monkeypatch):
        from dw.server.updater import DiffusersUpdater

        calls = self._fake_run(monkeypatch)

        result = DiffusersUpdater._run_pip(commit="deadbee")
        assert result.returncode == 0
        assert [args for args, _ in calls] == build_pip_commands(commit="deadbee")
        for _, kwargs in calls:
            # shell=True is never passed - subprocess.run defaults to
            # shell=False, and the call here never overrides it
            assert kwargs.get("shell", False) is False
            assert kwargs["capture_output"] is True
            assert kwargs["text"] is True

    def test_run_pip_keeps_every_command_output(self, monkeypatch):
        from dw.server.updater import DiffusersUpdater

        self._fake_run(monkeypatch)

        result = DiffusersUpdater._run_pip()
        assert "out1" in result.stdout and "out2" in result.stdout
        assert "err1" in result.stderr and "err2" in result.stderr

    def test_a_failed_git_install_stops_before_the_dependency_pass(
        self, monkeypatch
    ):
        from dw.server.updater import DiffusersUpdater

        calls = self._fake_run(monkeypatch, returncodes=[1])

        result = DiffusersUpdater._run_pip()
        assert result.returncode == 1
        assert len(calls) == 1

    def test_run_pip_revert_invokes_subprocess_with_release_pin(self, monkeypatch):
        from dw.server.updater import DiffusersUpdater

        calls = self._fake_run(monkeypatch)

        DiffusersUpdater._run_pip(revert=True)
        assert len(calls) == 1
        assert calls[0][0][-1] == f"{PYPI_PACKAGE}=={release_floor()}"
        assert "git+" not in calls[0][0][-1]
