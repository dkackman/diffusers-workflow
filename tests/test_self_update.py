"""The Update dw button's engine (dw/server/self_update.py): the commands, the
latest-release lookup and the restart exit. Nothing here runs pip, reaches the
network or stops the test process."""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from dw.server import self_update
from dw.server.self_update import DwUpdater, build_upgrade_commands


@pytest.fixture(autouse=True)
def _fresh_cache():
    self_update.forget_latest()
    yield
    self_update.forget_latest()


class TestUpgradeCommands:
    def test_upgrade_pins_the_installed_torch(self, tmp_path):
        cmds = build_upgrade_commands(
            "/v/bin/python",
            "2.14.1+cu130",
            "https://download.pytorch.org/whl/cu130",
            constraints_dir=tmp_path,
            git=True,
        )
        assert (tmp_path / "constraints.txt").read_text().strip() == (
            "torch==2.14.1+cu130"
        )
        first = cmds[0]
        assert "diffusers-workflow[server,mcp]" in first
        assert first[first.index("-c") + 1] == str(tmp_path / "constraints.txt")
        assert first[-2:] == [
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu130",
        ]
        assert cmds[-1][-4:] == ["setup", "--yes", "--no-open", "--no-service"]

    def test_diffusers_moves_to_main_between_them(self, tmp_path):
        cmds = build_upgrade_commands("/v/py", "2.1", None, tmp_path, git=True)
        assert any(
            c[:3] == ["/v/py", "-m", "pip"] and "--force-reinstall" in c for c in cmds
        )
        assert cmds[0] is not cmds[-1] and len(cmds) == 4

    def test_pypi_and_no_record_add_no_extra_index(self, tmp_path):
        for index in ("pypi", None):
            first = build_upgrade_commands("/v/py", "2.1", index, tmp_path, git=True)[0]
            assert "--extra-index-url" not in first

    def test_the_torch_index_comes_from_the_install_record(self, monkeypatch):
        record = {"torch_index": "https://download.pytorch.org/whl/cpu"}
        monkeypatch.setattr(self_update, "read_install_record", lambda: record)
        assert self_update.torch_index_from_record() == record["torch_index"]
        monkeypatch.setattr(
            self_update, "read_install_record", lambda: {"torch_index": "pypi"}
        )
        assert self_update.torch_index_from_record() is None
        monkeypatch.setattr(self_update, "read_install_record", lambda: None)
        assert self_update.torch_index_from_record() is None


class TestRunCommands:
    @staticmethod
    def wire(monkeypatch, codes, error=None):
        ran = []

        def run_commands(commands):
            ran.append(commands)
            if error is not None and commands == [["dw", "setup"]]:
                raise error
            code = codes.pop(0)
            return SimpleNamespace(returncode=code, stdout=f"out{code}", stderr="")

        monkeypatch.setattr(
            self_update,
            "build_upgrade_commands",
            lambda *a, **k: [["pip", "dw"], ["pip", "diffusers"], ["dw", "setup"]],
        )
        monkeypatch.setattr(self_update, "installed_torch_version", lambda: None)
        monkeypatch.setattr(self_update, "run_commands", run_commands)
        return ran

    def test_the_refresh_runs_after_the_installs(self, monkeypatch):
        ran = self.wire(monkeypatch, [0, 0])
        assert self_update._run_commands().returncode == 0
        assert ran == [[["pip", "dw"], ["pip", "diffusers"]], [["dw", "setup"]]]

    def test_a_failed_install_stops_before_the_refresh(self, monkeypatch):
        ran = self.wire(monkeypatch, [3])
        assert self_update._run_commands().returncode == 3 and len(ran) == 1

    def test_a_failed_refresh_still_succeeds_and_says_so(self, monkeypatch, caplog):
        self.wire(monkeypatch, [0, 1])
        result = self_update._run_commands()
        assert result.returncode == 0
        assert "registration refresh" in result.stderr
        assert "registration refresh" in caplog.text

    def test_a_refresh_that_cannot_start_still_succeeds(self, monkeypatch):
        self.wire(monkeypatch, [0], error=FileNotFoundError("no dw"))
        result = self_update._run_commands()
        assert result.returncode == 0 and "no dw" in result.stderr

    def test_the_refresh_is_the_dw_beside_the_interpreter(self, tmp_path):
        cmds = build_upgrade_commands("/v/py", None, None, tmp_path, git=True)
        assert Path(cmds[-1][0]).parent == Path(sys.executable).parent


class TestInstallCommand:
    def test_posix_and_windows_one_liners(self):
        assert self_update.install_command("darwin").endswith("bootstrap.sh | bash")
        assert self_update.install_command("linux").startswith("curl -LsSf ")
        windows = self_update.install_command("win32")
        assert windows.startswith("powershell -ExecutionPolicy ByPass")
        assert "bootstrap.ps1" in windows


class TestLatestVersion:
    def test_reads_pypi_info_version_and_caches(self):
        calls = []

        def fetch(url, timeout):
            calls.append(url)
            return {"info": {"version": "0.24.0"}}

        now = [0.0]
        assert self_update.latest_version(fetch, lambda: now[0]) == "0.24.0"
        now[0] = 599
        assert self_update.latest_version(fetch, lambda: now[0]) == "0.24.0"
        assert calls == ["https://pypi.org/pypi/diffusers-workflow/json"]
        now[0] = 601
        self_update.latest_version(fetch, lambda: now[0])
        assert len(calls) == 2

    def test_any_failure_is_none_and_is_cached_too(self):
        def boom(url, timeout):
            raise OSError("offline")

        assert self_update.latest_version(boom) is None
        assert (
            self_update.latest_version(lambda u, t: {"info": {"version": "9"}}) is None
        )


class TestDwUpdater:
    @staticmethod
    def exits():
        codes = []
        return codes, codes.append

    def test_success_exits_for_a_restart(self):
        codes, exit_fn = self.exits()
        updater = DwUpdater(run_fn=lambda: None, exit_fn=exit_fn, delay=0)
        updater.start()
        updater.join()
        assert codes == [75]
        assert updater.status()["status"] == "succeeded"

    def test_failure_reports_and_does_not_exit(self):
        codes, exit_fn = self.exits()
        failed = SimpleNamespace(returncode=2, stdout="out", stderr="boom")
        updater = DwUpdater(run_fn=lambda: failed, exit_fn=exit_fn, delay=0)
        updater.start()
        updater.join()
        state = updater.status()
        assert state["status"] == "failed" and "code 2" in state["error"]
        assert "boom" in state["log"] and codes == []

    def test_an_exception_is_a_failure(self):
        codes, exit_fn = self.exits()

        def run():
            raise RuntimeError("no interpreter")

        updater = DwUpdater(run_fn=run, exit_fn=exit_fn, delay=0)
        updater.start()
        updater.join()
        assert updater.status()["error"] == "no interpreter" and codes == []

    def test_one_at_a_time(self):
        release = []
        updater = DwUpdater(
            run_fn=lambda: time.sleep(0.05) if not release else None,
            exit_fn=lambda code: None,
            delay=0,
        )
        updater.start()
        with pytest.raises(ValueError):
            updater.start()
        updater.join()

    def test_request_restart_records_the_code_and_raises_sigint(self, monkeypatch):
        raised = []
        monkeypatch.setattr(self_update.signal, "raise_signal", raised.append)
        monkeypatch.setitem(self_update._restart, "code", None)
        self_update.request_restart()
        assert raised == [self_update.signal.SIGINT]
        assert self_update.restart_requested() == 75
