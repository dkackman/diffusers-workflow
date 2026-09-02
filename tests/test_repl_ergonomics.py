"""REPL behavior added by the Phase 0 ergonomics pass: quote handling,
single-argument clear, run-with-overrides, shortcuts, workflow listing."""

import json

from dw.repl import DiffusersWorkflowREPL
from dw.workflow import Workflow


class StubWorkerManager:
    """Records commands and answers success, so run tests need no worker."""

    def __init__(self):
        self.commands = []
        self.worker_active = True
        self.worker_process = None

    def ensure_worker(self, log_level="INFO"):
        pass

    def send_command(self, command):
        self.commands.append(command)

    def get_result(self, timeout=None):
        return {"type": "success", "message": "ok", "run_count": 1, "manifest": []}

    def shutdown_worker(self):
        self.worker_active = False

    def cancel(self):
        self.commands.append({"type": "cancel"})

    def executes(self):
        return [c for c in self.commands if c.get("type") == "execute"]


def make_repl(tmp_path):
    repl = DiffusersWorkflowREPL()
    repl.worker_manager = StubWorkerManager()
    workflow_def = {
        "id": "ergo",
        "variables": {"prompt": "default", "steps": 9},
        "steps": [],
    }
    workflow_file = tmp_path / "ergo.json"
    workflow_file.write_text(json.dumps(workflow_def))
    repl.current_workflow = Workflow(workflow_def, str(tmp_path), str(workflow_file))
    return repl


def test_arg_set_strips_matching_quotes(tmp_path):
    repl = make_repl(tmp_path)
    repl.onecmd('arg set prompt="a cat in a hat"')
    assert repl.workflow_args["prompt"] == "a cat in a hat"
    repl.onecmd("arg set prompt='single'")
    assert repl.workflow_args["prompt"] == "single"
    # an unmatched or interior quote is content, not wrapping
    repl.onecmd('arg set prompt=say "hi" now')
    assert repl.workflow_args["prompt"] == 'say "hi" now'


def test_set_shortcut_is_arg_set(tmp_path):
    repl = make_repl(tmp_path)
    repl.onecmd("set steps=30")
    assert repl.workflow_args == {"steps": "30"}


def test_arg_clear_single_and_all(tmp_path):
    repl = make_repl(tmp_path)
    repl.onecmd("set prompt=x")
    repl.onecmd("set steps=1")
    repl.onecmd("arg clear prompt")
    assert repl.workflow_args == {"steps": "1"}
    repl.onecmd("arg clear")
    assert repl.workflow_args == {}


def test_run_with_inline_overrides_sets_args_then_executes(tmp_path):
    repl = make_repl(tmp_path)
    repl.onecmd('run steps=30 prompt="two words"')
    assert repl.workflow_args == {"steps": "30", "prompt": "two words"}
    executes = repl.worker_manager.executes()
    assert len(executes) == 1
    assert executes[0]["arguments"] == {"steps": "30", "prompt": "two words"}


def test_run_refuses_malformed_or_unknown_overrides(tmp_path):
    repl = make_repl(tmp_path)
    repl.onecmd("run notanassignment")
    assert repl.worker_manager.executes() == [], "malformed override must not run"

    repl.onecmd("run bogus=1")
    assert repl.worker_manager.executes() == [], "unknown argument must not run"
    assert "bogus" not in repl.workflow_args


def test_workflow_names_are_relative_without_extension(tmp_path):
    repl = make_repl(tmp_path)
    (tmp_path / "sub").mkdir()
    (tmp_path / "A.json").write_text("{}")
    (tmp_path / "sub" / "B.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("")
    repl.globals["workflow_dir"] = str(tmp_path)
    names = repl.workflow_commands.workflow_names()
    assert "A" in names and "sub/B" in names
    assert all(not n.endswith(".json") for n in names)


def test_completion_offers_variables_for_run_and_set(tmp_path):
    repl = make_repl(tmp_path)
    assert repl.complete_set("pr", "set pr", 4, 6) == ["prompt="]
    assert repl.complete_run("st", "run st", 4, 6) == ["steps="]
    assert "load" in repl.complete_workflow("l", "workflow l", 9, 10)


# ------------------------------------------------ already-running server
#
# The REPL starts its own persistent worker, so running it beside the
# desktop app (or a plain dw-serve) puts two processes on one GPU. Warn,
# never refuse - pinning a step to CPU while the server runs is legitimate.


def test_warns_when_a_server_answers(capsys, monkeypatch):
    from dw.repl import warn_if_server_running

    monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", "/nonexistent")
    url = warn_if_server_running(probe=lambda base: True)
    assert url == "http://127.0.0.1:8765"
    out = capsys.readouterr().out
    assert "already running" in out
    assert "GPU memory" in out


def test_silent_when_nothing_answers(capsys, monkeypatch):
    from dw.repl import warn_if_server_running

    monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", "/nonexistent")
    assert warn_if_server_running(probe=lambda base: False) is None
    assert capsys.readouterr().out == ""


def test_probe_failure_is_not_fatal(capsys, monkeypatch):
    from dw.repl import warn_if_server_running

    monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", "/nonexistent")

    def boom(base):
        raise OSError("network down")

    assert warn_if_server_running(probe=boom) is None
    assert capsys.readouterr().out == ""


def test_reads_the_port_the_desktop_shell_recorded(tmp_path, monkeypatch):
    import json

    from dw.repl import running_server_url

    monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", str(tmp_path))
    (tmp_path / "server.json").write_text(
        json.dumps({"base_url": "http://127.0.0.1:9001"})
    )
    assert running_server_url() == "http://127.0.0.1:9001"


def test_malformed_server_file_falls_back_to_the_default_port(tmp_path, monkeypatch):
    from dw.repl import running_server_url

    monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", str(tmp_path))
    (tmp_path / "server.json").write_text("{not json")
    assert running_server_url() == "http://127.0.0.1:8765"
