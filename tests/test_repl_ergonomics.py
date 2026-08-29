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
