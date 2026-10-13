"""The engine records each step's role and the run's inputs in manifest.json."""

import copy
import json
from types import SimpleNamespace

from dw.workflow_run import RunRecord, write_run_manifest

from .test_runs import _workflow_definition, fake_pipeline  # noqa: F401


def _two_step_definition():
    definition = _workflow_definition()
    second = copy.deepcopy(definition["steps"][0])
    second["name"] = "second"
    second["pipeline"]["arguments"]["image"] = "previous_result:gen0"
    definition["steps"].append(second)
    return definition


def _stand_in(tmp_path, manifest):
    return SimpleNamespace(
        manifest=manifest,
        _run_dir=str(tmp_path),
        _run_version=1,
        _elided_steps=[],
        name="roles-test",
        file_spec=None,
    )


def test_terminal_manifest_records_roles_and_inputs(tmp_path, fake_pipeline):  # noqa: F811
    from dw.workflow import Workflow

    definition = _two_step_definition()
    Workflow(definition, str(tmp_path), "/w/workflows/Roles.json").run({})

    run_dir = next((tmp_path / "Roles").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert {e["step"]: e["role"] for e in manifest["steps"]} == {
        "gen0": "intermediate",
        "second": "final",
    }
    assert manifest["inputs"] == []


def test_a_terminal_manifest_lists_the_assets_the_steps_read(tmp_path):
    steps = [{"name": "a", "x": "asset:cat.png"}, {"name": "b"}]
    record = RunRecord(status="completed", expanded_steps=steps)
    workflow = _stand_in(tmp_path, [{"step": "a", "files": []}])
    write_run_manifest(workflow, record)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["inputs"] == [{"reference": "asset:cat.png", "kind": "asset"}]
    assert manifest["steps"][0]["role"] == "final"


def test_a_running_manifest_carries_inputs_but_no_roles(tmp_path):
    steps = [{"name": "a", "x": "asset:cat.png"}, {"name": "b"}]
    record = RunRecord(status="running", expanded_steps=steps)
    workflow = _stand_in(tmp_path, [{"step": "a", "files": []}])
    write_run_manifest(workflow, record)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["inputs"] == [{"reference": "asset:cat.png", "kind": "asset"}]
    assert all("role" not in e for e in manifest["steps"])
