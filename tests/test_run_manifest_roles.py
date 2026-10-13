"""The engine records each step's role and the run's inputs in manifest.json."""

import copy
import json
from types import SimpleNamespace

from dw.roles import role_basis, run_inputs
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
    record = RunRecord(
        status="completed", role_basis=role_basis(steps), inputs=run_inputs(steps)
    )
    workflow = _stand_in(tmp_path, [{"step": "a", "files": []}])
    write_run_manifest(workflow, record)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["inputs"] == [{"reference": "asset:cat.png", "kind": "asset"}]
    assert manifest["steps"][0]["role"] == "final"


def test_a_running_manifest_carries_inputs_but_no_roles(tmp_path):
    steps = [{"name": "a", "x": "asset:cat.png"}, {"name": "b"}]
    record = RunRecord(
        status="running", role_basis=role_basis(steps), inputs=run_inputs(steps)
    )
    workflow = _stand_in(tmp_path, [{"step": "a", "files": []}])
    write_run_manifest(workflow, record)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["inputs"] == [{"reference": "asset:cat.png", "kind": "asset"}]
    assert all("role" not in e for e in manifest["steps"])


def _source_run(tmp_path):
    """A finished run of the one-step workflow, and the output: reference
    naming the image it wrote."""
    from dw.workflow import Workflow

    Workflow(_workflow_definition(), str(tmp_path), "/w/workflows/Src.json").run({})
    run_dir = next((tmp_path / "Src").iterdir())
    image = next(p.name for p in run_dir.iterdir() if p.suffix == ".png")
    return f"output:Src/{run_dir.name}/{image}"


def _terminal_manifest(tmp_path, identity):
    run_dir = next((tmp_path / identity).iterdir())
    return json.loads((run_dir / "manifest.json").read_text())


def test_a_real_run_records_the_output_a_step_read(tmp_path, fake_pipeline):  # noqa: F811
    # The step's output: string is realized in place as the step runs; the
    # terminal manifest must still name it (it was always [] before)
    from dw.workflow import Workflow

    reference = _source_run(tmp_path)
    definition = _two_step_definition()
    definition["id"] = "Dst"
    definition["steps"][0]["pipeline"]["arguments"]["image"] = reference
    Workflow(definition, str(tmp_path), "/w/workflows/Dst.json").run({})

    manifest = _terminal_manifest(tmp_path, "Dst")
    assert manifest["status"] == "completed"
    assert manifest["inputs"] == [{"reference": reference, "kind": "output"}]
    # and roles still see the previous_result reference realization replaced
    assert {e["step"]: e["role"] for e in manifest["steps"]} == {
        "gen0": "intermediate",
        "second": "final",
    }


def test_a_real_run_records_an_output_carried_by_a_variable(tmp_path, fake_pipeline):  # noqa: F811
    from dw.workflow import Workflow

    reference = _source_run(tmp_path)
    definition = _workflow_definition()
    definition["id"] = "Var"
    definition["variables"] = {"image": reference}
    definition["steps"][0]["pipeline"]["arguments"]["image"] = "variable:image"
    Workflow(definition, str(tmp_path), "/w/workflows/Var.json").run({})

    manifest = _terminal_manifest(tmp_path, "Var")
    assert manifest["inputs"] == [{"reference": reference, "kind": "output"}]
