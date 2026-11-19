"""dw.plan_downloads.repos_named (#1445): the Hub repos a run names, read off
the expanded definition, and what admission hands the job."""

import json

import pytest

import dw.plan_downloads
from dw.library import library_path
from dw.plan_downloads import downloads_required, repos_named
from dw.server.admission import admit
from dw.workspace import Workspace


def pipeline_step(name="gen", model="org/base", **extra):
    pipeline = {
        "configuration": {"component_type": "{Fake}"},
        "from_pretrained_arguments": {"model_name": model},
        "arguments": {"prompt": "p"},
    }
    pipeline.update(extra)
    return {"name": name, "pipeline": pipeline}


def definition(*steps):
    return {"id": "named", "steps": list(steps)}


def named(spec, tmp_path):
    return repos_named(spec, str(tmp_path), str(tmp_path))


def test_a_step_pipeline_model_name(tmp_path):
    assert named(definition(pipeline_step()), tmp_path) == ["org/base"]


def test_a_components_model_name(tmp_path):
    step = pipeline_step()
    step["pipeline"]["components"] = {
        "transformer": {
            "configuration": {"component_type": "SomeModel"},
            "from_pretrained_arguments": {"model_name": "org/xformer"},
        }
    }
    assert named(definition(step), tmp_path) == ["org/base", "org/xformer"]


def test_a_lora_repo(tmp_path):
    step = pipeline_step(loras=[{"model_name": "org/lora", "scale": 1.0}])
    assert set(named(definition(step), tmp_path)) == {"org/base", "org/lora"}


def test_a_lora_file_counts_as_its_repo(tmp_path):
    step = pipeline_step(
        loras=[{"model_name": "org/lora", "weight_name": "w.safetensors"}]
    )
    assert set(named(definition(step), tmp_path)) == {"org/base", "org/lora"}


def test_a_load_components_override(tmp_path):
    step = pipeline_step()
    step["pipeline"]["configuration"]["load_components"] = {
        "pretrained_model_name_or_path": {"text_encoder": "org/encoder"}
    }
    assert set(named(definition(step), tmp_path)) == {"org/base", "org/encoder"}


def test_a_load_components_string_override(tmp_path):
    step = pipeline_step()
    step["pipeline"]["configuration"]["load_components"] = {
        "pretrained_model_name_or_path": "org/everything"
    }
    assert "org/everything" in named(definition(step), tmp_path)


def task_step(command, arguments):
    return {"name": "t", "task": {"command": command, "arguments": arguments}}


def test_a_task_weight_repo(tmp_path):
    step = task_step("upscale", {"model_name": "org/up"})
    assert named(definition(step), tmp_path) == ["org/up"]


def test_a_task_file_weight_counts_as_its_repo(tmp_path):
    step = task_step("upscale", {"model_name": "org/up", "filename": "x.pth"})
    assert named(definition(step), tmp_path) == ["org/up"]


def test_a_composed_childs_model_name(tmp_path):
    child = definition(pipeline_step(model="org/child-model"))
    (tmp_path / "child.json").write_text(json.dumps(child))
    parent = definition(
        pipeline_step(model="org/parent-model"),
        {"name": "c", "workflow": {"path": "child.json", "arguments": {}}},
    )
    assert named(parent, tmp_path) == ["org/parent-model", "org/child-model"]


@pytest.mark.parametrize(
    "name",
    [
        "/abs/path/model",
        "./local",
        "../up/model",
        "https://example.com/org/model",
        "variable:model_name",
        "~/models/x",
    ],
)
def test_a_local_path_a_url_or_a_reference_is_dropped(tmp_path, name):
    spec = definition(pipeline_step(model=name), pipeline_step("b", "org/real"))
    assert named(spec, tmp_path) == ["org/real"]


def test_the_result_is_deduplicated_in_first_seen_order(tmp_path):
    spec = definition(
        pipeline_step("a", "org/one"),
        pipeline_step("b", "org/two"),
        pipeline_step("c", "org/one"),
        task_step("upscale", {"model_name": "org/two", "filename": "x.pth"}),
    )
    assert named(spec, tmp_path) == ["org/one", "org/two"]


def test_a_definition_naming_nothing(tmp_path):
    assert named(definition(), tmp_path) == []


def test_an_unresolved_reference_is_never_returned(tmp_path):
    spec = definition(pipeline_step(model="variable:model_name"))
    spec["variables"] = {"model_name": "org/default"}
    assert named(spec, tmp_path) == []


def test_downloads_required_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dw.plan_downloads, "scan_models", lambda cache_dir=None: {"repos": []}
    )
    step = pipeline_step(loras=[{"model_name": "org/lora"}])
    step["pipeline"]["configuration"]["load_components"] = {
        "pretrained_model_name_or_path": {"text_encoder": "org/encoder"}
    }
    spec = definition(step, task_step("upscale", {"model_name": "org/up"}))
    required = downloads_required(
        spec, str(tmp_path), str(tmp_path), str(tmp_path / "hub"), lookup_sizes=False
    )
    assert {entry["repo"] for entry in required} == {
        "org/base",
        "org/lora",
        "org/encoder",
        "org/up",
    }
    assert set(named(spec, tmp_path)) == {entry["repo"] for entry in required}


# -- admission: the expansion, not the document --------------------------


@pytest.fixture
def root(tmp_path):
    return Workspace(tmp_path / "studio", "flag").ensure()


def admitted(root, workflow, arguments):
    return admit(
        workflow_path=None,
        workflow=workflow,
        arguments=arguments,
        base_dir=None,
        workspace=root,
        ceiling_index={},
        output_dir=root.outputs,
        workflow_dir=root.workflows,
        asset_library=library_path("assets", root),
        prompt_library=library_path("prompts", root),
    )


def variable_workflow():
    return {
        "id": "variable_model",
        "variables": {"model_name": "org/default", "prompt": "p"},
        "steps": [
            {
                "name": "gen",
                "pipeline": {
                    "configuration": {"component_type": "{Fake}", "no_generator": True},
                    "from_pretrained_arguments": {"model_name": "variable:model_name"},
                    "arguments": {"prompt": "variable:prompt"},
                },
            }
        ],
    }


def test_admission_names_the_default_resolved_repo(root):
    admission = admitted(root, variable_workflow(), {})
    assert admission.ok, admission.errors
    assert admission.repos_named == ["org/default"]


def test_admission_names_the_argument_supplied_repo(root):
    admission = admitted(root, variable_workflow(), {"model_name": "org/chosen"})
    assert admission.ok, admission.errors
    assert admission.repos_named == ["org/chosen"]
    assert not any(":" in repo for repo in admission.repos_named)


def test_a_refused_admission_names_nothing(root):
    admission = admitted(root, variable_workflow(), {"no_such_argument": 1})
    assert not admission.ok
    assert admission.repos_named == []
