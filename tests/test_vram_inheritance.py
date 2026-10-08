"""A VRAM ceiling a hand-built workflow inherits from the catalog by pipeline
identity (#502, stage B of #479): the index, the match, the warning, and the
agreement between templates that declare one identity.

The serving device is mocked as a 24 GB CUDA card, as in
test_h3_vram_ceiling.py, since a ceiling is only checked against the device
actually serving the run.
"""

import contextlib
import copy
import json
import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import dw.validation
from dw.server.app import create_app
from dw.server.jobs import JobManager
from dw.vram_estimate import pipeline_identity
from dw.vram_inheritance import (
    KIND,
    build_index,
    declarations,
    inherited_vram_warnings,
)
from dw.workflow import workflow_from_definition
from dw.library import library_path
from dw.workspace import Workspace

from .test_server import ScriptedWorkerManager, success_script

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS_DIR = os.path.join(REPO_ROOT, "workflows")

H3 = "MiniMaxAI/MiniMax-H3"
REF2VA = ("ModularPipeline", H3, "ref2va")
T2VA = ("ModularPipeline", H3, "t2va")

REF2VA_ESTIMATE = {
    "base_gb": 16.0,
    "bytes_per_voxel": 28.71,
    "gb_per_reference": 1.0,
    "voxel_variables": ["width", "height", "num_frames"],
    "reason": "the template's own calibration",
}
T2VA_ESTIMATE = {
    "base_gb": 16.3,
    "bytes_per_voxel": 28.71,
    "voxel_variables": ["width", "height", "num_frames"],
}
CARD = [{"device": "cuda", "name": "RTX 3090", "vram_gb": 24, "minutes": 7.88}]


@contextlib.contextmanager
def cuda_24gb():
    with (
        patch.object(dw.validation, "get_device_type", return_value="cuda"),
        patch.object(dw.validation, "device_capacity_gb", return_value=24.0),
    ):
        yield


def _references(count):
    return [
        {
            "reference_type": "diffusers.modular_pipelines.minimax_h3.MiniMaxH3ImageReference",
            "from_file": f"https://example.com/reference-{i}.png",
        }
        for i in range(count)
    ]


def _h3_step(name, workflow="ref2va", model=H3, num_frames=209, references=3):
    arguments = {
        "prompt": "x",
        "num_frames": num_frames,
        "width": 1344,
        "height": 768,
    }
    if references is not None:
        arguments["references"] = _references(references)
    return {
        "name": name,
        "pipeline": {
            "configuration": {"component_type": "ModularPipeline"},
            "from_pretrained_arguments": {"model_name": model, "workflow": workflow},
            "arguments": arguments,
        },
        "result": {"content_type": "video/mp4"},
    }


def _template(identity_workflow, estimate, cost=CARD, steps=1):
    """A synthetic single-identity template declaring an estimate."""
    definition = {
        "id": f"template-{identity_workflow}",
        "vram_estimate": copy.deepcopy(estimate),
        "steps": [
            _h3_step(f"shot{i}", workflow=identity_workflow) for i in range(steps)
        ],
    }
    if cost is not None:
        definition["cost"] = copy.deepcopy(cost)
    return definition


def _catalog():
    return [
        ("templates/minimax/reference-to-video", _template("ref2va", REF2VA_ESTIMATE)),
        ("templates/minimax/video-with-audio", _template("t2va", T2VA_ESTIMATE)),
    ]


def _inline(*steps, **extra):
    return {"id": "inline-h3", "steps": list(steps), **extra}


def _warnings(definition, index=None, arguments=None):
    with cuda_24gb():
        workflow = workflow_from_definition(definition, tempfile.mkdtemp())
        return dw.validation.run_warning_check(
            workflow,
            "inherited_vram_warnings",
            arguments,
            ceiling_index=build_index(_catalog()) if index is None else index,
        )


# --- the index ---------------------------------------------------------------


class TestIndex:
    def test_built_from_a_synthetic_catalog_by_identity(self):
        index = build_index(_catalog())
        assert set(index) == {REF2VA, T2VA}
        assert index[REF2VA]["template"] == "templates/minimax/reference-to-video"
        assert index[REF2VA]["vram_estimate"]["gb_per_reference"] == 1.0
        assert index[T2VA]["template"] == "templates/minimax/video-with-audio"

    def test_a_template_with_no_card_is_not_a_source(self):
        catalog = [("templates/x", _template("ref2va", REF2VA_ESTIMATE, cost=None))]
        assert build_index(catalog) == {}
        # ...but its numbers still count for agreement
        assert [name for _, name, _ in declarations(catalog)] == ["templates/x"]

    def test_a_template_with_no_estimate_is_not_a_source(self):
        definition = _template("ref2va", REF2VA_ESTIMATE)
        del definition["vram_estimate"]
        assert build_index([("templates/x", definition)]) == {}

    def test_a_multi_identity_template_is_not_a_source(self):
        definition = _template("ref2va", REF2VA_ESTIMATE)
        definition["steps"].append(
            {
                "name": "still",
                "pipeline": {
                    "configuration": {"component_type": "ZImagePipeline"},
                    "from_pretrained_arguments": {"model_name": "Tongyi/Z-Image"},
                    "arguments": {"prompt": "x"},
                },
                "result": {"content_type": "image/png"},
            }
        )
        assert build_index([("templates/x", definition)]) == {}

    def test_the_plainest_template_is_the_named_source(self):
        catalog = [
            ("templates/a-chain", _template("ref2va", REF2VA_ESTIMATE, steps=3)),
            ("templates/z-single", _template("ref2va", REF2VA_ESTIMATE, steps=1)),
        ]
        assert build_index(catalog)[REF2VA]["template"] == "templates/z-single"
        assert (
            build_index(list(reversed(catalog)))[REF2VA]["template"]
            == "templates/z-single"
        )

    def test_identity_resolves_a_template_variable(self):
        step = _h3_step("shot")
        step["pipeline"]["from_pretrained_arguments"]["model_name"] = "variable:model"
        assert pipeline_identity(step, {"model": H3}) == REF2VA

    def test_the_real_catalog_indexes_ref2va_from_reference_to_video(self):
        index = build_index(_real_catalog())
        assert index[REF2VA]["template"] == "templates/minimax/reference-to-video"


# --- matching ------------------------------------------------------------------


class TestMatch:
    def test_an_inline_ref2va_step_over_the_ceiling_warns(self):
        warnings = _warnings(_inline(_h3_step("shot", references=3)))
        assert len(warnings) == 1
        assert KIND in warnings[0]
        assert "templates/minimax/reference-to-video" in warnings[0]
        assert "1344*768*209" in warnings[0]
        assert "3 references" in warnings[0]
        assert "above the 24 GB" in warnings[0]
        assert "offload and quantization" in warnings[0]
        # The template's calibration story stays in the template
        assert "the template's own calibration" not in warnings[0]

    def test_under_the_ceiling_nothing(self):
        assert _warnings(_inline(_h3_step("shot", num_frames=124))) == []

    def test_another_model_gets_nothing(self):
        step = _h3_step("shot", model="someone/Other-H3-Finetune")
        assert _warnings(_inline(step)) == []

    def test_another_h3_workflow_gets_its_own_ceiling_not_ref2va(self):
        # t2va at 1344x768x209 takes no references: 16.3 + 6.19 GB, under 24
        # - ref2va's per-reference term would have put three on it
        step = _h3_step("shot", workflow="t2va", references=None)
        assert _warnings(_inline(step)) == []
        step = _h3_step("shot", workflow="t2va", num_frames=345, references=None)
        (warning,) = _warnings(_inline(step))
        assert "templates/minimax/video-with-audio" in warning
        assert "reference-to-video" not in warning

    def test_an_identity_the_catalog_does_not_declare_gets_nothing(self):
        step = _h3_step("shot", workflow="fl2va")
        assert _warnings(_inline(step)) == []

    def test_its_own_declaration_wins(self):
        # A lean estimate of its own: judged by that, and nothing inherited
        lean = {**REF2VA_ESTIMATE, "base_gb": 4.0}
        definition = _inline(
            _h3_step("shot", references=3), vram_estimate=lean, cost=CARD
        )
        assert _warnings(definition) == []
        with cuda_24gb():
            workflow = workflow_from_definition(definition, tempfile.mkdtemp())
            assert [
                e for e in workflow.validation_errors() if "GB VRAM" in e["message"]
            ] == []

    def test_its_own_declaration_refuses_as_stage_a_does(self):
        definition = _inline(
            _h3_step("shot", references=3),
            vram_estimate=REF2VA_ESTIMATE,
            cost=CARD,
        )
        assert _warnings(definition) == []
        with cuda_24gb():
            workflow = workflow_from_definition(definition, tempfile.mkdtemp())
            errors = [
                e for e in workflow.validation_errors() if "GB VRAM" in e["message"]
            ]
        assert len(errors) == 1

    def test_an_inherited_ceiling_never_refuses(self):
        definition = _inline(_h3_step("shot", references=3))
        with cuda_24gb():
            workflow = workflow_from_definition(definition, tempfile.mkdtemp())
            assert workflow.validation_errors() == []

    def test_for_each_names_the_largest_member(self):
        step = _h3_step("shot", references=None)
        arguments = step["pipeline"]["arguments"]
        arguments["num_frames"] = "item:num_frames"
        arguments["references"] = "item:references"
        step["for_each"] = "variable:shots"
        definition = _inline(
            step,
            variables={
                "shots": [
                    {"name": "light", "num_frames": 124, "references": _references(1)},
                    {"name": "heavy", "num_frames": 209, "references": _references(3)},
                ]
            },
        )
        (warning,) = _warnings(definition)
        assert warning.startswith("variables.shots[1]:")
        assert "shot@heavy" in warning

    def test_only_matching_steps_are_projected(self):
        # A step of another pipeline beside the ref2va step: projected by
        # neither ceiling, and the ref2va step keeps its own path
        other = _h3_step("other", model="someone/Other", num_frames=999)
        warnings = _warnings(_inline(other, _h3_step("shot", references=3)))
        assert len(warnings) == 1
        assert "999" not in warnings[0]

    def test_an_empty_index_warns_nothing(self):
        assert _warnings(_inline(_h3_step("shot")), index={}) == []

    def test_a_definition_that_does_not_expand_says_the_check_failed(self):
        # It used to warn nothing: the expansion's failure is
        # validation_errors' to report, but a warning source that cannot
        # run now says so rather than going quiet (B10)
        step = _h3_step("shot")
        step["for_each"] = []
        assert _warnings(_inline(step)) == [
            "internal: warning check 'inherited_vram_warnings' failed "
            "(ForEachError) - the server log has the detail"
        ]


def test_the_pure_function_skips_a_declared_estimate():
    definition = _inline(_h3_step("shot"), vram_estimate=REF2VA_ESTIMATE)
    assert (
        inherited_vram_warnings(
            definition, build_index(_catalog()), device_type="cuda", capacity_gb=24.0
        )
        == []
    )


# --- catalog agreement ---------------------------------------------------------


def _real_catalog():
    catalog = []
    for name, source in (
        library_path("workflows", None, primary=WORKFLOWS_DIR).entries()[0].items()
    ):
        with open(os.path.join(source.root, f"{name}.json")) as file:
            catalog.append((name, json.load(file)))
    return catalog


# What a ceiling is: the formula and the cards it is checked against. The
# reason is prose and may differ; the numbers may not
_ESTIMATE_FIELDS = (
    "base_gb",
    "bytes_per_voxel",
    "bytes_per_guide_voxel",
    "gb_per_reference",
    "voxel_variables",
)


def _numbers(definition):
    estimate = definition["vram_estimate"]
    cards = {
        (entry.get("device"), entry.get("vram_gb"))
        for entry in definition.get("cost") or []
    }
    return tuple(estimate.get(field) for field in _ESTIMATE_FIELDS), cards


def disagreements(catalog):
    """Every identity two templates declare different ceilings for. A
    template with no card agrees on the formula alone."""
    seen = {}
    found = []
    for identity, name, definition in declarations(catalog):
        formula, cards = _numbers(definition)
        if identity not in seen:
            seen[identity] = (name, formula)
            continue
        first, first_formula = seen[identity]
        if formula != first_formula:
            found.append((identity, first, name))
    return found


def test_every_template_declaring_one_identity_declares_the_same_numbers():
    catalog = _real_catalog()
    assert len(list(declarations(catalog))) > 1
    assert disagreements(catalog) == []


def test_every_card_a_shared_identity_declares_is_the_same_card():
    by_identity = {}
    for identity, name, definition in declarations(_real_catalog()):
        _, cards = _numbers(definition)
        if cards:
            by_identity.setdefault(identity, {})[name] = cards
    for identity, cards in by_identity.items():
        assert len({frozenset(c) for c in cards.values()}) == 1, (identity, cards)


def test_a_disagreement_is_caught():
    drifted = {**REF2VA_ESTIMATE, "base_gb": 15.0}
    catalog = [
        ("templates/a", _template("ref2va", REF2VA_ESTIMATE)),
        ("templates/b", _template("ref2va", drifted, cost=None)),
    ]
    assert disagreements(catalog) == [(REF2VA, "templates/a", "templates/b")]


# --- over the server -----------------------------------------------------------


@pytest.fixture
def api(tmp_path):
    workspace = Workspace(tmp_path / "studio", "flag").ensure()
    template = os.path.join(workspace.workflows, "templates", "minimax")
    os.makedirs(template)
    with open(os.path.join(template, "reference-to-video.json"), "w") as file:
        json.dump(_template("ref2va", REF2VA_ESTIMATE), file)
    manager = JobManager(
        workspace.outputs,
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=workspace.workflows,
    )
    app = create_app(
        workflow_dir=workspace.workflows,
        output_dir=workspace.outputs,
        job_manager=manager,
        workspace=workspace.root,
    )
    with cuda_24gb(), TestClient(app, base_url="http://localhost") as client:
        yield client


def _inherited(warnings):
    return [w for w in warnings if KIND in w]


def test_validate_answers_valid_with_the_inherited_warning(api):
    answer = api.post(
        "/api/validate", json={"workflow": _inline(_h3_step("shot", references=3))}
    ).json()
    assert answer["valid"] is True, answer
    (warning,) = _inherited(answer["warnings"])
    assert "templates/minimax/reference-to-video" in warning


def test_validate_under_the_ceiling_answers_no_inherited_warning(api):
    answer = api.post(
        "/api/validate", json={"workflow": _inline(_h3_step("shot", num_frames=124))}
    ).json()
    assert answer["valid"] is True, answer
    assert _inherited(answer["warnings"]) == []


def test_a_job_over_an_inherited_ceiling_queues_and_carries_the_warning(api):
    response = api.post(
        "/api/jobs",
        json={
            "workflow": _inline(_h3_step("shot", references=3)),
            "acknowledged_cost": True,
        },
    )
    assert response.status_code == 201, response.text
    (warning,) = _inherited(response.json()["warnings"])
    assert "templates/minimax/reference-to-video" in warning
