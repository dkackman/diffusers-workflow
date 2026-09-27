"""Category 4/5 coverage for #479's per-step VRAM projection, against the
real catalog rather than inline fixtures.

Every real-template check here mocks the serving device the same way
test_vram_estimate.py does (a 24 GB CUDA card), since `vram_estimate_errors`
is only checked against the device actually serving the run.
"""

import contextlib
import copy
import os
import tempfile
from unittest.mock import patch

import dw.workflow
import dw.vram_estimate
from dw.workflow import workflow_from_definition, workflow_from_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(REPO_ROOT, "workflows", "templates")
MINIMAX_DIR = os.path.join(TEMPLATES_DIR, "minimax")


@contextlib.contextmanager
def cuda_24gb():
    with (
        patch.object(dw.workflow, "get_device_type", return_value="cuda"),
        patch.object(dw.workflow, "device_capacity_gb", return_value=24.0),
    ):
        yield


def _vram_errors(errors):
    return [e for e in errors if "GB VRAM" in e["message"]]


def _load(path):
    return workflow_from_file(path, tempfile.mkdtemp())


def _template_json_files(root):
    for directory, _, files in os.walk(root):
        for name in files:
            if name.endswith(".json"):
                yield os.path.join(directory, name)


def _iter_steps(definition):
    steps = definition.get("steps") if isinstance(definition, dict) else None
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict):
                yield step


def _is_ref2va_step(step):
    pipeline = step.get("pipeline")
    if not isinstance(pipeline, dict):
        return False
    from_pretrained = pipeline.get("from_pretrained_arguments")
    return (
        isinstance(from_pretrained, dict)
        and from_pretrained.get("workflow") == "ref2va"
    )


# --- Category 4: catalog acceptance at 1344x768 on cuda/24GB ---------------


def test_reference_to_video_accepts_209_frames_at_1344x768():
    with cuda_24gb():
        workflow = _load(os.path.join(MINIMAX_DIR, "reference-to-video.json"))
        errors = _vram_errors(
            workflow.validation_errors(
                arguments={"width": 1344, "height": 768, "num_frames": 209}
            )
        )
        assert errors == []


def test_reference_to_video_refuses_260_frames_at_1344x768():
    with cuda_24gb():
        workflow = _load(os.path.join(MINIMAX_DIR, "reference-to-video.json"))
        errors = _vram_errors(
            workflow.validation_errors(
                arguments={"width": 1344, "height": 768, "num_frames": 260}
            )
        )
        assert len(errors) == 1


def _ref2va_definition(num_frames, num_references, width=1344, height=768):
    references = [
        {
            "reference_type": "diffusers.modular_pipelines.minimax_h3.MiniMaxH3ImageReference",
            "from_file": f"https://example.com/reference-{i}.png",
        }
        for i in range(num_references)
    ]
    return {
        "id": "inline-ref2va-vram",
        "cost": [
            {"device": "cuda", "name": "RTX 3090", "vram_gb": 24, "minutes": 7.88}
        ],
        "vram_estimate": {
            "base_gb": 16.0,
            "bytes_per_voxel": 28.71,
            "gb_per_reference": 1.0,
            "voxel_variables": ["width", "height", "num_frames"],
        },
        "steps": [
            {
                "name": "shot",
                "pipeline": {
                    "configuration": {"component_type": "ModularPipeline"},
                    "from_pretrained_arguments": {
                        "model_name": "MiniMaxAI/MiniMax-H3",
                        "workflow": "ref2va",
                    },
                    "arguments": {
                        "prompt": "x",
                        "references": references,
                        "num_frames": num_frames,
                        "width": width,
                        "height": height,
                    },
                },
                "result": {"content_type": "video/mp4"},
            }
        ],
    }


def test_inline_ref2va_refuses_three_references_at_209_frames():
    with cuda_24gb():
        workflow = workflow_from_definition(
            _ref2va_definition(209, 3), tempfile.mkdtemp()
        )
        assert len(_vram_errors(workflow.validation_errors())) == 1


def test_inline_ref2va_accepts_three_references_at_175_frames():
    with cuda_24gb():
        workflow = workflow_from_definition(
            _ref2va_definition(175, 3), tempfile.mkdtemp()
        )
        assert _vram_errors(workflow.validation_errors()) == []


def test_inline_ref2va_refuses_one_reference_at_260_frames():
    with cuda_24gb():
        workflow = workflow_from_definition(
            _ref2va_definition(260, 1), tempfile.mkdtemp()
        )
        assert len(_vram_errors(workflow.validation_errors())) == 1


def _dialogue_short_shots_with_three_references_on_deflect():
    """The 'deflect' entry (shots[1]) given 'cold_open''s (shots[0]'s) first
    three references - two subject portraits plus character_a's voice - so
    it carries 3 counted references once character_a_voice is non-null,
    rather than its own default single subject reference."""
    workflow = _load(os.path.join(MINIMAX_DIR, "dialogue-short.json"))
    default_shots = workflow.workflow_definition["variables"]["shots"]
    shots = copy.deepcopy(default_shots)
    assert shots[1]["name"] == "deflect"
    shots[1]["num_frames"] = 209
    shots[1]["references"] = copy.deepcopy(shots[0]["references"][:3])
    return workflow, shots


def test_dialogue_short_deflect_over_ceiling_with_a_named_voice():
    with cuda_24gb():
        workflow, shots = _dialogue_short_shots_with_three_references_on_deflect()
        errors = _vram_errors(
            workflow.validation_errors(
                arguments={
                    "width": 1344,
                    "height": 768,
                    "character_a_voice": "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav",
                    "shots": shots,
                }
            )
        )
        assert len(errors) == 1
        assert errors[0]["path"] == "arguments.shots[1]"
        assert "shot@deflect" in errors[0]["message"]


def test_dialogue_short_deflect_under_ceiling_without_a_named_voice():
    with cuda_24gb():
        workflow, shots = _dialogue_short_shots_with_three_references_on_deflect()
        errors = _vram_errors(
            workflow.validation_errors(
                arguments={"width": 1344, "height": 768, "shots": shots}
            )
        )
        assert errors == []


# --- Category 5: catalog sweeps ---------------------------------------------


def test_every_ref2va_minimax_template_declares_gb_per_reference():
    checked = 0
    for path in _template_json_files(MINIMAX_DIR):
        workflow = workflow_from_file(path, tempfile.mkdtemp())
        definition = workflow.workflow_definition
        if not any(_is_ref2va_step(step) for step in _iter_steps(definition)):
            continue
        checked += 1
        estimate = definition.get("vram_estimate")
        assert isinstance(estimate, dict), (
            f"{path} has a ref2va step but no vram_estimate"
        )
        assert "gb_per_reference" in estimate, (
            f"{path} vram_estimate has no gb_per_reference"
        )
    assert checked > 0, "expected at least one minimax ref2va template"


def test_every_template_with_vram_estimate_validates_clean_at_its_defaults():
    checked = 0
    for path in _template_json_files(TEMPLATES_DIR):
        workflow = workflow_from_file(path, tempfile.mkdtemp())
        definition = workflow.workflow_definition
        if not isinstance(definition.get("vram_estimate"), dict):
            continue
        checked += 1
        with cuda_24gb():
            errors = _vram_errors(workflow.validation_errors())
            assert errors == [], (
                f"{path} has a VRAM error at its own defaults: {errors}"
            )

            expanded = workflow.expanded_definition()
            projections = list(
                dw.vram_estimate._projections(
                    expanded, definition["vram_estimate"], None
                )
            )
            assert projections, f"{path} declares vram_estimate but projects no step"
    assert checked > 0, "expected at least one template declaring vram_estimate"


def test_ltx2_text_to_video_refuses_361_frames():
    with cuda_24gb():
        workflow = _load(os.path.join(TEMPLATES_DIR, "ltx2", "text-to-video.json"))
        errors = _vram_errors(workflow.validation_errors(arguments={"num_frames": 361}))
        assert len(errors) == 1
