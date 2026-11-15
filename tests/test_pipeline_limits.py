"""H3's frame and canvas limits at validation time, read off diffusers (#1361).

An inline H3 workflow with `num_frames: 25` and no `variable_constraints`
passed validate_workflow and was refused by the pipeline after the load.
dw/pipeline_limits.py refuses it at the step's own argument, with the numbers
the pipeline class publishes.
"""

import copy
import json

import pytest

from dw import pipeline_limits
from dw.pipeline_limits import (
    PIPELINE_LIMIT_CLASSES,
    _limits,
    accepted_frames,
    pipeline_limit_errors,
)
from dw.pipeline_processors.h3_rules import RENDER_GRID
from dw.workflow import Workflow

h3 = pytest.importorskip(
    "diffusers.modular_pipelines.minimax_h3.modular_pipeline",
    reason="diffusers without MiniMax-H3",
)

FRAMES_PATH = "steps[0].pipeline.arguments.num_frames"


def h3_step(workflow="ref2va", **arguments):
    return {
        "name": "shot",
        "pipeline": {
            "configuration": {"component_type": "ModularPipeline"},
            "from_pretrained_arguments": {
                "model_name": "MiniMaxAI/MiniMax-H3",
                "workflow": workflow,
            },
            "arguments": {"prompt": "x", "width": 960, "height": 544, **arguments},
        },
        "result": {"content_type": "video/mp4"},
    }


def definition(*steps, **extra):
    return {"id": "inline", "steps": list(steps), **extra}


def errors(*steps):
    return pipeline_limit_errors(definition(*steps))


def paths(*steps):
    return [e["path"] for e in errors(*steps)]


class TestLimitsComeFromDiffusers:
    def test_the_class_pointer_is_the_only_knowledge(self):
        module, name = PIPELINE_LIMIT_CLASSES["minimax_h3"]
        assert getattr(h3, name) is h3.MiniMaxH3ModularPipeline
        assert module == h3.__name__

    def test_the_limits_match_the_template_derivation(self):
        """The same derivation tests/test_variable_constraints.py pins the
        templates' num_frames rule to."""
        bare = object.__new__(h3.MiniMaxH3ModularPipeline)
        modulus = bare.vae_frames_per_chunk
        remainder = bare.vae_latents_per_chunk
        fps = h3.MINIMAX_H3_FPS
        low = next(
            n
            for n in range(1, 10_000)
            if n % modulus == remainder and n / fps >= bare.min_duration
        )
        high = max(
            n
            for n in range(1, 10_000)
            if n % modulus == remainder and n / fps <= bare.max_duration
        )
        limits = _limits()
        assert limits["fps"] == fps
        assert limits["vae_frames_per_chunk"] == modulus
        assert RENDER_GRID["remainder"] == remainder
        assert limits["canvas_multiple"] == bare.canvas_multiple
        assert accepted_frames(limits) == (low, high)
        assert (low, high) == (124, 345)


class TestNumFrames:
    @pytest.mark.parametrize("frames", [25, 107, 346])
    def test_outside_the_window_is_refused(self, frames):
        found = errors(h3_step(num_frames=frames))
        assert [e["path"] for e in found] == [FRAMES_PATH]
        assert "124 to 345" in found[0]["message"]

    @pytest.mark.parametrize("frames", [108, 124, 130, 345])
    def test_inside_the_window_passes(self, frames):
        assert errors(h3_step(num_frames=frames)) == []

    def test_the_message_names_seconds_fps_and_the_rounded_count(self):
        (error,) = errors(h3_step(num_frames=25))
        message = error["message"]
        assert "5 to 15 seconds at 24 fps" in message
        assert "rounds up to 39" in message
        assert "use 124" in message
        (error,) = errors(h3_step(num_frames=346))
        assert "rounds up to 362" in error["message"]
        assert "use 345" in error["message"]

    @pytest.mark.parametrize("workflow", ["t2va", "fl2va", "ref2va"])
    def test_every_h3_workflow_is_checked(self, workflow):
        assert paths(h3_step(workflow, num_frames=25)) == [FRAMES_PATH]

    def test_no_num_frames_keeps_the_default(self):
        assert errors(h3_step()) == []


class TestCanvas:
    def test_off_the_multiple_is_refused(self):
        found = errors(h3_step(width=950))
        assert [e["path"] for e in found] == ["steps[0].pipeline.arguments.width"]
        assert "multiple of 32" in found[0]["message"]
        assert "use 928 or 960" in found[0]["message"]

    def test_height_too(self):
        assert paths(h3_step(height=540)) == ["steps[0].pipeline.arguments.height"]

    def test_on_the_multiple_passes(self):
        assert errors(h3_step(width=960, height=544)) == []


class TestLeftAlone:
    @pytest.mark.parametrize(
        "value", ["previous_result:base.frames", "variable:frames", 25.5, True, None]
    )
    def test_a_reference_or_non_int_is_skipped(self, value):
        assert errors(h3_step(num_frames=value)) == []

    def test_a_non_h3_step_is_skipped(self):
        step = h3_step(num_frames=25, width=950)
        step["pipeline"] = {
            "configuration": {"component_type": "LTX2Pipeline"},
            "from_pretrained_arguments": {"model_name": "Lightricks/LTX-2"},
            "arguments": {"prompt": "x", "num_frames": 25, "width": 950},
        }
        assert errors(step) == []

    def test_a_missing_class_checks_nothing(self, monkeypatch):
        monkeypatch.setitem(
            PIPELINE_LIMIT_CLASSES,
            "minimax_h3",
            ("diffusers.modular_pipelines.minimax_h3.modular_pipeline", "Gone"),
        )
        assert errors(h3_step(num_frames=25, width=950)) == []

    def test_a_missing_module_checks_nothing(self, monkeypatch):
        monkeypatch.setitem(
            PIPELINE_LIMIT_CLASSES, "minimax_h3", ("no.such.module", "Gone")
        )
        assert errors(h3_step(num_frames=25)) == []

    def test_a_missing_attribute_checks_nothing(self, monkeypatch):
        monkeypatch.setattr(
            pipeline_limits,
            "LIMIT_PROPERTIES",
            (*pipeline_limits.LIMIT_PROPERTIES, "no_such_property"),
        )
        assert errors(h3_step(num_frames=25)) == []


INLINE_REF2VA = {
    "id": "inline-ref2va",
    "steps": [h3_step("ref2va", num_frames=25)],
}


class TestThroughValidation:
    """The real path: Workflow.validation_errors, substitution and expansion
    included."""

    def _validate(self, data, arguments=None):
        workflow = Workflow(copy.deepcopy(data), "outputs", "inline.json")
        return workflow.validation_errors(arguments)

    def test_the_issue_repro_is_refused_at_the_step(self):
        found = [e for e in self._validate(INLINE_REF2VA) if "num_frames" in e["path"]]
        assert [e["path"] for e in found] == [FRAMES_PATH]

    def test_a_variable_with_no_constraint_is_checked_after_substitution(self):
        data = {
            "id": "v",
            "variables": {"frames": 25},
            "steps": [h3_step(num_frames="variable:frames")],
        }
        found = [e["path"] for e in self._validate(data) if "num_frames" in e["path"]]
        assert found == [FRAMES_PATH]

    def test_a_constrained_variable_gives_exactly_one_error(self):
        data = {
            "id": "v",
            "variables": {"num_frames": 124},
            "variable_constraints": {
                "num_frames": {
                    "modulus": 17,
                    "remainder": 5,
                    "min_frames": 124,
                    "max_frames": 345,
                }
            },
            "steps": [h3_step(num_frames="variable:num_frames")],
        }
        found = [
            e["path"]
            for e in self._validate(data, {"num_frames": 22})
            if "num_frames" in e["path"]
        ]
        assert found == ["arguments.num_frames"]

    def test_a_for_each_member_is_named_at_the_written_step(self):
        step = h3_step(num_frames="item:num_frames")
        step["for_each"] = "variable:shots"
        step["pipeline"]["arguments"]["prompt"] = "item:prompt"
        data = {
            "id": "fe",
            "variables": {
                "shots": [
                    {"name": "ok", "prompt": "a", "num_frames": 124},
                    {"name": "short", "prompt": "b", "num_frames": 25},
                ]
            },
            "steps": [step],
        }
        found = [e for e in self._validate(data) if "num_frames" in e["path"]]
        assert [e["path"] for e in found] == [FRAMES_PATH]
        assert "short" in found[0]["message"]
        assert "ok" not in found[0]["message"].split("use 124")[-1]

    def test_the_reference_to_video_template_is_refused_once(self):
        path = "workflows/templates/minimax/reference-to-video.json"
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        workflow = Workflow(data, "outputs", path)
        found = [
            e["path"]
            for e in workflow.validation_errors({"num_frames": 22})
            if "num_frames" in e["path"]
        ]
        assert found == ["arguments.num_frames"]
