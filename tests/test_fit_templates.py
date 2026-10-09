"""The 2x LTX templates through fit_to_model (#602, stage B).

`upscale-clip` and `refine-clip` fit the caller's clip to the model before
the pipeline and restore after it. What has to hold is that the fitted video
reaches the pipeline in a form it takes - a pipeline's `video` and an
LTX2ReferenceCondition's `frames` both take an array, not an AudioVideo - so
this runs the real path: the template loaded and expanded, its steps
realized against a real asset, the `fit` step through the task dispatch, and
the pipeline step's arguments resolved by get_iterations exactly as Step.run
resolves them. Only the pipeline itself is not run.
"""

import os

import numpy
import pytest
from diffusers.utils import export_to_video

from dw.arguments import realize_args
from dw.library import ASSET_DIR_ENV_VAR
from dw.media_types import FittedVideo
from dw.previous_results import get_iterations
from dw.result import Result
from dw.tasks.task import Task
from dw.workflow import workflow_from_file
from dw.workflow_run import run_base_dir
from tests.test_examples import REPO_ROOT

TEMPLATES = os.path.join(REPO_ROOT, "workflows", "templates", "ltx2")

# A 4:3 clip shorter than either template's 121 frames: letterboxed into
# the 16:9 working size and held on its last frame
SRC_W, SRC_H, SRC_FRAMES, SRC_FPS = 192, 144, 20, 24


@pytest.fixture
def asset_clip(tmp_path, monkeypatch):
    library = tmp_path / "assets"
    library.mkdir()
    ramp = numpy.linspace(0.1, 0.9, SRC_FRAMES, dtype=numpy.float32)
    frames = [numpy.full((SRC_H, SRC_W, 3), value) for value in ramp]
    export_to_video(frames, str(library / "clip.mp4"), fps=SRC_FPS)
    monkeypatch.setenv(ASSET_DIR_ENV_VAR, str(library))
    return library


def realized_steps(name, tmp_path):
    """The template's steps as Step.run sees them before resolving its
    previous results."""
    workflow = workflow_from_file(
        os.path.join(TEMPLATES, f"{name}.json"), str(tmp_path / "out")
    )
    steps = workflow.expanded_definition()["steps"]
    realize_args(steps, run_base_dir(workflow))
    return {step["name"]: step for step in steps}


def run_fit(steps):
    """The `fit` step run through the task dispatch, held as its Result."""
    task = steps["fit"]["task"]
    output = Task(task, "cpu").run(task["arguments"])
    result = Result(steps["fit"]["result"])
    result.add_result(output)
    return output, result


class TestTheFittedVideoReachesThePipeline:
    def test_upscale_clip_conditions_on_the_fitted_video(self, asset_clip, tmp_path):
        steps = realized_steps("upscale-clip", tmp_path)
        output, result = run_fit(steps)

        (arguments,) = get_iterations(
            steps["upscaled"]["pipeline"]["arguments"], {"fit": result}
        )

        from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import (
            LTX2ReferenceCondition,
        )

        (condition,) = arguments["reference_conditions"]
        assert isinstance(condition, LTX2ReferenceCondition)
        frames = condition.frames
        assert frames is output["video"]
        assert isinstance(frames, FittedVideo)
        assert isinstance(frames, numpy.ndarray)
        assert frames.dtype == numpy.float32
        # Half the 960x544 output, at the template's 121 frames
        assert frames.shape == (121, 272, 480, 3)
        assert frames.fps == SRC_FPS
        assert 0.0 <= frames.min() and frames.max() <= 1.0
        # The output size still goes to the pipeline
        assert (arguments["width"], arguments["height"]) == (960, 544)

    def test_refine_clip_upsamples_the_fitted_video(self, asset_clip, tmp_path):
        steps = realized_steps("refine-clip", tmp_path)
        output, result = run_fit(steps)

        (arguments,) = get_iterations(
            steps["upscale"]["pipeline"]["arguments"], {"fit": result}
        )

        video = arguments["video"]
        assert video is output["video"]
        assert isinstance(video, FittedVideo)
        assert video.shape == (121, 288, 512, 3)
        assert video.fps == SRC_FPS

    @pytest.mark.parametrize("name", ["upscale-clip", "refine-clip"])
    def test_the_fit_record_says_what_restore_undoes(self, name, asset_clip, tmp_path):
        steps = realized_steps(name, tmp_path)
        output, _ = run_fit(steps)
        record = output["fit"]

        assert record["mode"] == "letterbox"
        assert (record["source_width"], record["source_height"]) == (SRC_W, SRC_H)
        assert record["source_frames"] == SRC_FRAMES
        assert record["model_frames"] == 121
        # 4:3 in 16:9 is pillarboxed: full height, narrower content
        box = record["content_box"]
        assert box["h"] == record["model_height"]
        assert box["w"] < record["model_width"]

    @pytest.mark.parametrize(
        "name,pipeline", [("upscale-clip", "upscaled"), ("refine-clip", "refine")]
    )
    def test_restore_takes_the_pipeline_output_and_the_record(self, name, pipeline):
        workflow = workflow_from_file(os.path.join(TEMPLATES, f"{name}.json"), "/tmp")
        steps = {s["name"]: s for s in workflow.workflow_definition["steps"]}

        assert steps["restore"]["task"]["command"] == "restore_to_source"
        assert steps["restore"]["task"]["arguments"] == {
            "video": f"previous_result:{pipeline}",
            "fit": "previous_result:fit.fit",
        }
        assert steps["with_source_audio"]["task"]["arguments"]["video"] == (
            "previous_result:restore"
        )
