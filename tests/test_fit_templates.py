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

import json
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
            "fps": "variable:frame_rate",
        }
        assert steps["with_source_audio"]["task"]["arguments"]["video"] == (
            "previous_result:restore"
        )


@pytest.mark.parametrize(
    "name,fit,pipeline,restore",
    [
        ("upscale-clip", "fit", "upscaled", "restore"),
        ("refine-clip", "fit", "refine", "restore"),
        ("refine-in-place", "fitted", "refine", "restored"),
        ("restore-deblur", "fit", "restored", "final"),
        ("restore-decompression", "fit", "restored", "final"),
    ],
)
def test_the_output_is_written_at_the_sources_frame_rate(name, fit, pipeline, restore):
    """#1107 (#1090 plan v1): 'frame_rate' defaults to "source" and is the
    output rate. The model runs at its 24 fps and stamps 24 on its own output;
    restore_to_source sets the rate, and nothing downstream declares one."""
    with open(os.path.join(TEMPLATES, f"{name}.json"), encoding="utf-8") as handle:
        definition = json.load(handle)
    steps = {s["name"]: s for s in definition["steps"]}

    assert definition["variables"]["frame_rate"] == "source"
    assert steps[pipeline]["pipeline"]["arguments"]["frame_rate"] == 24.0
    assert steps[pipeline]["result"]["fps"] == 24.0
    assert steps[restore]["task"]["arguments"]["fps"] == "variable:frame_rate"
    assert steps[restore]["task"]["arguments"]["fit"] == f"previous_result:{fit}.fit"
    names = list(steps)
    for step in [steps[n] for n in names[names.index(restore) :]]:
        assert "fps" not in step.get("result", {}), step
    # no declared result.fps anywhere names the "source" sentinel
    for step in definition["steps"]:
        assert step.get("result", {}).get("fps") != "variable:frame_rate", step["name"]


@pytest.fixture
def eight_fps_clip(tmp_path, monkeypatch):
    library = tmp_path / "assets"
    library.mkdir()
    ramp = numpy.linspace(0.1, 0.9, SRC_FRAMES, dtype=numpy.float32)
    frames = [numpy.full((SRC_H, SRC_W, 3), value) for value in ramp]
    export_to_video(frames, str(library / "clip.mp4"), fps=8)
    monkeypatch.setenv(ASSET_DIR_ENV_VAR, str(library))
    return library


@pytest.mark.parametrize(
    "template,model_step",
    [
        ("alpha-matte", "matte"),
        ("restore-colorize", "colorized"),
        ("restore-deblur", "restored"),
        ("restore-decompression", "restored"),
    ],
)
class TestTheModelPassKeepsTheSourcesRate:
    """#1079, #1080, #1108: the model runs at its 24 fps, but the output is written
    at the source's own rate unless 'frame_rate' says otherwise. The real
    path: the template expanded against an 8 fps asset, `fit` through the
    task dispatch, the model step's 24 fps output standing in for the
    pipeline, and `final` run and saved through its own result definition."""

    # A small working size keeps the stand-in matte cheap; the fixture's
    # asset is named, since deblur and decompression default to other names
    SMALL = {
        "width": 256,
        "height": 192,
        "num_frames": 25,
        "source_video": "asset:clip.mp4",
    }

    def written(self, template, model_step, tmp_path, **arguments):
        from dw.media import video_shape
        from dw.media_types import AudioVideo

        workflow = workflow_from_file(
            os.path.join(TEMPLATES, f"{template}.json"), str(tmp_path / "out")
        )
        steps = workflow.expanded_definition(arguments=dict(self.SMALL, **arguments))
        steps = steps["steps"]
        realize_args(steps, run_base_dir(workflow))
        steps = {step["name"]: step for step in steps}
        _, fit = run_fit(steps)

        model = Result(steps[model_step]["result"])
        model.add_result(
            AudioVideo(numpy.zeros((25, 192, 256, 3), numpy.float32), None, None, 24)
        )
        final = steps["final"]
        (task_arguments,) = get_iterations(
            final["task"]["arguments"], {"fit": fit, model_step: model}
        )
        output = Task(final["task"], "cpu").run(task_arguments)
        result = Result(final["result"])
        result.add_result(output)
        saved = tmp_path / "saved"
        saved.mkdir()
        result.save(str(saved), f"{template}-final.0")
        (name,) = [n for n in os.listdir(saved) if n.endswith(".mp4")]
        return video_shape(str(saved / name))

    def test_the_default_writes_the_sources_rate(
        self, template, model_step, eight_fps_clip, tmp_path
    ):
        shape = self.written(template, model_step, tmp_path)

        assert shape["fps"] == pytest.approx(8.0)
        assert (shape["width"], shape["height"], shape["frame_count"]) == (
            SRC_W,
            SRC_H,
            SRC_FRAMES,
        )

    def test_frame_rate_overrides_the_rate_not_the_frames(
        self, template, model_step, eight_fps_clip, tmp_path
    ):
        shape = self.written(template, model_step, tmp_path, frame_rate=24)

        assert shape["fps"] == pytest.approx(24.0)
        assert shape["frame_count"] == SRC_FRAMES


@pytest.fixture
def eight_fps_clip_with_audio(tmp_path, monkeypatch):
    from tests.test_media_info import write_mp4

    library = tmp_path / "assets"
    library.mkdir()
    write_mp4(library / "clip.mp4", frames=SRC_FRAMES, fps=8, width=SRC_W, height=SRC_H)
    monkeypatch.setenv(ASSET_DIR_ENV_VAR, str(library))
    return library


@pytest.mark.parametrize(
    "template,final_step",
    [
        ("refine-in-place", "with_source_audio"),
        ("upscale-clip", "with_source_audio"),
        ("refine-clip", "with_source_audio"),
    ],
)
class TestTheFitShapedRoutesKeepTheSourcesRate:
    """#1107 (#1090 plan v1, stage A): the real path. The template expanded
    against an 8 fps asset with a soundtrack; every task step run through the
    task dispatch in order, each pipeline step stood in for by a 24 fps clip
    at its own size; the deliverable saved through its own result definition.
    It comes back at the source's rate and frame count, with the source's
    track paired to frames / rate, and no fps warning on the way."""

    SMALL = {"width": 256, "height": 192, "num_frames": 25}

    def written(self, template, final_step, tmp_path, warnings, **arguments):
        import av

        from dw.events import RunContext, activate_context, deactivate_context
        from dw.media import stream_seconds, video_shape
        from dw.media_types import AudioVideo

        workflow = workflow_from_file(
            os.path.join(TEMPLATES, f"{template}.json"), str(tmp_path / "out")
        )
        assert workflow.validation_errors(arguments=dict(self.SMALL, **arguments)) == []
        steps = workflow.expanded_definition(arguments=dict(self.SMALL, **arguments))
        steps = steps["steps"]
        realize_args(steps, run_base_dir(workflow))

        def record(event):
            if event["event"] == "warning":
                warnings.append(event.get("code") or event["message"])

        token = activate_context(RunContext(on_event=record))
        try:
            results = {}
            for step in steps:
                result = Result(step.get("result", {}))
                if "pipeline" in step:
                    args = step["pipeline"]["arguments"]
                    # refine-clip's latent upsampler doubles; its refine
                    # renders at twice the working size
                    scale = (
                        2
                        if template == "refine-clip" and step["name"] == "refine"
                        else 1
                    )
                    frames = numpy.zeros(
                        (
                            self.SMALL["num_frames"],
                            args.get("height", self.SMALL["height"]) * scale,
                            args.get("width", self.SMALL["width"]) * scale,
                            3,
                        ),
                        numpy.float32,
                    )
                    result.add_result(AudioVideo(frames, None, None, 24))
                else:
                    (task_arguments,) = get_iterations(
                        step["task"]["arguments"], results
                    )
                    result.add_result(Task(step["task"], "cpu").run(task_arguments))
                results[step["name"]] = result
                if step["name"] == final_step:
                    saved = tmp_path / "saved"
                    saved.mkdir()
                    result.save(str(saved), f"{template}-final.0")
                    break
        finally:
            deactivate_context(token)

        (name,) = [n for n in os.listdir(saved) if n.endswith(".mp4")]
        path = str(saved / name)
        with av.open(path) as container:
            audio_seconds = stream_seconds(container.streams.audio[0])
        return video_shape(path), audio_seconds

    def test_the_default_writes_the_sources_rate(
        self, template, final_step, eight_fps_clip_with_audio, tmp_path
    ):
        warnings = []
        shape, audio_seconds = self.written(template, final_step, tmp_path, warnings)

        assert shape["fps"] == pytest.approx(8.0)
        assert shape["frame_count"] == SRC_FRAMES
        assert audio_seconds == pytest.approx(SRC_FRAMES / 8, abs=0.1)
        assert not any(
            "video_fps_defaulted" in w or "fps_mismatch" in w for w in warnings
        ), warnings

    def test_frame_rate_overrides_the_rate_not_the_frames(
        self, template, final_step, eight_fps_clip_with_audio, tmp_path
    ):
        warnings = []
        shape, audio_seconds = self.written(
            template, final_step, tmp_path, warnings, frame_rate=24
        )

        assert shape["fps"] == pytest.approx(24.0)
        assert shape["frame_count"] == SRC_FRAMES
        assert audio_seconds == pytest.approx(SRC_FRAMES / 24, abs=0.1)

    @pytest.mark.parametrize("bad", ["fast", 0])
    def test_a_bad_frame_rate_is_refused_at_validate(
        self, template, final_step, bad, eight_fps_clip_with_audio, tmp_path
    ):
        workflow = workflow_from_file(
            os.path.join(TEMPLATES, f"{template}.json"), str(tmp_path / "out")
        )
        errors = workflow.validation_errors(arguments={"frame_rate": bad})

        assert errors
        assert any("fps" in str(error) for error in errors), errors


@pytest.mark.parametrize("template", ["restore-deblur", "restore-decompression"])
@pytest.mark.parametrize("bad", ["fast", 0])
def test_a_bad_restore_frame_rate_is_refused_at_validate(template, bad, tmp_path):
    """#1108: 'frame_rate' reaches restore_to_source's `fps`, whose static
    check refuses anything but "source" or a positive number."""
    workflow = workflow_from_file(
        os.path.join(TEMPLATES, f"{template}.json"), str(tmp_path / "out")
    )
    errors = workflow.validation_errors(arguments={"frame_rate": bad})

    assert any("fps" in str(error) for error in errors), errors
