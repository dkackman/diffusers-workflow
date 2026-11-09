"""A chain probe (`stop_after`): the plan is the full take's, only its first
k segments render, and the take is muxed with the matched track cut to the
rendered frames. A second job of the same chain with `resume_from` naming
the probe's last segment file continues it to the end."""

import pytest
import torch
from PIL import Image

from dw import references
from dw.events import RunContext, activate_context, deactivate_context
from dw.guides import anchor_errors
from dw.pipeline_processors.chain import run_chain
from dw.pipeline_processors.chain_rules import chain_stop_after_problem
from dw.runs import activate_output_root, deactivate_output_root
from dw.server.observed_cost import observed_for
from dw.task_domains import frames_to_samples
from tests.test_chain import FakeAudioReference, FakeImageReference, FakePipeline
from tests.test_observed_cost import MINUTE, run, workflow

FPS = 24
RATE = 16000
FIRST_RUN = "20261109-100000-aaaaaaaa"
SECOND_RUN = "20261109-110000-bbbbbbbb"
WORKFLOW = "probed"
# 40 frames at 24 fps: segments of 9 frames less a 1-frame trim per seam,
# 9 + 8 * 4 = 41, plan five segments
TOTAL_FRAMES = 40
SAMPLES = round(TOTAL_FRAMES / FPS * RATE)


@pytest.fixture
def outputs(tmp_path):
    root = tmp_path / "outputs"
    for run_id in (FIRST_RUN, SECOND_RUN):
        (root / WORKFLOW / run_id).mkdir(parents=True)
    token = activate_output_root(str(root))
    yield root
    deactivate_output_root(token)


@pytest.fixture
def events():
    seen = []
    token = activate_context(RunContext(on_event=seen.append))
    yield seen
    deactivate_context(token)


def _output(arguments, index):
    count = arguments.get("num_frames", 9)
    frames = [Image.new("RGB", (16, 16), (40 * index, 90, 160))] * count
    samples = round(count / FPS * RATE)
    track = torch.full((1, 1, samples), 0.1 * (index + 1))
    return {"videos": [list(frames)], "audio": track, "sampling_rate": RATE}


def _pipeline(outputs, run):
    return FakePipeline(
        _output, output_dir=str(outputs / WORKFLOW / run), file_prefix="probed.video"
    )


def _track():
    """A matched track whose every sample says where it came from."""
    return torch.arange(SAMPLES, dtype=torch.float32).repeat(1, 1) / SAMPLES


def _arguments():
    return {
        "prompt": "test",
        "num_frames": 9,
        "references": [
            FakeImageReference(Image.new("RGB", (16, 16), "red")),
            FakeAudioReference(_track(), RATE),
        ],
    }


def _chain(**overrides):
    return {
        "match_audio": True,
        "fps": FPS,
        "trim_frames": 1,
        "segment_argument": "references",
        "save_segments": True,
        "keep_segments": True,
    } | overrides


def _segment_reference(index, run=FIRST_RUN):
    return references.make_ref(
        references.OUTPUT,
        f"{WORKFLOW}/{run}/probed.video.0.segment-{index:03d}.mp4",
    )


def _logs(events, kind):
    return [e for e in events if e.get("event") == "log" and e.get("kind") == kind]


def test_stop_after_renders_k_and_trims_the_track(outputs, events):
    pipeline = _pipeline(outputs, FIRST_RUN)

    result = run_chain(pipeline, _chain(stop_after=2), _arguments())

    assert len(pipeline.calls) == 2
    assert result.frames.frame_count == 9 + 8
    assert result.audio.shape[1] == frames_to_samples(9 + 8, FPS, RATE)
    assert torch.equal(
        torch.as_tensor(result.audio), _track()[:, : result.audio.shape[1]]
    )
    assert [shot["num_frames"] for shot in result.shots] == [9, 8]
    assert all(shot["planned_segments"] == 5 for shot in result.shots)
    (stopped,) = _logs(events, "chain_stopped_after")
    assert stopped["stop_after"] == 2 and stopped["segments"] == 5
    assert "2 of 5" in stopped["message"]


def test_stop_after_past_the_plan_renders_it_all(outputs, events):
    pipeline = _pipeline(outputs, FIRST_RUN)

    result = run_chain(pipeline, _chain(stop_after=9), _arguments())

    assert len(pipeline.calls) == 5
    assert result.frames.frame_count == TOTAL_FRAMES
    assert torch.equal(torch.as_tensor(result.audio), _track())
    assert all(shot["planned_segments"] == 5 for shot in result.shots)
    assert _logs(events, "chain_stopped_after") == []


def test_resume_continues_a_stopped_probe(outputs, events):
    probe = _pipeline(outputs, FIRST_RUN)
    run_chain(probe, _chain(stop_after=2), _arguments())
    full = run_chain(_pipeline(outputs, SECOND_RUN), _chain(), _arguments())

    pipeline = _pipeline(outputs, SECOND_RUN)
    result = run_chain(
        pipeline, _chain(resume_from=_segment_reference(1)), _arguments()
    )

    assert len(pipeline.calls) == 3
    assert result.frames.frame_count == full.frames.frame_count == TOTAL_FRAMES
    assert torch.equal(torch.as_tensor(result.audio), _track())
    assert [shot.get("resumed", False) for shot in result.shots] == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert all(shot["planned_segments"] == 5 for shot in result.shots)


def test_resume_at_or_past_stop_after_is_refused_before_copying(outputs, events):
    run_chain(_pipeline(outputs, FIRST_RUN), _chain(stop_after=2), _arguments())
    pipeline = _pipeline(outputs, SECOND_RUN)

    with pytest.raises(ValueError, match="stop_after 2"):
        run_chain(
            pipeline,
            _chain(stop_after=2, resume_from=_segment_reference(1)),
            _arguments(),
        )

    assert pipeline.calls == []
    assert list((outputs / WORKFLOW / SECOND_RUN).glob("*.mp4")) == []


def test_resume_of_a_complete_take_still_refuses(outputs, events):
    run_chain(_pipeline(outputs, FIRST_RUN), _chain(), _arguments())

    with pytest.raises(ValueError, match="nothing is left to render"):
        run_chain(
            _pipeline(outputs, SECOND_RUN),
            _chain(resume_from=_segment_reference(4)),
            _arguments(),
        )


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_stop_after_zero_is_refused(outputs, value):
    problem = chain_stop_after_problem({"stop_after": value})
    assert problem and "stop_after" in problem and "whole number" in problem
    pipeline = _pipeline(outputs, FIRST_RUN)

    with pytest.raises(ValueError, match="stop_after"):
        run_chain(pipeline, _chain(stop_after=value), _arguments())
    assert pipeline.calls == []


def test_stop_after_passes_null_and_unresolved_references():
    assert chain_stop_after_problem({}) is None
    assert chain_stop_after_problem({"stop_after": None}) is None
    kept = {"save_segments": True, "keep_segments": True}
    assert chain_stop_after_problem({"stop_after": 1, **kept}) is None
    assert chain_stop_after_problem({"stop_after": "variable:probe"}) is None


def test_a_probe_must_keep_its_segments():
    problem = chain_stop_after_problem({"stop_after": 2, "save_segments": True})
    assert "keep_segments" in problem
    problem = chain_stop_after_problem({"stop_after": 2})
    assert "save_segments" in problem


def test_stop_after_is_refused_at_validate():
    definition = {
        "steps": [
            {
                "name": "video",
                "pipeline": {
                    "configuration": {"component_type": "ModularPipeline"},
                    "from_pretrained_arguments": {
                        "model_name": "MiniMaxAI/MiniMax-H3",
                        "workflow": "t2va",
                    },
                    "chain": {"segments": 3, "stop_after": 0},
                    "arguments": {"prompt": "a cat"},
                },
            }
        ]
    }
    paths = [error["path"] for error in anchor_errors(definition)]
    assert "steps[0].pipeline.chain.stop_after" in paths


def test_a_probe_run_is_not_a_full_run_s_observed_basis():
    definition = workflow(["num_frames"], num_frames=124, probe=None)
    definition["steps"] = [{"chain": {"stop_after": "variable:probe"}, "arguments": {}}]

    observed = observed_for(
        definition, [run(10 * MINUTE, {"probe": 2}), run(30 * MINUTE, {})]
    )

    assert observed["runs"] == 1
    assert observed["cold_minutes"] == 30.0
