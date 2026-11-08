"""A chain resumed from an earlier run's kept segments (`resume_from`,
dw/pipeline_processors/chain_resume.py): the dead run's segment files are
copied into the new run's spill, the carry is read off the last of them, and
only the rest of the plan renders."""

import os

import pytest
import torch
from PIL import Image

from dw import references
from dw.events import RunContext, activate_context, deactivate_context
from dw.guides import anchor_errors
from dw.pipeline_processors.chain import run_chain
from dw.pipeline_processors.chain_resume import kept_count
from dw.pipeline_processors.chain_rules import (
    chain_resume_problem,
    resume_reference_problem,
)
from dw.runs import activate_output_root, deactivate_output_root
from dw.security import PathTraversalError
from tests.test_chain import FakePipeline

FPS = 24
RATE = 16000
FIRST_RUN = "20261108-100000-aaaaaaaa"
SECOND_RUN = "20261108-110000-bbbbbbbb"
WORKFLOW = "chained"


class SegmentFailed(RuntimeError):
    pass


def _factory(fail_at=None, audio=False):
    """Segments of 16x16 frames, `num_frames` long, coloured by call; frame 0
    reproduces the keyframe as an image-to-video pipeline does. Call
    `fail_at` (0-based) raises."""

    def output(arguments, index):
        if index == fail_at:
            raise SegmentFailed(f"segment {index + 1} died")
        count = arguments.get("num_frames", 8)
        frames = [Image.new("RGB", (16, 16), (40 * index, 90, 160))] * count
        if "image" in arguments:
            frames[0] = arguments["image"]
        if not audio:
            return {"videos": [list(frames)]}
        samples = round(count / FPS * RATE)
        track = torch.full((1, 1, samples), 0.1 * (index + 1))
        return {"videos": [list(frames)], "audio": track, "sampling_rate": RATE}

    return output


@pytest.fixture
def outputs(tmp_path):
    """An output root with this workflow's two run directories, active."""
    root = tmp_path / "outputs"
    for run in (FIRST_RUN, SECOND_RUN):
        (root / WORKFLOW / run).mkdir(parents=True)
    token = activate_output_root(str(root))
    yield root
    deactivate_output_root(token)


@pytest.fixture
def events():
    seen = []
    token = activate_context(RunContext(on_event=seen.append))
    yield seen
    deactivate_context(token)


def _warnings(events, kind):
    return [e for e in events if e.get("event") == "warning" and e.get("kind") == kind]


def _pipeline(outputs, run, factory):
    return FakePipeline(
        factory, output_dir=str(outputs / WORKFLOW / run), file_prefix="chained.video"
    )


def _chain(**overrides):
    return {
        "segments": 4,
        "fps": FPS,
        "trim_frames": 1,
        "save_segments": True,
        "keep_segments": True,
    } | overrides


def _arguments(num_frames=9):
    return {
        "prompt": "test",
        "num_frames": num_frames,
        "image": Image.new("RGB", (16, 16), "red"),
    }


def _dead_run(outputs, events, chain=None, arguments=None, fail_at=2, audio=False):
    """Run the chain until segment `fail_at + 1` raises; returns the hint."""
    pipeline = _pipeline(outputs, FIRST_RUN, _factory(fail_at, audio))
    with pytest.raises(SegmentFailed):
        run_chain(pipeline, chain or _chain(), arguments or _arguments())
    (hint,) = _warnings(events, "chain_resume_hint")
    return hint


def _first_file_reference():
    return references.make_ref(
        references.OUTPUT, f"{WORKFLOW}/{FIRST_RUN}/chained.video.0.segment-000.mp4"
    )


def test_a_last_frame_chain_resumes_after_k_kept_segments(outputs, events):
    hint = _dead_run(outputs, events)
    first_run = outputs / WORKFLOW / FIRST_RUN
    assert sorted(p.name for p in first_run.glob("*.mp4")) == [
        "chained.video.0.segment-000.mp4",
        "chained.video.0.segment-001.mp4",
    ]
    assert _first_file_reference() in hint["message"]
    assert hint["resume_from"] == _first_file_reference()

    pipeline = _pipeline(outputs, SECOND_RUN, _factory())
    result = run_chain(
        pipeline, _chain(resume_from=_first_file_reference()), _arguments()
    )

    assert len(pipeline.calls) == 2
    # The first rendered segment is keyed on the last kept file's last frame
    carried = pipeline.calls[0]["image"]
    assert carried.size == (16, 16)
    assert abs(carried.getpixel((8, 8))[0] - 40) <= 4
    assert len(result.shots) == 4
    assert all(shot.get("segment_file") for shot in result.shots)
    assert [shot.get("resumed", False) for shot in result.shots] == [
        True,
        True,
        False,
        False,
    ]
    assert [shot["num_frames"] for shot in result.shots] == [9, 8, 8, 8]
    assert result.frames.frame_count == 9 + 8 * 3
    (resumed,) = _warnings(events, "chain_resumed")
    assert resumed["message"].startswith(
        f"Resumed 2 of 4 segments from {_first_file_reference()}"
    )


def test_resume_refuses_a_kept_segment_of_another_length(outputs, events):
    _dead_run(outputs, events, arguments=_arguments(num_frames=29))
    pipeline = _pipeline(outputs, SECOND_RUN, _factory())

    with pytest.raises(ValueError) as refused:
        run_chain(
            pipeline,
            _chain(resume_from=_first_file_reference()),
            _arguments(num_frames=37),
        )

    message = str(refused.value)
    assert "segment 1" in message and "29" in message and "37" in message
    assert pipeline.calls == []
    # Nothing was copied into the new run before the refusal
    assert list((outputs / WORKFLOW / SECOND_RUN).glob("*.mp4")) == []


def test_guide_resume_warns_about_the_decoded_audio_guide(outputs, events):
    chain = _chain(continuity="guide", guide_frames=22, segments=3)
    del chain["trim_frames"]
    _dead_run(
        outputs,
        events,
        chain=chain,
        arguments={"prompt": "test", "num_frames": 39},
        fail_at=1,
        audio=True,
    )
    pipeline = _pipeline(outputs, SECOND_RUN, _factory(audio=True))

    result = run_chain(
        pipeline,
        {**chain, "resume_from": _first_file_reference()},
        {"prompt": "test", "num_frames": 39},
    )

    assert len(pipeline.calls) == 2
    (guide,) = pipeline.calls[0]["guides"]
    assert len(guide["video"].frames) == 22
    assert guide["video"].audio is not None
    assert [shot["num_frames"] for shot in result.shots] == [39, 17, 17]
    (resumed,) = _warnings(events, "chain_resumed")
    assert "Resumed 1 of 3 segments" in resumed["message"]
    assert "decoded" in resumed["message"]


def test_latent_carry_and_resume_are_refused_at_validate():
    chain = {
        "segments": 3,
        "continuity": "guide",
        "save_segments": True,
        "experimental_latent_carry": True,
        "resume_from": _first_file_reference(),
    }
    problem = chain_resume_problem(chain)
    assert "experimental_latent_carry" in problem and "latents" in problem

    definition = {
        "steps": [
            {
                "name": "video",
                "pipeline": {
                    "configuration": {"component_type": "ModularPipeline"},
                    "from_pretrained_arguments": {
                        "model_name": "MiniMaxAI/MiniMax-H3",
                        "workflow": "fl2va",
                    },
                    "chain": chain,
                    "arguments": {"prompt": "a cat"},
                },
            }
        ]
    }
    paths = [error["path"] for error in anchor_errors(definition)]
    assert "steps[0].pipeline.chain.resume_from" in paths


def test_a_bare_path_is_refused_as_the_location_policy_refuses_one(outputs):
    bare = f"{WORKFLOW}/{FIRST_RUN}/chained.video.0.segment-000.mp4"
    problem = resume_reference_problem(bare)
    assert problem.startswith("Refusing to read")
    pipeline = _pipeline(outputs, SECOND_RUN, _factory())

    with pytest.raises(PathTraversalError):
        run_chain(pipeline, _chain(resume_from=bare), _arguments())


def test_kept_files_are_copied_not_moved(outputs, events):
    _dead_run(outputs, events)
    originals = sorted((outputs / WORKFLOW / FIRST_RUN).glob("*.mp4"))
    sizes = [os.path.getsize(path) for path in originals]

    pipeline = _pipeline(outputs, SECOND_RUN, _factory())
    result = run_chain(
        pipeline, _chain(resume_from=_first_file_reference()), _arguments()
    )

    assert [os.path.getsize(path) for path in originals] == sizes
    copied = sorted((outputs / WORKFLOW / SECOND_RUN).glob("*.mp4"))
    assert [path.name for path in copied] == [
        f"chained.video.0.segment-{index:03d}.mp4" for index in range(4)
    ]
    assert [shot["segment_file"] for shot in result.shots] == [p.name for p in copied]


def test_the_estimate_counts_the_kept_segments(outputs, events):
    _dead_run(outputs, events)
    assert kept_count({"resume_from": _first_file_reference()}) == 2
    assert kept_count({"resume_from": None}) == 0
    # A reference that does not resolve yet prices every segment
    missing = references.make_ref(
        references.OUTPUT, f"{WORKFLOW}/latest/nothing.0.segment-000.mp4"
    )
    assert kept_count({"resume_from": missing}) == 0
