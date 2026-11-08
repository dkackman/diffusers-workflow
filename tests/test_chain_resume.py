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


def _iteration_reference(iteration, run=FIRST_RUN):
    return references.make_ref(
        references.OUTPUT,
        f"{WORKFLOW}/{run}/chained.video.{iteration}.segment-000.mp4",
    )


def _fanned_out_dead_run(outputs, events, keep_iteration_0=False):
    """A previous_result fan-out of two iterations through one wrapper:
    iteration 0 finishes, iteration 1 dies at its segment 3. Without
    `keep_iteration_0` iteration 0's files are removed, as a finished take's
    are once its video is written; returns iteration 1's hint."""
    pipeline = _pipeline(outputs, FIRST_RUN, _factory(fail_at=6))
    run_chain(pipeline, _chain(), _arguments())
    with pytest.raises(SegmentFailed):
        run_chain(pipeline, _chain(), _arguments())
    first_run = outputs / WORKFLOW / FIRST_RUN
    if not keep_iteration_0:
        for path in first_run.glob("chained.video.0.segment-*.mp4"):
            path.unlink()
    (hint,) = _warnings(events, "chain_resume_hint")
    return hint


def test_a_fan_out_resumes_each_iteration_from_its_own_files(outputs, events):
    hint = _fanned_out_dead_run(outputs, events)
    # The hint quotes the iteration that died
    assert hint["resume_from"] == _iteration_reference(1)

    pipeline = _pipeline(outputs, SECOND_RUN, _factory())
    chain = _chain(resume_from=hint["resume_from"])
    first = run_chain(pipeline, chain, _arguments())
    # Iteration 0 kept nothing beside the reference: it renders in full
    assert len(pipeline.calls) == 4
    assert not any(shot.get("resumed") for shot in first.shots)
    second = run_chain(pipeline, chain, _arguments())
    # Iteration 1 keeps its own two segments and renders the other two
    assert len(pipeline.calls) == 6
    assert [shot.get("resumed", False) for shot in second.shots] == [
        True,
        True,
        False,
        False,
    ]
    # Keyed on iteration 1's own segment 2 (the dead run's call 5), not
    # iteration 0's (call 1)
    carried = pipeline.calls[4]["image"]
    assert abs(carried.getpixel((8, 8))[0] - 200) <= 4
    skipped = [w for w in _warnings(events, "chain_resumed") if w["kept"] == 0]
    assert len(skipped) == 1 and "in full" in skipped[0]["message"]


def test_a_fan_out_renders_an_iteration_that_finished_in_full(outputs, events):
    hint = _fanned_out_dead_run(outputs, events, keep_iteration_0=True)

    pipeline = _pipeline(outputs, SECOND_RUN, _factory())
    chain = _chain(resume_from=hint["resume_from"])
    first = run_chain(pipeline, chain, _arguments())
    second = run_chain(pipeline, chain, _arguments())

    # Iteration 0's complete set is not the one named: it renders in full
    # rather than refusing the job
    assert len(pipeline.calls) == 6
    assert not any(shot.get("resumed") for shot in first.shots)
    assert sum(1 for shot in second.shots if shot.get("resumed")) == 2


def test_the_named_iteration_with_every_segment_kept_is_refused(outputs, events):
    pipeline = _pipeline(outputs, FIRST_RUN, _factory())
    run_chain(pipeline, _chain(), _arguments())

    with pytest.raises(ValueError, match="nothing is left to render"):
        run_chain(
            _pipeline(outputs, SECOND_RUN, _factory()),
            _chain(resume_from=_iteration_reference(0)),
            _arguments(),
        )


def _audio_at(result, sample):
    return abs(float(result.audio[..., sample].reshape(-1)[0]))


def test_kept_seams_are_declicked_as_a_seam_with_no_head_material(outputs, events):
    _dead_run(outputs, events, audio=True)
    pipeline = _pipeline(outputs, SECOND_RUN, _factory(audio=True))

    result = run_chain(
        pipeline, _chain(resume_from=_first_file_reference()), _arguments()
    )

    kept_seam = result.shots[1]["start_sample"]
    middle = result.shots[1]["start_sample"] + result.shots[1]["num_samples"] // 2
    assert _audio_at(result, middle) > 0.1
    # The kept seam fades out and in, as the chain's declick does
    assert _audio_at(result, kept_seam - 1) < 0.02
    assert _audio_at(result, kept_seam) < 0.02
    assert (
        result.audio.shape[-1]
        == result.shots[-1]["start_sample"] + (result.shots[-1]["num_samples"])
    )


def _first_call_voiced():
    """Audio on the dead run's first segment only."""
    voiced, silent = _factory(audio=True), _factory(fail_at=2)

    def output(arguments, index):
        return (voiced if index == 0 else silent)(arguments, index)

    return output


def test_a_mixed_silent_and_voiced_kept_set_is_refused(outputs, events):
    pipeline = _pipeline(outputs, FIRST_RUN, _first_call_voiced())
    with pytest.raises(SegmentFailed):
        run_chain(pipeline, _chain(), _arguments())

    resumed = _pipeline(outputs, SECOND_RUN, _factory(audio=True))
    with pytest.raises(ValueError) as refused:
        run_chain(resumed, _chain(resume_from=_first_file_reference()), _arguments())

    message = str(refused.value)
    assert "chained.video.0.segment-000.mp4" in message
    assert "chained.video.0.segment-001.mp4" in message
    assert resumed.calls == []
    assert list((outputs / WORKFLOW / SECOND_RUN).glob("*.mp4")) == []


@pytest.mark.parametrize(
    "kept_audio, live_audio", [(False, True), (True, False)], ids=["silent", "voiced"]
)
def test_kept_and_rendered_segments_must_agree_on_audio(
    outputs, events, kept_audio, live_audio
):
    _dead_run(outputs, events, audio=kept_audio)
    pipeline = _pipeline(outputs, SECOND_RUN, _factory(audio=live_audio))

    with pytest.raises(ValueError, match="audio") as refused:
        run_chain(pipeline, _chain(resume_from=_first_file_reference()), _arguments())

    assert "segment 3" in str(refused.value)
