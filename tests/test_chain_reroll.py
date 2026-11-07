"""A chain's check re-rolling a segment through run_chain (#1241): the
re-roll under save_segments, carry_pick, and on the last segment, and that the
check keeps the previous segment's small measures rather than its frames."""

from PIL import Image

from dw.pipeline_processors.chain import run_chain
from dw.pipeline_processors.chain_check import (
    CheckResult,
    check_segment,
    measure_frames,
)
from tests.test_chain import FakePipeline, modular_output


def _grey(level):
    return Image.new("RGB", (32, 32), (level, level, level))


def _output(bad_calls, sample_rate=100):
    """Segments of flat grey 100; the calls named in `bad_calls` (0-based over
    every render, re-rolls included) come back flat 250 - a picture jump."""

    def factory(arguments, index):
        out = modular_output(arguments, index, sample_rate=sample_rate)
        level = 250 if index in bad_calls else 100
        out["videos"] = [[_grey(level) for _ in out["videos"][0]]]
        return out

    return factory


def _chain(**overrides):
    return {
        "segments": 3,
        "fps": 4,
        "trim_frames": 2,
        "check": True,
        "retries": 1,
    } | overrides


def _levels(frames):
    return [frame.getpixel((0, 0))[0] for frame in frames]


def test_a_failed_segment_is_re_rolled_and_the_good_attempt_kept():
    pipeline = FakePipeline(_output({1}))
    result = run_chain(pipeline, _chain(), {"num_frames": 8, "seed": 5})
    assert len(pipeline.calls) == 4  # 3 segments + one re-roll of segment 2
    assert [s["attempts"] for s in result.shots] == [1, 2, 1]
    assert set(_levels(result.frames)) == {100}


def test_the_last_segment_is_re_rolled_too():
    pipeline = FakePipeline(_output({2}))
    result = run_chain(pipeline, _chain(), {"num_frames": 8, "seed": 5})
    assert len(pipeline.calls) == 4
    assert [s["attempts"] for s in result.shots] == [1, 1, 2]
    assert set(_levels(result.frames)) == {100}


def test_every_attempt_failing_keeps_one_and_warns():
    pipeline = FakePipeline(_output({1, 2}))
    result = run_chain(pipeline, _chain(), {"num_frames": 8, "seed": 5})
    assert result.shots[1]["attempts"] == 2
    assert result.shots[1]["check_failed"]


def test_a_re_roll_under_save_segments_keeps_the_good_attempt(tmp_path):
    pipeline = FakePipeline(
        _output({1}, sample_rate=8000), output_dir=str(tmp_path), file_prefix="wf-step"
    )
    result = run_chain(
        pipeline, _chain(save_segments=True), {"num_frames": 8, "seed": 5}
    )
    try:
        assert len(pipeline.calls) == 4
        assert [s["attempts"] for s in result.shots] == [1, 2, 1]
        # Lossy segment files: the bad attempt's 250 would sit far above 100
        means = [
            float(frame.float().mean()) for chunk in result.frames for frame in chunk
        ]
        assert means and max(means) < 150
    finally:
        result.frames.cleanup()


def test_a_re_roll_under_carry_pick_keeps_the_good_attempt():
    pipeline = FakePipeline(_output({1}))
    result = run_chain(
        pipeline,
        _chain(carry_pick={"sharpest": 2}),
        {"num_frames": 8, "seed": 5},
    )
    assert [s["attempts"] for s in result.shots] == [1, 2, 1]


def test_the_check_reads_the_previous_segments_measures_not_its_frames():
    previous = [_grey(100) for _ in range(8)]
    frames = [_grey(250) for _ in range(8)]
    by_frames = check_segment(previous, frames, frames, _chain_check())
    by_measures = check_segment(
        measure_frames(previous), frames, frames, _chain_check()
    )
    assert by_frames.failed == by_measures.failed
    assert by_frames.measures == by_measures.measures
    assert isinstance(by_measures, CheckResult) and by_measures.frame_data is not None


def test_the_run_holds_no_frames_of_the_previous_segment(monkeypatch):
    seen = []
    import dw.pipeline_processors.chain as chain_module

    real = chain_module.check_segment

    def spy(previous, *args, **kwargs):
        seen.append(previous)
        return real(previous, *args, **kwargs)

    monkeypatch.setattr(chain_module, "check_segment", spy)
    run_chain(FakePipeline(_output(set())), _chain(), {"num_frames": 8, "seed": 5})
    assert seen and all(isinstance(p, tuple) for p in seen)


def _chain_check():
    from dw.pipeline_processors.chain_check import SegmentCheck

    return SegmentCheck()
