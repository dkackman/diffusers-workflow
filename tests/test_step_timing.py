"""Per-step timing (#867): the StepTimer's arithmetic on a fake clock, its
ambient marks, and what a step's manifest entry and `step_end` carry."""

import pytest
import torch
from PIL import Image
from unittest.mock import patch

from dw.pipeline_processors import timing as timing_module
from dw.pipeline_processors.pipeline import Pipeline
from dw.pipeline_processors.timing import (
    StepTimer,
    denoise_start_hook,
    is_compiled,
    mark,
    timed_step,
)
from dw.server.api_models import ManifestEntry
from dw.step_cache import step_cache
from dw.workflow import Workflow
from dw import workflow_run as workflow_run_module

PLAN_KEYS = {
    "load_s",
    "load_cached",
    "encode_s",
    "denoise_s",
    "passes",
    "first_pass_s",
    "pass_s",
    "compiled",
    "decode_s",
    "save_s",
    "segments",
}


class FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self):
        self.now = 0.0

    def at(self, seconds):
        self.now = seconds
        return self.now

    def __call__(self):
        return self.now


def marked(clock, timer, *events):
    """Apply (time, mark name[, args]) events to the timer in order."""
    for time_, name, *args in events:
        clock.at(time_)
        getattr(timer, name)(*args)


def new_timer():
    clock = FakeClock()
    return clock, StepTimer(clock=clock)


def test_a_full_step_reports_every_span():
    clock, timer = new_timer()
    marked(
        clock,
        timer,
        (1.0, "load_started"),
        (1.0, "load_cached", True),
        (3.04, "load_ended"),
        (3.04, "call_started", False),
        (5.0, "denoise_started"),
        (6.0, "pass_ended"),
        (7.0, "pass_ended"),
        (8.0, "pass_ended"),
        (9.5, "call_ended"),
        (9.5, "save_started"),
        (10.0, "save_ended"),
    )

    assert timer.timing() == {
        "load_s": 2.0,
        "load_cached": True,
        "encode_s": 2.0,
        "denoise_s": 3.0,
        "passes": 3,
        "first_pass_s": 1.0,
        "pass_s": 1.0,
        "compiled": False,
        "decode_s": 1.5,
        "save_s": 0.5,
        "segments": 1,
    }


def test_every_seconds_field_is_rounded_to_a_tenth():
    clock, timer = new_timer()
    marked(
        clock,
        timer,
        (0.0, "load_started"),
        (0.26, "load_ended"),
        (0.26, "call_started"),
        (0.57, "denoise_started"),
        (1.0, "pass_ended"),
        (1.5, "pass_ended"),
        (1.84, "call_ended"),
        (1.84, "save_started"),
        (1.9, "save_ended"),
    )

    result = timer.timing()

    assert result["load_s"] == 0.3
    assert result["encode_s"] == 0.3
    assert result["first_pass_s"] == 0.4
    assert result["pass_s"] == 0.5
    assert result["decode_s"] == 0.3
    assert result["save_s"] == 0.1


def test_a_missing_denoise_start_nulls_encode_and_first_pass():
    clock, timer = new_timer()
    marked(
        clock,
        timer,
        (0.0, "call_started"),
        (2.0, "pass_ended"),
        (3.0, "pass_ended"),
        (4.0, "call_ended"),
    )

    result = timer.timing()

    assert result["encode_s"] is None
    assert result["first_pass_s"] is None
    assert result["denoise_s"] is None
    assert result["passes"] == 2
    assert result["decode_s"] == 1.0


def test_missing_load_and_save_marks_are_null_not_guessed():
    clock, timer = new_timer()
    marked(clock, timer, (0.0, "call_started"), (1.0, "call_ended"))
    result = timer.timing()

    assert result["load_s"] is None
    assert result["load_cached"] is None
    assert result["save_s"] is None
    assert result["compiled"] is None

    marked(clock, timer, (2.0, "save_started"))
    assert timer.timing()["save_s"] is None


def test_a_timer_with_no_call_has_no_segments_and_no_spans():
    _, timer = new_timer()

    result = timer.timing()

    assert result["segments"] == 0
    assert result["passes"] is None
    assert result["encode_s"] is None
    assert result["denoise_s"] is None
    assert result["pass_s"] is None


def test_pass_s_is_the_median_of_the_intervals_not_the_mean():
    clock, timer = new_timer()
    # Pass ends at 1, 2, 3, 13: intervals 1, 1, 10 - median 1, mean 4
    marked(
        clock,
        timer,
        (0.0, "call_started"),
        (0.0, "denoise_started"),
        (1.0, "pass_ended"),
        (2.0, "pass_ended"),
        (3.0, "pass_ended"),
        (13.0, "pass_ended"),
        (14.0, "call_ended"),
    )

    assert timer.timing()["pass_s"] == 1.0


def test_pass_s_is_null_with_fewer_than_two_passes():
    clock, timer = new_timer()
    marked(
        clock,
        timer,
        (0.0, "call_started"),
        (0.0, "denoise_started"),
        (1.0, "pass_ended"),
        (2.0, "call_ended"),
    )

    result = timer.timing()

    assert result["passes"] == 1
    assert result["pass_s"] is None
    assert result["first_pass_s"] == 1.0


def test_a_chain_sums_its_segments_and_takes_one_median_over_both():
    clock, timer = new_timer()
    marked(
        clock,
        timer,
        # Segment 1: encode 1, passes end 3 and 4 and 9 (intervals 1, 5), decode 1
        (0.0, "call_started"),
        (1.0, "denoise_started"),
        (3.0, "pass_ended"),
        (4.0, "pass_ended"),
        (9.0, "pass_ended"),
        (10.0, "call_ended"),
        # Segment 2: encode 2, passes end 14 and 15 (interval 1), decode 3
        (20.0, "call_started"),
        (22.0, "denoise_started"),
        (24.0, "pass_ended"),
        (25.0, "pass_ended"),
        (28.0, "call_ended"),
    )

    result = timer.timing()

    assert result["segments"] == 2
    assert result["encode_s"] == 3.0
    assert result["denoise_s"] == 8.0 + 3.0
    assert result["decode_s"] == 4.0
    assert result["first_pass_s"] == 2.0 + 2.0
    assert result["passes"] == 5
    # Intervals 1, 5 and 1; the gap between the segments is no interval
    assert result["pass_s"] == 1.0


def test_a_segment_with_no_denoise_start_and_no_passes_is_all_denoise():
    clock, timer = new_timer()
    marked(clock, timer, (2.0, "call_started"), (7.0, "call_ended"))

    result = timer.timing()

    assert result["denoise_s"] == 5.0
    assert result["encode_s"] is None
    assert result["decode_s"] is None
    assert result["passes"] is None


def test_load_cached_stays_false_once_any_load_was_fresh():
    _, timer = new_timer()

    timer.load_cached(False)
    timer.load_cached(True)

    assert timer.timing()["load_cached"] is False


def test_load_cached_true_holds_when_every_load_was_cached():
    _, timer = new_timer()

    timer.load_cached(True)
    timer.load_cached(True)

    assert timer.timing()["load_cached"] is True


def test_mark_with_no_timer_open_does_nothing():
    mark("call_started")
    mark("pass_ended")


def test_mark_reaches_the_open_timer():
    clock, timer = new_timer()

    with timed_step(timer):
        mark("call_started", compiled=True)

    assert timer.timing()["segments"] == 1
    assert timer.timing()["compiled"] is True


def test_a_step_with_no_timer_hides_the_outer_one_and_restores_it():
    clock, outer = new_timer()

    with timed_step(outer):
        with timed_step(None):
            mark("call_started")
            assert timing_module.current_timer() is None
        assert timing_module.current_timer() is outer
        mark("call_started")

    assert outer.timing()["segments"] == 1
    assert timing_module.current_timer() is None


def test_the_denoise_hook_marks_the_first_forward_once_and_removes_itself():
    module = torch.nn.Linear(2, 2)
    timer = StepTimer()

    with timed_step(timer):
        timer.call_started()
        with denoise_start_hook(module):
            module(torch.zeros(1, 2))
            assert not module._forward_pre_hooks
            segment = timer._segments[0]
            first = segment.denoise_started
            module(torch.zeros(1, 2))

    assert first is not None
    assert segment.denoise_started == first


def test_the_denoise_hook_is_removed_when_forward_never_ran():
    module = torch.nn.Linear(2, 2)

    with denoise_start_hook(module):
        assert len(module._forward_pre_hooks) == 1

    assert not module._forward_pre_hooks


def test_a_plain_module_is_not_compiled():
    assert is_compiled(torch.nn.Sequential(torch.nn.Linear(2, 2))) is False


def test_a_module_with_a_compiled_child_is_compiled():
    child = torch.nn.Linear(2, 2)
    child._compiled_call_impl = lambda *args, **kwargs: None
    parent = torch.nn.Sequential(child)

    assert is_compiled(parent) is True


def test_no_module_means_compiled_is_unknown():
    assert is_compiled(None) is None


# --- the manifest entry and step_end -------------------------------------


class RecordingContext:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, fields))


class FakeLoop:
    def __init__(self):
        self.steps = [None]
        self.workflow_id = "wf"
        self.run_context = RecordingContext()


class FakeStepResult:
    selected = None
    saved_shots = None


class FakeWorkflow:
    def __init__(self):
        self.manifest = []

    def _parent_progress_fields(self):
        return {}


def record(timing, reused):
    workflow, loop = FakeWorkflow(), FakeLoop()
    workflow_run_module.record_step(
        workflow,
        loop,
        0,
        {"name": "generate"},
        (FakeStepResult(), ["generate.png"], reused, False),
        [],
        timing=timing,
    )
    end = [fields for event, fields in loop.run_context.events if event == "step_end"]
    return workflow.manifest[0], end[0]


def full_timing(**overrides):
    return dict.fromkeys(PLAN_KEYS) | {"segments": 1} | overrides


def test_a_step_that_ran_carries_its_timing_in_entry_and_step_end():
    timing = full_timing(load_s=1.5)

    entry, step_end = record(timing, reused=False)

    assert entry["timing"] == timing
    assert step_end["timing"] == timing


def test_a_reused_step_carries_no_timing_even_when_one_is_passed():
    entry, step_end = record(full_timing(), reused=True)

    assert entry["reused"] is True
    assert "timing" not in entry
    assert "timing" not in step_end


def test_a_step_nothing_timed_carries_no_timing():
    entry, step_end = record(None, reused=False)

    assert "timing" not in entry
    assert "timing" not in step_end


class FakeDiffusionPipeline:
    """Called like a diffusers pipeline: no step callback, so the call is all
    there is to time."""

    def __call__(self, **kwargs):
        return type("Output", (), {"images": [Image.new("RGB", (8, 8))]})()


def _load_fake_pipeline(self, shared_components):
    self.pipeline = FakeDiffusionPipeline()


def run_workflow(tmp_path, step):
    step_cache.clear()
    workflow = Workflow(
        {"id": "timing_test", "seed": 1, "steps": [step]}, str(tmp_path), "test.json"
    )
    with patch.object(Pipeline, "load", _load_fake_pipeline):
        workflow.run({})
    return workflow


def test_a_pipeline_step_run_through_the_step_loop_records_its_timing(tmp_path):
    workflow = run_workflow(
        tmp_path,
        {
            "name": "generate",
            "pipeline": {
                "configuration": {"component_type": "{MockPipeline}"},
                "from_pretrained_arguments": {"model_name": "model-generate"},
                "arguments": {"prompt": "a cat"},
            },
        },
    )

    timing = workflow.manifest[0]["timing"]

    assert set(timing) == PLAN_KEYS
    assert timing["segments"] >= 1
    assert timing["save_s"] is not None
    assert timing["load_s"] is not None
    ManifestEntry.model_validate(workflow.manifest[0])


def test_a_task_step_run_through_the_step_loop_has_no_timing(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8)).save(source)

    workflow = run_workflow(
        tmp_path / "out",
        {
            "name": "grade",
            "task": {
                "command": "grade",
                "arguments": {"media": str(source), "contrast": 0.1},
            },
            "result": {"content_type": "image/png"},
        },
    )

    assert workflow.manifest[0]["step"] == "grade"
    assert "timing" not in workflow.manifest[0]


# --- the response model ---------------------------------------------------


def test_a_manifest_entry_without_timing_serializes_without_the_key():
    entry = ManifestEntry.model_validate(
        {"step": "generate", "files": ["a.png"], "subfolder": ""}
    )

    assert "timing" not in entry.model_dump(exclude_unset=True)
    assert "timing" not in entry.model_dump(exclude_none=True)


def test_a_manifest_entry_with_timing_validates_and_round_trips():
    timing = full_timing(
        load_s=1.2,
        load_cached=False,
        encode_s=0.5,
        denoise_s=10.0,
        passes=20,
        first_pass_s=1.0,
        pass_s=0.4,
        compiled=True,
        decode_s=0.7,
        save_s=0.1,
    )
    payload = {
        "step": "generate",
        "files": ["a.png"],
        "subfolder": "",
        "timing": timing,
    }

    entry = ManifestEntry.model_validate(payload)

    assert entry.model_dump(exclude_unset=True) == payload
    assert ManifestEntry.model_validate(entry.model_dump(exclude_unset=True)) == entry


def test_a_timing_with_a_mistyped_field_is_refused():
    with pytest.raises(ValueError):
        ManifestEntry.model_validate(
            {"step": "s", "files": [], "subfolder": "", "timing": {"segments": "x"}}
        )
