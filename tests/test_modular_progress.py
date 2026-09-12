"""Per-step progress from a pipeline that takes no step callback.

A ModularPipeline - H3, LTX-2, Qwen-Image - has no `callback_on_step_end`
parameter, so the route that reports every other pipeline's denoise steps
never fired for one: a `generating` phase, then nothing at all for however
many minutes the loop took. From outside, a slow run and a hung one were the
same thing. What those pipelines do have is a tqdm bar inside the denoise
block, and every advance of it is a step.
"""

import copy

import pytest
from PIL import Image
from tqdm.auto import tqdm

from dw.events import RunContext, WorkflowCancelled

from .test_phase_events import _pipeline_workflow, _run


class FakeOutput:
    def __init__(self):
        self.images = [Image.new("RGB", (2, 2))]


class FakeDenoiseBlock:
    """The block that owns the loop, the way a modular denoise block does:
    it opens its own bar and advances it once per step."""

    def __init__(self, steps=3, iterate=False):
        self.steps = steps
        self.iterate = iterate

    def progress_bar(self, iterable=None, total=None):
        if iterable is not None:
            return tqdm(iterable, disable=True)
        return tqdm(total=total, disable=True)

    def run(self):
        if self.iterate:
            for _ in self.progress_bar(range(self.steps)):
                pass
            return
        with self.progress_bar(total=self.steps) as bar:
            for _ in range(self.steps):
                bar.update()


class FakeBlocks:
    def __init__(self, denoise):
        self.sub_blocks = {"denoise": denoise}


class FakeModularPipeline:
    """No `callback_on_step_end` in the signature, and a public `blocks`
    that hands back a copy - both true of the real thing, and the second is
    why the patch has to go through `_blocks`."""

    def __init__(self, steps=3, iterate=False):
        self.denoise = FakeDenoiseBlock(steps, iterate)
        self._blocks = FakeBlocks(self.denoise)

    @property
    def blocks(self):
        return copy.deepcopy(self._blocks)

    def __call__(self, prompt=None, num_inference_steps=None, generator=None):
        self.denoise.run()
        return FakeOutput()


class FakePipelineWithBoth:
    """A classic pipeline: it has a bar *and* takes a callback. Only one of
    the two may report, or every step is counted twice."""

    def __init__(self):
        self._num_timesteps = 3

    def progress_bar(self, iterable=None, total=None):
        return tqdm(total=total, disable=True)

    def __call__(
        self,
        prompt=None,
        num_inference_steps=None,
        generator=None,
        callback_on_step_end=None,
    ):
        with self.progress_bar(total=3) as bar:
            for i in range(3):
                callback_on_step_end(self, i, 0, {})
                bar.update()
        return FakeOutput()


def _events(fake, on_event=None):
    collected = []

    def sink(event):
        collected.append(event)
        if on_event is not None:
            on_event(context, event)

    context = RunContext(on_event=sink)
    _run(_pipeline_workflow(), context, fake=fake)
    return collected


def _steps(events):
    return [
        (event["step"], event["total_steps"])
        for event in events
        if event["event"] == "pipeline_step"
    ]


def test_every_advance_of_the_bar_is_reported():
    assert _steps(_events(FakeModularPipeline())) == [(1, 3), (2, 3), (3, 3)]


def test_an_iterated_bar_reports_too():
    """The other shape a block writes the loop in."""
    assert _steps(_events(FakeModularPipeline(iterate=True))) == [
        (1, 3),
        (2, 3),
        (3, 3),
    ]


def test_decoding_follows_the_last_step():
    events = _events(FakeModularPipeline())

    names = [
        event.get("phase") if event["event"] == "phase" else event["event"]
        for event in events
        if event["event"] in ("phase", "pipeline_step")
    ]
    assert names == [
        "loading",
        "generating",
        "pipeline_step",
        "pipeline_step",
        "pipeline_step",
        "decoding",
    ]


def test_a_cancel_lands_inside_the_loop_rather_than_after_it():
    """The same checkpoint the callback route has: without it a cancel on a
    modular pipeline waits out the whole generation."""

    def cancel_on_first_step(context, event):
        if event["event"] == "pipeline_step":
            context.cancel()

    with pytest.raises(WorkflowCancelled):
        _events(FakeModularPipeline(steps=20), on_event=cancel_on_first_step)


def test_a_pipeline_that_takes_a_callback_is_not_counted_twice():
    assert _steps(_events(FakePipelineWithBoth())) == [(1, 3), (2, 3), (3, 3)]


def test_the_pipeline_is_handed_back_unpatched():
    """This process keeps a loaded pipeline between runs - a wrapper left on
    it would report into the run that has finished."""
    fake = FakeModularPipeline()

    _events(fake)

    assert "progress_bar" not in vars(fake.denoise)
    assert isinstance(fake.denoise.progress_bar(total=1), tqdm)
