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

from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks

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


class FakeSequentialBlocks(SequentialPipelineBlocks):
    """The dispatch a real `SequentialPipelineBlocks` performs: it walks its
    named sub-blocks in order, calling each with the pipeline and the state.
    `__call__` lives on the class, which is why the patch cannot be
    per-instance the way the progress-bar one is - and the real base class
    is what it subclasses, because only a sequence may be narrated by
    walking its sub-blocks."""

    def __init__(self, sub_blocks):
        self.sub_blocks = sub_blocks

    def __call__(self, pipeline, state):
        for block in self.sub_blocks.values():
            pipeline, state = block(pipeline, state)
        return pipeline, state


class FakeNamedBlock:
    def __init__(self, on_call=None):
        self.on_call = on_call

    def __call__(self, pipeline, state):
        if self.on_call is not None:
            self.on_call()
        return pipeline, state


class FakeBlockedPipeline:
    """A modular pipeline that runs its blocks the way the real one does:
    text encode, then reference encode, then the denoise loop."""

    def __init__(self, steps=3, on_encode=None):
        self.denoise = FakeDenoiseBlock(steps)
        self._blocks = FakeSequentialBlocks(
            {
                "text_encoder": FakeNamedBlock(),
                "vae_encoder": FakeNamedBlock(on_encode),
                "denoise": FakeNamedBlock(self.denoise.run),
            }
        )

    @property
    def blocks(self):
        return copy.deepcopy(self._blocks)

    def __call__(self, prompt=None, num_inference_steps=None, generator=None):
        self._blocks(self, None)
        return FakeOutput()


def _logs(events):
    return [event["message"] for event in events if event["event"] == "log"]


def test_each_block_of_the_lead_in_says_it_started():
    """#95: the encode that runs before the denoise loop emitted nothing, so
    a video reference - which takes minutes of it - looked exactly like a
    hang. The blocks have names, and naming each one as it starts is the
    difference between silence and 'it is encoding the reference'."""
    events = _events(FakeBlockedPipeline())

    assert _logs(events) == [
        "acme/model: text_encoder",
        "acme/model: vae_encoder",
        "acme/model: denoise",
    ]


def test_a_block_is_named_before_it_runs_rather_than_after():
    """After is no use: the whole point is the event that lands while the
    long block is still going, which is where the ten minutes go."""
    timeline = []
    pipeline = FakeBlockedPipeline(on_encode=lambda: timeline.append("encoding"))

    def record(context, event):
        if event["event"] == "log":
            timeline.append(event["message"])

    _events(pipeline, record)

    assert timeline == [
        "acme/model: text_encoder",
        "acme/model: vae_encoder",
        "encoding",
        "acme/model: denoise",
    ]


def test_the_block_dispatch_is_handed_back_unpatched():
    """The patch is on the class, so leaving it in place would outlive the
    run and report blocks into whatever context came next."""
    pipeline = FakeBlockedPipeline()
    original = FakeSequentialBlocks.__call__

    _events(pipeline)

    assert FakeSequentialBlocks.__call__ is original


class FakePipelineWithoutBlocks:
    """No step callback and no `_blocks` tree - the bar is the pipeline's
    own, the way a classic pipeline that predates the callback holds it."""

    def progress_bar(self, iterable=None, total=None):
        return tqdm(total=total, disable=True)

    def __call__(self, prompt=None, num_inference_steps=None, generator=None):
        with self.progress_bar(total=3) as bar:
            for _ in range(3):
                bar.update()
        return FakeOutput()


def test_a_pipeline_with_no_blocks_still_runs():
    """Not every pipeline without a step callback is modular: with no
    `_blocks` there is nothing to narrate, and its own bar still reports."""
    assert not hasattr(FakePipelineWithoutBlocks(), "_blocks")
    assert _steps(_events(FakePipelineWithoutBlocks())) == [(1, 3), (2, 3), (3, 3)]


class FakeConditionalBlocks:
    """What `AutoPipelineBlocks` does: it *picks* one sub-block on its
    inputs rather than running them all. Not a `SequentialPipelineBlocks`,
    and that is the whole point."""

    def __init__(self, sub_blocks, chosen):
        self.sub_blocks = sub_blocks
        self.chosen = chosen

    def __call__(self, pipeline, state):
        return self.sub_blocks[self.chosen](pipeline, state)


class FakeConditionalPipeline:
    def __init__(self):
        self.ran = []
        self.denoise = FakeDenoiseBlock(3)
        self._blocks = FakeConditionalBlocks(
            {
                "image_branch": FakeNamedBlock(lambda: self.ran.append("image")),
                "video_branch": FakeNamedBlock(lambda: self.ran.append("video")),
            },
            chosen="video_branch",
        )

    @property
    def blocks(self):
        return copy.deepcopy(self._blocks)

    def __call__(self, prompt=None, num_inference_steps=None, generator=None):
        self._blocks(self, None)
        return FakeOutput()


def test_a_container_that_chooses_one_block_is_left_alone():
    """Narrating by walking `sub_blocks` is only correct for a sequence. A
    conditional container runs one branch, so the same walk would run every
    branch of it - a wrong answer bought with a progress message."""
    pipeline = FakeConditionalPipeline()

    events = _events(pipeline)

    assert pipeline.ran == ["video"]
    assert _logs(events) == []
