"""
Unit tests for scheduler configuration - type replacement and sigma shift.

A pipeline can carry more than one scheduler: MiniMax-H3 steps video and audio
latents down two schedules inside a single transformer call, and their shifts
are set independently. No GPU is involved - the schedulers here are stand-ins
recording what was asked of them.
"""

import pytest

from dw.pipeline_processors.pipeline import load_and_configure_scheduler


class FakeScheduler:
    """A scheduler that takes a shift, the way MiniMaxH3Scheduler does."""

    def __init__(self, shift=12.0):
        self.config = {"shift": shift}
        self.shift = shift

    def set_shift(self, shift):
        self.shift = shift


class ShiftlessScheduler:
    """A scheduler with no sigma shift to set - most of them."""

    def __init__(self):
        self.config = {}


class FakePipeline:
    def __init__(self, **components):
        for name, component in components.items():
            setattr(self, name, component)


class TestSigmaShift:
    def test_a_shift_reaches_the_scheduler(self):
        pipeline = FakePipeline(scheduler=FakeScheduler())

        load_and_configure_scheduler({"shift": 6}, pipeline)

        assert pipeline.scheduler.shift == 6.0

    def test_a_shift_needs_no_scheduler_type(self):
        """A definition that only lowers the shift keeps the pipeline's own type"""
        scheduler = FakeScheduler()
        pipeline = FakePipeline(scheduler=scheduler)

        load_and_configure_scheduler({"shift": 3}, pipeline)

        assert pipeline.scheduler is scheduler

    def test_the_audio_scheduler_is_configured_separately(self):
        pipeline = FakePipeline(
            scheduler=FakeScheduler(12.0), audio_scheduler=FakeScheduler(3.0)
        )

        load_and_configure_scheduler({"shift": 6}, pipeline)
        load_and_configure_scheduler({"shift": 2}, pipeline, "audio_scheduler")

        assert pipeline.scheduler.shift == 6.0
        assert pipeline.audio_scheduler.shift == 2.0

    def test_configuring_one_leaves_the_other_alone(self):
        pipeline = FakePipeline(
            scheduler=FakeScheduler(12.0), audio_scheduler=FakeScheduler(3.0)
        )

        load_and_configure_scheduler({"shift": 6}, pipeline)

        assert pipeline.audio_scheduler.shift == 3.0

    def test_a_string_shift_is_accepted(self):
        """Schema validation runs before variable substitution, so a resolved
        'variable:' reference can arrive as a string"""
        pipeline = FakePipeline(scheduler=FakeScheduler())

        load_and_configure_scheduler({"shift": "6.0"}, pipeline)

        assert pipeline.scheduler.shift == 6.0

    def test_a_scheduler_without_a_shift_raises(self):
        pipeline = FakePipeline(scheduler=ShiftlessScheduler())

        with pytest.raises(ValueError, match="does not take a sigma shift"):
            load_and_configure_scheduler({"shift": 6}, pipeline)

    def test_an_unloaded_scheduler_raises(self):
        """A modular pipeline registers an unloaded component as None"""
        pipeline = FakePipeline(scheduler=None)

        with pytest.raises(ValueError, match="has not loaded it"):
            load_and_configure_scheduler({"shift": 6}, pipeline)


class TestSchedulerReplacement:
    def test_a_scheduler_type_replaces_the_scheduler(self):
        replacement = FakeScheduler()

        class SchedulerType:
            @staticmethod
            def from_config(config, **kwargs):
                SchedulerType.seen = (config, kwargs)
                return replacement

        pipeline = FakePipeline(scheduler=FakeScheduler(12.0))
        load_and_configure_scheduler(
            {
                "configuration": {"scheduler_type": SchedulerType},
                "from_config_args": {"use_karras_sigmas": True},
            },
            pipeline,
        )

        assert pipeline.scheduler is replacement
        assert SchedulerType.seen[1] == {"use_karras_sigmas": True}

    def test_a_replacement_can_also_take_a_shift(self):
        """The shift lands on the scheduler the replacement installed"""
        replacement = FakeScheduler(12.0)

        class SchedulerType:
            @staticmethod
            def from_config(config, **kwargs):
                return replacement

        pipeline = FakePipeline(scheduler=FakeScheduler(12.0))
        load_and_configure_scheduler(
            {"configuration": {"scheduler_type": SchedulerType}, "shift": 6},
            pipeline,
        )

        assert pipeline.scheduler is replacement
        assert replacement.shift == 6.0

    def test_no_definition_is_a_no_op(self):
        scheduler = FakeScheduler(12.0)
        pipeline = FakePipeline(scheduler=scheduler)

        load_and_configure_scheduler(None, pipeline)

        assert pipeline.scheduler.shift == 12.0
