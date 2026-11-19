"""
Unit tests for scheduler configuration - type replacement and sigma shift.

A pipeline can carry more than one scheduler: MiniMax-H3 steps video and audio
latents down two schedules inside a single transformer call, and their shifts
are set independently. No GPU is involved - the schedulers here are stand-ins
recording what was asked of them.
"""

import pytest

from dw.pipeline_processors.components import load_and_configure_scheduler


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


class TestSchedulerLoadLog:
    """The scheduler-load line reaches job events, not just the server log (#1487)"""

    @staticmethod
    def _capture(monkeypatch):
        events = []
        monkeypatch.setattr(
            "dw.pipeline_processors.components.emit_log",
            lambda message, **fields: events.append((message, fields)),
        )
        return events

    @staticmethod
    def _type():
        class FlowMatchEulerDiscreteScheduler:
            @staticmethod
            def from_config(config, **kwargs):
                return FakeScheduler()

        return FlowMatchEulerDiscreteScheduler

    def test_a_scheduler_type_emits_one_log_event(self, monkeypatch):
        events = self._capture(monkeypatch)
        pipeline = FakePipeline(scheduler=FakeScheduler())

        load_and_configure_scheduler(
            {"configuration": {"scheduler_type": self._type()}}, pipeline
        )

        assert len(events) == 1
        message, fields = events[0]
        assert message == "Loading scheduler: FlowMatchEulerDiscreteScheduler"
        assert fields["component"] == "scheduler"
        assert fields["scheduler_type"] == "FlowMatchEulerDiscreteScheduler"
        assert "from_config_args" not in fields

    def test_from_config_args_keys_are_named_not_values(self, monkeypatch):
        events = self._capture(monkeypatch)
        pipeline = FakePipeline(scheduler=FakeScheduler())

        load_and_configure_scheduler(
            {
                "configuration": {"scheduler_type": self._type()},
                "from_config_args": {"use_karras_sigmas": "secret-value"},
            },
            pipeline,
        )

        message, fields = events[0]
        assert fields["from_config_args"] == ["use_karras_sigmas"]
        assert "use_karras_sigmas" in message
        assert "secret-value" not in message

    def test_the_audio_scheduler_names_its_component(self, monkeypatch):
        events = self._capture(monkeypatch)
        pipeline = FakePipeline(
            scheduler=FakeScheduler(), audio_scheduler=FakeScheduler()
        )

        load_and_configure_scheduler(
            {"configuration": {"scheduler_type": self._type()}},
            pipeline,
            "audio_scheduler",
        )

        assert events[0][1]["component"] == "audio_scheduler"
        assert events[0][0].startswith("Loading audio_scheduler:")

    def test_a_type_without_a_name_falls_back_to_str(self, monkeypatch):
        events = self._capture(monkeypatch)

        class Nameless:
            def from_config(self, config, **kwargs):
                return FakeScheduler()

            def __str__(self):
                return "nameless-type"

        pipeline = FakePipeline(scheduler=FakeScheduler())
        load_and_configure_scheduler(
            {"configuration": {"scheduler_type": Nameless()}}, pipeline
        )

        assert events[0][1]["scheduler_type"] == "nameless-type"

    def test_shift_only_and_none_emit_nothing(self, monkeypatch):
        events = self._capture(monkeypatch)
        pipeline = FakePipeline(scheduler=FakeScheduler())

        load_and_configure_scheduler({"shift": 6}, pipeline)
        load_and_configure_scheduler(None, pipeline)

        assert events == []
