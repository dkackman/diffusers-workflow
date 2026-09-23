"""Numeric domains on task arguments, refused before a run rather than
interpreted during one.

`slice_audio(num_frames=-10)` used to reach Python's slice semantics and
return the track minus its last ten frames, reported succeeded with no
warnings (#139); `resample_audio(target_sample_rate=0)` fell through to the
44100 Hz save default and re-headered the samples unresampled, 38% off in
duration (#140). Both are now refused twice over: statically by
validate_workflow for a literal, and by the command itself for a value that
arrived from a variable or an earlier step.
"""

import numpy
import pytest

from dw.task_domains import (
    TASK_ARGUMENT_DOMAINS,
    CLOSED_UNIT,
    NON_NEGATIVE,
    NON_POSITIVE,
    POSITIVE,
    as_number,
    task_argument_errors,
)
from dw.tasks.audio_utils import resample_audio, resample_waveform, slice_audio
from dw.workflow import Workflow


def tone(seconds=1.0, sample_rate=32000):
    return numpy.zeros((1, int(seconds * sample_rate)), dtype=numpy.float32)


def task_step(command, arguments, name="step"):
    return {
        "name": name,
        "task": {"command": command, "arguments": arguments},
        "result": {"content_type": "audio/wav"},
    }


def errors_for(command, arguments):
    definition = {"id": "domains", "steps": [task_step(command, arguments)]}
    return task_argument_errors(definition)


class TestTheRegistryNamesRealArguments:
    """A domain declared against an argument that no longer exists checks
    nothing, silently - so every entry is pinned to the signature the
    dispatch actually forwards into."""

    def test_every_command_is_a_task_command(self):
        from dw.introspection import describe_task

        for command in TASK_ARGUMENT_DOMAINS:
            assert describe_task(command)["name"] == command

    def test_every_argument_is_a_parameter_of_its_command(self):
        from dw.introspection import describe_task

        for command, domains in TASK_ARGUMENT_DOMAINS.items():
            parameters = {p["name"] for p in describe_task(command)["parameters"]}
            assert set(domains) <= parameters, command

    def test_every_domain_is_one_of_the_declared_kinds(self):
        for domains in TASK_ARGUMENT_DOMAINS.values():
            assert set(domains.values()) <= {
                POSITIVE,
                NON_NEGATIVE,
                NON_POSITIVE,
                CLOSED_UNIT,
            }


class TestAsNumber:
    def test_a_number_supplied_as_a_string_is_read(self):
        # A variable declared null carries no type, so a value given for it on
        # the command line arrives as a string
        assert as_number("-10") == -10.0

    def test_a_reference_is_left_to_another_pass(self):
        assert as_number("variable:num_frames") is None
        assert as_number("previous_result:shot") is None

    def test_a_bool_is_not_a_number(self):
        assert as_number(True) is None

    def test_a_fraction_is(self):
        from fractions import Fraction

        assert as_number(Fraction(24000, 1001)) == pytest.approx(23.976, abs=1e-3)


class TestTheStaticPass:
    def test_a_negative_frame_count_is_refused_at_its_path(self):
        errors = errors_for(
            "slice_audio",
            {"audio": "asset:bed.wav", "start_frame": 0, "num_frames": -10, "fps": 24},
        )
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].task.arguments.num_frames"
        assert "num_frames" in errors[0]["message"]

    def test_a_zero_target_rate_is_refused(self):
        errors = errors_for(
            "resample_audio", {"audio": "asset:bed.wav", "target_sample_rate": 0}
        )
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].task.arguments.target_sample_rate"

    def test_a_positive_target_lufs_is_refused(self):
        errors = errors_for(
            "normalize_audio", {"audio": "asset:bed.wav", "target_lufs": 3.0}
        )
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].task.arguments.target_lufs"

    def test_an_out_of_range_temperature_is_refused(self):
        errors = errors_for("grade", {"media": "asset:a.png", "temperature": 5.0})
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].task.arguments.temperature"
        assert "-1.0 and 1.0" in errors[0]["message"]

    def test_an_out_of_range_tint_is_refused(self):
        errors = errors_for("grade", {"media": "asset:a.png", "tint": -1.5})
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].task.arguments.tint"

    def test_a_boundary_temperature_is_fine(self):
        assert errors_for("grade", {"media": "asset:a.png", "temperature": 1.0}) == []

    def test_a_zero_offset_is_fine(self):
        assert (
            errors_for(
                "slice_audio",
                {"audio": "a.wav", "start_frame": 0, "num_frames": 24, "fps": 24},
            )
            == []
        )

    def test_a_hard_cut_is_fine(self):
        # trim_frames 0 and crossfade_ms 0 are both ordinary requests
        assert (
            errors_for(
                "concat_videos",
                {"videos": [], "trim_frames": 0, "crossfade_ms": 0},
            )
            == []
        )

    def test_a_referenced_value_is_not_this_passs_business(self):
        assert (
            errors_for(
                "slice_audio",
                {
                    "audio": "a.wav",
                    "num_frames": "variable:frames",
                    "fps": "item:fps",
                },
            )
            == []
        )

    def test_an_untouched_command_is_left_alone(self):
        assert errors_for("compose_text", {"parts": ["a"]}) == []

    def test_every_bad_argument_is_reported_not_just_the_first(self):
        errors = errors_for(
            "slice_audio",
            {"audio": "a.wav", "start_frame": -1, "num_frames": -1, "fps": 0},
        )
        assert {e["path"].rsplit(".", 1)[-1] for e in errors} == {
            "start_frame",
            "num_frames",
            "fps",
        }


class TestThroughValidateWorkflow:
    def _workflow(self, arguments):
        return Workflow(
            {
                "id": "domains",
                "steps": [task_step("slice_audio", arguments)],
            },
            "outputs",
            None,
        )

    def test_a_run_that_would_slice_backwards_never_starts(self):
        errors = self._workflow(
            {"audio": "asset:bed.wav", "start_frame": 0, "num_frames": -10, "fps": 24}
        ).validation_errors()
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.num_frames"]

    def test_a_good_slice_validates(self):
        assert (
            self._workflow(
                {
                    "audio": "asset:bed.wav",
                    "start_frame": 0,
                    "num_frames": 24,
                    "fps": 24,
                }
            ).validation_errors()
            == []
        )

    def test_a_bad_value_supplied_as_an_argument_is_caught(self):
        # The static pass runs on the substituted definition, so a caller's
        # argument is checked as it will run
        workflow = Workflow(
            {
                "id": "domains",
                "variables": {"frames": 24},
                "steps": [
                    task_step(
                        "slice_audio",
                        {
                            "audio": "asset:bed.wav",
                            "start_frame": 0,
                            "num_frames": "variable:frames",
                            "fps": 24,
                        },
                    )
                ],
            },
            "outputs",
            None,
        )
        assert workflow.validation_errors() == []
        errors = workflow.validation_errors({"frames": -10})
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.num_frames"]


class TestAtRunTime:
    """The static pass cannot see a computed value, so the command refuses
    too - and refuses with the same words."""

    def test_slice_audio_refuses_a_negative_count(self):
        with pytest.raises(ValueError, match="num_frames"):
            slice_audio(
                tone(), start_frame=0, num_frames=-10, fps=24, sample_rate=32000
            )

    def test_slice_audio_refuses_a_zero_length(self):
        with pytest.raises(ValueError, match="num_frames"):
            slice_audio(tone(), start_frame=0, num_frames=0, fps=24, sample_rate=32000)

    def test_slice_audio_refuses_a_negative_start(self):
        with pytest.raises(ValueError, match="start_seconds"):
            slice_audio(
                tone(), start_seconds=-1, duration_seconds=0.5, sample_rate=32000
            )

    def test_slice_audio_still_slices(self):
        track = slice_audio(
            tone(), start_frame=0, num_frames=24, fps=24, sample_rate=32000
        )
        assert track.audio.shape == (1, 32000)
        assert track.sample_rate == 32000

    def test_resample_audio_refuses_a_zero_rate(self):
        with pytest.raises(ValueError, match="target_sample_rate"):
            resample_audio(tone(), 0, sample_rate=32000)

    def test_resample_audio_refuses_a_negative_rate(self):
        with pytest.raises(ValueError, match="target_sample_rate"):
            resample_audio(tone(), -44100, sample_rate=32000)

    def test_resample_audio_still_resamples(self):
        track = resample_audio(tone(), 16000, sample_rate=32000)
        assert track.audio.shape == (1, 16000)
        assert track.sample_rate == 16000

    def test_a_rate_given_as_a_string_is_read(self):
        track = resample_audio(tone(), "16000", sample_rate=32000)
        assert track.sample_rate == 16000

    def test_resample_waveform_refuses_a_rate_that_is_not_one(self):
        # The conversion the chained-video code reaches directly, without the
        # task's argument handling
        with pytest.raises(ValueError, match="above zero"):
            resample_waveform(tone(), 32000, 0)
        with pytest.raises(ValueError, match="above zero"):
            resample_waveform(tone(), 0, 32000)

    def test_a_track_is_never_relabelled_at_a_rate_it_is_not_at(self):
        from dw.tasks.audio_utils import _as_track

        with pytest.raises(ValueError, match="not a rate"):
            _as_track(tone(), 0, "slice_audio")


class TestTheDomainIsVisibleOverTheApi:
    """`get_task` reported 'annotation: null' and no domain, so an agent
    composing a call had nothing to read - which is how the negative frame
    count got written in the first place."""

    def test_get_task_reports_the_domain_beside_the_parameter(self):
        from dw.introspection import describe_task

        parameters = {p["name"]: p for p in describe_task("slice_audio")["parameters"]}
        assert parameters["num_frames"]["domain"] == POSITIVE
        assert parameters["start_frame"]["domain"] == NON_NEGATIVE

    def test_an_unconstrained_parameter_carries_no_domain_key(self):
        from dw.introspection import describe_task

        parameters = {p["name"]: p for p in describe_task("slice_audio")["parameters"]}
        assert "domain" not in parameters["audio"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
