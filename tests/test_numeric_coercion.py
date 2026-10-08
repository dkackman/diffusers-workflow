"""One numeric coercion in dw/tasks (#774): validate and run agree.

`number_problem` is the rule; `whole_number` and `real_number` raise it at run
time and `domain_violation` reports it in the static pass, so a value one
refuses the other refuses too - for every argument the registry declares.
"""

import numpy
import pytest

import dw.tasks.task  # noqa: F401 - registers every command
from dw.media_types import AudioVideo
from dw.task_domains import (
    TASK_ARGUMENT_DOMAINS,
    TASK_WHOLE_NUMBER_ARGUMENTS,
    check_arguments,
    coerce_arguments,
    domain_violation,
    real_number,
    task_argument_errors,
    whole_number,
)
from dw.tasks.cuts import plan_cuts
from dw.tasks.fit import fit_to_model
from dw.tasks.trim import trim_video
from dw.tasks.windows import window_video

WHOLE = [
    (command, name)
    for command, names in sorted(TASK_WHOLE_NUMBER_ARGUMENTS.items())
    for name in names
]
REAL = [
    (command, name)
    for command, domains in sorted(TASK_ARGUMENT_DOMAINS.items())
    for name in domains
    if (command, name) not in WHOLE
]
ALL = WHOLE + REAL
IDS = lambda pairs: [f"{c}.{n}" for c, n in pairs]  # noqa: E731

VALUES = [
    3,
    3.0,
    "3",
    "3.0",
    "3.5",
    3.5,
    True,
    False,
    "abc",
    float("inf"),
    float("nan"),
    -1,
    0,
]


def _value_id(value):
    return repr(value)


class TestWholeTable:
    @pytest.mark.parametrize("command, name", WHOLE, ids=IDS(WHOLE))
    @pytest.mark.parametrize("value", [3, 3.0, "3", "3.0"], ids=_value_id)
    def test_an_integral_value_is_the_int(self, command, name, value):
        result = whole_number(value, name, command)
        assert result == 3 and type(result) is int

    @pytest.mark.parametrize("command, name", WHOLE, ids=IDS(WHOLE))
    @pytest.mark.parametrize("value", ["3.5", True, "abc", float("inf")], ids=_value_id)
    def test_anything_else_is_refused_by_name(self, command, name, value):
        with pytest.raises(ValueError) as error:
            whole_number(value, name, command)
        assert f"'{name}'" in str(error.value)
        assert "whole number" in str(error.value)


class TestRealTable:
    @pytest.mark.parametrize("command, name", REAL, ids=IDS(REAL))
    def test_numbers_pass(self, command, name):
        assert real_number("3", name, command) == 3.0
        assert real_number(3.5, name, command) == 3.5

    @pytest.mark.parametrize("command, name", REAL, ids=IDS(REAL))
    @pytest.mark.parametrize("value", [True, "abc", float("inf")], ids=_value_id)
    def test_anything_else_is_refused_by_name(self, command, name, value):
        with pytest.raises(ValueError) as error:
            real_number(value, name, command)
        assert f"'{name}'" in str(error.value)


def _static_refuses(command, name, value):
    whole = name in TASK_WHOLE_NUMBER_ARGUMENTS.get(command, ())
    domain = TASK_ARGUMENT_DOMAINS[command][name]
    return domain_violation(command, name, value, domain, whole=whole) is not None


def _run_refuses(command, name, value):
    coerce = (
        whole_number
        if name in TASK_WHOLE_NUMBER_ARGUMENTS.get(command, ())
        else real_number
    )
    try:
        coerced = coerce(value, name, command)
        check_arguments(command, **{name: coerced})
    except ValueError:
        return True
    return False


class TestValidateAndRunAgree:
    @pytest.mark.parametrize("command, name", ALL, ids=IDS(ALL))
    @pytest.mark.parametrize("value", VALUES, ids=_value_id)
    def test_both_refuse_or_both_accept(self, command, name, value):
        assert _static_refuses(command, name, value) == _run_refuses(
            command, name, value
        ), (command, name, value)


def _workflow_errors(command, arguments):
    return task_argument_errors(
        {
            "id": "numeric",
            "steps": [
                {
                    "name": "step",
                    "task": {"command": command, "arguments": arguments},
                    "result": {"content_type": "application/json"},
                }
            ],
        }
    )


def _mentioning(errors, name):
    return [e for e in errors if f"'{name}'" in e["message"] or name in e["path"]]


class TestStaticWorkflow:
    @pytest.mark.parametrize("command, name", ALL, ids=IDS(ALL))
    def test_a_bool_is_refused(self, command, name):
        errors = _mentioning(_workflow_errors(command, {name: True}), name)
        assert errors, (command, name)

    @pytest.mark.parametrize("command, name", WHOLE, ids=IDS(WHOLE))
    def test_a_fractional_string_is_refused_as_not_whole(self, command, name):
        errors = _mentioning(_workflow_errors(command, {name: "3.5"}), name)
        assert any("whole number" in e["message"] for e in errors), errors


class TestPlanPins:
    def test_analyze_beats_sample_rate_fraction_is_refused_at_validate(self):
        errors = _workflow_errors("analyze_beats", {"sample_rate": "22050.5"})
        assert any(
            "whole number" in e["message"] and "sample_rate" in e["message"]
            for e in errors
        ), errors

    def test_analyze_beats_sample_rate_fraction_is_refused_at_run(self):
        from dw.tasks.beats import _coerce_arguments

        with pytest.raises(ValueError, match="whole number"):
            _coerce_arguments("22050.5", None, 60, 200, None)

    def test_plan_cuts_integral_modulus_string_has_no_validate_error(self):
        errors = _workflow_errors("plan_cuts", {"modulus": "17.0"})
        assert not _mentioning(errors, "modulus"), errors


# -- the code running ------------------------------------------------------


def _same(a, b):
    a = getattr(a, "frames", a)
    b = getattr(b, "frames", b)
    return numpy.array_equal(numpy.asarray(a), numpy.asarray(b))


def _pil_clip(count, fps=24):
    from PIL import Image

    frames = [Image.new("RGB", (4, 4), (i, 0, 0)) for i in range(count)]
    return AudioVideo(frames, None, None, fps=fps)


def _float_clip(count=5, width=64, height=48):
    frames = numpy.random.default_rng(0).random((count, height, width, 3))
    return AudioVideo(frames.astype(numpy.float32), None, None, fps=24)


class TestRealPaths:
    @pytest.mark.parametrize("num_frames", [3.0, "3.0"], ids=_value_id)
    def test_window_video_integral_num_frames(self, num_frames):
        clip = _pil_clip(20)
        assert _same(window_video(clip, 1, num_frames, 1), window_video(clip, 1, 3, 1))

    def test_window_video_fractional_num_frames(self):
        with pytest.raises(ValueError, match="whole number"):
            window_video(_pil_clip(20), 1, 3.5, 1)

    @pytest.mark.parametrize("num_frames", [3.0, "3.0"], ids=_value_id)
    def test_trim_video_integral_num_frames(self, num_frames):
        clip = _pil_clip(20)
        assert _same(trim_video(clip, 2, num_frames), trim_video(clip, 2, 3))

    def test_trim_video_fractional_num_frames(self):
        with pytest.raises(ValueError, match="whole number"):
            trim_video(_pil_clip(20), 2, 3.5)

    def test_fit_to_model_integral_floats(self):
        clip = _float_clip()
        floats = fit_to_model(clip, 32.0, 32.0, 5.0)["video"]
        ints = fit_to_model(clip, 32, 32, 5)["video"]
        assert _same(floats, ints)

    def test_fit_to_model_fractional_width(self):
        with pytest.raises(ValueError, match="whole number"):
            fit_to_model(_float_clip(), 32.5, 32, 5)

    def test_plan_cuts_integral_modulus_string(self):
        chunks = [
            {"start": 4.0, "end": 7.0, "text": " Walking in the moonlight"},
            {"start": 8.0, "end": 11.0, "text": " Dancing with the stars"},
        ]
        arguments = {
            "transcript": {"text": "x", "chunks": chunks},
            "duration_s": 20.0,
        }
        assert plan_cuts(**arguments, modulus="17.0") == plan_cuts(
            **arguments, modulus=17
        )

    def test_plan_cuts_fractional_modulus(self):
        with pytest.raises(ValueError, match="whole number"):
            plan_cuts(
                transcript={"text": "x", "chunks": []}, duration_s=20.0, modulus=17.5
            )


def _run_task(command, arguments):
    from dw.tasks.task import Task

    return Task({"command": command, "arguments": {}}, "cpu").run(arguments)


def _gradient():
    from PIL import Image

    pixels = numpy.tile(numpy.arange(0, 256, 4, dtype=numpy.uint8), (16, 1))
    return Image.fromarray(numpy.stack([pixels] * 3, axis=-1))


class TestCoerceArguments:
    """Every numeric argument reaches its handler as a number (#774 bounce)."""

    def test_a_numeric_string_becomes_a_number(self):
        coerced = coerce_arguments("grade", {"exposure": "0.5", "contrast": "1"})
        assert coerced == {"exposure": 0.5, "contrast": 1}
        assert isinstance(coerced["exposure"], float)

    def test_a_whole_number_argument_is_an_int(self):
        coerced = coerce_arguments("slice_audio", {"num_frames": "17.0"})
        assert coerced["num_frames"] == 17 and isinstance(coerced["num_frames"], int)

    def test_fps_stays_exact(self):
        from fractions import Fraction

        coerced = coerce_arguments("slice_audio", {"fps": "23.976"})
        assert coerced["fps"] == Fraction(2997, 125)

    def test_list_entries_are_read_one_by_one_and_refs_pass_through(self):
        coerced = coerce_arguments("mix_audio", {"gains": ["0.5", 1, "variable:x"]})
        assert coerced["gains"] == [0.5, 1, "variable:x"]

    def test_an_undeclared_argument_and_none_are_left_alone(self):
        arguments = {"media": "x", "exposure": None}
        assert coerce_arguments("grade", arguments) == arguments

    @pytest.mark.parametrize("value", [True, False, "abc", "nan", "inf"], ids=_value_id)
    def test_a_non_number_names_the_argument(self, value):
        with pytest.raises(ValueError, match="'exposure'"):
            coerce_arguments("grade", {"exposure": value})


class TestGradeExposure:
    """The bounce: grade.exposure had no domain, so validate passed what run failed on."""

    @pytest.mark.parametrize("value", [True, False, "abc", "nan", "inf"], ids=_value_id)
    def test_validate_refuses(self, value):
        errors = _mentioning(_workflow_errors("grade", {"exposure": value}), "exposure")
        assert errors, value
        assert not any(".." in e["message"] for e in errors), errors

    @pytest.mark.parametrize("value", [True, False, "abc", "nan", "inf"], ids=_value_id)
    def test_run_refuses(self, value):
        with pytest.raises(ValueError, match="'exposure'"):
            _run_task("grade", {"media": _gradient(), "exposure": value})

    def test_a_string_number_grades_like_the_number(self):
        image = _gradient()
        from_string = _run_task("grade", {"media": image, "exposure": "0.5"})
        control = _run_task("grade", {"media": image, "exposure": 0.5})
        assert _same(from_string, control)
        assert not _same(control, image)
