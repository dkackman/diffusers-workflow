"""The introspection contracts a UI and the submission warnings rely on."""

from dw.introspection import (
    describe_pipeline,
    unknown_call_arguments,
    workflow_argument_warnings,
    load_pipeline_class,
)

import pytest


def test_describe_merges_signature_and_docstring():
    description = describe_pipeline("ZImagePipeline")
    parameters = {p["name"]: p for p in description["parameters"]}
    assert parameters["num_inference_steps"]["default"] == 50
    assert parameters["num_inference_steps"]["required"] is False
    # docstring detail rides along when present
    assert "description" in parameters["prompt"]
    # self and *args/**kwargs never appear as parameters
    assert "self" not in parameters


def test_load_pipeline_class_rejects_non_bare_names():
    for bad in ("os.path", "../etc", "", "diffusers.ZImagePipeline", "no_such_thing"):
        with pytest.raises(ValueError):
            load_pipeline_class(bad)


def test_warnings_are_never_wrong():
    """The pre-load check only flags what is provably unaccepted."""
    # a class that cannot be resolved yields no warnings, not an error
    assert unknown_call_arguments("NoSuchPipeline", ["whatever"]) == []

    # escaped and dotted component types are left alone
    workflow = {
        "id": "w",
        "steps": [
            {
                "name": "escaped",
                "pipeline": {
                    "configuration": {"component_type": "{MockPipeline}"},
                    "arguments": {"bogus": 1},
                },
            },
            {
                "name": "dotted",
                "pipeline": {
                    "configuration": {"component_type": "some.module.Pipeline"},
                    "arguments": {"bogus": 1},
                },
            },
            {
                "name": "real",
                "pipeline": {
                    "configuration": {"component_type": "ZImagePipeline"},
                    "arguments": {"prompt": "p", "guidance_scael": 3},
                },
            },
        ],
    }
    warnings = workflow_argument_warnings(workflow)
    assert len(warnings) == 1
    assert "guidance_scael" in warnings[0] and "real" in warnings[0]


def test_a_reference_naming_no_variable_is_warned_about_before_anything_loads():
    """An agent wrote `variable:base_prompt, clear sky` expecting
    interpolation; validation passed and the run failed at resolution. The
    warning names where it sits, what it asked for, and what is declared."""
    workflow = {
        "variables": {"base_prompt": "a lighthouse", "steps": 4},
        "seed": "variable:seed",
        "steps": [
            {
                "name": "clear",
                "pipeline": {
                    "configuration": {"component_type": "ZImagePipeline"},
                    "arguments": {
                        "prompt": "variable:base_prompt, clear sky",
                        "num_inference_steps": "variable:steps",
                    },
                },
            }
        ],
    }
    warnings = workflow_argument_warnings(workflow)
    assert len(warnings) == 2
    assert warnings[0] == (
        "seed: 'variable:seed' names no declared variable; "
        "declared: base_prompt, steps"
    )
    assert warnings[1].startswith(
        "steps[0].pipeline.arguments.prompt: 'variable:base_prompt, clear sky' "
        "names no declared variable - a reference is the whole value"
    )


def test_a_declared_reference_anywhere_in_the_definition_is_not_warned_about():
    workflow = {
        "variables": {"p": "x", "n": 1},
        "steps": [
            {
                "name": "s",
                "task": {
                    "command": "nonexistent",
                    "arguments": {"a": ["variable:p", {"b": "variable:n"}]},
                },
            }
        ],
    }
    assert not any(
        "names no declared variable" in w for w in workflow_argument_warnings(workflow)
    )


def test_describe_class_init_target_reads_constructors():
    from dw.introspection import describe_class

    description = describe_class("BitsAndBytesConfig", target="init")
    names = [p["name"] for p in description["parameters"]]
    assert "load_in_4bit" in names and "bnb_4bit_quant_type" in names


def test_describe_class_load_target_merges_curated_knobs():
    from dw.introspection import describe_class

    description = describe_class("AutoencoderKL", target="load")
    names = [p["name"] for p in description["parameters"]]
    for knob in ("torch_dtype", "variant", "subfolder", "revision"):
        assert knob in names
    # the model path is the editor's own field, never a discovered argument
    assert "pretrained_model_name_or_path" not in names
    # **kwargs loading means nothing is provably wrong - no warnings possible
    assert description["accepts_kwargs"] is True


def test_class_enumeration_by_kind():
    from dw.introspection import list_classes

    assert "AutoencoderKL" in list_classes("models")
    assert "FlowMatchEulerDiscreteScheduler" in list_classes("schedulers")
    assert "BitsAndBytesConfig" in list_classes("quantization")
    with pytest.raises(ValueError):
        list_classes("nonsense")


def test_allowlist_admits_sdnq_and_nothing_else():
    from dw.introspection import load_allowed_class

    pytest.importorskip("sdnq")
    assert load_allowed_class("sdnq.SDNQConfig").__name__ == "SDNQConfig"
    for blocked in ("os.path", "subprocess.Popen", "dw.security.validate_path"):
        with pytest.raises(ValueError):
            load_allowed_class(blocked)


def test_scheduler_compatibles_reported_when_present():
    from dw.introspection import describe_class

    description = describe_class("EulerDiscreteScheduler", target="init")
    assert "DDIMScheduler" in description.get("compatibles", [])


def _concat_step(**arguments):
    return {
        "variables": {"trim": 0},
        "steps": [
            {
                "name": "cut",
                "task": {
                    "command": "concat_videos",
                    "arguments": {"videos": ["a.mp4", "b.mp4"], **arguments},
                },
            }
        ],
    }


def test_a_crossfade_with_nothing_trimmed_is_warned_about():
    """concat_videos draws its crossfade from the trimmed-off material, so at
    `trim_frames: 0` a `crossfade_ms` reads as active and does nothing - the
    cut-based templates all sit there. Only a value the author wrote is
    warned about; the argument's own default is not their mistake."""
    warnings = workflow_argument_warnings(_concat_step(crossfade_ms=200))
    assert len(warnings) == 1
    assert "crossfade_ms" in warnings[0] and "trim_frames" in warnings[0]
    assert "audio_bleed_ms" in warnings[0]

    warnings = workflow_argument_warnings(
        _concat_step(crossfade_ms=200, trim_frames=0)
    )
    assert len(warnings) == 1


def test_a_crossfade_over_a_trim_is_not_warned_about():
    assert workflow_argument_warnings(_concat_step(crossfade_ms=200, trim_frames=1)) == []
    assert workflow_argument_warnings(_concat_step(trim_frames=0)) == []
    assert workflow_argument_warnings(_concat_step(crossfade_ms=0)) == []
    # A referenced trim is unknown until the run - do not guess
    assert workflow_argument_warnings(
        _concat_step(crossfade_ms=200, trim_frames="variable:trim")
    ) == []
