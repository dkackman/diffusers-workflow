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
        "seed: 'variable:seed' names no declared variable; declared: base_prompt, steps"
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

    warnings = workflow_argument_warnings(_concat_step(crossfade_ms=200, trim_frames=0))
    assert len(warnings) == 1


def test_a_crossfade_over_a_trim_is_not_warned_about():
    assert (
        workflow_argument_warnings(_concat_step(crossfade_ms=200, trim_frames=1)) == []
    )
    assert workflow_argument_warnings(_concat_step(trim_frames=0)) == []
    assert workflow_argument_warnings(_concat_step(crossfade_ms=0)) == []
    # A referenced trim is unknown until the run - do not guess
    assert (
        workflow_argument_warnings(
            _concat_step(crossfade_ms=200, trim_frames="variable:trim")
        )
        == []
    )


def _bleed_step(variables=None, **arguments):
    return {
        "variables": {
            "audio_bleed_ms": 1800,
            "seam_fade_ms": None,
            **(variables or {}),
        },
        "steps": [
            {
                "name": "episode",
                "task": {
                    "command": "concat_videos",
                    "arguments": {
                        "videos": ["a.mp4", "b.mp4"],
                        "trim_frames": 0,
                        "audio_bleed_ms": "variable:audio_bleed_ms",
                        "seam_fade_ms": "variable:seam_fade_ms",
                        **arguments,
                    },
                },
            }
        ],
    }


def test_a_seam_fade_while_bleed_is_non_zero_is_warned_about():
    """concat_videos takes the bleed path, not the fade path, at a hard cut
    while audio_bleed_ms is non-zero, so a seam_fade_ms the caller passed
    alongside the template's own default bleed does nothing (#288)."""
    definition = _bleed_step()
    warnings = workflow_argument_warnings(definition, {"seam_fade_ms": 80})
    assert len(warnings) == 1
    assert "seam_fade_ms" in warnings[0] and "audio_bleed_ms" in warnings[0]

    # No caller arguments at all: seam_fade_ms resolves to its own null
    # default and is not "set", so this is silent
    assert workflow_argument_warnings(definition, None) == []

    # Caller zeroes the bleed alongside the fade: seam_fade_ms is live
    assert (
        workflow_argument_warnings(
            definition, {"seam_fade_ms": 80, "audio_bleed_ms": 0}
        )
        == []
    )


def test_a_bleed_gain_without_bleed_is_warned_about():
    """concat_videos applies audio_bleed_gain_db to the bled tail, so an
    audio_bleed_gain_db the caller passed with audio_bleed_ms at 0 - whether
    zeroed explicitly or just never set - does nothing (#290)."""
    definition = _concat_step(audio_bleed_ms=0, audio_bleed_gain_db=-6)
    warnings = workflow_argument_warnings(definition)
    assert len(warnings) == 1
    assert "audio_bleed_gain_db" in warnings[0] and "audio_bleed_ms" in warnings[0]

    # audio_bleed_ms simply omitted - the task's own default of 0 applies
    definition = _concat_step(audio_bleed_gain_db=-6)
    warnings = workflow_argument_warnings(definition)
    assert len(warnings) == 1
    assert "audio_bleed_gain_db" in warnings[0]

    # A non-zero bleed: the gain is live
    assert (
        workflow_argument_warnings(
            _concat_step(audio_bleed_ms=800, audio_bleed_gain_db=-6)
        )
        == []
    )

    # No gain passed at all: silent
    assert workflow_argument_warnings(_concat_step(audio_bleed_ms=0)) == []


def _dissolve_step(**arguments):
    return {
        "steps": [
            {
                "name": "cut",
                "task": {
                    "command": "dissolve_videos",
                    "arguments": {
                        "videos": ["a.mp4", "b.mp4"],
                        "dissolve_frames": 0,
                        **arguments,
                    },
                },
            }
        ],
    }


def test_a_match_levels_dbfs_without_match_levels_is_warned_about():
    """concat_videos and dissolve_videos only read match_levels_dbfs as the
    target inside match_levels() - called only when match_levels itself is
    truthy - so a caller who passes only the target dBFS and leaves
    match_levels unset (off by default) has the value silently dropped
    (#291)."""
    for step in (_concat_step, _dissolve_step):
        warnings = workflow_argument_warnings(step(match_levels_dbfs=-24))
        assert len(warnings) == 1
        assert "match_levels_dbfs" in warnings[0] and "match_levels" in warnings[0]

        # match_levels set: the target is live
        assert (
            workflow_argument_warnings(step(match_levels_dbfs=-24, match_levels="rms"))
            == []
        )

        # No target passed at all: silent
        assert workflow_argument_warnings(step(match_levels="rms")) == []
