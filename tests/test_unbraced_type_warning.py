"""#926: an unbraced type name handed to a `*_dtype` key warns at validate."""

from dw.argument_warnings import workflow_argument_warnings


def _definition(default):
    return {
        "variables": {"weights_dtype": default},
        "steps": [
            {
                "name": "s",
                "pipeline": {
                    "configuration": {"component_type": "{x}"},
                    "arguments": {"weights_dtype": "variable:weights_dtype"},
                },
            }
        ],
    }


def test_unbraced_name_warns_with_the_braced_spelling():
    warnings = workflow_argument_warnings(
        _definition("{int4}"), {"weights_dtype": "int8"}
    )
    assert any("{int8}" in w and "weights_dtype" in w for w in warnings)


def test_braced_name_is_quiet():
    assert (
        workflow_argument_warnings(_definition("{int4}"), {"weights_dtype": "{int8}"})
        == []
    )


def test_a_real_diffusers_type_is_quiet():
    warnings = workflow_argument_warnings(
        _definition("{int4}"), {"weights_dtype": "DiffusionPipeline"}
    )
    assert not any("needs braces" in w for w in warnings)
