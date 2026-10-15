"""#926: an unbraced type name handed to a `*_dtype` key warns at validate."""

from dw.type_references import unbraced_type_warnings
from dw.validation import WARNING_CHECKS


def _definition(value, extra=None):
    return {
        "steps": [
            {
                "name": "s",
                "pipeline": {
                    "configuration": {"component_type": "{x}"},
                    "arguments": {"weights_dtype": value, **(extra or {})},
                },
            }
        ],
    }


def test_unbraced_name_warns_with_the_braced_spelling():
    warnings = unbraced_type_warnings(_definition("int8"))
    assert len(warnings) == 1
    assert "{int8}" in warnings[0] and "weights_dtype" in warnings[0]


def test_braced_name_is_quiet():
    assert unbraced_type_warnings(_definition("{int8}")) == []


def test_a_real_diffusers_type_is_quiet():
    assert unbraced_type_warnings(_definition("DiffusionPipeline")) == []


def test_dotted_name_is_left_to_the_trust_gate():
    assert unbraced_type_warnings(_definition("os.path")) == []


def test_media_reference_media_type_is_quiet():
    media = {"media_type": "image", "location": "x.png"}
    assert unbraced_type_warnings(_definition("{int8}", {"image": media})) == []


def test_registered_as_a_warning_check():
    assert "unbraced_type_warnings" in [c.name for c in WARNING_CHECKS]
