import pytest
from dw.variables import replace_variables, set_variables, VariableNotFoundError


def test_replace_variables_in_dict():
    data = {"key1": "variable:test", "key2": "static_value"}
    variables = {"test": "replaced_value"}

    result = replace_variables(data, variables)
    assert result["key1"] == "replaced_value"
    assert result["key2"] == "static_value"


def test_replace_variables_in_list():
    data = ["variable:test", "static_value"]
    variables = {"test": "replaced_value"}

    result = replace_variables(data, variables)
    assert result[0] == "replaced_value"
    assert result[1] == "static_value"


def test_replace_variables_missing():
    data = {"key": "variable:missing"}
    variables = {"other": 1}

    with pytest.raises(VariableNotFoundError) as exc_info:
        replace_variables(data, variables)
    message = str(exc_info.value)
    assert "Variable <missing> not found" in message
    assert "other" in message


def test_replace_variables_does_not_mutate_input():
    data = {"key1": "variable:test", "key2": ["variable:test", "static"]}
    original = {"key1": "variable:test", "key2": ["variable:test", "static"]}
    variables = {"test": "replaced_value"}

    result = replace_variables(data, variables)

    # The input structure is left exactly as passed in
    assert data == original
    # ...while the returned structure has the substitution applied
    assert result["key1"] == "replaced_value"
    assert result["key2"][0] == "replaced_value"


def test_set_variables():
    variables = {"int_var": 1, "str_var": "test", "bool_var": True}
    values = {"int_var": "2", "str_var": "new_test", "bool_var": "false"}

    set_variables(values, variables)
    assert variables["int_var"] == 2
    assert variables["str_var"] == "new_test"
    assert variables["bool_var"] is False


def test_set_variables_boolean_true():
    variables = {"flag": False}
    values = {"flag": "true"}

    set_variables(values, variables)
    assert variables["flag"] is True


def test_set_variables_type_conversion():
    variables = {"count": 0, "ratio": 0.0}
    values = {"count": "42", "ratio": "3.14"}

    set_variables(values, variables)
    assert variables["count"] == 42
    assert abs(variables["ratio"] - 3.14) < 0.01


def test_replace_variables_nested():
    data = {"outer": {"inner": {"value": "variable:nested_var"}}}
    variables = {"nested_var": "replaced"}

    result = replace_variables(data, variables)
    assert result["outer"]["inner"]["value"] == "replaced"


def test_replace_variables_in_nested_list():
    data = {"items": [{"name": "variable:item1"}, {"name": "variable:item2"}]}
    variables = {"item1": "first", "item2": "second"}

    result = replace_variables(data, variables)
    assert result["items"][0]["name"] == "first"
    assert result["items"][1]["name"] == "second"


def test_set_variables_invalid_name():
    from dw.security import SecurityError

    variables = {"valid_name": "value"}
    values = {"invalid!name": "value"}

    with pytest.raises(SecurityError):
        set_variables(values, variables)


@pytest.mark.parametrize("value", ["0", "no", "off", "No", "OFF"])
def test_set_variables_boolean_false_aliases(value):
    variables = {"flag": True}
    values = {"flag": value}

    set_variables(values, variables)
    assert variables["flag"] is False


@pytest.mark.parametrize("value", ["1", "yes", "on", "Yes", "ON"])
def test_set_variables_boolean_true_aliases(value):
    variables = {"flag": False}
    values = {"flag": value}

    set_variables(values, variables)
    assert variables["flag"] is True


def test_set_variables_boolean_invalid_raises():
    variables = {"upscale": False}
    values = {"upscale": "maybe"}

    with pytest.raises(ValueError) as exc_info:
        set_variables(values, variables)
    assert "upscale" in str(exc_info.value)


def test_set_variables_list_default_splits_on_comma():
    variables = {"items": ["default"]}
    values = {"items": "a,b"}

    set_variables(values, variables)
    assert variables["items"] == ["a", "b"]


def test_set_variables_list_default_single_value():
    variables = {"items": ["default"]}
    values = {"items": "single"}

    set_variables(values, variables)
    assert variables["items"] == ["single"]


def test_set_variables_numeric_conversion_failure_raises():
    variables = {"num_inference_steps": 25}
    values = {"num_inference_steps": "abc"}

    with pytest.raises(ValueError) as exc_info:
        set_variables(values, variables)
    message = str(exc_info.value)
    assert "num_inference_steps" in message
    assert "abc" in message
    assert "int" in message
    # The unconvertible value must never silently pass through unconverted -
    # the variable is left at its prior value once the error is raised.
    assert variables["num_inference_steps"] == 25


def test_set_variables_unknown_name_raises():
    variables = {"prompt": "a cat", "steps": 25}
    values = {"promt": "a dog"}

    with pytest.raises(ValueError) as exc_info:
        set_variables(values, variables)
    message = str(exc_info.value)
    assert "promt" in message
    assert "prompt" in message
    assert "steps" in message


def test_set_variables_string_override_of_a_dict_default_passes_through():
    """Media variables are commonly declared as {'location': ...}; a user
    overriding one with a plain path string (dw-run image=/tmp/cat.png)
    must not be coerced with dict('/tmp/cat.png') and refused."""
    variables = {"image": {"location": "https://example/x.png"}}
    set_variables({"image": "/tmp/cat.png"}, variables)
    assert variables["image"] == "/tmp/cat.png"


def test_set_variables_string_override_of_a_null_default_passes_through():
    variables = {"mask": None}
    set_variables({"mask": "masks/a.png"}, variables)
    assert variables["mask"] == "masks/a.png"
