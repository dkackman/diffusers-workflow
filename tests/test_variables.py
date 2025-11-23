import pytest
from dw.variables import replace_variables, set_variables


def test_replace_variables_in_dict():
    data = {"key1": "variable:test", "key2": "static_value"}
    variables = {"test": "replaced_value"}

    replace_variables(data, variables)
    assert data["key1"] == "replaced_value"
    assert data["key2"] == "static_value"


def test_replace_variables_in_list():
    data = ["variable:test", "static_value"]
    variables = {"test": "replaced_value"}

    replace_variables(data, variables)
    assert data[0] == "replaced_value"
    assert data[1] == "static_value"


def test_replace_variables_missing():
    data = {"key": "variable:missing"}
    variables = {}

    with pytest.raises(Exception) as exc_info:
        replace_variables(data, variables)
    assert "Variable <missing> not found" in str(exc_info.value)


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
    data = {
        "outer": {
            "inner": {
                "value": "variable:nested_var"
            }
        }
    }
    variables = {"nested_var": "replaced"}
    
    replace_variables(data, variables)
    assert data["outer"]["inner"]["value"] == "replaced"


def test_replace_variables_in_nested_list():
    data = {
        "items": [
            {"name": "variable:item1"},
            {"name": "variable:item2"}
        ]
    }
    variables = {"item1": "first", "item2": "second"}
    
    replace_variables(data, variables)
    assert data["items"][0]["name"] == "first"
    assert data["items"][1]["name"] == "second"


def test_set_variables_invalid_name():
    from dw.security import SecurityError
    variables = {"valid_name": "value"}
    values = {"invalid!name": "value"}
    
    with pytest.raises(SecurityError):
        set_variables(values, variables)
