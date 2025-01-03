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
