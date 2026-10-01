"""The three `previous_result:` scanners, characterized.

`referenced_result_names` (step_cache), `find_previous_result_refs` and the
reference-error scan (previous_results) walk the same shapes and differ in
three ways each caller depends on: a `from_previous_result` value is recorded
bare, only step_cache also reads a prefixed spelling inside it, and the
error scan drops a `variable:` value. These pin that, whichever code
implements it.
"""

import pytest

from dw.previous_results import (
    _collect_reference_paths,
    find_previous_result_refs,
)
from dw.step_cache import referenced_result_names


def reference_paths(value):
    found = {}
    _collect_reference_paths(value, (), found)
    return found


SHAPES = {
    "prefixed string": ("previous_result:a", {"a"}, {(): "a"}, {(): "a"}),
    "bare from_": (
        {"from_previous_result": "a"},
        {"a"},
        {("from_previous_result",): "a"},
        {("from_previous_result",): "a"},
    ),
    "prefixed from_": (
        {"from_previous_result": "previous_result:a"},
        {"previous_result:a", "a"},
        {("from_previous_result",): "previous_result:a"},
        {("from_previous_result",): "previous_result:a"},
    ),
    "variable from_": (
        {"from_previous_result": "variable:v"},
        {"variable:v"},
        {("from_previous_result",): "variable:v"},
        {},
    ),
    "nested lists and dicts": (
        {"x": [{"y": "previous_result:a.b"}, ["previous_result:c"]], "z": 1},
        {"a.b", "c"},
        {("x", 0, "y"): "a.b", ("x", 1, 0): "c"},
        {("x", 0, "y"): "a.b", ("x", 1, 0): "c"},
    ),
    "other keys beside from_": (
        {
            "from_previous_result": "a",
            "scale": 2,
            "also": "previous_result:b",
            "inner": {"from_previous_result": "c"},
        },
        {"a", "b", "c"},
        {
            ("from_previous_result",): "a",
            ("also",): "b",
            ("inner", "from_previous_result"): "c",
        },
        {
            ("from_previous_result",): "a",
            ("also",): "b",
            ("inner", "from_previous_result"): "c",
        },
    ),
    "non-string leaves": (
        [1, 2.5, None, True, {"k": None}, "plain", "variable:v"],
        set(),
        {},
        {},
    ),
    "non-string from_ is walked": (
        {"from_previous_result": ["previous_result:a", 3]},
        {"a"},
        {("from_previous_result", 0): "a"},
        {("from_previous_result", 0): "a"},
    ),
    "from_ list entry in a list": (
        [{"from_previous_result": "a"}, {"from_previous_result": "b"}],
        {"a", "b"},
        {(0, "from_previous_result"): "a", (1, "from_previous_result"): "b"},
        {(0, "from_previous_result"): "a", (1, "from_previous_result"): "b"},
    ),
}


@pytest.mark.parametrize("shape", SHAPES)
def test_referenced_result_names(shape):
    value, names, _, _ = SHAPES[shape]
    assert referenced_result_names([value]) == names


@pytest.mark.parametrize("shape", SHAPES)
def test_find_previous_result_refs(shape):
    value, _, refs, _ = SHAPES[shape]
    assert find_previous_result_refs(value) == refs


@pytest.mark.parametrize("shape", SHAPES)
def test_reference_error_scan(shape):
    value, _, _, paths = SHAPES[shape]
    assert reference_paths(value) == paths
