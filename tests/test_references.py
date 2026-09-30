"""dw/references.py - the one place a reference prefix is spelled."""

import ast
import pathlib

import pytest

from dw import references
from dw.references import (
    ASSET,
    DEFERRED,
    GATHER,
    ITEM,
    PREVIOUS_RESULT,
    SUBSTITUTED,
    UNRESOLVED,
    VARIABLE,
    author_index,
    is_ref,
    make_ref,
    ref_name,
    render_path,
)


@pytest.mark.parametrize(
    "kind, value, expected",
    [
        (ASSET, "asset:iris.png", True),
        (ASSET, "output:x/latest/a.png", False),
        (ASSET, "an asset: in prose", False),
        (ASSET, None, False),
        (ASSET, 3, False),
        (UNRESOLVED, "gather:shot", True),
        (UNRESOLVED, "asset:iris.png", False),
    ],
)
def test_is_ref_is_a_prefix_test_on_strings_only(kind, value, expected):
    assert is_ref(kind, value) is expected


def test_ref_name_strips_its_own_prefix_only():
    assert ref_name(VARIABLE, "variable:num_frames") == "num_frames"
    assert ref_name(VARIABLE, "item:len") is None
    assert ref_name(VARIABLE, 7) is None


@pytest.mark.parametrize("helper", [ref_name, make_ref])
def test_a_prefix_set_is_refused_where_one_prefix_is_needed(helper):
    with pytest.raises(TypeError):
        helper(UNRESOLVED, "variable:x")


def test_make_ref_round_trips_through_ref_name():
    assert ref_name(ASSET, make_ref(ASSET, "gyre/frames/web.mp4")) == (
        "gyre/frames/web.mp4"
    )


def test_the_three_sets_nest():
    # a check that skips DEFERRED values skips every UNRESOLVED one, and an
    # UNRESOLVED skip covers every SUBSTITUTED one
    assert set(SUBSTITUTED) < set(UNRESOLVED) < set(DEFERRED)
    assert set(UNRESOLVED) - set(SUBSTITUTED) == {PREVIOUS_RESULT, GATHER}
    assert ITEM in SUBSTITUTED


@pytest.mark.parametrize(
    "source_indices, index, expected",
    [([0, 0, 1], 1, 0), ([0, 0, 1], 2, 1), (None, 4, 4), ([], 2, 2), ([5], 3, 3)],
)
def test_author_index_maps_an_expanded_step_back_to_its_template(
    source_indices, index, expected
):
    assert author_index(source_indices, index) == expected


def test_references_imports_nothing_from_dw():
    tree = ast.parse(pathlib.Path(references.__file__).read_text())
    imported = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert imported == []


def test_render_path_writes_a_json_path_the_way_schema_errors_do():
    path = ["steps", 3, "task", "arguments", "videos", 1]
    assert render_path(path) == "steps[3].task.arguments.videos[1]"
