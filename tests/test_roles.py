import pytest

from dw.roles import FINAL, INTERMEDIATE, annotate_roles, run_inputs, step_roles
from dw.schema import load_schema, validate_data_all


def _step(name, result=None, args=None):
    step = {"name": name, "task": {"command": "noop", "arguments": args or {}}}
    if result is not None:
        step["result"] = result
    return step


def _entry(step, *files, **extra):
    return {"step": step, "files": list(files), **extra}


def test_leaf_media_step_is_final_and_consumed_one_intermediate():
    steps = [
        _step("image"),
        _step("video", args={"image": "previous_result:image"}),
    ]
    manifest = [_entry("image", "a.png"), _entry("video", "b.mp4")]
    assert step_roles(steps, manifest) == {"image": INTERMEDIATE, "video": FINAL}


def test_property_reference_counts_as_consumption():
    steps = [
        _step("stems"),
        _step("mix", args={"audio": "previous_result:stems.vocals"}),
    ]
    manifest = [_entry("stems", "v.wav"), _entry("mix", "m.wav")]
    assert step_roles(steps, manifest)["stems"] == INTERMEDIATE


def test_text_only_consumer_does_not_demote():
    steps = [
        _step("video"),
        _step("check", args={"video": "previous_result:video"}),
    ]
    manifest = [_entry("video", "v.mp4"), _entry("check", "report.json")]
    roles = step_roles(steps, manifest)
    assert roles == {"video": FINAL, "check": FINAL}


def test_unsaved_consumer_still_demotes():
    steps = [
        _step("image"),
        _step("latent", args={"image": "previous_result:image"}),
        _step("video", args={"latent": "previous_result:latent"}),
    ]
    manifest = [_entry("image", "a.png"), _entry("video", "v.mp4")]
    roles = step_roles(steps, manifest)
    assert roles["image"] == INTERMEDIATE
    assert roles["latent"] == INTERMEDIATE
    assert roles["video"] == FINAL


def test_explicit_role_wins_over_inference_and_subfolder():
    steps = [
        _step("image", result={"role": "final", "subfolder": "intermediate"}),
        _step("video", args={"image": "previous_result:image"}),
    ]
    manifest = [_entry("image", "a.png"), _entry("video", "v.mp4")]
    assert step_roles(steps, manifest)["image"] == FINAL


def test_subfolder_convention_including_nested():
    steps = [
        _step("a", result={"subfolder": "intermediate/shots"}),
        _step("b", result={"subfolder": "final/act-1"}),
    ]
    manifest = [_entry("a", "x.png"), _entry("b", "y.png")]
    assert step_roles(steps, manifest) == {"a": INTERMEDIATE, "b": FINAL}


def test_other_subfolder_name_falls_through_to_inference():
    steps = [_step("a", result={"subfolder": "shots"})]
    assert step_roles(steps, [_entry("a", "x.png")]) == {"a": FINAL}


def test_from_previous_result_key_counts():
    steps = [
        _step("seg"),
        _step("inpaint", args={"mask": {"from_previous_result": "seg"}}),
    ]
    manifest = [_entry("seg", "m.png"), _entry("inpaint", "o.png")]
    assert step_roles(steps, manifest)["seg"] == INTERMEDIATE


def test_dotted_step_name_resolves_by_longest_match():
    steps = [
        _step("v1.0"),
        _step("up", args={"image": "previous_result:v1.0"}),
    ]
    manifest = [_entry("v1.0", "a.png"), _entry("up", "b.png")]
    assert step_roles(steps, manifest)["v1.0"] == INTERMEDIATE


def test_for_each_members_are_steps_of_their_own():
    steps = [
        _step("shot@0"),
        _step("shot@1"),
        _step(
            "join",
            args={"videos": ["previous_result:shot@0", "previous_result:shot@1"]},
        ),
    ]
    manifest = [
        _entry("shot@0", "0.mp4"),
        _entry("shot@1", "1.mp4"),
        _entry("join", "j.mp4"),
    ]
    roles = step_roles(steps, manifest)
    assert roles["shot@0"] == roles["shot@1"] == INTERMEDIATE
    assert roles["join"] == FINAL


def test_child_entries_take_parent_step_role():
    steps = [
        _step("child", args={}),
        _step("score", args={"video": "previous_result:child"}),
    ]
    manifest = [
        _entry("render", "r.mp4", parent_step="child"),
        _entry("score", "s.mp4"),
    ]
    annotated = annotate_roles(steps, manifest)
    assert annotated[0]["role"] == INTERMEDIATE
    assert annotated[1]["role"] == FINAL


def test_annotate_roles_copies_and_defaults_unknown_to_final():
    manifest = [_entry("ghost", "g.png")]
    annotated = annotate_roles([], manifest)
    assert annotated[0]["role"] == FINAL
    assert "role" not in manifest[0]


def test_run_inputs_dedups_in_first_use_order():
    steps = [
        _step("a", args={"image": "asset:cast/iris.png", "mask": "output:w/r/m.png"}),
        _step("b", args={"images": ["asset:cast/iris.png", "asset:bg.png"]}),
    ]
    assert run_inputs(steps) == [
        {"reference": "asset:cast/iris.png", "kind": "asset"},
        {"reference": "output:w/r/m.png", "kind": "output"},
        {"reference": "asset:bg.png", "kind": "asset"},
    ]


@pytest.mark.parametrize(
    "role,ok",
    [
        ("final", True),
        ("intermediate", True),
        ("variable:r", True),
        ("input", False),
        ("keeper", False),
    ],
)
def test_schema_role_values(role, ok):
    workflow = {
        "id": "t",
        "steps": [
            {
                "name": "s",
                "task": {"command": "noop", "arguments": {}},
                "result": {"role": role},
            }
        ],
    }
    errors = validate_data_all(workflow, load_schema("workflow"))
    role_errors = [e for e in errors if "role" in (e["path"] or "")]
    assert (not role_errors) == ok


def test_role_basis_survives_realization_of_the_steps():
    from dw.roles import role_basis

    steps = [
        {"name": "image", "result": {"subfolder": "x"}},
        {"name": "video", "arguments": {"image": "previous_result:image"}},
    ]
    basis = role_basis(steps)
    steps[1]["arguments"]["image"] = object()  # what a run does in place
    manifest = [_entry("image", "a.png"), _entry("video", "b.mp4")]
    assert step_roles(None, manifest, basis) == {"image": INTERMEDIATE, "video": FINAL}


def test_fallback_roles():
    from dw.roles import fallback_roles

    entries = [
        {"step": "a", "files": ["a.png"], "subfolder": "intermediate/x"},
        {"step": "b", "files": ["b.png"], "subfolder": ""},
    ]
    assert [e.get("role") for e in fallback_roles(entries)] == [INTERMEDIATE, None]
    assert fallback_roles([{"step": "a", "files": ["a.png"]}])[0]["role"] == FINAL
    assert fallback_roles([]) == []
