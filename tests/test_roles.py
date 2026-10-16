import json

import pytest

from dw.roles import (
    FINAL,
    INTERMEDIATE,
    ROLE_DECLARED_KEY,
    annotate_roles,
    consumed_outputs,
    listed_role,
    run_inputs,
    step_roles,
)
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


# A file a later run consumed (#928): read-time, across the runs of one
# output root, never rewriting a manifest

PRODUCER = "20261015-000000-aaaaaaaa"
OLDER = "20261014-000000-aaaaaaaa"
CONSUMER = "20261015-010000-bbbbbbbb"


def _producer(write_run, root, run_id=PRODUCER, version=1, **entry):
    run_dir = write_run(
        root,
        "shots",
        run_id,
        [{"step": "shot", "files": ["final/shot.png"], "role": "final", **entry}],
        files=["final/shot.png"],
    )
    if version != 1:
        manifest_path = f"{run_dir}/manifest.json"
        with open(manifest_path) as handle:
            manifest = json.load(handle)
        manifest["version"] = version
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle)
    return run_dir


def _consumer(write_run, root, reference, run_id=CONSUMER, kind="output", **kw):
    return write_run(
        root,
        "assembly",
        run_id,
        [{"step": "cut", "files": ["final/cut.mp4"], "role": "final"}],
        files=["final/cut.mp4"],
        inputs=[{"reference": reference, "kind": kind}],
        **kw,
    )


def test_annotate_roles_flags_a_role_declared_by_rule_1():
    steps = [
        _step("a", result={"role": "final"}),
        _step("b", result={"subfolder": "final"}),
        _step("c"),
    ]
    manifest = [_entry("a", "a.png"), _entry("b", "b.png"), _entry("c", "c.png")]
    annotated = {e["step"]: e for e in annotate_roles(steps, manifest)}
    assert annotated["a"][ROLE_DECLARED_KEY] is True
    assert ROLE_DECLARED_KEY not in annotated["b"]
    assert ROLE_DECLARED_KEY not in annotated["c"]


@pytest.mark.parametrize(
    "reference",
    [
        f"output:shots/{PRODUCER}/final/shot.png",
        "output:shots/v1/final/shot.png",
        "output:shots/latest/final/shot.png",
    ],
)
def test_a_file_a_later_run_read_is_consumed(tmp_path, write_run, reference):
    root = str(tmp_path)
    _producer(write_run, root)
    _consumer(write_run, root, reference)
    assert consumed_outputs(root) == {f"shots/{PRODUCER}/final/shot.png"}


def test_an_unpinned_latest_is_the_newest_run_before_the_consumer(tmp_path, write_run):
    root = str(tmp_path)
    _producer(write_run, root, OLDER)
    _consumer(write_run, root, "output:shots/latest/final/shot.png")
    # A run of the producer after the consumer is not what it read
    _producer(write_run, root, "20261015-020000-cccccccc", version=2)
    assert consumed_outputs(root) == {f"shots/{OLDER}/final/shot.png"}


def test_a_manifest_without_inputs_falls_back_to_the_realized_workflow(
    tmp_path, write_run
):
    root = str(tmp_path)
    _producer(write_run, root)
    write_run(
        root,
        "assembly",
        CONSUMER,
        [{"step": "cut", "files": ["cut.mp4"], "role": "final"}],
        realized={
            "steps": [
                {
                    "name": "cut",
                    "x": f"output:shots/{PRODUCER}/final/shot.png",
                }
            ]
        },
    )
    assert consumed_outputs(root) == {f"shots/{PRODUCER}/final/shot.png"}


def test_an_asset_kept_from_an_output_is_consumed(tmp_path, write_run):
    from dw.runs import record_kept_shots

    root = str(tmp_path / "outputs")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "hero.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (assets / "plain.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    record_kept_shots(
        str(assets),
        "hero.png",
        None,
        {"workspace": "studio", "source": f"shots/{PRODUCER}/final/shot.png"},
    )
    _producer(write_run, root)
    _consumer(write_run, root, "asset:hero.png", kind="asset")
    write_run(
        root,
        "other",
        "20261015-030000-dddddddd",
        [],
        inputs=[{"reference": "asset:plain.png", "kind": "asset"}],
    )

    def find(name):
        path = assets / name
        return str(path) if path.is_file() else None

    assert consumed_outputs(root, find, "studio") == {
        f"shots/{PRODUCER}/final/shot.png"
    }
    # Kept from another workspace's output: not this root's file
    assert consumed_outputs(root, find, "elsewhere") == set()


def test_a_consumer_under_another_root_consumes_nothing_here(tmp_path, write_run):
    here, there = str(tmp_path / "here"), str(tmp_path / "there")
    _producer(write_run, here)
    _producer(write_run, there)
    _consumer(write_run, there, f"output:shots/{PRODUCER}/final/shot.png")
    assert consumed_outputs(here) == set()


def test_an_unresolvable_reference_consumes_nothing(tmp_path, write_run):
    root = str(tmp_path)
    _producer(write_run, root)
    _consumer(write_run, root, "output:shots/v9/final/shot.png")
    _consumer(
        write_run,
        root,
        "output:../escape.png",
        run_id="20261015-040000-eeeeeeee",
    )
    assert consumed_outputs(root) == set()


@pytest.mark.parametrize(
    "role, declared, consumed, listed",
    [
        (FINAL, False, True, INTERMEDIATE),
        (FINAL, True, True, FINAL),
        (FINAL, False, False, FINAL),
        (INTERMEDIATE, False, True, INTERMEDIATE),
        (None, False, True, None),
    ],
)
def test_listed_role(role, declared, consumed, listed):
    assert listed_role(role, declared, consumed) == listed
