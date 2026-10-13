"""The gallery says which files of a run are its deliverable and which are
scratch: each entry's `role`, the listing's `role` filter, and the metadata's
`role` and `inputs` (dw/roles.py decides the roles)."""

import os

from dw.server.outputs import run_roles, split_run_name


def test_recorded_roles_reach_the_listing(gallery_client, output_root, write_run):
    write_run(
        output_root,
        "wf",
        "20261012-000000-aaaaaaaa",
        [
            {"step": "a", "files": ["a.png"], "role": "intermediate"},
            {"step": "b", "files": ["b.png"], "role": "final"},
        ],
        files=["a.png", "b.png"],
    )
    files = {f["label"]: f for f in gallery_client.get("/api/gallery").json()["files"]}
    assert files["a.png"]["role"] == "intermediate"
    assert files["b.png"]["role"] == "final"


def test_older_run_roles_inferred_from_realized_workflow(
    gallery_client, output_root, write_run
):
    realized = {
        "id": "wf",
        "steps": [
            {"name": "a", "task": {"command": "noop", "arguments": {}}},
            {
                "name": "b",
                "task": {
                    "command": "noop",
                    "arguments": {"image": "previous_result:a"},
                },
            },
        ],
    }
    write_run(
        output_root,
        "wf",
        "20261012-000001-bbbbbbbb",
        [{"step": "a", "files": ["a.png"]}, {"step": "b", "files": ["b.png"]}],
        realized=realized,
        files=["a.png", "b.png"],
    )
    files = {f["label"]: f for f in gallery_client.get("/api/gallery").json()["files"]}
    assert files["a.png"]["role"] == "intermediate"
    assert files["b.png"]["role"] == "final"


def test_role_filter(gallery_client, output_root, write_run):
    write_run(
        output_root,
        "wf",
        "20261012-000002-cccccccc",
        [
            {"step": "a", "files": ["a.png"], "role": "intermediate"},
            {"step": "b", "files": ["final/b.png"], "role": "final"},
        ],
        files=["a.png", "final/b.png"],
    )
    finals = gallery_client.get("/api/gallery", params={"role": "final"}).json()[
        "files"
    ]
    assert [f["label"] for f in finals] == ["b.png"]
    both = gallery_client.get(
        "/api/gallery", params={"role": ["final", "intermediate"]}
    ).json()["files"]
    assert len(both) == 2
    assert (
        gallery_client.get("/api/gallery", params={"role": "bogus"}).status_code == 400
    )


def test_flat_file_has_null_role_and_passes_final_filter(gallery_client, output_root):
    os.makedirs(os.path.join(output_root, "loose"), exist_ok=True)
    with open(os.path.join(output_root, "loose", "x.png"), "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
    entries = gallery_client.get("/api/gallery", params={"role": "final"}).json()[
        "files"
    ]
    loose = [f for f in entries if f["label"] == "x.png"]
    assert loose and loose[0].get("role") is None


def test_metadata_carries_role_and_inputs(gallery_client, output_root, write_run):
    write_run(
        output_root,
        "wf",
        "20261012-000003-dddddddd",
        [{"step": "b", "files": ["b.png"], "role": "final"}],
        files=["b.png"],
        inputs=[{"reference": "asset:iris.png", "kind": "asset"}],
    )
    meta = gallery_client.get(
        "/api/gallery/wf/20261012-000003-dddddddd/b.png/metadata"
    ).json()
    assert meta["role"] == "final"
    assert meta["inputs"] == [{"reference": "asset:iris.png", "kind": "asset"}]


def test_split_run_name(tmp_path):
    root = str(tmp_path)
    assert split_run_name(root, "wf/20261012-000003-dddddddd/final/b.png") == (
        os.path.join(root, "wf", "20261012-000003-dddddddd"),
        "final/b.png",
    )
    assert split_run_name(root, "loose/x.png") is None
    assert run_roles(os.path.join(root, "missing")) == {}


def test_metadata_for_a_subfolder_file_of_a_nested_identity_run(
    gallery_client, output_root, write_run
):
    write_run(
        output_root,
        "series/ep1",
        "20261012-000004-eeeeeeee",
        [{"step": "b", "files": ["final/b.png"], "role": "final"}],
        files=["final/b.png"],
        inputs=[{"reference": "output:wf/v1/a.png", "kind": "output"}],
    )
    meta = gallery_client.get(
        "/api/gallery/series/ep1/20261012-000004-eeeeeeee/final/b.png/metadata"
    ).json()
    assert meta["role"] == "final"
    assert meta["inputs"] == [{"reference": "output:wf/v1/a.png", "kind": "output"}]
