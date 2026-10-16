"""Copying runs and assets between workspaces (#853): the engine, the route and the MCP tool."""

import json
import os
import re
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from dw.notes import read_notes, set_file_note
from dw.runs import (
    RUN_ID_PATTERN,
    record_kept_shots,
    resolve_output_reference,
    run_versions,
)
from dw.security import InvalidInputError, PathTraversalError, SecurityError
from dw.server.app import create_app
from dw.server.jobs import JobManager
from dw.workspace import (
    ConfiguredWorkspace,
    Workspace,
    create_workspace,
)
from dw.workspace_copy import CopyRefused, copy_between
from dw_mcp.client import DwClient, api_path
from dw_mcp.workspaces import copy_to_workspace

from .test_server import ScriptedWorkerManager, success_script

R1 = "20260101-000000-aaaaaaaa"
R2 = "20260102-000000-bbbbbbbb"
R3 = "20260103-000000-cccccccc"


def test_run_ids_match_the_pattern():
    for run_id in (R1, R2, R3):
        assert RUN_ID_PATTERN.match(run_id)
        assert re.match(r"\d{8}-\d{6}-[0-9a-f]{8}$", run_id)


@pytest.fixture
def root(tmp_path):
    return Workspace(tmp_path / "studio", "flag").ensure()


@pytest.fixture
def src(root):
    return create_workspace(root, "src")


@pytest.fixture
def dst(root):
    return create_workspace(root, "dst")


def make_run(
    ws,
    identity,
    run_id,
    version=1,
    inputs=(),
    workflow_file=None,
    steps=None,
    notes=None,
    content=b"png-bytes",
):
    run_dir = os.path.join(ws.outputs, *identity.split("/"), run_id)
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "img.png"), "wb") as f:
        f.write(content)
    manifest = {
        "run_id": run_id,
        "version": version,
        "inputs": [{"reference": r, "kind": "output"} for r in inputs],
        "workflow": {"file": workflow_file or "relative.json"},
    }
    if steps is not None:
        manifest["steps"] = steps
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    if notes is not None:
        with open(os.path.join(run_dir, "notes.json"), "w") as f:
            json.dump(notes, f)
    return run_dir


def read_json(path):
    with open(path) as f:
        return json.load(f)


def nothing_written(ws):
    return not os.path.exists(ws.outputs) or not any(os.scandir(ws.outputs))


class TestHappyPath:
    def test_one_run_is_copied(self, src, dst):
        make_run(src, "wf", R1)
        reply = copy_between(src, dst, [f"wf/{R1}"])
        assert set(reply) == {
            "source",
            "destination",
            "copied",
            "already_present",
            "renumbered",
            "warnings",
            "bytes_linked",
            "bytes_copied",
        }
        assert reply["source"] == "src"
        assert reply["destination"] == "dst"
        assert reply["copied"] == {"runs": [f"wf/{R1}"], "assets": []}
        assert reply["already_present"] == []
        assert reply["renumbered"] == {}
        for name in ("img.png", "manifest.json"):
            assert os.path.isfile(os.path.join(dst.outputs, "wf", R1, name))

    def test_a_bare_workflow_copies_all_its_runs(self, src, dst):
        make_run(src, "wf", R1, 1)
        make_run(src, "wf", R2, 2)
        reply = copy_between(src, dst, ["wf"])
        assert reply["copied"]["runs"] == [f"wf/{R1}", f"wf/{R2}"]

    def test_a_multi_segment_identity(self, src, dst):
        make_run(src, "ltx2/Gyre", R1)
        reply = copy_between(src, dst, [f"ltx2/Gyre/{R1}"])
        assert reply["copied"]["runs"] == [f"ltx2/Gyre/{R1}"]
        assert os.path.isfile(os.path.join(dst.outputs, "ltx2", "Gyre", R1, "img.png"))

    def test_a_bare_multi_segment_identity(self, src, dst):
        make_run(src, "ltx2/Gyre", R1)
        reply = copy_between(src, dst, ["ltx2/Gyre"])
        assert reply["copied"]["runs"] == [f"ltx2/Gyre/{R1}"]

    def test_an_output_prefix_is_accepted(self, src, dst):
        make_run(src, "wf", R1)
        reply = copy_between(src, dst, [f"output:wf/{R1}"])
        assert reply["copied"]["runs"] == [f"wf/{R1}"]

    def test_repeating_a_copy_is_idempotent(self, src, dst):
        make_run(src, "wf", R1)
        copy_between(src, dst, [f"wf/{R1}"])
        again = copy_between(src, dst, [f"wf/{R1}"])
        assert again["already_present"] == [f"wf/{R1}"]
        assert again["copied"]["runs"] == []
        assert "no_jobs" not in again["warnings"]


class TestRenumbering:
    def test_a_collision_renumbers_in_run_id_order(self, src, dst):
        make_run(dst, "wf", "20251231-000000-dddddddd", 1)
        make_run(src, "wf", R2, 2)
        make_run(src, "wf", R1, 1)
        make_run(src, "wf", R3, 3)
        # v1 collides, so every copied run of the identity is renumbered
        reply = copy_between(src, dst, ["wf"])
        assert reply["renumbered"] == {
            f"wf/{R1}": {"from": 1, "to": 2},
            f"wf/{R2}": {"from": 2, "to": 3},
            f"wf/{R3}": {"from": 3, "to": 4},
        }
        copied = read_json(os.path.join(dst.outputs, "wf", R1, "manifest.json"))
        assert copied["version"] == 2
        versions = run_versions(os.path.join(dst.outputs, "wf"))
        assert versions[R1] == 2 and versions[R2] == 3 and versions[R3] == 4

    def test_two_runs_renumber_after_one(self, src, dst):
        make_run(dst, "wf", "20251231-000000-dddddddd", 1)
        make_run(src, "wf", R1, 1)
        make_run(src, "wf", R2, 2)
        reply = copy_between(src, dst, ["wf"])
        assert reply["renumbered"] == {
            f"wf/{R1}": {"from": 1, "to": 2},
            f"wf/{R2}": {"from": 2, "to": 3},
        }

    def test_no_collision_keeps_numbers(self, src, dst):
        make_run(dst, "wf", "20251231-000000-dddddddd", 1)
        make_run(src, "wf", R2, 2)
        reply = copy_between(src, dst, [f"wf/{R2}"])
        assert reply["renumbered"] == {}
        copied = read_json(os.path.join(dst.outputs, "wf", R2, "manifest.json"))
        assert copied["version"] == 2


class TestWarnings:
    def test_unresolved_output_refs(self, src, dst):
        make_run(dst, "other", R1)
        make_run(
            src,
            "wf",
            R2,
            inputs=["output:other/latest/img.png", "output:gone/latest/img.png"],
        )
        reply = copy_between(src, dst, [f"wf/{R2}"])
        unresolved = reply["warnings"]["unresolved_output_refs"]
        assert unresolved == {f"wf/{R2}": ["output:gone/latest/img.png"]}

    def test_all_refs_resolving_gives_no_warning(self, src, dst):
        make_run(dst, "other", R1)
        make_run(src, "wf", R2, inputs=["output:other/latest/img.png"])
        reply = copy_between(src, dst, [f"wf/{R2}"])
        assert "unresolved_output_refs" not in reply["warnings"]

    def test_absolute_paths_are_relative_to_the_source_root(self, src, dst, tmp_path):
        inside = os.path.join(src.outputs, "wf", "earlier", "img.png")
        workflow = os.path.join(src.workflows, "wf.json")
        make_run(
            src,
            "wf",
            R1,
            workflow_file=workflow,
            steps=[{"step": "s", "files": [inside, "relative/img.png"]}],
        )
        reply = copy_between(src, dst, [f"wf/{R1}"])
        paths = reply["warnings"]["absolute_paths"][f"wf/{R1}"]
        assert paths == sorted(["outputs/wf/earlier/img.png", "workflows/wf.json"])
        assert str(tmp_path) not in json.dumps(reply)

    def test_a_path_outside_the_source_root_is_not_reported(self, src, dst):
        make_run(src, "wf", R1, workflow_file="/somewhere/else/wf.json")
        reply = copy_between(src, dst, [f"wf/{R1}"])
        assert "absolute_paths" not in reply["warnings"]

    def test_no_jobs_note_after_a_copy(self, src, dst):
        make_run(src, "wf", R1)
        reply = copy_between(src, dst, [f"wf/{R1}"])
        assert "no_jobs" in reply["warnings"]


class TestRefusals:
    @pytest.mark.parametrize("entry", ["../x", "/abs", "wf/../../x", "wf/.."])
    def test_traversal(self, src, dst, entry):
        make_run(src, "wf", R1)
        with pytest.raises(SecurityError):
            copy_between(src, dst, [entry])
        assert nothing_written(dst)

    @pytest.mark.parametrize("entry", ["wf/", f"wf/{R1}/", f"wf//{R1}", "output:wf/"])
    def test_an_empty_segment(self, src, dst, entry):
        make_run(src, "wf", R1)
        with pytest.raises(InvalidInputError):
            copy_between(src, dst, [entry])
        assert nothing_written(dst)

    def test_a_symlinked_run_folder(self, src, dst, tmp_path):
        make_run(src, "elsewhere", R1)
        os.makedirs(os.path.join(src.outputs, "wf"))
        os.symlink(
            os.path.join(src.outputs, "elsewhere", R1),
            os.path.join(src.outputs, "wf", R1),
        )
        with pytest.raises(PathTraversalError):
            copy_between(src, dst, [f"wf/{R1}"])
        assert nothing_written(dst)

    def test_a_symlinked_identity_folder(self, src, dst):
        make_run(src, "real", R1)
        os.symlink(
            os.path.join(src.outputs, "real"), os.path.join(src.outputs, "linked")
        )
        with pytest.raises(PathTraversalError):
            copy_between(src, dst, [f"linked/{R1}"])
        assert nothing_written(dst)

    def test_a_symlink_inside_a_run(self, src, dst, tmp_path):
        run_dir = make_run(src, "wf", R1)
        os.symlink(os.path.join(run_dir, "img.png"), os.path.join(run_dir, "link.png"))
        with pytest.raises(PathTraversalError):
            copy_between(src, dst, [f"wf/{R1}"])
        assert nothing_written(dst)

    def test_a_missing_run(self, src, dst):
        make_run(src, "wf", R1)
        with pytest.raises(CopyRefused) as raised:
            copy_between(src, dst, [f"wf/{R2}"])
        assert raised.value.status == 404
        assert nothing_written(dst)

    def test_a_missing_workflow(self, src, dst):
        with pytest.raises(CopyRefused) as raised:
            copy_between(src, dst, ["nope"])
        assert raised.value.status == 404

    def test_same_workspace(self, src):
        make_run(src, "wf", R1)
        with pytest.raises(InvalidInputError):
            copy_between(src, src, [f"wf/{R1}"])

    def test_empty_runs(self, src, dst):
        with pytest.raises(InvalidInputError):
            copy_between(src, dst, [])

    def configured(self, ws):
        return ConfiguredWorkspace(ws.workflows, ws.assets, ws.outputs, ws.prompts)

    def test_configured_source(self, src, dst):
        make_run(src, "wf", R1)
        with pytest.raises(CopyRefused) as raised:
            copy_between(self.configured(src), dst, [f"wf/{R1}"])
        assert raised.value.status == 409
        assert nothing_written(dst)

    def test_configured_destination(self, src, dst):
        make_run(src, "wf", R1)
        with pytest.raises(CopyRefused) as raised:
            copy_between(src, self.configured(dst), [f"wf/{R1}"])
        assert raised.value.status == 409
        assert nothing_written(dst)

    def test_an_active_run_refuses_all_or_nothing(self, src, dst):
        make_run(src, "wf", R1, 1)
        make_run(src, "wf", R2, 2)
        with pytest.raises(CopyRefused) as raised:
            copy_between(src, dst, ["wf"], active_runs=[f"wf/{R2}"])
        assert raised.value.status == 409
        assert raised.value.conflicts == [f"wf/{R2}"]
        assert nothing_written(dst)
        assert not os.path.exists(os.path.join(dst.outputs, "wf", R1))


class TestResolution:
    def test_output_references_resolve_in_the_destination(self, src, dst):
        make_run(src, "wf", R1)
        copy_between(src, dst, ["wf"])
        latest = resolve_output_reference("output:wf/latest/img.png", dst.outputs)
        pinned = resolve_output_reference(f"output:wf/{R1}/img.png", dst.outputs)
        expected = os.path.join(os.path.realpath(dst.outputs), "wf", R1, "img.png")
        assert latest == expected
        assert pinned == expected
        assert os.path.isfile(pinned)


class TestIndependence:
    def test_notes_are_not_shared(self, src, dst):
        source_run = make_run(src, "wf", R1, notes={"note": "original"})
        copy_between(src, dst, [f"wf/{R1}"])
        copied = os.path.join(dst.outputs, "wf", R1, "notes.json")
        original = os.path.join(source_run, "notes.json")
        assert os.stat(copied).st_ino != os.stat(original).st_ino
        with open(copied, "w") as f:
            json.dump({"note": "edited"}, f)
        assert read_json(original) == {"note": "original"}

    def test_a_renumbered_manifest_leaves_the_source_alone(self, src, dst):
        make_run(dst, "wf", "20251231-000000-dddddddd", 1)
        source_run = make_run(src, "wf", R1, 1)
        copy_between(src, dst, [f"wf/{R1}"])
        copied = os.path.join(dst.outputs, "wf", R1, "manifest.json")
        original = os.path.join(source_run, "manifest.json")
        assert read_json(copied)["version"] == 2
        assert read_json(original)["version"] == 1
        assert os.stat(copied).st_ino != os.stat(original).st_ino


class TestMediaLinking:
    def test_media_is_hard_linked(self, src, dst):
        source_run = make_run(src, "wf", R1, content=b"x" * 100)
        reply = copy_between(src, dst, [f"wf/{R1}"])
        a = os.stat(os.path.join(source_run, "img.png"))
        b = os.stat(os.path.join(dst.outputs, "wf", R1, "img.png"))
        assert a.st_ino == b.st_ino
        assert reply["bytes_linked"] >= 100

    def test_a_failed_link_falls_back_to_a_copy(self, src, dst, monkeypatch):
        source_run = make_run(src, "wf", R1, content=b"x" * 100)

        def refuse(*args, **kwargs):
            raise OSError("cross-device")

        monkeypatch.setattr("dw.workspace_copy.os.link", refuse)
        reply = copy_between(src, dst, [f"wf/{R1}"])
        target = os.path.join(dst.outputs, "wf", R1, "img.png")
        source_file = os.path.join(source_run, "img.png")
        assert os.stat(target).st_ino != os.stat(source_file).st_ino
        with open(target, "rb") as a, open(source_file, "rb") as b:
            assert a.read() == b.read()
        assert reply["bytes_linked"] == 0
        assert reply["bytes_copied"] >= 100


def make_asset(
    ws, name, content=b"asset-bytes", note=None, shots=None, provenance=None
):
    path = os.path.join(ws.assets, *name.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    if note is not None:
        set_file_note(os.path.dirname(path), os.path.basename(path), note)
    if shots is not None or provenance is not None:
        record_kept_shots(
            os.path.dirname(path), os.path.basename(path), shots, provenance
        )
    return path


def no_assets(ws):
    return not os.path.exists(ws.assets) or not any(os.scandir(ws.assets))


SHOTS = [{"start": 0.0, "end": 1.5}]
PROVENANCE = {"job_id": "j1", "workspace": "src", "run": f"wf/{R1}"}


class TestAssets:
    def test_a_file_keeps_its_path(self, src, dst):
        source_file = make_asset(src, "cast/iris.png")
        reply = copy_between(src, dst, assets=["cast/iris.png"])
        target = os.path.join(dst.assets, "cast", "iris.png")
        assert reply["copied"] == {"runs": [], "assets": ["cast/iris.png"]}
        assert os.stat(target).st_ino == os.stat(source_file).st_ino
        assert reply["bytes_linked"] == len(b"asset-bytes")

    def test_an_asset_prefix_is_accepted(self, src, dst):
        make_asset(src, "iris.png")
        reply = copy_between(src, dst, assets=["asset:iris.png"])
        assert reply["copied"]["assets"] == ["iris.png"]

    def test_a_folder_copies_every_file_under_it(self, src, dst):
        make_asset(src, "proj/a.png", note="first")
        make_asset(src, "proj/deep/b.wav")
        make_asset(src, "other.png")
        reply = copy_between(src, dst, assets=["proj/"])
        assert reply["copied"]["assets"] == ["proj/a.png", "proj/deep/b.wav"]
        assert not os.path.exists(os.path.join(dst.assets, "other.png"))
        # The folder's sidecar travels per file, not as a file of its own
        assert read_json(os.path.join(dst.assets, "proj", "notes.json"))["files"] == {
            "a.png": "first"
        }

    def test_a_folder_without_a_slash_is_a_folder(self, src, dst):
        make_asset(src, "proj/a.png")
        reply = copy_between(src, dst, assets=["proj"])
        assert reply["copied"]["assets"] == ["proj/a.png"]

    def test_an_upload_stays_under_uploads(self, src, dst):
        make_asset(src, "uploads/abc.png")
        reply = copy_between(src, dst, assets=["uploads/abc.png"])
        assert reply["copied"]["assets"] == ["uploads/abc.png"]
        assert os.path.isfile(os.path.join(dst.assets, "uploads", "abc.png"))

    def test_json_is_copied_not_linked(self, src, dst):
        source_file = make_asset(src, "data/cues.json", content=b"{}")
        copy_between(src, dst, assets=["data/cues.json"])
        target = os.path.join(dst.assets, "data", "cues.json")
        assert os.stat(target).st_ino != os.stat(source_file).st_ino

    def test_an_identical_file_is_already_present(self, src, dst):
        make_asset(src, "iris.png")
        make_asset(dst, "iris.png")
        reply = copy_between(src, dst, assets=["iris.png"])
        assert reply["copied"]["assets"] == []
        assert reply["already_present"] == ["iris.png"]

    def test_repeating_a_copy_is_idempotent(self, src, dst):
        make_asset(src, "proj/a.png", note="n")
        copy_between(src, dst, assets=["proj/"])
        reply = copy_between(src, dst, assets=["proj/"])
        assert reply["copied"]["assets"] == []
        assert reply["already_present"] == ["proj/a.png"]

    def test_a_different_file_refuses_everything(self, src, dst):
        make_run(src, "wf", R1)
        make_asset(src, "proj/a.png")
        make_asset(src, "proj/b.png", content=b"mine")
        make_asset(dst, "proj/b.png", content=b"theirs")
        with pytest.raises(CopyRefused) as refused:
            copy_between(src, dst, [f"wf/{R1}"], ["proj/"])
        assert refused.value.status == 409
        assert refused.value.conflicts == ["proj/b.png"]
        assert "proj/b.png" in str(refused.value)
        assert nothing_written(dst)
        assert not os.path.exists(os.path.join(dst.assets, "proj", "a.png"))
        with open(os.path.join(dst.assets, "proj", "b.png"), "rb") as f:
            assert f.read() == b"theirs"

    def test_an_active_run_and_a_different_file_are_named_together(self, src, dst):
        make_run(src, "wf", R1)
        make_asset(src, "a.png", content=b"mine")
        make_asset(dst, "a.png", content=b"theirs")
        with pytest.raises(CopyRefused) as refused:
            copy_between(src, dst, [f"wf/{R1}"], ["a.png"], active_runs=[f"wf/{R1}"])
        assert refused.value.conflicts == [f"wf/{R1}", "a.png"]

    def test_runs_and_assets_together(self, src, dst):
        make_run(src, "wf", R1)
        make_asset(src, "a.png")
        reply = copy_between(src, dst, [f"wf/{R1}"], ["a.png"])
        assert reply["copied"] == {"runs": [f"wf/{R1}"], "assets": ["a.png"]}

    def test_notes_and_kept_shots_travel(self, src, dst):
        make_asset(
            src, "proj/a.mp4", note="the keeper", shots=SHOTS, provenance=PROVENANCE
        )
        copy_between(src, dst, assets=["proj/a.mp4"])
        target_dir = os.path.join(dst.assets, "proj")
        assert read_notes(target_dir)["files"] == {"a.mp4": "the keeper"}
        manifest = read_json(os.path.join(target_dir, "manifest.json"))
        [entry] = manifest["steps"]
        assert entry["files"] == ["a.mp4"]
        assert entry["shots"] == SHOTS
        # Where it was kept from is history: it does not become the destination
        assert entry["provenance"] == PROVENANCE
        assert entry["provenance"]["workspace"] == "src"

    def test_a_destinations_own_entries_are_kept(self, src, dst):
        make_asset(src, "proj/a.png", note="from src")
        make_asset(dst, "proj/z.png", note="already here", shots=SHOTS)
        copy_between(src, dst, assets=["proj/a.png"])
        target_dir = os.path.join(dst.assets, "proj")
        assert read_notes(target_dir)["files"] == {
            "a.png": "from src",
            "z.png": "already here",
        }
        steps = read_json(os.path.join(target_dir, "manifest.json"))["steps"]
        assert [s["files"] for s in steps] == [["z.png"]]

    def test_sidecars_are_not_shared(self, src, dst):
        make_asset(src, "proj/a.mp4", note="original", shots=SHOTS)
        source_dir = os.path.join(src.assets, "proj")
        notes_before = read_json(os.path.join(source_dir, "notes.json"))
        manifest_before = read_json(os.path.join(source_dir, "manifest.json"))
        copy_between(src, dst, assets=["proj/"])
        target_dir = os.path.join(dst.assets, "proj")
        for sidecar in ("notes.json", "manifest.json"):
            assert (
                os.stat(os.path.join(target_dir, sidecar)).st_ino
                != os.stat(os.path.join(source_dir, sidecar)).st_ino
            )
        set_file_note(target_dir, "a.mp4", "edited")
        record_kept_shots(target_dir, "a.mp4", [{"start": 0.0, "end": 9.0}])
        assert read_json(os.path.join(source_dir, "notes.json")) == notes_before
        assert read_json(os.path.join(source_dir, "manifest.json")) == manifest_before

    @pytest.mark.parametrize("entry", ["../x.png", "/abs/x.png", "a/../b.png", "./a"])
    def test_traversal(self, src, dst, entry):
        make_asset(src, "a/b.png")
        with pytest.raises(PathTraversalError):
            copy_between(src, dst, assets=[entry])
        assert no_assets(dst)

    def test_a_sidecar_named_directly(self, src, dst):
        make_asset(src, "proj/a.png", note="n")
        with pytest.raises(InvalidInputError):
            copy_between(src, dst, assets=["proj/notes.json"])

    def test_a_missing_asset(self, src, dst):
        with pytest.raises(CopyRefused) as refused:
            copy_between(src, dst, assets=["nope.png"])
        assert refused.value.status == 404

    def test_a_symlink_on_the_way_in_the_source(self, src, dst, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "a.png").write_bytes(b"x")
        os.makedirs(src.assets, exist_ok=True)
        os.symlink(outside, os.path.join(src.assets, "linked"))
        with pytest.raises(PathTraversalError):
            copy_between(src, dst, assets=["linked/a.png"])
        assert no_assets(dst)

    def test_a_symlink_inside_a_folder(self, src, dst, tmp_path):
        make_asset(src, "proj/a.png")
        outside = tmp_path / "secret.png"
        outside.write_bytes(b"x")
        os.symlink(outside, os.path.join(src.assets, "proj", "b.png"))
        with pytest.raises(PathTraversalError):
            copy_between(src, dst, assets=["proj/"])
        assert no_assets(dst)

    def test_a_symlink_on_the_way_in_the_destination(self, src, dst, tmp_path):
        make_asset(src, "proj/a.png")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        os.makedirs(dst.assets, exist_ok=True)
        os.symlink(elsewhere, os.path.join(dst.assets, "proj"))
        with pytest.raises(PathTraversalError):
            copy_between(src, dst, assets=["proj/a.png"])
        assert list(elsewhere.iterdir()) == []

    def test_nothing_named(self, src, dst):
        with pytest.raises(InvalidInputError):
            copy_between(src, dst)


@pytest.fixture
def server(root, tmp_path):
    def make(with_root=True):
        manager = JobManager(
            root.outputs,
            worker_manager=ScriptedWorkerManager(success_script),
            history_path=str(tmp_path / "jobs.sqlite"),
            workflow_dir=root.workflows,
        )
        kwargs = {"workspace": root.root} if with_root else {}
        app = create_app(
            workflow_dir=root.workflows,
            output_dir=root.outputs,
            job_manager=manager,
            prompt_dir=root.prompts,
            asset_dir=root.assets,
            **kwargs,
        )
        return TestClient(app, base_url="http://localhost")

    return make


class TestRoute:
    def test_happy_path(self, server, root, src, dst):
        make_run(src, "wf", R1)
        with server() as client:
            response = client.post(
                "/api/workspaces/dst/copy",
                json={"source": "src", "runs": [f"wf/{R1}"]},
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["copied"]["runs"] == [f"wf/{R1}"]
        assert os.path.isfile(os.path.join(dst.outputs, "wf", R1, "img.png"))

    def test_assets(self, server, src, dst):
        make_asset(src, "proj/a.png", note="n")
        with server() as client:
            response = client.post(
                "/api/workspaces/dst/copy",
                json={"source": "src", "assets": ["proj/"]},
            )
        assert response.status_code == 200, response.text
        assert response.json()["copied"] == {"runs": [], "assets": ["proj/a.png"]}
        assert os.path.isfile(os.path.join(dst.assets, "proj", "a.png"))

    def test_runs_and_assets(self, server, src, dst):
        make_run(src, "wf", R1)
        make_asset(src, "a.png")
        with server() as client:
            response = client.post(
                "/api/workspaces/dst/copy",
                json={"source": "src", "runs": [f"wf/{R1}"], "assets": ["a.png"]},
            )
        assert response.status_code == 200, response.text
        assert response.json()["copied"] == {
            "runs": [f"wf/{R1}"],
            "assets": ["a.png"],
        }

    def test_a_different_asset_is_409_and_writes_nothing(self, server, src, dst):
        make_run(src, "wf", R1)
        make_asset(src, "a.png", content=b"mine")
        make_asset(dst, "a.png", content=b"theirs")
        with server() as client:
            response = client.post(
                "/api/workspaces/dst/copy",
                json={"source": "src", "runs": [f"wf/{R1}"], "assets": ["a.png"]},
            )
        assert response.status_code == 409
        assert response.json()["detail"]["conflicts"] == ["a.png"]
        assert nothing_written(dst)

    def test_a_missing_asset_is_404(self, server, src, dst):
        with server() as client:
            response = client.post(
                "/api/workspaces/dst/copy",
                json={"source": "src", "assets": ["nope.png"]},
            )
        assert response.status_code == 404

    def test_asset_traversal_is_400(self, server, src, dst):
        with server() as client:
            response = client.post(
                "/api/workspaces/dst/copy",
                json={"source": "src", "assets": ["../x.png"]},
            )
        assert response.status_code == 400

    def test_an_upload_over_a_copied_asset_leaves_the_source_alone(
        self, server, src, dst
    ):
        source_file = make_asset(src, "uploads/cast.png", content=b"original")
        copy_between(src, dst, assets=["uploads/cast.png"])
        with server() as client:
            response = client.post(
                "/api/uploads",
                params={
                    "filename": "cast.png",
                    "asset_name": "cast.png",
                    "workspace": "dst",
                },
                content=b"replaced",
            )
        assert response.status_code == 201, response.text
        with open(os.path.join(dst.assets, "uploads", "cast.png"), "rb") as f:
            assert f.read() == b"replaced"
        with open(source_file, "rb") as f:
            assert f.read() == b"original"

    def test_empty_runs(self, server, src, dst):
        with server() as client:
            response = client.post(
                "/api/workspaces/dst/copy", json={"source": "src", "runs": []}
            )
        assert response.status_code == 400

    def test_same_workspace(self, server, src):
        with server() as client:
            response = client.post(
                "/api/workspaces/src/copy",
                json={"source": "src", "runs": [f"wf/{R1}"]},
            )
        assert response.status_code == 400

    def test_unknown_destination(self, server, src):
        make_run(src, "wf", R1)
        with server() as client:
            response = client.post(
                "/api/workspaces/nope/copy",
                json={"source": "src", "runs": [f"wf/{R1}"]},
            )
        assert response.status_code == 404

    def test_unknown_source(self, server, dst):
        with server() as client:
            response = client.post(
                "/api/workspaces/dst/copy",
                json={"source": "nope", "runs": [f"wf/{R1}"]},
            )
        assert response.status_code == 404

    def test_a_missing_run_is_404(self, server, src, dst):
        with server() as client:
            response = client.post(
                "/api/workspaces/dst/copy",
                json={"source": "src", "runs": [f"wf/{R1}"]},
            )
        assert response.status_code == 404

    def test_traversal_is_400(self, server, src, dst):
        with server() as client:
            response = client.post(
                "/api/workspaces/dst/copy",
                json={"source": "src", "runs": ["../x"]},
            )
        assert response.status_code == 400

    @pytest.mark.parametrize(
        "destination",
        ["../dst", "/tmp/dst", "dst/../x", ".", "..", "", "dst/"],
    )
    def test_a_bad_destination_name_is_refused_by_name(
        self, server, root, src, dst, destination
    ):
        # Built as the MCP client builds it: each name reaches the route
        # rather than being normalized into another URL on the way
        make_run(src, "wf", R1)
        before = sorted(os.listdir(root.root))
        with server() as client:
            response = client.post(
                api_path("api", "workspaces", destination, "copy"),
                json={"source": "src", "runs": [f"wf/{R1}"]},
            )
        assert response.status_code == 404, response.text
        assert response.json()["detail"] == f"No such workspace: {destination}"
        assert sorted(os.listdir(root.root)) == before
        assert nothing_written(dst)

    def test_an_active_job_conflicts(self, server, src, dst):
        make_run(src, "wf", R1)
        with server() as client:
            job = SimpleNamespace(
                status="running",
                run_dir=f"wf/{R1}",
                spec={"workspace": "src"},
            )
            client.app.state.job_manager.jobs["j1"] = job
            response = client.post(
                "/api/workspaces/dst/copy",
                json={"source": "src", "runs": [f"wf/{R1}"]},
            )
        assert response.status_code == 409
        assert response.json()["detail"]["conflicts"] == [f"wf/{R1}"]
        assert nothing_written(dst)

    def test_a_finished_job_does_not_conflict(self, server, src, dst):
        make_run(src, "wf", R1)
        with server() as client:
            client.app.state.job_manager.jobs["j1"] = SimpleNamespace(
                status="succeeded", run_dir=f"wf/{R1}", spec={"workspace": "src"}
            )
            response = client.post(
                "/api/workspaces/dst/copy",
                json={"source": "src", "runs": [f"wf/{R1}"]},
            )
        assert response.status_code == 200

    def test_no_workspace_root(self, server):
        with server(with_root=False) as client:
            response = client.post(
                "/api/workspaces/dst/copy",
                json={"source": "src", "runs": [f"wf/{R1}"]},
            )
        assert response.status_code == 409

    def test_the_default_is_the_source_when_omitted(self, server, root, dst):
        make_run(root, "wf", R1)
        with server() as client:
            response = client.post(
                "/api/workspaces/dst/copy", json={"runs": [f"wf/{R1}"]}
            )
        assert response.status_code == 200, response.text
        assert response.json()["source"] == "default"
        assert os.path.isfile(os.path.join(dst.outputs, "wf", R1, "img.png"))

    def test_the_default_can_be_the_destination(self, server, root, src):
        make_run(src, "wf", R1)
        with server() as client:
            response = client.post(
                "/api/workspaces/default/copy",
                json={"source": "src", "runs": [f"wf/{R1}"]},
            )
        assert response.status_code == 200, response.text
        assert os.path.isfile(os.path.join(root.outputs, "wf", R1, "img.png"))


class TestMcp:
    def recording(self, workspace=None):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json={"ok": True})

        client = DwClient(
            transport=httpx.MockTransport(handler),
            **({"workspace": workspace} if workspace else {}),
        )
        return client, seen

    def test_source_defaults_to_the_session_workspace(self):
        client, seen = self.recording("shots")
        copy_to_workspace(client, "final", ["wf/x"])
        assert seen[0].method == "POST"
        assert seen[0].url.path == "/api/workspaces/final/copy"
        body = json.loads(seen[0].content)
        assert body == {"source": "shots", "runs": ["wf/x"], "assets": []}

    def test_assets_pass_through(self):
        client, seen = self.recording("shots")
        copy_to_workspace(client, "final", assets=["proj/"])
        body = json.loads(seen[0].content)
        assert body == {"source": "shots", "runs": [], "assets": ["proj/"]}

    def test_an_explicit_source_passes_through(self):
        client, seen = self.recording("shots")
        copy_to_workspace(client, "final", ["wf/x"], source="other")
        assert json.loads(seen[0].content)["source"] == "other"


class TestTheOwners:
    """The rules a copy relies on live with their owners (#859 arch review):
    the entry shape in dw/security.py, the numbering in dw/runs.py."""

    def test_an_entry_follows_the_output_reference_segment_rule(self):
        from dw.security import validate_output_prefix, validate_output_reference

        assert validate_output_prefix("wf") == "wf"
        assert validate_output_prefix(f"ltx2/Gyre/{R1}") == f"ltx2/Gyre/{R1}"
        for name in ("wf@a", "w.f-1/x"):
            validate_output_prefix(name)
            validate_output_reference(f"{name}/latest/img.png")
        for bad in ("@wf", "wf/.hidden", "a/b/c/d/e/f/g/h", "wf/x y"):
            with pytest.raises(InvalidInputError):
                validate_output_prefix(bad)

    def test_copy_runs_in_pins_the_destinations_unrecorded_runs_first(self, tmp_path):
        from dw.runs import copy_runs_in

        source = tmp_path / "src" / "wf"
        dest = tmp_path / "dst" / "wf"
        for base, run_id, version in ((source, R2, 1), (dest, R1, None)):
            run_dir = base / run_id
            run_dir.mkdir(parents=True)
            manifest = {"run_id": run_id}
            if version is not None:
                manifest["version"] = version
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

        def place(run_id):
            os.makedirs(dest / run_id)
            (dest / run_id / "manifest.json").write_text(
                (source / run_id / "manifest.json").read_text()
            )

        renumbered = copy_runs_in(str(source), str(dest), [R2], place)

        assert renumbered == {R2: {"from": 1, "to": 2}}
        recorded = json.loads((dest / R1 / "manifest.json").read_text())
        assert recorded["version"] == 1
        assert run_versions(str(dest)) == {R1: 1, R2: 2}
        assert json.loads((source / R2 / "manifest.json").read_text())["version"] == 1
