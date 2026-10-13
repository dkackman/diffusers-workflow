"""Copying runs between workspaces (#853): the engine, the route and the MCP tool."""

import json
import os
import re
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from dw.runs import RUN_ID_PATTERN, resolve_output_reference, run_versions
from dw.security import InvalidInputError, PathTraversalError, SecurityError
from dw.server.app import create_app
from dw.server.jobs import JobManager
from dw.workspace import (
    ConfiguredWorkspace,
    Workspace,
    create_workspace,
)
from dw.workspace_copy import CopyRefused, copy_runs
from dw_mcp.client import DwClient
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
        reply = copy_runs(src, dst, [f"wf/{R1}"])
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
        reply = copy_runs(src, dst, ["wf"])
        assert reply["copied"]["runs"] == [f"wf/{R1}", f"wf/{R2}"]

    def test_a_multi_segment_identity(self, src, dst):
        make_run(src, "ltx2/Gyre", R1)
        reply = copy_runs(src, dst, [f"ltx2/Gyre/{R1}"])
        assert reply["copied"]["runs"] == [f"ltx2/Gyre/{R1}"]
        assert os.path.isfile(os.path.join(dst.outputs, "ltx2", "Gyre", R1, "img.png"))

    def test_a_bare_multi_segment_identity(self, src, dst):
        make_run(src, "ltx2/Gyre", R1)
        reply = copy_runs(src, dst, ["ltx2/Gyre"])
        assert reply["copied"]["runs"] == [f"ltx2/Gyre/{R1}"]

    def test_an_output_prefix_is_accepted(self, src, dst):
        make_run(src, "wf", R1)
        reply = copy_runs(src, dst, [f"output:wf/{R1}"])
        assert reply["copied"]["runs"] == [f"wf/{R1}"]

    def test_repeating_a_copy_is_idempotent(self, src, dst):
        make_run(src, "wf", R1)
        copy_runs(src, dst, [f"wf/{R1}"])
        again = copy_runs(src, dst, [f"wf/{R1}"])
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
        reply = copy_runs(src, dst, ["wf"])
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
        reply = copy_runs(src, dst, ["wf"])
        assert reply["renumbered"] == {
            f"wf/{R1}": {"from": 1, "to": 2},
            f"wf/{R2}": {"from": 2, "to": 3},
        }

    def test_no_collision_keeps_numbers(self, src, dst):
        make_run(dst, "wf", "20251231-000000-dddddddd", 1)
        make_run(src, "wf", R2, 2)
        reply = copy_runs(src, dst, [f"wf/{R2}"])
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
        reply = copy_runs(src, dst, [f"wf/{R2}"])
        unresolved = reply["warnings"]["unresolved_output_refs"]
        assert unresolved == {f"wf/{R2}": ["output:gone/latest/img.png"]}

    def test_all_refs_resolving_gives_no_warning(self, src, dst):
        make_run(dst, "other", R1)
        make_run(src, "wf", R2, inputs=["output:other/latest/img.png"])
        reply = copy_runs(src, dst, [f"wf/{R2}"])
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
        reply = copy_runs(src, dst, [f"wf/{R1}"])
        paths = reply["warnings"]["absolute_paths"][f"wf/{R1}"]
        assert paths == sorted(["outputs/wf/earlier/img.png", "workflows/wf.json"])
        assert str(tmp_path) not in json.dumps(reply)

    def test_a_path_outside_the_source_root_is_not_reported(self, src, dst):
        make_run(src, "wf", R1, workflow_file="/somewhere/else/wf.json")
        reply = copy_runs(src, dst, [f"wf/{R1}"])
        assert "absolute_paths" not in reply["warnings"]

    def test_no_jobs_note_after_a_copy(self, src, dst):
        make_run(src, "wf", R1)
        reply = copy_runs(src, dst, [f"wf/{R1}"])
        assert "no_jobs" in reply["warnings"]


class TestRefusals:
    @pytest.mark.parametrize("entry", ["../x", "/abs", "wf/../../x", "wf/.."])
    def test_traversal(self, src, dst, entry):
        make_run(src, "wf", R1)
        with pytest.raises(SecurityError):
            copy_runs(src, dst, [entry])
        assert nothing_written(dst)

    def test_a_symlinked_run_folder(self, src, dst, tmp_path):
        make_run(src, "elsewhere", R1)
        os.makedirs(os.path.join(src.outputs, "wf"))
        os.symlink(
            os.path.join(src.outputs, "elsewhere", R1),
            os.path.join(src.outputs, "wf", R1),
        )
        with pytest.raises(PathTraversalError):
            copy_runs(src, dst, [f"wf/{R1}"])
        assert nothing_written(dst)

    def test_a_symlinked_identity_folder(self, src, dst):
        make_run(src, "real", R1)
        os.symlink(
            os.path.join(src.outputs, "real"), os.path.join(src.outputs, "linked")
        )
        with pytest.raises(PathTraversalError):
            copy_runs(src, dst, [f"linked/{R1}"])
        assert nothing_written(dst)

    def test_a_symlink_inside_a_run(self, src, dst, tmp_path):
        run_dir = make_run(src, "wf", R1)
        os.symlink(os.path.join(run_dir, "img.png"), os.path.join(run_dir, "link.png"))
        with pytest.raises(PathTraversalError):
            copy_runs(src, dst, [f"wf/{R1}"])
        assert nothing_written(dst)

    def test_a_missing_run(self, src, dst):
        make_run(src, "wf", R1)
        with pytest.raises(CopyRefused) as raised:
            copy_runs(src, dst, [f"wf/{R2}"])
        assert raised.value.status == 404
        assert nothing_written(dst)

    def test_a_missing_workflow(self, src, dst):
        with pytest.raises(CopyRefused) as raised:
            copy_runs(src, dst, ["nope"])
        assert raised.value.status == 404

    def test_same_workspace(self, src):
        make_run(src, "wf", R1)
        with pytest.raises(InvalidInputError):
            copy_runs(src, src, [f"wf/{R1}"])

    def test_empty_runs(self, src, dst):
        with pytest.raises(InvalidInputError):
            copy_runs(src, dst, [])

    def configured(self, ws):
        return ConfiguredWorkspace(ws.workflows, ws.assets, ws.outputs, ws.prompts)

    def test_configured_source(self, src, dst):
        make_run(src, "wf", R1)
        with pytest.raises(CopyRefused) as raised:
            copy_runs(self.configured(src), dst, [f"wf/{R1}"])
        assert raised.value.status == 409
        assert nothing_written(dst)

    def test_configured_destination(self, src, dst):
        make_run(src, "wf", R1)
        with pytest.raises(CopyRefused) as raised:
            copy_runs(src, self.configured(dst), [f"wf/{R1}"])
        assert raised.value.status == 409
        assert nothing_written(dst)

    def test_an_active_run_refuses_all_or_nothing(self, src, dst):
        make_run(src, "wf", R1, 1)
        make_run(src, "wf", R2, 2)
        with pytest.raises(CopyRefused) as raised:
            copy_runs(src, dst, ["wf"], active_runs=[f"wf/{R2}"])
        assert raised.value.status == 409
        assert raised.value.conflicts == [f"wf/{R2}"]
        assert nothing_written(dst)
        assert not os.path.exists(os.path.join(dst.outputs, "wf", R1))


class TestResolution:
    def test_output_references_resolve_in_the_destination(self, src, dst):
        make_run(src, "wf", R1)
        copy_runs(src, dst, ["wf"])
        latest = resolve_output_reference("output:wf/latest/img.png", dst.outputs)
        pinned = resolve_output_reference(f"output:wf/{R1}/img.png", dst.outputs)
        expected = os.path.join(os.path.realpath(dst.outputs), "wf", R1, "img.png")
        assert latest == expected
        assert pinned == expected
        assert os.path.isfile(pinned)


class TestIndependence:
    def test_notes_are_not_shared(self, src, dst):
        source_run = make_run(src, "wf", R1, notes={"note": "original"})
        copy_runs(src, dst, [f"wf/{R1}"])
        copied = os.path.join(dst.outputs, "wf", R1, "notes.json")
        original = os.path.join(source_run, "notes.json")
        assert os.stat(copied).st_ino != os.stat(original).st_ino
        with open(copied, "w") as f:
            json.dump({"note": "edited"}, f)
        assert read_json(original) == {"note": "original"}

    def test_a_renumbered_manifest_leaves_the_source_alone(self, src, dst):
        make_run(dst, "wf", "20251231-000000-dddddddd", 1)
        source_run = make_run(src, "wf", R1, 1)
        copy_runs(src, dst, [f"wf/{R1}"])
        copied = os.path.join(dst.outputs, "wf", R1, "manifest.json")
        original = os.path.join(source_run, "manifest.json")
        assert read_json(copied)["version"] == 2
        assert read_json(original)["version"] == 1
        assert os.stat(copied).st_ino != os.stat(original).st_ino


class TestMediaLinking:
    def test_media_is_hard_linked(self, src, dst):
        source_run = make_run(src, "wf", R1, content=b"x" * 100)
        reply = copy_runs(src, dst, [f"wf/{R1}"])
        a = os.stat(os.path.join(source_run, "img.png"))
        b = os.stat(os.path.join(dst.outputs, "wf", R1, "img.png"))
        assert a.st_ino == b.st_ino
        assert reply["bytes_linked"] >= 100

    def test_a_failed_link_falls_back_to_a_copy(self, src, dst, monkeypatch):
        source_run = make_run(src, "wf", R1, content=b"x" * 100)

        def refuse(*args, **kwargs):
            raise OSError("cross-device")

        monkeypatch.setattr("dw.workspace_copy.os.link", refuse)
        reply = copy_runs(src, dst, [f"wf/{R1}"])
        target = os.path.join(dst.outputs, "wf", R1, "img.png")
        source_file = os.path.join(source_run, "img.png")
        assert os.stat(target).st_ino != os.stat(source_file).st_ino
        with open(target, "rb") as a, open(source_file, "rb") as b:
            assert a.read() == b.read()
        assert reply["bytes_linked"] == 0
        assert reply["bytes_copied"] >= 100


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

    def test_assets_are_not_supported(self, server, src, dst):
        make_run(src, "wf", R1)
        with server() as client:
            response = client.post(
                "/api/workspaces/dst/copy",
                json={"source": "src", "runs": [f"wf/{R1}"], "assets": ["a.png"]},
            )
        assert response.status_code == 400
        assert "not supported" in response.text
        assert nothing_written(dst)

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
        assert body == {"source": "shots", "runs": ["wf/x"]}

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
