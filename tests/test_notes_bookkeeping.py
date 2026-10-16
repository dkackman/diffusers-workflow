"""Run-directory bookkeeping stays correct when notes.json is present."""

import io
import os
import zipfile

from dw.notes import NOTES_FILE_NAME, read_notes, set_file_note, set_run_note
from dw.runs import new_run_id

from .test_server_exports import RUN_DIR, exporting_script, finished
from .test_server_exports import server as export_server  # noqa: F401
from .test_server_exports import workspace_root  # noqa: F401

IDENTITY = "wf"


def _run(write_run, output_root, files, seed=1):
    run_id = new_run_id({"id": IDENTITY, "seed": seed})
    run_dir = write_run(
        output_root,
        IDENTITY,
        run_id,
        [{"step": "gen", "files": list(files)}],
        files=files,
    )
    return run_id, run_dir


def test_run_with_only_notes_is_pruned_and_orphan(
    gallery_client, output_root, write_run
):
    run_id, run_dir = _run(write_run, output_root, ["a.png"])
    set_run_note(run_dir, "testing shift")
    assert (
        gallery_client.delete(f"/api/gallery/{IDENTITY}/{run_id}/a.png").status_code
        == 200
    )
    assert not os.path.exists(run_dir)

    orphan_id, orphan_dir = _run(write_run, output_root, [], seed=2)
    set_run_note(orphan_dir, "left behind")
    assert os.path.isfile(os.path.join(orphan_dir, NOTES_FILE_NAME))
    listed = gallery_client.get("/api/gallery?only_orphans=true").json()
    assert {r["name"] for r in listed["runs"]} == {f"{IDENTITY}/{orphan_id}"}


def test_single_output_delete_drops_its_note(gallery_client, output_root, write_run):
    run_id, run_dir = _run(write_run, output_root, ["a.png", "final/b.png"])
    set_file_note(run_dir, "a.png", "first")
    set_file_note(run_dir, "final/b.png", "second")
    response = gallery_client.delete(f"/api/gallery/{IDENTITY}/{run_id}/final/b.png")
    assert response.status_code == 200
    assert os.path.isfile(os.path.join(run_dir, "a.png"))
    assert read_notes(run_dir)["files"] == {"a.png": "first"}


def test_export_carries_notes(export_server):  # noqa: F811
    with export_server(exporting_script) as client:
        # the run's directory is made by the script, so note it once finished
        job_id = finished(client)
        output_root = client.app.state.job_manager.output_dir
        set_run_note(os.path.join(output_root, RUN_DIR), "keep this one")
        body = client.post(f"/api/jobs/{job_id}/export").json()
        assert os.path.isfile(os.path.join(body["directory"], NOTES_FILE_NAME))
        zipped = client.get(body["zip_url"])
    names = zipfile.ZipFile(io.BytesIO(zipped.content)).namelist()
    assert f"{job_id}/{NOTES_FILE_NAME}" in names


def test_a_failing_note_drop_does_not_fail_the_delete(
    gallery_client, output_root, write_run, monkeypatch
):
    run_id, run_dir = _run(write_run, output_root, ["a.png"])
    set_file_note(run_dir, "a.png", "first")

    def broken(*_args):
        raise OSError("disk full")

    monkeypatch.setattr("dw.server.routes.gallery.drop_file_note", broken)
    response = gallery_client.delete(f"/api/gallery/{IDENTITY}/{run_id}/a.png")
    assert response.status_code == 200
    assert response.json()["run_swept"] == run_id
    assert not os.path.exists(run_dir)
