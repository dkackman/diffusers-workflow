import json
import os
import threading

import pytest

from dw import notes as N


def test_clean_note_rules():
    assert N.clean_note(None) is None
    assert N.clean_note("   \n") is None
    assert N.clean_note("shift 5") == "shift 5"
    assert N.clean_note("x" * N.MAX_NOTE_LENGTH) == "x" * N.MAX_NOTE_LENGTH
    with pytest.raises(N.NoteTooLong):
        N.clean_note("x" * (N.MAX_NOTE_LENGTH + 1))
    with pytest.raises(TypeError):
        N.clean_note(5)


def test_run_and_file_notes_and_effective(tmp_path):
    d = str(tmp_path)
    assert N.read_notes(d) == {"run": None, "files": {}}
    N.set_run_note(d, "testing shift 3 vs 5")
    N.set_file_note(d, "final/b.png", "the keeper")
    notes = N.read_notes(d)
    assert N.effective_note(notes, "final/b.png") == ("the keeper", "file")
    assert N.effective_note(notes, "a.png") == ("testing shift 3 vs 5", "run")
    assert N.effective_note({"run": None, "files": {}}, "a.png") == (None, None)


def test_clearing_everything_removes_the_file(tmp_path):
    d = str(tmp_path)
    N.set_run_note(d, "x")
    N.set_file_note(d, "a.png", "y")
    assert os.path.exists(os.path.join(d, N.NOTES_FILE_NAME))
    N.set_run_note(d, "")
    N.drop_file_note(d, "a.png")
    assert not os.path.exists(os.path.join(d, N.NOTES_FILE_NAME))


def test_corrupt_file_reads_empty(tmp_path):
    (tmp_path / N.NOTES_FILE_NAME).write_text("{not json")
    assert N.read_notes(str(tmp_path)) == {"run": None, "files": {}}


def test_concurrent_file_notes_all_land(tmp_path):
    d = str(tmp_path)
    threads = [
        threading.Thread(target=N.set_file_note, args=(d, f"{i}.png", str(i)))
        for i in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(N.read_notes(d)["files"]) == 20


def test_workspace_note(tmp_path):
    root = str(tmp_path)
    assert N.read_workspace_note(root) is None
    N.set_workspace_note(root, "the anthem project")
    assert N.read_workspace_note(root) == "the anthem project"
    with open(tmp_path / N.WORKSPACE_FILE_NAME) as handle:
        assert json.load(handle) == {"notes": "the anthem project"}
    N.set_workspace_note(root, None)
    assert not os.path.exists(tmp_path / N.WORKSPACE_FILE_NAME)


def test_workspace_with_note_is_deletable(tmp_path):
    from dw.workspace import Workspace, create_workspace, delete_workspace

    root = Workspace(tmp_path / "studio", "flag").ensure()
    created = create_workspace(root, "shots")
    N.set_workspace_note(created.root, "x")
    assert os.path.exists(os.path.join(created.root, N.WORKSPACE_FILE_NAME))
    delete_workspace(root, "shots")
    assert not os.path.exists(created.root)
