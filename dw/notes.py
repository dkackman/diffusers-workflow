"""Notes: optional plain text a person or an agent leaves on a run, on a
file, on an asset or on a workspace, saying what it is for.

Two stores, both sidecars the media listings never show (.json is not a
media kind):

- `notes.json` in a directory - a run directory or an asset-library
  directory - holding `{"run": ..., "files": {<name relative to the
  directory>: ...}}`. A file's effective note is its own, else the run's.
  The engine rewrites manifest.json throughout a run, so notes live apart
  from it and only the server writes them.
- `workspace.json` at a workspace root, holding `{"notes": ...}`.

Callers resolve and validate the directory; this module joins only its
own fixed file names onto it. A write that leaves nothing removes the file.
"""

import json
import os
import threading

from .runs import write_json_atomic
from .workspace import WORKSPACE_FILE_NAME  # noqa: F401 - re-exported

NOTES_FILE_NAME = "notes.json"
MAX_NOTE_LENGTH = 4000

_locks = {}
_locks_guard = threading.Lock()


class NoteTooLong(ValueError):
    pass


def clean_note(note):
    """The note as stored: None for nothing, else the text, length-checked."""
    if note is None:
        return None
    if not isinstance(note, str):
        raise TypeError(f"a note is text, not {type(note).__name__}")
    if not note.strip():
        return None
    if len(note) > MAX_NOTE_LENGTH:
        raise NoteTooLong(
            f"a note is at most {MAX_NOTE_LENGTH} characters, got {len(note)}"
        )
    return note


def _lock(directory):
    key = os.path.realpath(directory)
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def _read_json(path):
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def read_notes(directory):
    data = _read_json(os.path.join(directory, NOTES_FILE_NAME))
    run = data.get("run") if isinstance(data.get("run"), str) else None
    files = data.get("files") if isinstance(data.get("files"), dict) else {}
    files = {
        k: v for k, v in files.items() if isinstance(k, str) and isinstance(v, str)
    }
    return {"run": run, "files": files}


def _store(directory, notes):
    path = os.path.join(directory, NOTES_FILE_NAME)
    if notes["run"] is None and not notes["files"]:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return
    write_json_atomic(path, {k: v for k, v in notes.items() if v})


def _update(directory, change):
    with _lock(directory):
        notes = read_notes(directory)
        change(notes)
        _store(directory, notes)


def set_run_note(directory, note):
    note = clean_note(note)
    _update(directory, lambda notes: notes.__setitem__("run", note))
    return note


def set_file_note(directory, name, note):
    note = clean_note(note)

    def change(notes):
        if note is None:
            notes["files"].pop(name, None)
        else:
            notes["files"][name] = note

    _update(directory, change)
    return note


def drop_file_note(directory, name):
    if os.path.exists(os.path.join(directory, NOTES_FILE_NAME)):
        set_file_note(directory, name, None)


def effective_note(notes, name):
    own = notes["files"].get(name)
    if own:
        return own, "file"
    if notes["run"]:
        return notes["run"], "run"
    return None, None


def read_workspace_note(root):
    note = _read_json(os.path.join(root, WORKSPACE_FILE_NAME)).get("notes")
    return note if isinstance(note, str) and note.strip() else None


def set_workspace_note(root, note):
    note = clean_note(note)
    path = os.path.join(root, WORKSPACE_FILE_NAME)
    with _lock(root):
        if note is None:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        else:
            write_json_atomic(path, {"notes": note})
    return note
