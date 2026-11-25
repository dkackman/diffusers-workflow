"""ExportSummary: what the export route hands back inline."""

import json

from dw.server import exports
from dw.server.exports import ExportSummary


def test_inline_json_reads_the_named_files(tmp_path, monkeypatch):
    """The three files are read under the names the export writes them by:
    a rename in one place moves the inline read with it."""
    monkeypatch.setattr(exports, "WORKFLOW_FILE_NAME", "w.json")
    monkeypatch.setattr(exports, "MANIFEST_FILE_NAME", "m.json")
    monkeypatch.setattr(exports, "JOB_FILE_NAME", "j.json")
    for name, body in (("w.json", {"id": "W"}), ("m.json", [1]), ("j.json", {"j": 2})):
        (tmp_path / name).write_text(json.dumps(body))
    found = ExportSummary(job_id="x", directory=str(tmp_path)).inline_json()
    assert found == {"workflow": {"id": "W"}, "manifest": [1], "job": {"j": 2}}


def test_inline_json_reads_none_for_an_unreadable_file(tmp_path):
    (tmp_path / exports.JOB_FILE_NAME).write_text("{not json")
    found = ExportSummary(job_id="x", directory=str(tmp_path)).inline_json()
    assert found == {"workflow": None, "manifest": None, "job": None}
