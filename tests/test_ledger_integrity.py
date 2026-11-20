"""The requests ledger's integrity (0.21.0 review): the atomic write, the
file's shape, the cross-process lock, the caps that no longer wedge a
workspace, and deleting a request."""

import json
import os

import pytest

from dw import runs
from dw.runs import write_json_atomic


class TestAtomicWrite:
    def test_the_file_is_synced_before_it_replaces_the_old_one(
        self, tmp_path, monkeypatch
    ):
        order = []
        real_fsync, real_replace = os.fsync, os.replace
        monkeypatch.setattr(
            runs.os, "fsync", lambda fd: (order.append("fsync"), real_fsync(fd))[1]
        )
        monkeypatch.setattr(
            runs.os,
            "replace",
            lambda a, b: (order.append("replace"), real_replace(a, b))[1],
        )
        write_json_atomic(str(tmp_path / "x.json"), {"a": 1})
        assert order == ["fsync", "replace"]
        assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1}

    def test_a_value_json_cannot_hold_leaves_no_temp_file(self, tmp_path):
        (tmp_path / "x.json").write_text('{"kept": true}')
        with pytest.raises(ValueError):
            # a cycle: json raises ValueError, which is not an OSError
            cycle = {}
            cycle["self"] = cycle
            write_json_atomic(str(tmp_path / "x.json"), cycle)
        assert os.listdir(tmp_path) == ["x.json"]
        assert json.loads((tmp_path / "x.json").read_text()) == {"kept": True}

    def test_a_failed_replace_leaves_no_temp_file(self, tmp_path, monkeypatch):
        def refuse(a, b):
            raise PermissionError("no")

        monkeypatch.setattr(runs.os, "replace", refuse)
        with pytest.raises(OSError):
            write_json_atomic(str(tmp_path / "x.json"), {"a": 1})
        assert os.listdir(tmp_path) == []
