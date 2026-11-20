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


class TestReservedNames:
    """A workspace is a folder at the root, and the default workspace keeps
    its ledger and its note as files there: a workspace of either name would
    be a directory where the file goes."""

    @pytest.mark.parametrize("name", ["requests.json", "workspace.json"])
    def test_a_workspace_cannot_take_a_root_file_name(self, tmp_path, name):
        from dw.security import InvalidInputError
        from dw.workspace import Workspace, create_workspace

        root = Workspace(tmp_path / "studio", "flag").ensure()
        with pytest.raises(InvalidInputError):
            create_workspace(root, name)
        assert not os.path.exists(os.path.join(root.root, name))

    def test_the_reserved_names_include_both_files(self):
        from dw.workspace import (
            REQUESTS_FILE_NAME,
            RESERVED_WORKSPACE_NAMES,
            WORKSPACE_FILE_NAME,
        )

        assert REQUESTS_FILE_NAME in RESERVED_WORKSPACE_NAMES
        assert WORKSPACE_FILE_NAME in RESERVED_WORKSPACE_NAMES
