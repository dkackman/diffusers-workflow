"""dw's disk-cache registry (dw/disk_caches.py) and its routes (#828)."""

import builtins
import json
import logging
import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

from dw import _inductor_cache_dir_default, disk_caches
from dw.disk_caches import (
    MARKER,
    NoEntries,
    NotClearable,
    UnknownCache,
    UnknownEntry,
)
from dw.server.app import create_app
from dw.server.jobs import JobManager, JobsActive

from tests.test_server import (
    ScriptedWorkerManager,
    admitted_for,
    success_script,
    valid_workflow,
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """HOME and every cache variable pointed into tmp_path, triton gone from
    sys.modules so the never-imported assertions mean something."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_HOME", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setenv("DW_QUANTIZED_CACHE_DIR", str(tmp_path / "quantized"))
    monkeypatch.delenv("DW_QUANTIZED_CACHE", raising=False)
    monkeypatch.delitem(sys.modules, "triton", raising=False)
    return home


@pytest.fixture(autouse=True)
def fresh_scans(monkeypatch):
    monkeypatch.setattr(disk_caches, "_scan_memo", None)


def populate(root, sizes=(10, 20, 30)):
    """Files in the root and a nested directory; returns (bytes, files)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "sub" / "deep").mkdir(parents=True, exist_ok=True)
    paths = [root / "a.bin", root / "sub" / "b.bin", root / "sub" / "deep" / "c.bin"]
    for path, size in zip(paths, sizes):
        path.write_bytes(b"x" * size)
    return sum(sizes), len(paths)


# ------------------------------------------------------------------ scan


def test_scan_counts_bytes_files_and_newest_mtime(env, tmp_path):
    root = tmp_path / "inductor"
    total, count = populate(root)
    (root / MARKER).write_text("marked")
    os.utime(root / "a.bin", (1000, 1000))
    os.utime(root / "sub" / "b.bin", (3000, 3000))
    os.utime(root / "sub" / "deep" / "c.bin", (2000, 2000))
    os.utime(root / MARKER, (9000, 9000))

    info = disk_caches.scan(disk_caches.get("inductor"))

    assert info["bytes"] == total
    assert info["files"] == count
    assert info["last_used"] == 3000
    assert info["clearable"] is True
    assert "not_clearable_reason" not in info


def test_scan_missing_root(env, tmp_path):
    info = disk_caches.scan(disk_caches.get("inductor"))

    assert not (tmp_path / "inductor").exists()
    assert info["bytes"] == 0
    assert info["files"] == 0
    assert info["last_used"] is None
    assert info["clearable"] is False
    assert isinstance(info["disk_free"], int)
    assert isinstance(info["disk_total"], int)


def test_scan_does_not_follow_symlinks(env, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "big.bin").write_bytes(b"x" * 100_000)
    target_file = tmp_path / "target.bin"
    target_file.write_bytes(b"x" * 50_000)
    root = tmp_path / "inductor"
    total, count = populate(root)
    os.symlink(outside, root / "dirlink")
    os.symlink(target_file, root / "filelink")

    info = disk_caches.scan(disk_caches.get("inductor"))

    assert info["bytes"] == total
    assert info["files"] == count


# ---------------------------------------------------------------- triton


def test_triton_root_cache_dir_wins(env, tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "tc"))
    monkeypatch.setenv("TRITON_HOME", str(tmp_path / "th"))
    assert disk_caches.get("triton").root() == str(tmp_path / "tc")


def test_triton_root_from_triton_home(env, tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_HOME", str(tmp_path / "th"))
    assert disk_caches.get("triton").root() == str(
        tmp_path / "th" / ".triton" / "cache"
    )


def test_triton_root_defaults_under_home(env):
    assert disk_caches.get("triton").root() == str(env / ".triton" / "cache")


def test_triton_is_never_imported(env, monkeypatch):
    real_import = builtins.__import__

    def refuse_triton(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            raise AssertionError("triton was imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_triton)
    triton = disk_caches.get("triton")
    triton.root()
    disk_caches.scan(triton)
    disk_caches.mark_roots()
    monkeypatch.undo()

    assert "triton" not in sys.modules


@pytest.mark.parametrize("installed", [True, False])
def test_triton_enabled_follows_find_spec(env, monkeypatch, installed):
    monkeypatch.setattr(
        "dw.disk_caches.importlib.util.find_spec",
        lambda name: object() if installed and name == "triton" else None,
    )
    assert disk_caches.get("triton").enabled() is installed


def test_missing_triton_root_is_not_created(env):
    triton = disk_caches.get("triton")

    assert disk_caches.mark_root(triton) is False
    disk_caches.mark_roots()

    assert not (env / ".triton").exists()
    info = disk_caches.scan(triton)
    assert info["bytes"] == 0
    assert info["files"] == 0
    assert info["clearable"] is False


# -------------------------------------------------------------- inductor


def test_inductor_root_from_env(env, tmp_path):
    assert disk_caches.get("inductor").root() == str(tmp_path / "inductor")


def test_inductor_default_is_xdg_dw_torchinductor(env, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR")
    cache = disk_caches.get("inductor")

    expected = str(tmp_path / "xdg" / "dw" / "torchinductor")
    assert _inductor_cache_dir_default(os.environ) == expected
    assert cache.default_root() == expected
    assert cache.root() == expected


def test_missing_inductor_root_is_created_and_marked(env, tmp_path):
    assert disk_caches.mark_root(disk_caches.get("inductor")) is True

    assert (tmp_path / "inductor" / MARKER).is_file()


# ---------------------------------------------------------- marker rule


def test_empty_root_gets_marker(env, tmp_path):
    (tmp_path / "inductor").mkdir()
    assert disk_caches.mark_root(disk_caches.get("inductor")) is True
    assert (tmp_path / "inductor" / MARKER).is_file()


def test_populated_inductor_default_is_grandfathered(env, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    default = _inductor_cache_dir_default(os.environ)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", default)
    populate(tmp_path / "xdg" / "dw" / "torchinductor")

    assert disk_caches.mark_root(disk_caches.get("inductor")) is True
    assert os.path.isfile(os.path.join(default, MARKER))


def test_populated_triton_default_is_grandfathered(env):
    root = env / ".triton" / "cache"
    populate(root)

    assert disk_caches.mark_root(disk_caches.get("triton")) is True
    assert (root / MARKER).is_file()


def test_populated_non_default_root_is_not_marked_or_cleared(env, tmp_path):
    root = tmp_path / "inductor"
    populate(root)

    assert disk_caches.mark_root(disk_caches.get("inductor")) is False
    assert not (root / MARKER).exists()
    info = disk_caches.scan(disk_caches.get("inductor"))
    assert info["clearable"] is False
    assert MARKER in info["not_clearable_reason"]
    with pytest.raises(NotClearable):
        disk_caches.clear("inductor")
    assert (root / "a.bin").exists()
    assert (root / "sub" / "deep" / "c.bin").exists()


def test_triton_home_populated_is_not_grandfathered(env, tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_HOME", str(tmp_path / "th"))
    root = tmp_path / "th" / ".triton" / "cache"
    populate(root)

    assert disk_caches.mark_root(disk_caches.get("triton")) is False
    assert not (root / MARKER).exists()
    assert disk_caches.scan(disk_caches.get("triton"))["clearable"] is False
    with pytest.raises(NotClearable):
        disk_caches.clear("triton")
    assert (root / "a.bin").exists()


def test_mark_roots_swallows_oserror(env, monkeypatch, caplog):
    def refuse(root):
        raise PermissionError("read-only")

    monkeypatch.setattr(disk_caches, "_write_marker", refuse)
    with caplog.at_level("WARNING", logger="dw"):
        disk_caches.mark_roots()

    assert "Could not mark the inductor cache" in caplog.text


# ----------------------------------------------------------------- clear


def marked_root(tmp_path):
    root = tmp_path / "inductor"
    result = populate(root)
    (root / MARKER).write_text("marked")
    return root, result


def test_clear_empties_root_and_keeps_marker(env, tmp_path):
    root, (total, count) = marked_root(tmp_path)

    result = disk_caches.clear("inductor")

    assert result == {"freed": total, "files": count}
    assert root.is_dir()
    assert [p.name for p in root.iterdir()] == [MARKER]


def test_clear_keeps_a_file_written_during_the_clear(env, tmp_path, monkeypatch):
    root, _ = marked_root(tmp_path)
    real_rmtree = disk_caches.shutil.rmtree

    def rmtree_after_a_write(path, *args, **kwargs):
        (root / "new.bin").write_bytes(b"new")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("dw.disk_caches.shutil.rmtree", rmtree_after_a_write)
    disk_caches.clear("inductor")

    assert (root / "new.bin").exists()
    assert not (root / "a.bin").exists()
    assert not (root / "sub").exists()


def test_clear_removes_a_stale_clearing_dir(env, tmp_path):
    root, _ = marked_root(tmp_path)
    stale = root / ".clearing-123-abcd"
    stale.mkdir()
    (stale / "old.bin").write_bytes(b"x" * 5)

    disk_caches.clear("inductor")

    assert sorted(p.name for p in root.iterdir()) == [MARKER]


@pytest.mark.parametrize("name", ["nope", "../x"])
def test_unknown_cache_names_the_known_ones(name):
    with pytest.raises(UnknownCache) as raised:
        disk_caches.clear(name)
    assert "inductor" in str(raised.value)
    assert "triton" in str(raised.value)


# ----------------------------------------------------------------- routes


@pytest.fixture(params=[None])
def api(env, tmp_path, request, monkeypatch):
    # No UI unless a test asks for one: with ui/dist built, its StaticFiles
    # mount at / answers a path no route matches, so a route test would
    # depend on whether this checkout has built the UI
    ui_dir = request.param
    if ui_dir is None:
        monkeypatch.setattr("dw.server.app.default_ui_dir", lambda: None)
    else:
        ui_dir = tmp_path / ui_dir
        ui_dir.mkdir()
        (ui_dir / "index.html").write_text("<!doctype html>")
    workflow_dir = tmp_path / "workflows"
    workflow_dir.mkdir()
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
    )
    app = create_app(
        workflow_dir=str(workflow_dir),
        output_dir=str(tmp_path / "outputs"),
        job_manager=manager,
        prompt_dir=str(prompt_dir),
        ui_dir=str(ui_dir) if ui_dir else None,
    )
    with TestClient(app, base_url="http://localhost") as client:
        client.manager = manager
        yield client


def tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def parent_tree(tmp_path):
    """What is under tmp_path, minus the server's own workspace files: the
    job history's SQLite -shm/-wal sidecars come and go with its connection,
    and outputs/ is created on the server's schedule, so a traversal test
    checks the parent for losses among the rest, not for equality."""
    return {
        p
        for p in tree(tmp_path)
        if not p.startswith("jobs.sqlite") and not p.startswith("outputs")
    }


class TestCacheRoutes:
    def test_list_reports_both_caches(self, api, tmp_path):
        root, (total, count) = marked_root(tmp_path)

        response = api.get("/api/caches")

        assert response.status_code == 200
        caches = {c["name"]: c for c in response.json()["caches"]}
        assert set(caches) == {"inductor", "triton", "quantized"}
        keys = {
            "name",
            "root",
            "enabled",
            "bytes",
            "files",
            "last_used",
            "clearable",
            "cost_of_clearing",
            "disk_free",
            "disk_total",
        }
        for cache in caches.values():
            assert keys <= set(cache)
            assert cache["cost_of_clearing"]
        inductor, triton = caches["inductor"], caches["triton"]
        assert inductor["root"] == str(root)
        assert inductor["bytes"] == total
        assert inductor["files"] == count
        assert inductor["clearable"] is True
        assert "not_clearable_reason" not in inductor
        assert "shared" in triton["cost_of_clearing"]
        assert triton["clearable"] is False
        assert triton["not_clearable_reason"]

    def test_clear_empties_a_marked_root(self, api, tmp_path):
        root, (total, count) = marked_root(tmp_path)

        response = api.delete("/api/caches/inductor")

        assert response.status_code == 200
        assert response.json() == {
            "name": "inductor",
            "cleared": True,
            "freed": total,
            "files": count,
        }
        assert tree(root) == [MARKER]

    def test_clear_refused_while_busy(self, api, tmp_path):
        root, _ = marked_root(tmp_path)
        before = tree(root)
        api.manager.is_busy = lambda: True

        response = api.delete("/api/caches/inductor")

        assert response.status_code == 409
        detail = response.json()["detail"]
        assert "running" in detail or "queued" in detail
        assert tree(root) == before

    def test_clear_refused_for_unmarked_root(self, api, tmp_path):
        root = tmp_path / "inductor"
        populate(root)
        before = tree(root)

        response = api.delete("/api/caches/inductor")

        assert response.status_code == 400
        assert MARKER in response.json()["detail"]
        assert tree(root) == before

    def test_unknown_name_is_404(self, api, tmp_path):
        marked_root(tmp_path)

        response = api.delete("/api/caches/nope")

        assert response.status_code == 404
        assert "inductor" in response.json()["detail"]
        assert "triton" in response.json()["detail"]

    # The decoded name has a "/", so no cache route matches: without a UI
    # nothing does (404); with one, the UI's StaticFiles mount at / takes the
    # path and refuses a DELETE (405). Either way the handler never runs.
    @pytest.mark.parametrize(
        "api, status", [(None, 404), ("ui", 405)], indirect=["api"]
    )
    def test_traversal_name_touches_nothing(self, api, tmp_path, status):
        root, _ = marked_root(tmp_path)
        victim = tmp_path / "x" / "victim.bin"
        victim.parent.mkdir()
        victim.write_bytes(b"x" * 8)
        before_root = tree(root)
        before_parent = parent_tree(tmp_path)

        response = api.delete("/api/caches/..%2Fx")

        assert response.status_code == status
        assert tree(root) == before_root
        assert victim.read_bytes() == b"x" * 8
        assert before_parent <= parent_tree(tmp_path)


# ------------------------------------------------------- quantized entries

KEY_A = "a" * 64
KEY_B = "b" * 64


def quantized_entry(root, key, sizes=(100, 200), entry=True, **fields):
    """An entry directory as quantized_cache.save_entry leaves it."""
    path = root / key
    path.mkdir(parents=True)
    for i, size in enumerate(sizes):
        (path / f"model-{i}.safetensors").write_bytes(b"q" * size)
    if entry is True:
        record = {
            "model_name": "MiniMaxAI/MiniMax-H3",
            "subfolder": "vae",
            "commit": "c" * 40,
            "key": key,
            "bytes": sum(sizes),
            "created": 1700000000.0,
            "last_hit": None,
            **fields,
        }
        (path / "entry.json").write_text(json.dumps(record))
    elif entry is not None:
        (path / "entry.json").write_text(entry)
    return path


def quantized_root(tmp_path):
    root = tmp_path / "quantized"
    root.mkdir()
    (root / MARKER).write_text("marked")
    return root


def test_quantized_root_and_enabled_follow_the_env(env, tmp_path, monkeypatch):
    cache = disk_caches.get("quantized")
    assert cache.root() == str(tmp_path / "quantized")
    assert cache.enabled() is False
    monkeypatch.setenv("DW_QUANTIZED_CACHE", "1")
    assert cache.enabled() is True


def test_entries_parsed_from_entry_json(env, tmp_path):
    root = quantized_root(tmp_path)
    quantized_entry(root, KEY_A, sizes=(100, 200), last_hit=1700000500.0)
    quantized_entry(
        root, KEY_B, sizes=(50,), subfolder="transformer", created=1700000900.0
    )
    (root / "not-a-key").mkdir()
    (root / f"{KEY_A}.tmp-1-abcd").mkdir()

    info = disk_caches.scan(disk_caches.get("quantized"))

    by_id = {e["id"]: e for e in info["entries"]}
    assert set(by_id) == {KEY_A, KEY_B}
    # newest first
    assert [e["id"] for e in info["entries"]] == [KEY_B, KEY_A]
    a = by_id[KEY_A]
    assert a["model"] == "MiniMaxAI/MiniMax-H3"
    assert a["subfolder"] == "vae"
    assert a["component"] == "vae"
    assert a["created"] == 1700000000.0
    assert a["last_hit"] == 1700000500.0
    assert a["unreadable"] is False
    # the files on disk, entry.json included
    entry_json = (root / KEY_A / "entry.json").stat().st_size
    assert a["bytes"] == 300 + entry_json
    assert a["files"] == 3
    assert by_id[KEY_B]["component"] == "transformer"


@pytest.mark.parametrize("entry", [None, "{not json", "[1, 2]"])
def test_unreadable_entry_is_listed_and_deletable(env, tmp_path, entry):
    root = quantized_root(tmp_path)
    quantized_entry(root, KEY_A, sizes=(40,), entry=entry)

    [listed] = disk_caches.scan(disk_caches.get("quantized"))["entries"]

    assert listed["id"] == KEY_A
    assert listed["unreadable"] is True
    assert listed["model"] is None
    assert listed["component"] is None

    result = disk_caches.delete_entry("quantized", KEY_A)

    assert result["freed"] >= 40
    assert not (root / KEY_A).exists()


def test_off_cache_still_lists_and_clears(env, tmp_path):
    root = quantized_root(tmp_path)
    quantized_entry(root, KEY_A)

    info = disk_caches.scan(disk_caches.get("quantized"))
    assert info["enabled"] is False
    assert [e["id"] for e in info["entries"]] == [KEY_A]
    assert info["clearable"] is True

    disk_caches.clear("quantized")

    assert [p.name for p in root.iterdir()] == [MARKER]


def test_missing_quantized_root_lists_no_entries_and_is_not_created(env, tmp_path):
    disk_caches.mark_roots()

    info = disk_caches.scan(disk_caches.get("quantized"))

    assert info["entries"] == []
    assert info["bytes"] == 0
    assert not (tmp_path / "quantized").exists()


def test_populated_quantized_default_is_grandfathered(env, tmp_path, monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HOME", str(tmp_path / "hf"))
    monkeypatch.delenv("DW_QUANTIZED_CACHE_DIR")
    root = tmp_path / "hf" / "dw-quantized"
    quantized_entry(root, KEY_A)

    assert disk_caches.mark_root(disk_caches.get("quantized")) is True
    assert (root / MARKER).is_file()

    assert disk_caches.get("quantized").default_root() == str(root)


def test_populated_non_default_quantized_root_is_not_marked(env, tmp_path):
    root = tmp_path / "quantized"
    quantized_entry(root, KEY_A)

    assert disk_caches.mark_root(disk_caches.get("quantized")) is False
    with pytest.raises(NotClearable):
        disk_caches.delete_entry("quantized", KEY_A)
    assert (root / KEY_A).is_dir()


def test_delete_entry_removes_only_that_entry(env, tmp_path):
    root = quantized_root(tmp_path)
    quantized_entry(root, KEY_A, sizes=(100,))
    quantized_entry(root, KEY_B, sizes=(7,))

    result = disk_caches.delete_entry("quantized", KEY_A)

    assert result["freed"] >= 100
    assert sorted(p.name for p in root.iterdir()) == sorted([MARKER, KEY_B])


@pytest.mark.parametrize(
    "entry",
    ["abc", "A" * 64, "a" * 63, "a" * 65, "../" + "a" * 61, "g" * 64, "", None],
)
def test_malformed_id_refused_before_any_path_is_built(
    env, tmp_path, monkeypatch, entry
):
    quantized_root(tmp_path)
    built = []
    monkeypatch.setattr(
        "dw.disk_caches.validate_path",
        lambda *a, **k: built.append(a) or a[0],
    )
    monkeypatch.setattr(
        "dw.quantized_cache.validate_path",
        lambda *a, **k: built.append(a) or a[0],
    )
    monkeypatch.setattr(
        "dw.disk_caches.os.path.join", lambda *a: built.append(a) or "/nowhere"
    )

    with pytest.raises(NoEntries):
        disk_caches.delete_entry("quantized", entry)
    assert built == []


def test_entry_ops_on_a_cache_without_entries(env, tmp_path):
    marked_root(tmp_path)
    with pytest.raises(NoEntries):
        disk_caches.delete_entry("inductor", KEY_A)


def test_unknown_entry(env, tmp_path):
    quantized_root(tmp_path)
    with pytest.raises(UnknownEntry):
        disk_caches.delete_entry("quantized", KEY_A)


@pytest.mark.parametrize("free, low", [(10 * 2**30, True), (60 * 2**30, False)])
def test_low_disk_on_both_arms_of_the_floor(env, tmp_path, monkeypatch, free, low):
    quantized_root(tmp_path)
    total = 1000 * 2**30  # floor = min(50 GiB, 100 GiB) = 50 GiB
    usage = disk_caches.shutil._ntuple_diskusage(total, total - free, free)
    monkeypatch.setattr("dw.disk_caches.shutil.disk_usage", lambda path: usage)

    info = disk_caches.scan(disk_caches.get("quantized"))

    assert info["low_disk"] is low
    assert "low_disk" not in disk_caches.scan(disk_caches.get("inductor"))


def test_low_disk_reads_the_quantized_floor(env):
    from dw import quantized_cache

    assert disk_caches.get("quantized").low_disk_floor is quantized_cache.floor_bytes


class TestEntryRoutes:
    def test_list_carries_entries_and_low_disk(self, api, tmp_path):
        root = quantized_root(tmp_path)
        quantized_entry(root, KEY_A)

        caches = {c["name"]: c for c in api.get("/api/caches").json()["caches"]}

        quantized = caches["quantized"]
        assert isinstance(quantized["low_disk"], bool)
        [entry] = quantized["entries"]
        assert entry["id"] == KEY_A
        assert entry["component"] == "vae"
        assert "entries" not in caches["inductor"]

    def test_delete_entry(self, api, tmp_path):
        root = quantized_root(tmp_path)
        quantized_entry(root, KEY_A, sizes=(100,))

        response = api.delete(f"/api/caches/quantized/entries/{KEY_A}")

        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "quantized"
        assert body["entry"] == KEY_A
        assert body["deleted"] is True
        assert body["freed"] >= 100
        assert not (root / KEY_A).exists()

    @pytest.mark.parametrize("entry", ["abc", "A" * 64, "z" * 64])
    def test_malformed_id_is_400(self, api, tmp_path, entry):
        root = quantized_root(tmp_path)
        quantized_entry(root, KEY_A)
        before = tree(root)

        response = api.delete(f"/api/caches/quantized/entries/{entry}")

        assert response.status_code == 400
        assert tree(root) == before

    def test_cache_without_entries_is_400(self, api, tmp_path):
        marked_root(tmp_path)
        response = api.delete(f"/api/caches/inductor/entries/{KEY_A}")
        assert response.status_code == 400

    def test_unknown_entry_is_404(self, api, tmp_path):
        root = quantized_root(tmp_path)
        quantized_entry(root, KEY_A)
        before = tree(root)

        response = api.delete(f"/api/caches/quantized/entries/{KEY_B}")

        assert response.status_code == 404
        assert tree(root) == before

    def test_unknown_cache_is_404(self, api, tmp_path):
        response = api.delete(f"/api/caches/nope/entries/{KEY_A}")
        assert response.status_code == 404

    @pytest.mark.parametrize("path", [f"/entries/{KEY_A}", ""])
    def test_refused_while_busy(self, api, tmp_path, path):
        root = quantized_root(tmp_path)
        quantized_entry(root, KEY_A)
        before = tree(root)
        api.manager.is_busy = lambda: True

        response = api.delete(f"/api/caches/quantized{path}")

        assert response.status_code == 409
        assert tree(root) == before

    @pytest.mark.parametrize("path", [f"/entries/{KEY_A}", ""])
    def test_refused_while_a_download_is_active(self, api, tmp_path, path):
        root = quantized_root(tmp_path)
        quantized_entry(root, KEY_A)
        before = tree(root)
        api.app.state.downloads.is_active = lambda: True

        response = api.delete(f"/api/caches/quantized{path}")

        assert response.status_code == 409
        assert "download" in response.json()["detail"]
        assert tree(root) == before

    def test_a_download_does_not_block_other_caches(self, api, tmp_path):
        marked_root(tmp_path)
        api.app.state.downloads.is_active = lambda: True

        assert api.delete("/api/caches/inductor").status_code == 200


def test_triton_default_created_after_start_is_clearable(env):
    triton = disk_caches.get("triton")
    disk_caches.mark_roots()
    assert disk_caches.scan(triton)["clearable"] is False

    # a compile creates the root after the server started
    populate(env / ".triton" / "cache")

    info = disk_caches.scan(triton)
    assert info["clearable"] is True
    assert info["files"] > 0
    assert disk_caches.clear("triton")["files"] > 0


# ------------------------------------------- the clear holds the queue (#838)


def test_job_submitted_during_a_clear_waits_for_it(api, tmp_path, monkeypatch):
    root, _ = marked_root(tmp_path)
    manager = api.manager
    seen = {}
    real_clear = disk_caches.clear

    def clear_with_a_submit_in_the_window(name):
        # Between the busy check and the clear: a job arrives
        seen["job"] = manager.submit(
            admitted=admitted_for(manager, valid_workflow()),
            workflow=valid_workflow(),
            arguments={"prompt": "hi"},
        )
        time.sleep(0.3)
        seen["status_during"] = seen["job"].status
        return real_clear(name)

    monkeypatch.setattr(disk_caches, "clear", clear_with_a_submit_in_the_window)

    response = api.delete("/api/caches/inductor")

    assert response.status_code == 200
    assert seen["status_during"] == "queued"
    job = seen["job"]
    deadline = time.time() + 5
    while job.status not in ("succeeded", "failed") and time.time() < deadline:
        time.sleep(0.02)
    assert job.status == "succeeded"
    assert tree(root) == [MARKER]


def test_hold_dispatch_refuses_when_busy_and_releases(api):
    manager = api.manager
    manager.is_busy = lambda: True
    with pytest.raises(JobsActive):
        with manager.hold_dispatch():
            pass
    assert manager._dispatch_holds == 0
    del manager.is_busy
    with manager.hold_dispatch():
        assert manager._dispatch_holds == 1
    assert manager._dispatch_holds == 0


# --------------------------------------------- orphaned staging (#839)


def test_scan_does_not_count_a_leftover_staging_directory(env, tmp_path):
    root, (total, count) = marked_root(tmp_path)
    leftover = root / ".clearing-1-abcd"
    leftover.mkdir()
    (leftover / "big.bin").write_bytes(b"x" * 500)

    info = disk_caches.scan(disk_caches.get("inductor"))

    assert (info["bytes"], info["files"]) == (total, count)


def test_mark_roots_and_clear_remove_a_leftover_staging_directory(env, tmp_path):
    root, _ = marked_root(tmp_path)
    leftover = root / ".clearing-1-abcd"
    leftover.mkdir()
    (leftover / "big.bin").write_bytes(b"x" * 500)

    disk_caches.mark_roots()
    assert not leftover.exists()
    assert (root / "a.bin").exists()

    again = root / ".clearing-2-ef01"
    again.mkdir()
    (again / "big.bin").write_bytes(b"x" * 500)
    result = disk_caches.clear("inductor")
    assert not again.exists()
    assert tree(root) == [MARKER]
    assert result["files"] == 3


# ------------------------------------------------- scan cost (#840)


def test_scan_all_reuses_its_answer_until_a_clear(env, tmp_path):
    root, (total, _) = marked_root(tmp_path)

    first = disk_caches.scan_all()
    (root / "late.bin").write_bytes(b"x" * 7)
    assert disk_caches.scan_all() is first

    disk_caches.clear("inductor")
    after = {c["name"]: c for c in disk_caches.scan_all()}
    assert after["inductor"]["bytes"] == 0


def test_scan_all_rescans_after_the_ttl(env, tmp_path, monkeypatch):
    root, (total, _) = marked_root(tmp_path)
    disk_caches.scan_all()
    (root / "late.bin").write_bytes(b"x" * 7)
    monkeypatch.setattr(disk_caches, "_SCAN_TTL", 0.0)

    inductor = {c["name"]: c for c in disk_caches.scan_all()}["inductor"]

    assert inductor["bytes"] == total + 7


# ---------------------------------- the refusal is logged once (#841)


def test_unmarked_root_refusal_is_info_at_startup_and_quiet_on_scans(
    env, tmp_path, caplog
):
    populate(tmp_path / "inductor")
    cache = disk_caches.get("inductor")

    with caplog.at_level(logging.INFO, logger="dw"):
        disk_caches.mark_roots()
        startup = [r for r in caplog.records if "not created by dw" in r.message]
        caplog.clear()
        disk_caches.scan(cache)
        disk_caches.scan_all()

    assert len(startup) >= 1
    assert not [r for r in caplog.records if "not created by dw" in r.message]
