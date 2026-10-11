"""dw's disk-cache registry (dw/disk_caches.py) and its routes (#828)."""

import builtins
import os
import sys

import pytest
from fastapi.testclient import TestClient

from dw import _inductor_cache_dir_default, disk_caches
from dw.disk_caches import MARKER, NotClearable, UnknownCache
from dw.server.app import create_app
from dw.server.jobs import JobManager

from tests.test_server import ScriptedWorkerManager, success_script


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
    monkeypatch.delitem(sys.modules, "triton", raising=False)
    return home


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


@pytest.fixture
def api(env, tmp_path):
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
    )
    with TestClient(app, base_url="http://localhost") as client:
        client.manager = manager
        yield client


def tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


class TestCacheRoutes:
    def test_list_reports_both_caches(self, api, tmp_path):
        root, (total, count) = marked_root(tmp_path)

        response = api.get("/api/caches")

        assert response.status_code == 200
        caches = {c["name"]: c for c in response.json()["caches"]}
        assert set(caches) == {"inductor", "triton"}
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

    def test_traversal_name_touches_nothing(self, api, tmp_path):
        root, _ = marked_root(tmp_path)
        before_root = tree(root)
        before_parent = tree(tmp_path)

        response = api.delete("/api/caches/..%2Fx")

        assert response.status_code == 404
        assert tree(root) == before_root
        assert tree(tmp_path) == before_parent
