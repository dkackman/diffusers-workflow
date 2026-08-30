"""The hub cache manager: scanning what from_pretrained left on disk and
deleting it through huggingface_hub's own strategy."""

import pytest

from dw.hub_cache import scan_models, delete_model


def make_repo(cache_dir, name="tiny", commit="aaaa1111", size=64):
    """A minimal hub-cache repo layout scan_cache_dir accepts."""
    repo = cache_dir / f"models--acme--{name}"
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(commit)
    (snapshot / "model.bin").write_bytes(b"x" * size)
    return repo


class TestScan:
    def test_repos_come_back_largest_first_with_revision_detail(self, tmp_path):
        make_repo(tmp_path, name="small", commit="a1", size=10)
        make_repo(tmp_path, name="big", commit="b1", size=1000)
        result = scan_models(tmp_path)
        assert [r["repo_id"] for r in result["repos"]] == ["acme/big", "acme/small"]
        assert result["size_on_disk"] >= 1010
        assert result["repos"][0]["revisions"][0]["refs"] == ["main"]
        assert result["disk_free"] > 0
        assert result["disk_total"] >= result["disk_free"]

    def test_a_missing_cache_dir_is_empty_not_an_error(self, tmp_path):
        result = scan_models(tmp_path / "never_downloaded_anything")
        assert result["repos"] == []
        assert result["size_on_disk"] == 0


class TestDelete:
    def test_delete_removes_the_repo_and_reports_freed_bytes(self, tmp_path):
        repo_path = make_repo(tmp_path, size=500)
        freed = delete_model("acme/tiny", cache_dir=tmp_path)
        assert freed >= 500
        assert not repo_path.exists()
        assert scan_models(tmp_path)["repos"] == []

    def test_delete_of_an_unknown_repo_is_refused_and_deletes_nothing(self, tmp_path):
        make_repo(tmp_path)
        with pytest.raises(ValueError, match="not in the hub cache"):
            delete_model("acme/other", cache_dir=tmp_path)
        assert len(scan_models(tmp_path)["repos"]) == 1


class FakeSibling:
    def __init__(self, size):
        self.size = size


class FakeInfo:
    def __init__(self, sizes):
        self.siblings = [FakeSibling(size) for size in sizes]


def fake_download(chunks, gate=None):
    """A download_fn that feeds `chunks` byte counts through the tracker,
    pausing at `gate` (if given) so a test can cancel mid-flight."""

    def download(repo_id, tqdm_class=None):
        tracker = tqdm_class(total=sum(chunks), desc=repo_id)
        for i, chunk in enumerate(chunks):
            if gate is not None and i == len(chunks) // 2:
                gate.wait(timeout=5)
            tracker.update(chunk)
        tracker.close()

    return download


class TestDownloadManager:
    def wait_status(self, manager, download_id, statuses, timeout=5.0):
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            status = manager.status(download_id)
            if status["status"] in statuses:
                return status
            time.sleep(0.01)
        raise AssertionError(f"never reached {statuses}: {status}")

    def test_download_completes_and_reports_progress(self):
        from dw.hub_cache import DownloadManager

        manager = DownloadManager(
            download_fn=fake_download([100, 200, 300]),
            info_fn=lambda repo_id: FakeInfo([100, 200, 300]),
        )
        started = manager.start("acme/tiny")
        status = self.wait_status(manager, started["id"], ["completed"])
        assert status["downloaded"] == 600
        assert status["total"] == 600
        assert status["error"] is None

    def test_cancel_stops_the_download_mid_flight(self):
        import threading

        from dw.hub_cache import DownloadManager

        gate = threading.Event()
        manager = DownloadManager(
            download_fn=fake_download([10] * 10, gate=gate),
            info_fn=lambda repo_id: FakeInfo([10] * 10),
        )
        started = manager.start("acme/tiny")
        manager.cancel(started["id"])
        gate.set()
        status = self.wait_status(manager, started["id"], ["cancelled"])
        assert status["downloaded"] < 100

    def test_failure_is_reported_not_raised(self):
        from dw.hub_cache import DownloadManager

        def broken(repo_id, tqdm_class=None):
            raise OSError("no such repo")

        manager = DownloadManager(
            download_fn=broken, info_fn=lambda repo_id: FakeInfo([])
        )
        started = manager.start("acme/missing")
        status = self.wait_status(manager, started["id"], ["failed"])
        assert "no such repo" in status["error"]

    def test_invalid_and_duplicate_repos_are_refused(self):
        import threading

        from dw.hub_cache import DownloadManager

        gate = threading.Event()
        manager = DownloadManager(
            download_fn=fake_download([10] * 10, gate=gate),
            info_fn=lambda repo_id: FakeInfo([10] * 10),
        )
        with pytest.raises(ValueError):
            manager.start("../../etc/passwd")

        started = manager.start("acme/tiny")
        with pytest.raises(ValueError, match="already downloading"):
            manager.start("acme/tiny")
        manager.cancel(started["id"])
        gate.set()
        self.wait_status(manager, started["id"], ["cancelled"])
