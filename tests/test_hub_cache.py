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
