"""The model-use index (dw/model_use.py, #1445): when a job last used a repo."""

import json

import pytest

from dw import model_use


def test_record_then_read_round_trips():
    model_use.record(["org/a", "org/b"], 100.0)
    assert model_use.read() == {"org/a": 100.0, "org/b": 100.0}


def test_two_records_merge_different_repos():
    model_use.record(["org/a"], 100.0)
    model_use.record(["org/b"], 200.0)
    assert model_use.read() == {"org/a": 100.0, "org/b": 200.0}


def test_a_newer_stamp_replaces_an_older_one():
    model_use.record(["org/a"], 100.0)
    model_use.record(["org/a"], 300.0)
    assert model_use.read() == {"org/a": 300.0}


def test_an_older_stamp_never_replaces_a_newer_one():
    model_use.record(["org/a"], 300.0)
    model_use.record(["org/a"], 100.0)
    assert model_use.read() == {"org/a": 300.0}


@pytest.mark.parametrize(
    "name", ["variable:model_name", "https://x/y", "", None, 5, "a b/c"]
)
def test_a_name_that_is_not_a_hub_id_is_never_a_key(name):
    model_use.record([name, "org/ok"], 1.0)
    assert model_use.read() == {"org/ok": 1.0}


def test_only_bad_names_write_nothing():
    model_use.record(["variable:x", ""], 1.0)
    assert model_use.read() == {}
    assert not model_use._path().exists()


def test_record_of_none_is_a_no_op():
    model_use.record(None, 1.0)
    assert model_use.read() == {}


def test_the_cap_drops_the_oldest_stamps_first(monkeypatch):
    monkeypatch.setattr(model_use, "MAX_REPOS", 5)
    for index in range(5):
        model_use.record([f"org/r{index}"], 100.0 + index)
    model_use.record(["org/new"], 500.0)
    kept = model_use.read()
    assert len(kept) == 5
    assert "org/r0" not in kept
    assert "org/new" in kept
    assert {"org/r1", "org/r2", "org/r3", "org/r4"} <= set(kept)


def test_the_cap_keeps_every_repo_of_the_current_call(monkeypatch):
    monkeypatch.setattr(model_use, "MAX_REPOS", 5)
    for index in range(4):
        model_use.record([f"org/old{index}"], 1000.0 + index)
    # The current call names 3 repos with the oldest stamp of all, and the
    # room left is 1: the old (newer-stamped) repos lose, the call's do not
    current = ["org/c0", "org/c1", "org/c2"]
    model_use.record(current, 1.0)
    kept = model_use.read()
    assert set(current) <= set(kept)
    assert len(kept) == 5


def test_a_call_naming_more_than_the_cap_keeps_all_of_them(monkeypatch):
    monkeypatch.setattr(model_use, "MAX_REPOS", 3)
    model_use.record(["org/old"], 9999.0)
    current = [f"org/c{index}" for index in range(5)]
    model_use.record(current, 1.0)
    kept = model_use.read()
    assert set(kept) == set(current)


def test_a_missing_file_reads_empty():
    assert not model_use._path().exists()
    assert model_use.read() == {}


@pytest.mark.parametrize(
    "text", ["{not json", "", "[1, 2]", '{"repos": []}', '{"version": 1}']
)
def test_a_corrupt_file_reads_empty(text):
    path = model_use._path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    assert model_use.read() == {}


def test_a_malformed_entry_is_dropped_and_the_rest_kept():
    path = model_use._path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "repos": {
                    "org/good": 5,
                    "org/text": "yesterday",
                    "org/flag": True,
                    "variable:x": 3,
                },
            }
        )
    )
    assert model_use.read() == {"org/good": 5}


def test_a_record_over_a_corrupt_file_starts_over():
    path = model_use._path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken")
    model_use.record(["org/a"], 7.0)
    assert model_use.read() == {"org/a": 7.0}


def test_record_into_an_unwritable_location_does_not_raise(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setattr(model_use, "_path", lambda: blocker / "sub" / "model-use.json")
    model_use.record(["org/a"], 1.0)
    model_use.forget("org/a")
    assert model_use.read() == {}


def test_forget_on_a_missing_file_is_a_no_op():
    model_use.forget("org/a")
    assert not model_use._path().exists()


def test_forget_drops_one_key_and_leaves_the_others():
    model_use.record(["org/a", "org/b", "org/c"], 10.0)
    model_use.forget("org/b")
    assert model_use.read() == {"org/a": 10.0, "org/c": 10.0}


def test_forget_of_an_unrecorded_repo_leaves_the_file():
    model_use.record(["org/a"], 10.0)
    model_use.forget("org/other")
    assert model_use.read() == {"org/a": 10.0}


def test_a_forgotten_repo_can_be_recorded_afresh():
    model_use.record(["org/a"], 500.0)
    model_use.forget("org/a")
    model_use.record(["org/a"], 100.0)
    assert model_use.read() == {"org/a": 100.0}
