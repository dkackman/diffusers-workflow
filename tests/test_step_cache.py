import pytest

from dw.step_cache import StepCache, deep_equal


class FakeResult:
    def __init__(self, label, saved_files=None):
        self.label = label
        # Default to no files: a hit verifies every saved file still exists,
        # and most of these tests are about key matching, not disk state
        self.saved_files = [] if saved_files is None else list(saved_files)


def test_deep_equal_matches_identical_nested_dicts():
    a = {"prompt": "a cat", "settings": {"steps": 9, "images": [1, 2, 3]}}
    b = {"prompt": "a cat", "settings": {"steps": 9, "images": [1, 2, 3]}}
    assert deep_equal(a, b)


def test_deep_equal_rejects_changed_nested_value():
    a = {"prompt": "a cat", "settings": {"steps": 9}}
    b = {"prompt": "a cat", "settings": {"steps": 25}}
    assert not deep_equal(a, b)


def test_step_cache_hit_on_unchanged_step_data_and_seed():
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    result = FakeResult("first")
    cache.put(step_data, 42, result)

    hit = cache.get(step_data, 42, hits_this_run=set())

    assert hit is result


def test_step_cache_miss_when_step_data_changes():
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    cache.put(step_data, 42, FakeResult("first"))

    changed = {"name": "gen", "pipeline": {"arguments": {"prompt": "a dog"}}}
    hit = cache.get(changed, 42, hits_this_run=set())

    assert hit is None


def test_step_cache_miss_when_seed_changes():
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    cache.put(step_data, 42, FakeResult("first"))

    hit = cache.get(step_data, 99, hits_this_run=set())

    assert hit is None


def test_step_cache_miss_when_referenced_step_did_not_hit_this_run():
    cache = StepCache()
    step_data = {
        "name": "video",
        "pipeline": {"arguments": {"image": "previous_result:image_generation"}},
    }
    cache.put(step_data, 7, FakeResult("first"))

    # image_generation was NOT in hits_this_run - it re-ran and may have changed
    hit = cache.get(step_data, 7, hits_this_run=set())

    assert hit is None


def test_step_cache_hit_when_referenced_step_did_hit_this_run():
    cache = StepCache()
    step_data = {
        "name": "video",
        "pipeline": {"arguments": {"image": "previous_result:image_generation"}},
    }
    result = FakeResult("first")
    cache.put(step_data, 7, result)

    hit = cache.get(step_data, 7, hits_this_run={"image_generation"})

    assert hit is result


def test_step_cache_hit_when_referenced_step_hit_via_property_suffix():
    cache = StepCache()
    step_data = {
        "name": "video",
        "pipeline": {"arguments": {"mask": "previous_result:segment.mask"}},
    }
    result = FakeResult("first")
    cache.put(step_data, 7, result)

    hit = cache.get(step_data, 7, hits_this_run={"segment"})

    assert hit is result


def test_step_cache_miss_on_first_run():
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}

    assert cache.get(step_data, 42, hits_this_run=set()) is None


def test_step_cache_hit_when_output_dir_unchanged():
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    result = FakeResult("first")
    cache.put(step_data, 42, result, "/out/a")

    hit = cache.get(step_data, 42, hits_this_run=set(), output_dir="/out/a")

    assert hit is result


def test_step_cache_miss_when_output_dir_changes():
    """A hit reuses the entry's saved_files/manifest paths verbatim, so a
    changed effective output dir must force a miss rather than silently
    keep pointing at the old directory's files."""
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    cache.put(step_data, 42, FakeResult("first"), "/out/a")

    hit = cache.get(step_data, 42, hits_this_run=set(), output_dir="/out/b")

    assert hit is None


def test_step_cache_hit_when_saved_files_still_exist(tmp_path):
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    kept = tmp_path / "kept.png"
    kept.write_bytes(b"x")
    result = FakeResult("first", [str(kept)])
    cache.put(step_data, 42, result)

    assert cache.get(step_data, 42, hits_this_run=set()) is result


def test_step_cache_miss_when_a_saved_file_was_deleted(tmp_path):
    """A hit reports the cached entry's saved_files into the manifest and
    job history - if the user deleted one of those files, the entry is
    stale and the step must re-run rather than point at a missing file."""
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    kept = tmp_path / "kept.png"
    kept.write_bytes(b"x")
    gone = tmp_path / "gone.png"
    gone.write_bytes(b"x")
    cache.put(step_data, 42, FakeResult("first", [str(kept), str(gone)]))

    gone.unlink()

    assert cache.get(step_data, 42, hits_this_run=set()) is None


def test_deep_equal_is_false_rather_than_raising_on_array_values():
    """A realized step argument can hold a numpy array (or a tensor), whose
    == yields an array, not a bool. That must degrade to a cache miss, not
    abort the run with 'truth value of an array is ambiguous'."""
    numpy = pytest.importorskip("numpy")

    a = {"latents": numpy.zeros(4), "steps": 9}
    b = {"latents": numpy.zeros(4), "steps": 9}

    assert deep_equal(a, b) is False


def test_deep_equal_is_false_when_comparison_raises_a_type_error():
    class Hostile:
        def __eq__(self, other):
            raise TypeError("no comparison for you")

    assert deep_equal({"x": Hostile()}, {"x": Hostile()}) is False


def test_step_cache_evicts_the_least_recently_used_entry_over_the_cap():
    """The cache holds realized media - unbounded growth would work against
    the very OOM avoidance release_unreferenced_results exists for."""
    cache = StepCache(max_entries=3)
    for name in ("a", "b", "c"):
        cache.put({"name": name}, 1, FakeResult(name))

    # touch 'a' so 'b' becomes the least recently used
    assert cache.get({"name": "a"}, 1, hits_this_run=set()) is not None

    cache.put({"name": "d"}, 1, FakeResult("d"))

    assert cache.get({"name": "b"}, 1, hits_this_run=set()) is None
    for name in ("a", "c", "d"):
        assert cache.get({"name": name}, 1, hits_this_run=set()) is not None


def test_step_cache_has_a_default_entry_cap():
    cache = StepCache()
    assert cache.max_entries == StepCache.DEFAULT_MAX_ENTRIES
    for i in range(StepCache.DEFAULT_MAX_ENTRIES + 5):
        cache.put({"name": f"step{i}"}, 1, FakeResult(str(i)))

    assert len(cache._entries) == StepCache.DEFAULT_MAX_ENTRIES
    assert cache.get({"name": "step0"}, 1, hits_this_run=set()) is None
    assert cache.get({"name": "step4"}, 1, hits_this_run=set()) is None
    assert (
        cache.get(
            {"name": f"step{StepCache.DEFAULT_MAX_ENTRIES}"}, 1, hits_this_run=set()
        )
        is not None
    )


def test_step_cache_clear():
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    cache.put(step_data, 42, FakeResult("first"))

    cache.clear()

    assert cache.get(step_data, 42, hits_this_run=set()) is None
