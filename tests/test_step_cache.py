import pytest

from dw.step_cache import StepCache, deep_equal, reference_resolves_to


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
    cache.put("w", step_data, 42, result, "/out", True)

    hit = cache.get("w", step_data, 42, set(), "/out", True)

    assert hit is result


def test_step_cache_miss_when_step_data_changes():
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    cache.put("w", step_data, 42, FakeResult("first"), "/out", True)

    changed = {"name": "gen", "pipeline": {"arguments": {"prompt": "a dog"}}}
    hit = cache.get("w", changed, 42, set(), "/out", True)

    assert hit is None


def test_step_cache_miss_when_seed_changes():
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    cache.put("w", step_data, 42, FakeResult("first"), "/out", True)

    hit = cache.get("w", step_data, 99, set(), "/out", True)

    assert hit is None


def test_step_cache_miss_when_referenced_step_did_not_hit_this_run():
    cache = StepCache()
    step_data = {
        "name": "video",
        "pipeline": {"arguments": {"image": "previous_result:image_generation"}},
    }
    cache.put("w", step_data, 7, FakeResult("first"), "/out", True)

    # image_generation was NOT in hits_this_run - it re-ran and may have changed
    hit = cache.get("w", step_data, 7, set(), "/out", True)

    assert hit is None


def test_step_cache_hit_when_referenced_step_did_hit_this_run():
    cache = StepCache()
    step_data = {
        "name": "video",
        "pipeline": {"arguments": {"image": "previous_result:image_generation"}},
    }
    result = FakeResult("first")
    cache.put("w", step_data, 7, result, "/out", True)

    hit = cache.get("w", step_data, 7, {"image_generation"}, "/out", True)

    assert hit is result


def test_step_cache_hit_when_referenced_step_hit_via_property_suffix():
    cache = StepCache()
    step_data = {
        "name": "video",
        "pipeline": {"arguments": {"mask": "previous_result:segment.mask"}},
    }
    result = FakeResult("first")
    cache.put("w", step_data, 7, result, "/out", True)

    hit = cache.get("w", step_data, 7, {"segment"}, "/out", True)

    assert hit is result


def test_step_cache_miss_on_first_run():
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}

    assert cache.get("w", step_data, 42, set(), "/out", True) is None


def test_step_cache_hit_when_output_dir_unchanged():
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    result = FakeResult("first")
    cache.put("w", step_data, 42, result, "/out/a", True)

    hit = cache.get("w", step_data, 42, set(), "/out/a", True)

    assert hit is result


def test_step_cache_miss_when_output_dir_changes():
    """A hit reuses the entry's saved_files/manifest paths verbatim, so a
    changed effective output dir must force a miss rather than silently
    keep pointing at the old directory's files."""
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    cache.put("w", step_data, 42, FakeResult("first"), "/out/a", True)

    hit = cache.get("w", step_data, 42, set(), "/out/b", True)

    assert hit is None


def test_step_cache_hit_when_saved_files_still_exist(tmp_path):
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    kept = tmp_path / "kept.png"
    kept.write_bytes(b"x")
    result = FakeResult("first", [str(kept)])
    cache.put("w", step_data, 42, result, "/out", True)

    assert cache.get("w", step_data, 42, set(), "/out", True) is result


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
    cache.put(
        "w", step_data, 42, FakeResult("first", [str(kept), str(gone)]), "/out", True
    )

    gone.unlink()

    assert cache.get("w", step_data, 42, set(), "/out", True) is None


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
        cache.put("w", {"name": name}, 1, FakeResult(name), "/out", True)

    # touch 'a' so 'b' becomes the least recently used
    assert cache.get("w", {"name": "a"}, 1, set(), "/out", True) is not None

    cache.put("w", {"name": "d"}, 1, FakeResult("d"), "/out", True)

    assert cache.get("w", {"name": "b"}, 1, set(), "/out", True) is None
    for name in ("a", "c", "d"):
        assert cache.get("w", {"name": name}, 1, set(), "/out", True) is not None


def test_step_cache_has_a_default_entry_cap():
    cache = StepCache()
    assert cache.max_entries == StepCache.DEFAULT_MAX_ENTRIES
    for i in range(StepCache.DEFAULT_MAX_ENTRIES + 5):
        cache.put("w", {"name": f"step{i}"}, 1, FakeResult(str(i)), "/out", True)

    assert len(cache._entries) == StepCache.DEFAULT_MAX_ENTRIES
    assert cache.get("w", {"name": "step0"}, 1, set(), "/out", True) is None
    assert cache.get("w", {"name": "step4"}, 1, set(), "/out", True) is None
    assert (
        cache.get(
            "w",
            {"name": f"step{StepCache.DEFAULT_MAX_ENTRIES}"},
            1,
            set(),
            "/out",
            True,
        )
        is not None
    )


def test_step_cache_clear():
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    cache.put("w", step_data, 42, FakeResult("first"), "/out", True)

    cache.clear()

    assert cache.get("w", step_data, 42, set(), "/out", True) is None


def test_entries_are_scoped_to_the_workflow_id():
    """Saved files are named '{workflow_id}-{step}.{i}', so an entry keyed by
    the bare step name lets a different workflow hit and republish the other
    workflow's file paths while writing none of its own."""
    cache = StepCache()
    step_data = {"name": "main", "pipeline": {"arguments": {"prompt": "a cat"}}}
    result = FakeResult("first")
    cache.put("workflow_a", step_data, 42, result, "/out", True)

    assert cache.get("workflow_b", step_data, 42, set(), "/out", True) is None
    assert cache.get("workflow_a", step_data, 42, set(), "/out", True) is result


def test_miss_when_the_upstream_entry_changed_since_this_entry_was_stored():
    """An upstream that hit this run is not enough - the entry must have been
    computed from the upstream generation now in the cache. A run cancelled
    between A's put and B's leaves B stale against the new A."""
    cache = StepCache()
    a = {"name": "A", "pipeline": {"arguments": {"prompt": "one"}}}
    b = {"name": "B", "pipeline": {"arguments": {"image": "previous_result:A"}}}
    cache.put("w", a, 1, FakeResult("a1"), "/out", True)
    cache.put("w", b, 1, FakeResult("b1"), "/out", True)

    # A re-ran with changed inputs and was re-put; the run was cancelled
    # before B's put, so B's entry still describes the old A
    cache.put(
        "w",
        {**a, "pipeline": {"arguments": {"prompt": "two"}}},
        1,
        FakeResult("a2"),
        "/out",
        True,
    )

    assert cache.get("w", b, 1, {"A"}, "/out", True) is None


def test_hit_when_the_upstream_generation_still_matches():
    cache = StepCache()
    a = {"name": "A", "pipeline": {"arguments": {"prompt": "one"}}}
    b = {"name": "B", "pipeline": {"arguments": {"image": "previous_result:A"}}}
    cache.put("w", a, 1, FakeResult("a1"), "/out", True)
    result = FakeResult("b1")
    cache.put("w", b, 1, result, "/out", True)

    assert cache.get("w", b, 1, {"A"}, "/out", True) is result


def test_unretained_entry_stores_a_result_with_an_empty_result_list():
    """A result no later step reads is on disk already - keeping its decoded
    frames or latent tensors alive for the life of the cache is what
    release_unreferenced_results exists to avoid."""
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    result = FakeResult("first")
    result.result_list = ["a decoded frame"]
    cache.put("w", step_data, 42, result, "/out", False)

    hit = cache.get("w", step_data, 42, set(), "/out", False)

    assert hit is not result
    assert hit.result_list == []
    assert hit.saved_files == result.saved_files
    assert result.result_list == ["a decoded frame"]


def test_miss_when_this_run_needs_a_result_the_entry_did_not_retain():
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    cache.put("w", step_data, 42, FakeResult("first"), "/out", False)

    assert cache.get("w", step_data, 42, set(), "/out", True) is None


def test_result_without_saved_files_raises_rather_than_hitting():
    """A result-like object with no saved_files must not silently pass the
    'every named file still exists' check."""
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}

    class NoSavedFiles:
        result_list = []

    cache.put("w", step_data, 42, NoSavedFiles(), "/out", True)

    with pytest.raises(AttributeError):
        cache.get("w", step_data, 42, set(), "/out", True)


def test_reference_resolves_to_matches_a_name_or_a_property_of_it():
    assert reference_resolves_to("gen", "gen")
    assert reference_resolves_to("gen.mask", "gen")
    assert not reference_resolves_to("generate", "gen")
    assert not reference_resolves_to("gen", "gen.mask")
