import numpy as np
import pytest

from dw.for_each import MAX_FOR_EACH_ENTRIES
from dw.step_cache import (
    StepCache,
    deep_equal,
    reference_resolves_to,
    normalized_downstream,
    _result_bytes,
)


class FakeResult:
    def __init__(self, label, saved_files=None, result_list=None):
        self.label = label
        # Default to no files: a hit verifies every saved file still exists,
        # and most of these tests are about key matching, not disk state
        self.saved_files = [] if saved_files is None else list(saved_files)
        self.result_list = [] if result_list is None else result_list


def _frames(count, height=64, width=64, channels=3):
    # A stand-in for decoded video frames - real weight, not a mock of it
    return [np.zeros((height, width, channels), dtype=np.uint8) for _ in range(count)]


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


def test_stale_entry_eviction_subtracts_its_size(tmp_path):
    """#418: dropping a stale entry (its saved_files no longer exist) must
    free its bytes the same as a replace or an LRU eviction does - otherwise
    the phantom bytes only ever grow, and once they pass max_retained_bytes
    every put() evicts down to one entry."""
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    gone = tmp_path / "gone.png"
    gone.write_bytes(b"x")
    result = FakeResult("first", [str(gone)], result_list=[{"videos": _frames(5)}])
    cache.put("w", step_data, 42, result, "/out", True)
    size_before = cache._retained_bytes
    assert size_before > 0

    gone.unlink()

    assert cache.get("w", step_data, 42, set(), "/out", True) is None
    assert cache._retained_bytes == 0


def test_deep_equal_compares_array_values_by_content_rather_than_raising():
    """A realized step argument can hold a numpy array (or a tensor), whose
    == yields an array, not a bool - that must not abort the run with 'truth
    value of an array is ambiguous'. Two arrays with the same content are a
    match (#253: a from_file() reference rebuilt from an unchanged asset on
    every run must still compare equal to the entry it was cached under),
    and two with different content are correctly a miss."""
    numpy = pytest.importorskip("numpy")

    a = {"latents": numpy.zeros(4), "steps": 9}
    b = {"latents": numpy.zeros(4), "steps": 9}
    c = {"latents": numpy.ones(4), "steps": 9}

    assert deep_equal(a, b) is True
    assert deep_equal(a, c) is False


def test_deep_equal_matches_a_reconstructed_dataclass_holding_an_image():
    """#253: a from_file() reference (MiniMaxH3ImageReference,
    LTX2ReferenceCondition, ...) is a dataclass wrapping in-memory media,
    rebuilt fresh by realize_args on every run of a for_each member that
    takes an asset: reference - identical content, but never `is` the same
    object and never `==` by the dataclass's generated equality, since
    PIL.Image has no value equality and comparing tensor/array fields with
    `==` cannot resolve to a bool. Without special-casing these, that made
    every such step an unconditional cache miss."""
    from dataclasses import dataclass

    from PIL import Image

    numpy = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")

    @dataclass
    class FakeImageReference:
        image: object
        weight: float = 1.0

    def load():
        # A fresh object each call, like from_file() reloading the same file
        return FakeImageReference(image=Image.new("RGB", (4, 4), color=(1, 2, 3)))

    assert deep_equal(load(), load()) is True
    assert (
        deep_equal(
            load(), FakeImageReference(image=Image.new("RGB", (4, 4), color=(9, 9, 9)))
        )
        is False
    )

    # The same holds for a reference built straight over an array or tensor
    assert deep_equal(numpy.zeros((2, 2)), numpy.zeros((2, 2))) is True
    assert deep_equal(torch.zeros(3), torch.zeros(3)) is True
    assert deep_equal(torch.zeros(3), torch.ones(3)) is False


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


def test_the_default_cap_fits_a_maximal_for_each_run():
    # A maximal for_each run over two groups plus fixed steps must fit, or a
    # run evicts its own earlier members before it finishes.
    assert StepCache.DEFAULT_MAX_ENTRIES >= 2 * MAX_FOR_EACH_ENTRIES + 8


def test_result_bytes_sums_array_frames_and_ignores_scalars():
    result = FakeResult("video", result_list=[{"videos": _frames(10), "fps": 24}])
    assert _result_bytes(result) == 10 * 64 * 64 * 3


def test_result_bytes_does_not_double_count_a_shared_object():
    # get_artifact_list's fitted-in-place audio can be referenced by more
    # than one artifact in the same result_list - the byte budget should
    # not charge for it twice
    audio = np.zeros(1000, dtype=np.float32)
    result = FakeResult("av", result_list=[{"audio": audio}, {"audio": audio}])
    assert _result_bytes(result) == audio.nbytes


def test_step_cache_evicts_retained_entries_over_the_byte_budget():
    """Every shot@ member of a for_each group is legitimately retained (the
    gather step genuinely reads all of them), but nothing bounded how much
    decoded media that retention pins resident once the run that needed it
    is done (#368) - the byte budget is the bound, on top of the entry cap."""
    frame_bytes = 64 * 64 * 3
    cache = StepCache(max_entries=128, max_retained_bytes=4 * frame_bytes)
    for name in ("shot@a", "shot@b", "shot@c"):
        result = FakeResult(name, result_list=[{"videos": _frames(2)}])
        cache.put("w", {"name": name}, 1, result, "/out", True)

    # 3 entries x 2 frames each = 6 frames worth, over the 4-frame budget -
    # the least recently used (shot@a) is evicted despite being well under
    # the entry-count cap
    assert cache.get("w", {"name": "shot@a"}, 1, set(), "/out", True) is None
    assert cache.get("w", {"name": "shot@b"}, 1, set(), "/out", True) is not None
    assert cache.get("w", {"name": "shot@c"}, 1, set(), "/out", True) is not None


def test_step_cache_has_a_default_byte_budget():
    cache = StepCache()
    assert cache.max_retained_bytes == StepCache.DEFAULT_MAX_RETAINED_BYTES


def test_step_cache_never_evicts_the_only_retained_entry_over_budget():
    cache = StepCache(max_retained_bytes=1)
    result = FakeResult("big", result_list=[{"videos": _frames(5)}])
    cache.put("w", {"name": "only"}, 1, result, "/out", True)

    assert cache.get("w", {"name": "only"}, 1, set(), "/out", True) is result


def test_step_cache_unretained_result_does_not_count_against_the_byte_budget():
    cache = StepCache(max_retained_bytes=1)
    heavy = FakeResult("heavy", result_list=[{"videos": _frames(5)}])
    cache.put("w", {"name": "unretained"}, 1, heavy, "/out", False)

    assert cache._retained_bytes == 0


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


def test_unretained_entry_does_not_keep_the_artifacts_save_extracted():
    """The same rule, one layer down: Result memoizes get_artifact_list in
    _artifact_cache, which save() has already filled with the decoded frames
    and waveform - possibly still on the GPU. A shallow copy shares that dict,
    so emptying result_list released nothing at all for a step that saved."""
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    result = FakeResult("first")
    result.result_list = ["a decoded frame"]
    frames = ["a decoded frame"]
    result._artifact_cache = {id(result.result_list): frames}
    cache.put("w", step_data, 42, result, "/out", False)

    hit = cache.get("w", step_data, 42, set(), "/out", False)

    assert hit._artifact_cache == {}
    # the original keeps its own - only the cached copy starts empty
    assert result._artifact_cache == {id(result.result_list): frames}


def test_retain_result_true_but_not_retainable_stores_without_result_list():
    """A chain step's save_segments cleans up its segment files during
    save(), before put() runs - Result.retainable turns False, and the entry
    must be stored the same stripped way as retain_result=False, whatever the
    caller passed."""

    class UnretainableResult(FakeResult):
        retainable = False

    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    result = UnretainableResult("first")
    result.result_list = ["a decoded frame"]

    cache.put("w", step_data, 42, result, "/out", True)

    # A downstream step that needs the result misses, exactly like an entry
    # stored with retain_result=False
    assert cache.get("w", step_data, 42, set(), "/out", True) is None

    hit = cache.get("w", step_data, 42, set(), "/out", False)
    assert hit is not None
    assert hit.result_list == []
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


def test_normalized_downstream_true_when_a_later_step_normalizes_it():
    steps = [
        {"name": "write_song", "pipeline": {"arguments": {}}},
        {
            "name": "balanced",
            "task": {
                "command": "normalize_audio",
                "arguments": {"audio": "previous_result:write_song", "peak_dbfs": -3.0},
            },
        },
    ]
    assert normalized_downstream(steps, "write_song")


def test_normalized_downstream_false_when_only_a_non_normalizing_step_reads_it():
    """slice_audio reading the raw track for conditioning does not change
    what write_song's own written file sounds like (#286)."""
    steps = [
        {"name": "write_song", "pipeline": {"arguments": {}}},
        {
            "name": "slice",
            "task": {
                "command": "slice_audio",
                "arguments": {"audio": "previous_result:write_song"},
            },
        },
    ]
    assert not normalized_downstream(steps, "write_song")


def test_normalized_downstream_false_when_nothing_references_it():
    steps = [
        {"name": "write_song", "pipeline": {"arguments": {}}},
        {
            "name": "balanced",
            "task": {
                "command": "normalize_audio",
                "arguments": {"audio": "previous_result:other_step", "peak_dbfs": -3.0},
            },
        },
    ]
    assert not normalized_downstream(steps, "write_song")
