import json
import pathlib

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


def test_stats_reports_the_stale_drop_that_freed_its_bytes(tmp_path):
    """#418 follow-up: stats() is the read-only view of the same accounting
    the stale-drop test above exercises directly - entries/retained_bytes
    must go back down through this surface too, since it's the one a caller
    outside the process (get_memory) actually reads."""
    cache = StepCache()
    assert cache.stats() == {
        "entries": 0,
        "max_entries": cache.max_entries,
        "retained_bytes": 0,
        "max_retained_bytes": cache.max_retained_bytes,
    }

    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    gone = tmp_path / "gone.png"
    gone.write_bytes(b"x")
    result = FakeResult("first", [str(gone)], result_list=[{"videos": _frames(5)}])
    cache.put("w", step_data, 42, result, "/out", True)

    stats = cache.stats()
    assert stats["entries"] == 1
    assert stats["retained_bytes"] > 0

    gone.unlink()
    assert cache.get("w", step_data, 42, set(), "/out", True) is None

    assert cache.stats() == {
        "entries": 0,
        "max_entries": cache.max_entries,
        "retained_bytes": 0,
        "max_retained_bytes": cache.max_retained_bytes,
    }


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


# Every template's step_pipeline_keys, recorded at 1f94c2a0 - before
# pipeline identity moved from dw/workflow.py into dw/step_cache.py - by
# hashing each template's expanded_definition() (declared defaults folded,
# constants resolved, for_each expanded; not realize_args-realized, so a
# component_type is still its name). A pipeline's key names what is loaded
# and resident, and a worker compares this run's key against the last run's:
# a hash that moved would reload every resident model once and miss every
# step-cache entry that recorded a borrowed key. A template added since has
# no row and fails until one is recorded; a constant a diffusers upgrade
# changed is a row to regenerate, not a parity failure.
#
# Four rows were regenerated deliberately when a borrow chain became part
# of identity: the one step in each of base-and-refiner (main),
# ltx2/generative-upscale (upscaled), ltx2/refine-clip (refine) and
# ltx2/two-stage (upscale) reuses components, so its effective key hashes the
# own keys of its reuse closure (itself and its sources). Every step that
# reuses nothing kept its 1f94c2a0 hash.
TEMPLATE_PIPELINE_KEYS = {
    "assemble-and-score.json": {},
    "attention-processor.json": {
        "generate_with_sdpa": "8dc3e92052c445d5dbf20bf6e8b6fe91794cfa3772bb7366e2390889b4e99459"
    },
    "audio-trim-fade.json": {},
    "base-and-refiner.json": {
        "main": "384b0dccd2842b0c235b9d325e53e48bef5bc1b5150fd2e836f154d7dde32507",
        "sdxl_base": "5d64d980c2fb69c77b5547b8d8bd32d7de0e8e469ba7cc6ca73a60ad0e0b9058",
    },
    "best-of-n-to-video.json": {
        "still@0": "4e1fd3c983134c89fd764ab8635c410b08a14947fa5a34c372b6567fca12a4d3",
        "still@1": "4e1fd3c983134c89fd764ab8635c410b08a14947fa5a34c372b6567fca12a4d3",
        "still@2": "4e1fd3c983134c89fd764ab8635c410b08a14947fa5a34c372b6567fca12a4d3",
        "still@3": "4e1fd3c983134c89fd764ab8635c410b08a14947fa5a34c372b6567fca12a4d3",
    },
    "community-pipeline.json": {
        "invert": "7013cd33f88d6c850c7775f0bdf6d7be74ef427bf6fdf34c3663ddd793028a25"
    },
    "compose-workflows.json": {},
    "consistent-set.json": {
        "base": "ee282369a48c066f07fd13d0e7fc77b54f0342b518d8d5927b7900d9f759b6c1",
        "first": "ee282369a48c066f07fd13d0e7fc77b54f0342b518d8d5927b7900d9f759b6c1",
        "second": "ee282369a48c066f07fd13d0e7fc77b54f0342b518d8d5927b7900d9f759b6c1",
        "third": "ee282369a48c066f07fd13d0e7fc77b54f0342b518d8d5927b7900d9f759b6c1",
    },
    "controlnet-component.json": {
        "main": "23e479d111c0c2883755e0d32ef48b686d89f8967b535524ed8c6cb6f863bb9d"
    },
    "controlnet.json": {
        "FluxCanny": "cfe38a8a886222292666951457910698f54fb42d631de87009f4319e267bb6f4"
    },
    "depth-marigold.json": {
        "depth": "9338475d3fb48eab6e3dc8eb53cb07a916470906e338bb02a71e3ed0416a447b"
    },
    "describe-and-regenerate.json": {},
    "dissolve-between-shots.json": {},
    "embed-metadata.json": {
        "generate": "ba05817f175f71163f8007c8e9af3073686df61dbb090a8dd45f26a9dd502fa2"
    },
    "expand-prompt.json": {},
    "generate-speech.json": {},
    "image-edit.json": {
        "edit": "ee282369a48c066f07fd13d0e7fc77b54f0342b518d8d5927b7900d9f759b6c1"
    },
    "image-processors.json": {},
    "image-to-image.json": {
        "main": "bf8dcfe42d46a7031793e8aea4460cef13b511ca93b1a40cb78b50a98c50f4f8"
    },
    "image-to-text.json": {},
    "image-variation.json": {
        "main": "eb340f983349eb499e6fe01e29c65baf3fefd807e1f6a6a74f561768225fc496",
        "prior": "3253c892afe8ff8d02c464e2b175c519cfac4585fef0e9d2231683719db61a81",
    },
    "inpaint.json": {
        "fill": "a52f958d8c596182fb83182bad33df3da491b63a3ad758ec5807103b401f92eb"
    },
    "interpolate-frames.json": {
        "generate_video": "e35d0272e90c76b6730987d5c60a8dfda18290ef71276eb72246e1468ad16dfb"
    },
    "ip-adapter.json": {
        "main": "d75dffe2b60f5b3b65ad094896aa982029a830d14d149c0eb099caa8acf78291"
    },
    "lora-styles.json": {
        "couple": "e0d4d24cbe4caec22ee9e2afc43e9cfd772971430ed633cd289ff362e8213816",
        "font-design": "8f28fcccd557b862030508fa1db006f365afbf9b394f981488673df4c46eb548",
        "home-decoration": "8f0b0e4608b143d62d89641cddb7a4ff8b97c0d3cd4ba476f973fe6701bc36cc",
        "portrait-illustration": "23e72f3a1c737781910ef385cfc156730bfbe9ecc36a67c53249c70e5dbfa404",
        "portrait-photography": "6fd461304bdd784ecd9e94aea9a2f06f66ef61426ff9dc6cb552bf08e69d0520",
        "ppt-templates": "b03b08bd86d82d633a07ce1e62326223609340ba17db59b8aebaa9c43a9db8b5",
        "sandstorm-visual-effect": "0ec17d1beaaa8edafcbc5df532db4bef528694619f398fdaaf19f772c0d44871",
        "sparklers-visual-effect": "d7dec11561b3645b6c316159e06acd20c6ff84883712c5ba1b252c1de88424ae",
        "storyboard": "25c647a6d3e2013861ffafc783d8e15232f344f0c8579030e9d46b94bc5383d4",
        "visual-identity-design": "004a829d2a7c1d55b36f279ffc67057a165ecbf8a109da55f93f6acb09545afa",
    },
    "lora.json": {
        "txt2img": "47aba88047464c927983d4666abacb74ea253bb310c847e3e955f4e02658b277"
    },
    "ltx2/chained-segments.json": {
        "chained_image_to_video": "3a36def4e09fe10832bc1821c7cb6fd82d9e050ba149e5b6e7afa6ec66f60f68"
    },
    "ltx2/diffusion-decode.json": {
        "diffusion_decode": "127a7f5e3740bab50724eda1d06c465446cb4d47fb14868b82b8dc8c62ecdd37",
        "latents": "9c52ab1464a239a6e192ec36ffb4687b3dd61a66363886f38bb7ab5ceb83d638",
    },
    "ltx2/enhance-prompt.json": {
        "enhanced_image_to_video": "91126ea32e5fca936be3a03ddb4f1ec1488d95614b127730b6bb417a8dc5a93e"
    },
    "ltx2/extend-clip.json": {
        "extended": "7102a40f1655f06fd5173c45ea4989b877dc336221c4ab219fa0d0d9d845a69b",
        "opening": "9c52ab1464a239a6e192ec36ffb4687b3dd61a66363886f38bb7ab5ceb83d638",
    },
    "ltx2/generative-upscale.json": {
        "low_resolution": "fab6e3ec8feb0909dc44660990636418bdccbbb045154e1a56b26f235435acf0",
        "upscaled": "eb12a8878241511cecd56d4c41a168e9b59a11803656bef3948c704d18fba112",
    },
    "ltx2/image-to-video.json": {
        "image_to_video": "3a36def4e09fe10832bc1821c7cb6fd82d9e050ba149e5b6e7afa6ec66f60f68"
    },
    "ltx2/keyframes.json": {
        "keyframes_to_video": "ea27aeb8d07376a1295ea2d102ddc61a8d64fa70aeae26e02f6f48e825a6d9ee"
    },
    "ltx2/reference-sheet.json": {
        "shot": "f561e8c08eb096c02bcd252244e6dc94c6eb82f133fea559221c4acce364b3f8"
    },
    "ltx2/refine-clip.json": {
        "refine": "cdc51f4b605ee60ec9359390b4a9fb7f60762b419637699527f5a7b1457c1f22",
        "upscale": "f97469b7a921acca65cf45135dafd45796a29297794eabb43d3adabeeed3c33d",
    },
    "ltx2/restore-deblur.json": {
        "restored": "46a91b9693a5d42b7e43229989cfff47d7f58c26cec910ed3d18afcb06257396"
    },
    "ltx2/restore-decompression.json": {
        "restored": "afcb2e67a7c03447b5860bfe19f4785480e214eb54a39a61a53ccbf0e51717f6"
    },
    "ltx2/text-to-video.json": {
        "text_to_video": "c4da9234234392f4df1a76b9d3ffb2cbff735540e278709da11e43f9b4bc7313"
    },
    "ltx2/two-stage.json": {
        "base": "21b2f1afafba342e7ca064bf431ddb00307a49bd1404eb3011c31b3d486cd239",
        "refine": "21b2f1afafba342e7ca064bf431ddb00307a49bd1404eb3011c31b3d486cd239",
        "upscale": "caa78ea250fb765b6aff640c03219cfbc3a9c02366fd31997776520a20c8ee75",
    },
    "ltx2/upscale-clip.json": {
        "upscaled": "0f0195201e3be833527a0bc1ef04d29a1dd048a8877678ab1769f663fd27575f"
    },
    "minimax/chain-matched-and-aligned.json": {
        "audio_aligned_chained_reference_to_video_audio": "29913e317c4b9d427e617b120cec9213fc99a8b5c859f2d173372d11788a88e8"
    },
    "minimax/chain-matched-to-audio.json": {
        "chained_reference_to_video_audio": "1bce70dfbd9418b3dbe7ca8abd6f109dd8cdaffbdc802edce327454f36c3f600"
    },
    "minimax/chain-video-continuity.json": {
        "chained_reference_to_video_audio": "1bce70dfbd9418b3dbe7ca8abd6f109dd8cdaffbdc802edce327454f36c3f600"
    },
    "minimax/chained-segments.json": {
        "chained_keyframe_to_video_audio": "da965b4931829682072f59a5b91cdd0c212b39f585b445cc60b7b5c717cbe4e1"
    },
    "minimax/composable-references.json": {
        "video_reference_to_video_audio": "1bce70dfbd9418b3dbe7ca8abd6f109dd8cdaffbdc802edce327454f36c3f600"
    },
    "minimax/dialogue-short.json": {
        "draw_character_a": "65e14c94aa1bfa3a41f44e1a278ac84b09804577b2cf78c1a5a066366db5e20f",
        "draw_character_b": "65e14c94aa1bfa3a41f44e1a278ac84b09804577b2cf78c1a5a066366db5e20f",
        "shot@button": "29913e317c4b9d427e617b120cec9213fc99a8b5c859f2d173372d11788a88e8",
        "shot@cold_open": "29913e317c4b9d427e617b120cec9213fc99a8b5c859f2d173372d11788a88e8",
        "shot@deflect": "29913e317c4b9d427e617b120cec9213fc99a8b5c859f2d173372d11788a88e8",
        "shot@react": "29913e317c4b9d427e617b120cec9213fc99a8b5c859f2d173372d11788a88e8",
        "shot@tag": "29913e317c4b9d427e617b120cec9213fc99a8b5c859f2d173372d11788a88e8",
    },
    "minimax/enhance-prompt-with-image.json": {
        "keyframe_to_video_audio": "da965b4931829682072f59a5b91cdd0c212b39f585b445cc60b7b5c717cbe4e1"
    },
    "minimax/enhance-prompt.json": {
        "text_to_video_audio": "6995ccd88fcd70ede12ac1a10a3624fbbdb47eabe8714d9a191dcb393efa7ee9"
    },
    "minimax/first-and-last-frame.json": {
        "first_and_last_frame_to_video_audio": "da965b4931829682072f59a5b91cdd0c212b39f585b445cc60b7b5c717cbe4e1"
    },
    "minimax/generated-subject-reference.json": {
        "draw_subject": "65e14c94aa1bfa3a41f44e1a278ac84b09804577b2cf78c1a5a066366db5e20f",
        "reference_to_video_audio": "1bce70dfbd9418b3dbe7ca8abd6f109dd8cdaffbdc802edce327454f36c3f600",
    },
    "minimax/image-to-video.json": {
        "keyframe_to_video_audio": "da965b4931829682072f59a5b91cdd0c212b39f585b445cc60b7b5c717cbe4e1"
    },
    "minimax/last-frame-only.json": {
        "last_frame_to_video_audio": "da965b4931829682072f59a5b91cdd0c212b39f585b445cc60b7b5c717cbe4e1"
    },
    "minimax/music-video.json": {
        "draw_singer": "65e14c94aa1bfa3a41f44e1a278ac84b09804577b2cf78c1a5a066366db5e20f",
        "shot@closeup": "29913e317c4b9d427e617b120cec9213fc99a8b5c859f2d173372d11788a88e8",
        "shot@finale": "29913e317c4b9d427e617b120cec9213fc99a8b5c859f2d173372d11788a88e8",
        "shot@room": "29913e317c4b9d427e617b120cec9213fc99a8b5c859f2d173372d11788a88e8",
        "shot@wide_open": "29913e317c4b9d427e617b120cec9213fc99a8b5c859f2d173372d11788a88e8",
        "write_song": "1abd10e3c899fb5bbc490baa64467ed46d5005f78fa8f94bc777f3ddcaf81fb5",
    },
    "minimax/music.json": {
        "generate_music": "1abd10e3c899fb5bbc490baa64467ed46d5005f78fa8f94bc777f3ddcaf81fb5"
    },
    "minimax/reference-to-video.json": {
        "reference_to_video_audio": "1bce70dfbd9418b3dbe7ca8abd6f109dd8cdaffbdc802edce327454f36c3f600"
    },
    "minimax/shots-batch.json": {
        "shot@shot_1": "6995ccd88fcd70ede12ac1a10a3624fbbdb47eabe8714d9a191dcb393efa7ee9",
        "shot@shot_2": "6995ccd88fcd70ede12ac1a10a3624fbbdb47eabe8714d9a191dcb393efa7ee9",
        "shot@shot_3": "6995ccd88fcd70ede12ac1a10a3624fbbdb47eabe8714d9a191dcb393efa7ee9",
        "shot@shot_4": "6995ccd88fcd70ede12ac1a10a3624fbbdb47eabe8714d9a191dcb393efa7ee9",
        "shot@shot_5": "6995ccd88fcd70ede12ac1a10a3624fbbdb47eabe8714d9a191dcb393efa7ee9",
    },
    "minimax/storyboard.json": {
        "board_1_launch": "65e14c94aa1bfa3a41f44e1a278ac84b09804577b2cf78c1a5a066366db5e20f",
        "board_2_gutter": "65e14c94aa1bfa3a41f44e1a278ac84b09804577b2cf78c1a5a066366db5e20f",
        "board_3_shore": "65e14c94aa1bfa3a41f44e1a278ac84b09804577b2cf78c1a5a066366db5e20f",
        "voyage": "ee8e8884e8f049bf7de6aabe053a8c389ab6cd0c42108a04cc6fd1049e671300",
    },
    "minimax/video-with-audio-768p.json": {
        "text_to_video_audio": "748d703dd1745057386c6a167578c2f1b31ddad3df102364a5ec5389bcc5b001"
    },
    "minimax/video-with-audio.json": {
        "text_to_video_audio": "6995ccd88fcd70ede12ac1a10a3624fbbdb47eabe8714d9a191dcb393efa7ee9"
    },
    "minimax/voice-timbre-reference.json": {
        "reference_to_video_audio": "1bce70dfbd9418b3dbe7ca8abd6f109dd8cdaffbdc802edce327454f36c3f600"
    },
    "multi-image-reference.json": {
        "txt2img": "6e563b308f1508e0f24a660dcf25636774e2b70c4ccc723e8975421a7224b48d"
    },
    "outpaint.json": {
        "outpaint": "a52f958d8c596182fb83182bad33df3da491b63a3ad758ec5807103b401f92eb"
    },
    "prompt-weighting.json": {
        "txt2img": "5d31f1fa9f9fa65e19e82f6a5e9634ae4c5f2a0f4a7880b5f186ccd43860f560"
    },
    "qr-code.json": {
        "main": "f52e468931a9e9d546696cb9fccc365e7347b47e07730470b7a3ac419722fce5"
    },
    "recenter-crop.json": {},
    "restore-faces.json": {
        "generate": "8655c1f1635a7f9a1d3ce02a9cbf6f0f9e8b27130052d6e2758fcb7838270c14"
    },
    "segment-and-inpaint.json": {
        "inpaint": "a52f958d8c596182fb83182bad33df3da491b63a3ad758ec5807103b401f92eb"
    },
    "segment.json": {},
    "step-caching.json": {
        "txt2img": "ee477459eb2c2860c84b8da2913bf9c891666cc9c99d271e6255853539726786"
    },
    "sub-workflow.json": {},
    "surface-normals.json": {
        "normals": "5f0d7e4762dd0483a2bdc767b3bff57502323ff231940cb817634b38d1709b64"
    },
    "text-to-image.json": {
        "main": "4e1fd3c983134c89fd764ab8635c410b08a14947fa5a34c372b6567fca12a4d3"
    },
    "transcribe-audio.json": {},
    "upscale-diffusion.json": {},
    "upscale-spandrel.json": {},
}

TEMPLATES_DIR = pathlib.Path(__file__).resolve().parents[1] / "workflows" / "templates"


@pytest.mark.parametrize(
    "template",
    sorted(
        p.relative_to(TEMPLATES_DIR).as_posix() for p in TEMPLATES_DIR.rglob("*.json")
    ),
)
def test_every_template_keeps_its_pipeline_keys(template, tmp_path):
    from dw.step_cache import step_pipeline_keys
    from dw.workflow import Workflow

    assert template in TEMPLATE_PIPELINE_KEYS, (
        f"{template} has no recorded keys - add its step_pipeline_keys row"
    )
    path = TEMPLATES_DIR / template
    workflow = Workflow(json.loads(path.read_text()), str(tmp_path), str(path))
    keys = step_pipeline_keys(workflow.expanded_definition()["steps"])
    assert keys == TEMPLATE_PIPELINE_KEYS[template]
