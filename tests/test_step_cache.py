from dw.step_cache import StepCache, deep_equal


class FakeResult:
    def __init__(self, label):
        self.label = label
        self.saved_files = [f"{label}.png"]


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


def test_step_cache_clear():
    cache = StepCache()
    step_data = {"name": "gen", "pipeline": {"arguments": {"prompt": "a cat"}}}
    cache.put(step_data, 42, FakeResult("first"))

    cache.clear()

    assert cache.get(step_data, 42, hits_this_run=set()) is None
