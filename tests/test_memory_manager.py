import gc

import torch
import pytest

from dw.pipeline_processors.memory_manager import MemoryManager


class FakeModel:
    """Stands in for an nn.Module: tracks device via .to(), and can be told
    to raise OutOfMemoryError on its next .to() call to a given device."""

    def __init__(self, name):
        self.name = name
        self.device = torch.device("cpu")
        self.oom_on = None

    def to(self, device):
        device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )
        if self.oom_on is not None and str(device) == str(self.oom_on):
            self.oom_on = None  # only OOM once, so a retry after eviction succeeds
            raise torch.OutOfMemoryError(f"{self.name} cannot fit on {device}")
        self.device = device
        return self


def test_load_moves_component_with_no_contention():
    mm = MemoryManager()
    model = FakeModel("a")
    mm.register(model, priority=1)

    mm.load(model, "cuda:0", "cpu")

    assert str(model.device) == "cuda:0"


def test_load_evicts_lower_priority_resident_on_oom():
    mm = MemoryManager()
    low = FakeModel("low")
    high = FakeModel("high")
    mm.register(low, priority=1)
    mm.register(high, priority=5)

    mm.load(low, "cuda:0", "cpu")
    assert str(low.device) == "cuda:0"

    high.oom_on = "cuda:0"
    mm.load(high, "cuda:0", "cpu")

    assert str(high.device) == "cuda:0"
    assert str(low.device) == "cpu"  # evicted to make room


def test_load_evicts_least_recently_used_when_priority_ties():
    mm = MemoryManager()
    first = FakeModel("first")
    second = FakeModel("second")
    mm.register(first, priority=1)
    mm.register(second, priority=1)

    mm.load(first, "cuda:0", "cpu")
    mm.load(second, "cuda:0", "cpu")  # both would now be "on cuda:0" in real life;
    # here they're just two independent fakes, so simulate contention directly:

    third = FakeModel("third")
    mm.register(third, priority=1)
    third.oom_on = "cuda:0"
    mm.load(third, "cuda:0", "cpu")

    # first was loaded before second, so it's the least-recently-used tie-break
    assert str(first.device) == "cpu"
    assert str(second.device) == "cuda:0"  # untouched - not the eviction candidate


def test_load_reraises_when_nothing_left_to_evict():
    mm = MemoryManager()
    model = FakeModel("only")
    mm.register(model, priority=1)
    model.oom_on = "cuda:0"

    with pytest.raises(torch.OutOfMemoryError):
        mm.load(model, "cuda:0", "cpu")


def test_unregister_removes_component_from_eviction_pool():
    mm = MemoryManager()
    victim = FakeModel("victim")
    mm.register(victim, priority=1)
    mm.load(victim, "cuda:0", "cpu")
    mm.unregister(victim)

    other = FakeModel("other")
    mm.register(other, priority=1)
    other.oom_on = "cuda:0"

    with pytest.raises(torch.OutOfMemoryError):
        mm.load(other, "cuda:0", "cpu")  # victim is gone, nothing left to evict


def test_registry_does_not_keep_a_component_alive():
    """Registration must not pin a component for the life of the process -
    when its pipeline is released and nothing else references it, the entry
    goes away on its own."""
    mm = MemoryManager()
    model = FakeModel("transient")
    mm.register(model, priority=1)
    assert len(mm._entries) == 1

    del model
    gc.collect()

    assert mm.live_entry_count() == 0


def test_eviction_skips_and_prunes_a_dead_reference():
    """A component that has been garbage collected is no benefit to evict,
    so it must not be chosen as the eviction candidate."""
    mm = MemoryManager()
    ghost = FakeModel("ghost")
    mm.register(ghost, priority=1)
    mm.load(ghost, "cuda:0", "cpu")
    del ghost
    gc.collect()

    other = FakeModel("other")
    mm.register(other, priority=1)
    other.oom_on = "cuda:0"

    # the only other entry is dead, so there is nothing real to evict
    with pytest.raises(torch.OutOfMemoryError):
        mm.load(other, "cuda:0", "cpu")

    assert mm.live_entry_count() == 1


def test_clear_empties_the_registry():
    mm = MemoryManager()
    kept = FakeModel("kept")
    mm.register(kept, priority=1)
    mm.load(kept, "cuda:0", "cpu")

    mm.clear()

    assert mm.live_entry_count() == 0

    other = FakeModel("other")
    mm.register(other, priority=1)
    other.oom_on = "cuda:0"
    with pytest.raises(torch.OutOfMemoryError):
        mm.load(other, "cuda:0", "cpu")  # registry was cleared, nothing to evict


def test_register_tolerates_a_component_that_cannot_be_weak_referenced():
    """Registration is best-effort - a component that does not support weak
    references simply is not tracked, rather than breaking its own load."""

    class NoWeakRef:
        __slots__ = ("device",)

        def __init__(self):
            self.device = torch.device("cpu")

        def to(self, device):
            self.device = torch.device(device)
            return self

    mm = MemoryManager()
    model = NoWeakRef()
    mm.register(model, priority=1)

    mm.load(model, "cuda:0", "cpu")

    assert str(model.device) == "cuda:0"
