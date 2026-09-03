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
