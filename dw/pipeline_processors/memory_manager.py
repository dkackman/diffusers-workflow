"""Cross-step model residency tracking for 'residency: on_demand' components.

Each on_demand component (apply_on_demand_placement, pipeline.py) moves
itself to the accelerator around its own calls and back off when idle, with
no awareness of what else is currently resident. On a card too small to
hold two such components at once, the second one's move to the accelerator
raises torch.OutOfMemoryError instead of making room.

MemoryManager is a process-wide registry: every on_demand component is
registered with a priority, and a move that hits OOM evicts whichever
*other* registered component is lowest-priority and least-recently-used,
then retries - the same trade Mellon's memory manager (utils/memory_menager.py)
makes for its node graph. It is not consulted by 'offload: model/sequential'
or 'group_offload', which install diffusers' own hooks - dw does not own
the individual .to() calls those make, so there is nothing to intercept.
"""

import time
import logging

import torch

logger = logging.getLogger("dw")


def _as_device(device):
    return torch.device(device) if not isinstance(device, torch.device) else device


class MemoryManager:
    def __init__(self):
        self._entries = {}  # id(component) -> {component, priority, last_used, device}

    def register(self, component, priority=1):
        self._entries[id(component)] = {
            "component": component,
            "priority": priority,
            "last_used": time.time(),
            # Not every component exposes .device - a bare nn.Module does not.
            # The starting value is only a guess anyway: load() and mark_idle()
            # record where the component actually ended up
            "device": getattr(component, "device", torch.device("cpu")),
        }

    def unregister(self, component):
        self._entries.pop(id(component), None)

    def load(self, component, device, offload_device):
        """Move `component` onto `device`, evicting lower-priority residents on OOM."""
        entry = self._entries.get(id(component))
        if entry is not None:
            entry["last_used"] = time.time()

        while True:
            try:
                component.to(device)
                if entry is not None:
                    entry["device"] = _as_device(device)
                return
            except torch.OutOfMemoryError:
                victim_id = self._pick_eviction_candidate(
                    device, exclude_id=id(component)
                )
                if victim_id is None:
                    raise
                self._evict(victim_id, offload_device)

    def mark_idle(self, component, offload_device):
        """Record that `component` has been moved back to `offload_device`."""
        entry = self._entries.get(id(component))
        if entry is not None:
            entry["device"] = _as_device(offload_device)

    def _pick_eviction_candidate(self, device, exclude_id):
        device = str(device)
        candidates = [
            (entry["priority"], entry["last_used"], comp_id)
            for comp_id, entry in self._entries.items()
            if comp_id != exclude_id and str(entry["device"]) == device
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda c: (c[0], c[1]))
        return candidates[0][2]

    def _evict(self, comp_id, offload_device):
        entry = self._entries.get(comp_id)
        if entry is None:
            return
        logger.debug(
            f"Evicting a lower-priority on-demand component to free {entry['device']}"
        )
        entry["component"].to(offload_device)
        entry["device"] = _as_device(offload_device)


memory_manager = MemoryManager()
