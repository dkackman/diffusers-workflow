"""The worker pool's cards and the policy that places a job on one.

A WorkerSlot is one `--devices` card: its worker, the job it runs and the
lock that serializes talking to it. The functions here read only the slots
they are given and take no lock - fit (`check_fits`, `dispatch_need`), the
largest card (`largest_ceiling_gb`) and affinity (`choose_slot`). The
JobManager (`dw/server/jobs.py`) calls them under its own lock where the
answer must agree with dispatch (`JobManager.route`, `_next_dispatch`).
"""

import logging
import threading

from ..devices import device_ordinal

logger = logging.getLogger("dw")


class WorkerSlot:
    """One card's worker and what the dispatcher knows about it: the job it
    is running, and the lock that serializes talking to it. A job holds its
    slot's lock for its whole run, as the single worker's did; a memory or
    cache request takes it with a bounded wait."""

    def __init__(self, manager):
        self.manager = manager
        self.lock = threading.Lock()
        self.current_job_id = None
        self.started_at = None
        # The workflow identity the worker last ran - JobManager.identity_of
        self.last_identity = None
        # This card's last memory reading and when it was taken
        self.last_memory = None
        self.last_memory_at = None
        # The worker process _rank_for_oom last looked at
        self.ranked_pid = None

    @property
    def device(self):
        return getattr(self.manager, "device", None)

    def capacity_gb(self):
        """The card's VRAM in the catalog's GB, or None where it cannot be
        read - a card of unknown size takes any job, so an unreadable
        ceiling never strands the queue."""
        try:
            return self.manager.capacity_gb()
        except Exception:
            logger.debug("Could not read a worker's card capacity", exc_info=True)
            return None

    def ceiling_gb(self):
        """The most a declared vram_estimate may project on this card, or
        None where it cannot be read (WorkerManager.ceiling_gb)."""
        try:
            return self.manager.ceiling_gb()
        except Exception:
            logger.debug("Could not read a worker's card ceiling", exc_info=True)
            return None

    def label(self):
        """The card as a job record names it, or None."""
        try:
            return self.manager.device_label()
        except Exception:
            logger.debug("Could not read the worker's device", exc_info=True)
            return None

    def device_fields(self):
        """The card as a job record stores it - `("cuda:1", "NVIDIA GeForce
        RTX 3090")` - or `(None, None)` where it cannot be read."""
        try:
            return self.manager.device_fields()
        except Exception:
            logger.debug("Could not read the worker's device", exc_info=True)
            return None, None

    def ordinal(self):
        """The card as `cuda:1` - what a `device` argument names it by."""
        return device_ordinal(self.device)

    def warm_identity(self):
        """The workflow identity this card's worker holds warm, or None when
        no worker is running - a worker that is not running holds nothing."""
        if not self.manager.worker_active:
            return None
        return self.last_identity

    def alive(self):
        process = getattr(self.manager, "worker_process", None)
        return bool(
            self.manager.worker_active and process is not None and process.is_alive()
        )

    def fits(self, need):
        """Whether a job needing `need` - (GB, hard) or None - may start
        here. A hard (declared vram_estimate) need is held to the card's
        ceiling, as admission holds it; a soft catalog `cost` figure to the
        card's size in the catalog's GB."""
        if need is None:
            return True
        gb, hard = need
        capacity = self.ceiling_gb() if hard else self.capacity_gb()
        return capacity is None or gb <= capacity


# ---------------------------------------------------------------- affinity


def slot_for(slots, device):
    """The slot running `device` ('cuda:1'), or raise ValueError naming
    the cards this server has."""
    wanted = device_ordinal(device)
    for slot in slots:
        if slot.ordinal() == wanted:
            return slot
    cards = ", ".join(str(slot.ordinal()) for slot in slots)
    raise ValueError(f"This server has no worker on {device}: its cards are {cards}")


def choose_slot(candidates, identity=None, preferred=None):
    """The candidate a job with `identity` goes to: the card a rerun
    prefers, then the card whose worker holds that identity warm, then
    a card whose worker is not running (nothing warm to evict), then
    the first in `--devices` order."""
    if preferred is not None:
        for slot in candidates:
            if slot.ordinal() == preferred:
                return slot
    if identity is not None:
        for slot in candidates:
            if slot.warm_identity() == identity:
                return slot
    for slot in candidates:
        if not slot.manager.worker_active:
            return slot
    return candidates[0]


# ---------------------------------------------------------------- VRAM fit


def capacities(slots, hard=False):
    """(slot, GB) for every card whose size can be read - its ceiling
    for a hard need, its catalog size for a soft one (WorkerSlot.fits)."""
    readings = [
        (slot, slot.ceiling_gb() if hard else slot.capacity_gb()) for slot in slots
    ]
    return [(slot, capacity) for slot, capacity in readings if capacity]


def largest_card(slots):
    """(slot, GB) of the card with the most VRAM it can be held to, or
    None when no size could be read."""
    readings = capacities(slots, hard=True)
    if not readings:
        return None
    return max(readings, key=lambda reading: reading[1])


def card_name(slot):
    """How a refusal names a card, whichever sentence it is in."""
    return slot.label() or slot.device or "this server's card"


def largest_ceiling(slots):
    """(GB, label) of the card `largest_ceiling_gb` reads, the label
    naming it in a refusal; (None, None) when no size could be read."""
    largest = largest_card(slots)
    if largest is None:
        return None, None
    slot, capacity = largest
    return capacity, f"the largest card here ({card_name(slot)})"


def largest_ceiling_gb(slots):
    """The most VRAM any card here can be held to (WorkerSlot.ceiling_gb),
    or None when no card's size could be read (no torch, or the probe
    failed). Admission checks a declared vram_estimate against this
    rather than the process's own device, which under --devices is only
    the first card."""
    readings = capacities(slots, hard=True)
    return max(capacity for _, capacity in readings) if readings else None


def unfit_message(slots, need):
    """Why no card here can ever run a job needing `need` (admission's
    `vram_need`, (GB, hard)), or None when one can. Only a declared
    vram_estimate's projection is hard: a catalog `cost` figure is what
    a card the model was measured on held, not a floor, so it orders
    dispatch but refuses nothing."""
    if not need or not need[1] or need[0] is None:
        return None
    card = largest_card(slots)
    if card is None:
        return None
    slot, largest = card
    if need[0] <= largest:
        return None
    name = card_name(slot)
    return (
        f"This job needs {need[0]:.1f} GB of VRAM, more than any card here "
        f"has: the largest is {name} ({largest:g} GB usable)"
    )


def check_fits(slots, admission):
    """Raise ValueError - a 400 at the route - for a request that is
    not admissible, or that needs more VRAM than the largest card here.
    An inadmissible request's message gains the card sentence when that
    applies too, so the caller learns both at once."""
    unfit = unfit_message(slots, getattr(admission, "vram_need", None))
    if not admission.ok:
        message = admission.message()
        raise ValueError(f"{message}; {unfit}" if unfit else message)
    if unfit:
        raise ValueError(unfit)


def dispatch_need(slots, need):
    """What a job must find on a free card before it starts - (GB, hard)
    for WorkerSlot.fits - or None for any card. A soft (`cost`) figure no
    card here meets is dropped, so the job runs on whatever card is free,
    as it would have on one card."""
    if not need or need[0] is None:
        return None
    gb, hard = need
    if hard or any(capacity >= gb for _, capacity in capacities(slots)):
        return (gb, hard)
    return None
