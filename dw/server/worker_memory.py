"""Asking a card's worker about its memory and step cache, and clearing them.

Each card keeps its own last memory reading (WorkerSlot.last_memory). A
card's worker is asked live when it is idle; while it runs a job the run's
last report answers instead, labelled with why it is not live. Every
request takes the card's `slot.lock` with a bounded wait, because a job
holds that lock for its whole run, and a request must fall back rather than
block behind it.

The functions take the JobManager's `lock` to read which job a card runs.
The lock order is the dispatcher's: a slot lock is never taken while `lock`
is held.
"""

import queue
import time
import uuid
import logging

from ..worker_protocol import (
    ClearMemory,
    MemoryCleared,
    MemoryStatus,
    MemoryStatusReply,
    ProbeCache,
)
from .pool import slot_for

logger = logging.getLogger("dw")


class WorkerBusy(RuntimeError):
    """A memory clear found no idle card to clear (a 409 at the route)."""


def _running(slot, lock):
    """The job `slot`'s card is running, or None."""
    with lock:
        return slot.current_job_id


def record_memory(info, slot):
    """Remember a card's reading and when it was taken, so a later cached
    answer can say how old it is."""
    slot.last_memory = info
    slot.last_memory_at = time.time() if info is not None else None


def cached_memory(reason, slot):
    """A card's last reading, labelled with why it is not a live one. A
    caller comparing two readings must compare only `live: true` ones - a
    cached `info` was taken at another moment, and while a job loads a
    model it understates what is resident by however much has loaded
    since."""
    info = slot.last_memory
    age = None
    if info is not None and slot.last_memory_at is not None:
        age = round(time.time() - slot.last_memory_at, 1)
    return {
        "live": False,
        "info": info,
        "stale": info is not None,
        "reason": reason,
        "age_seconds": age,
    }


def probe_cache(slot, lock, command, timeout=5):
    """Which steps `slot`'s step cache would serve for `command` (the
    fields an execute command carries, minus its type), or None when the
    answer cannot be had right now - a job is running on that card, its
    worker is busy, or it did not answer in time. Never blocks a request
    behind a running job, for the same reason memory_status does not.

    No worker running is a definite answer, not an unknown one: the
    cache lives in the worker process, so a worker that is not running
    holds nothing.
    """
    if _running(slot, lock) is not None:
        return None
    if not slot.manager.worker_active:
        return []
    if not slot.lock.acquire(timeout=2):
        return None
    # A probe that timed out still answers eventually, onto the same
    # queue the next request reads - so each carries an id and request()
    # discards every reply that is not its own, rather than reporting
    # the previous workflow's hit list as this plan's
    try:
        reply = slot.manager.request(
            ProbeCache(request_id=uuid.uuid4().hex, **command), timeout
        )
    except (RuntimeError, queue.Empty) as e:
        logger.debug(f"Worker did not answer the cache probe: {e}")
        return None
    finally:
        slot.lock.release()
    cached = getattr(reply, "cached", None)
    return list(cached) if isinstance(cached, list) else None


def memory_status(slots, lock, timeout=5, device=None):
    """Memory per card. With `device`, that card's reading, naming it;
    without, the first card's reading at the top level (as the
    single-worker server answered) plus `workers`, one entry per card.

    Live stats when a card's worker is idle; the run's last report while
    it is busy. The lock acquire is bounded: a job holds its card's lock
    for its whole duration, and a poll that raced a job start must fall
    back to the cached reading, not block for hours.

    `live` says whether `info` was measured by this call. `stale` and
    `reason` say why it was not, and `age_seconds` how old the cached
    reading is; `info` is null when there has never been a reading, which
    means nothing is resident rather than that the answer is unknown.

    Raises ValueError for a `device` this server has no worker on."""
    if device is not None:
        return _slot_memory(slot_for(slots, device), lock, timeout)
    workers = [_slot_memory(slot, lock, timeout) for slot in slots]
    first = {key: value for key, value in workers[0].items() if key != "device"}
    return {**first, "workers": workers}


def _slot_memory(slot, lock, timeout):
    """One card's memory_status entry, naming the card."""
    return {"device": slot.ordinal(), **read_slot_memory(slot, lock, timeout)}


def read_slot_memory(slot, lock, timeout):
    """One card's reading: live when its worker is idle, else the cached
    one with the reason."""
    if _running(slot, lock) is not None:
        return cached_memory("job_running", slot)
    if not slot.manager.worker_active:
        return cached_memory("worker_stopped", slot)
    if not slot.lock.acquire(timeout=2):
        return cached_memory("worker_busy", slot)
    try:
        reply = slot.manager.request(MemoryStatus(request_id=uuid.uuid4().hex), timeout)
    except (RuntimeError, queue.Empty) as e:
        # A dead worker is exactly when the last reading taken before it
        # died is worth the most, so report that rather than failing the
        # request. Without this the caller gets a 503 at the one moment
        # it most wants a number
        detail = slot.manager.crash_details()
        logger.warning(f"Worker unavailable for memory status: {detail or e}")
        if detail is not None:
            # crash_details only answers for a process the OS has reaped,
            # so a worker that is merely slow to reply keeps its state -
            # a timeout is not evidence of death
            slot.manager.mark_crashed()
        return cached_memory("worker_unreachable", slot)
    finally:
        slot.lock.release()
    if isinstance(reply, MemoryStatusReply):
        record_memory(reply.info, slot)
        return {
            "live": True,
            "info": slot.last_memory,
            "stale": False,
            "reason": None,
            "age_seconds": 0.0,
        }
    return cached_memory("worker_unreachable", slot)


def clear_memory(slots, lock, timeout=30, device=None):
    """Drop every loaded pipeline and the step cache on each idle card -
    `device`'s alone when one is named - and report the readings taken
    right after. A card running a job is left alone: clearing under a
    run would rip out what it is using. The 30s timeout (vs.
    `memory_status`'s 5s) allows for this: actually freeing CUDA memory
    takes longer than reading a counter does.

    Returns `{cleared, info, workers}`, `workers` one entry per card:
    `{device, cleared: true, info}`, or `{device, cleared: false,
    reason: "job_running", job}` for a card running a job. `info` is
    the first cleared card's reading - null when nothing was resident,
    as with no worker running. With `device`, `{cleared, info, device}`
    for that card alone.

    Raises WorkerBusy when no card asked about is idle, ValueError for a
    `device` this server has no worker on, and RuntimeError when a
    worker does not answer with a clear."""
    asked = [slot_for(slots, device)] if device is not None else slots
    workers = [clear_slot(slot, lock, timeout) for slot in asked]
    cleared = [entry for entry in workers if entry["cleared"]]
    if not cleared:
        if device is not None:
            busy = workers[0]
            raise WorkerBusy(f"{busy['device']} is running job {busy['job']}")
        raise WorkerBusy("Every card is running a job")
    if device is not None:
        return {
            "cleared": True,
            "info": cleared[0]["info"],
            "device": cleared[0]["device"],
        }
    return {"cleared": True, "info": cleared[0]["info"], "workers": workers}


def clear_slot(slot, lock, timeout):
    """Clear one card's worker, or say why it was left alone."""
    device = slot.ordinal()
    job_id = _running(slot, lock)
    if job_id is not None:
        return {
            "device": device,
            "cleared": False,
            "reason": "job_running",
            "job": job_id,
        }
    if not slot.manager.worker_active:
        # Nothing to clear, and not a fault: the pipelines and the step
        # cache both live in the worker process, so no worker running
        # means both are already gone. An on-demand worker is
        # legitimately absent on an idle server (#206). A null `info` is
        # "no reading was taken", not a failure
        slot.last_identity = None
        return {"device": device, "cleared": True, "info": None}
    if not slot.lock.acquire(timeout=2):
        # A job was dispatched to it since the check above
        return {
            "device": device,
            "cleared": False,
            "reason": "job_running",
            "job": _running(slot, lock),
        }
    try:
        reply = slot.manager.request(ClearMemory(request_id=uuid.uuid4().hex), timeout)
    finally:
        slot.lock.release()
    if not isinstance(reply, MemoryCleared):
        raise RuntimeError(f"unexpected worker reply: {reply.TYPE}")
    # Its pipelines and step cache are gone: nothing is warm there now
    slot.last_identity = None
    record_memory(reply.info, slot)
    return {"device": device, "cleared": True, "info": slot.last_memory}
