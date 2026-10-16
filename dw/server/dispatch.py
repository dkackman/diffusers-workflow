"""The dispatcher's pick: which queued job starts next, on which card.

`Dispatcher` is a mixin of `JobManager` (`dw/server/jobs.py`) - its methods
read the manager's queue and dispatch tables (`_pending`, `_needs`,
`_host_needs`, `_preferred`) and are called holding the manager's `_lock`.
Queue order, except that a job too big for every free card keeps its place
while a smaller one behind it takes the card (backfill), and a job that fits
a free card is also held while the host's RAM can't hold it beside the
running jobs (#869; the figure is `dw/server/host_ram_need.py`'s, the
comparison `pool.host_ram_wait`).
"""

from ..host_memory import host_memory_stats, process_rss_mb
from . import pool
from .job_record import QUEUED

# How often a job held for host RAM is looked at again while nothing else
# wakes the dispatcher - memory can be freed outside dw (#869)
HOST_RAM_RECHECK_SECONDS = 10.0


class _HostReading:
    """The host's MemAvailable for one dispatch pass, read on first ask."""

    def __init__(self):
        self._read = False
        self._available = None

    def available_mb(self):
        if not self._read:
            self._read = True
            # None where the platform can't say (macOS without psutil): no gate
            self._available = (host_memory_stats() or {}).get("available_mb")
        return self._available


def _worker_rss_mb(slot):
    """What `slot`'s worker holds in host RAM now, or None (not running, or
    unreadable)."""
    pid = slot.manager.pid() if hasattr(slot.manager, "pid") else None
    return process_rss_mb(pid) if pid is not None else None


class Dispatcher:
    """The queue scan `JobManager._run_loop` asks for its next job."""

    def _next_dispatch(self):
        """(slot, job) for the first queued job, in queue order, that fits
        a free card - the one it has an affinity for (`_choose_slot`) - or
        None. A job too big for every free card keeps its place, and the
        scan goes on past it: a smaller job behind it takes the card
        meanwhile (backfill). A job the host's RAM can't hold beside the
        running ones keeps its place the same way (`_host_ram_wait`).
        Called holding _lock."""
        if self._dispatch_holds:
            return None
        free = [slot for slot in self.slots if slot.current_job_id is None]
        if not free:
            return None
        host = _HostReading()
        for job_id in list(self._pending):
            job = self.jobs.get(job_id)
            if job is None or job.status != QUEUED:
                self._pending.remove(job_id)  # cancelled while waiting
                self._forget_dispatch(job, job_id)
                continue
            need = self._needs.get(job_id)
            fitting = [slot for slot in free if slot.fits(need)]
            if not fitting:
                # Waiting for a card, not for host RAM
                job.waiting_for = None
                continue
            slot = pool.choose_slot(
                fitting, self._job_identity(job), self._preferred.get(job_id)
            )
            reason = self._host_ram_wait(job_id, slot, host)
            if reason is not None:
                if job.waiting_for is None:
                    # Once per hold, not once per pass: a wait is a moment,
                    # so an event and a live field, never a warning (#869)
                    job.add_event(
                        {"event": "queue_wait", "reason": "host_ram", "message": reason}
                    )
                job.waiting_for = reason
                continue
            self._pending.remove(job_id)
            self._forget_dispatch(job)
            return slot, job
        return None

    def _forget_dispatch(self, job, job_id=None):
        """Drop what dispatch keeps for a job leaving the queue - started or
        cancelled. `job` may be None for one already gone."""
        job_id = job.id if job is not None else job_id
        self._needs.pop(job_id, None)
        self._host_needs.pop(job_id, None)
        self._preferred.pop(job_id, None)
        if job is not None:
            job.waiting_for = None

    def _held_for_host_ram(self):
        """The queued jobs held for host RAM now. Called holding _lock."""
        for job_id in self._pending:
            job = self.jobs.get(job_id)
            if job is not None and job.waiting_for is not None:
                yield job

    def _host_ram_wait(self, job_id, slot, host):
        """Why the queued `job_id` must not start on `slot` yet for host RAM,
        or None (pool.host_ram_wait, #869). `host` is this pass's reading,
        taken once and only when a job with a figure asks."""
        need = self._host_needs.get(job_id)
        if need is None:
            return None
        running = [
            (other, self.jobs.get(other.current_job_id))
            for other in self.slots
            if other.current_job_id is not None
        ]
        if not running:
            return None
        available = host.available_mb()
        if available is None:
            return None
        return pool.host_ram_wait(
            need,
            available,
            [
                (
                    job.host_ram_need[0] if job and job.host_ram_need else None,
                    _worker_rss_mb(other),
                )
                for other, job in running
            ],
            _worker_rss_mb(slot),
        )
