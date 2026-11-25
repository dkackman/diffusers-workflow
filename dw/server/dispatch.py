"""The dispatcher's pick: which queued job starts next, on which card.

`Dispatcher` is a mixin of `JobManager` (`dw/server/jobs.py`) - its methods
read the manager's queue and dispatch tables (`_pending`, `_needs`,
`_host_needs`, `_preferred`) and are called holding the manager's `_lock`.
Queue order, except that a job too big for every free card keeps its place
while a smaller one behind it takes the card (backfill), and a job that fits
a free card is also held while the host's RAM can't hold it beside the
running jobs (#869; the figure is `dw/server/host_ram_need.py`'s, the
comparison `pool.host_ram_wait`). What a dispatch does to the other workers
is here as well: the warm idle ones a held job asked released
(`_release_idle_workers`, #1407) and the OOM-killer ranking set as a job
starts (`_rank_for_oom`, #881).

The loop that runs the pick is here too: `_run_loop` hands each pick to its
own thread (`_run_job`), which drives the card's worker (`_run_on`) and
folds its messages into the Job (`dw/server/job_results.py`).
"""

import functools
import logging
import threading
import time

from ..devices import device_ordinal
from ..host_memory import (
    HostClaims,
    committed_rss_mb,
    device_term_mb,
    host_memory_stats,
)
from ..worker_protocol import Execute
from . import job_results, pool, worker_memory
from .job_record import CANCELLED, FAILED, QUEUED, RUNNING, TERMINAL_STATES

logger = logging.getLogger("dw")

# How often a job held for host RAM is looked at again while nothing else
# wakes the dispatcher - memory can be freed outside dw (#869)
HOST_RAM_RECHECK_SECONDS = 10.0

# A running job is charged the host RAM it has still to grow into only while
# it is loading. Past this, its worker's RSS is its footprint: a recorded peak
# is a lower bound taken at phase boundaries, and charging an hours-long job
# the gap to a stale projection starves every queued job on an otherwise idle
# card (#1357)
HOST_RAM_SETTLED_SECONDS = 900.0


def _gpu_busy_message(running, ahead, now=None):
    """The `waiting_for` text of a job queued with no card free. `running` is
    [(device, started_at)], one per busy card; the minutes are measured at
    `now`, so a read after the pass is current (#1618)."""
    now = time.time() if now is None else now

    def card(device, started):
        if started is None:
            return device
        return f"{device}, {max(0, int((now - started) // 60))} min in"

    if len(running) == 1:
        behind = f"the running job on {card(*running[0])}"
    else:
        behind = (
            f"{len(running)} running jobs on every card ("
            + "; ".join(card(*r) for r in running)
            + ")"
        )
    return (
        f"queued behind {behind} and {ahead} queued job(s) ahead; "
        "no start estimate - the queue runs in order and a task-only "
        "job holds a card like any other"
    )


class _HostReading:
    """The host's MemAvailable for one dispatch pass, read on first ask, and
    the jobs other dw.serve processes on this host are running (#1227)."""

    def __init__(self, claims=None):
        self._read = False
        self._available = None
        self._claims = claims
        self._foreign = None

    def foreign(self):
        if self._foreign is None:
            self._foreign = self._claims.foreign() if self._claims else []
        return self._foreign

    def available_mb(self):
        if not self._read:
            self._read = True
            # None where the platform can't say (macOS without psutil): no gate
            self._available = (host_memory_stats() or {}).get("available_mb")
        return self._available


def _worker_rss_mb(slot):
    """What `slot`'s worker has committed in host RAM now, or None (not
    running, or unreadable) - its mapped checkpoint pages left out, on the
    same basis as the committed peaks a need is projected from.

    Called holding the dispatcher's lock, so it reads /proc/<pid>/status
    only, never smaps (seconds for a large worker). The driver-pinned term
    comes from the worker's last memory report (`slot.last_memory`), taken
    outside the lock; with none yet, the reading is status-only."""
    pid = slot.manager.pid() if hasattr(slot.manager, "pid") else None
    if pid is None:
        return None
    committed = committed_rss_mb(pid, device=False)
    if committed is None:
        return None
    return committed + _pinned_mb(slot)


def _pinned_mb(slot):
    """The driver-pinned host memory of `slot`'s worker at its last report,
    published with its claim so another server reads it on this basis."""
    return device_term_mb(getattr(slot, "last_memory", None)) or 0.0


def _worker_pid(slot):
    pid = slot.manager.pid() if hasattr(slot.manager, "pid") else None
    return pid if isinstance(pid, int) and pid > 0 else None


HOST_RAM_RELEASE_SECONDS = 60  # don't re-release a card's worker sooner


class Dispatcher:
    """The queue scan `JobManager._run_loop` asks for its next job."""

    def _host_claims(self):
        if getattr(self, "_claims", None) is None:
            self._claims = HostClaims()
        return self._claims

    def _running_claims(self):
        """[(worker pid, projected peak MB)] for this server's running jobs
        that have a figure and a worker to read. Called holding _lock."""
        out = []
        for slot in self.slots:
            job = self.jobs.get(slot.current_job_id) if slot.current_job_id else None
            pid = _worker_pid(slot)
            if job is not None and job.host_ram_need and pid is not None:
                out.append((pid, job.host_ram_need[0], _pinned_mb(slot)))
        return out

    def _start_worker(self, slot, job):
        """Start `slot`'s worker if it is not up, then claim `job` for the
        other servers on the host. `_next_dispatch` can only claim a job whose
        worker already has a pid; a cold card's worker is spawned here, in
        the job thread, so without this the job runs unseen by them (#1227)."""
        slot.manager.ensure_worker(self.log_level)
        if not job.host_ram_need:
            return
        with self._lock:
            claims = self._host_claims()
            with claims.lock():
                claims.publish(self._running_claims())

    def _rank_for_oom(self, slot, job=None):
        """Score the dispatching worker 1000 for the kernel's OOM killer and
        put every other live worker back to its spawn-time floor, on every
        dispatch, so a host-RAM squeeze kills the newest job whatever the
        workers' sizes (#881). The floor rather than 0: an unprivileged write
        below it is refused (WorkerManager.reset_oom_score_adj). Nothing is
        written while it is the only worker, so one card runs exactly as
        before. A write the kernel refuses is a warning on the job, since a
        tie at 1000 is the failure this exists to prevent."""
        manager = slot.manager
        if not hasattr(manager, "set_oom_score_adj"):
            return
        others = [
            other.manager for other in self.slots if other is not slot and other.alive()
        ]
        if not others:
            return
        refused = []
        if manager.set_oom_score_adj(1000) is False:
            refused.append((manager, 1000))
        for other in others:
            if (
                hasattr(other, "reset_oom_score_adj")
                and other.reset_oom_score_adj() is False
            ):
                refused.append((other, other.oom_score_floor or 0))
        for worker, value in refused:
            message = (
                f"Could not set oom_score_adj to {value} on the {worker.device} "
                f"worker (pid {worker.pid()}): a host-RAM squeeze may not kill "
                "the newest job first"
            )
            logger.warning(message)
            if job is not None:
                job.add_event({"event": "warning", "message": message})

    def _next_dispatch(self):
        """(slot, job) for the next job to start, or None - `_pick` under the
        host-wide claims lock, so a server starting a heavy job and another
        deciding on one at the same moment see each other's claim (#1227).
        Called holding _lock."""
        claims = self._host_claims()
        with claims.lock():
            running = self._running_claims()
            claims.publish(running)
            picked = self._pick(claims)
            if picked is not None:
                slot, job = picked
                pid = _worker_pid(slot)
                need = job.host_ram_need
                if need and pid is not None:
                    claims.publish(running + [(pid, need[0], _pinned_mb(slot))])
            return picked

    def _pick(self, claims):
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
            self._hint_gpu_busy()
            return None
        host = _HostReading(claims)
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
                self._set_wait(job, None, None)
                continue
            slot = pool.choose_slot(
                fitting, self._job_identity(job), self._preferred.get(job_id)
            )
            reason = self._host_ram_wait(job_id, slot, host)
            if reason is not None:
                self._set_wait(job, "host_ram", reason)
                continue
            self._pending.remove(job_id)
            self._forget_dispatch(job)
            return slot, job
        return None

    @staticmethod
    def _set_wait(job, kind, message):
        """Record why a queued job is held: `kind` None clears it. The
        `queue_wait` event goes out once per hold, not once per pass: a wait
        is a moment, so an event and a live field, never a warning (#869)."""
        if kind is not None and getattr(job, "wait_kind", None) != kind:
            job.add_event({"event": "queue_wait", "reason": kind, "message": message})
        job.wait_kind = kind
        job.waiting_for = message  # the setter drops any wait_render

    def _hint_gpu_busy(self):
        """Tell every queued job that every card is busy: how many jobs run,
        on which cards and for how long, and how many are ahead of it. Called
        holding _lock, when no card is free (#1355). The queue stays FIFO;
        this only lets a consumer tell waiting from stuck."""
        running = []
        for slot in self.slots:
            if slot.current_job_id is None:
                continue
            # A server run without --devices has slot.device None: name the
            # card it runs on anyway (#1618)
            running.append(
                (device_ordinal(slot.device), getattr(slot, "started_at", None))
            )
        ahead = 0
        for job_id in list(self._pending):
            job = self.jobs.get(job_id)
            if job is None or job.status != QUEUED:
                continue
            render = functools.partial(_gpu_busy_message, running, ahead)
            self._set_wait(job, "gpu_busy", render())
            # The minutes are worked out when waiting_for is read, not once
            # per pass: nothing wakes the dispatcher while a job runs (#1618)
            job.wait_render = render
            ahead += 1

    def _forget_dispatch(self, job, job_id=None):
        """Drop what dispatch keeps for a job leaving the queue - started or
        cancelled. `job` may be None for one already gone."""
        job_id = job.id if job is not None else job_id
        self._needs.pop(job_id, None)
        self._host_needs.pop(job_id, None)
        self._preferred.pop(job_id, None)
        if job is not None:
            job.waiting_for = None
            job.wait_kind = None

    def _release_idle_workers(self):
        """Clear the warm idle workers held jobs asked released for host RAM
        (#1407), say so on each job, and let the next pass re-check. Called
        holding _wake; drops it while a worker clears (seconds)."""
        asked, self._release_idle = self._release_idle, {}
        self._wake.release()
        try:
            for job_id, cards in asked.items():
                job = self.jobs.get(job_id)
                for card in cards:
                    self._released_at[id(card)] = time.time()
                    try:
                        worker_memory.clear_slot(card, self._lock, 30)
                    except Exception as exc:
                        logger.warning(
                            "releasing %s for host RAM failed: %s", card.device, exc
                        )
                        continue
                    if job is not None:
                        job.add_event(
                            {
                                "event": "queue_wait",
                                "reason": "host_ram",
                                "message": f"released the warm models on {card.device} "
                                "to free host RAM for this job; its next job there "
                                "loads cold",
                            }
                        )
        finally:
            self._wake.acquire()

    def _held_for_host_ram(self):
        """The queued jobs held for host RAM now. Called holding _lock."""
        for job_id in self._pending:
            job = self.jobs.get(job_id)
            if job is not None and getattr(job, "wait_kind", None) == "host_ram":
                yield job

    @staticmethod
    def _growth_peak(slot, job):
        """The projected peak to charge a running job's remaining growth
        against, or None once it has run past HOST_RAM_SETTLED_SECONDS (#1357)."""
        if job is None or not job.host_ram_need:
            return None
        started = getattr(slot, "started_at", None)
        if started is not None and time.time() - started > HOST_RAM_SETTLED_SECONDS:
            return None
        return job.host_ram_need[0]

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
        foreign = host.foreign()
        # A warm worker idling on another card holds its models in host RAM
        # and waits on nothing, so it is never "running" and never frees by
        # itself (#1407)
        idle = [
            (other, rss)
            for other in self.slots
            if other is not slot
            and other.current_job_id is None
            and (rss := _worker_rss_mb(other))
        ]
        if not running and not foreign and not idle:
            return None
        available = host.available_mb()
        if available is None:
            return None
        reason = pool.host_ram_wait(
            need,
            available,
            [
                (
                    self._growth_peak(other, job),
                    _worker_rss_mb(other),
                )
                for other, job in running
            ]
            + foreign
            # (None, None) charges nothing: it only lets the gate run with
            # no job running, when an idle worker is all that holds the RAM
            or [(None, None)],
            _worker_rss_mb(slot),
        )
        if reason is None or not idle:
            return reason
        now = time.time()
        mine = self._release_idle.setdefault(job_id, set())
        for other, rss in idle:
            if now - self._released_at.get(id(other), 0) > HOST_RAM_RELEASE_SECONDS:
                mine.add(other)
        gb = sum(rss for _, rss in idle) / 1024
        cards = ", ".join(str(other.device or "a card") for other, _ in idle)
        return (
            f"waiting for host RAM: {reason.split(': ', 1)[-1]}; releasing the "
            f"warm idle worker on {cards} (~{gb:.0f} GB held) to make room"
        )

    # ---------------------------------------------------------------- runner

    def _run_loop(self):
        """The dispatcher: hand each free card the first queued job that
        fits it, and start that job on its own thread."""
        while True:
            with self._wake:
                while True:
                    if self._stop.is_set():
                        return
                    picked = self._next_dispatch()
                    if picked is not None:
                        break
                    if self._release_idle:
                        self._release_idle_workers()
                        continue
                    # Host RAM frees outside dw too, so a job held for it is
                    # looked at again on a timer, not only on a wake (#869)
                    held = any(self._held_for_host_ram())
                    self._wake.wait(HOST_RAM_RECHECK_SECONDS if held else None)
                slot, job = picked
                job.status = RUNNING
                job.started_at = time.time()
                job.device_ordinal, job.device_card = slot.device_fields()
                slot.current_job_id = job.id
                slot.started_at = job.started_at
                # The worker switches to this identity for the run
                slot.last_identity = self._job_identity(job)
            job.add_event({"event": "job_status", "status": RUNNING})
            # Before the job's thread exists, so _finish always writes last
            self._record_open(job)
            self._changed(job)
            thread = threading.Thread(
                target=self._run_job,
                args=(slot, job),
                daemon=True,
                name=f"job-{job.id}",
            )
            with self._lock:
                self._job_threads.add(thread)
            thread.start()

    def _run_job(self, slot, job):
        """A dispatched job's thread: run it, then leave the set shutdown()
        joins."""
        try:
            self._run_on(slot, job)
        finally:
            with self._lock:
                self._job_threads.discard(threading.current_thread())

    def _run_on(self, slot, job):
        """Run one dispatched job on its slot's worker. The dispatcher has
        already marked it RUNNING and the slot busy."""
        manager = slot.manager
        with slot.lock:
            try:
                if job.status != RUNNING or job.cancel_requested:
                    # Cancelled between dispatch and here: nothing was sent,
                    # so the outcome is the cancel itself
                    outcome = (CANCELLED, None, None)
                else:
                    self._start_worker(slot, job)
                    self._rank_for_oom(slot, job)
                    command = Execute(
                        # The snapshot admission checked, which the worker runs
                        # as it is rather than reading the file again
                        definition=job.spec.get("definition"),
                        file_spec=job.spec.get("file_spec"),
                        source=job.spec.get("source"),
                        workflow_dir=job.spec.get("workflow_dir"),
                        # The job's own roots, so a job queued for one workspace
                        # still runs in it after the manager has served another
                        output_dir=job.spec.get("output_dir") or self.output_dir,
                        arguments=job.spec["arguments"],
                        log_level=self.log_level,
                        asset_dir=job.spec.get("asset_dir") or None,
                    )
                    manager.send_command(command.to_wire())
                    if job.cancel_requested:
                        # Asked while the worker was still idle; now it has
                        # the job, and a second cancel lands behind it
                        manager.cancel()
                    outcome = job_results.consume_results(
                        job,
                        manager,
                        slot,
                        job.spec.get("output_dir") or self.output_dir,
                        worker_memory.record_memory,
                        on_run_start=lambda: self._on_run_start(job),
                    )
            except Exception as e:
                logger.error(f"Job {job.id} failed: {e}", exc_info=True)
                outcome = (FAILED, str(e), None)
            finally:
                # Cleared BEFORE the terminal status becomes visible - a
                # client seeing "succeeded" must find the card idle - and
                # the dispatcher woken, since the card is free again
                with self._wake:
                    slot.current_job_id = None
                    slot.started_at = None
                    self._wake.notify_all()
        if job.status not in TERMINAL_STATES:
            status, error, traceback_text = outcome
            self._finish(job, status, error=error, traceback_text=traceback_text)
