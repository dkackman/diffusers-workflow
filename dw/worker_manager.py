"""
Worker process management: the GPU worker's lifecycle for JobManager.

Handles starting, stopping, and communicating with the worker process
that keeps models loaded in GPU memory.
"""

import math
import multiprocessing
import queue as queue_module
import logging
import signal
import sys
import time
from typing import Optional
from . import device_capacity_gb, get_device
from .devices import device_label, pinned_environment, worker_environment
from .worker import worker_main
from .worker_protocol import Cancel, Shutdown, WorkerCrashed, parse_reply

logger = logging.getLogger("dw")

# Worker lifecycle timeouts
WORKER_SHUTDOWN_TIMEOUT_SECONDS = 10
WORKER_TERMINATE_TIMEOUT_SECONDS = 5

# How often get_result checks worker liveness while waiting for a message
WORKER_LIVENESS_POLL_SECONDS = 1.0


class WorkerManager:
    """Manages the worker process lifecycle and communication."""

    def __init__(self, device: Optional[str] = None):
        """Initialize worker manager with no active worker.

        Args:
            device: The card this manager's worker runs on, as the server
                addresses it ('cuda:1'). None is the device dw runs on -
                `get_device()`, read when the worker is spawned. A CUDA
                device named with an index pins the worker to that card
                (dw/devices.py)
        """
        self.device = device
        self._device_label = None
        self._capacity_read = False
        self._capacity_gb = None
        self._ceiling_gb = None
        # What set_oom_score_adj last wrote for the running process
        self.oom_score_adj = None
        self.worker_process: Optional[multiprocessing.Process] = None
        self.command_queue: Optional[multiprocessing.Queue] = None
        self.result_queue: Optional[multiprocessing.Queue] = None
        self.worker_active = False

    def ensure_worker(self, log_level: str = "INFO"):
        """Start worker process if not running.

        Args:
            log_level: Logging level for the worker process
        """
        if self.worker_process is None or not self.worker_process.is_alive():
            logger.info("Starting worker process...")
            self.command_queue = multiprocessing.Queue()
            self.result_queue = multiprocessing.Queue()

            self.worker_process = multiprocessing.Process(
                target=worker_main,
                args=(self.command_queue, self.result_queue, log_level),
            )
            # Pinned in the parent: a spawned child copies os.environ at
            # start(), and dw/__init__.py imports torch before anything in
            # the child could set CUDA_VISIBLE_DEVICES itself
            with pinned_environment(worker_environment(self.device or get_device())):
                self.worker_process.start()
            self.worker_active = True
            self.oom_score_adj = None
            logger.info("Worker process started")

    def device_label(self):
        """This worker's card as a job record names it -
        `"cuda:1 NVIDIA GeForce RTX 3090"` - read once, in the server
        process, which sees every card by its ordinal."""
        if self._device_label is None:
            self._device_label = device_label(self.device)
        return self._device_label

    def shutdown_worker(self):
        """Gracefully shutdown worker process."""
        if self.worker_process and self.worker_process.is_alive():
            logger.info("Shutting down worker process...")
            try:
                if self.command_queue:
                    self.command_queue.put(Shutdown().to_wire())
                self.worker_process.join(timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS)

                if self.worker_process.is_alive():
                    logger.warning("Worker did not shutdown gracefully, terminating...")
                    self.worker_process.terminate()
                    self.worker_process.join(timeout=WORKER_TERMINATE_TIMEOUT_SECONDS)

                    if self.worker_process.is_alive():
                        logger.error("Worker did not terminate, killing...")
                        self.worker_process.kill()

            except Exception as e:
                logger.error(f"Error shutting down worker: {e}")
            finally:
                self.worker_active = False
                self.worker_process = None
                self.command_queue = None
                self.result_queue = None

    def send_command(self, command: dict):
        """Send a command to the worker process.

        Args:
            command: Dictionary containing command type and parameters
        """
        if not self.worker_active or not self.command_queue:
            raise RuntimeError("Worker process is not active")
        self.command_queue.put(command)

    def get_result(self, timeout: Optional[float] = None):
        """Get a result from the worker process.

        With no timeout this waits as long as the worker is alive - a video
        generation or a cold model download takes however long it takes, and
        progress events keep the caller informed in the meantime. The wait
        polls so a worker that dies mid-run raises instead of blocking forever.

        Args:
            timeout: Optional timeout in seconds; None waits indefinitely
                while the worker process is alive

        Returns:
            Result dictionary from worker

        Raises:
            RuntimeError: If worker is not active or dies while waiting
            queue.Empty: If an explicit timeout elapses
        """
        if not self.worker_active or not self.result_queue:
            raise RuntimeError("Worker process is not active")

        if timeout is not None:
            return self.result_queue.get(timeout=timeout)

        while True:
            try:
                return self.result_queue.get(timeout=WORKER_LIVENESS_POLL_SECONDS)
            except queue_module.Empty:
                if self.worker_process is None or not self.worker_process.is_alive():
                    raise RuntimeError("Worker process died while waiting for results")

    def request(self, command, timeout: float):
        """Send a request command and return the worker's typed reply to it.

        Reads until the reply carrying the command's request_id arrives. A
        reply that is not its own - one whose reader gave up before it landed
        - is discarded and logged at DEBUG, so it can never be taken for this
        request's answer. `timeout` bounds the whole wait, however many
        stale replies are read on the way.

        Raises:
            RuntimeError: If the worker is not active, as get_result does
            queue.Empty: If no reply of its own arrives within `timeout`
        """
        self.send_command(command.to_wire())
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue_module.Empty()
            reply = parse_reply(self.get_result(timeout=remaining))
            if getattr(reply, "request_id", None) == command.request_id:
                return reply
            if isinstance(reply, WorkerCrashed):
                # A crash answers every waiting request: nobody else will
                # read it, and the worker is gone
                self.mark_crashed()
                raise RuntimeError(f"Worker crashed: {reply.message}")
            logger.debug(
                "Discarding a worker reply that does not answer %s %s: %s",
                command.TYPE,
                command.request_id,
                type(reply).__name__,
            )

    def cancel(self):
        """Ask the worker to cancel the workflow it is running."""
        self.send_command(Cancel().to_wire())

    def crash_details(self):
        """Why the worker process is gone, as far as the OS will say.

        A worker killed by a signal - the OOM killer's SIGKILL above all -
        never reaches worker_main's except clause, so there is no traceback
        to report and the exit code is the whole diagnosis. Returns None
        while the process is alive or was never started.
        """
        process = self.worker_process
        if process is None or process.is_alive():
            return None
        exitcode = process.exitcode
        if exitcode is None:
            return None
        if exitcode < 0:
            signal_number = -exitcode
            try:
                name = signal.Signals(signal_number).name
            except ValueError:
                name = f"signal {signal_number}"
            detail = f"killed by {name}"
            if signal_number == signal.SIGKILL:
                # By far the likeliest cause on a box that loads models
                # measured in tens of gigabytes, and the one thing the user
                # can act on - no Python-level error will have been logged
                detail += " (typically the out-of-memory killer)"
            return detail
        return f"exited with code {exitcode}"

    def mark_crashed(self):
        """Record that the worker process died on its own - no shutdown
        handshake to attempt, just clear the tracking state."""
        self.worker_active = False
        self.worker_process = None

    def pid(self):
        """The worker process's pid while it is alive, else None."""
        process = self.worker_process
        if process is None or not process.is_alive():
            return None
        return process.pid

    def _read_capacity(self):
        """Measure the card once, in the server process, which sees every
        card by its ordinal: a card does not change size."""
        if not self._capacity_read:
            self._capacity_read = True
            capacity = device_capacity_gb(self.device)
            if capacity:
                self._capacity_gb = math.ceil(capacity)
                self._ceiling_gb = round(capacity, 1)

    def capacity_gb(self):
        """What this worker's card holds, in the GB a catalog `cost` entry's
        `vram_gb` is written in - the card's GiB rounded up, so a 3090
        (23.6 GiB) is the "24" the catalog measured on - or None where the
        card cannot be read."""
        self._read_capacity()
        return self._capacity_gb

    def ceiling_gb(self):
        """The most a declared `vram_estimate` may project on this card - its
        GiB to one decimal (a 3090's 23.6), the figure admission's own
        ceiling holds a projection to on a device no `cost` entry describes
        (dw/vram_estimate.py `_entries_for`), so the two gates agree. Falls
        back to capacity_gb when only that is known."""
        self._read_capacity()
        return self._ceiling_gb if self._ceiling_gb is not None else self._capacity_gb

    def set_oom_score_adj(self, value):
        """Ask the kernel's OOM killer to pick this worker ahead of anything
        scored lower (Linux only; best effort). In a pool the worker started
        later is scored higher, so a host-RAM squeeze kills the newer job
        rather than whichever worker happens to be larger (#462). Raising a
        process's own score needs no privilege. Returns whether it took."""
        pid = self.pid()
        if pid is None or not sys.platform.startswith("linux"):
            return False
        value = max(-1000, min(1000, int(value)))
        try:
            with open(f"/proc/{int(pid)}/oom_score_adj", "w") as file:
                file.write(str(value))
        except OSError as e:
            logger.debug(f"Could not set oom_score_adj for worker {pid}: {e}")
            return False
        self.oom_score_adj = value
        return True
