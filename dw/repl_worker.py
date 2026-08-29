"""
Worker process management for REPL.

Handles starting, stopping, and communicating with the worker process
that keeps models loaded in GPU memory.
"""

import multiprocessing
import queue as queue_module
import logging
from typing import Optional
from .worker import worker_main

logger = logging.getLogger("dw")

# REPL constants
WORKER_SHUTDOWN_TIMEOUT_SECONDS = 10
WORKER_TERMINATE_TIMEOUT_SECONDS = 5

# How often get_result checks worker liveness while waiting for a message
WORKER_LIVENESS_POLL_SECONDS = 1.0


class WorkerManager:
    """Manages the worker process lifecycle and communication."""

    def __init__(self):
        """Initialize worker manager with no active worker."""
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
            self.worker_process.start()
            self.worker_active = True
            logger.info("Worker process started")

    def shutdown_worker(self):
        """Gracefully shutdown worker process."""
        if self.worker_process and self.worker_process.is_alive():
            logger.info("Shutting down worker process...")
            try:
                if self.command_queue:
                    self.command_queue.put({"type": "shutdown"})
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

    def cancel(self):
        """Ask the worker to cancel the workflow it is running."""
        self.send_command({"type": "cancel"})
