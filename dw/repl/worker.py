"""
Worker process management for REPL.

Handles starting, stopping, and communicating with the worker process
that keeps models loaded in GPU memory.
"""

import multiprocessing
import logging
from typing import Optional
from ..worker import worker_main

logger = logging.getLogger("dw")

# REPL constants
WORKER_RESULT_TIMEOUT_SECONDS = 300  # 5 minutes
WORKER_SHUTDOWN_TIMEOUT_SECONDS = 10
WORKER_TERMINATE_TIMEOUT_SECONDS = 5


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

    def get_result(self, timeout: float = WORKER_RESULT_TIMEOUT_SECONDS):
        """Get a result from the worker process.

        Args:
            timeout: Timeout in seconds for waiting for result

        Returns:
            Result dictionary from worker

        Raises:
            RuntimeError: If worker is not active
        """
        if not self.worker_active or not self.result_queue:
            raise RuntimeError("Worker process is not active")
        return self.result_queue.get(timeout=timeout)

