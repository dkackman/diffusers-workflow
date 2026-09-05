"""
Persistent worker process for workflow execution.
Keeps models loaded in GPU memory across multiple runs.
"""

import os
import sys
import queue
import logging
import threading
import traceback
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dw.workflow import workflow_from_file, workflow_from_definition
from dw.step_cache import step_cache
from dw.assets import activate_asset_dir, deactivate_asset_dir
from dw.log_setup import setup_logging, set_log_level
from dw.settings import load_settings, resolve_path
from dw.security import validate_output_path
from dw.events import RunContext, WorkflowCancelled
from dw import get_device_type, empty_device_cache, device_memory_stats

logger = logging.getLogger("dw.worker")

# Memory management constants
MEMORY_GROWTH_THRESHOLD_MB = 500  # Warn if GPU memory grows by more than this

# How often (in seconds) the main loop wakes up to check whether the parent
# process is still alive when no command has arrived. Short enough that an
# orphaned worker exits promptly, long enough to avoid busy-waiting.
COMMAND_POLL_TIMEOUT_SECONDS = 5


class WorkflowWorker:
    """
    Persistent worker that keeps workflows and models loaded in memory.
    Monitors workflow file for changes and reloads when necessary.
    """

    def __init__(self, command_queue, result_queue, log_level="INFO"):
        """
        Initialize the worker with communication queues.

        Args:
            command_queue: Queue for receiving commands from REPL
            result_queue: Queue for sending results back to REPL
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.command_queue = command_queue
        self.result_queue = result_queue

        # Capture the parent PID at startup so the main loop can detect
        # orphaning (parent died/was killed without sending "shutdown") and
        # exit cleanly instead of blocking forever on the command queue.
        self.parent_pid = os.getppid()

        # Log to the same file the rest of dw uses - the worker is a separate
        # process, and ConcurrentRotatingFileHandler exists to share it safely
        settings = load_settings()
        setup_logging(resolve_path(settings.log_filename), log_level)

        # Workflow state. Identity (path, or id for inline definitions) decides
        # when the model cache is dropped wholesale - switching workflows frees
        # the old one's models before the new one loads
        self.workflow_identity = None
        self.pending_shutdown = False

        # Pipeline cache - persists across runs
        self.loaded_pipelines = {}
        self.shared_components = {}

        # Memory tracking
        self.run_count = 0
        self.last_memory_mb = 0

        logger.info("Worker process initialized")

    def run(self):
        """
        Main worker loop - processes commands until shutdown.
        """
        logger.info("Worker entering command loop")

        try:
            while True:
                try:
                    # Wait for a command from the REPL, but poll with a
                    # timeout rather than blocking forever. If nothing
                    # arrives, check whether the parent process is still
                    # alive - if it has died (e.g. crashed or was killed)
                    # without sending "shutdown", we'd otherwise sit here
                    # forever as an orphaned, unkillable-by-normal-means
                    # child. Exit cleanly instead.
                    try:
                        command = self.command_queue.get(
                            timeout=COMMAND_POLL_TIMEOUT_SECONDS
                        )
                    except queue.Empty:
                        if self._parent_is_dead():
                            logger.info(
                                f"Parent process (pid {self.parent_pid}) is gone - "
                                "worker exiting"
                            )
                            break
                        continue

                    command_type = command.get("type")

                    logger.debug(f"Received command: {command_type}")

                    if command_type == "execute":
                        self._handle_execute(command)
                        if self.pending_shutdown:
                            self._handle_shutdown()
                            break
                    elif command_type == "cancel":
                        # Nothing running - a cancel that raced the run's end
                        logger.debug("Ignoring cancel with no workflow running")
                    elif command_type == "shutdown":
                        self._handle_shutdown()
                        break
                    elif command_type == "ping":
                        self._handle_ping()
                    elif command_type == "clear_memory":
                        self._handle_clear_memory()
                    elif command_type == "memory_status":
                        self._handle_memory_status()
                    else:
                        self.result_queue.put(
                            {
                                "type": "error",
                                "message": f"Unknown command type: {command_type}",
                            }
                        )

                except KeyboardInterrupt:
                    logger.info("Worker interrupted by keyboard")
                    break
                except Exception as e:
                    logger.error(f"Error processing command: {e}", exc_info=True)
                    self.result_queue.put(
                        {
                            "type": "error",
                            "message": f"Command processing error: {str(e)}",
                            "traceback": traceback.format_exc(),
                        }
                    )

        finally:
            logger.info("Worker shutting down")
            self._cleanup_all()

    def _handle_execute(self, command: Dict[str, Any]):
        """
        Execute a workflow, reusing loaded models if possible.

        The command names the workflow either by path (workflow_path) or as an
        inline definition (workflow, with an optional base_dir that relative
        paths inside it resolve against). Models stay cached between runs of
        the same workflow identity; pipelines are cached by what they load, so
        an edited workflow keeps every pipeline whose definition is unchanged.
        A {"type": "cancel"} command sent during execution stops the run at
        the next step boundary or diffusion step.

        Args:
            command: Dictionary with workflow_path or workflow (+ base_dir),
                arguments, output_dir, log_level, and optionally asset_dir -
                the workspace library 'asset:' resolves against
        """
        arguments = command["arguments"]
        output_dir = command["output_dir"]
        log_level = command.get("log_level", "INFO")

        try:
            set_log_level(log_level)

            workflow, identity = self._load_workflow(command, output_dir)
            workflow.validate()

            # Switching to a different workflow frees the old one's models
            # before the new one loads - on one accelerator, holding both is
            # what runs out of memory
            if identity != self.workflow_identity:
                if self.workflow_identity is not None:
                    self.result_queue.put(
                        {
                            "type": "output",
                            "message": "Workflow changed - releasing cached models...",
                        }
                    )
                    self._cleanup_all()
                self.workflow_identity = identity

            self.result_queue.put(
                {"type": "workflow_loaded", "workflow_name": workflow.name}
            )
            self.result_queue.put(
                {
                    "type": "output",
                    "message": f"Executing workflow: {workflow.name}",
                }
            )

            # Progress events stream to the client as they happen; the watcher
            # thread keeps the command queue live so cancel works mid-run
            context = RunContext(
                on_event=lambda event: self.result_queue.put(
                    {"type": "progress", **event}
                )
            )
            watcher = self._watch_commands(context)
            # Which workspace's assets this job's 'asset:' references resolve
            # against. A server holds several workspaces and each has its own
            # library, so the root travels with the job rather than being
            # pinned in the environment the way the shared prompt library is
            asset_token = (
                activate_asset_dir(command["asset_dir"])
                if command.get("asset_dir")
                else None
            )
            try:
                workflow.run(arguments, self.loaded_pipelines, context=context)
            finally:
                watcher.stop()
                if asset_token is not None:
                    deactivate_asset_dir(asset_token)

            # Drop cached pipelines this run no longer touched - an edited
            # workflow that removed or redefined a step leaves those behind
            for cache_key in list(self.loaded_pipelines):
                if cache_key not in context.touched_pipelines:
                    logger.info("Evicting cached pipeline no longer in workflow")
                    del self.loaded_pipelines[cache_key]

            self.run_count += 1

            # Aggressive memory cleanup after execution
            self._cleanup_between_runs()

            # Report memory status
            memory_info = self._get_memory_info()
            self.result_queue.put({"type": "memory_info", "info": memory_info})

            self.result_queue.put(
                {
                    "type": "success",
                    "message": "Workflow completed successfully",
                    "run_count": self.run_count,
                    "manifest": getattr(workflow, "manifest", []),
                }
            )

        except WorkflowCancelled:
            self._cleanup_between_runs()
            self.result_queue.put(
                {"type": "cancelled", "message": "Workflow run cancelled"}
            )
        except Exception as e:
            logger.error(f"Error executing workflow: {e}", exc_info=True)
            self.result_queue.put(
                {
                    "type": "error",
                    "message": f"Workflow execution error: {str(e)}",
                    "traceback": traceback.format_exc(),
                }
            )

    def _load_workflow(self, command: Dict[str, Any], output_dir: str):
        """Build the Workflow a command names, and its cache identity."""
        workflow_dir = command.get("workflow_dir")
        if "workflow_path" in command and command["workflow_path"] is not None:
            workflow_path = command["workflow_path"]
            workflow = workflow_from_file(workflow_path, output_dir, workflow_dir)
            return workflow, ("path", workflow_path)

        workflow_data = command["workflow"]
        workflow = workflow_from_definition(
            workflow_data, output_dir, command.get("base_dir"), workflow_dir
        )
        return workflow, ("inline", workflow_data.get("id"))

    def _watch_commands(self, context):
        """Watch the command queue during a run so cancel and ping still work.

        Returns an object with stop(); anything that is not cancel, ping or
        shutdown is refused, since one workflow runs at a time.
        """
        stop_event = threading.Event()
        worker = self

        def watch():
            while not stop_event.is_set():
                try:
                    command = worker.command_queue.get(timeout=0.25)
                except queue.Empty:
                    continue
                command_type = command.get("type")
                if command_type == "cancel":
                    logger.info("Cancel requested")
                    context.cancel()
                    worker.result_queue.put(
                        {"type": "output", "message": "Cancelling..."}
                    )
                elif command_type == "ping":
                    worker._handle_ping()
                elif command_type == "shutdown":
                    # Stop the run, then let the main loop see the shutdown
                    context.cancel()
                    worker.pending_shutdown = True
                else:
                    worker.result_queue.put(
                        {
                            "type": "error",
                            "message": f"Cannot handle '{command_type}' while a "
                            "workflow is running",
                        }
                    )

        thread = threading.Thread(target=watch, daemon=True, name="command-watcher")
        thread.start()

        class _Watcher:
            def stop(self):
                stop_event.set()
                thread.join()

        return _Watcher()

    def _handle_shutdown(self):
        """Handle graceful shutdown request."""
        logger.info("Shutdown requested")
        self._cleanup_all()
        self.result_queue.put({"type": "shutdown_complete"})

    def _handle_ping(self):
        """Respond to ping to prove worker is alive."""
        self.result_queue.put({"type": "pong", "run_count": self.run_count})

    def _handle_clear_memory(self):
        """Handle explicit memory clear request."""
        logger.info("Memory clear requested")
        self._cleanup_all()
        memory_info = self._get_memory_info()
        self.result_queue.put({"type": "memory_cleared", "info": memory_info})

    def _handle_memory_status(self):
        """Report current memory usage."""
        memory_info = self._get_memory_info()
        self.result_queue.put({"type": "memory_status", "info": memory_info})

    def _cleanup_between_runs(self):
        """
        Aggressive memory cleanup between workflow runs.
        Keeps models loaded but cleans up intermediate tensors and garbage.
        """
        import gc

        logger.debug("Performing inter-run memory cleanup")

        # Force garbage collection
        gc.collect()

        # Clean up GPU cache if available (CUDA or MPS). Don't synchronize
        # here as it's expensive and unnecessary.
        try:
            empty_device_cache()
        except Exception as e:
            logger.warning(f"Could not clean GPU cache: {e}")

        # Check for memory growth
        current_memory = self._get_gpu_memory_mb()
        if current_memory > 0:
            if self.last_memory_mb > 0:
                growth = current_memory - self.last_memory_mb
                if growth > MEMORY_GROWTH_THRESHOLD_MB:
                    logger.warning(
                        f"GPU memory grew by {growth:.1f}MB "
                        f"({self.last_memory_mb:.1f}MB -> {current_memory:.1f}MB)"
                    )
            self.last_memory_mb = current_memory

        logger.debug("Inter-run cleanup complete")

    def _cleanup_all(self):
        """
        Complete cleanup - clear all cached models and components.
        Called when workflow changes or on shutdown.
        """
        import gc
        from .tasks.model_cache import clear_model_cache

        logger.info("Performing full cleanup")

        # Clear pipeline cache and any models task handlers cached
        self.loaded_pipelines.clear()
        self.shared_components.clear()
        clear_model_cache()
        # Drop cached step results too - stale results would otherwise
        # survive a memory clear and keep getting served for steps whose
        # models/components were just evicted
        step_cache.clear()

        # Reset state
        self.run_count = 0
        self.last_memory_mb = 0

        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()

        # Aggressive GPU cleanup (CUDA or MPS) - empty cache and synchronize
        # to ensure all operations complete before we go on to reset stats.
        try:
            empty_device_cache(synchronize=True)

            # Try to reset CUDA memory stats - no MPS equivalent exists
            if get_device_type() == "cuda":
                import torch

                try:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()
                except (RuntimeError, AttributeError) as e:
                    logger.debug(f"Could not reset memory stats: {e}")

        except Exception as e:
            logger.warning(f"Could not perform GPU cleanup: {e}")

        logger.info("Full cleanup complete")

    def _parent_is_dead(self) -> bool:
        """
        Check whether the process that spawned this worker is still around.

        On POSIX, a process gets reparented to init (traditionally pid 1,
        though some systems use a subreaper) once its original parent exits,
        so a changed getppid() is the standard signal that we've been
        orphaned.

        Returns:
            True if the parent appears to be gone, False otherwise.
        """
        current_ppid = os.getppid()
        return current_ppid != self.parent_pid or current_ppid == 1

    def _get_gpu_memory_mb(self) -> float:
        """
        Get current GPU memory usage in MB.

        Returns:
            Memory usage in MB, or 0 if not available
        """
        try:
            return device_memory_stats()["allocated_mb"]
        except (RuntimeError, AttributeError) as e:
            logger.debug(f"Could not get GPU memory: {e}")
        return 0.0

    def _get_memory_info(self) -> Dict[str, Any]:
        """
        Get detailed memory information.

        Returns:
            Dictionary with memory statistics
        """
        info = {
            "run_count": self.run_count,
            "gpu_available": False,
            "gpu_memory_allocated_mb": 0.0,
            "gpu_memory_reserved_mb": 0.0,
            "gpu_memory_free_mb": 0.0,
            "gpu_device_name": None,
        }

        try:
            stats = device_memory_stats()
            info["gpu_available"] = stats["available"]
            info["gpu_device_name"] = stats["device_name"]
            info["gpu_memory_allocated_mb"] = stats["allocated_mb"]
            info["gpu_memory_reserved_mb"] = stats["reserved_mb"]
            # free/total are only ever None when CUDA's mem_get_info call
            # itself failed - leave gpu_memory_free_mb at its 0.0 default and
            # gpu_memory_total_mb unset in that case, same as before.
            if stats["free_mb"] is not None:
                info["gpu_memory_free_mb"] = stats["free_mb"]
            if stats["total_mb"] is not None:
                info["gpu_memory_total_mb"] = stats["total_mb"]
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.debug(f"Could not access GPU: {e}")

        return info


def worker_main(command_queue, result_queue, log_level="INFO"):
    """
    Entry point for worker process.

    Args:
        command_queue: Queue for receiving commands
        result_queue: Queue for sending results
        log_level: Logging level
    """
    try:
        worker = WorkflowWorker(command_queue, result_queue, log_level)
        worker.run()
    except Exception as e:
        logger.error(f"Worker crashed: {e}", exc_info=True)
        try:
            result_queue.put(
                {
                    "type": "worker_crashed",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
            )
        except (OSError, RuntimeError) as queue_error:
            logger.error(f"Failed to send crash notification to queue: {queue_error}")
        sys.exit(1)


if __name__ == "__main__":
    # For testing - won't normally be run directly
    import multiprocessing

    # Set spawn method for CUDA compatibility
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    cmd_queue = multiprocessing.Queue()
    res_queue = multiprocessing.Queue()

    print("Starting worker in test mode...")
    worker_main(cmd_queue, res_queue, "DEBUG")
