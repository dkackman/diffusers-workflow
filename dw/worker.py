"""
Persistent worker process for workflow execution.
Keeps models loaded in GPU memory across multiple runs.
"""

import os
import sys
import json
import hashlib
import logging
import traceback
import gc
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dw.workflow import workflow_from_file
from dw.log_setup import setup_logging

logger = logging.getLogger("dw.worker")


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
        
        # Setup logging
        setup_logging(log_level)
        
        # Workflow state
        self.current_workflow = None
        self.workflow_path = None
        self.workflow_hash = None
        self.output_dir = None
        
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
                    # Wait for command from REPL
                    command = self.command_queue.get()
                    command_type = command.get('type')
                    
                    logger.debug(f"Received command: {command_type}")
                    
                    if command_type == 'execute':
                        self._handle_execute(command)
                    elif command_type == 'shutdown':
                        self._handle_shutdown()
                        break
                    elif command_type == 'ping':
                        self._handle_ping()
                    elif command_type == 'clear_memory':
                        self._handle_clear_memory()
                    elif command_type == 'memory_status':
                        self._handle_memory_status()
                    else:
                        self.result_queue.put({
                            'type': 'error',
                            'message': f"Unknown command type: {command_type}"
                        })
                        
                except KeyboardInterrupt:
                    logger.info("Worker interrupted by keyboard")
                    break
                except Exception as e:
                    logger.error(f"Error processing command: {e}", exc_info=True)
                    self.result_queue.put({
                        'type': 'error',
                        'message': f"Command processing error: {str(e)}",
                        'traceback': traceback.format_exc()
                    })
                    
        finally:
            logger.info("Worker shutting down")
            self._cleanup_all()

    def _handle_execute(self, command: Dict[str, Any]):
        """
        Execute a workflow, reusing loaded models if possible.
        
        Args:
            command: Dictionary with workflow_path, arguments, output_dir, log_level
        """
        workflow_path = command['workflow_path']
        arguments = command['arguments']
        output_dir = command['output_dir']
        log_level = command.get('log_level', 'INFO')
        
        try:
            # Update logging level if changed
            setup_logging(log_level)
            
            # Check if workflow file changed
            current_hash = self._compute_file_hash(workflow_path)
            workflow_changed = (
                current_hash != self.workflow_hash or
                workflow_path != self.workflow_path
            )
            
            if workflow_changed:
                self.result_queue.put({
                    'type': 'output',
                    'message': 'Workflow file changed - reloading models...'
                })
                
                # Cleanup old workflow
                self._cleanup_all()
                
                # Load new workflow
                self.result_queue.put({
                    'type': 'output',
                    'message': f'Loading workflow from {workflow_path}'
                })
                
                self.current_workflow = workflow_from_file(workflow_path, output_dir)
                self.workflow_path = workflow_path
                self.workflow_hash = current_hash
                self.output_dir = output_dir
                
                self.result_queue.put({
                    'type': 'workflow_loaded',
                    'workflow_name': self.current_workflow.name
                })
            else:
                self.result_queue.put({
                    'type': 'output',
                    'message': 'Reusing loaded models from cache'
                })
            
            # Execute workflow with cached pipelines
            self.result_queue.put({
                'type': 'output',
                'message': f'Executing workflow: {self.current_workflow.name}'
            })
            
            self.current_workflow.run(arguments, self.loaded_pipelines)
            
            self.run_count += 1
            
            # Aggressive memory cleanup after execution
            self._cleanup_between_runs()
            
            # Report memory status
            memory_info = self._get_memory_info()
            self.result_queue.put({
                'type': 'memory_info',
                'info': memory_info
            })
            
            self.result_queue.put({
                'type': 'success',
                'message': 'Workflow completed successfully',
                'run_count': self.run_count
            })
            
        except Exception as e:
            logger.error(f"Error executing workflow: {e}", exc_info=True)
            self.result_queue.put({
                'type': 'error',
                'message': f"Workflow execution error: {str(e)}",
                'traceback': traceback.format_exc()
            })

    def _handle_shutdown(self):
        """Handle graceful shutdown request."""
        logger.info("Shutdown requested")
        self._cleanup_all()
        self.result_queue.put({
            'type': 'shutdown_complete'
        })

    def _handle_ping(self):
        """Respond to ping to prove worker is alive."""
        self.result_queue.put({
            'type': 'pong',
            'run_count': self.run_count
        })

    def _handle_clear_memory(self):
        """Handle explicit memory clear request."""
        logger.info("Memory clear requested")
        self._cleanup_all()
        memory_info = self._get_memory_info()
        self.result_queue.put({
            'type': 'memory_cleared',
            'info': memory_info
        })

    def _handle_memory_status(self):
        """Report current memory usage."""
        memory_info = self._get_memory_info()
        self.result_queue.put({
            'type': 'memory_status',
            'info': memory_info
        })

    def _cleanup_between_runs(self):
        """
        Aggressive memory cleanup between workflow runs.
        Keeps models loaded but cleans up intermediate tensors and garbage.
        """
        import gc
        
        logger.debug("Performing inter-run memory cleanup")
        
        # Force garbage collection
        gc.collect()
        
        # Clean up CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Don't synchronize here as it's expensive and unnecessary
        except Exception as e:
            logger.warning(f"Could not clean CUDA cache: {e}")
        
        # Check for memory growth
        current_memory = self._get_gpu_memory_mb()
        if current_memory > 0:
            if self.last_memory_mb > 0:
                growth = current_memory - self.last_memory_mb
                if growth > 500:  # More than 500MB growth
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
        
        logger.info("Performing full cleanup")
        
        # Clear pipeline cache
        self.loaded_pipelines.clear()
        self.shared_components.clear()
        
        # Reset state
        self.current_workflow = None
        self.run_count = 0
        self.last_memory_mb = 0
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Aggressive CUDA cleanup
        try:
            import torch
            if torch.cuda.is_available():
                # Empty cache
                torch.cuda.empty_cache()
                
                # Synchronize to ensure all operations complete
                torch.cuda.synchronize()
                
                # Try to reset memory stats
                try:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()
                except:
                    pass
                    
        except Exception as e:
            logger.warning(f"Could not perform CUDA cleanup: {e}")
        
        logger.info("Full cleanup complete")

    def _compute_file_hash(self, path: str) -> str:
        """
        Compute SHA256 hash of workflow file to detect changes.
        
        Args:
            path: Path to workflow file
            
        Returns:
            Hex digest of file hash
        """
        try:
            with open(path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error computing file hash: {e}")
            return ""

    def _get_gpu_memory_mb(self) -> float:
        """
        Get current GPU memory usage in MB.
        
        Returns:
            Memory usage in MB, or 0 if not available
        """
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
        except:
            pass
        return 0.0

    def _get_memory_info(self) -> Dict[str, Any]:
        """
        Get detailed memory information.
        
        Returns:
            Dictionary with memory statistics
        """
        info = {
            'run_count': self.run_count,
            'gpu_available': False,
            'gpu_memory_allocated_mb': 0.0,
            'gpu_memory_reserved_mb': 0.0,
            'gpu_memory_free_mb': 0.0,
            'gpu_device_name': None
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                info['gpu_available'] = True
                info['gpu_device_name'] = torch.cuda.get_device_name(0)
                info['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
                info['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
                
                # Try to get free memory
                try:
                    free, total = torch.cuda.mem_get_info()
                    info['gpu_memory_free_mb'] = free / 1024 / 1024
                    info['gpu_memory_total_mb'] = total / 1024 / 1024
                except:
                    pass
        except:
            pass
            
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
            result_queue.put({
                'type': 'worker_crashed',
                'message': str(e),
                'traceback': traceback.format_exc()
            })
        except:
            pass
        sys.exit(1)


if __name__ == '__main__':
    # For testing - won't normally be run directly
    import multiprocessing
    
    # Set spawn method for CUDA compatibility
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    
    cmd_queue = multiprocessing.Queue()
    res_queue = multiprocessing.Queue()
    
    print("Starting worker in test mode...")
    worker_main(cmd_queue, res_queue, "DEBUG")
