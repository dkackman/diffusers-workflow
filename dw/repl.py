import cmd
import sys
import argparse
import logging
import os
import multiprocessing
from . import startup
from .workflow import workflow_from_file
from .worker import worker_main
from .security import (
    validate_path,
    validate_workflow_path,
    validate_output_path,
    validate_variable_name,
    validate_string_input,
    sanitize_command_args,
    SecurityError,
    PathTraversalError,
    InvalidInputError,
)
import torch

# CRITICAL: Set multiprocessing start method to 'spawn' for CUDA compatibility
# Must be done before any multiprocessing operations
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set, ignore
        pass

logger = logging.getLogger("dw")


class DiffusersWorkflowREPL(cmd.Cmd):
    """Interactive command line interface for Diffusers Workflow"""

    intro = "Welcome to Diffusers Workflow REPL. Type help or ? to list commands.\n"
    prompt = "dw> "
    use_rawinput = True  # Ensure we're using raw_input for command reading

    def __init__(self):
        # Initialize cmd.Cmd first, before setting up our globals
        cmd.Cmd.__init__(self)
        # Initialize globals dictionary with default values
        self.globals = {
            "output_dir": "./outputs",  # Default output directory
            "log_level": "INFO",  # Default log level
            "workflow_dir": "./examples",  # Default workflow directory
        }
        self.current_workflow = None
        self.workflow_args = {}  # Store workflow arguments

        # Worker process management
        self.worker_process = None
        self.command_queue = None
        self.result_queue = None
        self.worker_active = False

    def preloop(self):
        """Hook method executed once when cmdloop() is called."""
        try:
            import readline

            history_file = os.path.expanduser("~/.dw_history")
            readline.read_history_file(history_file)
        except (ImportError, FileNotFoundError):
            pass

    def postloop(self):
        """Hook method executed once when cmdloop() is about to return."""
        try:
            import readline

            history_file = os.path.expanduser("~/.dw_history")
            readline.write_history_file(history_file)
        except (ImportError, FileNotFoundError):
            pass

    def emptyline(self):
        """Override emptyline to do nothing instead of repeating last command."""
        pass

    def do_help(self, arg):
        """List available commands with "help" or detailed help with "help cmd"."""
        if not arg:
            print("\nDiffusers Workflow REPL - Available Commands")
            print("=" * 60)
            print("\nCommand Groups (use '<command> ?' for subcommands):")
            print("  workflow  - Load and manage workflows")
            print("  arg       - Manage workflow arguments")
            print("  model     - Control model loading and execution")
            print("  memory    - Monitor and manage GPU memory")
            print("  config    - Configure global settings")
            print("\nOther Commands:")
            print("  help      - Show this help message")
            print("  exit      - Exit the REPL")
            print("  quit      - Exit the REPL")
            print("\nExamples:")
            print("  workflow load FluxDev")
            print('  arg set prompt="a cat"')
            print("  model run")
            print("  memory show")
            print("  config set output_dir=./outputs")
            print()
        else:
            super().do_help(arg)

    def do_exit(self, arg):
        """Exit the REPL"""
        self._shutdown_worker()
        print("Goodbye!")
        return True

    def do_quit(self, arg):
        """Exit the REPL (alias for exit)"""
        return self.do_exit(arg)

    # ========================================================================
    # CONFIG COMMANDS - Global configuration settings
    # ========================================================================

    def do_config(self, arg):
        """Configure global settings. Usage: config ? | show | set <name>=<value>"""
        if not arg or arg == "?":
            print("\nConfig commands:")
            print("  config show              - Show all configuration settings")
            print("  config set <name>=<value> - Set a configuration value")
            print("\nAvailable settings:")
            print("  output_dir   - Directory for output files (default: ./outputs)")
            print(
                "  log_level    - Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )
            print(
                "  workflow_dir - Default directory for workflows (default: ./examples)"
            )
            print()
            return

        parts = arg.split(None, 1)
        subcommand = parts[0]
        subarg = parts[1] if len(parts) > 1 else ""

        if subcommand == "show":
            self._config_show(subarg)
        elif subcommand == "set":
            self._config_set(subarg)
        else:
            print(f"Unknown config subcommand: {subcommand}")
            print("Use 'config ?' for help")

    def _config_show(self, arg):
        """Show configuration settings"""
        print("\nCurrent configuration:")
        for name, value in self.globals.items():
            print(f"  {name}={value}")
        print()

    def _config_set(self, arg):
        """Set a configuration value"""
        if not arg:
            # If no argument, show all config (backward compatibility with 'set')
            self._config_show(arg)
            return

        try:
            name, value = arg.split("=", 1)
            name = name.strip()
            value = value.strip()

            # Special handling for output_dir
            if name == "output_dir":
                try:
                    value = validate_output_path(value, None)
                    # Check if directory exists
                    if not os.path.exists(value):
                        print(f"Warning: Directory '{value}' does not exist")
                except SecurityError as e:
                    print(f"Error: Invalid output directory: {e}")
                    return

            # Special handling for log_level
            elif name == "log_level":
                valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                value = value.upper()
                if value not in valid_levels:
                    print(f"Error: Log level must be one of: {', '.join(valid_levels)}")
                    return
                # Update the log level
                logging.getLogger().setLevel(value)
                print(f"Log level set to {value}")

            elif name == "workflow_dir":
                try:
                    value = validate_path(value, allow_create=False)
                    if not os.path.exists(value):
                        print(f"Warning: Directory '{value}' does not exist")
                        return
                except SecurityError as e:
                    print(f"Error: Invalid workflow directory: {e}")
                    return
            else:
                print(f"Warning: Unknown setting '{name}'")

            self.globals[name] = value
            print(f"Set {name}={value}")

        except ValueError:
            print("Error: Invalid format. Use: config set name=value")

    # ========================================================================
    # ARG COMMANDS - Workflow argument management
    # ========================================================================

    def do_arg(self, arg):
        """Manage workflow arguments. Usage: arg ? | show | set <name>=<value> | clear"""
        if arg == "?":
            print("\nArg commands:")
            print("  arg show              - Show available and current arguments")
            print("  arg set <name>=<value> - Set an argument value")
            print("  arg clear             - Clear all argument values")
            print()
            return

        if not arg:
            # If no argument, show args (backward compatibility)
            self._arg_show(arg)
            return

        parts = arg.split(None, 1)
        subcommand = parts[0]
        subarg = parts[1] if len(parts) > 1 else ""

        if subcommand == "show":
            self._arg_show(subarg)
        elif subcommand == "set":
            self._arg_set(subarg)
        elif subcommand == "clear":
            self._arg_clear(subarg)
        else:
            # Try to parse as set command for backward compatibility
            self._arg_set(arg)

    def _arg_show(self, arg):
        """Show workflow arguments"""
        if not self.current_workflow:
            print("Error: No workflow loaded. Use 'workflow load' command first")
            return

        print("\nAvailable variables in workflow and their default values:")
        workflow_vars = self.current_workflow.variables
        if not workflow_vars:
            print("  No variables defined in workflow")
        else:
            for var_name, var_def in workflow_vars.items():
                print(f"  {var_name}: {var_def}")

        print("\nCurrent argument values:")
        if not self.workflow_args:
            print("  No arguments set")
        else:
            for name, value in self.workflow_args.items():
                print(f"  {name}={value}")
        print()

    def _arg_set(self, arg):
        """Set a workflow argument"""
        if not self.current_workflow:
            print("Error: No workflow loaded. Use 'workflow load' command first")
            return

        if not arg:
            print("Error: Please specify argument name and value")
            print("Usage: arg set <name>=<value>")
            return

        try:
            name, value = arg.split("=", 1)
            name = name.strip()
            value = value.strip()

            # Validate variable name
            try:
                name = validate_variable_name(name)
                value = validate_string_input(value, max_length=10000, allow_empty=True)
            except InvalidInputError as e:
                print(f"Error: Invalid input: {e}")
                return

            # Verify this is a valid variable name for the workflow
            if name not in self.current_workflow.variables:
                print(f"Error: '{name}' is not defined in workflow variables")
                return

            self.workflow_args[name] = value
            print(f"Set argument {name}={value}")

        except ValueError:
            print("Error: Invalid format. Use: arg set name=value")

    def _arg_clear(self, arg):
        """Clear all workflow arguments"""
        self.workflow_args = {}
        print("All workflow arguments cleared")

    # ========================================================================
    # MEMORY COMMANDS - GPU memory management
    # ========================================================================

    def do_memory(self, arg):
        """Manage GPU memory. Usage: memory ? | show | clear"""
        if not arg or arg == "?":
            print("\nMemory commands:")
            print("  memory show  - Show current GPU memory usage")
            print("  memory clear - Clear GPU memory and cached models")
            print()
            return

        parts = arg.split(None, 1)
        subcommand = parts[0]
        subarg = parts[1] if len(parts) > 1 else ""

        if subcommand == "show":
            self._memory_show(subarg)
        elif subcommand == "clear":
            self._memory_clear(subarg)
        else:
            print(f"Unknown memory subcommand: {subcommand}")
            print("Use 'memory ?' for help")

    def _memory_show(self, arg):
        """Show current GPU memory usage"""
        if not self.worker_active:
            print("No worker process running")
            return

        try:
            self.command_queue.put({"type": "memory_status"})
            result = self.result_queue.get(timeout=5)

            if result["type"] == "memory_status":
                self._print_memory_info(result.get("info", {}))
            else:
                print(f"Unexpected response: {result}")
        except Exception as e:
            print(f"Error getting memory status: {e}")

    def _memory_clear(self, arg):
        """Clear GPU memory and cached models"""
        if not self.worker_active:
            print("No worker process running")
            return

        try:
            print("Clearing GPU memory...")
            self.command_queue.put({"type": "clear_memory"})

            # Wait for response
            result = self.result_queue.get(timeout=30)
            if result["type"] == "memory_cleared":
                self._print_memory_info(result.get("info", {}))
                print("GPU memory cleared successfully")
            else:
                print(f"Unexpected response: {result}")
        except Exception as e:
            print(f"Error clearing memory: {e}")
            self._shutdown_worker()

    # ========================================================================
    # WORKFLOW COMMANDS - Workflow management
    # ========================================================================

    def do_workflow(self, arg):
        """Manage workflows. Usage: workflow ? | load <file> | reload | status"""
        if not arg or arg == "?":
            print("\nWorkflow commands:")
            print("  workflow load <file> - Load a workflow from a JSON file")
            print("  workflow reload      - Reload the current workflow from disk")
            print("  workflow status      - Show current workflow information")
            print()
            return

        parts = arg.split(None, 1)
        subcommand = parts[0]
        subarg = parts[1] if len(parts) > 1 else ""

        if subcommand == "load":
            self._workflow_load(subarg)
        elif subcommand == "reload":
            self._workflow_reload(subarg)
        elif subcommand == "status":
            self._workflow_status(subarg)
        else:
            print(f"Unknown workflow subcommand: {subcommand}")
            print("Use 'workflow ?' for help")

    def _workflow_load(self, arg):
        """Load a workflow from a JSON file"""
        if not arg:
            print("Error: Please specify a workflow file path or name")
            return

        try:
            file_path = validate_string_input(arg.strip(), max_length=1000)
        except InvalidInputError as e:
            print(f"Error: Invalid file path: {e}")
            return

        # If this isn't an absolute path or relative path starting with ./ or ../
        if not os.path.isabs(file_path) and not file_path.startswith(("./", "../")):
            # Treat as a workflow name in the default directory
            # Add .json extension if not present
            if not file_path.endswith(".json"):
                file_path = f"{file_path}.json"
            try:
                file_path = validate_path(
                    os.path.join(self.globals["workflow_dir"], file_path),
                    self.globals["workflow_dir"],
                    allow_create=False,
                )
            except SecurityError as e:
                print(f"Error: Invalid workflow path: {e}")
                return
        else:
            try:
                file_path = validate_workflow_path(
                    file_path, self.globals["workflow_dir"]
                )
            except SecurityError as e:
                print(f"Error: Invalid workflow path: {e}")
                return

        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            return

        try:
            output_dir = self.globals["output_dir"]
            if not os.path.exists(output_dir):
                print(f"Warning: Output directory {output_dir} does not exist")

            workflow = workflow_from_file(file_path, output_dir)

            # Try to validate the workflow immediately
            try:
                workflow.validate()

                # If workflow changed, shutdown worker (will restart on next run)
                old_path = (
                    self.current_workflow.file_spec if self.current_workflow else None
                )
                if old_path != file_path:
                    self._shutdown_worker()

                self.current_workflow = workflow
                # Clear any existing arguments when loading new workflow
                self.workflow_args = {}
                print(f"Loaded workflow: {workflow.name}")
                print("Workflow validated successfully")

            except Exception as e:
                print(f"Warning: Workflow validation failed: {str(e)}")

        except Exception as e:
            print(f"Error loading workflow: {str(e)}")
            self.current_workflow = None

    def _workflow_reload(self, arg):
        """Reload the current workflow from its file"""
        if not self.current_workflow:
            print("Error: No workflow loaded. Use 'workflow load' command first")
            return

        try:
            file_path = self.current_workflow.file_spec
            print(f"Reloading workflow from: {file_path}")

            # Load and validate the workflow
            workflow = workflow_from_file(file_path, self.globals["output_dir"])
            workflow.validate()

            # Replace current workflow
            self.current_workflow = workflow
            print(f"Reloaded workflow: {workflow.name}")
            print("Workflow validated successfully")

        except Exception as e:
            print(f"Error reloading workflow: {str(e)}")

    def _workflow_status(self, arg):
        """Show current workflow status"""
        if self.current_workflow is None:
            print("No workflow currently loaded")
        else:
            print(f"\nCurrent workflow: {self.current_workflow.name}")
            print(f"File: {self.current_workflow.file_spec}")
            print()

    # ========================================================================
    # MODEL COMMANDS - Model execution and control
    # ========================================================================

    def do_model(self, arg):
        """Control model execution. Usage: model ? | run | restart"""
        if not arg or arg == "?":
            print("\nModel commands:")
            print("  model run     - Execute the currently loaded workflow")
            print("  model restart - Restart the worker process (clears cache)")
            print()
            return

        parts = arg.split(None, 1)
        subcommand = parts[0]
        subarg = parts[1] if len(parts) > 1 else ""

        if subcommand == "run":
            self._model_run(subarg)
        elif subcommand == "restart":
            self._model_restart(subarg)
        else:
            print(f"Unknown model subcommand: {subcommand}")
            print("Use 'model ?' for help")

    def _model_run(self, arg):
        """Run the currently loaded workflow with set arguments"""
        if not self.current_workflow:
            print("Error: No workflow loaded. Use 'workflow load' command first")
            return

        try:
            # Validate inputs
            output_dir = validate_output_path(self.globals["output_dir"], None)
            workflow_spec = validate_workflow_path(self.current_workflow.file_spec)

            # Ensure worker is running
            self._ensure_worker()

            print(f"Running workflow: {self.current_workflow.name}")
            if self.workflow_args:
                print(f"Using arguments: {self.workflow_args}")

            # Send execute command to worker
            self.command_queue.put(
                {
                    "type": "execute",
                    "workflow_path": workflow_spec,
                    "arguments": self.workflow_args,
                    "output_dir": output_dir,
                    "log_level": self.globals["log_level"],
                }
            )

            # Process results from worker
            while True:
                try:
                    result = self.result_queue.get(timeout=300)  # 5 minute timeout
                    result_type = result.get("type")

                    if result_type == "output":
                        print(result["message"])
                    elif result_type == "workflow_loaded":
                        print(f"Models loaded for workflow: {result['workflow_name']}")
                    elif result_type == "memory_info":
                        self._print_memory_info(result["info"])
                    elif result_type == "success":
                        print(result["message"])
                        run_count = result.get("run_count", 0)
                        print(
                            f"(Workflow has been executed {run_count} time(s) in this session)"
                        )
                        break
                    elif result_type == "error":
                        print("\n" + "=" * 80)
                        print(f"ERROR: {result['message']}")
                        if "traceback" in result:
                            print("\nTraceback:")
                            print(result["traceback"])
                        print("=" * 80)
                        print("Worker process encountered an error and stopped.\n")
                        break
                    elif result_type == "worker_crashed":
                        print("\n" + "=" * 80)
                        print(f"WORKER CRASHED: {result['message']}")
                        if "traceback" in result:
                            print("\nTraceback:")
                            print(result["traceback"])
                        print("=" * 80)
                        print(
                            "Worker process has terminated. Use 'model restart' to start a new worker.\n"
                        )
                        self.worker_active = False
                        break
                    else:
                        print(f"Unknown result type: {result_type}")

                except Exception as e:
                    print("\n" + "=" * 80)
                    print(f"ERROR receiving results: {e}")
                    print("=" * 80)
                    print("Worker communication failed. Shutting down worker.\n")
                    self._shutdown_worker()
                    break

        except SecurityError as e:
            print("\n" + "=" * 80)
            print(f"SECURITY ERROR: {e}")
            print("=" * 80 + "\n")
        except Exception as e:
            print("\n" + "=" * 80)
            print(f"ERROR running workflow: {str(e)}")
            print("=" * 80)
            print("Shutting down worker.\n")
            self._shutdown_worker()

    def _model_restart(self, arg):
        """Restart the worker process"""
        print("Restarting worker process...")
        self._shutdown_worker()
        print("Worker shutdown complete")
        print("Worker will restart on next run")

    def default(self, line):
        """Handle unknown commands"""
        print(f"Unknown command: {line}")
        print("Type 'help' or '?' for a list of commands")

    def _ensure_worker(self):
        """Start worker process if not running"""
        if self.worker_process is None or not self.worker_process.is_alive():
            print("Starting worker process...")
            self.command_queue = multiprocessing.Queue()
            self.result_queue = multiprocessing.Queue()

            self.worker_process = multiprocessing.Process(
                target=worker_main,
                args=(self.command_queue, self.result_queue, self.globals["log_level"]),
            )
            self.worker_process.start()
            self.worker_active = True
            print("Worker process started")

    def _shutdown_worker(self):
        """Gracefully shutdown worker process"""
        if self.worker_process and self.worker_process.is_alive():
            print("Shutting down worker process...")
            try:
                self.command_queue.put({"type": "shutdown"})
                self.worker_process.join(timeout=10)

                if self.worker_process.is_alive():
                    print("Worker did not shutdown gracefully, terminating...")
                    self.worker_process.terminate()
                    self.worker_process.join(timeout=5)

                    if self.worker_process.is_alive():
                        print("Worker did not terminate, killing...")
                        self.worker_process.kill()

            except Exception as e:
                print(f"Error shutting down worker: {e}")
            finally:
                self.worker_active = False
                self.worker_process = None

    def _print_memory_info(self, info):
        """Print formatted memory information"""
        if not info.get("gpu_available"):
            print("GPU not available")
            return

        print(f"\nGPU Memory Status:")
        print(f"  Device: {info.get('gpu_device_name', 'Unknown')}")
        print(f"  Allocated: {info.get('gpu_memory_allocated_mb', 0):.1f} MB")
        print(f"  Reserved: {info.get('gpu_memory_reserved_mb', 0):.1f} MB")

        if "gpu_memory_free_mb" in info:
            print(f"  Free: {info.get('gpu_memory_free_mb', 0):.1f} MB")
            print(f"  Total: {info.get('gpu_memory_total_mb', 0):.1f} MB")

        print(f"  Runs in this session: {info.get('run_count', 0)}")
        print()


def main():
    """Start the REPL interface"""
    parser = argparse.ArgumentParser(description="Start Diffusers Workflow REPL.")
    parser.add_argument(
        "-l",
        "--log_level",
        type=str,
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    args = parser.parse_args()

    # Initialize logging
    startup(args.log_level)

    try:
        repl = DiffusersWorkflowREPL()
        repl.cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in REPL: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
