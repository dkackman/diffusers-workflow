"""
Interactive REPL for Diffusers Workflow.

Main entry point for the REPL interface. Delegates command handling
to specialized command classes and worker management to WorkerManager.
"""

import cmd
import sys
import argparse
import logging
import os
import multiprocessing
from . import startup
from .repl_worker import WorkerManager
from .repl_commands import (
    ConfigCommands,
    ArgCommands,
    MemoryCommands,
    WorkflowCommands,
    ModelCommands,
)

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

        # Initialize worker manager
        self.worker_manager = WorkerManager()

        # Initialize command handlers
        self.config_commands = ConfigCommands(self)
        self.arg_commands = ArgCommands(self)
        self.memory_commands = MemoryCommands(self)
        self.workflow_commands = WorkflowCommands(self)
        self.model_commands = ModelCommands(self)

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
        self.worker_manager.shutdown_worker()
        print("Goodbye!")
        return True

    def do_quit(self, arg):
        """Exit the REPL (alias for exit)"""
        return self.do_exit(arg)

    # ========================================================================
    # Command delegation to specialized handlers
    # ========================================================================

    def do_config(self, arg):
        """Configure global settings."""
        self.config_commands.do_config(arg)

    def do_arg(self, arg):
        """Manage workflow arguments."""
        self.arg_commands.do_arg(arg)

    def do_memory(self, arg):
        """Manage GPU memory."""
        self.memory_commands.do_memory(arg)

    def do_workflow(self, arg):
        """Manage workflows."""
        self.workflow_commands.do_workflow(arg)

    def do_model(self, arg):
        """Control model execution."""
        self.model_commands.do_model(arg)

    def default(self, line):
        """Handle unknown commands"""
        print(f"Unknown command: {line}")
        print("Type 'help' or '?' for a list of commands")

    # ========================================================================
    # Helper methods
    # ========================================================================

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
