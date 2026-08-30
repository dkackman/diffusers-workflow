"""
Interactive REPL for Diffusers Workflow.

Main entry point for the REPL interface. Delegates command handling
to specialized command classes and worker management to WorkerManager.
"""

import cmd
import sys
import argparse
import difflib
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

    intro = (
        "Welcome to the Diffusers Workflow REPL.\n"
        "  'workflow list' shows available workflows, 'help' gets you oriented.\n"
    )
    prompt = "dw> "
    use_rawinput = True  # Ensure we're using raw_input for command reading

    def __init__(self):
        # Initialize cmd.Cmd first, before setting up our globals
        cmd.Cmd.__init__(self)
        # Initialize globals dictionary with default values
        self.globals = {
            "output_dir": "./outputs",  # Default output directory
            "log_level": "INFO",  # Default log level
            "workflow_dir": "./workflows",  # Default workflow directory
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

    # Group commands whose detailed help lives in the group handler - 'help
    # workflow' and 'workflow ?' must tell the same story
    COMMAND_GROUPS = ("workflow", "arg", "memory", "config")

    def do_help(self, arg):
        """List available commands with "help" or detailed help with "help cmd"."""
        if not arg:
            print("\nDiffusers Workflow REPL")
            print("=" * 60)
            print("\nTypical session:")
            print("  workflow list                 See available workflows")
            print("  workflow load FluxDev         Load one (validated immediately)")
            print("  arg show                      See its variables and defaults")
            print('  arg set prompt="a cat"        Override a variable')
            print("  workflow run                  Run it")
            print("  workflow run steps=30         Set arguments and run in one line")
            print("\nModels stay loaded in a worker process between runs, so the")
            print("second run of a workflow skips straight to inference.")
            print("\nCommand groups ('help <group>' or '<group> ?' for details):")
            print("  workflow  - list, load, reload, status, run, restart")
            print("  arg       - show, set, clear")
            print("  memory    - show, clear")
            print("  config    - show, set")
            print("\nShortcuts:")
            print("  run [<name>=<value> ...]  same as 'workflow run'")
            print("  load <name>               same as 'workflow load'")
            print("  set <name>=<value>        same as 'arg set'")
            print("  <TAB>                     completes commands, workflow names,")
            print("                            and argument names")
            print("\nWhile a workflow runs:")
            print("  Progress prints per step and per denoise step. Ctrl+C cancels")
            print("  the run (models stay cached); a second Ctrl+C stops the worker.")
            print("\n'exit' or 'quit' leaves the REPL.")
            print()
        elif arg.strip() in self.COMMAND_GROUPS:
            # Route to the group's own help so there is one source of truth
            getattr(self, f"do_{arg.strip()}")("?")
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
        """Manage workflows. 'workflow ?' lists subcommands."""
        self.workflow_commands.do_workflow(arg)

    # ========================================================================
    # Shortcuts for the hot loop: run / load / set
    # ========================================================================

    def do_run(self, arg):
        """Run the loaded workflow. Shortcut for 'workflow run [<name>=<value> ...]'."""
        self.workflow_commands.do_workflow(f"run {arg}".strip())

    def do_load(self, arg):
        """Load a workflow. Shortcut for 'workflow load <name>'."""
        self.workflow_commands.do_workflow(f"load {arg}".strip())

    def do_set(self, arg):
        """Set a workflow argument. Shortcut for 'arg set <name>=<value>'."""
        self.arg_commands.do_arg(f"set {arg}".strip())

    # ========================================================================
    # Tab completion
    # ========================================================================

    WORKFLOW_SUBCOMMANDS = ("list", "load", "reload", "status", "run", "restart")
    ARG_SUBCOMMANDS = ("show", "set", "clear")
    MEMORY_SUBCOMMANDS = ("show", "clear")
    CONFIG_SUBCOMMANDS = ("show", "set")
    CONFIG_KEYS = ("output_dir", "log_level", "workflow_dir")

    @staticmethod
    def _matches(candidates, text):
        return [candidate for candidate in candidates if candidate.startswith(text)]

    def _variable_candidates(self):
        """The loaded workflow's variable names, ready for name=value entry."""
        if not self.current_workflow:
            return []
        return [f"{name}=" for name in self.current_workflow.variables]

    def complete_workflow(self, text, line, begidx, endidx):
        words = line.split()
        # Completing the subcommand itself
        if len(words) == 1 or (len(words) == 2 and not line.endswith(" ")):
            return self._matches(self.WORKFLOW_SUBCOMMANDS, text)
        if words[1] == "load":
            return self._matches(self.workflow_commands.workflow_names(), text)
        if words[1] == "run":
            return self._matches(self._variable_candidates(), text)
        return []

    def complete_arg(self, text, line, begidx, endidx):
        words = line.split()
        if len(words) == 1 or (len(words) == 2 and not line.endswith(" ")):
            return self._matches(self.ARG_SUBCOMMANDS, text)
        if words[1] == "set":
            return self._matches(self._variable_candidates(), text)
        if words[1] == "clear":
            return self._matches(list(self.workflow_args), text)
        return []

    def complete_memory(self, text, line, begidx, endidx):
        return self._matches(self.MEMORY_SUBCOMMANDS, text)

    def complete_config(self, text, line, begidx, endidx):
        words = line.split()
        if len(words) == 1 or (len(words) == 2 and not line.endswith(" ")):
            return self._matches(self.CONFIG_SUBCOMMANDS, text)
        if words[1] == "set":
            return self._matches([f"{key}=" for key in self.CONFIG_KEYS], text)
        return []

    def complete_load(self, text, line, begidx, endidx):
        return self._matches(self.workflow_commands.workflow_names(), text)

    def complete_run(self, text, line, begidx, endidx):
        return self._matches(self._variable_candidates(), text)

    def complete_set(self, text, line, begidx, endidx):
        return self._matches(self._variable_candidates(), text)

    def default(self, line):
        """Handle unknown commands"""
        print(f"Unknown command: {line}")
        command = line.split()[0] if line.split() else ""
        known = [name[3:] for name in dir(self) if name.startswith("do_")]
        suggestions = difflib.get_close_matches(command, known, n=3, cutoff=0.6)
        if suggestions:
            print(f"Did you mean: {', '.join(suggestions)}?")
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
