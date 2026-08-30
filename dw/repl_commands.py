"""
Command handlers for REPL.

Contains all the command implementations organized by category:
- Config commands
- Argument commands
- Memory commands
- Workflow commands
"""

import contextlib
import os
import shlex
import logging
from .security import (
    validate_path,
    validate_workflow_path,
    validate_output_path,
    validate_variable_name,
    validate_string_input,
    SecurityError,
    InvalidInputError,
    MAX_VARIABLE_VALUE_LENGTH,
    MAX_FILE_PATH_LENGTH,
)
from .workflow import workflow_from_file

logger = logging.getLogger("dw")


class ConfigCommands:
    """Handles configuration commands."""

    def __init__(self, repl):
        """Initialize with reference to REPL instance."""
        self.repl = repl

    def do_config(self, arg: str):
        """Configure global settings. Usage: config ? | show | set <name>=<value>"""
        if not arg or arg == "?":
            print("\nConfig commands:")
            print("  config show               - Show all configuration settings")
            print("  config set <name>=<value> - Set a configuration value")
            print("\nAvailable settings:")
            print("  output_dir   - Directory for output files (default: ./outputs)")
            print(
                "  log_level    - Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )
            print("  workflow_dir - Where 'workflow load <name>' and 'workflow list'")
            print("                 look for workflows (default: ./examples)")
            print()
            print("These apply to this REPL session. Standing settings (device,")
            print("log file, TF32) live in ~/.diffusers_helper/settings.json")
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

    def _config_show(self, arg: str):
        """Show configuration settings"""
        print("\nCurrent configuration:")
        for name, value in self.repl.globals.items():
            print(f"  {name}={value}")
        print()

    def _config_set(self, arg: str):
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
                # Update the log level for dw's own logger
                logging.getLogger("dw").setLevel(value)
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

            self.repl.globals[name] = value
            print(f"Set {name}={value}")

        except ValueError:
            print("Error: Invalid format. Use: config set name=value")


def display_value(value, limit=90):
    """A value shortened for terminal display - a paragraph-long prompt
    default would otherwise bury every other line of arg show."""
    text = str(value)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


class ArgCommands:
    """Handles workflow argument management commands."""

    def __init__(self, repl):
        """Initialize with reference to REPL instance."""
        self.repl = repl

    def do_arg(self, arg: str):
        """Manage workflow arguments. Usage: arg ? | show | set <name>=<value> | clear"""
        if arg == "?":
            print("\nArg commands:")
            print(
                "  arg show               - Show workflow variables and current values"
            )
            print("  arg set <name>=<value> - Set an argument (quotes optional):")
            print('                             arg set prompt="a cat in a hat"')
            print("  arg clear [<name>]     - Clear one argument, or all of them")
            print()
            print("Arguments override the workflow's variables on the next run and")
            print("reset when a different workflow is loaded. Shortcuts: 'set n=v'")
            print("is 'arg set', and 'workflow run n=v ...' sets and runs in one line.")
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

    def _arg_show(self, arg: str):
        """Show workflow arguments"""
        if not self.repl.current_workflow:
            print("Error: No workflow loaded. Use 'workflow load' command first")
            return

        print("\nAvailable variables in workflow and their default values:")
        workflow_vars = self.repl.current_workflow.variables
        if not workflow_vars:
            print("  No variables defined in workflow")
        else:
            for var_name, var_def in workflow_vars.items():
                print(f"  {var_name}: {display_value(var_def)}")

        print("\nCurrent argument values:")
        if not self.repl.workflow_args:
            print("  No arguments set")
        else:
            for name, value in self.repl.workflow_args.items():
                print(f"  {name}={display_value(value)}")
        print()

    def _arg_set(self, arg: str):
        """Set a workflow argument.

        Returns:
            True when the argument was set, False otherwise - callers that
            set several in a row (workflow run n=v ...) stop at the first
            failure instead of running with half the overrides.
        """
        if not self.repl.current_workflow:
            print("Error: No workflow loaded. Use 'workflow load' command first")
            return False

        if not arg:
            print("Error: Please specify argument name and value")
            print("Usage: arg set <name>=<value>")
            return False

        try:
            name, value = arg.split("=", 1)
            name = name.strip()
            value = value.strip()

            # There is no shell here to strip quotes, so a value quoted the
            # way every example shows it would otherwise keep literal quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]

            # Validate variable name
            try:
                name = validate_variable_name(name)
                value = validate_string_input(
                    value, max_length=MAX_VARIABLE_VALUE_LENGTH, allow_empty=True
                )
            except InvalidInputError as e:
                print(f"Error: Invalid input: {e}")
                return False

            # Verify this is a valid variable name for the workflow
            if name not in self.repl.current_workflow.variables:
                print(f"Error: '{name}' is not defined in workflow variables")
                available = ", ".join(self.repl.current_workflow.variables)
                print(f"Available variables: {available or '(none)'}")
                return False

            self.repl.workflow_args[name] = value
            print(f"Set argument {name}={value}")
            return True

        except ValueError:
            print("Error: Invalid format. Use: arg set name=value")
            return False

    def _arg_clear(self, arg: str):
        """Clear one workflow argument, or all of them"""
        name = arg.strip()
        if name:
            if name in self.repl.workflow_args:
                del self.repl.workflow_args[name]
                print(f"Cleared argument {name}")
            else:
                print(f"Argument '{name}' is not set")
            return
        self.repl.workflow_args = {}
        print("All workflow arguments cleared")


class MemoryCommands:
    """Handles GPU memory management commands."""

    def __init__(self, repl):
        """Initialize with reference to REPL instance."""
        self.repl = repl

    def do_memory(self, arg: str):
        """Manage GPU memory. Usage: memory ? | show | clear"""
        if not arg or arg == "?":
            print("\nMemory commands:")
            print("  memory show  - Show the worker's GPU memory usage")
            print("  memory clear - Release cached models and empty the device cache")
            print()
            print("The worker process starts on the first 'workflow run', so there")
            print("is nothing to show before that. 'workflow restart' frees")
            print("everything by stopping the worker process itself.")
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

    def _memory_show(self, arg: str):
        """Show current GPU memory usage"""
        if not self.repl.worker_manager.worker_active:
            print("No worker process running")
            return

        try:
            self.repl.worker_manager.send_command({"type": "memory_status"})
            result = self.repl.worker_manager.get_result(timeout=5)

            if result["type"] == "memory_status":
                self.repl._print_memory_info(result.get("info", {}))
            else:
                print(f"Unexpected response: {result}")
        except Exception as e:
            print(f"Error getting memory status: {e}")

    def _memory_clear(self, arg: str):
        """Clear GPU memory and cached models"""
        if not self.repl.worker_manager.worker_active:
            print("No worker process running")
            return

        try:
            print("Clearing GPU memory...")
            self.repl.worker_manager.send_command({"type": "clear_memory"})

            # Wait for response
            result = self.repl.worker_manager.get_result(timeout=30)
            if result["type"] == "memory_cleared":
                self.repl._print_memory_info(result.get("info", {}))
                print("GPU memory cleared successfully")
            else:
                print(f"Unexpected response: {result}")
        except Exception as e:
            print(f"Error clearing memory: {e}")
            self.repl.worker_manager.shutdown_worker()


class WorkflowCommands:
    """Handles workflow management commands."""

    def __init__(self, repl):
        """Initialize with reference to REPL instance."""
        self.repl = repl

    def do_workflow(self, arg: str):
        """Manage workflows. Usage: workflow ? | list | load <file> | reload | status | run | restart"""
        if not arg or arg == "?":
            print("\nWorkflow commands:")
            print(
                "  workflow list           - List workflows in the workflow directory"
            )
            print(
                "  workflow load <name>    - Load by name from workflow_dir, or by path"
            )
            print(
                "  workflow reload         - Re-read the current workflow file (keeps args)"
            )
            print(
                "  workflow status         - Show workflow, arguments and worker state"
            )
            print("  workflow run            - Run the loaded workflow")
            print("  workflow run <n>=<v> .. - Set arguments, then run, in one line")
            print("  workflow run ask <arg>  - Prompt for one argument value, then run")
            print(
                "  workflow restart        - Stop the worker process (frees all models)"
            )
            print()
            print("Runs execute in a persistent worker, so models stay loaded between")
            print("runs. After editing the workflow file, just run again - pipelines")
            print("whose definition did not change stay cached. Ctrl+C during a run")
            print("cancels it (models stay cached); a second Ctrl+C stops the worker.")
            print()
            return

        parts = arg.split(None, 1)
        subcommand = parts[0]
        subarg = parts[1] if len(parts) > 1 else ""

        if subcommand == "list":
            self._workflow_list(subarg)
        elif subcommand == "load":
            self._workflow_load(subarg)
        elif subcommand == "reload":
            self._workflow_reload(subarg)
        elif subcommand == "status":
            self._workflow_status(subarg)
        elif subcommand == "run":
            self._workflow_run(subarg)
        elif subcommand == "restart":
            self._workflow_restart(subarg)
        else:
            print(f"Unknown workflow subcommand: {subcommand}")
            print("Use 'workflow ?' for help")

    def workflow_names(self):
        """Workflow names under workflow_dir, as load accepts them.

        Names are paths relative to workflow_dir with the .json dropped
        ('flux/FluxDev'), sorted for stable listing and completion.
        """
        workflow_dir = self.repl.globals["workflow_dir"]
        names = []
        if not os.path.isdir(workflow_dir):
            return names
        for root, _dirs, files in os.walk(workflow_dir):
            for file_name in files:
                if file_name.endswith(".json"):
                    relative = os.path.relpath(
                        os.path.join(root, file_name), workflow_dir
                    )
                    names.append(relative[: -len(".json")])
        return sorted(names)

    def _workflow_list(self, arg: str):
        """List workflows available in the workflow directory"""
        workflow_dir = self.repl.globals["workflow_dir"]
        names = self.workflow_names()
        if not names:
            print(f"No workflows found in {workflow_dir}")
            print(
                "Point workflow_dir somewhere else with: config set workflow_dir=<path>"
            )
            return
        print(f"\nWorkflows in {workflow_dir}:")
        for name in names:
            print(f"  {name}")
        print("\nLoad one with: workflow load <name>")
        print()

    def _workflow_load(self, arg: str):
        """Load a workflow from a JSON file"""
        if not arg:
            print("Error: Please specify a workflow file path or name")
            return

        try:
            file_path = validate_string_input(
                arg.strip(), max_length=MAX_FILE_PATH_LENGTH
            )
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
                    os.path.join(self.repl.globals["workflow_dir"], file_path),
                    self.repl.globals["workflow_dir"],
                    allow_create=False,
                )
            except SecurityError as e:
                print(f"Error: Invalid workflow path: {e}")
                return
        else:
            try:
                file_path = validate_workflow_path(
                    file_path, self.repl.globals["workflow_dir"]
                )
            except SecurityError as e:
                print(f"Error: Invalid workflow path: {e}")
                return

        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            return

        try:
            output_dir = self.repl.globals["output_dir"]
            if not os.path.exists(output_dir):
                print(f"Warning: Output directory {output_dir} does not exist")

            workflow = workflow_from_file(file_path, output_dir)

            # Try to validate the workflow immediately
            try:
                workflow.validate()

                # The worker handles workflow switches itself now - it frees
                # the old workflow's models before loading the new one - so
                # loading a different file no longer restarts the process
                self.repl.current_workflow = workflow
                # Clear any existing arguments when loading new workflow
                self.repl.workflow_args = {}
                print(f"Loaded workflow: {workflow.name}")
                print("Workflow validated successfully")

            except Exception as e:
                print(f"Warning: Workflow validation failed: {str(e)}")

        except Exception as e:
            print(f"Error loading workflow: {str(e)}")
            self.repl.current_workflow = None

    def _workflow_reload(self, arg: str):
        """Reload the current workflow from its file"""
        if not self.repl.current_workflow:
            print("Error: No workflow loaded. Use 'workflow load' command first")
            return

        try:
            file_path = self.repl.current_workflow.file_spec
            print(f"Reloading workflow from: {file_path}")

            # Load and validate the workflow
            workflow = workflow_from_file(file_path, self.repl.globals["output_dir"])
            workflow.validate()

            # Replace current workflow
            self.repl.current_workflow = workflow
            print(f"Reloaded workflow: {workflow.name}")
            print("Workflow validated successfully")

        except Exception as e:
            print(f"Error reloading workflow: {str(e)}")

    def _workflow_status(self, arg: str):
        """Show current workflow status"""
        if self.repl.current_workflow is None:
            print("No workflow currently loaded")
            print(
                "Use 'workflow list' to see what is available, then 'workflow load <name>'"
            )
            return

        workflow = self.repl.current_workflow
        print(f"\nCurrent workflow: {workflow.name}")
        print(f"  File: {workflow.file_spec}")
        print(f"  Output directory: {self.repl.globals['output_dir']}")

        variables = workflow.variables
        if variables:
            print("  Variables:")
            for name, default in variables.items():
                if name in self.repl.workflow_args:
                    print(
                        f"    {name} = {display_value(self.repl.workflow_args[name])}"
                        f"  (default: {display_value(default, limit=40)})"
                    )
                else:
                    print(f"    {name} = {display_value(default)}")
        else:
            print("  Variables: none defined")

        manager = self.repl.worker_manager
        worker_alive = (
            manager.worker_active
            and manager.worker_process is not None
            and manager.worker_process.is_alive()
        )
        if worker_alive:
            print("  Worker: running (models stay cached between runs)")
        else:
            print("  Worker: not running (starts on the next 'workflow run')")
        print()

    @contextlib.contextmanager
    def _history_suppressed(self):
        """Readline auto-history off for one input, so values are not saved"""
        readline = None
        try:
            import readline
        except ImportError:
            pass

        suppress = readline is not None and hasattr(readline, "set_auto_history")
        if suppress:
            readline.set_auto_history(False)
        try:
            yield
        finally:
            if suppress:
                readline.set_auto_history(True)

    def _prompt_for_argument(self, arg_name):
        """Ask the user for one workflow argument's value.

        Returns:
            True when the value was set, False on invalid input or cancel
        """
        # Validate that the argument exists in the workflow's variables
        if arg_name not in self.repl.current_workflow.variables:
            print(f"Error: '{arg_name}' is not defined in workflow variables")
            print(
                f"Available variables: {', '.join(self.repl.current_workflow.variables.keys())}"
            )
            return False

        try:
            with self._history_suppressed():
                user_value = input(f"Enter value for '{arg_name}': ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled")
            return False

        try:
            validated_name = validate_variable_name(arg_name)
            validated_value = validate_string_input(
                user_value,
                max_length=MAX_VARIABLE_VALUE_LENGTH,
                allow_empty=True,
            )
        except InvalidInputError as e:
            print(f"Error: Invalid input: {e}")
            return False

        self.repl.workflow_args[validated_name] = validated_value
        print(f"Set argument {validated_name}={validated_value}")
        print()
        return True

    def _workflow_run(self, arg: str):
        """Run the currently loaded workflow with set arguments"""
        if not self.repl.current_workflow:
            print("Error: No workflow loaded. Use 'workflow load' command first")
            return

        # Handle "run ask <arg_name>" subcommand. "ask" with nothing after it
        # (any amount of whitespace) is caught here, so the name is never empty
        if arg.strip() == "ask":
            print("Error: Please specify an argument name")
            print("Usage: workflow run ask <arg_name>")
            return

        if arg.startswith("ask "):
            if not self._prompt_for_argument(arg[4:].strip()):
                return
        elif arg.strip():
            # Inline overrides: workflow run steps=30 prompt="a cat". Same
            # semantics as arg set for each pair, then run.
            try:
                assignments = shlex.split(arg)
            except ValueError as e:
                print(f"Error: Could not parse arguments: {e}")
                return
            for assignment in assignments:
                if "=" not in assignment:
                    print(f"Error: Expected <name>=<value>, got '{assignment}'")
                    print("Usage: workflow run [<name>=<value> ...] | ask <arg>")
                    return
            for assignment in assignments:
                if not self.repl.arg_commands._arg_set(assignment):
                    return

        try:
            # Validate inputs
            output_dir = validate_output_path(self.repl.globals["output_dir"], None)
            workflow_spec = validate_workflow_path(self.repl.current_workflow.file_spec)

            # Ensure worker is running
            self.repl.worker_manager.ensure_worker(self.repl.globals["log_level"])

            print(f"Running workflow: {self.repl.current_workflow.name}")
            if self.repl.workflow_args:
                print(f"Using arguments: {self.repl.workflow_args}")

            # Send execute command to worker
            self.repl.worker_manager.send_command(
                {
                    "type": "execute",
                    "workflow_path": workflow_spec,
                    "arguments": self.repl.workflow_args,
                    "output_dir": output_dir,
                    "log_level": self.repl.globals["log_level"],
                }
            )

            # Process results from worker. There is no fixed timeout - a video
            # generation or cold model download takes however long it takes,
            # progress events stream in while it works, and get_result raises
            # if the worker dies. Ctrl+C asks the worker to cancel the run; a
            # second Ctrl+C gives up on it and shuts the worker down.
            cancelling = False
            inline_progress = False

            def end_inline():
                nonlocal inline_progress
                if inline_progress:
                    print()
                    inline_progress = False

            while True:
                try:
                    result = self.repl.worker_manager.get_result()
                except KeyboardInterrupt:
                    end_inline()
                    if not cancelling:
                        cancelling = True
                        print("Cancelling... (Ctrl+C again to force-stop the worker)")
                        try:
                            self.repl.worker_manager.cancel()
                        except Exception as e:
                            print(f"Could not send cancel: {e}")
                            self.repl.worker_manager.shutdown_worker()
                            break
                        continue
                    print("Force-stopping worker...")
                    self.repl.worker_manager.shutdown_worker()
                    break
                except Exception as e:
                    end_inline()
                    print("\n" + "=" * 80)
                    print(f"ERROR receiving results: {e}")
                    print("=" * 80)
                    print("Worker communication failed. Shutting down worker.\n")
                    self.repl.worker_manager.shutdown_worker()
                    break

                result_type = result.get("type")

                if result_type == "progress":
                    event = result.get("event")
                    if event == "step_start":
                        end_inline()
                        print(
                            f"[{result['index'] + 1}/{result['total_steps']}] "
                            f"{result['step']}"
                        )
                    elif event == "iteration_start":
                        if result["total_iterations"] > 1:
                            end_inline()
                            print(
                                f"  iteration {result['iteration']}/"
                                f"{result['total_iterations']}"
                            )
                    elif event == "pipeline_step":
                        total = result.get("total_steps") or "?"
                        print(
                            f"\r  denoise step {result['step']}/{total}",
                            end="",
                            flush=True,
                        )
                        inline_progress = True
                    elif event == "step_end":
                        end_inline()
                        for saved_file in result.get("files", []):
                            print(f"  saved {saved_file}")
                    # workflow_start / workflow_end stay quiet - the step
                    # lines carry the story
                elif result_type == "output":
                    end_inline()
                    print(result["message"])
                elif result_type == "workflow_loaded":
                    end_inline()
                    print(f"Workflow loaded: {result['workflow_name']}")
                elif result_type == "memory_info":
                    end_inline()
                    self.repl._print_memory_info(result["info"])
                elif result_type == "success":
                    end_inline()
                    print(result["message"])
                    break
                elif result_type == "cancelled":
                    end_inline()
                    print(result.get("message", "Workflow run cancelled"))
                    break
                elif result_type == "error":
                    end_inline()
                    print("\n" + "=" * 80)
                    print(f"ERROR: {result['message']}")
                    if "traceback" in result:
                        print("\nTraceback:")
                        print(result["traceback"])
                    print("=" * 80)
                    print("Worker process encountered an error and stopped.\n")
                    break
                elif result_type == "worker_crashed":
                    end_inline()
                    print("\n" + "=" * 80)
                    print(f"WORKER CRASHED: {result['message']}")
                    if "traceback" in result:
                        print("\nTraceback:")
                        print(result["traceback"])
                    print("=" * 80)
                    print(
                        "Worker process has terminated. Use 'workflow restart' to start a new worker.\n"
                    )
                    # Already crashed - no shutdown handshake to attempt
                    self.repl.worker_manager.mark_crashed()
                    break
                else:
                    end_inline()
                    print(f"Unknown result type: {result_type}")

        except SecurityError as e:
            print("\n" + "=" * 80)
            print(f"SECURITY ERROR: {e}")
            print("=" * 80 + "\n")
        except Exception as e:
            print("\n" + "=" * 80)
            print(f"ERROR running workflow: {str(e)}")
            print("=" * 80)
            print("Shutting down worker.\n")
            self.repl.worker_manager.shutdown_worker()

    def _workflow_restart(self, arg: str):
        """Restart the worker process"""
        print("Restarting worker process...")
        self.repl.worker_manager.shutdown_worker()
        print("Worker shutdown complete")
        print("Worker will restart on next run")
