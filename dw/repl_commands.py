"""
Command handlers for REPL.

Contains all the command implementations organized by category:
- Config commands
- Argument commands
- Memory commands
- Workflow commands
"""

import os
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

            self.repl.globals[name] = value
            print(f"Set {name}={value}")

        except ValueError:
            print("Error: Invalid format. Use: config set name=value")


class ArgCommands:
    """Handles workflow argument management commands."""

    def __init__(self, repl):
        """Initialize with reference to REPL instance."""
        self.repl = repl

    def do_arg(self, arg: str):
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
                print(f"  {var_name}: {var_def}")

        print("\nCurrent argument values:")
        if not self.repl.workflow_args:
            print("  No arguments set")
        else:
            for name, value in self.repl.workflow_args.items():
                print(f"  {name}={value}")
        print()

    def _arg_set(self, arg: str):
        """Set a workflow argument"""
        if not self.repl.current_workflow:
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
                value = validate_string_input(
                    value, max_length=MAX_VARIABLE_VALUE_LENGTH, allow_empty=True
                )
            except InvalidInputError as e:
                print(f"Error: Invalid input: {e}")
                return

            # Verify this is a valid variable name for the workflow
            if name not in self.repl.current_workflow.variables:
                print(f"Error: '{name}' is not defined in workflow variables")
                return

            self.repl.workflow_args[name] = value
            print(f"Set argument {name}={value}")

        except ValueError:
            print("Error: Invalid format. Use: arg set name=value")

    def _arg_clear(self, arg: str):
        """Clear all workflow arguments"""
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
        """Manage workflows. Usage: workflow ? | load <file> | reload | status | run | restart"""
        if not arg or arg == "?":
            print("\nWorkflow commands:")
            print("  workflow load <file> - Load a workflow from a JSON file")
            print("  workflow reload      - Reload the current workflow from disk")
            print("  workflow status      - Show current workflow information")
            print("  workflow run         - Execute the currently loaded workflow")
            print("  workflow run ask <arg> - Prompt for an argument value and run")
            print("  workflow restart     - Restart the worker process (clears cache)")
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
        elif subcommand == "run":
            self._workflow_run(subarg)
        elif subcommand == "restart":
            self._workflow_restart(subarg)
        else:
            print(f"Unknown workflow subcommand: {subcommand}")
            print("Use 'workflow ?' for help")

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

                # If workflow changed, shutdown worker (will restart on next run)
                old_path = (
                    self.repl.current_workflow.file_spec
                    if self.repl.current_workflow
                    else None
                )
                if old_path != file_path:
                    self.repl.worker_manager.shutdown_worker()

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
        else:
            print(f"\nCurrent workflow: {self.repl.current_workflow.name}")
            print(f"File: {self.repl.current_workflow.file_spec}")
            print()

    def _workflow_run(self, arg: str):
        """Run the currently loaded workflow with set arguments"""
        if not self.repl.current_workflow:
            print("Error: No workflow loaded. Use 'workflow load' command first")
            return

        # Handle "run ask <arg_name>" subcommand
        if arg.strip() == "ask":
            print("Error: Please specify an argument name")
            print("Usage: workflow run ask <arg_name>")
            return

        if arg.startswith("ask "):
            arg_name = arg[4:].strip()  # Remove "ask " prefix
            if not arg_name:
                print("Error: Please specify an argument name")
                print("Usage: workflow run ask <arg_name>")
                return

            # Validate that the argument exists in the workflow's variables
            if arg_name not in self.repl.current_workflow.variables:
                print(f"Error: '{arg_name}' is not defined in workflow variables")
                print(
                    f"Available variables: {', '.join(self.repl.current_workflow.variables.keys())}"
                )
                return

            # Prompt user for the value
            # Temporarily disable readline history to prevent the input value from being saved
            readline_available = False
            try:
                import readline

                readline_available = True
                # Disable auto-history for this input
                if hasattr(readline, "set_auto_history"):
                    readline.set_auto_history(False)
            except (ImportError, AttributeError):
                pass

            try:
                prompt_text = f"Enter value for '{arg_name}': "
                user_value = input(prompt_text).strip()

                # Validate the input
                try:
                    validated_name = validate_variable_name(arg_name)
                    validated_value = validate_string_input(
                        user_value,
                        max_length=MAX_VARIABLE_VALUE_LENGTH,
                        allow_empty=True,
                    )
                except InvalidInputError as e:
                    print(f"Error: Invalid input: {e}")
                    return

                # Set the argument
                self.repl.workflow_args[validated_name] = validated_value
                print(f"Set argument {validated_name}={validated_value}")
                print()

            except (EOFError, KeyboardInterrupt):
                print("\nCancelled")
                return
            finally:
                # Always re-enable readline history after input
                if readline_available:
                    try:
                        import readline

                        if hasattr(readline, "set_auto_history"):
                            readline.set_auto_history(True)
                    except (ImportError, AttributeError):
                        pass

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

            # Process results from worker
            from .repl_worker import WORKER_RESULT_TIMEOUT_SECONDS

            while True:
                try:
                    result = self.repl.worker_manager.get_result(
                        timeout=WORKER_RESULT_TIMEOUT_SECONDS
                    )
                    result_type = result.get("type")

                    if result_type == "output":
                        print(result["message"])
                    elif result_type == "workflow_loaded":
                        print(f"Models loaded for workflow: {result['workflow_name']}")
                    elif result_type == "memory_info":
                        self.repl._print_memory_info(result["info"])
                    elif result_type == "success":
                        print(result["message"])
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
                            "Worker process has terminated. Use 'workflow restart' to start a new worker.\n"
                        )
                        # Mark worker as inactive (already crashed, no need to shutdown)
                        self.repl.worker_manager.worker_active = False
                        self.repl.worker_manager.worker_process = None
                        break
                    else:
                        print(f"Unknown result type: {result_type}")

                except Exception as e:
                    print("\n" + "=" * 80)
                    print(f"ERROR receiving results: {e}")
                    print("=" * 80)
                    print("Worker communication failed. Shutting down worker.\n")
                    self.repl.worker_manager.shutdown_worker()
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
            self.repl.worker_manager.shutdown_worker()

    def _workflow_restart(self, arg: str):
        """Restart the worker process"""
        print("Restarting worker process...")
        self.repl.worker_manager.shutdown_worker()
        print("Worker shutdown complete")
        print("Worker will restart on next run")
