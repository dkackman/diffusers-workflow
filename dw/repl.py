import cmd
import sys
import argparse
import logging
import os
from . import startup
from .workflow import workflow_from_file
from .security import (
    validate_path, validate_workflow_path, validate_output_path,
    validate_variable_name, validate_string_input, sanitize_command_args,
    SecurityError, PathTraversalError, InvalidInputError
)
import torch
import subprocess 

logger = logging.getLogger("dw")

class DiffusersWorkflowREPL(cmd.Cmd):
    """Interactive command line interface for Diffusers Workflow"""
    
    intro = 'Welcome to Diffusers Workflow REPL. Type help or ? to list commands.\n'
    prompt = 'dw> '
    use_rawinput = True  # Ensure we're using raw_input for command reading
    
    def __init__(self):
        # Initialize cmd.Cmd first, before setting up our globals
        cmd.Cmd.__init__(self)
        # Initialize globals dictionary with default values
        self.globals = {
            'output_dir': './outputs',  # Default output directory
            'log_level': 'INFO',  # Default log level
            'workflow_dir': './examples'  # Default workflow directory
        }
        self.current_workflow = None
        self.workflow_args = {}  # Store workflow arguments
    
    def preloop(self):
        """Hook method executed once when cmdloop() is called."""
        try:
            import readline
            readline.read_history_file('.dw_history')
        except (ImportError, FileNotFoundError):
            pass
    
    def postloop(self):
        """Hook method executed once when cmdloop() is about to return."""
        try:
            import readline
            readline.write_history_file('.dw_history')
        except (ImportError, FileNotFoundError):
            pass
    
    def do_help(self, arg):
        """List available commands with "help" or detailed help with "help cmd"."""
        super().do_help(arg)
    
    def do_exit(self, arg):
        """Exit the REPL"""
        print('Goodbye!')
        return True
    
    def do_set(self, arg):
        """Set a global value. Usage: set global_name=value"""
        if not arg:
            # If no argument, show all globals
            print("Current globals:")
            for name, value in self.globals.items():
                print(f"{name}={value}")
            return

        try:
            name, value = arg.split('=', 1)
            name = name.strip()
            value = value.strip()
            
            # Special handling for output_dir
            if name == 'output_dir':
                try:
                    value = validate_output_path(value, None)
                    # Check if directory exists
                    if not os.path.exists(value):
                        print(f"Warning: Directory '{value}' does not exist")
                except SecurityError as e:
                    print(f"Error: Invalid output directory: {e}")
                    return
            
            # Special handling for log_level
            elif name == 'log_level':
                valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
                value = value.upper()
                if value not in valid_levels:
                    print(f"Error: Log level must be one of: {', '.join(valid_levels)}")
                    return
                # Update the log level
                logging.getLogger().setLevel(value)
                print(f"Log level set to {value}")

            elif name == 'workflow_dir':
                try:
                    value = validate_path(value, allow_create=False)
                    if not os.path.exists(value):
                        print(f"Warning: Directory '{value}' does not exist")
                        return
                except SecurityError as e:
                    print(f"Error: Invalid workflow directory: {e}")
                    return
                
            self.globals[name] = value
            print(f"Set {name}={value}")
            
        except ValueError:
            print("Error: Invalid format. Use: set global_name=value")
    
    def do_arg(self, arg):
        """Set or view workflow arguments. Usage: arg [name=value]"""
        if not self.current_workflow:
            print("Error: No workflow loaded. Use 'load' command first")
            return
        
        if not arg:               
            print("\nAvailable variables in workflow and their default values:")
            workflow_vars = self.current_workflow.variables
            if not workflow_vars:
                print("  No variables defined in workflow")
            else:
                for var_name, var_def in workflow_vars.items():
                    print(f"  {var_name}: {var_def}")

            # Show current arguments and available variables
            print("\nCurrent argument values:")
            if not self.workflow_args:
                print("  No arguments set")
            else:
                for name, value in self.workflow_args.items():
                    print(f"  {name}={value}")

            return

        try:
            name, value = arg.split('=', 1)
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
            print("Error: Invalid format. Use: arg name=value")

    def do_clear_args(self, arg):
        """Clear all workflow arguments"""
        self.workflow_args = {}
        print("All workflow arguments cleared")
    
    def do_load(self, arg):
        """Load a workflow from a JSON file. Usage: load [path/to/]workflow[.json]"""
        if not arg:
            print("Error: Please specify a workflow file path or name")
            return
        
        try:
            file_path = validate_string_input(arg.strip(), max_length=1000)
        except InvalidInputError as e:
            print(f"Error: Invalid file path: {e}")
            return
        
        # If this isn't an absolute path or relative path starting with ./ or ../
        if not os.path.isabs(file_path) and not file_path.startswith(('./','../')):
            # Treat as a workflow name in the default directory
            # Add .json extension if not present
            if not file_path.endswith('.json'):
                file_path = f"{file_path}.json"
            try:
                file_path = validate_path(
                    os.path.join(self.globals['workflow_dir'], file_path),
                    self.globals['workflow_dir'],
                    allow_create=False
                )
            except SecurityError as e:
                print(f"Error: Invalid workflow path: {e}")
                return
        else:
            try:
                file_path = validate_workflow_path(file_path, self.globals['workflow_dir'])
            except SecurityError as e:
                print(f"Error: Invalid workflow path: {e}")
                return
        
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            return

        try:
            output_dir = self.globals['output_dir']
            if not os.path.exists(output_dir):
                print(f"Warning: Output directory {output_dir} does not exist")
            
            workflow = workflow_from_file(file_path, output_dir)
            
            # Try to validate the workflow immediately
            try:
                workflow.validate()
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
    
    def do_status(self, arg):
        """Show current workflow status"""
        if self.current_workflow is None:
            print("No workflow currently loaded")
        else:
            print(f"Current workflow: {self.current_workflow.name}")
            print(f"File: {self.current_workflow.file_spec}")
    
    def do_run(self, arg):
        """Run the currently loaded workflow with set arguments"""
        if not self.current_workflow:
            print("Error: No workflow loaded. Use 'load' command first")
            return
        
        try:
            print(f"Running workflow: {self.current_workflow.name}")
            print(f"Using arguments: {self.workflow_args}")
            
            # Build command line arguments with security validation
            try:
                # Validate all inputs before building command
                output_dir = validate_output_path(self.globals['output_dir'], None)
                log_level = validate_string_input(self.globals['log_level'], 20)
                workflow_spec = validate_workflow_path(self.current_workflow.file_spec)
                
                # Build base command
                cmd = [
                    "python",
                    "-Xfrozen_modules=off",
                    "-m",
                    "dw.run",
                    "-o", output_dir,
                    "-l", log_level,
                    workflow_spec
                ]
                
                # Add workflow arguments as name=value pairs with validation
                for name, value in self.workflow_args.items():
                    validated_name = validate_variable_name(name)
                    validated_value = validate_string_input(str(value), max_length=10000, allow_empty=True)
                    cmd.append(f"{validated_name}={validated_value}")
                
                # Sanitize command arguments
                sanitized_cmd = sanitize_command_args(cmd)
                
            except SecurityError as e:
                print(f"Error: Security validation failed: {e}")
                return
            
            # Run the workflow in a subprocess with streaming output
            process = subprocess.Popen(
                sanitized_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True,
                shell=False  # Never use shell=True for security
            )
            
            # Stream output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.rstrip())
                
            # Get return code and handle completion
            return_code = process.poll()
            if return_code == 0:
                print("Workflow completed successfully")
            else:
                print(f"Workflow failed with return code {return_code}")
            
        except Exception as e:
            print(f"Error launching workflow: {str(e)}")
    
    def do_reload(self, arg):
        """Reload the current workflow from its file"""
        if not self.current_workflow:
            print("Error: No workflow loaded. Use 'load' command first")
            return
        
        try:
            file_path = self.current_workflow.file_spec
            print(f"Reloading workflow from: {file_path}")
            
            # Load and validate the workflow
            workflow = workflow_from_file(file_path, self.globals['output_dir'])
            workflow.validate()
            
            # Replace current workflow
            self.current_workflow = workflow
            print(f"Reloaded workflow: {workflow.name}")
            print("Workflow validated successfully")
            
        except Exception as e:
            print(f"Error reloading workflow: {str(e)}")
    
    def default(self, line):
        """Handle unknown commands"""
        print(f"Unknown command: {line}")
        print("Type 'help' for a list of commands")

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