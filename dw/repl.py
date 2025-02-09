import cmd
import sys
import argparse
import logging
import os
from . import startup
from .workflow import workflow_from_file

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
            'log_level': 'INFO'  # Default log level
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
                # Convert to absolute path
                value = os.path.abspath(value)
                # Check if directory exists
                if not os.path.exists(value):
                    print(f"Warning: Directory '{value}' does not exist")
            
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
        """Load a workflow from a JSON file. Usage: load path/to/workflow.json"""
        if not arg:
            print("Error: Please specify a workflow file path")
            return
            
        file_path = os.path.abspath(arg.strip())
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
            
            result = self.current_workflow.run(self.workflow_args)
            
            print("Workflow completed successfully")
            if result:
                print(f"Saved {len(result)} results to {self.globals['output_dir']}")
            
        except Exception as e:
            print(f"Error running workflow: {str(e)}")
    
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