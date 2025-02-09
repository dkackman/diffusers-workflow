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
        # Initialize cmd.Cmd first, before setting up our variables
        cmd.Cmd.__init__(self)
        # Initialize variables dictionary with default values
        self.variables = {
            'output_dir': './outputs',  # Default output directory
            'log_level': 'INFO'  # Default log level
        }
        self.current_workflow = None
    
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
        """Set a variable value. Usage: set variable=value"""
        if not arg:
            # If no argument, show all variables
            print("Current variables:")
            for name, value in self.variables.items():
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
            
            self.variables[name] = value
            print(f"Set {name}={value}")
            
        except ValueError:
            print("Error: Invalid format. Use: set variable=value")
    
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
            output_dir = self.variables['output_dir']
            if not os.path.exists(output_dir):
                print(f"Warning: Output directory {output_dir} does not exist")
            
            self.current_workflow = workflow_from_file(file_path, output_dir)
            print(f"Loaded workflow: {self.current_workflow.name}")
            
            # Try to validate the workflow immediately
            try:
                self.current_workflow.validate()
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