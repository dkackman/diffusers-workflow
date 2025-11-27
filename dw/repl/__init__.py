"""
REPL subpackage for Diffusers Workflow.

Contains the interactive REPL interface components:
- worker: Worker process management
- commands: Command handlers for REPL
"""

# Re-export the main REPL class from the parent repl.py module
# This allows: from dw.repl import DiffusersWorkflowREPL
# Note: There's a naming conflict - dw.repl is both a module (repl.py) and a package (repl/)
# We work around this by loading the parent module file directly
import os
import importlib.util

# Get the parent directory and load repl.py
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_repl_file = os.path.join(_parent_dir, "repl.py")

if os.path.exists(_repl_file):
    # Load the parent repl.py module
    _spec = importlib.util.spec_from_file_location("dw.repl_module", _repl_file)
    if _spec and _spec.loader:
        _repl_module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_repl_module)
        # Export the class
        DiffusersWorkflowREPL = _repl_module.DiffusersWorkflowREPL
        __all__ = ["DiffusersWorkflowREPL"]
    else:
        __all__ = []
else:
    __all__ = []
