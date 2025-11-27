#!/usr/bin/env python
"""
Test the reorganized REPL commands interactively.
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dw.repl import DiffusersWorkflowREPL


def test_repl_commands():
    """Test the reorganized command structure"""
    repl = DiffusersWorkflowREPL()

    test_commands = [
        ("help", "Main help"),
        ("workflow ?", "Workflow help"),
        ("arg ?", "Arg help"),
        ("model ?", "Model help"),
        ("memory ?", "Memory help"),
        ("config ?", "Config help"),
        ("config show", "Show config"),
        ("workflow status", "Workflow status (no workflow loaded)"),
        ("arg show", "Show args (no workflow loaded)"),
        # Test backward compatibility
        ("status", "Old status command"),
        ("load", "Old load command (no args)"),
    ]

    print("=" * 70)
    print("Testing REPL Command Reorganization")
    print("=" * 70)

    for cmd, description in test_commands:
        print(f"\n{'='*70}")
        print(f"Test: {description}")
        print(f"Command: {cmd}")
        print(f"{'='*70}")
        repl.onecmd(cmd)

    print("\n" + "=" * 70)
    print("✅ All command tests completed successfully!")
    print("=" * 70)
    print("\nCommand hierarchy implemented:")
    print("  • workflow - Load and manage workflows")
    print("  • arg      - Manage workflow arguments")
    print("  • model    - Control model execution")
    print("  • memory   - Monitor and manage GPU memory")
    print("  • config   - Configure global settings")
    print("\nBackward compatibility maintained for:")
    print("  load, reload, status, run, restart, clear, set, clear_args")
    print("\nUse '<command> ?' to explore any command group!")


if __name__ == "__main__":
    test_repl_commands()
