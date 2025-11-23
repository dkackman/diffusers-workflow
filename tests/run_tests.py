"""
Test runner script with coverage reporting
Run: python -m tests.run_tests
"""

import sys
import subprocess


def main():
    """Run test suite with coverage"""

    print("=" * 70)
    print("Running diffusers-workflow Test Suite")
    print("=" * 70)
    print()

    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("ERROR: pytest is not installed")
        print("Install with: pip install pytest pytest-cov")
        return 1

    # Run tests with coverage
    args = [
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--color=yes",  # Colored output
        "tests/",  # Test directory
    ]

    # Add coverage if pytest-cov is available
    try:
        import pytest_cov

        args.extend(
            [
                "--cov=dw",  # Coverage for dw package
                "--cov-report=term",  # Terminal report
                "--cov-report=html",  # HTML report
            ]
        )
    except ImportError:
        print("Note: pytest-cov not installed, skipping coverage report")
        print("Install with: pip install pytest-cov")
        print()

    # Run pytest
    exit_code = pytest.main(args)

    print()
    print("=" * 70)
    if exit_code == 0:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 70)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
