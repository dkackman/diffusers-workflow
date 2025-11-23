#!/bin/bash
# Setup script for test infrastructure

echo "======================================"
echo "Setting up test infrastructure"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version
echo ""

# Install test dependencies
echo "Installing test dependencies..."
pip install -r requirements-test.txt
echo ""

# Verify pytest installation
echo "Verifying pytest installation..."
python3 -m pytest --version
echo ""

# Run tests
echo "======================================"
echo "Running test suite..."
echo "======================================"
python3 -m pytest tests/ -v --tb=short

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "✓ All tests passed!"
    echo "======================================"
else
    echo ""
    echo "======================================"
    echo "✗ Some tests failed"
    echo "======================================"
    exit 1
fi
