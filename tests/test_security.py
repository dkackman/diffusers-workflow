"""
Tests for security module
"""

import pytest
import os
import tempfile
from dw.security import (
    validate_path,
    validate_workflow_path,
    validate_url,
    validate_variable_name,
    validate_string_input,
    sanitize_command_args,
    SecurityError,
    PathTraversalError,
    InvalidInputError,
)


def test_path_validation():
    """Test path validation functionality"""
    # Valid path should work
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.json")
        with open(test_file, "w") as f:
            f.write("{}")

        # Should validate successfully
        validated = validate_path(test_file, allow_create=False)
        assert os.path.isabs(validated)

        # Path traversal should fail
        with pytest.raises(PathTraversalError):
            validate_path("../../../etc/passwd")

        # Should work with base directory restriction
        validated = validate_path(test_file, base_dir=temp_dir, allow_create=False)
        assert validated == os.path.abspath(test_file)

        # Should fail when outside base directory
        with pytest.raises(PathTraversalError):
            validate_path("/etc/passwd", base_dir=temp_dir)


def test_url_validation():
    """Test URL validation"""
    # Valid URLs should work
    assert (
        validate_url("https://example.com/image.jpg") == "https://example.com/image.jpg"
    )
    assert validate_url("http://localhost:8080/api") == "http://localhost:8080/api"

    # Invalid schemes should fail
    with pytest.raises(InvalidInputError):
        validate_url("file:///etc/passwd")

    with pytest.raises(InvalidInputError):
        validate_url("ftp://example.com/file")

    # Malformed URLs should fail
    with pytest.raises(InvalidInputError):
        validate_url("not-a-url")


def test_variable_name_validation():
    """Test variable name validation"""
    # Valid names should work
    assert validate_variable_name("prompt") == "prompt"
    assert validate_variable_name("num_images_per_prompt") == "num_images_per_prompt"
    assert validate_variable_name("test_var_123") == "test_var_123"

    # Invalid names should fail
    with pytest.raises(InvalidInputError):
        validate_variable_name("invalid-name!")

    with pytest.raises(InvalidInputError):
        validate_variable_name("123invalid")

    with pytest.raises(InvalidInputError):
        validate_variable_name("")

    # Too long names should fail
    with pytest.raises(InvalidInputError):
        validate_variable_name("a" * 101)


def test_string_input_validation():
    """Test string input validation"""
    # Valid strings should work
    assert validate_string_input("hello world") == "hello world"
    assert validate_string_input("", allow_empty=True) == ""

    # Empty strings should fail when not allowed
    with pytest.raises(InvalidInputError):
        validate_string_input("", allow_empty=False)

    # Too long strings should fail
    with pytest.raises(InvalidInputError):
        validate_string_input("a" * 1001)

    # Strings with null bytes should fail
    with pytest.raises(InvalidInputError):
        validate_string_input("hello\x00world")

    # Strings with invalid control characters should fail (not tab/newline/CR)
    with pytest.raises(InvalidInputError):
        validate_string_input("hello\x01world")


def test_command_sanitization():
    """Test command argument sanitization"""
    # Normal arguments should work
    args = ["python", "-m", "dw.run", "workflow.json"]
    sanitized = sanitize_command_args(args)
    assert len(sanitized) == len(args)
    assert (
        sanitized == args
    )  # With shell=False, arguments pass through after validation

    # Arguments with semicolons should fail
    with pytest.raises(InvalidInputError):
        sanitize_command_args(["rm", "-rf", "; rm -rf /"])

    # Arguments with $ should fail
    with pytest.raises(InvalidInputError):
        sanitize_command_args(["echo", "$(malicious_command)"])

    # Arguments with pipes should fail
    with pytest.raises(InvalidInputError):
        sanitize_command_args(["cat", "/etc/passwd | grep root"])


if __name__ == "__main__":
    pytest.main([__file__])
