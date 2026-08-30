"""
Tests for security module
"""

import pytest
import os
import tempfile
from dw.security import (
    MAX_FILENAME_LENGTH,
    MAX_JSON_SIZE,
    MAX_PATH_LENGTH,
    validate_path,
    validate_file_extension,
    validate_json_size,
    validate_output_path,
    validate_workflow_path,
    validate_prompt_path,
    validate_prompt_reference,
    validate_url,
    validate_variable_name,
    validate_string_input,
    safe_join_path,
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
        # Use realpath to handle macOS symlinks (/var -> /private/var)
        assert validated == os.path.realpath(os.path.abspath(test_file))

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


class TestValidatePathRejections:
    """Inputs validate_path must refuse outright"""

    def test_an_empty_path_is_rejected(self):
        with pytest.raises(InvalidInputError):
            validate_path("")

    def test_a_null_byte_is_rejected(self):
        # A null byte truncates the path inside the C library, so a name that
        # passes extension checks can open a different file entirely
        with pytest.raises(InvalidInputError):
            validate_path("workflow.json\x00.png")

    def test_an_over_long_path_is_rejected(self):
        with pytest.raises(InvalidInputError, match="Path too long"):
            validate_path("/tmp/" + "a" * MAX_PATH_LENGTH)

    def test_an_over_long_filename_is_rejected(self):
        with pytest.raises(InvalidInputError, match="Filename too long"):
            validate_path("/tmp/" + "a" * (MAX_FILENAME_LENGTH + 1))

    @pytest.mark.parametrize(
        "path", ["/dev/random", "/proc/self/environ", "/sys/kernel/debug", "~/secrets"]
    )
    def test_sensitive_locations_are_rejected(self, path):
        with pytest.raises(PathTraversalError):
            validate_path(path)

    def test_traversal_is_caught_even_when_it_stays_inside_the_base_dir(self):
        # ".." is rejected on the raw string, before normalization can hide it
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(PathTraversalError):
                validate_path(os.path.join(temp_dir, "sub", "..", "ok.json"), temp_dir)

    def test_windows_style_traversal_is_caught(self):
        # Separators are normalized before the pattern check, so a backslash
        # form is refused on Linux too
        with pytest.raises(PathTraversalError):
            validate_path(r"..\..\windows\system32\config\sam")

    def test_a_missing_path_is_rejected_when_creation_is_not_allowed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = os.path.join(temp_dir, "nope.json")
            with pytest.raises(InvalidInputError, match="does not exist"):
                validate_path(missing, allow_create=False)

    def test_a_missing_path_is_allowed_when_creation_is(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = os.path.join(temp_dir, "new", "output.png")
            assert validate_path(missing, allow_create=True).endswith("output.png")

    def test_a_sibling_of_the_base_dir_is_rejected(self):
        # /tmp/base-evil must not pass as inside /tmp/base on a prefix match
        with tempfile.TemporaryDirectory() as parent:
            base = os.path.join(parent, "base")
            sibling = os.path.join(parent, "base-evil")
            os.makedirs(base)
            os.makedirs(sibling)

            with pytest.raises(PathTraversalError):
                validate_path(os.path.join(sibling, "f.json"), base_dir=base)

    def test_the_base_dir_itself_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            assert validate_path(temp_dir, base_dir=temp_dir) == os.path.realpath(
                temp_dir
            )


class TestValidateFileExtension:
    def test_the_extension_check_is_case_insensitive(self):
        assert validate_file_extension("photo.PNG", {".png"}) == "photo.PNG"

    def test_a_disallowed_extension_is_rejected(self):
        with pytest.raises(InvalidInputError, match="File extension not allowed"):
            validate_file_extension("payload.exe", {".png", ".jpg"})

    def test_a_missing_extension_is_rejected(self):
        with pytest.raises(InvalidInputError):
            validate_file_extension("README", {".json"})

    def test_only_the_final_extension_counts(self):
        # A double extension must not slip through on the inner one
        with pytest.raises(InvalidInputError):
            validate_file_extension("workflow.json.sh", {".json"})


class TestValidateWorkflowPath:
    def test_a_json_workflow_validates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "wf.json")
            with open(path, "w") as f:
                f.write("{}")

            assert validate_workflow_path(path) == os.path.realpath(path)

    def test_a_non_json_workflow_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "wf.yaml")
            with open(path, "w") as f:
                f.write("{}")

            with pytest.raises(InvalidInputError):
                validate_workflow_path(path)

    def test_a_missing_workflow_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(InvalidInputError):
                validate_workflow_path(os.path.join(temp_dir, "absent.json"))


class TestValidatePromptReference:
    @pytest.mark.parametrize("name", ["foo", "foo/bar", "foo.bar-1", "a_1/B-2.x"])
    def test_a_plain_or_one_folder_name_validates(self, name):
        assert validate_prompt_reference(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "a/b/c",  # deeper than the library's one folder level
            "../escape",
            "/etc/passwd",
            ".hidden",
            "folder/.hidden",
            "-dashfirst",
        ],
    )
    def test_anything_else_is_rejected(self, name):
        with pytest.raises(InvalidInputError):
            validate_prompt_reference(name)

    def test_an_overlong_name_is_rejected(self):
        with pytest.raises(InvalidInputError):
            validate_prompt_reference("a" * 201)


class TestValidatePromptPath:
    def test_a_json_prompt_inside_the_library_validates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "p.json")
            with open(path, "w") as f:
                f.write("{}")

            assert validate_prompt_path(path, temp_dir) == os.path.realpath(path)

    def test_a_path_outside_the_library_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            outside = os.path.join(temp_dir, "outside")
            library = os.path.join(temp_dir, "prompts")
            os.makedirs(outside)
            os.makedirs(library)
            path = os.path.join(outside, "p.json")
            with open(path, "w") as f:
                f.write("{}")

            with pytest.raises(SecurityError):
                validate_prompt_path(path, library)

    def test_a_non_json_prompt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "p.txt")
            with open(path, "w") as f:
                f.write("text")

            with pytest.raises(InvalidInputError):
                validate_prompt_path(path, temp_dir)


def test_validate_output_path_allows_a_directory_that_does_not_exist_yet():
    # dw.run creates the output directory after validating it
    with tempfile.TemporaryDirectory() as temp_dir:
        target = os.path.join(temp_dir, "outputs")

        assert validate_output_path(target, temp_dir) == os.path.join(
            os.path.realpath(temp_dir), "outputs"
        )


class TestValidateJsonSize:
    def test_a_small_file_passes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "small.json")
            with open(path, "w") as f:
                f.write("{}")

            assert validate_json_size(path) is None

    def test_an_oversized_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "big.json")
            with open(path, "wb") as f:
                # Sparse write - no need to actually produce 50MB of bytes
                f.truncate(MAX_JSON_SIZE + 1)

            with pytest.raises(InvalidInputError, match="JSON file too large"):
                validate_json_size(path)

    def test_a_missing_file_is_reported_as_an_input_error(self):
        # The OSError is translated so callers only handle SecurityError
        with pytest.raises(InvalidInputError, match="Cannot check file size"):
            validate_json_size("/nonexistent/file.json")


class TestSafeJoinPath:
    def test_plain_components_join(self):
        assert safe_join_path("outputs", "run1", "image.png") == os.path.join(
            "outputs", "run1", "image.png"
        )

    def test_empty_components_are_skipped_by_validation(self):
        # validate_string_input would reject "" as empty, so the loop skips it
        # rather than refusing the whole join
        assert safe_join_path("outputs", "", "image.png") == os.path.join(
            "outputs", "image.png"
        )

    @pytest.mark.parametrize(
        "component", ["..", "../etc", "sub/dir", r"sub\dir", "a/../b"]
    )
    def test_a_component_carrying_a_separator_or_traversal_is_rejected(self, component):
        with pytest.raises(InvalidInputError, match="invalid characters"):
            safe_join_path("outputs", component)

    def test_an_over_long_component_is_rejected(self):
        with pytest.raises(InvalidInputError, match="String too long"):
            safe_join_path("outputs", "a" * (MAX_FILENAME_LENGTH + 1))

    def test_a_component_with_a_null_byte_is_rejected(self):
        with pytest.raises(InvalidInputError):
            safe_join_path("outputs", "image\x00.png")


class TestSanitizeCommandArgs:
    def test_non_string_arguments_are_coerced(self):
        assert sanitize_command_args(["--steps", 25, 1.5]) == ["--steps", "25", "1.5"]

    @pytest.mark.parametrize(
        "arg", ["`whoami`", "$HOME", "a|b", "a&b", "a;b", "a>b", "a<b", "a\nb", "a\rb"]
    )
    def test_every_shell_metacharacter_is_rejected(self, arg):
        with pytest.raises(InvalidInputError, match="dangerous characters"):
            sanitize_command_args([arg])

    def test_an_empty_argument_list_is_fine(self):
        assert sanitize_command_args([]) == []


class TestValidateUrl:
    def test_an_empty_url_is_rejected(self):
        with pytest.raises(InvalidInputError, match="URL cannot be empty"):
            validate_url("")

    @pytest.mark.parametrize(
        "url",
        [
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "file:///etc/passwd",
        ],
    )
    def test_dangerous_schemes_are_rejected(self, url):
        with pytest.raises(InvalidInputError):
            validate_url(url)

    def test_a_scheme_without_a_host_is_rejected(self):
        with pytest.raises(InvalidInputError, match="valid domain"):
            validate_url("https://")

    def test_every_security_error_shares_one_base_class(self):
        # Callers catch SecurityError; both subclasses must be caught by it
        assert issubclass(PathTraversalError, SecurityError)
        assert issubclass(InvalidInputError, SecurityError)


if __name__ == "__main__":
    pytest.main([__file__])
