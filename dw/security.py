"""
Security utilities for input validation and safe file operations.
"""

import os
import re
import logging
from pathlib import Path
from urllib.parse import urlparse
from typing import Union, List, Optional

logger = logging.getLogger("dw")

# Security constants
MAX_PATH_LENGTH = 4096
MAX_FILENAME_LENGTH = 255
MAX_JSON_SIZE = 50 * 1024 * 1024  # 50MB
MAX_VARIABLE_NAME_LENGTH = 100
MAX_VARIABLE_VALUE_LENGTH = 10000
MAX_CONSTANT_NAME_LENGTH = 200
DEFAULT_MAX_STRING_LENGTH = 1000
MAX_FILE_PATH_LENGTH = 1000
ALLOWED_JSON_EXTENSIONS = {".json"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}

# Dangerous path patterns (handle both Unix and Windows paths)
DANGEROUS_PATTERNS = [
    r"\.\.",  # Parent directory traversal (.. anywhere)
    r"~[\\/]",  # Home directory expansion
    r"^/dev/",  # Device files (Unix)
    r"^/proc/",  # Process files (Unix)
    r"^/sys/",  # System files (Unix)
]


class SecurityError(Exception):
    """Base exception for security-related errors."""

    pass


class PathTraversalError(SecurityError):
    """Raised when path traversal attempt is detected."""

    pass


class InvalidInputError(SecurityError):
    """Raised when input validation fails."""

    pass


class UntrustedWorkflowError(SecurityError):
    """Raised when an untrusted workflow reaches the code-execution surface.

    Loading a workflow JSON file is not a passive data-load: pre_load_modules,
    a dotted '*_type'/'*_dtype'/'dtype'/'config_type' value, and a 'constant:'
    reference all run importlib.import_module() on a name the file supplies,
    which executes that module's top-level code. See docs/SECURITY.md's Trust
    model section. Untrusted here means "not explicitly vouched for by
    --trust-workflows" - the default - not "known malicious".
    """

    pass


# The environment variable a parent process sets to hand its --trust-workflows
# choice down to a spawned worker subprocess - multiprocessing's 'spawn' start
# method launches a fresh interpreter that inherits os.environ, the same way
# DW_PROMPT_DIR reaches the worker (see dw/serve.py, dw/repl.py)
TRUST_WORKFLOWS_ENV_VAR = "DW_TRUST_WORKFLOWS"

# Dotted type/pre_load_modules references that resolve under one of these
# top-level packages are treated as part of the diffusers ecosystem the tool
# already assumes, and are allowed even for an untrusted workflow - everything
# else requires --trust-workflows. This is the dependency set pyproject.toml
# declares for exactly this purpose ("Quantization backends - config_objects.py
# loads them dynamically"), plus the framework packages workflows target
# directly, plus 'dw' itself - a workflow's own community_pipelines component
# lives in this package, not a third party one
TRUSTED_TOP_LEVEL_PACKAGES = (
    "diffusers",
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "peft",
    "sdnq",
    "torchao",
    "optimum",  # optimum-quanto
    "gguf",
    "bitsandbytes",
    "dw",
)


def set_trust_workflows(trusted: bool) -> None:
    """Record the process-wide --trust-workflows choice.

    Called once at CLI/server startup, before any workflow loads. A spawned
    worker subprocess reads the same choice back from the environment
    variable this sets, rather than needing it passed as an argument.
    """
    os.environ[TRUST_WORKFLOWS_ENV_VAR] = "1" if trusted else "0"


def workflows_are_trusted() -> bool:
    """Whether the process has been told to trust workflow files fully.

    Defaults to untrusted (False) when nothing has set the flag, which is
    the secure default for any code path that loads a workflow without
    going through the CLI/server startup that calls set_trust_workflows().
    """
    return os.environ.get(TRUST_WORKFLOWS_ENV_VAR) == "1"


def _top_level_package(dotted_name: str) -> str:
    return dotted_name.split(".", 1)[0]


def require_trusted_dotted_name(dotted_name: str, what: str) -> None:
    """Refuse a dotted-name import outside the diffusers ecosystem unless
    the workflow is trusted.

    Args:
        dotted_name: The module.path.Name a workflow supplied
        what: Short phrase naming what kind of value this was ('a *_type
            value', 'a config_type value', ...), for the error message

    Raises:
        UntrustedWorkflowError: If untrusted and the name is outside
            TRUSTED_TOP_LEVEL_PACKAGES
    """
    if workflows_are_trusted():
        return

    top_level = _top_level_package(dotted_name)
    if top_level in TRUSTED_TOP_LEVEL_PACKAGES:
        return

    raise UntrustedWorkflowError(
        f"Refusing to load {what} '{dotted_name}': it imports the "
        f"'{top_level}' module, which is outside the ecosystem "
        f"({', '.join(TRUSTED_TOP_LEVEL_PACKAGES)}) this workflow is "
        f"allowed to reach untrusted. Loading a workflow JSON file can "
        f"execute arbitrary Python - see docs/SECURITY.md. Pass "
        f"--trust-workflows if you trust this workflow's source."
    )


def require_trusted_pre_load_modules(module_names) -> None:
    """Refuse a pre_load_modules entry outside the diffusers ecosystem
    unless the workflow is trusted.

    pre_load_modules exists to run a module's import-time registration
    side effects - sdnq registering its quantization method with diffusers
    is the pattern the bundled example workflows use - so an in-ecosystem
    module name is allowed the same way an in-ecosystem dotted type
    reference is; anything else requires trust.

    Raises:
        UntrustedWorkflowError: If untrusted and any name is outside
            TRUSTED_TOP_LEVEL_PACKAGES
    """
    for module_name in module_names or []:
        require_trusted_dotted_name(module_name, "pre_load_modules entry")


REMOTE_CODE_ARGUMENTS = ("trust_remote_code", "custom_pipeline")


def require_trusted_from_pretrained_arguments(arguments, what: str) -> None:
    """Refuse from_pretrained arguments that make diffusers/transformers
    download and execute Python from the Hub unless the workflow is trusted.

    `trust_remote_code: true` runs a repo's own modeling code; `custom_pipeline`
    fetches a pipeline module from the Hub (or a local path) and imports it.
    Both are arbitrary code chosen by the workflow file, reached without any
    importlib call of ours - so the importlib gate alone would leave them open.

    Args:
        arguments: The from_pretrained_arguments block
        what: The component being loaded, for the error message

    Raises:
        UntrustedWorkflowError: If untrusted and either argument is set
    """
    if workflows_are_trusted() or not arguments:
        return
    for key in REMOTE_CODE_ARGUMENTS:
        if arguments.get(key):
            raise UntrustedWorkflowError(
                f"Refusing to load {what}: its from_pretrained_arguments set "
                f"'{key}', which downloads and executes Python from the model "
                f"repository. Loading a workflow JSON file can execute arbitrary "
                f"Python - see docs/SECURITY.md. Pass --trust-workflows if you "
                f"trust this workflow's source."
            )


def validate_path(
    path: Union[str, Path], base_dir: Optional[str] = None, allow_create: bool = True
) -> str:
    """
    Validate and sanitize file paths to prevent path traversal attacks.

    Args:
        path: The path to validate
        base_dir: Optional base directory to restrict access to
        allow_create: Whether to allow creation of non-existent paths

    Returns:
        Absolute, sanitized path

    Raises:
        PathTraversalError: If path contains dangerous patterns
        InvalidInputError: If path is invalid or too long
    """
    if not path:
        raise InvalidInputError("Path cannot be empty")

    path_str = str(path)

    # Check path length
    if len(path_str) > MAX_PATH_LENGTH:
        raise InvalidInputError(f"Path too long: {len(path_str)} > {MAX_PATH_LENGTH}")

    # Check for null bytes
    if "\x00" in path_str:
        raise InvalidInputError("Path contains null bytes")

    # Normalize path separators for consistent checking across platforms
    normalized_path = path_str.replace("\\", "/")

    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, normalized_path, re.IGNORECASE):
            raise PathTraversalError(
                f"Path contains dangerous pattern matching {pattern}"
            )

    # Convert to absolute path and resolve. normpath after realpath is a
    # no-op for correctness but keeps the value in the normalized shape
    # security scanners recognize as sanitized
    try:
        abs_path = os.path.abspath(os.path.expanduser(path_str))
        resolved_path = os.path.normpath(os.path.realpath(abs_path))
    except (OSError, ValueError) as e:
        raise InvalidInputError(f"Invalid path: {e}")

    # Check if path is within base directory if specified
    if base_dir:
        try:
            base_abs = os.path.abspath(os.path.expanduser(base_dir))
            base_real = os.path.normpath(os.path.realpath(base_abs))
        except (OSError, ValueError) as e:
            raise InvalidInputError(f"Invalid base directory: {e}")

        # Containment on the fully resolved paths: equal to the base, or a
        # descendant of it. The os.sep suffix stops a sibling with the base
        # as a name prefix (/base-evil vs /base); realpath above already
        # collapsed symlinks and '..' on both sides, which also makes a
        # different-drive Windows path fail the prefix test
        if resolved_path != base_real and not resolved_path.startswith(
            base_real + os.sep
        ):
            raise PathTraversalError(f"Path outside allowed directory: {resolved_path}")

    # Check filename length
    filename = os.path.basename(resolved_path)
    if len(filename) > MAX_FILENAME_LENGTH:
        raise InvalidInputError(
            f"Filename too long: {len(filename)} > {MAX_FILENAME_LENGTH}"
        )

    # Check if path exists or creation is allowed
    if not os.path.exists(resolved_path) and not allow_create:
        raise InvalidInputError(f"Path does not exist: {resolved_path}")

    logger.debug(f"Validated path: {path_str} -> {resolved_path}")
    return resolved_path


def validate_file_extension(path: str, allowed_extensions: set) -> str:
    """
    Validate file extension against allowed list.

    Args:
        path: File path to validate
        allowed_extensions: Set of allowed extensions (with dots)

    Returns:
        The validated path

    Raises:
        InvalidInputError: If extension is not allowed
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in allowed_extensions:
        raise InvalidInputError(f"File extension not allowed: {ext}")
    return path


def validate_workflow_path(path: str, workflow_dir: str = None) -> str:
    """Validate workflow file paths."""
    validated = validate_path(path, workflow_dir, allow_create=False)
    return validate_file_extension(validated, ALLOWED_JSON_EXTENSIONS)


def validate_prompt_path(path: str, prompt_dir: str) -> str:
    """Validate stored prompt file paths, confined to the prompt directory."""
    validated = validate_path(path, prompt_dir, allow_create=False)
    return validate_file_extension(validated, ALLOWED_JSON_EXTENSIONS)


def validate_output_path(path: str, output_dir: str) -> str:
    """Validate output file paths."""
    return validate_path(path, output_dir, allow_create=True)


def validate_url(url: str) -> str:
    """
    Validate URL format and scheme.

    Args:
        url: URL to validate

    Returns:
        The validated URL

    Raises:
        InvalidInputError: If URL is invalid or uses dangerous scheme
    """
    if not url:
        raise InvalidInputError("URL cannot be empty")

    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise InvalidInputError(f"URL scheme not allowed: {parsed.scheme}")
        if not parsed.netloc:
            raise InvalidInputError("URL must have a valid domain")
        return url
    except Exception as e:
        raise InvalidInputError(f"Invalid URL: {e}")


def sanitize_command_args(args: List[str]) -> List[str]:
    """
    Sanitize command arguments for subprocess execution with shell=False.

    When using subprocess with a list of arguments and shell=False, Python
    handles argument separation safely without shell interpretation. This
    function validates that arguments don't contain shell metacharacters
    that could be dangerous if shell=True were accidentally used.

    Args:
        args: List of command arguments

    Returns:
        List of validated arguments (no modification needed for shell=False)

    Raises:
        InvalidInputError: If arguments contain dangerous content
    """
    sanitized = []

    for arg in args:
        if not isinstance(arg, str):
            arg = str(arg)

        # Check for dangerous characters that would be problematic with shell=True
        # Even though we use shell=False, this prevents accidental security issues
        if any(char in arg for char in ["`", "$", "|", "&", ";", ">", "<", "\n", "\r"]):
            raise InvalidInputError(f"Argument contains dangerous characters: {arg}")

        # With shell=False, we don't need shlex.quote() - Python handles it safely
        # Just validate and pass through
        sanitized.append(arg)

    return sanitized


def validate_variable_name(name: str) -> str:
    """
    Validate variable names to prevent injection attacks.

    Args:
        name: Variable name to validate

    Returns:
        The validated variable name

    Raises:
        InvalidInputError: If name is invalid
    """
    if not name:
        raise InvalidInputError("Variable name cannot be empty")

    # Allow only alphanumeric characters, underscores, and hyphens
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_-]*\Z", name):
        raise InvalidInputError(f"Invalid variable name: {name}")

    if len(name) > MAX_VARIABLE_NAME_LENGTH:
        raise InvalidInputError(
            f"Variable name too long: {len(name)} > {MAX_VARIABLE_NAME_LENGTH}"
        )

    return name


# A stored prompt's name: a file name, optionally under one folder. Each
# segment starts with a word character, which precludes '..', hidden files,
# and absolute paths without a second scan. Anchored with \Z, not $ - $ also
# matches before a trailing newline, which would admit names no listing can
# round-trip (the same reason the variable and constant patterns use \Z)
PROMPT_REFERENCE_PATTERN = r"^[\w][\w.-]*(/[\w][\w.-]*)?\Z"
MAX_PROMPT_REFERENCE_LENGTH = 200


def validate_prompt_reference(name: str) -> str:
    """
    Validate the name a 'prompt:' reference points at.

    The name is joined onto the prompt directory to find the file, so it is
    checked before anything touches the filesystem - a plain name or one
    folder deep, matching how the prompt library is organized.

    Args:
        name: Prompt name to validate

    Returns:
        The validated name

    Raises:
        InvalidInputError: If name is invalid
    """
    if not name:
        raise InvalidInputError("Prompt name cannot be empty")

    if not re.match(PROMPT_REFERENCE_PATTERN, name):
        raise InvalidInputError(
            f"Invalid prompt name: {name} - a prompt is named by its file under "
            f"the prompt directory, at most one folder deep, like "
            f"'scenic_landscape' or 'minimax/fox_dawn'"
        )

    if len(name) > MAX_PROMPT_REFERENCE_LENGTH:
        raise InvalidInputError(
            f"Prompt name too long: {len(name)} > {MAX_PROMPT_REFERENCE_LENGTH}"
        )

    return name


# A stored asset's name: a file name with its extension, optionally under
# folders. Each segment starts with a word character, which precludes '..',
# hidden files and absolute paths; the depth cap keeps a name a name. A prompt
# is named without its extension and lives at most one folder deep - an asset
# carries its extension, because which file it is depends on it, and media
# libraries nest deeper than prompt libraries do
ASSET_REFERENCE_PATTERN = r"^[\w][\w.-]*(/[\w][\w.-]*){0,4}\Z"
MAX_ASSET_REFERENCE_LENGTH = 400


def validate_asset_reference(name: str) -> str:
    """
    Validate the name an 'asset:' reference points at.

    The name is joined onto the asset directory to find the file, so it is
    checked before anything touches the filesystem. Containment in the
    library is checked separately, by the validate_path call that joins it.

    Args:
        name: Asset name to validate

    Returns:
        The validated name

    Raises:
        InvalidInputError: If name is invalid
    """
    if not name:
        raise InvalidInputError("Asset name cannot be empty")

    if not re.match(ASSET_REFERENCE_PATTERN, name):
        raise InvalidInputError(
            f"Invalid asset name: {name} - an asset is named by its file under "
            f"the asset directory, with its extension and at most four folders "
            f"deep, like 'iris.jpg' or 'gyre/frames/iris.jpg'"
        )

    if len(name) > MAX_ASSET_REFERENCE_LENGTH:
        raise InvalidInputError(
            f"Asset name too long: {len(name)} > {MAX_ASSET_REFERENCE_LENGTH}"
        )

    return name


# A dotted python name: identifiers separated by dots, and nothing else
CONSTANT_NAME_PATTERN = r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*\Z"


def validate_constant_name(name: str) -> str:
    """
    Validate the dotted name of a python constant a workflow references.

    Resolving a name imports the module it lives in, which runs that module's
    code, so the name is checked before anything is imported - identifiers and
    dots only, no relative paths and nothing to evaluate.

    Args:
        name: Dotted name to validate

    Returns:
        The validated name

    Raises:
        InvalidInputError: If name is invalid
    """
    if not name:
        raise InvalidInputError("Constant name cannot be empty")

    if not re.match(CONSTANT_NAME_PATTERN, name):
        raise InvalidInputError(
            f"Invalid constant name: {name} - a constant is named by its module "
            f"and the attribute to read from it, like "
            f"'diffusers.pipelines.ltx2.utils.DISTILLED_SIGMA_VALUES'"
        )

    if len(name) > MAX_CONSTANT_NAME_LENGTH:
        raise InvalidInputError(
            f"Constant name too long: {len(name)} > {MAX_CONSTANT_NAME_LENGTH}"
        )

    return name


# A git commit: hex digits only, short (7) to full (40) SHA-1 length. Used
# to pin the diffusers updater's "git+URL@<commit>" install target - a git
# ref grammar accepts far more than a hash would (branch names, shell-
# adjacent punctuation), so this is deliberately narrower than "anything
# git allows"
COMMIT_HASH_PATTERN = r"^[0-9a-fA-F]{7,40}\Z"


def validate_commit_hash(commit: str) -> str:
    """
    Validate a git commit hash before it is interpolated into a
    'git+<url>@<commit>' pip install target.

    Args:
        commit: Commit hash to validate

    Returns:
        The validated commit hash

    Raises:
        InvalidInputError: If the hash is not 7-40 hex characters
    """
    if not commit:
        raise InvalidInputError("Commit hash cannot be empty")

    if not re.match(COMMIT_HASH_PATTERN, commit):
        raise InvalidInputError(
            f"Invalid commit hash: {commit} - expected 7 to 40 hex characters"
        )

    return commit


def validate_json_size(file_path: str) -> None:
    """
    Validate JSON file size before loading.

    Args:
        file_path: Path to JSON file

    Raises:
        InvalidInputError: If file is too large
    """
    try:
        size = os.path.getsize(file_path)
        if size > MAX_JSON_SIZE:
            raise InvalidInputError(f"JSON file too large: {size} > {MAX_JSON_SIZE}")
    except OSError as e:
        raise InvalidInputError(f"Cannot check file size: {e}")


def validate_string_input(
    value: str, max_length: int = DEFAULT_MAX_STRING_LENGTH, allow_empty: bool = False
) -> str:
    """
    Validate string input for basic safety.

    Args:
        value: String to validate
        max_length: Maximum allowed length
        allow_empty: Whether empty strings are allowed

    Returns:
        The validated string

    Raises:
        InvalidInputError: If string is invalid
    """
    if not allow_empty and not value:
        raise InvalidInputError("String cannot be empty")

    if len(value) > max_length:
        raise InvalidInputError(f"String too long: {len(value)} > {max_length}")

    # Check for null bytes and control characters
    if "\x00" in value or any(ord(c) < 32 for c in value if c not in "\t\n\r"):
        raise InvalidInputError("String contains invalid characters")

    return value


def safe_join_path(*parts: str) -> str:
    """
    Safely join path components with validation.

    Args:
        *parts: Path components to join

    Returns:
        Safely joined path

    Raises:
        InvalidInputError: If any component is invalid
    """
    # Validate each component
    for part in parts:
        if not part:
            continue
        validate_string_input(part, MAX_FILENAME_LENGTH)
        if ".." in part or "/" in part or "\\" in part:
            raise InvalidInputError(
                f"Path component contains invalid characters: {part}"
            )

    return os.path.join(*parts)
