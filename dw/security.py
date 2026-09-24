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
# A stored prompt resolved into a sub-workflow's argument (the H3 Context-IR system
# prompt, about 11k characters) passes through this guard, so it sits above that with
# headroom.
MAX_VARIABLE_VALUE_LENGTH = 20000
MAX_CONSTANT_NAME_LENGTH = 200
DEFAULT_MAX_STRING_LENGTH = 1000
MAX_FILE_PATH_LENGTH = 1000
ALLOWED_JSON_EXTENSIONS = {".json"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
# The most pixels an image a caller names is decoded at. A PNG header can claim
# any size, and Pillow only warns below twice its own MAX_IMAGE_PIXELS, so this
# is checked after Image.open and before anything decodes. An 8K frame is 33M.
# dw_mcp keeps its own copy (it cannot import dw), pinned equal by a test.
MAX_DECODE_PIXELS = 50_000_000

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


# What an untrusted workflow may name as a class. The package allowlist above
# is necessary but not sufficient: every '*_type' value is a class the run
# constructs or calls from_pretrained on with the workflow's own arguments,
# and a class defined inside an allowed package can still do anything in its
# constructor - start a process, open a file for writing. So a class reached
# untrusted must also be one of the kinds below, each of which is a
# checkpoint-loading component or plain data. Each base is (module, name) and
# is matched by identity through sys.modules: a class can only subclass a
# base whose module is already imported, so nothing is imported to check,
# and a package that is not installed simply contributes no bases.
CONSTRUCTIBLE_BASE_CLASSES = (
    # diffusers: models, pipelines, schedulers, quantization configs
    ("diffusers.models.modeling_utils", "ModelMixin"),
    ("diffusers.pipelines.pipeline_utils", "DiffusionPipeline"),
    ("diffusers.modular_pipelines.modular_pipeline", "ModularPipeline"),
    ("diffusers.schedulers.scheduling_utils", "SchedulerMixin"),
    ("diffusers.quantizers.quantization_config", "QuantizationConfigMixin"),
    # transformers: models, tokenizers, processors, quantization configs
    ("transformers.modeling_utils", "PreTrainedModel"),
    ("transformers.tokenization_utils_base", "PreTrainedTokenizerBase"),
    ("transformers.processing_utils", "ProcessorMixin"),
    ("transformers.feature_extraction_utils", "FeatureExtractionMixin"),
    ("transformers.image_processing_base", "ImageProcessingMixin"),
    ("transformers.utils.quantization_config", "QuantizationConfigMixin"),
    # torchao quantization configs (quant_type); sdnq's SDNQConfig is a
    # diffusers QuantizationConfigMixin and is accepted through that
    ("torchao.core.config", "AOBaseConfig"),
)

# Modules whose classes are the libraries' auto-dispatch factories
# (AutoPipelineFor*, transformers' Auto*): they have no base in common with
# what they dispatch to, construct nothing themselves, and load only through
# from_pretrained - whose remote-code arguments are gated separately
# (require_trusted_from_pretrained_arguments)
CONSTRUCTIBLE_FACTORY_MODULES = (
    "diffusers.pipelines.auto_pipeline",
    "transformers.models.auto.",
)

# Reference and condition descriptions (MiniMaxH3ImageReference,
# LTX2VideoCondition, ...) are dataclasses with no base of their own. A
# dataclass's generated __init__ only assigns its fields, so one defined in
# these packages is plain data - unless it adds a __post_init__, which runs
# arbitrary code; those are accepted only once reviewed and listed below
CONSTRUCTIBLE_DATACLASS_PACKAGES = (
    "diffusers.pipelines.",
    "diffusers.modular_pipelines.",
)
REVIEWED_POST_INIT_DATACLASSES = (
    # __post_init__ only defaults fps to a module constant
    "diffusers.modular_pipelines.minimax_h3.references.MiniMaxH3VideoReference",
)

# attn_processor_type names an attention processor, constructed with no
# arguments. They share no base class; diffusers defines them in its models
# package, named '...Processor' (or '...Processor2_0'). Their constructors
# record configuration, except that one (LTX2VideoVaeNeighborhoodNattenProcessor)
# fetches a kernel from the Hub - from a repo diffusers hardcodes, never one
# its arguments name, and only when DIFFUSERS_DISABLE_REMOTE_CODE is unset
# (see dw/kernel_availability.py)
_ATTENTION_PROCESSOR_NAME = re.compile(r"Processor(\d+_\d+)?$")
_ATTENTION_PROCESSOR_PACKAGE = "diffusers.models."


def _constructible_bases():
    import sys

    bases = []
    for module_name, class_name in CONSTRUCTIBLE_BASE_CLASSES:
        module = sys.modules.get(module_name)
        base = getattr(module, class_name, None) if module is not None else None
        if isinstance(base, type):
            bases.append(base)
    return tuple(bases)


def is_constructible_class(cls) -> bool:
    """Whether an untrusted workflow may name `cls` as a type to construct.

    True for a subclass of CONSTRUCTIBLE_BASE_CLASSES, a class defined in
    CONSTRUCTIBLE_FACTORY_MODULES, a dataclass defined in
    CONSTRUCTIBLE_DATACLASS_PACKAGES with no unreviewed __post_init__, or a
    diffusers attention processor. See the comments on each for why.
    """
    import dataclasses

    if not isinstance(cls, type):
        return False
    # Real inheritance only: issubclass would also honour an ABC's register()
    # and __subclasshook__, and AOBaseConfig is an ABC
    bases = _constructible_bases()
    if any(base in cls.__mro__ for base in bases):
        return True
    module = getattr(cls, "__module__", None)
    if not isinstance(module, str):
        return False
    qualified = f"{module}.{cls.__qualname__}"
    if any(
        module == prefix.rstrip(".") or module.startswith(prefix)
        for prefix in CONSTRUCTIBLE_FACTORY_MODULES
    ):
        return True
    if dataclasses.is_dataclass(cls) and module.startswith(
        CONSTRUCTIBLE_DATACLASS_PACKAGES
    ):
        return (
            getattr(cls, "__post_init__", None) is None
            or qualified in REVIEWED_POST_INIT_DATACLASSES
        )
    return module.startswith(_ATTENTION_PROCESSOR_PACKAGE) and bool(
        _ATTENTION_PROCESSOR_NAME.search(cls.__name__)
    )


def require_constructible_class(name: str, cls, what: str) -> None:
    """Refuse a class an untrusted workflow may not construct.

    Raises:
        UntrustedWorkflowError: If untrusted and `cls` is not
            is_constructible_class
    """
    if workflows_are_trusted() or is_constructible_class(cls):
        return
    raise UntrustedWorkflowError(
        f"Refusing to load {what} '{name}': an untrusted workflow may only "
        f"name a model, pipeline, scheduler, tokenizer, processor, "
        f"quantization config, auto-pipeline/auto-model factory, attention "
        f"processor or reference/condition dataclass from the diffusers "
        f"ecosystem, and '{getattr(cls, '__module__', '?')}."
        f"{getattr(cls, '__qualname__', '?')}' is none of these - its "
        f"constructor could run anything with the workflow's arguments. Pass "
        f"--trust-workflows if you trust this workflow's source."
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


def contained(path: Union[str, Path], root: Union[str, Path]) -> bool:
    """Whether path, symlinks resolved, lies inside root, symlinks resolved.

    For a listing that walks a directory with os.walk: a file symlink shows
    up among the names like any other file, and naming it would carry the
    target's name, size or content out of the root. Both sides are resolved,
    so a root that is itself a link (a data volume) still contains its own
    files, while a link inside it pointing elsewhere does not.
    """
    real_root = os.path.realpath(root)
    real = os.path.realpath(path)
    return real == real_root or real.startswith(real_root.rstrip(os.sep) + os.sep)


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

    # urllib.parse and urllib3 disagree on where a backslash ends the host:
    # 'http://169.254.169.254\@example.com/' is example.com to the check and
    # 169.254.169.254 to the request. No valid URL needs one, so refuse it
    # rather than pick a parser
    if "\\" in url:
        raise InvalidInputError(
            f"Invalid URL: '{url}' contains a backslash, which parsers read "
            f"differently - percent-encode it as %5C if it belongs in the path"
        )

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
PROMPT_REFERENCE_CHARACTERS = r"[\w.-]"
PROMPT_REFERENCE_PATTERN = r"^[\w][\w.-]*(/[\w][\w.-]*)?\Z"
MAX_PROMPT_REFERENCE_LENGTH = 200


def _name_fault(name: str, allowed: str) -> str:
    """Which part of `name` the pattern objected to, as a clause, or ''.

    A name that fails one of these patterns used to be echoed back beside a
    description of a *valid* name and nothing else, so a caller who had
    passed a name the server itself produced had to bisect it character by
    character to find the objection (#162). Naming the character turns that
    into a one-line fix.
    """
    if not allowed:
        return ""
    for position, character in enumerate(name):
        if character == "/":
            continue
        if not re.fullmatch(allowed, character):
            return (
                f" - {character!r} (position {position}) is not a character "
                f"this kind of name may contain"
            )
    for position, segment in enumerate(name.split("/")):
        if not segment:
            return f" - segment {position + 1} is empty"
        if not re.fullmatch(r"\w", segment[0]):
            return (
                f" - segment {segment!r} starts with {segment[0]!r}, and every "
                f"segment must start with a letter, digit or underscore"
            )
    return ""


def _validate_name(
    name: str,
    pattern: str,
    max_length: int,
    what: str,
    hint: str,
    allowed: str = "",
) -> str:
    """Shared body of the reference/name validators below: an empty check, a
    length check, then the pattern - length before pattern so a name that
    fails both reports the shorter, cheaper-to-fix complaint first, matching
    what each validator already reported on its own.

    Args:
        name: The value to validate
        pattern: Regex the name must fully match
        max_length: Longest allowed length
        what: Short label for the messages ('Prompt name', 'Asset name', ...)
        hint: The rest of the "invalid" message, describing what a valid one
            looks like

    Raises:
        InvalidInputError: If name is invalid
    """
    if not name:
        raise InvalidInputError(f"{what} cannot be empty")

    if len(name) > max_length:
        raise InvalidInputError(f"{what} too long: {len(name)} > {max_length}")

    if not re.match(pattern, name):
        raise InvalidInputError(
            f"Invalid {what.lower()}: {name}{_name_fault(name, allowed)} - {hint}"
        )

    return name


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
    return _validate_name(
        name,
        PROMPT_REFERENCE_PATTERN,
        MAX_PROMPT_REFERENCE_LENGTH,
        "Prompt name",
        "a prompt is named by its file under the prompt directory, at most "
        "one folder deep, like 'scenic_landscape' or 'minimax/fox_dawn'",
        allowed=PROMPT_REFERENCE_CHARACTERS,
    )


# A stored asset's name: a file name with its extension, optionally under
# folders. Each segment starts with a word character, which precludes '..',
# hidden files and absolute paths; the depth cap keeps a name a name. A prompt
# is named without its extension and lives at most one folder deep - an asset
# carries its extension, because which file it is depends on it, and media
# libraries nest deeper than prompt libraries do
# '@' for the same reason OUTPUT_REFERENCE_PATTERN carries it: keeping a
# `for_each` member's file as an asset defaults its name to that file's
# base name, which carries the '@' the engine wrote (#162)
ASSET_REFERENCE_CHARACTERS = r"[\w.@-]"
ASSET_REFERENCE_PATTERN = r"^[\w][\w.@-]*(/[\w][\w.@-]*){0,4}\Z"
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
    return _validate_name(
        name,
        ASSET_REFERENCE_PATTERN,
        MAX_ASSET_REFERENCE_LENGTH,
        "Asset name",
        "an asset is named by its file under the asset directory, with its "
        "extension and at most four folders deep, like 'iris.jpg' or "
        "'gyre/frames/iris.jpg'",
        allowed=ASSET_REFERENCE_CHARACTERS,
    )


# A generated output's name: the workflow's identity, the run, and the file -
# deeper than an asset name because the identity itself can nest, and the run
# id is a segment of its own.
#
# '@' is here because the engine writes it: a `for_each` member is named
# '<group>@<entry>' and its files carry that in their base name, so a whole
# class of files the server named could not be named back to it (#162). It
# is safe in a path - not a separator, not '..', and containment is still
# checked by the validate_path that joins the name onto the output root -
# and a name still may not *start* with it.
OUTPUT_REFERENCE_CHARACTERS = r"[\w.@-]"
OUTPUT_REFERENCE_PATTERN = r"^[\w][\w.@-]*(/[\w][\w.@-]*){1,6}\Z"
MAX_OUTPUT_REFERENCE_LENGTH = 500


def validate_output_reference(name: str) -> str:
    """
    Validate the name an 'output:' reference points at.

    The name is joined onto the output directory to find the file, so it is
    checked before anything touches the filesystem. Containment is checked
    separately, by the validate_path call that joins it - after any 'latest'
    segment has been expanded, so what is checked is the real path.

    Args:
        name: Output name to validate

    Returns:
        The validated name

    Raises:
        InvalidInputError: If name is invalid
    """
    return _validate_name(
        name,
        OUTPUT_REFERENCE_PATTERN,
        MAX_OUTPUT_REFERENCE_LENGTH,
        "Output name",
        "an output is named by the workflow that made it, the run, and the "
        "file, like 'ltx2/Gyre/latest/Gyre-still.0-0.0.png'",
        allowed=OUTPUT_REFERENCE_CHARACTERS,
    )


# A step's result 'subfolder': a relative path under the run directory. The
# segment rule is OUTPUT_REFERENCE_PATTERN's, so every subfolder the engine
# writes is one a later workflow can name with 'output:'. It also refuses a
# backslash, which DANGEROUS_PATTERNS does not - '"final\\x"' would be one
# directory on POSIX and two on Windows
SUBFOLDER_PATTERN = r"^[\w][\w.-]*(/[\w][\w.-]*)*\Z"
MAX_SUBFOLDER_LENGTH = 200


def validate_subfolder(name: str) -> str:
    """
    Validate the shape of a result 'subfolder'.

    Containment is checked separately, by the validate_output_path call
    that joins it onto the run directory.

    Args:
        name: The subfolder as written in the workflow

    Returns:
        The validated name

    Raises:
        InvalidInputError: If the name is not a valid subfolder
    """
    return _validate_name(
        name,
        SUBFOLDER_PATTERN,
        MAX_SUBFOLDER_LENGTH,
        "Subfolder",
        "a subfolder is one or more path segments under the run directory, "
        "each starting with a letter, digit or underscore, like 'final' or "
        "'shots/act-1'",
    )


def validate_file_base_name(name: str) -> str:
    """
    Validate a result 'file_base_name': a name, never a path.

    A separator here used to pass validation and then fail at open() because
    the directory did not exist. Placement is what 'subfolder' is for.

    Raises:
        InvalidInputError: If the name carries a path separator
    """
    if "/" in name or "\\" in name:
        raise InvalidInputError(
            f"Invalid file_base_name: {name} - a file_base_name is a name, not "
            f"a path; to write into a subfolder of the run directory set "
            f"'subfolder' on the result instead"
        )
    return name


def validate_content_type(value: str) -> str:
    """
    Validate a result 'content_type': a MIME type, never a bare word.

    A bare word like 'video' passed here clean and then failed deep inside
    a writer, in a traceback naming neither the field nor the value - the
    writer's own dispatch matches 'video' as a startswith prefix of
    'video/mp4' and takes that branch anyway, then fails for lack of a real
    extension to write. This only checks the shape; which MIME types this
    engine actually has a writer for is the caller's business.

    Raises:
        InvalidInputError: If the value is not a string, or not shaped like
            a MIME type
    """
    if not isinstance(value, str):
        raise InvalidInputError(f"Invalid content_type: {value!r} - expected a string")
    if value.count("/") != 1 or "" in value.split("/"):
        raise InvalidInputError(
            f"Invalid content_type: {value!r} - content_type wants a MIME "
            f"type like 'video/mp4' or 'image/png', not a bare word"
        )
    return value


# A workspace's name: one path segment, starting with a word character, so
# '..', hidden names and anything with a separator in it are all excluded
# before the name is joined onto the workspace root
WORKSPACE_NAME_PATTERN = r"^[\w][\w.-]*\Z"
MAX_WORKSPACE_NAME_LENGTH = 100


def validate_workspace_name(name: str) -> str:
    """
    Validate a workspace name.

    Args:
        name: Workspace name to validate

    Returns:
        The validated name

    Raises:
        InvalidInputError: If the name is not one a workspace can take
    """
    from .workspace import RESERVED_WORKSPACE_NAMES

    _validate_name(
        name,
        WORKSPACE_NAME_PATTERN,
        MAX_WORKSPACE_NAME_LENGTH,
        "Workspace name",
        "a workspace is one folder under the workspace root, named with "
        "letters, numbers, dot, dash or underscore",
    )

    if name in RESERVED_WORKSPACE_NAMES:
        raise InvalidInputError(
            f"'{name}' is one of the workspace root's own folders "
            f"({', '.join(RESERVED_WORKSPACE_NAMES)}) and cannot name a workspace"
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
