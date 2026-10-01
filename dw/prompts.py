"""The prompt library: stored prompts a workflow references by name.

A prompt is one JSON file under the prompt directory - its text plus the
metadata the library pages show (description, intended model, tags). A
workflow argument written as 'prompt:name' or 'prompt:folder/name' loads
the file's text at run time, so the prompt is shared by reference rather
than copied into every workflow that uses it.
"""

import json
import logging

from . import references
from .schema import load_schema, validate_data
from .library import (
    PROMPT_DIR_ENV_VAR,
    PROMPTS_KIND,
    LibraryPath,
    LibraryRoot,
    WORKSPACE_ORIGIN,
    library_path_from_env,
)
from .security import validate_prompt_reference
from .workspace import PROMPTS_SUBDIR, discover_library

logger = logging.getLogger("dw")

# The prefix marking a value as a reference to a stored prompt. The name after it
# is rooted at the prompt directory, not the workflow file - prompts are a shared
# library, and the same reference means the same text from every workflow
PROMPT_PREFIX = references.PROMPT

# The prefixes a stored prompt's text may not begin with. Resolved text is
# substituted where the reference stood, so text that itself looks like a
# reference would be resolved again - or worse, expand a step's iterations
RESERVED_TEXT_PREFIXES = (
    references.PREVIOUS_RESULT,
    references.VARIABLE,
    references.CONSTANT,
    references.ASSET,
    references.OUTPUT,
    PROMPT_PREFIX,
)


def get_prompt_dir(base_dir=None):
    """The directory stored prompts are rooted at.

    DW_PROMPT_DIR names it explicitly - the server sets it from --prompt-dir,
    and the spawned worker inherits it. Below that, see
    workspace.discover_library for the shared precedence (a named workspace,
    then ./prompts, then a walk up from base_dir, then the workspace's
    prompts/ as the fallback).

    Read at call time, not import time, so a test or worker sees the current
    value.

    Args:
        base_dir: The workflow file's directory, when one anchors the search
    """
    return discover_library(PROMPTS_SUBDIR, PROMPT_DIR_ENV_VAR, base_dir)


def prompt_library(prompt_dir=None, base_dir=None, exact=False):
    """The prompt library as a `LibraryPath`: the library a save would write
    to first, then the read-only ones an entry point put on the path (the
    prompts a --examples-dir tree brings with it). A name found earlier
    shadows the same name later, the way it does on the workflow search path.

    Args:
        prompt_dir: The first directory; defaults to get_prompt_dir()
        base_dir: The workflow file's directory, anchoring discovery when no
            prompt directory is configured
        exact: Search `prompt_dir` alone, ignoring the pinned path - for a
            caller asking whether one root holds a name
    """
    primary = prompt_dir or get_prompt_dir(base_dir)
    if exact:
        return LibraryPath(PROMPTS_KIND, [LibraryRoot(primary, WORKSPACE_ORIGIN, True)])
    return library_path_from_env(PROMPTS_KIND, primary)


def prompt_search_path(prompt_dir=None, base_dir=None):
    """Every directory a 'prompt:' reference is looked for in, in order."""
    return [root.root for root in prompt_library(prompt_dir, base_dir).roots()]


def resolve_prompt_reference(
    reference, prompt_dir=None, base_dir=None, exact=False, library=None
):
    """Resolve a 'prompt:' reference to the file it names.

    Args:
        reference: The 'prompt:name' or 'prompt:folder/name' string
        prompt_dir: Directory the name is rooted at; defaults to get_prompt_dir()
        base_dir: The workflow file's directory, anchoring discovery when no
            prompt directory is configured
        exact: Resolve against `prompt_dir` alone rather than the whole
            search path (see prompt_library)
        library: The `LibraryPath` to resolve over, for a caller that holds
            the search path itself (the server's); it replaces `prompt_dir`,
            `base_dir` and `exact`

    Returns:
        The validated absolute path of the prompt file

    Raises:
        InvalidInputError: If the name is not a valid prompt name
        ValueError: If no prompt file exists under that name in any directory
            on the search path
    """
    name = validate_prompt_reference(reference.removeprefix(PROMPT_PREFIX).strip())
    library = library or prompt_library(prompt_dir, base_dir, exact)
    found = library.find(name, refuse=True)
    if found:
        return found[0]
    searched = ", ".join(root.root for root in library.roots())
    raise ValueError(
        f"No prompt named '{name}' in {searched} - a prompt reference names "
        f"a .json file under the prompt directory, without the extension"
    )


def load_prompt(path):
    """Read and validate one prompt file.

    Args:
        path: Path of the prompt file, already validated

    Returns:
        The prompt as a dict

    Raises:
        ValueError: If the file is not JSON or does not match the prompt schema
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Prompt file {path} is not valid JSON: {error}") from error

    status, message = validate_data(data, load_schema("prompt"))
    if not status:
        raise ValueError(f"Prompt file {path} is not a valid prompt: {message}")

    return data


def fetch_prompt(reference, prompt_dir=None, base_dir=None):
    """Read the text a 'prompt:' reference names.

    Args:
        reference: The 'prompt:name' or 'prompt:folder/name' string
        prompt_dir: Directory the name is rooted at; defaults to get_prompt_dir()
        base_dir: The workflow file's directory, anchoring discovery when no
            prompt directory is configured

    Returns:
        The prompt file's text field

    Raises:
        ValueError: If the prompt is missing, invalid, or its text is itself
            a reference
    """
    path = resolve_prompt_reference(reference, prompt_dir, base_dir)
    text = load_prompt(path)["text"]

    # Arguments are realized more than once, and iteration expansion scans the
    # realized template - text that begins like a reference would be treated
    # as one on the next pass, so it is data that may not masquerade as syntax
    if text.startswith(RESERVED_TEXT_PREFIXES):
        raise ValueError(
            f"Prompt '{reference}' has text beginning with a reference prefix "
            f"({', '.join(RESERVED_TEXT_PREFIXES)}) - a prompt's text may not "
            f"itself be a reference"
        )

    logger.info(f"Loaded prompt {reference} from {path}")
    return text
