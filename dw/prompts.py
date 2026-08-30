"""The prompt library: stored prompts a workflow references by name.

A prompt is one JSON file under the prompt directory - its text plus the
metadata the library pages show (description, intended model, tags). A
workflow argument written as 'prompt:name' or 'prompt:folder/name' loads
the file's text at run time, so the prompt is shared by reference rather
than copied into every workflow that uses it.
"""

import json
import logging
import os

from .schema import load_schema, validate_data
from .security import validate_prompt_path, validate_prompt_reference

logger = logging.getLogger("dw")

# The prefix marking a value as a reference to a stored prompt. The name after it
# is rooted at the prompt directory, not the workflow file - prompts are a shared
# library, and the same reference means the same text from every workflow
PROMPT_PREFIX = "prompt:"

# The prefixes a stored prompt's text may not begin with. Resolved text is
# substituted where the reference stood, so text that itself looks like a
# reference would be resolved again - or worse, expand a step's iterations
RESERVED_TEXT_PREFIXES = ("previous_result:", "variable:", "constant:", PROMPT_PREFIX)


def get_prompt_dir():
    """The directory stored prompts are rooted at.

    DW_PROMPT_DIR names it explicitly - the server sets it from --prompt-dir,
    and the spawned worker inherits it. Without one it is ./prompts in the
    working directory, the sibling of ./examples the CLI runs from. Read at
    call time, not import time, so a test or worker sees the current value.
    """
    return os.environ.get("DW_PROMPT_DIR") or os.path.abspath("./prompts")


def resolve_prompt_reference(reference, prompt_dir=None):
    """Resolve a 'prompt:' reference to the file it names.

    Args:
        reference: The 'prompt:name' or 'prompt:folder/name' string
        prompt_dir: Directory the name is rooted at; defaults to get_prompt_dir()

    Returns:
        The validated absolute path of the prompt file

    Raises:
        InvalidInputError: If the name is not a valid prompt name
        ValueError: If no prompt file exists under that name
    """
    name = validate_prompt_reference(reference.removeprefix(PROMPT_PREFIX).strip())
    prompt_dir = prompt_dir or get_prompt_dir()
    path = os.path.join(prompt_dir, name + ".json")
    if not os.path.isfile(path):
        raise ValueError(
            f"No prompt named '{name}' in {prompt_dir} - a prompt reference names "
            f"a .json file under the prompt directory, without the extension"
        )
    return validate_prompt_path(path, prompt_dir)


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


def fetch_prompt(reference, prompt_dir=None):
    """Read the text a 'prompt:' reference names.

    Args:
        reference: The 'prompt:name' or 'prompt:folder/name' string
        prompt_dir: Directory the name is rooted at; defaults to get_prompt_dir()

    Returns:
        The prompt file's text field

    Raises:
        ValueError: If the prompt is missing, invalid, or its text is itself
            a reference
    """
    path = resolve_prompt_reference(reference, prompt_dir)
    text = load_prompt(path)["text"]

    # Arguments are realized more than once, and iteration expansion scans the
    # realized template - text that begins like a reference would be treated
    # as one on the next pass, so it is data that may not masquerade as syntax
    if text.startswith(RESERVED_TEXT_PREFIXES):
        raise ValueError(
            f"Prompt '{reference}' has text beginning with a reference prefix - "
            f"a prompt's text may not itself be a reference"
        )

    logger.info(f"Loaded prompt {reference} from {path}")
    return text
