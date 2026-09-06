"""The asset library: input media a workflow references by name.

A workflow's media paths resolve against the workflow file's own directory,
which means a workflow that reads anything has to keep that thing beside it -
the reason generated media ends up gitignored inside a source tree. An
'asset:name' reference is rooted at the asset library instead, the way
'prompt:name' is rooted at the prompt library, so the same reference means the
same file from every workflow and neither has to live next to the other.

A reference resolves to a path, not to a value: 'asset:frames/iris.jpg'
becomes the absolute path of that file, and whatever would have loaded a path
written there loads it unchanged.
"""

import contextvars
import logging
import os

from .security import validate_asset_reference, validate_path
from .workspace import ASSETS_SUBDIR, discover_library

logger = logging.getLogger("dw")

# The prefix marking a value as a reference to a stored asset
ASSET_PREFIX = "asset:"

# Set by an entry point from --asset-dir, and inherited by a spawned worker,
# the way DW_PROMPT_DIR is
ASSET_DIR_ENV_VAR = "DW_ASSET_DIR"


# The asset library of the run in progress. A server holds several
# workspaces and each has its own assets, so this cannot be a process-wide
# environment variable there the way the prompt library can - there is one
# prompt library, shared, but assets belong to a workspace. Set per job by
# the worker; unset for the CLI and REPL, which have one workspace per
# process and read the environment below
_active_asset_dir = contextvars.ContextVar("dw_asset_dir", default=None)


def activate_asset_dir(directory):
    """Make an asset library the active one; returns a token for deactivate."""
    return _active_asset_dir.set(directory)


def deactivate_asset_dir(token):
    _active_asset_dir.reset(token)


def get_asset_dir(base_dir=None):
    """The directory 'asset:' references are rooted at.

    A library activated for this run wins outright - that is the server
    telling the worker which workspace's assets this job uses. Otherwise
    discovery mirrors the prompt library's - see workspace.discover_library
    for the shared precedence (DW_ASSET_DIR, then a named workspace, then
    ./assets, then a walk up from base_dir, then the workspace's assets/ as
    the fallback).

    Args:
        base_dir: The workflow file's directory, when one anchors the search
    """
    active = _active_asset_dir.get()
    if active:
        return active

    return discover_library(ASSETS_SUBDIR, ASSET_DIR_ENV_VAR, base_dir)


def is_asset_reference(value):
    """Whether a value references a file in the asset library."""
    return isinstance(value, str) and value.startswith(ASSET_PREFIX)


def resolve_asset_reference(reference, asset_dir=None, base_dir=None):
    """Resolve an 'asset:' reference to the file it names.

    Args:
        reference: The 'asset:name.ext' or 'asset:folder/name.ext' string
        asset_dir: Directory the name is rooted at; defaults to get_asset_dir()
        base_dir: The workflow file's directory, anchoring discovery when no
            asset directory is configured

    Returns:
        The validated absolute path of the asset file

    Raises:
        InvalidInputError: If the name is not a valid asset name
        PathTraversalError: If the name escapes the asset directory
        ValueError: If no file exists under that name
    """
    name = validate_asset_reference(reference.removeprefix(ASSET_PREFIX).strip())
    asset_dir = asset_dir or get_asset_dir(base_dir)
    # Confined to the library: the name is joined onto a directory, so the
    # containment check is what makes a name a name rather than a path
    # allow_create leaves "does not exist" to the check below, which can say
    # what an asset reference is instead of what a path is
    path = validate_path(os.path.join(asset_dir, name), asset_dir)
    if not os.path.isfile(path):
        raise ValueError(
            f"Asset '{name}' not found in {asset_dir} - an 'asset:' reference "
            f"names a file in the asset library, with its extension, like "
            f"'asset:iris.jpg' or 'asset:gyre/frame_1.jpg'"
        )
    logger.debug(f"Resolved {reference} to {path}")
    return path


def fetch_asset(reference, asset_dir=None, base_dir=None):
    """The path an 'asset:' reference names, for whatever loads paths."""
    return resolve_asset_reference(reference, asset_dir, base_dir)
