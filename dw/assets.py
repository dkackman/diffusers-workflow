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

import logging
import os

from .security import validate_asset_reference, validate_path
from .workspace import resolve_workspace

logger = logging.getLogger("dw")

# The prefix marking a value as a reference to a stored asset
ASSET_PREFIX = "asset:"

# Set by an entry point from --asset-dir, and inherited by a spawned worker,
# the way DW_PROMPT_DIR is
ASSET_DIR_ENV_VAR = "DW_ASSET_DIR"


def get_asset_dir(base_dir=None):
    """The directory 'asset:' references are rooted at.

    Discovery mirrors the prompt library's, for the same reasons:
    DW_ASSET_DIR names it outright; then a workspace someone named, whose
    assets/ is the library by definition; then assets/ in the working
    directory when it exists; then the walk from the workflow file's
    directory up toward the filesystem root, which is how a workflow in a
    tree reaches the library that tree keeps; and finally the workspace's
    assets/.

    Args:
        base_dir: The workflow file's directory, when one anchors the search
    """
    explicit = os.environ.get(ASSET_DIR_ENV_VAR)
    if explicit:
        return explicit

    workspace = resolve_workspace()
    if workspace.is_explicit:
        return workspace.assets

    working_directory_library = os.path.abspath("./assets")
    if os.path.isdir(working_directory_library):
        return working_directory_library
    if base_dir:
        current = os.path.abspath(base_dir)
        while True:
            candidate = os.path.join(current, "assets")
            if os.path.isdir(candidate):
                return candidate
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
    return workspace.assets


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


def resolve_asset_values(value, base_dir=None):
    """Replace any 'asset:' reference in a value with the path it names.

    A list is walked, because an 'image' argument may be a list of them and
    the key conventions hand the whole list to the loader at once - by then
    it is too late for a reference to be recognized. Dictionaries are left
    alone: realize_args recurses into those itself, and each of their values
    reaches this on the way through.
    """
    if is_asset_reference(value):
        return fetch_asset(value, base_dir=base_dir)
    if isinstance(value, list):
        # In place: a list argument keeps its identity, the way every other
        # value realize_args touches does
        for index, item in enumerate(value):
            value[index] = resolve_asset_values(item, base_dir)
        return value
    return value
