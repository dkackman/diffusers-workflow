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

from . import references
from .library import (
    ASSET_DIR_ENV_VAR,
    ASSETS_KIND,
    library_path_from_env,
)
from .security import validate_asset_reference
from .workspace import ASSETS_SUBDIR, discover_library

logger = logging.getLogger("dw")

# The asset library of the run in progress. A server holds several
# workspaces and each has its own assets, so this cannot be a process-wide
# environment variable there the way the prompt library can - there is one
# prompt library, shared, but assets belong to a workspace. Set per job by
# the worker; unset for the CLI, which has one workspace per
# process and reads the environment below
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
    return references.is_ref(references.ASSET, value)


def asset_library(asset_dir=None, base_dir=None):
    """The asset library as a `LibraryPath`: the workspace's own first, then
    the read-only ones an entry point put on the path (the shared `common`
    library and the assets a --examples-dir tree brings with it), so an
    example workflow reaches the media it ships with while an upload still
    lands in the workspace.

    Args:
        asset_dir: The first directory; defaults to get_asset_dir()
        base_dir: The workflow file's directory, anchoring discovery when no
            asset directory is configured
    """
    primary = asset_dir or get_asset_dir(base_dir)
    return library_path_from_env(ASSETS_KIND, primary)


def resolve_asset_reference(reference, asset_dir=None, base_dir=None, library=None):
    """Resolve an 'asset:' reference to the file it names.

    Args:
        reference: The 'asset:name.ext' or 'asset:folder/name.ext' string
        asset_dir: Directory the name is rooted at; defaults to get_asset_dir()
        base_dir: The workflow file's directory, anchoring discovery when no
            asset directory is configured
        library: The `LibraryPath` to resolve over, for a caller that holds
            the search path itself (the server's, for a workspace); it
            replaces `asset_dir` and `base_dir`, which are ignored
            when it is given

    Returns:
        The validated absolute path of the asset file

    Raises:
        InvalidInputError: If the name is not a valid asset name
        PathTraversalError: If the name escapes the asset directory
        ValueError: If no file exists under that name in any directory on
            the search path
    """
    name = references.ref_name(references.ASSET, reference)
    if name is None:
        # A bare name resolves as written: callers guard with is_asset_reference,
        # a direct caller need not
        name = reference
    name = validate_asset_reference(name.strip())
    library = library or asset_library(asset_dir, base_dir)
    # Confined to the library it was found in: the name is joined onto a
    # directory, so the containment check is what makes a name a name
    # rather than a path
    found = library.find(name, refuse=True)
    if found:
        logger.debug(f"Resolved {reference} to {found[0]}")
        return found[0]
    searched = ", ".join(root.root for root in library.roots())
    raise ValueError(
        f"Asset '{name}' not found in {searched} - an 'asset:' reference "
        f"names a file in the asset library, with its extension, like "
        f"'asset:iris.jpg' or 'asset:gyre/frame_1.jpg'"
    )


def fetch_asset(reference, asset_dir=None, base_dir=None):
    """The path an 'asset:' reference names, for whatever loads paths."""
    return resolve_asset_reference(reference, asset_dir, base_dir)
