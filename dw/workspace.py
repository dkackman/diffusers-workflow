"""The workspace: the directory holding what a user makes.

A workflow file, the assets it reads, the prompts it references and the files
it writes started life beside the code, which was right when the only entry
point was a CLI run from a checkout. It stops being right once an agent or a
remote client is the one authoring: generated media does not belong in the
source tree, and an agent's day-to-day workflows do not belong in the example
corpus this repository ships.

A workspace names one directory for all of it:

    <workspace>/
      workflows/    mine, writable
      prompts/      the stored prompt library
      assets/       input media
      outputs/      generated files

Resolution order, first hit wins:

1. an explicit path (a --workspace flag)
2. the DW_WORKSPACE environment variable
3. 'workspace' in ~/.diffusers_helper/settings.json
4. the working directory, when it looks like a workspace - it holds any of
   workflows/, prompts/ or outputs/
5. ~/diffusers-workspace

Rule 4 is what keeps a checkout working unchanged: the repository root holds
all three, so running from it resolves to it, and every default lands exactly
where it landed before there was a workspace at all. Only a working directory
with none of those markers falls through to the home workspace.

Nothing here creates a directory. Resolution is a pure question about paths -
an entry point that is about to write calls ensure() once it knows it needs to.
"""

import os
from pathlib import Path

# Set by an entry point that resolved a workspace, so a spawned worker
# subprocess inherits the same answer - multiprocessing's 'spawn' start method
# launches a fresh interpreter that inherits os.environ, the way DW_PROMPT_DIR
# and DW_TRUST_WORKFLOWS already reach the worker
WORKSPACE_ENV_VAR = "DW_WORKSPACE"

# Carries alongside it how the workspace was chosen. Without it a workspace
# merely inferred from the working directory would come back to the worker
# looking like one the user named, and an inferred workspace deliberately
# yields to discovery that predates it - see Workspace.is_explicit
WORKSPACE_SOURCE_ENV_VAR = "DW_WORKSPACE_SOURCE"

# Deliberately not '~/diffusers-workflow': that is where this repository
# gets cloned, and a default that lands inside a checkout is the coupling
# workspaces exist to remove
DEFAULT_WORKSPACE = "~/diffusers-workspace"

WORKFLOWS_SUBDIR = "workflows"
PROMPTS_SUBDIR = "prompts"
ASSETS_SUBDIR = "assets"
OUTPUTS_SUBDIR = "outputs"

SUBDIRS = (WORKFLOWS_SUBDIR, PROMPTS_SUBDIR, ASSETS_SUBDIR, OUTPUTS_SUBDIR)

# What makes a directory recognizable as a workspace. assets/ is deliberately
# not a marker - a bare assets/ folder is a common thing to have lying around,
# where these three together say "content lives here"
MARKER_SUBDIRS = (WORKFLOWS_SUBDIR, PROMPTS_SUBDIR, OUTPUTS_SUBDIR)

# How a workspace was chosen, in resolution order. Everything but the last two
# is an answer someone gave on purpose; see Workspace.is_explicit
FLAG = "flag"
ENVIRONMENT = "environment"
SETTINGS = "settings"
WORKING_DIRECTORY = "working directory"
DEFAULT = "default"

EXPLICIT_SOURCES = (FLAG, ENVIRONMENT, SETTINGS)
ALL_SOURCES = EXPLICIT_SOURCES + (WORKING_DIRECTORY, DEFAULT)


class Workspace:
    """A resolved workspace root and the four directories under it."""

    def __init__(self, root, source):
        self.root = os.path.abspath(os.path.expanduser(str(root)))
        self.source = source

    @property
    def is_explicit(self):
        """Whether someone named this workspace, rather than it being
        inferred from the working directory or fallen back to.

        Discovery that predates workspaces - the prompt library's walk up
        from the workflow file - stays ahead of an inferred workspace and
        behind a named one, so turning this on changes nothing for a caller
        who has not asked for a workspace.
        """
        return self.source in EXPLICIT_SOURCES

    @property
    def workflows(self):
        return os.path.join(self.root, WORKFLOWS_SUBDIR)

    @property
    def prompts(self):
        return os.path.join(self.root, PROMPTS_SUBDIR)

    @property
    def assets(self):
        return os.path.join(self.root, ASSETS_SUBDIR)

    @property
    def outputs(self):
        return os.path.join(self.root, OUTPUTS_SUBDIR)

    def ensure(self):
        """Create the workspace and its subdirectories if they are missing.

        Called by an entry point that is about to write, not by resolution -
        asking where the workspace is should never leave a directory behind.
        """
        for subdir in SUBDIRS:
            Path(self.root, subdir).mkdir(parents=True, exist_ok=True)
        return self

    def __repr__(self):
        return f"Workspace({self.root!r}, from {self.source})"

    def __eq__(self, other):
        return (
            isinstance(other, Workspace)
            and self.root == other.root
            and self.source == other.source
        )


def looks_like_workspace(path):
    """Whether a directory holds the subfolders that mark a workspace."""
    return any(os.path.isdir(os.path.join(path, name)) for name in MARKER_SUBDIRS)


def resolve_workspace(explicit=None):
    """The workspace this process works in.

    Args:
        explicit: A path from a --workspace flag, when one was given

    Returns:
        A Workspace, which may not exist on disk yet
    """
    if explicit:
        return Workspace(explicit, FLAG)

    from_environment = os.environ.get(WORKSPACE_ENV_VAR)
    if from_environment:
        source = os.environ.get(WORKSPACE_SOURCE_ENV_VAR)
        return Workspace(
            from_environment, source if source in ALL_SOURCES else ENVIRONMENT
        )

    # Imported here, not at module scope: dw.settings reads a file, and
    # resolution is called from argument parsing on every entry point
    from .settings import load_settings

    from_settings = load_settings().workspace
    if from_settings:
        return Workspace(from_settings, SETTINGS)

    working_directory = os.path.abspath(os.getcwd())
    if looks_like_workspace(working_directory):
        return Workspace(working_directory, WORKING_DIRECTORY)

    return Workspace(DEFAULT_WORKSPACE, DEFAULT)


def set_workspace(workspace):
    """Pin a resolved workspace in the environment, so a spawned worker
    subprocess and anything resolving later in this process agree with the
    entry point that chose it.

    Args:
        workspace: The Workspace to pin, or a path

    Returns:
        The Workspace that was pinned
    """
    if not isinstance(workspace, Workspace):
        workspace = Workspace(workspace, FLAG)
    os.environ[WORKSPACE_ENV_VAR] = workspace.root
    os.environ[WORKSPACE_SOURCE_ENV_VAR] = workspace.source
    return workspace
