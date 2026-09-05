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

# A server can hold several workspaces: the root's own workflows/assets/
# outputs are the default one, and a named workspace is a subdirectory of
# the root holding the same three. The prompt library is not among them -
# there is one, at the root, shared by every workspace, because a stored
# prompt is shared by reference and 'prompt:scenic' resolving to different
# text per workspace would break exactly that
DEFAULT_WORKSPACE_NAME = "default"

# Names a workspace cannot take, because the root's own folders already
# use them
RESERVED_WORKSPACE_NAMES = SUBDIRS

# What a named workspace holds - prompts excluded, per above
NAMED_SUBDIRS = (WORKFLOWS_SUBDIR, ASSETS_SUBDIR, OUTPUTS_SUBDIR)

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
    """A resolved workspace root and the four directories under it.

    A named workspace is the same thing rooted one level down, with its
    prompt library pointing back at the root's - `prompts_root` is what
    carries that, and it is why a named workspace has three folders where
    the default has four.
    """

    def __init__(self, root, source, name=DEFAULT_WORKSPACE_NAME, prompts_root=None):
        self.root = os.path.abspath(os.path.expanduser(str(root)))
        self.source = source
        self.name = name
        self._prompts_root = (
            os.path.abspath(os.path.expanduser(str(prompts_root)))
            if prompts_root
            else None
        )

    @property
    def is_default(self):
        """Whether this is the root's own workspace rather than a named one."""
        return self._prompts_root is None

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
        """The shared library: a named workspace points back at the root's."""
        return self._prompts_root or os.path.join(self.root, PROMPTS_SUBDIR)

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
        for subdir in NAMED_SUBDIRS if self._prompts_root else SUBDIRS:
            Path(self.root, subdir).mkdir(parents=True, exist_ok=True)
        if self._prompts_root:
            Path(self._prompts_root).mkdir(parents=True, exist_ok=True)
        return self

    def describe(self):
        """What a client needs to name this workspace and its folders."""
        return {
            "name": self.name,
            "default": self.is_default,
            "root": self.root,
            "workflows": self.workflows,
            "assets": self.assets,
            "outputs": self.outputs,
            "prompts": self.prompts,
        }

    def __repr__(self):
        return f"Workspace({self.root!r}, {self.name}, from {self.source})"

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


def _holds_a_workspace(path):
    """Whether a directory is a named workspace rather than some other
    folder someone left at the root."""
    return any(os.path.isdir(os.path.join(path, name)) for name in NAMED_SUBDIRS)


def workspace_names(workspace):
    """Every workspace under this root, the default first.

    The default is the root's own folders, so it is always present even on a
    root that holds nothing else - there is always somewhere to work.
    """
    names = []
    try:
        for entry in sorted(os.listdir(workspace.root)):
            if entry in RESERVED_WORKSPACE_NAMES or entry.startswith("."):
                continue
            path = os.path.join(workspace.root, entry)
            if os.path.isdir(path) and _holds_a_workspace(path):
                names.append(entry)
    except OSError:
        names = []
    return [DEFAULT_WORKSPACE_NAME] + names


def named_workspace(workspace, name):
    """One workspace under this root, by name.

    The default name resolves to the root's own workspace; any other name
    resolves to '<root>/<name>', sharing the root's prompt library. The name
    is validated before it is joined, so nothing here can leave the root.
    """
    from .security import validate_workspace_name

    if name is None or name == DEFAULT_WORKSPACE_NAME:
        return workspace
    validate_workspace_name(name)
    return Workspace(
        os.path.join(workspace.root, name),
        workspace.source,
        name=name,
        prompts_root=os.path.join(workspace.root, PROMPTS_SUBDIR),
    )


def workspace_exists(workspace, name):
    """Whether a name resolves to a workspace that is actually there."""
    if name is None or name == DEFAULT_WORKSPACE_NAME:
        return True
    return name in workspace_names(workspace)


def create_workspace(workspace, name):
    """Make a new named workspace under this root.

    Raises:
        InvalidInputError: If the name is not one a workspace can take
        FileExistsError: If a workspace of that name is already there
    """
    from .security import validate_workspace_name

    validate_workspace_name(name)
    if name in workspace_names(workspace):
        raise FileExistsError(f"Workspace '{name}' already exists")
    return named_workspace(workspace, name).ensure()


def workspace_contents(workspace):
    """How much a workspace holds, for a client about to offer to delete it:
    file counts and total bytes per folder. Counting is the point - the
    number is what makes 'delete this workspace' an informed choice."""
    summary = {}
    for folder in NAMED_SUBDIRS:
        directory = os.path.join(workspace.root, folder)
        files = 0
        total = 0
        for current, _dirs, names in os.walk(directory):
            for entry in names:
                try:
                    total += os.path.getsize(os.path.join(current, entry))
                except OSError:
                    continue
                files += 1
        summary[folder] = {"files": files, "bytes": total}
    return summary


def delete_workspace(workspace, name):
    """Remove a named workspace and everything in it.

    The default workspace is the root itself and is never deletable - it
    holds the shared prompt library, and there has to be somewhere to work.
    """
    import shutil

    if name is None or name == DEFAULT_WORKSPACE_NAME:
        raise ValueError("The default workspace cannot be deleted")
    if name not in workspace_names(workspace):
        raise FileNotFoundError(f"No such workspace: {name}")
    target = named_workspace(workspace, name)
    shutil.rmtree(target.root)
    return target
