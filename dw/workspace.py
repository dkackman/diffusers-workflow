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
import time
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

# The library shared by every workspace under one root:
# '<root>/common/assets'. The prompt library is shared because a 'prompt:'
# reference is shared by reference; assets are per workspace because an
# upload belongs to the work that made it - but a recurring cast is neither,
# and reaching it from a fresh workspace meant copying the files in
# (2026-09-11). A shared library sits on every workspace's asset search
# path, behind its own, so a workspace name still shadows a shared one and
# a write still lands in the workspace unless it says otherwise
COMMON_SUBDIR = "common"

# Where a job export lands: '<root>/exports/<job id>/'. Not a workspace
# folder - it is beside them, holding gathered copies rather than working
# content - but it is a name a workspace may not take, and workspace_names
# must not mistake it for one
EXPORTS_SUBDIR = "exports"

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
# use them - its four content folders, and the exports gathered beside them
RESERVED_WORKSPACE_NAMES = SUBDIRS + (EXPORTS_SUBDIR, COMMON_SUBDIR)

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

    def __init__(
        self,
        root,
        source,
        name=DEFAULT_WORKSPACE_NAME,
        prompts_root=None,
        common_root=None,
    ):
        self.root = os.path.abspath(os.path.expanduser(str(root)))
        self.source = source
        self.name = name
        self._prompts_root = (
            os.path.abspath(os.path.expanduser(str(prompts_root)))
            if prompts_root
            else None
        )
        # Like the prompt library, the shared one belongs to the root rather
        # than to this workspace - a named workspace is handed the root's
        self._common_root = (
            os.path.abspath(os.path.expanduser(str(common_root)))
            if common_root
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

    @property
    def common_assets(self):
        """The asset library every workspace under this root shares."""
        root = self._common_root or os.path.join(self.root, COMMON_SUBDIR)
        return os.path.join(root, ASSETS_SUBDIR)

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
            "common_assets": self.common_assets,
        }

    def __repr__(self):
        return f"Workspace({self.root!r}, {self.name}, from {self.source})"

    def __eq__(self, other):
        return (
            isinstance(other, Workspace)
            and self.root == other.root
            and self.source == other.source
        )


def _has_subdirs(path, names, all_of):
    """Whether a directory holds some (any_of) or all (all_of) of these
    subfolder names - the shared test behind looks_like_workspace's "any
    marker is enough" and _holds_a_workspace's stricter "all three, or it
    isn't a workspace"."""
    test = all if all_of else any
    return test(os.path.isdir(os.path.join(path, name)) for name in names)


def looks_like_workspace(path):
    """Whether a directory holds the subfolders that mark a workspace."""
    return _has_subdirs(path, MARKER_SUBDIRS, all_of=False)


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


# Read-only libraries searched after the one a run writes to. A tree named
# by --examples-dir brings the prompts and assets its workflows reference
# along with it, and neither is reachable from the workspace's own library -
# the search path is what closes that. Carried in the environment, joined by
# os.pathsep, so a spawned worker inherits it the way it inherits
# DW_PROMPT_DIR and DW_ASSET_DIR
PROMPT_PATH_ENV_VAR = "DW_PROMPT_PATH"
ASSET_PATH_ENV_VAR = "DW_ASSET_PATH"

LIBRARY_PATH_ENV_VARS = {
    PROMPTS_SUBDIR: PROMPT_PATH_ENV_VAR,
    ASSETS_SUBDIR: ASSET_PATH_ENV_VAR,
}


def library_fallbacks(subdir, primary=None):
    """The read-only roots a library is searched in after its own, in order.

    Args:
        subdir: 'prompts' or 'assets'
        primary: The library that is searched first, dropped from the result
            when it also appears here - a checkout serving as both the
            workspace and the examples tree has one library, not two

    Returns:
        A list of absolute paths, each an existing directory
    """
    raw = os.environ.get(LIBRARY_PATH_ENV_VARS[subdir], "")
    first = os.path.abspath(os.path.expanduser(str(primary))) if primary else None
    roots = []
    for entry in raw.split(os.pathsep):
        if not entry.strip():
            continue
        root = os.path.abspath(os.path.expanduser(entry))
        if root == first or root in roots or not os.path.isdir(root):
            continue
        roots.append(root)
    return roots


def set_library_fallbacks(subdir, roots):
    """Pin a library's read-only roots in the environment, so the worker
    subprocess resolves a reference exactly as the entry point would."""
    joined = os.pathsep.join(
        os.path.abspath(os.path.expanduser(str(root))) for root in roots or []
    )
    if joined:
        os.environ[LIBRARY_PATH_ENV_VARS[subdir]] = joined
    else:
        os.environ.pop(LIBRARY_PATH_ENV_VARS[subdir], None)
    return joined


def example_libraries(examples_dirs):
    """The prompt and asset libraries the trees named by --examples-dir
    bring with them.

    A workflows/ tree inside a checkout keeps the prompts and assets its
    workflows reference beside it, so the sibling of each examples directory
    is where they are; a directory that holds them itself - someone pointing
    --examples-dir at a whole workspace - is checked first, and either can
    be missing.

    Args:
        examples_dirs: The directories --examples-dir named, in order

    Returns:
        {'prompts': [roots], 'assets': [roots]}, deduplicated, in the order
        the examples directories were given
    """
    found = {PROMPTS_SUBDIR: [], ASSETS_SUBDIR: []}
    for directory in examples_dirs or []:
        root = os.path.abspath(os.path.expanduser(str(directory)))
        for holder in (root, os.path.dirname(root)):
            for subdir in found:
                candidate = os.path.join(holder, subdir)
                if os.path.isdir(candidate) and candidate not in found[subdir]:
                    found[subdir].append(candidate)
    return found


def discover_library(subdir, env_var, base_dir=None):
    """Where a shared library (prompts, assets) is rooted, by the precedence
    get_prompt_dir and get_asset_dir both need: an environment variable names
    it outright; then a workspace someone named (a --workspace flag,
    DW_WORKSPACE, or the 'workspace' setting), whose <subdir>/ is the library
    by definition; then <subdir>/ in the working directory when that exists;
    then the walk from base_dir up toward the filesystem root, looking for
    the <subdir>/ folder of whatever tree base_dir lives in; and finally the
    workspace's <subdir>/ as the fallback.

    The middle two steps predate workspaces and stay below the named-workspace
    check on purpose - a workspace merely inferred from the working directory
    or fallen back to must not preempt a library a workflow already reaches.

    Args:
        subdir: The library's folder name under a workspace ('prompts',
            'assets')
        env_var: The environment variable that names it explicitly
        base_dir: The workflow file's directory, when one anchors the search
    """
    explicit = os.environ.get(env_var)
    if explicit:
        return explicit

    workspace = resolve_workspace()
    if workspace.is_explicit:
        return getattr(workspace, subdir)

    working_directory_library = os.path.abspath(f"./{subdir}")
    if os.path.isdir(working_directory_library):
        return working_directory_library
    if base_dir:
        current = os.path.abspath(base_dir)
        while True:
            candidate = os.path.join(current, subdir)
            if os.path.isdir(candidate):
                return candidate
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
    return getattr(workspace, subdir)


def _holds_a_workspace(path):
    """Whether a directory is a named workspace rather than some other
    folder someone left at the root.

    All three folders, not any one: create_workspace makes all three, and a
    looser test claims too much - a checkout used as the workspace root has
    dw/workflows/ (the packaged builtins) right there, and one folder would
    make the source package a workspace, listed, browsable and deletable.
    """
    return _has_subdirs(path, NAMED_SUBDIRS, all_of=True)


def _foreign_entries(path):
    """What a directory holds besides a workspace's own three folders and
    the exports it may have gathered."""
    ignored = NAMED_SUBDIRS + (EXPORTS_SUBDIR,)
    try:
        return sorted(entry for entry in os.listdir(path) if entry not in ignored)
    except OSError:
        return []


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
    name = validate_workspace_name(name)
    return Workspace(
        os.path.join(workspace.root, name),
        workspace.source,
        name=name,
        prompts_root=os.path.join(workspace.root, PROMPTS_SUBDIR),
        common_root=os.path.join(workspace.root, COMMON_SUBDIR),
    )


def create_workspace(workspace, name):
    """Make a new named workspace under this root.

    Raises:
        InvalidInputError: If the name is not one a workspace can take
        FileExistsError: If a workspace of that name is already there
    """
    from .security import validate_workspace_name

    name = validate_workspace_name(name)
    if name in workspace_names(workspace):
        raise FileExistsError(f"Workspace '{name}' already exists")
    return named_workspace(workspace, name).ensure()


def _tree_usage(directory):
    """Files and bytes under one directory, as (files, bytes).

    Walks with scandir and stats through the DirEntry, which reuses the stat
    the directory read already did - the difference matters on an outputs
    tree of thousands of generated files. Symlinks are counted as neither
    file nor directory, so a link into a model cache cannot inflate the
    number or send the walk outside the workspace. Anything unreadable is
    skipped: this is a size to glance at, not an audit.
    """
    files = 0
    total = 0
    stack = [directory]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
                    files += 1
            except OSError:
                continue
    return files, total


def workspace_contents(workspace):
    """How much a workspace holds, for a client about to offer to delete it:
    file counts and total bytes per folder. Counting is the point - the
    number is what makes 'delete this workspace' an informed choice."""
    summary = {}
    for folder in NAMED_SUBDIRS:
        files, total = _tree_usage(os.path.join(workspace.root, folder))
        summary[folder] = {"files": files, "bytes": total}
    return summary


# How long a computed workspace size stays good enough to answer with. The
# number is a glance at how much a workspace holds, not an accounting
# figure, and re-walking a large outputs tree for every listing would cost
# more than the precision is worth - a running job moves it continuously
# anyway
USAGE_CACHE_SECONDS = 60

# the folders counted (a tuple of paths) -> (monotonic time, usage)
_usage_cache = {}


def _own_folders(workspace):
    """The folders whose bytes count as this workspace's own.

    A workspace's four folder properties are the candidates, deduplicated,
    minus any that is not actually inside its root: a named workspace's
    prompt library belongs to the root and is shared by every workspace, so
    counting it here would count it once per workspace. A workspace with no
    root of its own (a server configured folder by folder, where each can
    point anywhere) has nothing to exclude, and every folder it names is its
    own by definition.
    """
    folders = []
    for path in (
        workspace.workflows,
        workspace.prompts,
        workspace.assets,
        workspace.outputs,
    ):
        if not path or path in folders:
            continue
        if workspace.root and not _is_within(path, workspace.root):
            continue
        folders.append(path)
    return folders


def _is_within(path, root):
    """Whether a path is the root or sits under it."""
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives on Windows
        return False


def workspace_usage(workspace, max_age=USAGE_CACHE_SECONDS):
    """Roughly how much disk a workspace occupies: files and total bytes.

    Only the folders that are the workspace's own are counted, per
    _own_folders - which is what keeps the shared prompt library from being
    added to every workspace's total, and named workspaces from being
    counted inside the default one (they sit beside its folders, not in
    them).
    """
    folders = _own_folders(workspace)
    key = tuple(folders)
    now = time.monotonic()
    cached = _usage_cache.get(key)
    if cached and now - cached[0] < max_age:
        return cached[1]
    files = 0
    total = 0
    for directory in folders:
        if not os.path.isdir(directory):
            continue
        count, size = _tree_usage(directory)
        files += count
        total += size
    usage = {"files": files, "bytes": total}
    _usage_cache[key] = (now, usage)
    return usage


def forget_workspace_usage():
    """Drop every cached size - after creating or deleting a workspace,
    where a stale answer would be visibly wrong rather than merely old."""
    _usage_cache.clear()


def delete_workspace(workspace, name):
    """Remove a named workspace and everything in it.

    The default workspace is the root itself and is never deletable - it
    holds the shared prompt library, and there has to be somewhere to work.

    A directory that holds anything besides the three workspace folders is
    refused too: rmtree is unrecoverable, and a .git, an __init__.py or a
    stray file says this is something that merely contains a workspace, not
    one that is only a workspace.

    Raises:
        ValueError: For the default workspace
        FileNotFoundError: If no workspace has that name
        NotAWorkspaceError: If the directory holds more than a workspace
    """
    import shutil

    if name is None or name == DEFAULT_WORKSPACE_NAME:
        raise ValueError("The default workspace cannot be deleted")
    if name not in workspace_names(workspace):
        raise FileNotFoundError(f"No such workspace: {name}")
    target = named_workspace(workspace, name)
    foreign = _foreign_entries(target.root)
    if foreign:
        raise NotAWorkspaceError(name, foreign)
    shutil.rmtree(target.root)
    return target


class ConfiguredWorkspace(Workspace):
    """The default workspace on a server started with individual directory
    overrides (`--workflow-dir`, `--output-dir`, etc.), rather than a single
    `--workspace` root.

    Those overrides mean the default workspace's four folders are not
    reliably `<root>/workflows` and friends - each can point anywhere - so
    they cannot be derived the way Workspace derives them from a root. This
    subclass instead takes the four directories directly and answers the
    same four properties from them, `root` carrying whatever the caller has
    (a workspace root when there is one, otherwise whatever describes the
    configuration - possibly None) for `describe()` and logging only; it
    plays no part in resolving the four folders below.
    """

    def __init__(self, workflows, assets, outputs, prompts, root=None):
        self.root = os.path.abspath(os.path.expanduser(str(root))) if root else None
        self.source = DEFAULT
        self.name = DEFAULT_WORKSPACE_NAME
        self._workflows = os.path.abspath(os.path.expanduser(str(workflows)))
        # assets is optional - a server configured with no asset library at
        # all, same as Workspace's own callers see via app.state.asset_dir
        self._assets = (
            os.path.abspath(os.path.expanduser(str(assets))) if assets else None
        )
        self._outputs = os.path.abspath(os.path.expanduser(str(outputs)))
        self._prompts = os.path.abspath(os.path.expanduser(str(prompts)))

    @property
    def is_default(self):
        return True

    @property
    def workflows(self):
        return self._workflows

    @property
    def assets(self):
        return self._assets

    @property
    def outputs(self):
        return self._outputs

    @property
    def prompts(self):
        return self._prompts

    @property
    def common_assets(self):
        """The shared library, when there is a root to hang it off. A server
        configured from three loose directories has no root and so no shared
        library - there is nothing for it to be common to."""
        return (
            os.path.join(self.root, COMMON_SUBDIR, ASSETS_SUBDIR) if self.root else None
        )

    def ensure(self):
        """Not implemented here: a ConfiguredWorkspace's folders were each
        already resolved by the entry point that configured them, and each
        has its own idea of who creates it (the CLI's own ensure() calls,
        JobManager, etc.) - there is no single 'the workspace' to create."""
        raise NotImplementedError(
            "ConfiguredWorkspace folders are created by whatever configured "
            "them, not by the workspace itself"
        )


class NotAWorkspaceError(ValueError):
    """A directory named as a workspace holds more than a workspace does."""

    def __init__(self, name, entries):
        self.name = name
        self.entries = entries
        super().__init__(
            f"'{name}' holds more than a workspace's own folders "
            f"({', '.join(entries)}) and will not be deleted - remove those "
            f"by hand if it really is a workspace"
        )
