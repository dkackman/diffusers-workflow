"""The content libraries' search paths: which roots, in which order, who may write.

A library (workflows today; prompts and assets next) used to be a single
directory that was both the library and the place saves landed. With the
repository's own workflows/ as that directory - the default when a checkout is
the workspace - every save from the editor or an MCP client wrote into the
example corpus.

A search path separates the two. Reads resolve front to back; writes only
ever go to the front:

    <workspace>/workflows/   the user's own, writable
    <examples dirs>          read-only, --examples-dir

A name found in an earlier root shadows the same name in a later one, so a
workspace copy of an example is the one that runs. Saving over a read-only
workflow is not an error and not an overwrite: it writes a copy into the
writable root, which is what "open an example, change it, save" should do.

`LibraryPath` is the one implementation of those rules - `find`, `entries`,
`writable_root`, `require_writable`. What differs between libraries is three
small strategies chosen by `kind`: how a name maps to a file, which named
validator confines it, and how a root is listed. This module imports nothing
from `dw.server`; a lister that lives there (the asset media walk) is passed
in.
"""

import logging
import os

from . import references
from .security import (
    InvalidInputError,
    PathTraversalError,
    SecurityError,
    contained,
    validate_path,
    validate_prompt_path,
    validate_workflow_path,
)
from .workspace import (
    example_libraries,
    resolve_workspace,
)

logger = logging.getLogger("dw")

# What a source is, for a client deciding whether to offer save or delete
WORKSPACE_ORIGIN = "workspace"
EXAMPLES_ORIGIN = "examples"
BUILTIN_ORIGIN = "builtin"
# Not a workflow source - the asset library shared by every workspace under
# one root reports itself this way, so a client can tell a shared asset from
# one of its own
COMMON_ORIGIN = "common"


# The environment is how the worker subprocess learns the libraries: spawn
# inherits it, it does not inherit the argument parser. The primary (the
# directory a save lands in) is named by one variable per library; the
# read-only roots after it are an os.pathsep list in another. About thirty
# test sites set these, so the names and the format are a pinned interface.
ASSET_DIR_ENV_VAR = "DW_ASSET_DIR"
PROMPT_DIR_ENV_VAR = "DW_PROMPT_DIR"
PROMPT_PATH_ENV_VAR = "DW_PROMPT_PATH"
ASSET_PATH_ENV_VAR = "DW_ASSET_PATH"
# The same idea for workflows, which a sub-workflow step names: a stored
# template lives in an examples tree the workspace's own workflows/ cannot
# reach, so composing one used to mean copying it in (#90)
WORKFLOW_PATH_ENV_VAR = "DW_WORKFLOW_PATH"


def builtin_root():
    """The packaged workflows that ship inside dw/ - what 'builtin:' names."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflows")


def catalog_root(directory):
    """The nearest ancestor of `directory` literally named 'workflows', else
    the directory itself.

    The root a run with no workflow_dir of its own confines a relative
    sub-workflow reference to, so a template under templates/ can still climb
    to a sibling models/ without leaving the catalog.
    """
    directory = os.path.normpath(os.path.abspath(directory))
    parts = directory.split(os.sep)
    try:
        index = len(parts) - 1 - parts[::-1].index("workflows")
    except ValueError:
        return directory

    return os.sep.join(parts[: index + 1])


def workflow_output_subfolder(file_spec):
    """The subfolder a workflow's outputs land in, mirroring its position
    under the nearest directory literally named 'workflows' in its path.

    'workflows/ltx/Foo.json' -> 'ltx'; 'workflows/Foo.json' (or a builtin,
    always dw/workflows/<name>.json) -> '' (flat, no spurious subfolder);
    a path with no 'workflows' segment at all (an inline definition's
    synthetic file_spec, say) -> '' as a fallback. The *last* 'workflows'
    segment wins, matching the packaged dw/workflows tree when a checkout
    also has a top-level workflows/ directory somewhere in its ancestry.
    """
    if not file_spec:
        return ""

    directory = os.path.dirname(os.path.abspath(file_spec))
    parts = os.path.normpath(directory).split(os.sep)
    try:
        index = len(parts) - 1 - parts[::-1].index("workflows")
    except ValueError:
        return ""

    return os.path.join(*parts[index + 1 :]) if index + 1 < len(parts) else ""


class LibraryRoot:
    """One root on a library's search path."""

    def __init__(self, root, origin, writable):
        self.root = os.path.abspath(os.path.expanduser(str(root)))
        self.origin = origin
        self.writable = writable

    def contains(self, path):
        """Whether a path resolves inside this root - the containment check
        the security layer already implements, asked as a question rather
        than raised as an error."""
        try:
            validate_path(path, self.root)
            return True
        except SecurityError:
            return False

    def to_dict(self):
        return {"root": self.root, "origin": self.origin, "writable": self.writable}

    def __repr__(self):
        return f"LibraryRoot({self.root!r}, {self.origin}, writable={self.writable})"


class ReadOnlyLibraryError(Exception):
    """A write was aimed at an entry of a read-only root."""

    def __init__(self, name, root, kind):
        self.name = name
        self.root = root
        self.kind = kind
        super().__init__(
            f"'{name}' is in the read-only {root.origin} library ({root.root}); "
            f"only the workspace's own {kind} can be deleted"
        )


# kind -> how a name becomes a file name under a root. Workflows and prompts
# are JSON documents; an asset's name is its relative path, literally
WORKFLOWS_KIND = "workflows"
PROMPTS_KIND = "prompts"
ASSETS_KIND = "assets"
_JSON_KINDS = {WORKFLOWS_KIND, PROMPTS_KIND}
LIBRARY_KINDS = (WORKFLOWS_KIND, PROMPTS_KIND, ASSETS_KIND)
LIBRARY_PATH_ENV_VARS = {
    PROMPTS_KIND: PROMPT_PATH_ENV_VAR,
    ASSETS_KIND: ASSET_PATH_ENV_VAR,
    WORKFLOWS_KIND: WORKFLOW_PATH_ENV_VAR,
}


class LibraryPath:
    """An ordered search path of `LibraryRoot`s for one kind of library.

    The rules every library shares, written once: a name resolves front to
    back and only inside its root (a symlink that leaves it is a miss); a
    listing offers each name once, from the earliest root, and reports the
    later copies as shadowed; saves go to the writable root.
    """

    def __init__(self, kind, roots):
        if kind not in LIBRARY_KINDS:
            raise ValueError(f"Unknown library kind: {kind}")
        self.kind = kind
        self._roots = tuple(roots)

    def roots(self):
        """The roots, in search order."""
        return list(self._roots)

    def existing(self):
        """This path without the roots that are not directories yet - the
        front of a path is kept before a save creates it, and a client
        listing or serving from the library has nothing to read there."""
        return LibraryPath(
            self.kind, [root for root in self._roots if os.path.isdir(root.root)]
        )

    def __repr__(self):
        return f"LibraryPath({self.kind!r}, {list(self._roots)!r})"

    def file_name(self, name):
        """The file a name has under a root: JSON libraries take `.json`
        appended when the name has no extension - the catalog reports names
        without it, and run_workflow's workflow_path takes either (#90)."""
        if self.kind in _JSON_KINDS and not name.endswith(".json"):
            return f"{name}.json"
        return name

    def path_in(self, root, name, allow_create=False):
        """The on-disk path a name has in one root, or None when the name
        does not resolve inside it. Containment is the security layer's, so a
        name that tries to traverse simply does not resolve."""
        candidate = os.path.join(root.root, self.file_name(name))
        try:
            if self.kind == WORKFLOWS_KIND and not allow_create:
                # The named validators CodeQL models for py/path-injection;
                # each also enforces the .json extension
                return validate_workflow_path(candidate, root.root)
            if self.kind == PROMPTS_KIND and not allow_create:
                return validate_prompt_path(candidate, root.root)
            # Assets: containment alone, with a non-None base
            return validate_path(candidate, root.root, allow_create=allow_create)
        except SecurityError:
            return None

    def find(self, name, refuse=False):
        """The first root that has this name, as (path, root), or None.

        Front to back, so a workspace copy shadows the example it was copied
        from. Each candidate is confined to its own root, so a symlink that
        points outside it is a miss rather than a leak. A resolver that owes
        its caller a refusal rather than a "not found" - an `asset:` or
        `prompt:` reference naming a link out of the library - passes
        `refuse=True`, and the security layer's error propagates instead.
        """
        for root in self._roots:
            if refuse:
                candidate = os.path.join(root.root, self.file_name(name))
                if self.kind == PROMPTS_KIND:
                    # A prompt that is not a file here - a dangling link
                    # included - is a miss; only one that is present is
                    # confined, and refused when it leaves the library.
                    # A name that leaves the root before any link is
                    # followed is refused without probing the disk there
                    candidate = os.path.normpath(candidate)
                    inside = os.path.join(os.path.normpath(root.root), "")
                    if not candidate.startswith(inside):
                        return validate_prompt_path(candidate, root.root), root
                    if not os.path.isfile(candidate):
                        continue
                    return validate_prompt_path(candidate, root.root), root
                path = validate_path(candidate, root.root)
                if os.path.isfile(path):
                    return path, root
                continue
            path = self.path_in(root, name)
            if path and os.path.isfile(path):
                return path, root
        return None

    def root_for_path(self, path):
        """Which root a resolved path belongs to, or None if it is outside
        every root - which is what makes a path a library entry rather than
        an arbitrary file."""
        for root in self._roots:
            if root.contains(path):
                return root
        return None

    def entries(self, lister=None):
        """(winners, shadowed): every name the path offers with the root it
        comes from, sorted by name, and the later copies a name hid.

        `lister(root_dir)` yields the names under one root. Workflows and
        prompts list by walking for JSON files; an asset library is listed by
        the server's media walk, which lives above this module, so it is
        passed in rather than imported. `shadowed` is a list of
        `(name, root, shadowed_by)`, `shadowed_by` being the winning root.
        """
        if lister is None:
            if self.kind not in _JSON_KINDS:
                raise ValueError(f"A {self.kind} library needs a lister")
            lister = workflow_names
        winners = {}
        shadowed = []
        for root in self._roots:
            for name in lister(root.root):
                if name in winners:
                    shadowed.append((name, root, winners[name]))
                else:
                    winners[name] = root
        return dict(sorted(winners.items())), shadowed

    def describe(self):
        """The path as a listing reports it: `[{origin, root, writable}]`, in
        search order - the one `libraries` field of every library listing."""
        return [root.to_dict() for root in self._roots]

    def writable_root(self, shared=False):
        """The root saves go to: the front of the path. `shared` asks for the
        shared (common) root instead, for the libraries that have one."""
        for root in self._roots:
            if root.writable and (root.origin == COMMON_ORIGIN) == shared:
                return root
        return None

    def require_writable(self, root, name):
        """The root itself, or `ReadOnlyLibraryError` when it is read-only."""
        if not root.writable:
            raise ReadOnlyLibraryError(name, root, self.kind)
        return root


def shadowed_listing(hidden):
    """`LibraryPath.entries`' shadowed list as a listing reports it:
    `[{name, origin, shadowed_by}]`, the origin being the hidden copy's and
    `shadowed_by` the origin of the root whose copy won."""
    return [
        {"name": name, "origin": root.origin, "shadowed_by": winner.origin}
        for name, root, winner in hidden
    ]


def workflow_names(root):
    """Workflow names under a root, as relative paths without .json.

    A file symlink resolving outside the root is not a name here: a listing
    opens every file it names, and reads by name already refuse the link."""
    names = []
    if not os.path.isdir(root):
        return names
    for directory, _dirs, files in os.walk(root):
        for file_name in files:
            if file_name.endswith(".json"):
                path = os.path.join(directory, file_name)
                if not contained(path, root):
                    continue
                relative = os.path.relpath(path, root)
                names.append(relative[: -len(".json")].replace(os.sep, "/"))
    return sorted(names)


def _assemble(kind, primary, candidates):
    """A path from its writable front and the roots behind it.

    `candidates` is `[(directory, origin, writable)]`. A candidate that is
    the front, or that an earlier candidate already named, is dropped, and so
    is one that is not a directory - the same for every library. The front is
    kept whether or not it exists yet: a save creates it. A front of None is
    a server with no library of that kind: the path is the roots behind it.
    """
    roots = [primary] if primary else []
    seen = {primary.root} if primary else set()
    for directory, origin, writable in candidates:
        root = LibraryRoot(directory, origin, writable)
        if root.root in seen or not os.path.isdir(root.root):
            continue
        seen.add(root.root)
        roots.append(root)
    return LibraryPath(kind, roots)


def _front_directory(kind, workspace):
    return getattr(workspace, kind)


def read_only_candidates(kind, workspace, examples_dirs=None, include_builtin=False):
    """The roots behind the writable front, in order, as
    `[(directory, origin, writable)]` - unfiltered, so a caller can pin them
    whole (`pin_library_path`) or drop the missing ones (`library_path`)."""
    candidates = []
    if kind == WORKFLOWS_KIND:
        candidates += [(d, EXAMPLES_ORIGIN, False) for d in examples_dirs or []]
        if include_builtin:
            candidates.append((builtin_root(), BUILTIN_ORIGIN, False))
        return candidates
    if kind == ASSETS_KIND:
        common = getattr(workspace, "common_assets", None)
        if common:
            candidates.append((common, COMMON_ORIGIN, True))
    libraries = example_libraries(examples_dirs)
    return candidates + [(d, EXAMPLES_ORIGIN, False) for d in libraries[kind]]


def library_path(
    kind, workspace, examples_dirs=None, primary=None, include_builtin=False
):
    """The search path of one library for a workspace.

    Workflows: the workspace's `workflows/` (writable), then each examples
    directory (read-only), then - when `include_builtin` asks - the packaged
    workflows.

    Prompts: the prompt directory (writable), then the `prompts/` each
    examples directory brings (read-only).

    Assets: the workspace's `assets/` (writable), then the library every
    workspace under the root shares (`common`; writable, but only a write
    that says it is shared goes there - `writable_root()` still answers the
    workspace's), then the examples' `assets/` (read-only).

    `primary` replaces the writable front, which is how a job's own
    directory names its path. A read-only root that is the writable one - a
    checkout whose workflows/ is both the workspace library and the examples
    - appears once, writable, rather than twice with two different answers
    about whether it can be saved to. A root that is not a directory is
    dropped, for every kind; the writable front is kept whether or not it
    exists yet; a workspace with no library of that kind (a server configured
    folder by folder, with no asset directory) has no front at all.

    The packaged workflows are off the path by default. They are the pieces
    a 'builtin:' sub-workflow step names, resolved by
    resolve_sub_workflow_reference (below) - not workflows anyone browses or runs on
    their own, and listing them would put a handful of fragments in front of
    every user who never asked for them.
    """
    if kind not in LIBRARY_KINDS:
        raise ValueError(f"Unknown library kind: {kind}")
    if include_builtin and kind != WORKFLOWS_KIND:
        raise ValueError("Only the workflows library has builtin entries")

    front = primary or _front_directory(kind, workspace)
    candidates = read_only_candidates(kind, workspace, examples_dirs, include_builtin)
    return _assemble(
        kind, LibraryRoot(front, WORKSPACE_ORIGIN, True) if front else None, candidates
    )


def _pinned_roots(kind):
    """The read-only roots the entry point pinned in the environment: absolute,
    existing directories, each once, in order."""
    roots = []
    for entry in os.environ.get(LIBRARY_PATH_ENV_VARS[kind], "").split(os.pathsep):
        if not entry.strip():
            continue
        root = os.path.abspath(os.path.expanduser(entry))
        if root not in roots and os.path.isdir(root):
            roots.append(root)
    return roots


def library_path_from_env(kind, primary=None):
    """The search path the worker (or any engine caller) resolves against:
    `primary` first, then the roots the entry point pinned with
    `pin_library_path`.

    Origins are re-derived rather than carried, so the worker's tags match
    the API's: a root equal to the workspace's `common/assets` is `common`
    (writable, as the constructor builds it), any other pinned root is
    `examples` (read-only).

    `primary` may be None - an unconfined workflow run has no front. A
    workflow `primary` that is itself a pinned read-only root is a run
    confined to an examples directory, so it is tagged `examples` and read-only
    rather than `workspace`; for prompts and assets the primary is always the
    workspace's own. The exception is the resolved workspace's own `workflows/`
    (a checkout that is its own examples directory), which stays `workspace`.
    Accepted limits: a server whose `--workflow-dir` override equals an
    examples directory is still tagged `examples` here; and in a checkout, a
    named workspace's job confined to that same `workflows/` is tagged
    `workspace`, where the API says `examples` for that workspace - the worker
    knows the server's root, not the job's workspace. Nothing reads the tag.
    """
    common = None
    if kind == ASSETS_KIND:
        try:
            common = os.path.abspath(resolve_workspace().common_assets)
        except Exception:
            logger.debug("Could not resolve the common asset library", exc_info=True)

    def tagged(directory):
        if directory == common:
            return directory, COMMON_ORIGIN, True
        return directory, EXAMPLES_ORIGIN, False

    pinned = _pinned_roots(kind)
    if primary is None:
        roots = []
        for directory in pinned:
            roots.append(LibraryRoot(*tagged(directory)))
        return LibraryPath(kind, roots)

    front = LibraryRoot(primary, WORKSPACE_ORIGIN, True)
    if kind == WORKFLOWS_KIND and front.root in pinned:
        # Unless it is the workspace's own workflows/ - a checkout whose
        # examples directory is its library - which the API keeps writable
        try:
            own = os.path.abspath(resolve_workspace().workflows)
        except Exception:
            own = None
            logger.debug("Could not resolve the workspace workflows", exc_info=True)
        if front.root != own:
            front = LibraryRoot(*tagged(front.root))
    return _assemble(kind, front, [tagged(directory) for directory in pinned])


def pin_library_path(kind, workspace, examples_dirs=None):
    """Pin a library's read-only roots in the environment, so the worker
    subprocess resolves a reference exactly as the entry point would.

    The whole tail - everything behind the writable front, which has its own
    `DW_*_DIR` variable - as an os.pathsep list, deduplicated only against
    itself. Nothing is dropped for equalling the server's own primary or for
    not existing yet: a job carries a primary of its own, and the read side
    (`library_path_from_env`) drops what equals that one and what is missing
    when it resolves. An empty tail removes the variable. Returns the value
    written.
    """
    name = LIBRARY_PATH_ENV_VARS[kind]
    roots = []
    for directory, _origin, _writable in read_only_candidates(
        kind, workspace, examples_dirs
    ):
        root = LibraryRoot(directory, None, False).root
        if root not in roots:
            roots.append(root)
    joined = os.pathsep.join(roots)
    if joined:
        os.environ[name] = joined
    else:
        os.environ.pop(name, None)
    return joined


def suggest_workflow_names(library, name, limit=3):
    """Catalog names an unresolved `name` might have meant, for an error
    message rather than a second round trip.

    The catalog is organised in directories (`templates/minimax/dialogue-short`)
    and a caller - a skill, an earlier turn - often has only the trailing
    name (`dialogue-short`). Preferred answer: every catalog entry `name` is
    a unique path suffix of, since that is unambiguous; failing that, a
    close spelling match (`difflib`), for a typo rather than a shortened
    path. Empty when neither finds anything worth naming.
    """
    import difflib

    stripped = name[: -len(".json")] if name.endswith(".json") else name
    catalog_names = list(library.entries()[0])
    suffix_matches = [
        candidate
        for candidate in catalog_names
        if candidate == stripped or candidate.endswith(f"/{stripped}")
    ]
    if suffix_matches:
        return suffix_matches[:limit]

    # A typo is measured against the catalog entry's own name, not against
    # its directory prefix - "dialog-short" scores 0.92 against
    # "dialogue-short" and 0.55 against "templates/minimax/dialogue-short",
    # so comparing full paths lets a real typo miss the cutoff (#397). The
    # query's own prefix is stripped the same way, so "sub/Basik" is
    # measured as "Basik" against "Basic" rather than against "sub/Basic".
    query_basename = stripped.rsplit("/", 1)[-1]
    by_basename = {}
    for candidate in catalog_names:
        by_basename.setdefault(candidate.rsplit("/", 1)[-1], []).append(candidate)
    close_bases = difflib.get_close_matches(
        query_basename, list(by_basename.keys()), n=limit, cutoff=0.6
    )
    matches = []
    for base in close_bases:
        for candidate in by_basename[base]:
            if candidate not in matches:
                matches.append(candidate)
    return matches[:limit]


def _candidate_names(name):
    """A name as written, and with .json appended when it has no extension -
    the catalog reports names without it, and run_workflow's workflow_path
    takes either (#90)."""
    names = [name]
    if not name.endswith(".json"):
        names.append(f"{name}.json")
    return names


class SubWorkflowNotFound(Exception):
    """A sub-workflow step's path names nothing on the search path."""

    def __init__(self, path, tried):
        self.path = path
        # In order, each candidate once - two roots can resolve one name to
        # the same file, and saying so twice reads as two failures
        self.tried = list(dict.fromkeys(tried))
        detail = "\n  ".join(self.tried)
        super().__init__(
            f"Sub-workflow '{path}' could not be resolved. It is read as a "
            "catalog name from list_workflows (with or without .json), or a "
            "path relative to the workflow that names it. Looked in:"
            f"\n  {detail}"
        )


def _describe_roots(roots):
    """Roots this resolution consulted, for a refusal message - never empty
    text, since a run with none configured still owes an answer."""
    return ", ".join(roots) if roots else "(no workflow sources configured)"


def resolve_sub_workflow(path, base_dir, confine_to):
    """Where a sub-workflow step's `path` resolves to, and the root the
    child is confined to, as (path, root). `root` is a `LibraryRoot` tagged
    for what it is - `workspace` for the run's own root, `examples` for a
    fallback - or None when the run is unconfined.

    Order, first hit wins:

      1. relative to the directory of the workflow that names it - which is
         how a template reaches '../models/x.json', and stays first so an
         existing composition keeps meaning what it did
      2. the same, with '.json' supplied
      3. the run's own workflow root (a workspace's workflows/), by catalog
         name, with or without '.json'
      4. each read-only root on the search path, the same way - which is
         what lets a stored template be composed rather than copied (#90)

    An absolute path is taken as written and confined to whichever root
    holds it, so the sandbox still refuses one that belongs to no source.

    A relative path that climbs out of the root it will be handed back with is
    a PathTraversalError rather than a SubWorkflowNotFound - a refusal, not a
    name that was absent. Otherwise raises SubWorkflowNotFound, naming every
    candidate it looked at.
    """
    # The run's own root is the writable front of the path unless it is one
    # of the pinned read-only roots; every other root is a read-only
    # fallback. Each is tagged for what it is, so the root a caller gets back
    # says whether it is the workspace's own
    library = library_path_from_env(WORKFLOWS_KIND, confine_to or None)
    primary = library.roots()[0] if confine_to else None
    roots = [root.root for root in library.roots()]

    tried = []
    if os.path.isabs(path):
        candidate = os.path.normpath(path)
        tried.append(candidate)
        holder = library.root_for_path(candidate)
        if holder is not None and os.path.isfile(candidate):
            return candidate, holder
        if confine_to:
            # A real confinement boundary was named and nothing on the
            # search path holds this candidate - refuse here, naming every
            # root this resolution consulted, rather than handing an
            # unqualified path back to validate_workflow_path for a refusal
            # that names only the rejected path and not where it looked
            # (#422)
            raise PathTraversalError(
                f"Path outside every workflow source: {candidate}. Looked in: "
                + _describe_roots(roots)
            )
        # Unconfined (a bare CLI run naming no workflow_dir) - hand it back
        # as before and let validate_workflow_path's base=None passthrough
        # decide, since there is no boundary to report roots for
        return candidate, None

    if base_dir:
        # The root this candidate would be handed back with: the caller's
        # confinement when it named one, else the catalog root the run itself
        # would confine to. Containment is checked before the stat, so a name
        # that climbs out of the catalog is never even looked at - which is
        # what the callers' own validate_workflow_path caught a step too late,
        # leaving this `os.path.isfile` the dw/path-injection query's only
        # unguarded sink. A climb out is refused rather than reported as
        # absent, the distinction the search path below keeps: the
        # PathTraversalError propagates to the caller
        root = confine_to or catalog_root(base_dir)
        search_roots = [root] + [r for r in roots if r != root]
        for name in _candidate_names(path):
            # normpath first: validate_path refuses a '..' outright, so a
            # climb that stays inside the root has to be collapsed before it
            # is judged. Its return value is what gets stat'ed - and being
            # the validator's own, it is contained by construction
            try:
                candidate = validate_path(
                    os.path.normpath(os.path.join(base_dir, name)),
                    root,
                    allow_create=True,
                )
            except PathTraversalError as e:
                # Same refusal, naming the search path rather than only the
                # resolved (rejected) path (#422)
                raise PathTraversalError(
                    f"{e} Looked in: {_describe_roots(search_roots)}"
                ) from e
            tried.append(candidate)
            if os.path.isfile(candidate):
                return candidate, primary

    for root in library.roots():
        # allow_create, because what is being asked is where the name would
        # land rather than whether something is there - None means the name
        # traverses out of the root, and the file check is the next line
        candidate = library.path_in(root, path, allow_create=True)
        if candidate is None:
            tried.append(f"{os.path.join(root.root, path)} (outside the root)")
            continue
        tried.append(candidate)
        if os.path.isfile(candidate):
            return candidate, root

    raise SubWorkflowNotFound(path, tried)


def resolve_sub_workflow_reference(path, base_dir, confine_to):
    """Where one sub-workflow step's `path` resolves to, validated and
    confined, as (path, root) - `root` is the directory the child is confined
    to (None when the run is unconfined). The one preamble every site that
    asks shares: a run (`create_step_action`), validation, the realized
    workflow's digest and the observed-cost lookup, so a path one can open is
    a path the others can.

    `base_dir` is the directory of the workflow that names the step;
    `confine_to` its `workflow_dir`.

      - `builtin:<name>.json` is looked up only in `builtin_root()`, the
        packaged workflows, and confined to it whatever `confine_to` is
      - any other relative path in an unconfined run is confined to the
        catalog root (`catalog_root`) so it can reach a sibling folder
        but not leave the catalog
      - everything else goes through `resolve_sub_workflow`'s search path

    Raises SubWorkflowNotFound, SecurityError or InvalidInputError, each
    carrying the message the run itself would fail with.
    """
    builtin_name = references.ref_name(references.BUILTIN, path)
    if builtin_name is not None:
        if (
            not builtin_name.endswith(".json")
            or "/" in builtin_name
            or "\\" in builtin_name
        ):
            raise InvalidInputError(
                f"Invalid builtin workflow name: {builtin_name}. It must "
                "be a bare '<name>.json' filename with no path segments - "
                f"'builtin:' only looks in the packaged workflows root: "
                f"{builtin_root()}"
            )
        root = builtin_root()
        resolved = os.path.join(root, builtin_name)
        # The name is bare, so it lands in `root`; the validator's answer is
        # what gets stat'ed
        candidate = validate_path(resolved, root, allow_create=True)
        if not os.path.isfile(candidate):
            raise SubWorkflowNotFound(path, [resolved])
        return validate_workflow_path(resolved, root), root
    if confine_to is None and not os.path.isabs(path):
        confine_to = catalog_root(base_dir)
    resolved, library_root = resolve_sub_workflow(path, base_dir, confine_to)
    confine_to = library_root.root if library_root else None
    return validate_workflow_path(resolved, confine_to), confine_to
