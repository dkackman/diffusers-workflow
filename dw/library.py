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

from .security import (
    PathTraversalError,
    SecurityError,
    contained,
    validate_path,
    validate_workflow_path,
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


def builtin_root():
    """The packaged workflows that ship inside dw/ - what 'builtin:' names."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflows")


def catalog_root(directory):
    """The nearest ancestor of `directory` literally named 'workflows', else
    the directory itself.

    The root a run with no workflow_dir of its own confines a relative
    sub-workflow reference to, so a template under templates/ can still climb
    to a sibling models/ without leaving the catalog. `catalog_root_dir`
    (dw/workflow.py) is this rule asked for a file rather than a directory.
    """
    directory = os.path.normpath(os.path.abspath(directory))
    parts = directory.split(os.sep)
    try:
        index = len(parts) - 1 - parts[::-1].index("workflows")
    except ValueError:
        return directory

    return os.sep.join(parts[: index + 1])


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

    def __init__(self, name, root):
        self.name = name
        self.root = root
        super().__init__(
            f"'{name}' comes from the read-only {root.origin} directory {root.root}"
        )


# kind -> how a name becomes a file name under a root. Workflows and prompts
# are JSON documents; an asset's name is its relative path, literally
WORKFLOWS_KIND = "workflows"
PROMPTS_KIND = "prompts"
ASSETS_KIND = "assets"
_JSON_KINDS = {WORKFLOWS_KIND, PROMPTS_KIND}
LIBRARY_KINDS = (WORKFLOWS_KIND, PROMPTS_KIND, ASSETS_KIND)


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
                # The named validator CodeQL models for py/path-injection;
                # it also enforces the .json extension
                return validate_workflow_path(candidate, root.root)
            return validate_path(candidate, root.root, allow_create=allow_create)
        except SecurityError:
            return None

    def find(self, name):
        """The first root that has this name, as (path, root), or None.

        Front to back, so a workspace copy shadows the example it was copied
        from. Each candidate is confined to its own root, so a symlink that
        points outside it is a miss rather than a leak.
        """
        for root in self._roots:
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

    def writable_root(self, shared=False):
        """The root saves go to: the front of the path. `shared` asks for the
        shared (common) root instead, for the libraries that have one."""
        for root in self._roots:
            if root.writable and (root.origin == COMMON_ORIGIN) == shared:
                return root
        return None

    def require_writable(self, root, name=None):
        """The root itself, or `ReadOnlyLibraryError` when it is read-only."""
        if not root.writable:
            raise ReadOnlyLibraryError(name, root)
        return root


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


def library_path(
    kind, workspace, examples_dirs=None, primary=None, include_builtin=False
):
    """The search path of one library for a workspace.

    Workflows: the workspace's `workflows/` (writable), then each examples
    directory (read-only), then - when `include_builtin` asks - the packaged
    workflows. `primary` replaces the writable root, which is how a job's own
    directory names its path.

    A read-only root that is the writable one - a checkout whose workflows/ is
    both the workspace library and the examples - appears once, writable,
    rather than twice with two different answers about whether it can be
    saved to. A read-only root that is not a directory is dropped. The
    writable root is kept whether or not it exists yet: a save creates it.

    The packaged workflows are off the path by default. They are the pieces
    a 'builtin:' sub-workflow step names, resolved by the engine where that
    step is read (dw/workflow.py) - not workflows anyone browses or runs on
    their own, and listing them would put a handful of fragments in front of
    every user who never asked for them.
    """
    if kind != WORKFLOWS_KIND:
        raise ValueError(f"No search path is built for {kind} libraries yet")
    roots = [LibraryRoot(primary or workspace.workflows, WORKSPACE_ORIGIN, True)]
    candidates = [(directory, EXAMPLES_ORIGIN) for directory in examples_dirs or []]
    if include_builtin:
        candidates.append((builtin_root(), BUILTIN_ORIGIN))

    seen = {roots[0].root}
    for directory, origin in candidates:
        root = LibraryRoot(directory, origin, False)
        if root.root in seen or not os.path.isdir(root.root):
            continue
        seen.add(root.root)
        roots.append(root)
    return LibraryPath(kind, roots)


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


def fallback_roots(primary=None):
    """The read-only workflow roots a sub-workflow name is resolved against
    after the directory the run is confined to.

    Pinned in DW_WORKFLOW_PATH by dw.serve, the same way the prompt and
    asset libraries are, so the worker resolves a composed template exactly
    as the API would.
    """
    from .workspace import WORKFLOWS_SUBDIR, library_fallbacks

    return library_fallbacks(WORKFLOWS_SUBDIR, primary)


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
    primary = None
    if confine_to:
        primary = LibraryRoot(confine_to, WORKSPACE_ORIGIN, True)
    # The run's own root is the writable front of the path; every other root
    # is a read-only fallback. Each is tagged for what it is, so the root a
    # caller gets back says whether it is the workspace's own
    library = LibraryPath(
        WORKFLOWS_KIND,
        ([primary] if primary else [])
        + [
            LibraryRoot(root, EXAMPLES_ORIGIN, False)
            for root in fallback_roots(primary.root if primary else None)
            if not primary or root != primary.root
        ],
    )
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
