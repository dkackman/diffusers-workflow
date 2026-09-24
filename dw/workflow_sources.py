"""Where workflows are read from, and the one place they are written to.

A workflow directory used to be a single directory that was both the library
and the place saves landed. With the repository's own workflows/ as that
directory - the default when a checkout is the workspace - every save from
the editor or an MCP client wrote into the example corpus.

A search path separates the two. Reads resolve front to back; writes only
ever go to the front:

    <workspace>/workflows/   the user's own, writable
    <examples dirs>          read-only, --examples-dir

A name found in an earlier source shadows the same name in a later one, so a
workspace copy of an example is the one that runs. Saving over a read-only
workflow is not an error and not an overwrite: it writes a copy into the
writable source, which is what "open an example, change it, save" should do.
"""

import logging
import os

from .security import SecurityError, validate_path

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


class WorkflowSource:
    """One root on the search path."""

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

    def names(self):
        """Workflow names under this root, as relative paths without .json."""
        return workflow_names(self.root)

    def to_dict(self):
        return {"root": self.root, "origin": self.origin, "writable": self.writable}

    def __repr__(self):
        return f"WorkflowSource({self.root!r}, {self.origin}, writable={self.writable})"


def workflow_names(root):
    """Workflow names under a root, as relative paths without .json."""
    names = []
    if not os.path.isdir(root):
        return names
    for directory, _dirs, files in os.walk(root):
        for file_name in files:
            if file_name.endswith(".json"):
                relative = os.path.relpath(os.path.join(directory, file_name), root)
                names.append(relative[: -len(".json")].replace(os.sep, "/"))
    return sorted(names)


def workflow_sources(workflow_dir, examples_dirs=None, include_builtin=False):
    """The search path: the writable directory first, then read-only roots.

    A read-only root that is the writable one - a checkout whose workflows/
    is both the workspace library and the examples - appears once, writable,
    rather than twice with two different answers about whether it can be
    saved to.

    The packaged workflows are off the path by default. They are the pieces
    a 'builtin:' sub-workflow step names, resolved by the engine where that
    step is read (dw/workflow.py) - not workflows anyone browses or runs on
    their own, and listing them would put a handful of fragments in front of
    every user who never asked for them.
    """
    sources = [WorkflowSource(workflow_dir, WORKSPACE_ORIGIN, True)]
    candidates = [(directory, EXAMPLES_ORIGIN) for directory in examples_dirs or []]
    if include_builtin:
        candidates.append((builtin_root(), BUILTIN_ORIGIN))

    seen = {sources[0].root}
    for root, origin in candidates:
        source = WorkflowSource(root, origin, False)
        if source.root in seen:
            continue
        seen.add(source.root)
        sources.append(source)
    return sources


def writable_source(sources):
    """The source saves go to: the front of the path."""
    for source in sources:
        if source.writable:
            return source
    return None


def source_for_path(sources, path):
    """Which source a resolved path belongs to, or None if it is outside
    every root - which is what makes a path a workflow rather than an
    arbitrary file."""
    for source in sources:
        if source.contains(path):
            return source
    return None


def resolve_in_source(source, name, allow_create=False):
    """The on-disk path a name has in one source, or None when the name does
    not resolve inside it. Containment is the security layer's, so a name
    that tries to traverse simply does not resolve."""
    if not name.endswith(".json"):
        name = f"{name}.json"
    try:
        return validate_path(
            os.path.join(source.root, name), source.root, allow_create=allow_create
        )
    except SecurityError:
        return None


def find_workflow(sources, name):
    """The first source that has this name, as (path, source).

    Front to back, so a workspace copy shadows the example it was copied
    from. (None, None) when no source has it.
    """
    for source in sources:
        path = resolve_in_source(source, name)
        if path and os.path.isfile(path):
            return path, source
    return None, None


def suggest_workflow_names(sources, name, limit=3):
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
    catalog_names = list(listing(sources).keys())
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


def listing(sources):
    """Every name the search path offers, each with the source it comes
    from - a name in an earlier source shadowing the same name later."""
    found = {}
    for source in sources:
        for name in source.names():
            found.setdefault(name, source)
    return dict(sorted(found.items()))


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


def resolve_sub_workflow(path, base_dir, confine_to):
    """Where a sub-workflow step's `path` resolves to, and the root the
    child is confined to, as (path, root).

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
    roots = []
    if confine_to:
        roots.append(os.path.abspath(os.path.expanduser(str(confine_to))))
    for root in fallback_roots(roots[0] if roots else None):
        if root not in roots:
            roots.append(root)

    tried = []
    if os.path.isabs(path):
        candidate = os.path.normpath(path)
        tried.append(candidate)
        for root in roots:
            source = WorkflowSource(root, EXAMPLES_ORIGIN, False)
            if source.contains(candidate) and os.path.isfile(candidate):
                return candidate, root
        # No root holds it - hand it back confined as it was, so the
        # security layer writes the refusal it always did
        return candidate, confine_to

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
        for name in _candidate_names(path):
            # normpath first: validate_path refuses a '..' outright, so a
            # climb that stays inside the root has to be collapsed before it
            # is judged. Its return value is what gets stat'ed - and being
            # the validator's own, it is contained by construction
            candidate = validate_path(
                os.path.normpath(os.path.join(base_dir, name)),
                root,
                allow_create=True,
            )
            tried.append(candidate)
            if os.path.isfile(candidate):
                return candidate, confine_to

    for root in roots:
        source = WorkflowSource(root, EXAMPLES_ORIGIN, False)
        # allow_create, because what is being asked is where the name would
        # land rather than whether something is there - None means the name
        # traverses out of the root, and the file check is the next line
        candidate = resolve_in_source(source, path, allow_create=True)
        if candidate is None:
            tried.append(f"{os.path.join(root, path)} (outside the root)")
            continue
        tried.append(candidate)
        if os.path.isfile(candidate):
            return candidate, root

    raise SubWorkflowNotFound(path, tried)
