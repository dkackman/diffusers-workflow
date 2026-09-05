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


def builtin_root():
    """The packaged workflows that ship inside dw/ - what 'builtin:' names."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflows")


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


def listing(sources):
    """Every name the search path offers, each with the source it comes
    from - a name in an earlier source shadowing the same name later."""
    found = {}
    for source in sources:
        for name in source.names():
            found.setdefault(name, source)
    return dict(sorted(found.items()))
