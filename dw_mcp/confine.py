"""Where the mounted MCP surface may read and write files.

Served by `dw.serve --mcp`, a path a caller names is a path on the
operator's box rather than on the caller's own machine, so a read
(upload_asset's file_path) or a write (download_output's destination) is
confined to the directories the server works in. A stdio `dw-mcp` runs on
the user's own machine and is not confined. Both directions use this one
module: two copies of it drifted once, when a per-call workspace reached
the write side only (#389).
"""

import os

from dw_mcp.client import DwApiError


def _real(path):
    return os.path.normpath(
        os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))
    )


def remote_roots(client, workspace, keys, writable_libraries=False, *, refusal):
    """The directories a mounted surface is confined to, or None for a stdio
    client. `keys` are the `/api/server` directories that count; with
    `writable_libraries`, every writable library `/api/assets` lists counts
    too (the shared asset library is only visible there). `workspace` is the
    caller's per-call pin, sent to both routes. A mounted server that names
    none is refused with `refusal`, the caller's own message."""
    if not getattr(client, "mounted", False):
        return None
    directories = (
        client.get_json("/api/server", workspace=workspace).get("directories") or {}
    )
    values = [directories.get(key) for key in keys]
    if writable_libraries:
        try:
            libraries = (
                client.get_json("/api/assets", workspace=workspace).get("libraries")
                or []
            )
        except DwApiError:
            libraries = []
        values += [
            library.get("root")
            for library in libraries
            if isinstance(library, dict) and library.get("writable")
        ]
    roots = []
    for value in values:
        if value and _real(value) not in roots:
            roots.append(_real(value))
    if not roots:
        raise DwApiError(refusal)
    return roots


def contains(path, roots):
    """Whether `path`, resolved, lies in one of `roots`. Resolved through the
    real path of its nearest existing ancestor - the file itself usually
    does not exist yet, and realpath of a missing path leaves symlinks in
    its existing prefix unresolved on some platforms - so neither a '~', an
    absolute path nor a symlink carries it out."""
    probe = path
    while not os.path.exists(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    resolved = os.path.normpath(
        os.path.join(os.path.realpath(probe), os.path.relpath(path, probe))
    )
    return any(resolved == root or resolved.startswith(root + os.sep) for root in roots)
