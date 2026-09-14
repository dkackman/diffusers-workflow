"""The asset library: the input media a workflow reaches by an 'asset:'
reference, and the one way to get a local file into it.

An agent authoring remotely can name assets that already exist on the box,
but until it can put bytes there it can only write workflows it has no way
to supply inputs for. `upload_asset` is that path: the file's bytes go up as
the request body - the same route the browser's file picker uses - and what
comes back is the reference, not a path, because the reference is what a
workflow carries and the path means nothing on the machine the agent is on.
"""

import os

from dw_mcp.client import DwApiError, api_path

# Twin of the server's own limit (dw/server/app.py). Checked here as well so
# a 200MB file fails before it is read and pushed, not after
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

# What the library holds, and what the upload route accepts. Duplicated from
# dw/security.py rather than imported: importing anything under dw/ pulls in
# torch, which this pure HTTP client must not do
ALLOWED_UPLOAD_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".webp",
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".webm",
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
    }
)


def _remote_roots(client):
    """The directories a remote read is confined to, or None when local.

    The mirror of media.py's `_remote_root` (#113), for the other direction.
    Only the mounted MCP surface is remote: there `upload_asset` runs inside
    dw.serve, so `file_path` names a file on the operator's box rather than
    on the calling agent's machine, and an unconfined read is an arbitrary
    file read plus a path-existence oracle (#138). A stdio `dw-mcp` returns
    None and keeps reading whatever the user can, because there "local file"
    is genuinely their own.
    """
    if not getattr(client, "mounted", False):
        return None

    directories = (client.get_json("/api/server").get("directories")) or {}
    roots = []
    for key in ("workspace", "workflows", "assets", "outputs", "prompts"):
        value = directories.get(key)
        if not value:
            continue
        resolved = os.path.normpath(
            os.path.realpath(os.path.abspath(os.path.expanduser(str(value))))
        )
        if resolved not in roots:
            roots.append(resolved)
    if not roots:
        raise DwApiError(
            "This server cannot say which directories it works in, so it "
            "will not read a file off its own disk for you. Upload the "
            "bytes through the web UI's file picker, or keep a generated "
            "file with keep_output."
        )
    return roots


def _confine_source(path, roots, named):
    """Refuse a source outside `roots`, before anything looks at the file.

    Ordered ahead of the existence and extension checks on purpose: a
    refusal that depends on whether the file is there turns the tool into a
    path-existence oracle for the whole box, which is the condition this
    closes as much as the read itself (#138). Containment is on the resolved
    real path, so a symlink cannot carry the read out.
    """
    probe = path
    while not os.path.exists(probe) and os.path.dirname(probe) != probe:
        probe = os.path.dirname(probe)
    resolved = os.path.normpath(
        os.path.join(os.path.realpath(probe), os.path.relpath(path, probe))
    )
    if any(resolved == root or resolved.startswith(root + os.sep) for root in roots):
        return
    raise DwApiError(
        f"Refusing to read {named} - this MCP endpoint is served by "
        f"dw.serve, so the file would be read off the server, where a "
        f"source is confined to the directories it works in "
        f"({', '.join(roots)}). A file that is already there is reachable "
        f"as an 'asset:' reference; to put a new one there, upload it "
        f"through the web UI's file picker, or promote a generated file "
        f"with keep_output."
    )


def list_assets(client):
    """The input media on the server, each with the 'asset:' reference a
    workflow argument carries.

    Spans the whole search path: each entry's 'origin' says whether it is
    this workspace's own ('workspace'), the library every workspace shares
    ('common'), or one a read-only examples tree brought with it.
    """
    return client.get_json("/api/assets")


def delete_asset(client, name):
    """Remove one file from the asset library.

    Deletes from whichever library holds it - the workspace's own before
    the shared one, the order 'asset:' resolves in. An asset from a
    read-only examples library answers 403.
    """
    return client.delete_json(api_path("api", "assets", name))


def keep_output(
    client, name, asset_name=None, overwrite=False, shared=False, workspace=None
):
    """Keep a generated file as an input asset, under a stable name.

    A run's files are named by the run that made them, which is the wrong
    thing to build on: 'latest' moves and a pinned run id breaks when
    outputs are pruned. Keeping one gives it an 'asset:' name that stays
    put, so a later workflow can rely on it.

    The copy happens on the server, inside the workspace - downloading a
    render here only to upload it back would move the bytes twice for
    nothing.

    `shared` keeps it in the library every workspace under the server's
    root shares instead, which is where something a later episode in its
    own workspace has to reach belongs.
    """
    return client.post_json(
        "/api/assets/keep",
        {
            "name": name,
            "asset_name": asset_name,
            "overwrite": overwrite,
            "shared": shared,
        },
        workspace=workspace,
    )


def upload_asset(client, file_path, asset_name=None, shared=False):
    """Put a local image, video or audio file into the server's asset
    library and get back the reference a workflow can use.

    The file is read from the machine this MCP server runs on, which is not
    necessarily the machine dw.serve runs on - that is the point of the
    tool.

    Over a `dw.serve --mcp` endpoint that machine *is* the server, so there
    `file_path` is confined to the directories the server works in, and the
    refusal comes before the file is looked for so it cannot be used to
    probe which paths exist (#138). A stdio `dw-mcp` is unconfined, because
    there the file really is the caller's own.

    `asset_name` is the name it is stored under - 'cast/priya-voice.wav'
    rather than the random one an upload gets by default. A recurring cast
    referenced as 'asset:uploads/084eaecc....wav' in every workflow cannot
    be told apart without opening each file, which is the whole reason to
    name one (2026-09-11). The extension comes from the uploaded file when
    the name has none.

    `shared` puts it in the library every workspace shares rather than in
    the session's own, which is what a recurring cast needs: assets are
    per workspace, so a cast uploaded while making episode one was
    invisible from the workspace episode four was made in.
    """
    path = os.path.abspath(os.path.expanduser(str(file_path)))
    roots = _remote_roots(client)
    if roots is not None:
        _confine_source(path, roots, file_path)
    if not os.path.isfile(path):
        raise DwApiError(f"No such file: {file_path}")

    extension = os.path.splitext(path)[1].lower()
    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        raise DwApiError(
            f"{os.path.basename(path)} is not a kind the asset library takes "
            f"({', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))})."
        )

    size = os.path.getsize(path)
    if size > MAX_UPLOAD_BYTES:
        raise DwApiError(
            f"{os.path.basename(path)} is {size} bytes, over the "
            f"{MAX_UPLOAD_BYTES} byte upload limit."
        )

    try:
        with open(path, "rb") as handle:
            body = handle.read()
    except OSError as e:
        raise DwApiError(f"Could not read {file_path}: {e}")

    params = {"filename": os.path.basename(path)}
    if asset_name:
        params["asset_name"] = asset_name
    if shared:
        params["shared"] = "true"
    result = client.post_bytes("/api/uploads", body, params=params)
    # 'path' from a server with no asset library is an absolute path on that
    # machine; from one with a library it is already the reference. Report
    # whichever it gave, named for what it is
    return {
        "reference": result.get("path"),
        "url": result.get("url"),
        "uploaded": os.path.basename(path),
        "size": size,
    }
