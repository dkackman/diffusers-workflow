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

from dw_mcp.client import DwApiError

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


def list_assets(client):
    """The input media on the server, each with the 'asset:' reference a
    workflow argument carries."""
    return client.get_json("/api/assets")


def keep_output(client, name, asset_name=None, overwrite=False):
    """Keep a generated file as an input asset, under a stable name.

    A run's files are named by the run that made them, which is the wrong
    thing to build on: 'latest' moves and a pinned run id breaks when
    outputs are pruned. Keeping one gives it an 'asset:' name that stays
    put, so a later workflow can rely on it.

    The copy happens on the server, inside the workspace - downloading a
    render here only to upload it back would move the bytes twice for
    nothing.
    """
    return client.post_json(
        "/api/assets/keep",
        {"name": name, "asset_name": asset_name, "overwrite": overwrite},
    )


def upload_asset(client, file_path):
    """Put a local image, video or audio file into the server's asset
    library and get back the reference a workflow can use.

    The file is read from the machine this MCP server runs on, which is not
    necessarily the machine dw.serve runs on - that is the point of the
    tool.
    """
    path = os.path.abspath(os.path.expanduser(str(file_path)))
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

    result = client.post_bytes(
        "/api/uploads", body, params={"filename": os.path.basename(path)}
    )
    # 'path' from a server with no asset library is an absolute path on that
    # machine; from one with a library it is already the reference. Report
    # whichever it gave, named for what it is
    return {
        "reference": result.get("path"),
        "url": result.get("url"),
        "uploaded": os.path.basename(path),
        "size": size,
    }
