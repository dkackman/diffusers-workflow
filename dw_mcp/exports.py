"""Bundling a finished job so it can leave the server.

The one thing this module has to keep saying: the directory it makes is on
the machine running dw.serve, which over a `dw.serve --mcp` endpoint is the
GPU box and not where the agent is. The zip URL is the way to it from
anywhere else - but when the server requires a bearer token (#353), that URL
is for the person to open, not for this agent to fetch on their behalf; see
`export_job`'s `auth_required` / `open_url`.
"""

from dw_mcp.client import api_path


def export_job(client, job_id, overwrite=False):
    """Gather one finished job into a directory on the machine running
    dw.serve: workflow.json (realized), manifest.json, job.json, README,
    assets/, inputs/, outputs/. The export copies every output and input
    file rather than linking them, so a video job's export costs its size
    again on the server's disk; `total_bytes` in the result reports what
    was copied. Returns the directory, the zip URL(s), the file list with
    sizes and the total. The three JSON files are in the zip, not repeated
    here. The directory is on the server machine, not this one.

    `auth_required` says whether the zip needs this server's bearer token
    to open - a token this agent has no way to attach to a browser or hand
    to someone else's tooling. When it is true, `open_url` is for the
    *person* to open, not for this agent to fetch: hand it to them (see
    `next`). When it is false, `open_url` may be fetched directly. It is
    `absolute_zip_url` when the server has one configured (`DW_PUBLIC_URL`
    / the `public_url` setting), else the relative `zip_url`."""
    body = client.post_json(
        api_path("api", "jobs", job_id, "export"),
        params={"overwrite": "true" if overwrite else "false"},
    )
    directory = body.get("directory")
    zip_url = body.get("zip_url")
    absolute_zip_url = body.get("absolute_zip_url")
    auth_required = bool(body.get("auth_required"))
    if auth_required:
        open_url = absolute_zip_url or zip_url
        next_text = (
            "The directory is on the server, and the zip is behind this "
            "server's bearer token - hand open_url to the person and let "
            "them open it themselves; do not fetch it. "
            + (
                "It is already absolute."
                if absolute_zip_url
                else "It is relative - tell the person the server's own "
                "address, since none is configured (DW_PUBLIC_URL/public_url)."
            )
            + " Individual results stay reachable inline via "
            "get_output_image/get_output_audio/get_output_frames without "
            "opening the zip at all. workflow.json, manifest.json and "
            "job.json are inside it - they are not repeated here; "
            "get_job_workflow and get_job serve them individually."
        )
    else:
        open_url = absolute_zip_url or zip_url
        next_text = (
            "The directory is on the server. To give the user the files, "
            "fetch open_url and unpack it into exports/ under the session's "
            "working directory - it is the user's deliverable, not a "
            "temporary file, so not a scratch or temp directory. The archive "
            "already unpacks into one folder named after the job id; do not "
            "create that folder first or the id is doubled in the path. "
            "workflow.json, manifest.json and job.json are inside it - they "
            "are not repeated here; get_job_workflow and get_job serve them "
            "individually."
        )
    result = {
        "job_id": job_id,
        "where": f"{directory} on the machine running the MCP server",
        "directory": directory,
        "zip_url": zip_url,
        "auth_required": auth_required,
        "open_url": open_url,
        "files": body.get("files") or [],
        "total_bytes": body.get("total_bytes"),
        "missing": body.get("missing") or [],
        "next": next_text,
    }
    if absolute_zip_url is not None:
        result["absolute_zip_url"] = absolute_zip_url
    return result
