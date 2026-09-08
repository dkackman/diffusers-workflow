"""Bundling a finished job so it can leave the server.

The one thing this module has to keep saying: the directory it makes is on
the machine running dw.serve, which over a `dw.serve --mcp` endpoint is the
GPU box and not where the agent is. The zip URL is the way to it from
anywhere else.
"""

from dw_mcp.client import api_path


def export_job(client, job_id, overwrite=False):
    """Gather one finished job into a directory on the machine running
    dw.serve: workflow.json (realized), manifest.json, job.json, README,
    assets/, inputs/, outputs/. The export copies every output and input
    file rather than linking them, so a video job's export costs its size
    again on the server's disk; `total_bytes` in the result reports what
    was copied. Returns the directory, the zip URL, the file list with
    sizes and the total, and the three JSON files inline. The directory is
    on the server machine, not this one - use the zip URL to fetch it
    elsewhere."""
    body = client.post_json(
        api_path("api", "jobs", job_id, "export"),
        params={"overwrite": "true" if overwrite else "false"},
    )
    directory = body.get("directory")
    return {
        "job_id": job_id,
        "where": f"{directory} on the machine running the MCP server",
        "directory": directory,
        "zip_url": body.get("zip_url"),
        "files": body.get("files") or [],
        "total_bytes": body.get("total_bytes"),
        "missing": body.get("missing") or [],
        "workflow": body.get("workflow"),
        "manifest": body.get("manifest"),
        "job": body.get("job"),
        "next": "Report the directory as a path on the server, and hand the "
        "user the zip URL if they want the files locally.",
    }
