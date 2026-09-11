"""Exporting a job over MCP: the directory is on the server, and the tool
says so - the lesson download_output taught."""

import httpx
import pytest

from dw_mcp import exports
from dw_mcp.client import DwApiError, DwClient

SUMMARY = {
    "job_id": "job-1",
    "directory": "/srv/studio/exports/job-1",
    "files": [
        {"path": "workflow.json", "bytes": 412},
        {"path": "outputs/still.png", "bytes": 90210},
    ],
    "total_bytes": 90622,
    "missing": [],
    "zip_url": "/exports/job-1.zip",
    "workflow": {"id": "w", "steps": []},
    "manifest": {"run_id": "20260908-120000-abcdef01"},
    "job": {"id": "job-1", "status": "succeeded"},
}


def scripted(routes):
    seen = []

    def handler(request):
        key = (request.method, request.url.path)
        seen.append({"key": key, "params": dict(request.url.params)})
        if key not in routes:
            return httpx.Response(404, json={"detail": f"unrouted {key}"})
        status, body = routes[key]
        return httpx.Response(status, json=body)

    return DwClient(transport=httpx.MockTransport(handler)), seen


def exporting(status=201, body=None):
    return scripted({("POST", "/api/jobs/job-1/export"): (status, body or SUMMARY)})


def test_it_returns_the_directory_the_zip_and_the_file_list():
    client, seen = exporting()

    result = exports.export_job(client, "job-1")

    assert result["job_id"] == "job-1"
    assert result["directory"] == "/srv/studio/exports/job-1"
    assert result["zip_url"] == "/exports/job-1.zip"
    assert result["total_bytes"] == 90622
    assert [entry["path"] for entry in result["files"]] == [
        "workflow.json",
        "outputs/still.png",
    ]
    assert len(seen) == 1


def test_the_three_json_files_stay_in_the_zip():
    """A music-video export inlined 55 KB of workflow, manifest and job row
    that the zip already carries and get_job_workflow / get_job already
    serve - it blew past the tool output limit. The listing says they are
    there; the bytes are not repeated."""
    client, _ = exporting()

    result = exports.export_job(client, "job-1")

    assert "workflow" not in result
    assert "manifest" not in result
    assert "job" not in result
    assert "get_job_workflow" in result["next"]


def test_it_says_where_the_directory_is():
    client, _ = exporting()

    result = exports.export_job(client, "job-1")

    assert result["where"] == (
        "/srv/studio/exports/job-1 on the machine running the MCP server"
    )


def test_overwrite_travels_as_a_query_parameter():
    client, seen = exporting()

    exports.export_job(client, "job-1", overwrite=True)

    assert seen[0]["params"]["overwrite"] == "true"


def test_a_409_reaches_the_model_as_a_readable_refusal():
    client, _ = exporting(status=409, body={"detail": "An export already exists"})

    with pytest.raises(DwApiError) as caught:
        exports.export_job(client, "job-1")

    assert "already exists" in str(caught.value)


def test_the_docstring_says_copying_costs_disk_and_names_total_bytes():
    # A caller reading only the handler's docstring has to learn this before
    # exporting a video job fills the server's disk a second time - the
    # export copies files rather than linking them.
    assert "copies every output and input" in exports.export_job.__doc__
    assert "total_bytes" in exports.export_job.__doc__


def test_the_next_hint_sends_the_zip_to_the_working_directory():
    """The drill showed an agent unpacking the export into its scratchpad
    and doubling the job id in the path; the hint is where that is steered."""
    client, _ = exporting()
    hint = exports.export_job(client, "job-1")["next"]
    assert "working directory" in hint
    assert "temp" in hint
    assert "do not create that folder" in hint
