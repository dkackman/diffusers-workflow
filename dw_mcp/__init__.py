"""MCP server for diffusers-workflow.

A stdio MCP server that is an HTTP client of a running `dw.serve`. It owns
no job state and no GPU worker - every tool is a call against the REST API
that the web UI already uses.

The surface covers the REST API except three things: the SSE event stream
(`get_job_events` reads its polling twin, `/event-log`), the two bulk zips of
the gallery and of the assets, and the SPA's static mount.
"""
