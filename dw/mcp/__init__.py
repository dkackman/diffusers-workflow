"""MCP server for diffusers-workflow.

A stdio MCP server that is an HTTP client of a running `dw.serve`. It owns
no job state and no GPU worker - every tool is a call against the REST API
that the web UI already uses.
"""

from dw import __version__

__all__ = ["__version__"]
