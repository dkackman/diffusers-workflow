# dw_mcp

Guidance for the `dw_mcp/` stdio MCP server package. What it covers and how its modules split is in the `dw_mcp/__init__.py` docstring and docs/MCP.md; the rules that hold across modules are in `docs/ARCHITECTURE.md`.

## Writing text an agent reads

- A tool description (the docstring in `dw_mcp/tools_*.py`) or the instructions in `dw_mcp/server.py` must put what an agent has to act on first, and point at a guide section rather than restate it. `tests/test_mcp_server.py` pins the length (`CLIENT_TEXT_LIMIT`, `SURFACE_BUDGET`) but not the ordering, and the instructions and `list_workflows` sit within a few dozen characters of the limit.
- `MAX_WAIT_SECONDS` (`dw_mcp/diagnose.py`) is interpolated into the tool descriptions, so no doc or skill quotes the number.
