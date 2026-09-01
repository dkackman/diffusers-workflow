"""`python -m dw.mcp` / `dw-mcp`: serve the tool surface over stdio."""

import argparse
import sys

from dw.mcp.client import DwClient, resolve_base_url
from dw.mcp.server import build_server


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="dw-mcp",
        description="MCP server for diffusers-workflow. Requires a running "
        "dw.serve - start one with `dw-serve` first.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Base URL of the running dw.serve "
        "(default: $DW_MCP_URL, else http://127.0.0.1:8765)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds to wait on any one API request (default: 30)",
    )
    args = parser.parse_args(argv)

    client = DwClient(base_url=resolve_base_url(args.url), timeout=args.timeout)
    try:
        build_server(client).run(transport="stdio")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
