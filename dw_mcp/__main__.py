"""`python -m dw_mcp` / `dw-mcp`: serve the tool surface over stdio."""

import argparse
import sys

import httpx

from dw_mcp.client import (
    DwApiError,
    DwClient,
    is_loopback_url,
    resolve_base_url,
    resolve_token,
)
from dw_mcp.server import build_server


def _refuse(message):
    print(f"dw-mcp: {message}", file=sys.stderr)
    return 2


def main(argv=None, transport=None):
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
        "--token",
        default=None,
        help="Bearer token the dw.serve was started with, if any "
        "(default: $DW_API_TOKEN). Required when --url is not loopback.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Which of the server's workspaces to work in (default: "
        "$DW_MCP_WORKSPACE, else the server's default). A name on the "
        "server, not a directory here - list_workspaces shows them",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds to wait on any one API request (default: 30)",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        default=False,
        help="Skip the startup GET /api/health that confirms the server is "
        "reachable and the token is accepted",
    )
    args = parser.parse_args(argv)

    base_url = resolve_base_url(args.url)
    token = resolve_token(args.token)

    # A remote dw.serve with no token would let anyone on that network run
    # workflows as this user; refusing here is the client-side half of the
    # server's own non-loopback-without-token warning.
    if not is_loopback_url(base_url) and not token:
        return _refuse(
            f"{base_url} is not a loopback address and no token is set. "
            "A dw.serve reachable from other machines must be started with "
            "--token / DW_API_TOKEN, and the same token passed here with "
            "--token or DW_API_TOKEN."
        )

    client = DwClient(
        base_url=base_url,
        timeout=args.timeout,
        token=token,
        transport=transport,
        workspace=args.workspace,
    )
    try:
        if not args.no_probe:
            # A remote URL or token that is wrong is a misconfiguration the
            # user must fix before anything works, so it is fatal. On
            # loopback it usually means "dw.serve is not up yet", which a
            # tool call reports for itself with the same message - warn and
            # serve anyway rather than making the agent restart us.
            code = _probe(client, base_url, required=not is_loopback_url(base_url))
            if code:
                return code
        build_server(client).run(transport="stdio")
    finally:
        client.close()
    return 0


def _probe(client, base_url, required=True):
    """One GET /api/health so a wrong URL or token fails here, once, with a
    message - not on every tool call as an unexplained 401.

    Returns 2 when the probe failed and `required`; 0 otherwise. A failed
    probe always prints, so a loopback server that is merely not up yet
    still says so.
    """
    try:
        health = client.get_json("/api/health")
    except (DwApiError, httpx.HTTPError) as e:
        text = str(e)
        if "401" in text or "token" in text.lower():
            code = _refuse(
                f"dw.serve at {base_url} requires a bearer token and rejected "
                f"the one given (or none was given): {text}"
            )
        else:
            code = _refuse(f"could not reach dw.serve at {base_url}: {text}")
        return code if required else 0
    print(
        "dw-mcp: connected to {host} (dw {version}, {device}) at {url}".format(
            host=health.get("hostname", "?"),
            version=health.get("version", "?"),
            device=health.get("device", "?"),
            url=base_url,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
