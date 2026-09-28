"""python -m dw.run: a CLI client of a running dw.serve.

Every job used to run in-process, its own copy of workflow loading,
validation and the security checks duplicating what the server's admission
(`dw/server/admission.py`) already does for every other caller. This module
now only talks HTTP: it queues a job, prints its progress, and reports the
result - the same path the web UI and MCP server use. It never starts a
server; run one first with `python -m dw.serve`.
"""

import argparse
import json
import os
import sys
import time

from dw_mcp.client import (
    DwApiError,
    DwClient,
    api_path,
    resolve_base_url,
    resolve_token,
)
from .security import (
    validate_variable_name,
    validate_string_input,
    SecurityError,
    MAX_VARIABLE_VALUE_LENGTH,
)

# Matches dw_mcp.diagnose.WAIT_POLL_SECONDS / dw.server.app.SSE_POLL_SECONDS -
# the cadence every other poller in this codebase already uses.
POLL_SECONDS = 1.0

TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})

# The exact prefix of dw/server/app.py's resolve_workflow_reference() 400 -
# the one case where resending the file inline is the right fallback rather
# than a caller mistake to report as-is. Matching text is brittle, but the
# alternative is a server change this module does not make.
_UNREACHABLE_PREFIX = "workflow_path must name a workflow the server can reach"


class _CliError(Exception):
    """A name=value pair that failed local validation - reported before any
    request is made."""


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="dw-run", description="Run a workflow against a running dw.serve."
    )
    parser.add_argument(
        "workflow",
        help="A catalog name, a path the server resolves, or a local workflow file",
    )
    parser.add_argument(
        "variables",
        nargs="*",
        help="Optional parameters in name=value format",
    )
    parser.add_argument(
        "--server",
        default=None,
        help="Base URL of dw.serve (default: DW_MCP_URL, else http://127.0.0.1:8765)",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Which of the server's workspaces to run in (default: the "
        "server's default workspace)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Bearer token for an authenticated dw.serve (default: DW_API_TOKEN)",
    )
    return parser


def _parse_variables(pairs):
    """Validated name=value pairs, as string arguments - the server coerces
    them exactly as the in-process run used to."""
    variables = {}
    for pair in pairs:
        if "=" not in pair:
            raise _CliError(f"Error: Variable '{pair}' is not in name=value format")
        name, value = pair.split("=", 1)
        try:
            validated_name = validate_variable_name(name.strip())
            validated_value = validate_string_input(
                value.strip(), max_length=MAX_VARIABLE_VALUE_LENGTH, allow_empty=True
            )
        except SecurityError as e:
            raise _CliError(f"Error: Invalid variable input: {e}")
        variables[validated_name] = validated_value
    return variables


def _cannot_reach_path(error):
    return error.status_code == 400 and str(error).startswith(_UNREACHABLE_PREFIX)


def _submit(client, workflow_arg, arguments):
    """Queue the job. A local file is sent by its absolute path first, so
    run-directory identity and its own relative paths stay right; only when
    the server answers that it cannot reach that path is the file's JSON
    resent inline, after printing a notice - an unrecognized name or a
    server-side path is never retried this way, since there is no local
    file behind it to send."""
    is_local_file = os.path.isfile(workflow_arg)
    workflow_path = os.path.abspath(workflow_arg) if is_local_file else workflow_arg
    payload = {"workflow_path": workflow_path, "arguments": arguments}
    try:
        return client.post_json(api_path("api", "jobs"), payload)
    except DwApiError as e:
        if is_local_file and _cannot_reach_path(e):
            print(
                f"note: the server cannot read {workflow_path}; sent its "
                "definition inline - relative paths in it resolve against "
                "the server's workflows/"
            )
            with open(workflow_arg, "r", encoding="utf-8") as f:
                definition = json.load(f)
            return client.post_json(
                api_path("api", "jobs"),
                {"workflow": definition, "arguments": arguments},
            )
        raise


def _print_event(event):
    kind = event.get("event")
    if kind == "step_start":
        print(f"step_start: {event.get('step')}")
    elif kind == "step_end":
        print(f"step_end: {event.get('step')}")
    elif kind == "warning":
        print(f"warning: {event.get('message')}")


def _wait_for_completion(client, job_id):
    last_seq = -1
    while True:
        page = client.get_json(
            api_path("api", "jobs", job_id, "event-log"), params={"after": last_seq}
        )
        for event in page.get("events", []):
            last_seq = event.get("seq", last_seq)
            _print_event(event)
        if page.get("status") in TERMINAL_STATUSES:
            break
        time.sleep(POLL_SECONDS)
    return client.get_json(api_path("api", "jobs", job_id))


def _report(detail):
    status = detail.get("status")
    print(f"status: {status}")
    print(f"run_dir: {detail.get('run_dir')}")
    for entry in detail.get("manifest") or []:
        for file_path in entry.get("files", []):
            print(file_path)
    if status == "succeeded":
        return 0
    print(f"error: {detail.get('error')}")
    return 1


def main(argv=None, client=None):
    args = _build_parser().parse_intermixed_args(argv)

    try:
        arguments = _parse_variables(args.variables)
    except _CliError as e:
        print(e)
        return 1

    owns_client = client is None
    if client is None:
        client = DwClient(
            base_url=resolve_base_url(args.server),
            token=resolve_token(args.token),
            workspace=args.workspace,
        )

    try:
        try:
            job = _submit(client, args.workflow, arguments)
        except DwApiError as e:
            if e.status_code is None:
                print(
                    f"error: no dw.serve at {client.base_url} - start one "
                    "with: python -m dw.serve"
                )
                return 2
            if e.status_code in (401, 403):
                print(
                    "error: the server refused the token (set DW_API_TOKEN "
                    "or pass --token)"
                )
                return 2
            print(f"Error: {e}")
            return 1

        job_id = job["id"]
        try:
            detail = _wait_for_completion(client, job_id)
        except KeyboardInterrupt:
            try:
                client.post_json(api_path("api", "jobs", job_id, "cancel"), {})
            except DwApiError:
                pass
            print("cancelled")
            return 130

        return _report(detail)
    finally:
        if owns_client:
            client.close()


if __name__ == "__main__":
    sys.exit(main())
