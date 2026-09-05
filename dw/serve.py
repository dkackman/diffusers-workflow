"""Run the workflow engine as a local HTTP server.

    python -m dw.serve
    python -m dw.serve --port 8765 --workflow-dir ./workflows

Binds to localhost by default - this serves your GPU to your own tools,
not to the network. Interactive API docs at http://127.0.0.1:8765/docs
"""

import argparse
import multiprocessing
import os
import sys

# Spawn start method before anything touches multiprocessing (CUDA/MPS)
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Serve diffusers workflows over HTTP.")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)"
    )
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    parser.add_argument(
        "--workspace",
        default=None,
        help="Directory holding the workflows, prompts, assets and outputs "
        "this server serves (default: DW_WORKSPACE, else the 'workspace' "
        "setting, else the working directory when it looks like a "
        "workspace, else ~/diffusers-workspace). --workflow-dir, "
        "--output-dir and --prompt-dir each override one of its folders",
    )
    parser.add_argument(
        "--workflow-dir",
        default=None,
        help="Directory of workflow JSON files (default: the workspace's "
        "workflows/)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory results are written to (default: the workspace's " "outputs/)",
    )
    parser.add_argument(
        "--prompt-dir",
        default=None,
        help="Directory of stored prompt files (default: discovered the way "
        "a CLI run discovers it - DW_PROMPT_DIR, else ./prompts if it "
        "exists, else the nearest prompts/ above the workflow directory)",
    )
    parser.add_argument(
        "--asset-dir",
        default=None,
        help="Directory of input media 'asset:' references resolve against, "
        "and where browser uploads are saved (default: the workspace's "
        "assets/)",
    )
    parser.add_argument(
        "--output-layout",
        choices=("run", "flat"),
        default=None,
        help="'run' (default) gives each job its own directory under "
        "<output_dir>/<workflow>/, with a manifest.json beside its files; "
        "'flat' writes the way it did before run directories",
    )
    parser.add_argument(
        "-l",
        "--log_level",
        default="INFO",
        help="DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Static bearer token required on every /api/ request "
        "(default: DW_API_TOKEN env var, else no authentication)",
    )
    parser.add_argument(
        "--trust-workflows",
        action="store_true",
        default=False,
        help="Trust every workflow this server loads - inline (from a "
        "POST /api/jobs body, e.g. MCP-authored), a stored one, or one at "
        "an arbitrary path - to execute arbitrary Python: allow "
        "pre_load_modules and any dotted *_type/*_dtype/dtype/config_type "
        "value, not just ones inside the diffusers/torch/transformers/"
        "quantization-backend ecosystem the tool already depends on. Off "
        "by default - see docs/SECURITY.md's Trust model. A server "
        "anything can submit jobs to (including an MCP client) should "
        "stay untrusted.",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        default=False,
        help="Also serve the MCP tool surface at /mcp over Streamable HTTP, "
        "behind the same token as /api, so an agent on another machine "
        "needs no local install (`claude mcp add --transport http dw "
        "http://<host>:<port>/mcp --header 'Authorization: Bearer <token>'`). "
        "Refused on a non-loopback --host without a token.",
    )
    args = parser.parse_args()

    # Resolved and pinned before anything derives a directory from it - the
    # spawned worker inherits the environment variable, the way it inherits
    # the prompt directory and the trust flag below
    from .workspace import resolve_workspace, set_workspace

    workspace = set_workspace(resolve_workspace(args.workspace))
    workflow_dir = args.workflow_dir or workspace.workflows
    output_dir = args.output_dir or workspace.outputs
    asset_dir = os.path.abspath(args.asset_dir or workspace.assets)

    # A workspace's workflow folder is where the UI and MCP clients save, so
    # it has to exist for a first run in a fresh workspace. Only the folder
    # actually defaulted to is created - an explicit --workflow-dir that does
    # not exist stays the operator's typo, not a new empty directory. The
    # output folder is created by the job manager on the same reasoning
    if not args.workflow_dir:
        os.makedirs(workflow_dir, exist_ok=True)

    # Created either way: it is the upload destination and a static mount,
    # both of which need it to exist before the first request
    os.makedirs(asset_dir, exist_ok=True)

    # Pinned like the prompt directory, so 'asset:' resolves to the same
    # library in the worker that the upload route writes into
    os.environ["DW_ASSET_DIR"] = asset_dir

    if args.output_layout:
        from .runs import set_output_layout

        set_output_layout(args.output_layout)

    token = args.token or os.environ.get("DW_API_TOKEN") or None

    from .server.app import LOOPBACK_HOSTS

    # A hard error, where the REST-only case below is a warning: an MCP
    # endpoint can author and run workflows, and unlike the web UI there is
    # no page to paste a token into. Raised before startup() and before the
    # worker subprocess is ever spawned.
    if args.mcp and args.host not in LOOPBACK_HOSTS and not token:
        print(
            f"dw-serve: --mcp on {args.host} needs a token. An MCP endpoint "
            "can author and run workflows, and unlike the web UI there is "
            "no page to type a token into - pass --token or set "
            "DW_API_TOKEN, or bind to 127.0.0.1.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    # Set before create_app / before the worker subprocess is ever spawned -
    # 'spawn' launches a fresh interpreter that inherits this environment
    # variable, so the job runner sees the same trust choice the API does
    from .security import set_trust_workflows

    set_trust_workflows(args.trust_workflows)

    # Resolve the prompt directory once, with the same discovery a CLI run
    # uses, anchored at the workflow directory - then pin it, so the library
    # the Prompts page edits and the one 'prompt:' references resolve against
    # are always the same directory. The spawned worker inherits the variable.
    from .prompts import get_prompt_dir

    prompt_dir = os.path.abspath(
        args.prompt_dir or get_prompt_dir(base_dir=os.path.abspath(workflow_dir))
    )
    os.environ["DW_PROMPT_DIR"] = prompt_dir

    try:
        import uvicorn
    except ImportError:
        print(
            "The server needs fastapi and uvicorn: pip install fastapi 'uvicorn[standard]'"
        )
        raise SystemExit(1)

    from . import startup

    startup(args.log_level)

    import logging

    logger = logging.getLogger("dw")

    if args.host not in LOOPBACK_HOSTS and not token:
        logger.warning(
            "Binding to %s with no API token configured (--token or "
            "DW_API_TOKEN) - anything that can reach this address can "
            "queue jobs, read and write workflows/prompts, and browse "
            "generated output. Set a token, or bind to 127.0.0.1 if this "
            "server does not need to be reachable off this machine.",
            args.host,
        )

    from .server.app import create_app

    from .server.app import default_ui_dir

    app = create_app(
        # absolute, so the path the UI hands back on submit is unambiguous
        workflow_dir=os.path.abspath(workflow_dir),
        output_dir=output_dir,
        log_level=args.log_level,
        prompt_dir=prompt_dir,
        asset_dir=asset_dir,
        workspace=workspace.root,
        host=args.host,
        token=token,
        mcp=args.mcp,
        port=args.port,
    )
    ui = " - UI at /" if default_ui_dir() else ""
    mcp = " - MCP at /mcp" if args.mcp else ""
    print(
        f"diffusers-workflow server on http://{args.host}:{args.port}"
        f"  (docs at /docs{ui}{mcp})",
        # stdout is a pipe under systemd or nohup, where block buffering
        # would otherwise hold this line back until shutdown
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
