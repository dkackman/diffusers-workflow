"""Run the workflow engine as a local HTTP server.

    python -m dw.serve
    python -m dw.serve --port 8765 --workflow-dir ./workflows

Binds to localhost by default - this serves your GPU to your own tools,
not to the network. Interactive API docs at http://127.0.0.1:8765/docs
"""

import argparse
import multiprocessing
import os

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
        "--workflow-dir", default="./workflows", help="Directory of workflow JSON files"
    )
    parser.add_argument(
        "--output-dir", default="./outputs", help="Directory results are written to"
    )
    parser.add_argument(
        "--prompt-dir",
        default=None,
        help="Directory of stored prompt files (default: discovered the way "
        "a CLI run discovers it - DW_PROMPT_DIR, else ./prompts if it "
        "exists, else the nearest prompts/ above the workflow directory)",
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
    args = parser.parse_args()

    token = args.token or os.environ.get("DW_API_TOKEN") or None

    # Resolve the prompt directory once, with the same discovery a CLI run
    # uses, anchored at the workflow directory - then pin it, so the library
    # the Prompts page edits and the one 'prompt:' references resolve against
    # are always the same directory. The spawned worker inherits the variable.
    from .prompts import get_prompt_dir

    prompt_dir = os.path.abspath(
        args.prompt_dir or get_prompt_dir(base_dir=os.path.abspath(args.workflow_dir))
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

    if args.host not in ("127.0.0.1", "localhost", "::1") and not token:
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
        workflow_dir=args.workflow_dir,
        output_dir=args.output_dir,
        log_level=args.log_level,
        prompt_dir=prompt_dir,
        host=args.host,
        token=token,
    )
    ui = " - UI at /" if default_ui_dir() else ""
    print(
        f"diffusers-workflow server on http://{args.host}:{args.port}  (docs at /docs{ui})"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
