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
        "--prompt-dir", default="./prompts", help="Directory of stored prompt files"
    )
    parser.add_argument(
        "-l",
        "--log_level",
        default="INFO",
        help="DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    args = parser.parse_args()

    # The engine resolves 'prompt:' references through this variable, and the
    # spawned worker process inherits it - one setting covers both
    os.environ["DW_PROMPT_DIR"] = os.path.abspath(args.prompt_dir)

    try:
        import uvicorn
    except ImportError:
        print(
            "The server needs fastapi and uvicorn: pip install fastapi 'uvicorn[standard]'"
        )
        raise SystemExit(1)

    from . import startup

    startup(args.log_level)

    from .server.app import create_app

    from .server.app import default_ui_dir

    app = create_app(
        workflow_dir=args.workflow_dir,
        output_dir=args.output_dir,
        log_level=args.log_level,
        prompt_dir=args.prompt_dir,
    )
    ui = " - UI at /" if default_ui_dir() else ""
    print(
        f"diffusers-workflow server on http://{args.host}:{args.port}  (docs at /docs{ui})"
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
