"""Write the server's OpenAPI document, strict and normalized, to
ui/src/lib/generated/openapi.json - what `npm run gen:api` turns into the UI's
response types. `--check` exits 1 when the committed copy is stale;
`--stdout` prints instead of writing.

Normalized so it is the same on every machine and release: `info.version`
is "0", and the app is built with fixed directories and no token."""

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["DW_STRICT_RESPONSES"] = "1"  # before dw is imported: no index signatures

REPO = Path(__file__).resolve().parent.parent
# This checkout's dw, not whichever one the venv's editable install points
# at - a worktree sharing the main checkout's venv would otherwise dump the
# main checkout's contract
sys.path.insert(0, str(REPO))
OPENAPI_PATH = REPO / "ui" / "src" / "lib" / "generated" / "openapi.json"


def openapi_document() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        # dw creates its settings root on import, and the app's JobManager
        # opens the job history under it; point that at the scratch
        # directory first, so reading the schema never touches the history
        # a running dw.serve owns
        os.environ["DIFFUSERS_HELPER_ROOT"] = os.path.join(tmp, "helper")
        from dw.server.app import create_app

        app = create_app(
            workflow_dir=os.path.join(tmp, "workflows"),
            output_dir=os.path.join(tmp, "outputs"),
            prompt_dir=os.path.join(tmp, "prompts"),
        )
        try:
            document = app.openapi()
        finally:
            app.state.job_manager.shutdown()
    document["info"]["version"] = "0"
    return document


def render() -> str:
    return json.dumps(openapi_document(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    text = render()
    if "--stdout" in argv:
        sys.stdout.write(text)
        return 0
    if "--check" in argv:
        current = OPENAPI_PATH.read_text() if OPENAPI_PATH.exists() else ""
        if current != text:
            print(
                f"{OPENAPI_PATH.relative_to(REPO)} is stale: run python scripts/dump_openapi.py"
            )
            return 1
        return 0
    OPENAPI_PATH.parent.mkdir(parents=True, exist_ok=True)
    OPENAPI_PATH.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
