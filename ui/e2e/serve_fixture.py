"""Start the server for the e2e suite against a throwaway copy of the repo's
example workflows and prompts.

The specs save and delete files through the UI. Pointing the server at the
real workflows/ and prompts/ folders meant a failed run could leave scratch
files in the repo - or, on a locator mismatch, delete a real one. The copy
keeps every example the specs look for while confining what they write.
The prompt directory is passed explicitly: the server would otherwise
discover ./prompts in the working directory - the real library. The scratch
directory is also the workspace, so the asset library the server creates and
uploads into lands there rather than in the checkout. The settings root
(DIFFUSERS_HELPER_ROOT) is scratch too: it holds the job history the plan
prices from, and the developer's real one gave a box that had run a catalog
model an inherited-cost warning CI never sees, so a spec expecting a clean
validation passed in CI and failed locally.

Extra arguments (--port ...) pass through to dw.serve. The process replaces
itself with the server so Playwright's shutdown signal reaches it directly.
"""

import os
import base64
import json
import shutil
import sqlite3
import sys
import tempfile

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
scratch = tempfile.mkdtemp(prefix="dw-e2e-")
for name in ("workflows", "prompts"):
    shutil.copytree(os.path.join(root, name), os.path.join(scratch, name))
outputs = os.path.join(scratch, "outputs")
os.makedirs(outputs)
os.makedirs(os.path.join(scratch, "assets"))

# Two throwaway PNGs so the gallery specs have something to select. A 1x1
# image written by hand rather than by PIL - the fixture runs before the
# server and has no reason to import the imaging stack.
ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)
for name in ("e2e-one.png", "e2e-two.png"):
    with open(os.path.join(outputs, name), "wb") as handle:
        handle.write(ONE_PIXEL_PNG)

# Two runs of the same workflow, in the <identity>/<run id>/<file> shape a
# real execution writes - so the gallery has something to fold under one
# folder heading instead of one per run.
for run_id in ("20260906-101500-a1b2c3d4", "20260906-101612-ffffffff"):
    run_dir = os.path.join(outputs, "e2e-run-group", run_id)
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "still.png"), "wb") as handle:
        handle.write(ONE_PIXEL_PNG)

helper = os.path.join(scratch, "helper")
os.environ["DIFFUSERS_HELPER_ROOT"] = helper

# Two jobs a dead server's restart cancelled before they started: the
# tombstones the jobs page offers to rerun as one batch (#1473). They share
# the dead server's instance id, which is what groups them. The schema is the
# server's own, created through JobHistory, so the rows carry every column a
# real tombstone has.
sys.path.insert(0, root)
from dw.server.job_history import JobHistory  # noqa: E402
from dw.server.job_record import RESTART_PREFIX  # noqa: E402

os.makedirs(helper, exist_ok=True)
history_db = os.path.join(helper, "jobs.sqlite")
JobHistory(history_db)
template = os.path.join(scratch, "workflows", "templates", "text-to-image.json")
with sqlite3.connect(history_db) as connection:
    for index, job_id in enumerate(("e2erestart01", "e2erestart02")):
        connection.execute(
            "INSERT INTO jobs (id, workflow, status, created_at, finished_at,"
            " arguments, spec, error, events, workspace, server_instance)"
            " VALUES (?, ?, 'cancelled', ?, ?, '{}', ?, ?, '[]', 'default', ?)",
            (
                job_id,
                "text-to-image",
                1767225600.0 + index,
                1767225660.0,
                json.dumps({"workflow_path": template, "workspace": "default"}),
                f"{RESTART_PREFIX} queued, never started",
                "e2e-dead-server",
            ),
        )

os.chdir(root)
os.execv(
    sys.executable,
    [
        sys.executable,
        "-m",
        "dw.serve",
        "--workspace",
        scratch,
        "--workflow-dir",
        os.path.join(scratch, "workflows"),
        "--output-dir",
        outputs,
        "--prompt-dir",
        os.path.join(scratch, "prompts"),
        *sys.argv[1:],
    ],
)
