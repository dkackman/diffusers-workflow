"""Start the server for the e2e suite against a throwaway copy of the repo's
example workflows and prompts.

The specs save and delete files through the UI. Pointing the server at the
real workflows/ and prompts/ folders meant a failed run could leave scratch
files in the repo - or, on a locator mismatch, delete a real one. The copy
keeps every example the specs look for while confining what they write.
The prompt directory is passed explicitly: the server would otherwise
discover ./prompts in the working directory - the real library. The scratch
directory is also the workspace, so the asset library the server creates and
uploads into lands there rather than in the checkout.

Extra arguments (--port ...) pass through to dw.serve. The process replaces
itself with the server so Playwright's shutdown signal reaches it directly.
"""

import os
import base64
import shutil
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
