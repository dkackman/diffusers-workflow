"""Start the server for the e2e suite against a throwaway copy of the repo's
example workflows and prompts.

The specs save and delete files through the UI. Pointing the server at the
real workflows/ and prompts/ folders meant a failed run could leave scratch
files in the repo - or, on a locator mismatch, delete a real one. The copy
keeps every example the specs look for while confining what they write.
The prompt directory is passed explicitly: the server would otherwise
discover ./prompts in the working directory - the real library.

Extra arguments (--port ...) pass through to dw.serve. The process replaces
itself with the server so Playwright's shutdown signal reaches it directly.
"""

import os
import shutil
import sys
import tempfile

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
scratch = tempfile.mkdtemp(prefix="dw-e2e-")
for name in ("workflows", "prompts"):
    shutil.copytree(os.path.join(root, name), os.path.join(scratch, name))
os.makedirs(os.path.join(scratch, "outputs"))

os.chdir(root)
os.execv(
    sys.executable,
    [
        sys.executable,
        "-m",
        "dw.serve",
        "--workflow-dir",
        os.path.join(scratch, "workflows"),
        "--output-dir",
        os.path.join(scratch, "outputs"),
        "--prompt-dir",
        os.path.join(scratch, "prompts"),
        *sys.argv[1:],
    ],
)
