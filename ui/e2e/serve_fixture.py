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
import struct
import tempfile
import wave
import zlib
from datetime import datetime, timedelta, timezone

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
scratch = tempfile.mkdtemp(prefix="dw-e2e-")
for name in ("workflows", "prompts"):
    shutil.copytree(os.path.join(root, name), os.path.join(scratch, name))
outputs = os.path.join(scratch, "outputs")
os.makedirs(outputs)
os.makedirs(os.path.join(scratch, "assets"))

# The server's writable LoRA library is <root>/loras. Two entries for one
# made-up base - a trial and a rejected one - so the editor's catalog
# suggestions have a deterministic list, with the rejected entry to hide.
shutil.copytree(
    os.path.join(root, "ui", "e2e", "fixtures", "loras"),
    os.path.join(scratch, "loras"),
)

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

# The media the find-and-reuse journey looks for, in one run: a tall PNG whose
# `parameters` tEXt chunk (before IDAT, as the engine's embed_metadata writes
# it) holds the recipe, a small PNG with none, two seconds of silence and a
# three-line note. Written with the standard library, as the fixture runs
# before the server and imports no imaging stack.
FIND_RUN = "20260910-120000-0f1d0f1d"


def png_chunk(kind, data):
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def solid_png(width, height, rgb, text=None):
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = (b"\x00" + bytes(rgb) * width) * height
    chunks = [png_chunk(b"IHDR", header)]
    if text is not None:
        chunks.append(png_chunk(b"tEXt", b"parameters\x00" + text.encode("latin-1")))
    chunks += [png_chunk(b"IDAT", zlib.compress(rows)), png_chunk(b"IEND", b"")]
    return b"\x89PNG\r\n\x1a\n" + b"".join(chunks)


find_dir = os.path.join(outputs, "e2e-find", FIND_RUN)
os.makedirs(find_dir)
recipe = {
    "prompt": "a lighthouse at dusk",
    "seed": 4242,
    "model_name": "e2e/tiny",
    "arguments": {"prompt": "a lighthouse at dusk", "seed": 4242, "steps": 12},
}
with open(os.path.join(find_dir, "lighthouse.png"), "wb") as handle:
    handle.write(solid_png(400, 1600, (40, 90, 160), json.dumps(recipe)))
with open(os.path.join(find_dir, "harbour.png"), "wb") as handle:
    handle.write(solid_png(64, 64, (160, 120, 40)))
with wave.open(os.path.join(find_dir, "hum.wav"), "wb") as hum:
    hum.setnchannels(1)
    hum.setsampwidth(2)
    hum.setframerate(8000)
    hum.writeframes(b"\x00\x00" * 8000 * 2)
with open(os.path.join(find_dir, "notes.txt"), "w") as handle:
    handle.write("Keeper's log, first night\nthe lamp is lit\nnothing on the water\n")

helper = os.path.join(scratch, "helper")
os.environ["DIFFUSERS_HELPER_ROOT"] = helper

# The hub cache is scratch too, seeded with one made-up repo so the models
# journey has a row to delete without ever reaching the developer's cache.
# Set before the server import: huggingface_hub reads it when it loads.
hub_cache = os.path.join(scratch, "hub-cache")
os.environ["HF_HUB_CACHE"] = hub_cache
revision = "0123456789abcdef0123456789abcdef01234567"
repo_dir = os.path.join(hub_cache, "models--e2e--tiny")
os.makedirs(os.path.join(repo_dir, "snapshots", revision))
os.makedirs(os.path.join(repo_dir, "refs"))
with open(os.path.join(repo_dir, "snapshots", revision, "config.json"), "w") as handle:
    handle.write("{}")
with open(os.path.join(repo_dir, "refs", "main"), "w") as handle:
    handle.write(revision)

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

# The job that made them: a history row the job page and the viewer's recipe
# read, its manifest naming the pictures as the step's results.
find_started = 1767484810.0
with sqlite3.connect(history_db) as connection:
    connection.execute(
        "INSERT INTO jobs (id, workflow, status, created_at, started_at,"
        " finished_at, arguments, spec, manifest, error, events, workspace,"
        " run_id, run_dir)"
        " VALUES ('e2efind00001', 'text-to-image', 'succeeded', ?, ?, ?, ?, ?, ?,"
        " NULL, '[]', 'default', ?, ?)",
        (
            find_started - 10,
            find_started,
            find_started + 60,
            json.dumps({"prompt": "a lighthouse at dusk"}),
            json.dumps({"workflow_path": template, "workspace": "default"}),
            json.dumps(
                [
                    {
                        "step": "image",
                        "files": [
                            f"e2e-find/{FIND_RUN}/{name}"
                            for name in ("lighthouse.png", "harbour.png")
                        ],
                        "role": "final",
                    }
                ]
            ),
            FIND_RUN,
            f"e2e-find/{FIND_RUN}",
        ),
    )

# One request in the default workspace's ledger (requests.json at the
# workspace root), for the Requests pages: a done item with two succeeded
# jobs, each leaving a run with an image, and neither accepted; a planned
# item whose one dependency is done, so a job could be queued for it; a
# planned item waiting on that one; and an item dropped from the plan after
# a job that failed was linked to it. The ledger writes the file itself, so
# it has the shape the server reads; the jobs are history rows carrying the
# `request` and `item` a linked submit records.
from dw import ledger  # noqa: E402

REQUEST_ID = "e2e-launch"
linked = {
    "hero": [
        {
            "job": "e2ereqdone01",
            "status": "succeeded",
            "run_id": "20260907-090000-0badf00d",
        },
        {
            "job": "e2ereqdone02",
            "status": "succeeded",
            "run_id": "20260907-093000-0badf00d",
        },
    ],
    "cut": [{"job": "e2ereqcut001", "status": "failed", "run_id": None}],
}
with sqlite3.connect(history_db) as connection:
    for key, jobs in linked.items():
        for index, job in enumerate(jobs):
            done = job["status"] == "succeeded"
            run = os.path.join("e2e-request", job["run_id"]) if done else None
            if done:
                os.makedirs(os.path.join(outputs, run))
                with open(os.path.join(outputs, run, "hero.png"), "wb") as handle:
                    handle.write(ONE_PIXEL_PNG)
            started = 1767312010.0 + 1800 * index
            connection.execute(
                "INSERT INTO jobs (id, workflow, status, created_at, started_at,"
                " finished_at, arguments, spec, manifest, error, events, workspace,"
                " run_id, run_dir, request, item)"
                " VALUES (?, 'text-to-image', ?, ?, ?, ?, '{}', ?, ?, ?, '[]',"
                " 'default', ?, ?, ?, ?)",
                (
                    job["job"],
                    job["status"],
                    started - 10,
                    started,
                    started + 90,
                    json.dumps({"workflow_path": template, "workspace": "default"}),
                    json.dumps(
                        [
                            {
                                "step": "image",
                                "files": [f"{run}/hero.png"],
                                "role": "final",
                            }
                        ]
                        if done
                        else []
                    ),
                    None if done else "e2e: this attempt failed",
                    job["run_id"],
                    run,
                    REQUEST_ID,
                    key,
                ),
            )


def plan_item(key, title, **extra):
    return {"key": key, "title": title, "workflow": "text-to-image", **extra}


plan = {
    "title": "E2E launch posters",
    "goal": "Two finished posters",
    "prompt": "make me two posters for the launch",
    "why": "opened",
    "revision": 0,
    "items": [
        plan_item("hero", "Hero poster", estimate_minutes=2),
        plan_item("alt", "Alternate poster", depends_on=["hero"]),
        plan_item("sheet", "Contact sheet", depends_on=["alt"]),
        plan_item("cut", "Cut scene"),
    ],
}


def jobs_linked(_request, key):
    return linked.get(key, [])


ledger.put_request(scratch, REQUEST_ID, plan, by="agent", linked_jobs=jobs_linked)
# Left out of the next plan with a job linked to it: kept, as dropped
ledger.put_request(
    scratch,
    REQUEST_ID,
    {
        **plan,
        "prompt": None,
        "why": "dropped the cut scene",
        "revision": 1,
        "items": plan["items"][:3],
    },
    by="agent",
    linked_jobs=jobs_linked,
)

# A second request, for the specs that edit a plan and queue from it
# (accept, drop, restore, reorder, the conflict with an agent's write, the
# run dialog), so the first stays as the read-only specs expect: `board` is
# done with two succeeded takes and neither accepted; `shot` failed once;
# `cut` waits on `shot`; `title` has never run and waits on nothing, so it
# is ready; `score` failed once and nothing depends on it. `title` runs
# `e2e-card`, a workflow of one text task - no model, nothing to download,
# done in a second or two - so the run dialog's job can really be queued.
EDIT_ID = "e2e-board"
with open(os.path.join(scratch, "workflows", "e2e-card.json"), "w") as handle:
    json.dump(
        {
            "id": "E2ECard",
            "description": "A title card as one line of text.",
            "variables": {"line": "Title card"},
            "steps": [
                {
                    "name": "card",
                    "task": {
                        "command": "compose_text",
                        "arguments": {"parts": ["variable:line"]},
                    },
                    "result": {"content_type": "text/plain"},
                }
            ],
        },
        handle,
    )
edit_linked = {
    "board": [
        {
            "job": "e2ebrdtake01",
            "status": "succeeded",
            "run_id": "20260908-090000-0badf00d",
        },
        {
            "job": "e2ebrdtake02",
            "status": "succeeded",
            "run_id": "20260908-093000-0badf00d",
        },
    ],
    "shot": [{"job": "e2ebrdshot01", "status": "failed", "run_id": None}],
    "score": [{"job": "e2ebrdscore1", "status": "failed", "run_id": None}],
}
with sqlite3.connect(history_db) as connection:
    for key, jobs in edit_linked.items():
        for index, job in enumerate(jobs):
            done = job["status"] == "succeeded"
            run = os.path.join("e2e-board", job["run_id"]) if done else None
            if done:
                os.makedirs(os.path.join(outputs, run))
                with open(os.path.join(outputs, run, "board.png"), "wb") as handle:
                    handle.write(ONE_PIXEL_PNG)
            started = 1767398410.0 + 1800 * index
            connection.execute(
                "INSERT INTO jobs (id, workflow, status, created_at, started_at,"
                " finished_at, arguments, spec, manifest, error, events, workspace,"
                " run_id, run_dir, request, item)"
                " VALUES (?, 'text-to-image', ?, ?, ?, ?, '{}', ?, ?, ?, '[]',"
                " 'default', ?, ?, ?, ?)",
                (
                    job["job"],
                    job["status"],
                    started - 10,
                    started,
                    started + 60,
                    json.dumps({"workflow_path": template, "workspace": "default"}),
                    json.dumps(
                        [
                            {
                                "step": "image",
                                "files": [f"{run}/board.png"],
                                "role": "final",
                            }
                        ]
                        if done
                        else []
                    ),
                    None if done else "e2e: this attempt failed",
                    job["run_id"],
                    run,
                    EDIT_ID,
                    key,
                ),
            )

ledger.put_request(
    scratch,
    EDIT_ID,
    {
        "title": "E2E storyboard",
        "goal": "A board, a shot and a cut",
        "prompt": "storyboard the opening",
        "why": "opened",
        "revision": 0,
        "items": [
            plan_item("board", "Board sheet"),
            plan_item("shot", "Opening shot", depends_on=["board"]),
            plan_item("cut", "Final cut", depends_on=["shot"]),
            {
                **plan_item("title", "Title card", estimate_minutes=3),
                "workflow": "e2e-card",
                "arguments": {"line": "The opening"},
                # held for review for the next hour, for the hold's specs
                "review_until": (
                    datetime.now(timezone.utc) + timedelta(hours=1)
                ).isoformat(timespec="seconds"),
            },
            plan_item("score", "Score"),
        ],
    },
    by="agent",
    linked_jobs=lambda _request, key: edit_linked.get(key, []),
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
