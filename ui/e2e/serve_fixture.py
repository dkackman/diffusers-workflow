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

It refuses to start on a ui/dist older than the UI's sources (bundle_check).

Extra arguments (--port ...) pass through to dw.serve. The process replaces
itself with the server so Playwright's shutdown signal reaches it directly.
"""

import base64
import json
import os
import shutil
import sqlite3
import struct
import sys
import tempfile
import wave
import zlib
from datetime import datetime, timedelta, timezone

from bundle_check import bundle_problem

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
problem = bundle_problem(os.path.join(root, "ui"))
if problem:
    sys.exit(problem)
# playwright.config.ts makes the directory and passes it on, so a spec can
# reach what the fixture seeds (the hub cache, to put back what a case deletes)
scratch = os.environ.get("DW_E2E_SCRATCH") or tempfile.mkdtemp(prefix="dw-e2e-")
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

# One asset and one workflow with a proof picture, each a real image, so the
# specs that look at a tile's or a card's change mark have a picture to cover
with open(os.path.join(scratch, "assets", "e2e-proof-asset.png"), "wb") as handle:
    handle.write(solid_png(96, 96, (60, 140, 90)))
proof_dir = os.path.join(outputs, "card-proof", "20260910-130000-0f1d0f1d")
os.makedirs(proof_dir)
with open(os.path.join(proof_dir, "proof.png"), "wb") as handle:
    handle.write(solid_png(128, 96, (150, 70, 70)))
# Old, so it is not one of the Overview's recent files ahead of lighthouse.png
os.utime(os.path.join(proof_dir, "proof.png"), (1767312000.0, 1767312000.0))
with open(os.path.join(scratch, "workflows", "card-proof.json"), "w") as handle:
    json.dump(
        {
            "id": "CardProof",
            "description": "A workflow that has run once, for its card's picture.",
            "variables": {"line": "Proof"},
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


def seed_jobs(request_id, jobs_by_item, folder, filename, first_start, seconds):
    """History rows for a request's linked jobs, the shape a linked submit
    records: each succeeded job leaves a run under `folder` holding one
    `filename`, each failed one an error (its own `error`, or a stock one).
    The jobs of one item start half an hour apart from `first_start`."""
    with sqlite3.connect(history_db) as connection:
        for key, jobs in jobs_by_item.items():
            for index, job in enumerate(jobs):
                done = job["status"] == "succeeded"
                run = os.path.join(folder, job["run_id"]) if done else None
                if done:
                    os.makedirs(os.path.join(outputs, run))
                    with open(os.path.join(outputs, run, filename), "wb") as handle:
                        handle.write(ONE_PIXEL_PNG)
                started = first_start + 1800 * index
                files = [f"{run}/{filename}"] if done else []
                manifest = [{"step": "image", "files": files, "role": "final"}]
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
                        started + seconds,
                        json.dumps({"workflow_path": template, "workspace": "default"}),
                        json.dumps(manifest if done else []),
                        None if done else job.get("error", "e2e: this attempt failed"),
                        job["run_id"],
                        run,
                        request_id,
                        key,
                    ),
                )


seed_jobs(REQUEST_ID, linked, "e2e-request", "hero.png", 1767312010.0, 90)


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
# `e2e-title`, a workflow of one text task - no model, nothing to download,
# done in a second or two - so the run dialog's job can really be queued.
EDIT_ID = "e2e-board"
# The `title` items run `e2e-title`, the same workflow under a name of its
# own. A task-only workflow that has run is quoted from its history, and
# other specs queue `e2e-card` on this one server - so a quote read off
# `e2e-card` depends on which specs ran first. Nothing queues `e2e-title`
# before the run dialog's own submit, so until then it has no figure.
for card_name, card_id in (("e2e-card", "E2ECard"), ("e2e-title", "E2ETitle")):
    with open(os.path.join(scratch, "workflows", f"{card_name}.json"), "w") as handle:
        json.dump(
            {
                "id": card_id,
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
seed_jobs(EDIT_ID, edit_linked, "e2e-board", "board.png", 1767398410.0, 60)

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
                "workflow": "e2e-title",
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

# A third request, for the review-a-request journey, that wants something of
# every kind from the person: `hero` is done with two succeeded takes and
# neither accepted; `plate` failed once on a long CUDA error, its workflow
# `e2e-card` so it can run again; `comp` failed once and waits on `plate`;
# `title` on `e2e-title` is held for the next hour. Its takes are under a
# folder and name the gallery specs' searches (`e2e`, `still`) do not find.
REVIEW_ID = "e2e-review"
review_linked = {
    "hero": [
        {
            "job": "e2erevhero01",
            "status": "succeeded",
            "run_id": "20260909-090000-5eed5eed",
        },
        {
            "job": "e2erevhero02",
            "status": "succeeded",
            "run_id": "20260909-093000-5eed5eed",
        },
    ],
    "plate": [
        {
            "job": "e2erevplate1",
            "status": "failed",
            "run_id": None,
            "error": "CUDA out of memory. Tried to allocate 2.00 GiB\n"
            "Traceback (most recent call last): …",
        }
    ],
    "comp": [{"job": "e2erevcomp01", "status": "failed", "run_id": None}],
}
# The list puts held requests first; the journey asserts only that nothing
# ahead of this one is anything but held, so no write timing is relied on
seed_jobs(REVIEW_ID, review_linked, "review-takes", "hero-take.png", 1767441610.0, 60)
# Written last, the takes would be the gallery's newest files and push the
# find-and-reuse journey's lighthouse off the overview's recent row: dated
# back to when their jobs ended
for job in review_linked["hero"]:
    take = os.path.join(outputs, "review-takes", job["run_id"], "hero-take.png")
    os.utime(take, (1767441670.0, 1767441670.0))
ledger.put_request(
    scratch,
    REVIEW_ID,
    {
        "title": "E2E review",
        "goal": "A hero still, a plate, a composite and a title card",
        "prompt": "a still I can review before the composite",
        "why": "opened",
        "revision": 0,
        "items": [
            plan_item("hero", "Hero still"),
            {
                **plan_item("plate", "Background plate"),
                "workflow": "e2e-card",
                "arguments": {"line": "A plate"},
            },
            plan_item("comp", "Composite", depends_on=["plate"]),
            {
                **plan_item("title", "Title card"),
                "workflow": "e2e-title",
                "arguments": {"line": "The review"},
                "review_until": (
                    datetime.now(timezone.utc) + timedelta(hours=1)
                ).isoformat(timespec="seconds"),
            },
        ],
    },
    by="agent",
    linked_jobs=lambda _request, key: review_linked.get(key, []),
)

# The first import of a pipeline module, done on the main thread before the
# server reports ready, so no spec waits out or races it (#1603, #1640)
SERVE = (
    "from dw.introspection import load_allowed_class;"
    "load_allowed_class('FluxPipeline');"
    "from dw.serve import main;"
    "main()"
)

os.chdir(root)
os.execv(
    sys.executable,
    [
        sys.executable,
        "-c",
        SERVE,
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
