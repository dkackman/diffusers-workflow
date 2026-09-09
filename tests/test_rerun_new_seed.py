"""Rerunning a seeded workflow.

A workflow that pins its seed is a workflow the step cache can serve whole:
the same definition, the same arguments and the same output root make every
step a hit, so "Run again" finishes in a third of a second and republishes
the earlier run's files with `reused: true`. That is the cache working, but
it leaves the button promising something it did not do. A rerun can instead
draw a new seed, which is what "run it again" means for a generation - and
the workflow says which variable to draw it into.
"""

import json

import pytest
from fastapi.testclient import TestClient

from dw.server.app import create_app
from dw.server.jobs import JobManager

from .test_server import ScriptedWorkerManager, success_script, valid_workflow
from .test_server import wait_for_status


def seeded_workflow(job_id="seeded"):
    """A workflow whose seed points at a declared variable - the shape every
    catalog template uses, `workflows/models/z-image.json` included."""
    workflow = valid_workflow(job_id)
    workflow["seed"] = "variable:seed"
    workflow["variables"] = {**workflow["variables"], "seed": 42}
    return workflow


def literal_seed_workflow(job_id="pinned"):
    workflow = valid_workflow(job_id)
    workflow["seed"] = 7
    return workflow


@pytest.fixture
def server(tmp_path):
    (tmp_path / "workflows").mkdir(exist_ok=True)

    def make(script=success_script):
        manager = JobManager(
            str(tmp_path / "outputs"),
            worker_manager=ScriptedWorkerManager(script),
            history_path=str(tmp_path / "jobs.sqlite"),
        )
        app = create_app(
            workflow_dir=str(tmp_path / "workflows"),
            output_dir=str(tmp_path / "outputs"),
            job_manager=manager,
        )
        return TestClient(app, base_url="http://localhost")

    return make


def submit(client, workflow, arguments=None):
    job = client.post(
        "/api/jobs", json={"workflow": workflow, "arguments": arguments or {}}
    ).json()
    wait_for_status(client, job["id"], ["succeeded"])
    return job["id"]


def test_the_workflow_route_names_the_seed_variable(server):
    """Which variable the seed reads is the client's cue that a new-seed
    rerun is available at all - it is a property of the workflow, so the
    server answers it rather than the page parsing definitions."""
    with server() as client:
        seeded = submit(client, seeded_workflow())
        literal = submit(client, literal_seed_workflow())
        seedless = submit(client, valid_workflow())

        assert client.get(f"/api/jobs/{seeded}/workflow").json()["seed_variable"] == (
            "seed"
        )
        # A literal seed cannot be overridden by an argument, and a seedless
        # workflow already draws a fresh seed every run - neither offers one
        assert client.get(f"/api/jobs/{literal}/workflow").json()["seed_variable"] is (
            None
        )
        assert client.get(f"/api/jobs/{seedless}/workflow").json()["seed_variable"] is (
            None
        )


def test_a_rerun_with_a_new_seed_draws_one_into_that_variable(server):
    with server() as client:
        original = submit(client, seeded_workflow(), {"prompt": "a cat"})

        rerun = client.post(f"/api/jobs/{original}/rerun", json={"new_seed": True})
        assert rerun.status_code == 201
        arguments = rerun.json()["arguments"]

        assert arguments["prompt"] == "a cat"  # everything else rides along
        assert isinstance(arguments["seed"], int)
        assert arguments["seed"] != 42
        # JSON numbers are IEEE doubles in every browser that reads this back
        assert arguments["seed"] < 2**53


def test_two_new_seed_reruns_do_not_draw_the_same_seed(server):
    with server() as client:
        original = submit(client, seeded_workflow())
        seeds = {
            client.post(f"/api/jobs/{original}/rerun", json={"new_seed": True}).json()[
                "arguments"
            ]["seed"]
            for _ in range(5)
        }
        assert len(seeds) == 5


def test_a_plain_rerun_still_repeats_the_original_arguments(server):
    """The default is unchanged: same spec, same arguments, and the step
    cache is free to serve it."""
    with server() as client:
        original = submit(client, seeded_workflow(), {"prompt": "a cat"})
        rerun = client.post(f"/api/jobs/{original}/rerun")
        assert rerun.status_code == 201
        assert rerun.json()["arguments"] == {"prompt": "a cat"}


def test_a_new_seed_rerun_of_a_workflow_that_has_no_seed_variable_is_refused(server):
    """Refused rather than quietly run as an ordinary rerun: the caller
    asked for a different image and would otherwise get the cached one."""
    with server() as client:
        literal = submit(client, literal_seed_workflow())
        response = client.post(f"/api/jobs/{literal}/rerun", json={"new_seed": True})
        assert response.status_code == 400
        assert "seed" in response.json()["detail"].lower()


def test_a_job_launched_from_a_file_finds_its_seed_variable(tmp_path, server):
    """The seed reference lives in the workflow as written, not in the
    realized copy the run wrote - which pins the seed to the integer it
    used, and would look like a literal."""
    workflow_dir = tmp_path / "workflows"
    workflow_dir.mkdir(exist_ok=True)
    (workflow_dir / "Seeded.json").write_text(json.dumps(seeded_workflow()))

    with server() as client:
        job = client.post("/api/jobs", json={"workflow_path": "Seeded"}).json()
        wait_for_status(client, job["id"], ["succeeded"])

        assert client.get(f"/api/jobs/{job['id']}/workflow").json()[
            "seed_variable"
        ] == ("seed")
        rerun = client.post(f"/api/jobs/{job['id']}/rerun", json={"new_seed": True})
        assert rerun.status_code == 201
        assert isinstance(rerun.json()["arguments"]["seed"], int)
