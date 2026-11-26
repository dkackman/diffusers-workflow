"""What a request shows without one job read per attempt: each attempt's
and each result's proof output, the list's `cover` and its `ready` count -
all derived on the read, none stored."""

import os
import shutil

import pytest

from dw import ledger
from dw.server.outputs import proof_output
from dw.workspace import REQUESTS_FILE_NAME

from .test_ledger import item, plan, write
from . import test_ledger_job_link as link
from .test_ledger_job_link import WS, finish, items_of, submit, view, write_plan

# The job-link tests' server over a scripted worker, as this module's fixtures
root, client, planned = link.root, link.client, link.planned


def listed(client, request_id="cats"):
    r = client.get(f"/api/workspaces/{WS}/requests")
    assert r.status_code == 200, r.text
    return next(row for row in r.json()["requests"] if row["id"] == request_id)


# --- the one file that stands for a job -----------------------------------


class TestProofOutput:
    def test_the_first_visual_file_of_the_final_entries(self):
        manifest = [
            {"step": "ref", "role": "intermediate", "files": ["r/1/ref.png"]},
            {"step": "talk", "role": "final", "files": ["r/1/words.txt"]},
            {"step": "shot", "role": "final", "files": ["r/1/a.mp4", "r/1/b.png"]},
        ]
        assert proof_output(manifest) == {"name": "r/1/a.mp4", "kind": "video"}

    def test_a_final_sound_stands_before_an_intermediate_picture(self):
        manifest = [
            {"step": "art", "role": "intermediate", "files": ["r/1/art.png"]},
            {"step": "song", "role": "final", "files": ["r/1/song.wav"]},
        ]
        assert proof_output(manifest) == {"name": "r/1/song.wav", "kind": "audio"}

    def test_a_manifest_without_roles_uses_every_entry(self):
        manifest = [
            {"step": "say", "files": ["r/1/say.txt"]},
            {"step": "gen", "files": ["r/1/data.json", "r/1/still.png"]},
        ]
        assert proof_output(manifest) == {"name": "r/1/still.png", "kind": "image"}
        assert proof_output(manifest[:1]) == {"name": "r/1/say.txt", "kind": "text"}

    def test_final_entries_with_nothing_to_show_fall_back_on_the_rest(self):
        manifest = [
            {"step": "gen", "role": "intermediate", "files": ["r/1/still.png"]},
            {"step": "plan", "role": "final", "files": ["r/1/plan.json"]},
        ]
        assert proof_output(manifest) == {"name": "r/1/still.png", "kind": "image"}

    @pytest.mark.parametrize(
        "manifest",
        [
            None,
            [],
            "nope",
            [None, {"files": None}, {"files": [5]}],
            [{"files": ["r/1/model.safetensors"]}],
            [{"files": ["/srv/outputs/r/1/still.png"]}],
            [{"files": ["../r/1/still.png"]}],
            [{"files": ["r/../../still.png"]}],
        ],
    )
    def test_nothing_to_show_is_none_and_never_a_path_outside(self, manifest):
        assert proof_output(manifest) is None


# --- the views, over job dicts --------------------------------------------


def job(key, n, output=None, **fields):
    return {
        "job": f"job-{key}-{n}",
        "item": key,
        "status": "succeeded",
        "run_id": f"r{n}",
        "run_exists": True,
        "created_at": n,
        "started_at": n,
        "finished_at": n + 60,
        "output": output,
        **fields,
    }


def png(name):
    return {"name": f"shots/{name}.png", "kind": "image"}


class TestViews:
    def test_an_attempt_and_the_result_carry_the_output(self, tmp_path):
        write(tmp_path, plan())
        jobs = [
            job("a", 1, png("one")),
            job("a", 2, lambda: png("two")),
            job("a", 3, png("gone"), run_exists=False),
            job("a", 4, status="failed"),
        ]
        found = ledger.get_request(str(tmp_path), "cats", lambda rid: jobs)
        a = found["items"][0]
        assert [attempt["output"] for attempt in a["attempts"]] == [
            png("one"),
            png("two"),
            None,  # its run directory is gone, and its files with it
            None,
        ]
        assert a["result"] == {
            "job": "job-a-2",
            "run_id": "r2",
            "output": png("two"),
        }
        assert found["items"][1]["result"] is None

    def test_nothing_derived_is_stored(self, tmp_path):
        write(tmp_path, plan(), jobs_for=lambda rid: [job("a", 1, png("one"))])
        text = (tmp_path / REQUESTS_FILE_NAME).read_text()
        for word in ("output", "cover", "ready", "shots/one.png", "attempts"):
            assert word not in text

    def test_a_plan_read_writes_back_as_it_came(self, tmp_path):
        jobs = [job("a", 1, png("one"))]
        found = write(tmp_path, plan(), jobs_for=lambda rid: jobs)
        again = write(
            tmp_path,
            plan(revision=1, items=found["items"]),
            jobs_for=lambda rid: jobs,
        )
        assert again["items"][0]["result"]["output"] == png("one")

    def test_the_cover_is_the_three_latest_results(self, tmp_path):
        keys = ["a", "b", "c", "d", "e"]
        write(tmp_path, plan(items=[item(k) for k in keys]))
        jobs = [
            job("a", 10, png("a")),
            job("b", 40, png("b")),
            job("c", 30, png("c")),
            job("d", 20, None),
            job("e", 50, status="failed"),
            job("e", 5, png("e-old")),
        ]
        row = ledger.list_requests(str(tmp_path), lambda rid: jobs)[0]
        # d's result shows nothing, and e's latest attempt failed
        assert row["cover"] == [png("b"), png("c"), png("a")]

    def test_the_cover_of_a_request_with_no_result_is_empty(self, tmp_path):
        write(tmp_path, plan())
        assert ledger.list_requests(str(tmp_path))[0]["cover"] == []

    def test_the_list_reads_only_the_proofs_it_shows(self, tmp_path):
        keys = [f"k{n}" for n in range(40)]
        write(tmp_path, plan(items=[item(k) for k in keys]))
        read = []

        def lazy(name):
            def proof():
                read.append(name)
                return None if name.endswith(("7", "8")) else png(name)

            return proof

        jobs = [job(k, n, lazy(k)) for n, k in enumerate(keys)]
        row = ledger.list_requests(str(tmp_path), lambda rid: jobs)[0]
        assert [c["name"] for c in row["cover"]] == [
            "shots/k39.png",
            "shots/k36.png",
            "shots/k35.png",
        ]
        assert read == ["k39", "k38", "k37", "k36", "k35"]
        # and never more than COVER_SCAN, however few of them show anything
        del read[:]
        jobs = [job(k, n, lazy("x7")) for n, k in enumerate(keys)]
        assert ledger.list_requests(str(tmp_path), lambda rid: jobs)[0]["cover"] == []
        assert len(read) == ledger.COVER_SCAN < len(keys)


class TestReady:
    def plan_items(self):
        return [
            item("a", workflow="models/flux-dev"),
            item("b", ["a"], workflow="my-shot"),
            item("c", ["a"]),  # no stored workflow: nothing to queue
            item("d", workflow="my-shot"),
        ]

    def ready(self, found):
        return [i["key"] for i in found["items"] if i["ready"]]

    def test_planned_with_a_workflow_and_every_dependency_done(self, tmp_path):
        write(tmp_path, plan(items=self.plan_items()))
        root = str(tmp_path)  # noqa: F811
        none = ledger.get_request(root, "cats")
        assert self.ready(none) == ["a", "d"]
        assert ledger.list_requests(root)[0]["ready"] == 2
        jobs = [job("a", 1), job("d", 2, status="failed")]
        found = ledger.get_request(root, "cats", lambda rid: jobs)
        # a is done, which frees b; d has an attempt, so it is not planned
        assert self.ready(found) == ["b"]
        assert ledger.list_requests(root, lambda rid: jobs)[0]["ready"] == 1
        running = [job("a", 1, status="running")]
        assert self.ready(ledger.get_request(root, "cats", lambda rid: running)) == [
            "d"
        ]

    def test_a_closed_request_has_nothing_ready(self, tmp_path):
        write(tmp_path, plan(items=self.plan_items(), state="done"), by="user")
        assert self.ready(ledger.get_request(str(tmp_path), "cats")) == []
        assert ledger.list_requests(str(tmp_path))[0]["ready"] == 0

    def test_a_dropped_item_is_never_ready(self, tmp_path):
        write(tmp_path, plan(items=self.plan_items()))
        linked = lambda rid, key: [{"job": "j", "status": "failed"}]  # noqa: E731
        found = write(
            tmp_path,
            plan(revision=1, items=self.plan_items()[:1]),
            linked_jobs=linked,
        )
        assert self.ready(found) == ["a"]


# --- over HTTP, from the job store ----------------------------------------


class TestOverHttp:
    def test_a_finished_job_is_its_items_proof(self, planned):
        done = finish(planned, submit(planned))
        name = done["manifest"][0]["files"][0]
        assert not os.path.isabs(name) and name.endswith("/still.png")
        proof = {"name": name, "kind": "image"}
        a = items_of(planned)["a"]
        assert a["attempts"][0]["output"] == proof
        assert a["result"]["output"] == proof
        assert listed(planned)["cover"] == [proof]
        # the name is the gallery's own, relative to the workspace's outputs
        assert os.path.isfile(os.path.join(planned.root.outputs, name))
        # and it reads the same once the job is history only
        planned.manager.jobs.pop(done["id"])
        assert items_of(planned)["a"]["result"]["output"] == proof
        assert listed(planned)["cover"] == [proof]

    def test_no_absolute_path_reaches_a_read(self, planned):
        finish(planned, submit(planned))
        planned.manager.jobs.clear()
        for body in (
            planned.get(f"/api/workspaces/{WS}/requests").text,
            planned.get(f"/api/workspaces/{WS}/requests/cats").text,
        ):
            assert str(planned.root.root) not in body

    def test_a_queued_or_failed_attempt_has_no_output(self, planned):
        planned.script.fail_next = True
        finish(planned, submit(planned))
        a = items_of(planned)["a"]
        assert a["attempts"][0]["output"] is None and a["result"] is None
        assert listed(planned)["cover"] == []

    def test_a_deleted_run_takes_its_proof_with_it(self, planned):
        done = finish(planned, submit(planned))
        shutil.rmtree(os.path.join(planned.root.outputs, done["run_dir"]))
        a = items_of(planned)["a"]
        assert a["attempts"][0]["output"] is None and a["result"] is None
        assert listed(planned)["cover"] == []

    def test_polling_the_list_reads_each_manifest_once(self, planned, monkeypatch):
        for key in ("a", "b"):
            finish(planned, submit(planned, item_key=key))
        planned.manager.jobs.clear()
        history = planned.manager.history
        reads = []
        real = history.manifest_of

        def counted(job_id):
            reads.append(job_id)
            return real(job_id)

        monkeypatch.setattr(history, "manifest_of", counted)
        for _ in range(5):
            assert len(listed(planned)["cover"]) == 2
            view(planned)
        assert len(reads) == 2

    def test_ready_over_http(self, client):
        write_plan(
            client,
            items=[item("a", workflow="my-shot"), item("b", ["a"], workflow="my-shot")],
        )
        assert listed(client)["ready"] == 1
        assert [i["ready"] for i in view(client)["items"]] == [True, False]
        finish(client, submit(client))
        assert listed(client)["ready"] == 1
        assert [i["ready"] for i in view(client)["items"]] == [False, True]
