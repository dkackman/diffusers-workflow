"""#676: JobManager runs a pool of workers, one per card - concurrency, crash
isolation, cancel routing, VRAM-aware dispatch with backfill, the fit check at
submit, the OOM ranking, and what /api/health reports."""

import threading
import time

import pytest

from dw.server.jobs import JobManager
from dw.vram_estimate import required_vram_gb
from tests.test_server import (  # noqa: F401
    ScriptedWorkerManager,
    admitted_for,
    crashing_script,
    server,
    success_script,
    valid_workflow,
)


class Gate:
    """A script that holds its job running until released, then plays `then`."""

    def __init__(self, then=success_script):
        self.started = threading.Event()
        self.release = threading.Event()
        self.then = then

    def __call__(self, command):
        self.started.set()
        self.release.wait(timeout=10)
        yield from self.then(command)

    def wait_started(self):
        assert self.started.wait(timeout=5), "the job never reached its worker"


def card(script, device, capacity_gb, name="Test GPU"):
    manager = ScriptedWorkerManager(script)
    manager.device = device
    manager._capacity_read = True
    manager._capacity_gb = capacity_gb
    manager._device_label = f"{device} {name}"
    return manager


@pytest.fixture
def pool(tmp_path):
    """Builds a JobManager over the given managers, shut down afterwards."""
    built = []

    def make(*managers):
        manager = JobManager(
            str(tmp_path / "outputs"),
            worker_managers=list(managers),
            history_path=str(tmp_path / "jobs.sqlite"),
        )
        built.append((manager, managers))
        return manager

    yield make
    for manager, managers in built:
        # Let any held script go so shutdown does not wait on one
        for worker in managers:
            if isinstance(worker.script, Gate):
                worker.script.release.set()
        manager.shutdown()


def submit(manager, name, vram_need=None):
    return manager.submit(
        admitted=admitted_for(manager, valid_workflow(name)),
        workflow=valid_workflow(name),
        vram_need=vram_need,
    )


def wait_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def wait_for(job, statuses, timeout=5.0):
    assert wait_until(lambda: job.status in statuses, timeout), (
        f"job {job.id} is {job.status}, not {statuses}"
    )


class TestConcurrency:
    def test_two_workers_run_two_jobs_at_once(self, pool):
        gate_a, gate_b = Gate(), Gate()
        a = card(gate_a, "cuda:0", 24)
        b = card(gate_b, "cuda:1", 24)
        manager = pool(a, b)

        first = submit(manager, "first")
        second = submit(manager, "second")
        gate_a.wait_started()
        gate_b.wait_started()

        wait_for(first, {"running"})
        wait_for(second, {"running"})
        assert first.status == second.status == "running"
        assert {first.device, second.device} == {"cuda:0 Test GPU", "cuda:1 Test GPU"}
        assert all(manager.is_worker_busy(slot) for slot in manager.slots)

        gate_a.release.set()
        gate_b.release.set()
        wait_for(first, {"succeeded"})
        wait_for(second, {"succeeded"})


class TestCrashIsolation:
    def test_a_crashed_worker_fails_only_its_own_job(self, pool):
        crash = Gate(then=crashing_script)
        survivor = Gate()
        manager = pool(card(crash, "cuda:0", 24), card(survivor, "cuda:1", 24))

        doomed = submit(manager, "doomed")
        fine = submit(manager, "fine")
        crash.wait_started()
        survivor.wait_started()

        crash.release.set()
        wait_for(doomed, {"failed"})
        assert "Worker crashed" in doomed.error
        # The other card's job is untouched, and still running
        assert fine.status == "running"

        survivor.release.set()
        wait_for(fine, {"succeeded"})


class TestCancel:
    def test_cancel_reaches_the_worker_running_the_job(self, pool):
        gate_a, gate_b = Gate(), Gate()
        a = card(gate_a, "cuda:0", 24)
        b = card(gate_b, "cuda:1", 24)
        cancels = {"a": 0, "b": 0}
        for key, worker, gate in (("a", a, gate_a), ("b", b, gate_b)):
            original = worker.cancel

            def counted(key=key, original=original, gate=gate):
                cancels[key] += 1
                original()
                gate.release.set()

            worker.cancel = counted
        manager = pool(a, b)

        first = submit(manager, "first")
        gate_a.wait_started()
        second = submit(manager, "second")
        gate_b.wait_started()
        wait_for(first, {"running"})
        wait_for(second, {"running"})

        manager.cancel(second.id)
        wait_for(second, {"cancelled"})

        assert cancels == {"a": 0, "b": 1}
        assert first.status == "running"
        gate_a.release.set()
        wait_for(first, {"succeeded"})


class TestBackfill:
    def test_a_job_too_big_for_the_free_card_waits_while_a_smaller_one_takes_it(
        self, pool
    ):
        small_gate, big_gate = Gate(), Gate()
        small = card(small_gate, "cuda:0", 24)
        big = card(big_gate, "cuda:1", 48)
        manager = pool(small, big)

        # Occupy the 48 GB card: a job needing 30 GB cannot go on the 24
        holder = submit(manager, "holder", vram_need=(30, True))
        big_gate.wait_started()
        wait_for(holder, {"running"})
        assert holder.device == "cuda:1 Test GPU"

        wide = submit(manager, "wide", vram_need=(40, True))
        narrow = submit(manager, "narrow", vram_need=(10, True))
        small_gate.wait_started()
        wait_for(narrow, {"running"})
        assert narrow.device == "cuda:0 Test GPU"
        assert wide.status == "queued"

        # The 24 GB card frees but cannot take the 40 GB job
        small_gate.release.set()
        wait_for(narrow, {"succeeded"})
        assert wide.status == "queued"

        big_gate.release.set()
        wait_for(holder, {"succeeded"})
        wait_for(wide, {"succeeded"})
        assert wide.device == "cuda:1 Test GPU"

    def test_a_soft_need_no_card_meets_runs_anyway(self, pool):
        manager = pool(
            card(success_script, "cuda:0", 24), card(success_script, "cuda:1", 24)
        )
        job = submit(manager, "soft", vram_need=(80, False))
        wait_for(job, {"succeeded"})


class TestFitAtSubmit:
    class Admission:
        ok = True

        def __init__(self, vram_need):
            self.vram_need = vram_need

    def test_a_job_too_big_for_every_card_is_refused_naming_the_largest(self, pool):
        manager = pool(
            card(success_script, "cuda:0", 24, name="Small GPU"),
            card(success_script, "cuda:1", 48, name="Large GPU"),
        )
        with pytest.raises(ValueError) as refusal:
            manager.check_fits(self.Admission((60, True)))
        message = str(refusal.value)
        assert "48" in message
        assert "cuda:1 Large GPU" in message
        assert "more than any card here has" in message

        # What the largest card holds, or a soft figure, is let through
        manager.check_fits(self.Admission((48, True)))
        manager.check_fits(self.Admission((60, False)))

    def test_the_route_answers_400(self, server, monkeypatch):  # noqa: F811
        monkeypatch.setattr(
            "dw.server.admission.required_vram_gb", lambda *args: (10_000, True)
        )
        with server(success_script) as client:
            worker = client.app.state.job_manager.worker_manager
            worker._capacity_read = True
            worker._capacity_gb = 24
            worker._device_label = "cuda:0 Test GPU"

            response = client.post("/api/jobs", json={"workflow": valid_workflow()})

            assert response.status_code == 400
            assert "more than any card here has" in str(response.json())
            assert worker.commands == []


class FakeProcess:
    def __init__(self, pid):
        self.pid = pid

    def is_alive(self):
        return True


class TestOomRanking:
    def test_the_later_started_worker_is_ranked_first(self, pool):
        a = card(success_script, "cuda:0", 24)
        b = card(success_script, "cuda:1", 24)
        written = []
        for worker, pid in ((a, 101), (b, 102)):
            worker.pid = lambda pid=pid: pid
            worker.set_oom_score_adj = lambda value, worker=worker: written.append(
                (worker.device, value)
            )
        manager = pool(a, b)
        first, second = manager.slots

        # The only worker alive: nothing is written
        a.worker_active = True
        a.worker_process = FakeProcess(101)
        manager._rank_for_oom(first)
        assert written == []

        # A second starts beside it, with the first still at no adjustment
        b.worker_active = True
        b.worker_process = FakeProcess(102)
        assert a.oom_score_adj is None
        manager._rank_for_oom(second)
        assert written == [("cuda:1", 100)]


class TestHealth:
    def test_one_device_reports_one_worker(self, server):  # noqa: F811
        with server(success_script) as client:
            health = client.get("/api/health").json()

        assert len(health["workers"]) == 1
        assert {"device", "name", "vram_gb", "current_job", "alive"} <= set(
            health["workers"][0]
        )
        assert health["workers"][0]["current_job"] is None
        assert health["worker_alive"] is False

    def test_two_workers_are_each_reported(self, pool):
        gate_a = Gate()
        a = card(gate_a, "cuda:0", 24)
        b = card(success_script, "cuda:1", 48)
        manager = pool(a, b)
        # Hold card 1 out of the dispatcher's reach so the job lands on card 0
        job = submit(manager, "running", vram_need=(10, True))
        gate_a.wait_started()
        wait_for(job, {"running"})

        workers = manager.workers()

        assert [entry["device"] for entry in workers] == ["cuda:0", "cuda:1"]
        assert [entry["vram_gb"] for entry in workers] == [24, 48]
        assert workers[0]["current_job"] == job.id
        assert workers[1]["current_job"] is None


class TestRequiredVram:
    ESTIMATE = {
        "voxel_variables": ["width", "height"],
        "base_gb": 10,
        "bytes_per_voxel": 1024**3,
    }

    def definition(self, **extra):
        return {
            "variables": {"width": 2, "height": 3},
            "steps": [
                {
                    "name": "gen",
                    "pipeline": {
                        "configuration": {"component_type": "{Fake}"},
                        "from_pretrained_arguments": {"model_name": "m"},
                        "arguments": {"prompt": "p"},
                    },
                }
            ],
            **extra,
        }

    def test_a_declared_estimate_is_hard_and_is_the_largest_projection(self):
        definition = self.definition(vram_estimate=self.ESTIMATE)
        assert required_vram_gb(definition, None, "cuda") == (16, True)
        # The caller's arguments change the projection
        assert required_vram_gb(definition, {"width": 4}, "cuda") == (22, True)

    def test_only_cost_entries_give_the_smallest_matching_card_softly(self):
        definition = self.definition(
            cost=[
                {"device": "cuda", "name": "A100", "vram_gb": 80},
                {"device": "cuda", "name": "RTX 4090", "vram_gb": 24},
                {"device": "mps", "name": "M3 Max", "vram_gb": 8},
            ]
        )
        assert required_vram_gb(definition, None, "cuda") == (24, False)
        assert required_vram_gb(definition, None, "mps") == (8, False)
        assert required_vram_gb(definition, None, "cpu") is None

    def test_neither_is_none(self):
        assert required_vram_gb(self.definition(), None, "cuda") is None
        assert required_vram_gb("not a definition") is None
