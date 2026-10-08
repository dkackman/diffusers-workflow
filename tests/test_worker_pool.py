"""#676: JobManager runs a pool of workers, one per card - concurrency, crash
isolation, cancel routing, VRAM-aware dispatch with backfill, the fit check at
submit, the OOM ranking, and what /api/health reports."""

import threading
import time

import pytest

from dw.server.jobs import JobManager, WorkerBusy
from dw.vram_estimate import _entries_for, required_vram_gb
from tests.test_server import (  # noqa: F401
    ScriptedWorkerManager,
    admitted_for,
    crashing_script,
    no_hub,
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

    def test_a_declared_need_is_held_to_the_ceiling_admission_uses(
        self, pool, monkeypatch
    ):
        # A 3090 reads 23.57 GiB: the catalog calls it 24, but admission
        # holds a declared vram_estimate to 23.6 on a card no cost entry
        # describes. The fit check has to agree with it at the boundary.
        monkeypatch.setattr(
            "dw.worker_manager.device_capacity_gb", lambda device=None: 23.57
        )
        worker = ScriptedWorkerManager(success_script)
        worker.device = "cuda:0"
        worker._device_label = "cuda:0 NVIDIA GeForce RTX 3090"
        manager = pool(worker)

        admission_ceiling = _entries_for([], "cuda", 23.57)[0]["vram_gb"]
        assert worker.ceiling_gb() == admission_ceiling == 23.6
        assert worker.capacity_gb() == 24

        with pytest.raises(ValueError) as refusal:
            manager.check_fits(self.Admission((24, True)))
        assert "the largest is cuda:0 NVIDIA GeForce RTX 3090 (23.6 GB usable)" in str(
            refusal.value
        )
        manager.check_fits(self.Admission((23.6, True)))

        # A catalog cost figure of 24 is what a 3090 was measured at
        job = submit(manager, "soft", vram_need=(24, False))
        wait_for(job, {"succeeded"})

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
        # name is the GPU's own name; the job's device is ordinal then name
        assert [entry["name"] for entry in workers] == ["Test GPU", "Test GPU"]
        assert job.device == f"{workers[0]['device']} {workers[0]['name']}"
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


def submit_on(manager, name, device):
    """A job preferring `device`, as a rerun prefers its original's card."""
    return manager.submit(
        admitted=admitted_for(manager, valid_workflow(name)),
        workflow=valid_workflow(name),
        preferred_device=device,
    )


def finish(manager, *jobs):
    """Wait for every job to succeed and its card to be free again."""
    for job in jobs:
        wait_for(job, {"succeeded"})
    assert wait_until(lambda: all(s.current_job_id is None for s in manager.slots))


def rearm(gate):
    gate.started.clear()
    gate.release.clear()


def answering(worker):
    """Makes the scripted worker answer clear_memory (it answers
    memory_status already) and name its own card in every reading."""
    original = worker.send_command

    def send(command):
        if command["type"] == "memory_status":
            worker._results.put(
                {
                    "type": "memory_status",
                    "request_id": command["request_id"],
                    "info": {"card": worker.device},
                }
            )
            worker.commands.append(command)
        elif command["type"] == "clear_memory":
            worker.commands.append(command)
            worker._results.put(
                {
                    "type": "memory_cleared",
                    "request_id": command["request_id"],
                    "info": {"card": worker.device, "cleared": True},
                }
            )
        else:
            original(command)

    worker.send_command = send
    return worker


def types(worker):
    return [command["type"] for command in worker.commands]


class TestIdentityAffinity:
    def test_a_second_run_lands_on_the_card_that_ran_the_workflow_first(self, pool):
        gate_a = Gate()
        a = card(gate_a, "cuda:0", 24)
        b = card(success_script, "cuda:1", 24)
        manager = pool(a, b)

        # Card 0 is busy with Y, so X's first run goes to card 1
        holder = submit(manager, "y")
        gate_a.wait_started()
        wait_for(holder, {"running"})
        first = submit(manager, "x")
        wait_for(first, {"succeeded"})
        assert first.device == "cuda:1 Test GPU"
        gate_a.release.set()
        finish(manager, holder, first)

        # Both free and both warm: without affinity the first card is taken
        second = submit(manager, "x")
        wait_for(second, {"succeeded"})

        assert second.device == "cuda:1 Test GPU"

    def test_another_workflow_is_not_pulled_to_the_warm_card(self, pool):
        manager = pool(
            card(success_script, "cuda:0", 24), card(success_script, "cuda:1", 24)
        )
        warm = submit(manager, "x")
        finish(manager, warm)
        assert warm.device == "cuda:0 Test GPU"

        # Card 0 holds x warm; card 1 has no worker running, so y goes there
        other = submit(manager, "y")
        wait_for(other, {"succeeded"})

        assert other.device == "cuda:1 Test GPU"

    def test_a_cleared_card_holds_nothing_warm(self, pool):
        gate_a = Gate()
        a = answering(card(gate_a, "cuda:0", 24))
        b = answering(card(success_script, "cuda:1", 24))
        manager = pool(a, b)
        holder = submit(manager, "y")
        gate_a.wait_started()
        first = submit(manager, "x")
        wait_for(first, {"succeeded"})
        gate_a.release.set()
        finish(manager, holder, first)
        assert manager.slots[1].warm_identity() is not None

        manager.clear_memory(device="cuda:1")

        assert manager.slots[1].warm_identity() is None
        again = submit(manager, "x")
        wait_for(again, {"succeeded"})
        assert again.device == "cuda:0 Test GPU"


class TestRerunAffinity:
    def original_on_second_card(self, pool, gate_a):
        a = card(gate_a, "cuda:0", 24)
        b = card(success_script, "cuda:1", 24)
        manager = pool(a, b)
        holder = submit(manager, "y")
        gate_a.wait_started()
        original = submit(manager, "x")
        wait_for(original, {"succeeded"})
        assert original.device == "cuda:1 Test GPU"
        gate_a.release.set()
        finish(manager, holder, original)
        # Nothing warm anywhere: only the rerun's own preference can decide
        for slot in manager.slots:
            slot.last_identity = None
        return manager, original

    def test_a_rerun_lands_on_the_original_card_when_it_is_free(self, pool):
        manager, original = self.original_on_second_card(pool, Gate())

        rerun = manager.rerun(
            original.id, admitted=admitted_for(manager, valid_workflow("x"))
        )
        wait_for(rerun, {"succeeded"})

        assert rerun.device == "cuda:1 Test GPU"

    def test_a_rerun_takes_the_other_free_card_when_the_original_is_busy(self, pool):
        gate_a = Gate()
        manager, original = self.original_on_second_card(pool, gate_a)
        # Hold the original's card
        gate_b = Gate()
        manager.slots[1].manager.script = gate_b
        holder = submit_on(manager, "z", "cuda:1")
        gate_b.wait_started()
        wait_for(holder, {"running"})
        assert holder.device == "cuda:1 Test GPU"

        rerun = manager.rerun(
            original.id, admitted=admitted_for(manager, valid_workflow("x"))
        )
        wait_for(rerun, {"succeeded"})

        assert rerun.device == "cuda:0 Test GPU"
        assert holder.status == "running"
        gate_b.release.set()
        finish(manager, holder)

    def test_a_rerun_of_a_job_only_history_remembers_prefers_its_card(self, pool):
        manager, original = self.original_on_second_card(pool, Gate())
        # Dropped from memory, as a long-running server drops old jobs
        del manager.jobs[original.id]

        assert manager._original_device(original.id) == "cuda:1"
        assert manager._original_device("nonexistent") is None


class TestProbeRouting:
    COMMAND = {
        "definition": valid_workflow("x"),
        "file_spec": "x",
        "source": "inline",
        "arguments": {},
        "output_dir": "/tmp",
    }

    def test_the_probe_goes_to_the_worker_the_job_would_be_routed_to(self, pool):
        gate_a = Gate()
        a = card(gate_a, "cuda:0", 24)
        b = card(success_script, "cuda:1", 24)
        manager = pool(a, b)
        holder = submit(manager, "y")
        gate_a.wait_started()
        first = submit(manager, "x")
        wait_for(first, {"succeeded"})
        gate_a.release.set()
        finish(manager, holder, first)
        identity = manager.identity_of("inline", "x", valid_workflow("x"))
        assert manager.route(identity) is manager.slots[1]
        b.cached_steps = ["gen"]
        a.commands.clear()
        b.commands.clear()

        assert manager.probe_cache(self.COMMAND) == ["gen"]

        assert "probe_cache" in types(b)
        assert "probe_cache" not in types(a)

    def test_a_named_slot_overrides_the_route(self, pool):
        a = card(success_script, "cuda:0", 24)
        b = card(success_script, "cuda:1", 24)
        manager = pool(a, b)
        a.ensure_worker()
        b.ensure_worker()
        a.cached_steps = ["from-a"]
        b.cached_steps = ["from-b"]

        assert manager.probe_cache(self.COMMAND, slot=manager.slots[1]) == ["from-b"]
        assert manager.probe_cache(self.COMMAND, slot=manager.slots[0]) == ["from-a"]


class TestMemoryPerCard:
    def two(self, pool, script_a=success_script, script_b=success_script):
        a = answering(card(script_a, "cuda:0", 24))
        b = answering(card(script_b, "cuda:1", 24))
        manager = pool(a, b)
        a.ensure_worker()
        b.ensure_worker()
        return manager, a, b

    def test_without_a_device_there_is_one_entry_per_card(self, pool):
        manager, _, _ = self.two(pool)

        status = manager.memory_status()

        assert [entry["device"] for entry in status["workers"]] == ["cuda:0", "cuda:1"]
        assert [entry["info"] for entry in status["workers"]] == [
            {"card": "cuda:0"},
            {"card": "cuda:1"},
        ]
        assert all(entry["live"] for entry in status["workers"])
        # The top level is the first card's reading, as ever
        assert status["info"] == {"card": "cuda:0"}

    def test_a_device_answers_for_that_card_alone(self, pool):
        manager, a, b = self.two(pool)

        status = manager.memory_status(device="cuda:1")

        assert status["device"] == "cuda:1"
        assert status["info"] == {"card": "cuda:1"}
        assert "workers" not in status
        assert types(a) == []

    def test_an_unknown_device_is_a_value_error(self, pool):
        manager, _, _ = self.two(pool)

        with pytest.raises(ValueError, match="cuda:9"):
            manager.memory_status(device="cuda:9")
        with pytest.raises(ValueError, match="cuda:9"):
            manager.clear_memory(device="cuda:9")

    def test_last_memory_is_kept_per_card(self, pool):
        manager, _, _ = self.two(pool)

        manager.memory_status()

        first, second = manager.slots
        assert first.last_memory == {"card": "cuda:0"}
        assert second.last_memory == {"card": "cuda:1"}
        assert manager.last_memory == {"card": "cuda:0"}

        manager.clear_memory(device="cuda:1")
        assert first.last_memory == {"card": "cuda:0"}
        assert second.last_memory == {"card": "cuda:1", "cleared": True}

    def test_clearing_one_card_leaves_the_other_alone(self, pool):
        manager, a, b = self.two(pool)

        result = manager.clear_memory(device="cuda:1")

        assert result["cleared"] is True
        assert result["device"] == "cuda:1"
        assert types(b) == ["clear_memory"]
        assert types(a) == []

    def test_clearing_every_card_clears_each_idle_one(self, pool):
        manager, a, b = self.two(pool)

        result = manager.clear_memory()

        assert [entry["device"] for entry in result["workers"]] == ["cuda:0", "cuda:1"]
        assert all(entry["cleared"] for entry in result["workers"])
        assert types(a) == types(b) == ["clear_memory"]

    def test_a_named_card_running_a_job_is_refused_though_the_other_is_idle(self, pool):
        gate_b = Gate()
        manager, a, b = self.two(pool, script_b=gate_b)
        job = submit_on(manager, "busy", "cuda:1")
        gate_b.wait_started()
        wait_for(job, {"running"})
        assert job.device == "cuda:1 Test GPU"
        a.commands.clear()

        with pytest.raises(WorkerBusy, match="cuda:1"):
            manager.clear_memory(device="cuda:1")

        assert types(a) == []
        assert "clear_memory" not in types(b)
        gate_b.release.set()
        finish(manager, job)

    def test_clearing_every_card_reports_the_busy_one_and_clears_the_rest(self, pool):
        gate_b = Gate()
        manager, a, b = self.two(pool, script_b=gate_b)
        job = submit_on(manager, "busy", "cuda:1")
        gate_b.wait_started()
        wait_for(job, {"running"})

        result = manager.clear_memory()

        first, second = result["workers"]
        assert first["device"] == "cuda:0" and first["cleared"] is True
        assert second == {
            "device": "cuda:1",
            "cleared": False,
            "reason": "job_running",
            "job": job.id,
        }
        assert types(a) == ["clear_memory"]
        assert "clear_memory" not in types(b)
        gate_b.release.set()
        finish(manager, job)

    def test_every_card_busy_refuses_the_clear(self, pool):
        gate_a, gate_b = Gate(), Gate()
        manager, _, _ = self.two(pool, gate_a, gate_b)
        jobs = [submit_on(manager, "a", "cuda:0"), submit_on(manager, "b", "cuda:1")]
        gate_a.wait_started()
        gate_b.wait_started()

        with pytest.raises(WorkerBusy):
            manager.clear_memory()

        gate_a.release.set()
        gate_b.release.set()
        finish(manager, *jobs)


class TestMemoryRoutes:
    @pytest.fixture
    def pooled(self, pool, tmp_path):
        from fastapi.testclient import TestClient

        from dw.server.app import create_app

        gate_b = Gate()
        a = answering(card(success_script, "cuda:0", 24))
        b = answering(card(gate_b, "cuda:1", 24))
        manager = pool(a, b)
        a.ensure_worker()
        b.ensure_worker()
        workflows = tmp_path / "workflows"
        workflows.mkdir()
        prompts = tmp_path / "prompts"
        prompts.mkdir()
        app = create_app(
            workflow_dir=str(workflows),
            output_dir=str(tmp_path / "outputs"),
            job_manager=manager,
            prompt_dir=str(prompts),
        )
        client = TestClient(app, base_url="http://localhost")
        yield client, manager, gate_b
        gate_b.release.set()

    def test_an_unknown_device_is_a_400(self, pooled):
        client, _, _ = pooled

        assert client.get("/api/memory", params={"device": "cuda:9"}).status_code == 400
        assert (
            client.post("/api/memory/clear", params={"device": "cuda:9"}).status_code
            == 400
        )

    def test_a_device_reads_that_card(self, pooled):
        client, _, _ = pooled

        body = client.get("/api/memory", params={"device": "cuda:1"}).json()
        everything = client.get("/api/memory").json()

        assert body["device"] == "cuda:1"
        assert [w["device"] for w in everything["workers"]] == ["cuda:0", "cuda:1"]

    def test_clearing_a_busy_card_is_a_409_though_the_other_is_idle(self, pooled):
        client, manager, gate_b = pooled
        job = submit_on(manager, "busy", "cuda:1")
        gate_b.wait_started()
        wait_for(job, {"running"})

        busy = client.post("/api/memory/clear", params={"device": "cuda:1"})
        idle = client.post("/api/memory/clear", params={"device": "cuda:0"})

        assert busy.status_code == 409
        assert idle.status_code == 200
        assert idle.json()["device"] == "cuda:0"


@pytest.mark.usefixtures("no_hub")
class TestPricedFor:
    def test_the_estimate_names_the_card_the_job_is_routed_to(self, pool, tmp_path):
        from fastapi.testclient import TestClient

        from dw.server.app import create_app

        a = card(success_script, "cuda:0", 24, name="First GPU")
        b = card(success_script, "cuda:1", 24, name="Second GPU")
        manager = pool(a, b)
        a.ensure_worker()
        b.ensure_worker()
        workflows = tmp_path / "workflows"
        workflows.mkdir()
        prompts = tmp_path / "prompts"
        prompts.mkdir()
        client = TestClient(
            create_app(
                workflow_dir=str(workflows),
                output_dir=str(tmp_path / "outputs"),
                job_manager=manager,
                prompt_dir=str(prompts),
            ),
            base_url="http://localhost",
        )

        def priced(name):
            answer = client.post(
                "/api/validate?sizes=false", json={"workflow": valid_workflow(name)}
            ).json()
            assert answer["valid"], answer
            return answer["plan"]["estimate"]["priced_for"]

        # Both warm with something else: the first card
        manager.slots[0].last_identity = ("inline", "other")
        manager.slots[1].last_identity = ("inline", "x")
        assert priced("unrelated") == "cuda:0 First GPU"
        # The card that last ran this workflow
        assert priced("x") == "cuda:1 Second GPU"
