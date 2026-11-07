"""The host-RAM need a run is projected to hold (#869, #909).

`memory_key` names what decides a run's footprint (pipelines, quantization,
offload, arguments resolved); `JobHistory` stores it per run; `projected_peak`
prefers the observed peak of this machine's successful runs with that key over
the fit from the workflow's own history; validate reports it as
`plan.host_memory`.
"""

import copy
import json
import os
import sqlite3

import pytest
from fastapi.testclient import TestClient

from dw.host_memory_projection import host_memory_warnings, projected_mb
from dw.server.app import create_app
from dw.host_memory import HostClaims
from dw.server.host_ram_need import (
    declared_mb,
    host_memory_block,
    host_too_small,
    memory_key,
    projected_peak,
)
from dw.server.job_history import JobHistory
from dw.server.job_record import SUCCEEDED, Job
from dw.server.jobs import JobManager
from dw.workspace import Workspace

from .test_server import ScriptedWorkerManager, success_script, valid_workflow

TEMPLATE = "workflows/templates/minimax/reference-to-video.json"


def template():
    with open(TEMPLATE) as file:
        return json.load(file)


def pipeline_step(definition):
    return next(step for step in definition["steps"] if "pipeline" in step)


def recorded_job(job_id, definition, peak, status=SUCCEEDED, arguments=None):
    job = Job(
        {
            "workflow_name": "w",
            "catalog_name": "w",
            "arguments": arguments or {},
            "definition": definition,
        }
    )
    job.id = job_id
    job.status = status
    job.started_at = 10.0
    job.finished_at = 70.0
    job.host_memory_job_peak_rss_mb = peak
    return job


def fit_rows(*peaks):
    return [{"host_memory_job_peak_rss_mb": peak, "arguments": "{}"} for peak in peaks]


class TestMemoryKey:
    def test_the_template_has_a_key(self):
        assert memory_key(template()) is not None

    def test_a_renamed_copy_has_the_same_key(self):
        definition = template()
        renamed = copy.deepcopy(definition)
        renamed["id"] = "SomethingElse"
        renamed["name"] = "something-else"

        assert memory_key(renamed) == memory_key(definition)

    def test_an_inline_copy_without_its_metadata_has_the_same_key(self):
        definition = template()
        inline = copy.deepcopy(definition)
        for field in ("id", "name", "description", "summary", "cost", "cost_drivers"):
            inline.pop(field, None)

        assert memory_key(inline) == memory_key(definition)

    def test_a_different_weights_dtype_is_a_different_key(self):
        definition = template()
        assert "weights_dtype" in definition["variables"]

        int4 = memory_key(definition, {"weights_dtype": "{int4}"})
        int8 = memory_key(definition, {"weights_dtype": "{int8}"})

        assert int4 is not None and int8 is not None
        assert int4 != int8

    def test_a_different_offload_is_a_different_key(self):
        definition = valid_workflow("offload")
        offloaded = copy.deepcopy(definition)
        pipeline_step(offloaded)["pipeline"]["configuration"]["offload"] = "model"

        assert memory_key(offloaded) != memory_key(definition)

    def test_a_different_group_offload_setting_changes_the_template_key(self):
        definition = template()
        changed = copy.deepcopy(definition)
        text = json.dumps(changed).replace("block_level", "leaf_level")
        assert text != json.dumps(changed)

        assert memory_key(json.loads(text)) != memory_key(definition)

    def test_a_task_only_definition_has_no_key(self):
        definition = {
            "id": "tasks",
            "steps": [
                {
                    "name": "n",
                    "task": {"command": "normalize_audio", "arguments": {}},
                }
            ],
        }

        assert memory_key(definition) is None

    def test_an_undeclared_variable_has_no_key(self):
        definition = template()
        definition["variables"].pop("weights_dtype")

        assert memory_key(definition) is None


class TestProjectedPeak:
    def test_an_observed_peak_beats_the_fit(self, tmp_path):
        definition = template()
        history = JobHistory(str(tmp_path / "jobs.sqlite"))
        history.record(recorded_job("a", definition, 50000))
        history.record(recorded_job("b", definition, 60000))
        rows = fit_rows(1000, 2000)
        entries = {"references": 2}
        assert projected_mb(definition, entries, rows) is not None

        need = projected_peak(definition, {}, history, rows=rows, list_entries=entries)

        assert need == (60000, "observed", 2)

    def test_a_renamed_run_shares_the_templates_observed_history(self, tmp_path):
        definition = template()
        renamed = copy.deepcopy(definition)
        renamed["id"] = "Renamed"
        history = JobHistory(str(tmp_path / "jobs.sqlite"))
        history.record(recorded_job("a", renamed, 42000))

        assert projected_peak(definition, {}, history) == (42000, "observed", 1)

    def test_the_fit_answers_when_nothing_was_observed(self, tmp_path):
        history = JobHistory(str(tmp_path / "jobs.sqlite"))
        rows = fit_rows(30000, 40000) + [
            {"host_memory_job_peak_rss_mb": None, "arguments": "{}"}
        ]

        need = projected_peak(
            template(), {}, history, rows=rows, list_entries={"references": 3}
        )

        assert need == (40000, "fit", 2)

    def test_no_figure_without_history_or_rows(self, tmp_path):
        history = JobHistory(str(tmp_path / "jobs.sqlite"))

        assert projected_peak(template(), {}, history) is None
        assert projected_peak(template(), {}, None) is None

    def test_a_failed_run_is_not_observed(self, tmp_path):
        definition = template()
        history = JobHistory(str(tmp_path / "jobs.sqlite"))
        history.record(recorded_job("a", definition, 99000, status="failed"))

        assert history.peaks_for_key(memory_key(definition)) == []
        assert projected_peak(definition, {}, history) is None

    def test_a_run_without_a_peak_is_not_observed(self, tmp_path):
        definition = template()
        history = JobHistory(str(tmp_path / "jobs.sqlite"))
        history.record(recorded_job("a", definition, None))

        assert history.peaks_for_key(memory_key(definition)) == []

    def test_another_key_is_not_observed(self, tmp_path):
        definition = template()
        history = JobHistory(str(tmp_path / "jobs.sqlite"))
        history.record(recorded_job("a", definition, 50000))

        other = projected_peak(definition, {"weights_dtype": "{int8}"}, history)

        assert other is None


def shots_definition(released=False):
    """A for_each over a shot list, holding its pipeline resident across the
    list (the accumulating shape) unless `released`."""
    step = {
        "name": "shot",
        "for_each": "variable:shots",
        "pipeline": {
            "configuration": {"component_type": "ModularPipeline"},
            "from_pretrained_arguments": {"model_name": "acme/video"},
        },
        "run": {"prompt": "item:prompt"},
    }
    if released:
        step["release_pipeline"] = True
    return {
        "id": "shots",
        "variables": {"shots": [{"prompt": "a"}]},
        "steps": [step],
    }


def shot_rows(*runs):
    """Fit rows of (shot count, peak MB)."""
    return [
        {
            "host_memory_job_peak_rss_mb": peak,
            "arguments": json.dumps({"shots": [{"prompt": "p"}] * count}),
        }
        for count, peak in runs
    ]


class TestALongerListThanObserved:
    """#910: the memory key leaves list length out, so for an accumulating
    shape the observed max (from shorter lists) under-projects a longer one;
    the fit's growth is taken when it is the larger, so dispatch, the plan's
    host_memory and validate's warning agree."""

    def _history(self, tmp_path, definition):
        history = JobHistory(str(tmp_path / "jobs.sqlite"))
        for job_id, count, peak in (("a", 2, 20000), ("b", 4, 30000)):
            history.record(
                recorded_job(
                    job_id,
                    definition,
                    peak,
                    arguments={"shots": [{"prompt": "p"}] * count},
                )
            )
        return history

    def test_an_accumulating_list_takes_the_larger_fit(self, tmp_path):
        definition = shots_definition()
        history = self._history(tmp_path, definition)
        rows = shot_rows((2, 20000), (4, 30000))

        need = projected_peak(
            definition,
            {"shots": [{"prompt": "p"}] * 8},
            history,
            rows=rows,
            list_entries={"shots": 8},
        )

        assert need == (50000, "fit", 2)
        warnings = host_memory_warnings(definition, {"shots": 8}, rows, 40000)
        assert warnings and "~50000 MB" in warnings[0]

    def test_a_list_inside_the_observed_range_keeps_the_observed_max(self, tmp_path):
        definition = shots_definition()
        history = self._history(tmp_path, definition)

        need = projected_peak(
            definition,
            {"shots": [{"prompt": "p"}] * 3},
            history,
            rows=shot_rows((2, 20000), (4, 30000)),
            list_entries={"shots": 3},
        )

        assert need == (30000, "observed", 2)

    def test_a_released_list_keeps_the_observed_max(self, tmp_path):
        definition = shots_definition(released=True)
        history = self._history(tmp_path, definition)

        need = projected_peak(
            definition,
            {"shots": [{"prompt": "p"}] * 8},
            history,
            rows=shot_rows((2, 20000), (4, 30000)),
            list_entries={"shots": 8},
        )

        assert need == (30000, "observed", 2)


def old_shape_database(path):
    """A jobs.sqlite from before the memory_key column, holding one successful
    run with a peak."""
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE jobs (
                id TEXT PRIMARY KEY, workflow TEXT, status TEXT,
                created_at REAL, started_at REAL, finished_at REAL,
                arguments TEXT, spec TEXT, manifest TEXT, warnings TEXT,
                error TEXT, events TEXT, host_memory_job_peak_rss_mb REAL
            )""")
        connection.execute(
            "INSERT INTO jobs (id, workflow, status, arguments, spec, manifest,"
            " events, host_memory_job_peak_rss_mb) VALUES (?,?,?,?,?,?,?,?)",
            ("old", "w", SUCCEEDED, "{}", "{}", "[]", "[]", 77000.0),
        )


class TestMigration:
    def test_a_row_from_before_the_column_counts_as_no_data(self, tmp_path):
        path = str(tmp_path / "jobs.sqlite")
        old_shape_database(path)

        history = JobHistory(path)

        with sqlite3.connect(path) as connection:
            columns = [row[1] for row in connection.execute("PRAGMA table_info(jobs)")]
            (key,) = connection.execute("SELECT memory_key FROM jobs").fetchone()
        assert "memory_key" in columns
        assert key is None
        assert history.peaks_for_key(memory_key(template())) == []
        assert projected_peak(template(), {}, history) is None

    def test_opening_a_migrated_database_again_is_harmless(self, tmp_path):
        path = str(tmp_path / "jobs.sqlite")
        old_shape_database(path)
        JobHistory(path)

        JobHistory(path)

        with sqlite3.connect(path) as connection:
            columns = [row[1] for row in connection.execute("PRAGMA table_info(jobs)")]
        assert columns.count("memory_key") == 1


class TestHostMemoryBlock:
    def test_no_need_is_no_block(self):
        assert host_memory_block(None) is None

    def test_the_figure_is_rounded(self):
        assert host_memory_block((61234.6, "observed", 2)) == {
            "projected_mb": 61235,
            "basis": "observed",
            "runs": 2,
        }


@pytest.fixture
def server(tmp_path):
    root = Workspace(tmp_path / "studio", "flag").ensure()
    with open(os.path.join(root.workflows, "Heavy.json"), "w") as file:
        json.dump(valid_workflow("heavy"), file)
    manager = JobManager(
        root.outputs,
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=root.workflows,
    )
    app = create_app(
        workflow_dir=root.workflows,
        output_dir=root.outputs,
        job_manager=manager,
        prompt_dir=root.prompts,
        asset_dir=root.assets,
        workspace=root.root,
    )
    return manager, TestClient(app, base_url="http://localhost")


def validate(client):
    response = client.post("/api/validate", json={"workflow_path": "Heavy"})
    assert response.status_code == 200
    return response.json()


class TestValidateRoute:
    def test_no_history_has_no_host_memory_in_the_plan(self, server):
        manager, client = server
        with client:
            result = validate(client)

        assert result["valid"] is True
        assert "host_memory" not in result["plan"]

    def test_an_observed_run_is_the_plans_host_memory(self, server):
        manager, client = server
        manager.history.record(recorded_job("a", valid_workflow("heavy"), 51234.4))
        with client:
            result = validate(client)

        assert result["plan"]["host_memory"] == {
            "projected_mb": 51234,
            "basis": "observed",
            "runs": 1,
        }


class TestCommittedPeak:
    """A worker's RSS counts the checkpoint pages it maps (RssFile) - page
    cache the kernel can drop and a second worker mapping the same weights
    shares, which MemAvailable already counts as free. An H3 chain peaked at
    ~122 GB RSS with ~43 GB of it committed (RssAnon), so the gate held every
    queued H3 job for a figure three times what it would take."""

    def test_the_committed_peak_is_the_need(self, tmp_path):
        definition = template()
        history = JobHistory(str(tmp_path / "jobs.sqlite"))
        job = recorded_job("a", definition, 122000)
        job.host_memory_job_peak_committed_mb = 43000
        history.record(job)

        assert projected_peak(definition, {}, history) == (43000, "observed", 1)

    def test_runs_that_measured_it_supersede_rss_only_rows(self, tmp_path):
        definition = template()
        history = JobHistory(str(tmp_path / "jobs.sqlite"))
        history.record(recorded_job("old", definition, 122000))
        job = recorded_job("new", definition, 120000)
        job.host_memory_job_peak_committed_mb = 44000
        history.record(job)

        assert history.peaks_for_key(memory_key(definition)) == [44000]

    def test_rss_only_history_still_answers(self, tmp_path):
        definition = template()
        history = JobHistory(str(tmp_path / "jobs.sqlite"))
        history.record(recorded_job("old", definition, 122000))

        assert history.peaks_for_key(memory_key(definition)) == [122000]


class TestRecordRunMemory:
    def test_the_committed_peak_is_the_largest_committed_reading(self):
        from dw.server.job_results import record_run_memory

        class Slot:
            last_memory = None

        job = recorded_job("a", template(), None)
        for committed in (5000, 43000, 30000):
            record_run_memory(
                job,
                # RssAnon alone is not the committed figure: shmem and the
                # driver's pinned pages are in host_memory_committed_mb
                {"host_memory_committed_mb": committed, "host_memory_rss_anon_mb": 1},
                Slot(),
                lambda info, slot: None,
            )

        assert job.host_memory_job_peak_committed_mb == 43000


class TestWorkerCommittedRss:
    def test_a_workers_committed_rss_leaves_out_its_mapped_files(self, tmp_path):
        from dw.host_memory import committed_rss_mb

        status = tmp_path / "status"
        status.write_text(
            "VmRSS:\t108026044 kB\nRssAnon:\t42561384 kB\nRssFile:\t64928348 kB\n"
        )

        assert committed_rss_mb(None, status_path=str(status)) == pytest.approx(
            42561384 / 1024
        )


class TestTheFitOnCommittedHistory:
    """The fit is a list run's projection from the workflow's own rows, and
    those rows carried only RSS: an accumulating list's fit on ~122 GB-RSS
    runs beat the ~43 GB committed observed max and became the need again,
    a comparison of two bases. Once a row has the committed peak, the fit
    reads committed rows."""

    def _history(self, tmp_path, definition):
        history = JobHistory(str(tmp_path / "jobs.sqlite"))
        for job_id, count, rss, committed in (
            ("a", 2, 100000, 20000),
            ("b", 4, 120000, 30000),
        ):
            job = recorded_job(
                job_id, definition, rss, arguments={"shots": [{"prompt": "p"}] * count}
            )
            job.host_memory_job_peak_committed_mb = committed
            history.record(job)
        return history

    def test_a_list_runs_fit_reads_committed_rows(self, tmp_path):
        definition = shots_definition()
        history = self._history(tmp_path, definition)
        rows = [
            {**row, "host_memory_job_peak_committed_mb": committed}
            for row, committed in zip(
                shot_rows((2, 100000), (4, 120000)), (20000, 30000)
            )
        ]

        need = projected_peak(
            definition,
            {"shots": [{"prompt": "p"}] * 8},
            history,
            rows=rows,
            list_entries={"shots": 8},
        )

        # committed: 20000 at 2, 30000 at 4 -> 50000 at 8 (RSS would be 160000)
        assert need == (50000, "fit", 2)

    def test_the_history_rows_carry_the_committed_peak(self, tmp_path):
        definition = shots_definition()
        history = self._history(tmp_path, definition)

        rows = next(iter(history.finished_runs().values()))

        assert sorted(r["host_memory_job_peak_committed_mb"] for r in rows) == [
            20000,
            30000,
        ]

    def test_rss_only_rows_still_fit_on_rss(self):
        definition = shots_definition()
        fit = projected_mb(
            definition, {"shots": 8}, shot_rows((2, 100000), (4, 120000))
        )
        assert fit == {"projected_mb": 160000, "runs": 2}


class TestHostRamMargin:
    """The stored peak is the largest of readings taken at phase boundaries,
    a lower bound on the true peak, so the gate keeps a margin above it."""

    def test_a_need_that_only_just_fits_waits_for_its_margin(self):
        from dw.server.pool import host_ram_wait

        # 40 GB need, 42 GB free: inside the 10% margin
        assert host_ram_wait(40 * 1024, 42 * 1024, [(None, None)]) is not None
        assert host_ram_wait(40 * 1024, 45 * 1024, [(None, None)]) is None

    def test_a_small_need_keeps_the_minimum_margin(self):
        from dw.server.pool import HOST_RAM_MARGIN_MIN_MB, host_ram_wait

        need = 4 * 1024
        free = need + HOST_RAM_MARGIN_MIN_MB - 1
        assert host_ram_wait(need, free, [(None, None)]) is not None
        assert host_ram_wait(need, free + 2, [(None, None)]) is None

    def test_a_running_jobs_peak_is_charged_with_its_margin(self):
        from dw.server.pool import host_ram_wait

        # The running job sits at its recorded 30 GB peak, which is a lower
        # bound: 3 GB more of growth is charged, so 46 GB available is 43 GB
        # free against the 44 GB the 40 GB need takes with its own margin
        need = 40 * 1024
        assert host_ram_wait(need, 46 * 1024, [(30 * 1024, 30 * 1024)]) is not None
        assert host_ram_wait(need, 48 * 1024, [(30 * 1024, 30 * 1024)]) is None

    def test_the_margin(self):
        from dw.server.pool import with_margin

        assert with_margin(100 * 1024) == 110 * 1024
        assert with_margin(1024) == 1024 + 2048


class TestDispatchWorkerReading:
    def test_a_slots_worker_is_read_as_committed_memory(self, monkeypatch):
        from dw.server import dispatch

        asked = []
        monkeypatch.setattr(
            dispatch,
            "committed_rss_mb",
            lambda pid, device=True: asked.append((pid, device)) or 43000.0,
        )

        class Manager:
            def pid(self):
                return 4242

        class Slot:
            manager = Manager()

        assert dispatch._worker_rss_mb(Slot()) == 43000.0
        assert asked == [(4242, False)]

    def test_a_slot_without_a_worker_is_no_reading(self, monkeypatch):
        from dw.server import dispatch

        monkeypatch.setattr(
            dispatch, "committed_rss_mb", lambda pid, device=True: 1 / 0
        )

        class Manager:
            def pid(self):
                return None

        class Slot:
            manager = Manager()

        class NoPid:
            manager = object()

        assert dispatch._worker_rss_mb(Slot()) is None
        assert dispatch._worker_rss_mb(NoPid()) is None


class TestDispatchNeverReadsSmaps:
    def _proc(self, tmp_path, monkeypatch):
        from dw import host_memory

        status = tmp_path / "status"
        status.write_text("RssAnon:\t2048 kB\nRssFile:\t999 kB\nRssShmem:\t1024 kB\n")
        monkeypatch.setattr(host_memory, "_has_proc", lambda: True)
        monkeypatch.setattr(
            host_memory,
            "_proc_path",
            lambda pid, name: str(status) if name == "status" else 1 / 0,
        )
        monkeypatch.setattr(host_memory, "device_mapped_mb", lambda *a, **k: 1 / 0)

    def _slot(self, last_memory):
        class Manager:
            def pid(self):
                return 4242

        class Slot:
            manager = Manager()

        slot = Slot()
        slot.last_memory = last_memory
        return slot

    def test_adds_the_last_reported_device_figure(self, tmp_path, monkeypatch):
        from dw.server import dispatch

        self._proc(tmp_path, monkeypatch)
        report = {
            "host_memory_committed_mb": 10.0,
            "host_memory_rss_anon_mb": 4.0,
            "host_memory_rss_shmem_mb": 1.0,
        }
        assert dispatch._worker_rss_mb(self._slot(report)) == 3.0 + 5.0

    def test_no_report_falls_back_to_status_only(self, tmp_path, monkeypatch):
        from dw.server import dispatch

        self._proc(tmp_path, monkeypatch)
        assert dispatch._worker_rss_mb(self._slot(None)) == 3.0
        assert dispatch._worker_rss_mb(self._slot({})) == 3.0


H3_TEMPLATE = "workflows/templates/minimax/chained-segments.json"


def h3():
    with open(H3_TEMPLATE) as file:
        return json.load(file)


class TestDeclaredHostRam:
    """#1227: the tier rows declare their host RAM; int8 is the heavy one."""

    def test_each_tier_declares_its_figure(self):
        definition = h3()
        assert declared_mb(definition, {"tier": "draft"}) == 51 * 1024
        assert declared_mb(definition, {"tier": "standard"}) == 51 * 1024
        assert declared_mb(definition, {"tier": "best"}) == 69 * 1024
        assert declared_mb(definition, None) == 51 * 1024

    def test_an_explicit_argument_beats_the_bundle(self):
        assert declared_mb(h3(), {"tier": "best", "host_ram_gb": 80}) == 80 * 1024

    def test_a_definition_without_a_declaration_has_none(self):
        assert declared_mb(template(), {}) is None
        assert declared_mb({"host_ram_gb": 0}, {}) is None
        assert declared_mb({"host_ram_gb": "variable:nope"}, {}) is None

    def test_the_declaration_is_a_floor_under_an_observed_peak(self):
        definition = h3()
        assert projected_peak(definition, {"tier": "best"}, None) == (
            69 * 1024,
            "declared",
            0,
        )

    def test_a_larger_observed_peak_wins(self, tmp_path):
        history = JobHistory(str(tmp_path / "jobs.sqlite"))
        definition = h3()
        arguments = {"tier": "best"}
        history.record(recorded_job("a", definition, 80000.0, arguments=arguments))
        mb, basis, _ = projected_peak(definition, arguments, history)
        assert (mb, basis) == (80000.0, "observed")

    def test_a_host_too_small_for_the_declaration_is_refused(self):
        best = host_too_small(h3(), {"tier": "best"}, 64 * 1024)
        assert best is not None and "64 GB" in best and "69 GB" in best
        # int4 on the same 64 GB host: 51 GB plus margin fits
        assert host_too_small(h3(), {"tier": "standard"}, 64 * 1024) is None
        # a 128 GB host holds either, one at a time
        assert host_too_small(h3(), {"tier": "best"}, 125 * 1024) is None
        # no reading of the host, no refusal
        assert host_too_small(h3(), {"tier": "best"}, None) is None


class TestSmallHostIsRefusedUpFront:
    def test_validate_refuses_the_best_tier_on_a_64_gb_host(self, server, monkeypatch):
        monkeypatch.setattr(
            "dw.server.admission.host_memory_stats",
            lambda: {"total_mb": 64 * 1024},
        )
        manager, client = server
        with client:
            response = client.post(
                "/api/validate", json={"workflow_path": "Heavy", "arguments": {}}
            )
            assert response.status_code == 200
            assert response.json()["valid"] is True

    def test_admission_adds_the_error(self, monkeypatch):
        from dw.server import admission

        monkeypatch.setattr(
            admission, "host_memory_stats", lambda: {"total_mb": 64 * 1024}
        )

        class Candidate:
            workflow_definition = h3()

        class Fake:
            workflow = Candidate()
            arguments = {"tier": "best"}
            errors = []

        fake = Fake()
        admission._refuse_for_small_host(fake)
        assert fake.errors[0]["path"] == "host_ram_gb"
        assert "64 GB" in fake.errors[0]["message"]


class TestHostClaims:
    def test_another_live_servers_claim_is_seen_and_ones_own_is_not(self, tmp_path):
        mine = HostClaims(directory=str(tmp_path), pid=os.getpid())
        theirs = HostClaims(directory=str(tmp_path), pid=os.getppid())
        mine.publish([(999999, 1000.0)])
        theirs.publish([(999998, 2000.0)])
        assert mine.foreign() == [(2000.0, None)]
        assert theirs.foreign() == [(1000.0, None)]

    def test_a_dead_servers_claim_is_dropped(self, tmp_path):
        dead = HostClaims(directory=str(tmp_path), pid=2**22 + 12345)
        dead.publish([(999999, 1000.0)])
        reader = HostClaims(directory=str(tmp_path), pid=os.getpid())
        assert reader.foreign() == []
        assert not os.path.exists(os.path.join(str(tmp_path), f"{dead.pid}.json"))

    def test_withdrawing_removes_the_claim(self, tmp_path):
        theirs = HostClaims(directory=str(tmp_path), pid=os.getppid())
        theirs.publish([(999999, 1000.0)])
        theirs.withdraw()
        assert HostClaims(directory=str(tmp_path)).foreign() == []
