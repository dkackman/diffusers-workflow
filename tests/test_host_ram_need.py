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
from dw.server.host_ram_need import host_memory_block, memory_key, projected_peak
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
