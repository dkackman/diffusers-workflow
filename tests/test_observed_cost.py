"""What this box's own job history says a workflow costs.

`cost` is a maintainer's claim, defined in the schema as never derived. This
is the sibling that is nothing but derivation: finished runs of one workflow,
bucketed so a 141-frame run never informs a 124-frame figure, split cold from
warm, with step-cache hits thrown out (#93, from #91).

Each rule here is a way the naive median would lie, so each has a test that
fails if the rule is dropped.
"""

import glob
import json
import os

import pytest

from dw.server.jobs import JobHistory
from dw.server.observed_cost import (
    ObservedCosts,
    declared_drivers,
    observed_for,
)
from tests.test_examples import REPO_ROOT

MINUTE = 60.0


def run(
    duration,
    arguments=None,
    had_load=True,
    manifest=None,
    at_cap=False,
    finished_at=1_700_000_000.0,
):
    return {
        "started_at": finished_at - duration,
        "finished_at": finished_at,
        "duration": duration,
        "arguments": json.dumps(arguments or {}),
        "manifest": json.dumps(manifest if manifest is not None else [{"step": "a"}]),
        "had_load": had_load,
        "events_at_cap": at_cap,
    }


def workflow(drivers=None, **variables):
    definition = {"id": "w", "variables": variables or {"num_frames": 124}}
    if drivers is not None:
        definition["cost_drivers"] = drivers
    return definition


class TestComparability:
    def test_runs_at_the_same_driver_values_are_one_bucket(self):
        """Different prompts and seeds, same frame count: one figure."""
        definition = workflow(["num_frames"], num_frames=124, prompt="a")

        observed = observed_for(
            definition,
            [
                run(8 * MINUTE, {"prompt": "one", "seed": 1}),
                run(9 * MINUTE, {"prompt": "two", "seed": 2}),
            ],
        )

        assert observed["runs"] == 2
        assert observed["cold_runs"] == 2
        assert observed["cold_minutes"] == 8.5
        assert observed["comparable"] == "drivers"
        assert observed["drivers"] == {"num_frames": 124}

    def test_a_run_at_a_different_driver_value_is_a_different_bucket(self):
        """The reported bucket is the one a caller gets by default, so a
        resized run never lands in it - the figure stays comparable to the
        curated `cost`, which was measured at the defaults."""
        definition = workflow(["num_frames"], num_frames=124)

        observed = observed_for(
            definition,
            [
                run(8 * MINUTE, {}),
                run(8 * MINUTE, {}),
                run(30 * MINUTE, {"num_frames": 345}),
            ],
        )

        assert observed["runs"] == 2
        assert observed["cold_minutes"] == 8.0

    def test_a_supplied_value_equal_to_the_default_is_the_same_bucket(self):
        definition = workflow(["num_frames"], num_frames=124)

        observed = observed_for(
            definition, [run(8 * MINUTE, {}), run(8 * MINUTE, {"num_frames": 124})]
        )

        assert observed["runs"] == 2

    def test_a_numeric_string_argument_is_the_same_bucket(self):
        """The CLI and REPL hand every argument over as a string."""
        definition = workflow(["num_frames"], num_frames=124)

        observed = observed_for(
            definition, [run(8 * MINUTE, {}), run(8 * MINUTE, {"num_frames": "124"})]
        )

        assert observed["runs"] == 2

    def test_a_list_driver_buckets_on_its_length(self):
        """A `shots` list costs what its number of entries costs. Bucketing
        on contents would give every run its own bucket, which reports
        nothing at all."""
        definition = workflow(["shots"], shots=[{"name": "a"}, {"name": "b"}])

        observed = observed_for(
            definition,
            [
                run(10 * MINUTE, {"shots": [{"name": "x"}, {"name": "y"}]}),
                run(12 * MINUTE, {"shots": [{"name": "p"}, {"name": "q"}]}),
                run(30 * MINUTE, {"shots": [{"name": "a"}] * 5}),
            ],
        )

        assert observed["runs"] == 2
        assert observed["cold_minutes"] == 11.0
        assert observed["drivers"] == {"shots": 2}

    def test_without_drivers_only_a_run_that_overrode_nothing_counts(self):
        """Thin, but never wrong - the fallback degrades to honesty rather
        than to a number describing runs of different sizes."""
        definition = workflow(None, num_frames=124)

        observed = observed_for(
            definition,
            [run(8 * MINUTE, {}), run(30 * MINUTE, {"num_frames": 345})],
        )

        assert observed["runs"] == 1
        assert observed["comparable"] == "default-arguments"
        assert "drivers" not in observed

    def test_an_argument_the_workflow_declares_no_variable_for_is_ignored(self):
        """A stale argument name cannot make a comparable run incomparable."""
        definition = workflow(None, num_frames=124)

        observed = observed_for(definition, [run(8 * MINUTE, {"gone": 1})])

        assert observed["runs"] == 1

    def test_a_driver_naming_no_variable_is_dropped(self):
        """Its effective value would be None for every run - one bucket
        wearing the look of a partition."""
        definition = workflow(["absent"], num_frames=124)

        assert declared_drivers(definition) == []
        observed = observed_for(
            definition, [run(8 * MINUTE, {}), run(30 * MINUTE, {"num_frames": 345})]
        )
        assert observed["comparable"] == "default-arguments"
        assert observed["runs"] == 1


class TestColdIsNotWarm:
    def test_the_two_are_reported_separately_each_with_its_runs(self):
        """Averaging 13.6 s and 6.3 s gives a number describing neither."""
        definition = workflow(["num_frames"], num_frames=124)

        observed = observed_for(
            definition,
            [
                run(10 * MINUTE, had_load=True),
                run(12 * MINUTE, had_load=True),
                run(2 * MINUTE, had_load=False),
            ],
        )

        assert observed["cold_minutes"] == 11.0 and observed["cold_runs"] == 2
        assert observed["warm_minutes"] == 2.0 and observed["warm_runs"] == 1
        assert observed["runs"] == 3

    def test_a_side_with_no_runs_is_absent_rather_than_null(self):
        observed = observed_for(
            workflow(["num_frames"], num_frames=124), [run(10 * MINUTE, had_load=True)]
        )

        assert "warm_minutes" not in observed and "warm_runs" not in observed

    def test_a_run_whose_events_were_trimmed_counts_as_neither(self):
        """Persisted events stop at 200, so a long run's `loading` phase can
        be gone. Calling it warm because the evidence was trimmed is the one
        answer that would be wrong."""
        definition = workflow(["num_frames"], num_frames=124)

        observed = observed_for(
            definition,
            [
                run(10 * MINUTE, had_load=True),
                run(40 * MINUTE, had_load=False, at_cap=True),
            ],
        )

        assert observed["cold_runs"] == 1
        assert "warm_minutes" not in observed
        assert observed["unclassified_runs"] == 1
        assert observed["runs"] == 2

    def test_there_is_no_unqualified_minutes_figure(self):
        """Curated `cost.minutes` includes model load. A number not saying
        which of the two it is would be read as comparable to it."""
        observed = observed_for(
            workflow(["num_frames"], num_frames=124),
            [run(10 * MINUTE), run(2 * MINUTE, had_load=False)],
        )

        assert "minutes" not in observed and "median_minutes" not in observed


class TestACachedRunIsNotARun:
    def test_a_fully_reused_run_is_excluded(self):
        """It finished in seconds and wrote nothing; counting it collapses
        the figure for exactly the templates that get re-run most."""
        definition = workflow(["num_frames"], num_frames=124)

        observed = observed_for(
            definition,
            [
                run(10 * MINUTE),
                run(0.05 * MINUTE, manifest=[{"reused": True}, {"reused": True}]),
            ],
        )

        assert observed["runs"] == 1
        assert observed["cold_minutes"] == 10.0

    def test_a_partly_reused_run_still_counts(self):
        definition = workflow(["num_frames"], num_frames=124)

        observed = observed_for(
            definition,
            [run(10 * MINUTE, manifest=[{"reused": True}, {"step": "b"}])],
        )

        assert observed["runs"] == 1


class TestWhatItReports:
    def test_nothing_comparable_is_none_rather_than_an_empty_block(self):
        assert observed_for(workflow(["num_frames"], num_frames=124), []) is None

    def test_the_range_says_how_much_is_behind_the_median(self):
        observed = observed_for(
            workflow(["num_frames"], num_frames=124),
            [run(8 * MINUTE), run(9 * MINUTE), run(20 * MINUTE)],
        )

        assert observed["cold_minutes"] == 9.0
        assert observed["cold_range_minutes"] == [8.0, 20.0]

    def test_since_says_how_old_the_oldest_run_is(self):
        observed = observed_for(
            workflow(["num_frames"], num_frames=124),
            [
                run(8 * MINUTE, finished_at=1_700_000_000.0),
                run(8 * MINUTE, finished_at=1_600_000_000.0),
            ],
        )

        assert observed["since"] == "2020-09-13T12:26:40Z"

    def test_the_device_is_this_boxs(self):
        observed = observed_for(
            workflow(["num_frames"], num_frames=124),
            [run(8 * MINUTE)],
            device="cuda",
            device_name="NVIDIA GeForce RTX 3090",
        )

        assert observed["device"] == "cuda"
        assert observed["name"] == "NVIDIA GeForce RTX 3090"

    def test_a_zero_length_run_is_not_a_measurement(self):
        assert observed_for(workflow(None), [run(0)]) is None


class TestOffTheJobRow:
    """Condition 2 of the approval: bucket from the job row, not the run
    directory - one query, and figures that survive a pruned run dir."""

    def test_the_history_groups_finished_runs_by_workflow_name(self, tmp_path):
        history = JobHistory(tmp_path / "jobs.sqlite")
        with history._connect() as connection:
            connection.execute(
                "INSERT INTO jobs (id, status, started_at, finished_at, arguments,"
                " manifest, events, workflow_name) VALUES (?,?,?,?,?,?,?,?)",
                (
                    "a",
                    "succeeded",
                    100.0,
                    700.0,
                    "{}",
                    "[]",
                    json.dumps([{"type": "phase", "phase": "loading"}]),
                    "templates/x",
                ),
            )
            connection.execute(
                "INSERT INTO jobs (id, status, started_at, finished_at, arguments,"
                " manifest, events, workflow_name) VALUES (?,?,?,?,?,?,?,?)",
                (
                    "b",
                    "succeeded",
                    100.0,
                    220.0,
                    "{}",
                    "[]",
                    json.dumps([{"type": "phase", "phase": "generating"}]),
                    "templates/x",
                ),
            )

        rows = history.finished_runs()

        assert set(rows) == {("default", "templates/x")}
        assert [row["duration"] for row in rows[("default", "templates/x")]] == [
            600.0,
            120.0,
        ]
        assert [row["had_load"] for row in rows[("default", "templates/x")]] == [
            True,
            False,
        ]
        assert not any(
            row["events_at_cap"] for row in rows[("default", "templates/x")]
        )

    def test_a_failed_or_unnamed_run_is_not_history(self, tmp_path):
        history = JobHistory(tmp_path / "jobs.sqlite")
        with history._connect() as connection:
            connection.executemany(
                "INSERT INTO jobs (id, status, started_at, finished_at, arguments,"
                " manifest, events, workflow_name) VALUES (?,?,?,?,?,?,?,?)",
                [
                    ("a", "failed", 1.0, 2.0, "{}", "[]", "[]", "templates/x"),
                    ("b", "cancelled", 1.0, 2.0, "{}", "[]", "[]", "templates/x"),
                    # No name: recorded before the column, or run inline
                    ("c", "succeeded", 1.0, 2.0, "{}", "[]", "[]", None),
                    # Never finished
                    ("d", "succeeded", 1.0, None, "{}", "[]", "[]", "templates/x"),
                ],
            )

        assert history.finished_runs() == {}

    def test_the_watermark_moves_when_a_row_lands(self, tmp_path):
        history = JobHistory(tmp_path / "jobs.sqlite")
        before = history.watermark()

        with history._connect() as connection:
            connection.execute(
                "INSERT INTO jobs (id, status, finished_at, workflow_name)"
                " VALUES (?,?,?,?)",
                ("a", "succeeded", 5.0, "templates/x"),
            )

        assert history.watermark() != before

    def test_the_watermark_moves_when_a_row_is_orphaned(self, tmp_path):
        """`orphan_workflow_history` (#274) detaches a row by clearing
        `workflow_name` rather than deleting it - an ordinary `COUNT(*)`
        would not see that, and `ObservedCosts` would keep serving the
        purged figure until an unrelated job happened to land."""
        history = JobHistory(tmp_path / "jobs.sqlite")
        with history._connect() as connection:
            connection.execute(
                "INSERT INTO jobs (id, status, started_at, finished_at,"
                " workspace, workflow_name) VALUES (?,?,?,?,?,?)",
                ("a", "succeeded", 1.0, 5.0, "default", "templates/x"),
            )
        before = history.watermark()
        assert ("default", "templates/x") in history.finished_runs()

        history.orphan_workflow_history("default", "templates/x")

        assert history.watermark() != before
        assert history.finished_runs() == {}

    def test_the_aggregate_recomputes_only_when_the_table_moves(self, tmp_path):
        history = JobHistory(tmp_path / "jobs.sqlite")
        costs = ObservedCosts(history)
        with history._connect() as connection:
            connection.execute(
                "INSERT INTO jobs (id, status, started_at, finished_at, arguments,"
                " manifest, events, workflow_name) VALUES (?,?,?,?,?,?,?,?)",
                ("a", "succeeded", 0.0, 600.0, "{}", "[]", "[]", "templates/x"),
            )

        assert costs.rows_for("templates/x")
        reads = []
        original = history.finished_runs
        history.finished_runs = lambda: (reads.append(1), original())[1]

        costs.rows_for("templates/x")
        costs.rows_for("templates/x")

        assert reads == [], "a warm cache re-queried the job table"

    def test_a_listing_reads_the_watermark_once(self, tmp_path):
        """list_workflows attaches a figure to every catalog entry; the
        watermark is a COUNT(*) under the history lock the worker also
        needs, so a listing takes it once, not once per workflow."""
        from dw.server.app import attach_observed

        history = JobHistory(tmp_path / "jobs.sqlite")
        with history._connect() as connection:
            connection.execute(
                "INSERT INTO jobs (id, status, started_at, finished_at, arguments,"
                " manifest, events, workflow_name) VALUES (?,?,?,?,?,?,?,?)",
                ("a", "succeeded", 0.0, 600.0, "{}", "[]", "[]", "templates/x"),
            )
        costs = ObservedCosts(history)
        reads = []
        original = history.watermark
        history.watermark = lambda: (reads.append(1), original())[1]

        details = {
            name: {"cost_drivers": {}, "variable_names": []}
            for name in ("templates/x", "templates/y", "templates/z")
        }
        attach_observed(details, costs)

        assert len(reads) == 1

    def test_the_device_is_read_from_symbols_that_exist(self):
        """Caught in deployment: the lookup imported `get_memory_stats`,
        which is spelled `device_memory_stats`, and one try around both
        calls turned the ImportError into `device: null` on a box plainly
        running on CUDA - the silent null this field exists to replace. This
        asserts the symbols resolve rather than that any particular card is
        present, since the suite runs on CUDA, MPS and CPU."""
        from dw import device_memory_stats, get_device, get_device_type

        assert callable(get_device) and callable(get_device_type)
        assert "device_name" in device_memory_stats()

        kind, _card = ObservedCosts(None).device()
        assert kind == get_device_type(get_device())

    def test_the_device_is_looked_up_once(self):
        costs = ObservedCosts(None)
        first = costs.device()

        costs._device = ("sentinel", "sentinel")

        assert costs.device() == ("sentinel", "sentinel") and first is not None

    def test_no_history_is_no_figure_rather_than_an_error(self):
        costs = ObservedCosts(None)

        assert costs.rows_for("templates/x") == []
        assert costs.observed("templates/x", workflow(None)) is None


class TestWorkspaceScoping:
    """#274: two workspaces that each save a workflow called the same name
    are different workflows, so a workspace-writable save's history must not
    leak into the other's. A shared, read-only catalog source keeps pooling
    across every workspace that ran it (#154) - that is what omitting
    `workspace` still means."""

    def _seeded_history(self, tmp_path):
        history = JobHistory(tmp_path / "jobs.sqlite")
        with history._connect() as connection:
            connection.executemany(
                "INSERT INTO jobs (id, status, started_at, finished_at,"
                " arguments, manifest, events, workspace, workflow_name)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    ("a", "succeeded", 0.0, 600.0, "{}", "[]", "[]", "ws-one", "shots"),
                    ("b", "succeeded", 0.0, 300.0, "{}", "[]", "[]", "ws-two", "shots"),
                    (
                        "c",
                        "succeeded",
                        0.0,
                        60.0,
                        "{}",
                        "[]",
                        "[]",
                        "ws-one",
                        "templates/catalog",
                    ),
                    (
                        "d",
                        "succeeded",
                        0.0,
                        90.0,
                        "{}",
                        "[]",
                        "[]",
                        "ws-two",
                        "templates/catalog",
                    ),
                ],
            )
        return history

    def test_a_workspace_writable_names_history_is_scoped_to_its_own_workspace(
        self, tmp_path
    ):
        history = self._seeded_history(tmp_path)
        costs = ObservedCosts(history)

        assert [row["duration"] for row in costs.rows_for("shots", workspace="ws-one")] == [
            600.0
        ]
        assert [row["duration"] for row in costs.rows_for("shots", workspace="ws-two")] == [
            300.0
        ]

    def test_a_shared_catalog_name_pools_across_every_workspace(self, tmp_path):
        history = self._seeded_history(tmp_path)
        costs = ObservedCosts(history)

        durations = sorted(
            row["duration"] for row in costs.rows_for("templates/catalog")
        )

        assert durations == [60.0, 90.0]

    def test_a_workspace_with_no_rows_of_its_own_sees_none(self, tmp_path):
        history = self._seeded_history(tmp_path)
        costs = ObservedCosts(history)

        assert costs.rows_for("shots", workspace="ws-three") == []


class TestDeleteWorkflowPurgesHistory:
    """#274: deleting a workflow must purge its (workspace, name) history so
    a name reused afterwards - in this workspace or a fresh copy of it - does
    not inherit the deleted copy's figures."""

    def test_orphaning_removes_only_that_workspace_and_name(self, tmp_path):
        history = JobHistory(tmp_path / "jobs.sqlite")
        with history._connect() as connection:
            connection.executemany(
                "INSERT INTO jobs (id, status, started_at, finished_at,"
                " arguments, manifest, events, workspace, workflow_name)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    ("a", "succeeded", 0.0, 60.0, "{}", "[]", "[]", "ws-one", "shots"),
                    ("b", "succeeded", 0.0, 90.0, "{}", "[]", "[]", "ws-two", "shots"),
                ],
            )

        history.orphan_workflow_history("ws-one", "shots")
        rows = history.finished_runs()

        assert ("ws-one", "shots") not in rows
        assert [row["duration"] for row in rows[("ws-two", "shots")]] == [90.0]


# Workflows that carry a curated `cost` and deliberately declare no drivers,
# with why. Listed rather than inferred, so adding a measured workflow is a
# decision about what moves its cost rather than a silent omission.
NO_DRIVERS = {
    # What it costs is the size of the image handed to it, which is an
    # asset rather than a variable - the accepted limit of static drivers.
    # `tile_size` only trades memory against passes over that same image.
    "workflows/templates/upscale-spandrel.json": "cost follows the input image",
    # Three prompt edits over one image: nothing numeric moves the run.
    "workflows/templates/consistent-set.json": "no variable moves the cost",
}


class TestTheCatalogsDriversAreReal:
    """The sweep: a driver naming no variable checks nothing, which is the
    failure this whole module exists to avoid."""

    def test_every_declared_driver_is_a_variable_of_its_workflow(self):
        for path in sorted(
            glob.glob(
                os.path.join(REPO_ROOT, "workflows", "**", "*.json"), recursive=True
            )
        ):
            with open(path, encoding="utf-8") as handle:
                definition = json.load(handle)
            declared = definition.get("cost_drivers")
            if declared is None:
                continue
            variables = definition.get("variables") or {}
            for name in declared:
                assert name in variables, (
                    f"{os.path.relpath(path, REPO_ROOT)} declares cost driver "
                    f"'{name}', which it does not declare as a variable"
                )

    def test_a_driver_is_a_size_not_a_piece_of_text(self):
        """A prompt, a seed or a URL moves the output, not the wall clock,
        and bucketing on one gives every run its own bucket and so a
        permanent `runs: 1`. `num_images_per_prompt` is a count and is
        fine - the rule is the value's *type*, not the name's spelling."""
        never = {"seed", "prompt", "negative_prompt"}
        for path in sorted(
            glob.glob(
                os.path.join(REPO_ROOT, "workflows", "**", "*.json"), recursive=True
            )
        ):
            with open(path, encoding="utf-8") as handle:
                definition = json.load(handle)
            variables = definition.get("variables") or {}
            where = os.path.relpath(path, REPO_ROOT)
            for name in definition.get("cost_drivers") or []:
                assert name not in never, f"{where} buckets on '{name}'"
                assert not isinstance(variables.get(name), str), (
                    f"{where} buckets on '{name}', whose default is text"
                )

    def test_every_workflow_carrying_a_curated_cost_declares_its_drivers(self):
        """With no drivers the fallback is default-arguments-only, and real
        runs pass arguments - so an undeclared catalog keeps answering null
        for exactly the templates somebody bothered to measure."""
        missing = []
        for path in sorted(
            glob.glob(
                os.path.join(REPO_ROOT, "workflows", "**", "*.json"), recursive=True
            )
        ):
            with open(path, encoding="utf-8") as handle:
                definition = json.load(handle)
            if not definition.get("cost"):
                continue
            if definition.get("cost_drivers"):
                continue
            relative = os.path.relpath(path, REPO_ROOT)
            if relative in NO_DRIVERS:
                continue
            missing.append(relative)
        assert missing == []

    def test_the_minimax_templates_declare_drivers(self):
        """#91's report was about these specifically."""
        found = glob.glob(
            os.path.join(REPO_ROOT, "workflows", "templates", "minimax", "*.json")
        )
        assert found
        for path in found:
            with open(path, encoding="utf-8") as handle:
                definition = json.load(handle)
            assert definition.get("cost_drivers"), os.path.relpath(path, REPO_ROOT)


@pytest.mark.parametrize("value", [None, "", "[]", "not json"])
def test_a_manifest_that_will_not_parse_is_not_a_cached_run(value):
    from dw.server.observed_cost import _every_step_was_reused

    assert _every_step_was_reused(value) is False


class TestTheBucketAPlanAsksFor:
    """#154: the listing asks with no arguments and gets the defaults'
    bucket; a plan asks with the caller's own values and gets the figure for
    the run it is about to quote."""

    def test_the_callers_values_choose_the_bucket(self):
        definition = workflow(["num_frames"], num_frames=124)

        observed = observed_for(
            definition,
            [
                run(8 * MINUTE, {"num_frames": 124}),
                run(16 * MINUTE, {"num_frames": 248}),
            ],
            arguments={"num_frames": 248},
        )

        assert observed["cold_minutes"] == 16.0
        assert observed["cold_runs"] == 1
        # And says which run it is a figure for, not what the defaults are
        assert observed["drivers"] == {"num_frames": 248}

    def test_a_shape_this_box_has_never_run_reports_nothing(self):
        definition = workflow(["num_frames"], num_frames=124)

        assert (
            observed_for(
                definition,
                [run(8 * MINUTE, {"num_frames": 124})],
                arguments={"num_frames": 500},
            )
            is None
        )

    def test_without_declared_drivers_an_override_is_not_comparable(self):
        """Nothing is declared to matter, so a run that changed something
        cannot be said to be comparable to one that did not."""
        definition = workflow(num_frames=124)

        assert (
            observed_for(
                definition,
                [run(8 * MINUTE)],
                arguments={"num_frames": 124},
            )
            is None
        )

    def test_a_list_driver_buckets_on_the_callers_length(self):
        definition = workflow(["shots"], shots=[{"name": "a"}, {"name": "b"}])

        observed = observed_for(
            definition,
            [
                run(10 * MINUTE, {"shots": [{"name": "a"}, {"name": "b"}]}),
                run(20 * MINUTE, {"shots": [{"name": n} for n in "abcd"]}),
            ],
            arguments={"shots": [{"name": n} for n in "wxyz"]},
        )

        assert observed["cold_minutes"] == 20.0
        assert observed["drivers"] == {"shots": 4}
