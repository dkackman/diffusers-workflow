"""Host-RAM projection for a list-driven or composed run (#243).

Warn, not refuse: this module never fails a run, only says the caller's own
history projects past what this machine's RAM can hold. Each rule below is a
way a naive projection would mislead, so each has a test that fails if the
rule is dropped.
"""

import json

from dw.host_memory_projection import host_memory_warnings, releases_between_iterations

MB = 1.0


def row(peak_mb, arguments=None):
    return {
        "arguments": json.dumps(arguments or {}),
        "host_memory_peak_rss_mb": peak_mb,
    }


def for_each_workflow(release=False):
    step = {
        "name": "shot",
        "for_each": "variable:shots",
        "task": "noop",
    }
    if release:
        step["release_pipeline"] = True
    return {"id": "w", "variables": {"shots": []}, "steps": [step]}


class TestReleaseDetection:
    def test_no_for_each_step_is_treated_as_released(self):
        definition = {"id": "w", "variables": {}, "steps": [{"name": "a"}]}
        assert releases_between_iterations(definition) is True

    def test_release_pipeline_on_the_for_each_step_is_released(self):
        assert releases_between_iterations(for_each_workflow(release=True)) is True

    def test_no_release_flag_keeps_the_pipeline_resident(self):
        assert releases_between_iterations(for_each_workflow(release=False)) is False


class TestProjection:
    def test_cold_start_warns_nothing(self):
        definition = for_each_workflow(release=False)
        warnings = host_memory_warnings(definition, {"shots": 12}, [], 10_000 * MB)
        assert warnings == []

    def test_history_with_no_memory_reading_warns_nothing(self):
        definition = for_each_workflow(release=False)
        rows = [
            {
                "arguments": json.dumps({"shots": [1, 2, 3]}),
                "host_memory_peak_rss_mb": None,
            }
        ]
        warnings = host_memory_warnings(definition, {"shots": 12}, rows, 10_000 * MB)
        assert warnings == []

    def test_resident_pipeline_projects_per_entry_times_requested_count(self):
        definition = for_each_workflow(release=False)
        # 3-entry runs each peaked at 3000 MB -> 1000 MB/entry; 12 entries
        # projects to 12000 MB, over an 8000 MB ceiling
        rows = [row(3000 * MB, {"shots": [1, 2, 3]}) for _ in range(3)]
        warnings = host_memory_warnings(definition, {"shots": 12}, rows, 8_000 * MB)
        assert len(warnings) == 1
        assert "12000" in warnings[0] or "12,000" in warnings[0]

    def test_resident_pipeline_under_the_ceiling_warns_nothing(self):
        definition = for_each_workflow(release=False)
        rows = [row(3000 * MB, {"shots": [1, 2, 3]}) for _ in range(3)]
        warnings = host_memory_warnings(definition, {"shots": 4}, rows, 8_000 * MB)
        assert warnings == []

    def test_released_pipeline_projects_the_largest_single_iteration(self):
        definition = for_each_workflow(release=True)
        # Released between iterations: projection does not scale with count,
        # since only one entry's pipeline is ever resident at a time
        rows = [row(3000 * MB, {"shots": [1, 2, 3]}), row(3200 * MB, {"shots": [1]})]
        warnings = host_memory_warnings(definition, {"shots": 12}, rows, 8_000 * MB)
        assert warnings == []

    def test_released_pipeline_over_the_ceiling_still_warns(self):
        definition = for_each_workflow(release=True)
        rows = [row(9000 * MB, {"shots": [1, 2, 3]})]
        warnings = host_memory_warnings(definition, {"shots": 12}, rows, 8_000 * MB)
        assert len(warnings) == 1

    def test_no_list_in_this_request_warns_nothing(self):
        definition = for_each_workflow(release=False)
        rows = [row(9000 * MB, {"shots": [1, 2, 3]})]
        warnings = host_memory_warnings(definition, {}, rows, 100 * MB)
        assert warnings == []

    def test_identical_shape_that_already_succeeded_warns_nothing(self):
        # #254: a 2-entry run that just succeeded at ~62 GB shouldn't warn
        # about a repeat request for the same 2 entries, even though 62 GB
        # is itself above the ceiling - it already happened without incident.
        definition = for_each_workflow(release=False)
        rows = [row(62197 * MB, {"shots": [1, 2]})]
        warnings = host_memory_warnings(definition, {"shots": 2}, rows, 57788 * MB)
        assert warnings == []

    def test_larger_request_than_any_survived_shape_still_warns(self):
        definition = for_each_workflow(release=False)
        rows = [row(62197 * MB, {"shots": [1, 2]})]
        warnings = host_memory_warnings(definition, {"shots": 4}, rows, 57788 * MB)
        assert len(warnings) == 1
