"""A `shots` argument to an assessment probe reaching past a
statically-knowable video's real length, warned about at validate time
rather than only at run time - #425.

Exercises the real decode path (`probe_media` against a genuine mp4 file)
rather than a mock of it, mirroring tests/test_slice_preflight.py (#402).
"""

import os
import tempfile

from dw.runs import activate_output_root, deactivate_output_root
from dw.shot_span_preflight import shot_span_warnings
from dw.workflow import workflow_from_definition

from tests.test_assess import write_mp4


def workflow_dir_with_asset(monkeypatch, name, frames):
    base_dir = tempfile.mkdtemp()
    asset_dir = os.path.join(base_dir, "assets")
    os.makedirs(asset_dir)
    write_mp4(os.path.join(asset_dir, name), frames=frames, fps=24)
    monkeypatch.setenv("DW_ASSET_DIR", asset_dir)
    return base_dir


def seams_workflow(video, shots):
    return {
        "id": "check",
        "steps": [
            {
                "name": "seams",
                "task": {
                    "command": "analyze_seams",
                    "arguments": {"video": video, "shots": shots},
                },
                "result": {"content_type": "application/json"},
            }
        ],
    }


class TestTheCheck:
    def test_a_shots_record_past_a_short_asset_is_warned(self, monkeypatch):
        # Mirrors the issue's own repro shape: a shot reaching well past the
        # file's real frame count.
        base_dir = workflow_dir_with_asset(monkeypatch, "clip.mp4", frames=248)
        definition = seams_workflow(
            "asset:clip.mp4",
            [
                {"name": "a", "start_frame": 0, "num_frames": 124},
                {"name": "b", "start_frame": 124, "num_frames": 300},
            ],
        )

        warnings = shot_span_warnings(definition, base_dir=base_dir)

        assert len(warnings) == 1
        assert "'b'" in warnings[0]
        assert "176 past the file's 248 frames" in warnings[0]

    def test_shots_within_the_source_validate_clean(self, monkeypatch):
        base_dir = workflow_dir_with_asset(monkeypatch, "clip.mp4", frames=248)
        definition = seams_workflow(
            "asset:clip.mp4",
            [
                {"name": "a", "start_frame": 0, "num_frames": 124},
                {"name": "b", "start_frame": 124, "num_frames": 124},
            ],
        )

        assert shot_span_warnings(definition, base_dir=base_dir) == []

    def test_a_previous_result_video_is_left_to_the_run(self):
        definition = seams_workflow(
            "previous_result:make_cut",
            [{"name": "a", "start_frame": 0, "num_frames": 300}],
        )

        assert shot_span_warnings(definition) == []

    def test_an_output_reference_too_short_is_warned(self):
        output_root = tempfile.mkdtemp()
        run_dir = os.path.join(output_root, "cut", "20260101-000000-abc")
        os.makedirs(run_dir)
        write_mp4(os.path.join(run_dir, "final.mp4"), frames=48, fps=24)
        definition = seams_workflow(
            "output:cut/20260101-000000-abc/final.mp4",
            [{"name": "a", "start_frame": 0, "num_frames": 300}],
        )

        token = activate_output_root(output_root)
        try:
            warnings = shot_span_warnings(definition)
        finally:
            deactivate_output_root(token)

        assert len(warnings) == 1
        assert "final.mp4" not in warnings[0]
        assert "252 past the file's 48 frames" in warnings[0]

    def test_nothing_is_reported_for_a_definition_with_no_probe_step(self):
        assert shot_span_warnings({"steps": [{"name": "a", "task": {}}]}) == []


class TestWiredIntoTheWorkflow:
    def test_reachable_from_the_workflow_method(self, monkeypatch):
        base_dir = workflow_dir_with_asset(monkeypatch, "clip.mp4", frames=248)
        definition = seams_workflow(
            "asset:clip.mp4",
            [
                {"name": "a", "start_frame": 0, "num_frames": 124},
                {"name": "b", "start_frame": 124, "num_frames": 300},
            ],
        )
        workflow = workflow_from_definition(
            definition, os.path.join(base_dir, "workflow.json")
        )

        warnings = workflow.shot_span_warnings()

        assert len(warnings) == 1
        assert "'b'" in warnings[0]
