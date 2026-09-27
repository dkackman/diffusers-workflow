"""A dissolve_videos/concat_videos frame-size mismatch, refused before the
run when the sizes are already knowable - #504.

Both tasks only discover a size mismatch after decoding every input
(`check_same_frame_size`, dw/tasks/video_utils.py); these tests exercise the
real decode path (`probe_media` against a genuine mp4) rather than a mock of
it, so a fixture pair that disagrees in size is the same pair the run itself
would have failed on.
"""

import os
import tempfile

from tests.test_dissolve_frame_errors import write_mp4, workflow_dir_with_asset

from dw.runs import activate_output_root, deactivate_output_root
from dw.video_size_errors import video_size_errors
from dw.workflow import workflow_from_definition


def join_workflow(command, videos, extra_arguments=None):
    arguments = {"videos": videos}
    if extra_arguments:
        arguments.update(extra_arguments)
    return {
        "id": "joining",
        "steps": [
            {
                "name": "join",
                "task": {"command": command, "arguments": arguments},
                "result": {"content_type": "video/mp4", "fps": 6},
            }
        ],
    }


class TestTheCheck:
    def test_mismatched_asset_sizes_are_refused_for_dissolve_videos(self, monkeypatch):
        base_dir = workflow_dir_with_asset(monkeypatch, ("a.mp4", 12), ("b.mp4", 12))
        write_mp4(os.path.join(base_dir, "assets", "b.mp4"), frames=12, width=64, height=32)
        definition = join_workflow(
            "dissolve_videos", ["asset:a.mp4", "asset:b.mp4"]
        )

        problems = video_size_errors(definition, base_dir=base_dir)

        assert len(problems) == 1
        assert problems[0]["path"] == "steps[0].task.arguments.videos"
        assert "32x16" in problems[0]["message"]
        assert "64x32" in problems[0]["message"]

    def test_mismatched_asset_sizes_are_refused_for_concat_videos(self, monkeypatch):
        base_dir = workflow_dir_with_asset(monkeypatch, ("a.mp4", 12), ("b.mp4", 12))
        write_mp4(os.path.join(base_dir, "assets", "b.mp4"), frames=12, width=64, height=32)
        definition = join_workflow("concat_videos", ["asset:a.mp4", "asset:b.mp4"])

        problems = video_size_errors(definition, base_dir=base_dir)

        assert len(problems) == 1
        assert "concat_videos needs every video at one size" in problems[0]["message"]

    def test_matching_asset_sizes_validate_clean(self, monkeypatch):
        base_dir = workflow_dir_with_asset(monkeypatch, ("a.mp4", 12), ("b.mp4", 12))
        definition = join_workflow("dissolve_videos", ["asset:a.mp4", "asset:b.mp4"])

        assert video_size_errors(definition, base_dir=base_dir) == []

    def test_a_previous_result_video_is_left_to_the_run(self):
        definition = join_workflow(
            "dissolve_videos", ["previous_result:make_a", "previous_result:make_b"]
        )

        assert video_size_errors(definition) == []

    def test_a_mismatch_against_an_unresolvable_entry_is_left_to_the_run(
        self, monkeypatch
    ):
        base_dir = workflow_dir_with_asset(monkeypatch, ("a.mp4", 12))
        definition = join_workflow(
            "dissolve_videos", ["asset:a.mp4", "previous_result:make_b"]
        )

        assert video_size_errors(definition, base_dir=base_dir) == []

    def test_an_output_reference_mismatch_is_refused(self):
        output_root = tempfile.mkdtemp()
        run_dir = os.path.join(output_root, "clip", "20260101-000000-abc")
        os.makedirs(run_dir)
        write_mp4(os.path.join(run_dir, "a.mp4"), frames=12, width=32, height=16)
        write_mp4(os.path.join(run_dir, "b.mp4"), frames=12, width=64, height=32)
        definition = join_workflow(
            "concat_videos",
            [
                "output:clip/20260101-000000-abc/a.mp4",
                "output:clip/20260101-000000-abc/b.mp4",
            ],
        )

        token = activate_output_root(output_root)
        try:
            problems = video_size_errors(definition)
        finally:
            deactivate_output_root(token)

        assert len(problems) == 1
        assert "32x16" in problems[0]["message"]
        assert "64x32" in problems[0]["message"]

    def test_nothing_is_reported_for_a_definition_with_no_join_step(self):
        assert video_size_errors({"steps": [{"name": "a", "task": {}}]}) == []

    def test_a_single_video_needs_no_size_check(self, monkeypatch):
        base_dir = workflow_dir_with_asset(monkeypatch, ("a.mp4", 5))
        definition = join_workflow("dissolve_videos", ["asset:a.mp4"])

        assert video_size_errors(definition, base_dir=base_dir) == []


class TestTheValidationPass:
    def test_wired_into_validation_errors(self, monkeypatch):
        base_dir = workflow_dir_with_asset(monkeypatch, ("a.mp4", 12), ("b.mp4", 12))
        write_mp4(os.path.join(base_dir, "assets", "b.mp4"), frames=12, width=64, height=32)
        definition = join_workflow("concat_videos", ["asset:a.mp4", "asset:b.mp4"])
        workflow = workflow_from_definition(
            definition, os.path.join(base_dir, "workflow.json")
        )

        problems = workflow.validation_errors()

        assert any(
            problem["path"] == "steps[0].task.arguments.videos"
            for problem in problems
        )
