"""A join_windows window list of the wrong length, refused before the run -
#601.

These tests use real mp4 fixtures and the real header probe, with a plain
hand-written windowed workflow (a `for_each` window step and a `join_windows`
step) rather than a template.
"""

from dw.media import probe_metadata
from dw.window_count_errors import window_count_errors
from dw.workflow import workflow_from_definition

from .test_dissolve_frame_errors import workflow_dir_with_asset

NUM_FRAMES = 17
OVERLAP = 4  # stride 13: a 50-frame source needs ceil(50 / 13) = 4 windows


def window_workflow(window_count, num_frames=NUM_FRAMES, overlap=OVERLAP):
    return {
        "id": "windowed",
        "variables": {
            "source_video": "asset:long.mp4",
            "windows": [{"name": f"w{i}", "index": i} for i in range(window_count)],
        },
        "steps": [
            {
                "name": "window",
                "for_each": "variable:windows",
                "task": {
                    "command": "window_video",
                    "arguments": {
                        "video": "variable:source_video",
                        "index": "item:index",
                        "num_frames": num_frames,
                        "overlap": overlap,
                    },
                },
                "result": {"content_type": "video/mp4", "fps": 6},
            },
            {
                "name": "join",
                "task": {
                    "command": "join_windows",
                    "arguments": {
                        "videos": "gather:window",
                        "source": "variable:source_video",
                        "num_frames": num_frames,
                        "overlap": overlap,
                    },
                },
                "result": {"content_type": "video/mp4", "fps": 6},
            },
        ],
    }


def validate(monkeypatch, tmp_path, window_count, frames=50):
    import os

    base_dir = workflow_dir_with_asset(monkeypatch, tmp_path, ("long.mp4", frames))
    workflow = workflow_from_definition(
        window_workflow(window_count), os.path.join(base_dir, "workflow.json")
    )
    return [
        problem
        for problem in workflow.validation_errors()
        if "join_windows needs" in problem["message"]
    ]


def expanded(count, **arguments):
    """An already-expanded definition: `videos` a literal list of references."""
    base = {
        "videos": [f"previous_result:window@w{i}" for i in range(count)],
        "source": "asset:long.mp4",
        "num_frames": NUM_FRAMES,
        "overlap": OVERLAP,
    }
    base.update(arguments)
    return {
        "steps": [
            {
                "name": "join",
                "task": {"command": "join_windows", "arguments": base},
            }
        ]
    }


class TestThroughValidation:
    def test_a_short_list_is_refused(self, monkeypatch, tmp_path):
        problems = validate(monkeypatch, tmp_path, 3)

        assert len(problems) == 1
        assert problems[0]["path"] == "steps[1]"
        assert "needs 4 windows" in problems[0]["message"]
        assert "got 3" in problems[0]["message"]
        assert "add 1 entry" in problems[0]["message"]

    def test_the_exact_list_validates_clean(self, monkeypatch, tmp_path):
        assert validate(monkeypatch, tmp_path, 4) == []

    def test_a_long_list_names_the_drop(self, monkeypatch, tmp_path):
        problems = validate(monkeypatch, tmp_path, 5)

        assert len(problems) == 1
        assert problems[0]["path"] == "steps[1]"
        assert "needs 4 windows" in problems[0]["message"]
        assert "got 5" in problems[0]["message"]
        assert "drop 1 entry (index 4)" in problems[0]["message"]

    def test_an_unreadable_literal_source_is_refused_at_its_path(
        self, monkeypatch, tmp_path
    ):
        """SE-F042 (#630): the source the join reads is path-gated here, the
        same as window_video's video, rather than only when the run joins."""
        import os

        from dw.trust import TRUST_WORKFLOWS_ENV_VAR

        base_dir = workflow_dir_with_asset(monkeypatch, tmp_path, ("long.mp4", 50))
        monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")
        definition = window_workflow(4)
        definition["steps"][1]["task"]["arguments"]["source"] = "/etc/passwd"
        workflow = workflow_from_definition(
            definition, os.path.join(base_dir, "workflow.json")
        )

        paths = [problem["path"] for problem in workflow.validation_errors()]

        assert "steps[1].task.arguments.source" in paths


class TestDirectCalls:
    def test_a_short_list_is_reported(self, monkeypatch, tmp_path):
        base_dir = workflow_dir_with_asset(monkeypatch, tmp_path, ("long.mp4", 50))

        problems = window_count_errors(expanded(3), base_dir=base_dir)

        assert len(problems) == 1
        assert problems[0]["path"] == "steps[0]"
        assert "needs 4 windows" in problems[0]["message"]

    def test_the_exact_list_is_silent(self, monkeypatch, tmp_path):
        base_dir = workflow_dir_with_asset(monkeypatch, tmp_path, ("long.mp4", 50))

        assert window_count_errors(expanded(4), base_dir=base_dir) == []

    def test_a_previous_result_source_is_silent(self, monkeypatch, tmp_path):
        base_dir = workflow_dir_with_asset(monkeypatch, tmp_path, ("long.mp4", 50))
        definition = expanded(3, source="previous_result:something")

        assert window_count_errors(definition, base_dir=base_dir) == []

    def test_a_non_literal_num_frames_is_silent(self, monkeypatch, tmp_path):
        base_dir = workflow_dir_with_asset(monkeypatch, tmp_path, ("long.mp4", 50))
        definition = expanded(3, num_frames="variable:num_frames")

        assert window_count_errors(definition, base_dir=base_dir) == []

    def test_a_missing_asset_is_silent(self, monkeypatch, tmp_path):
        base_dir = workflow_dir_with_asset(monkeypatch, tmp_path, ("long.mp4", 50))
        definition = expanded(3, source="asset:absent.mp4")

        assert window_count_errors(definition, base_dir=base_dir) == []

    def test_an_overlap_not_below_num_frames_is_left_to_join_windows_errors(
        self, monkeypatch, tmp_path
    ):
        base_dir = workflow_dir_with_asset(monkeypatch, tmp_path, ("long.mp4", 50))

        for overlap in (NUM_FRAMES, NUM_FRAMES + 3):
            definition = expanded(3, overlap=overlap)
            assert window_count_errors(definition, base_dir=base_dir) == []

    def test_a_videos_that_is_not_a_list_is_silent(self, monkeypatch, tmp_path):
        base_dir = workflow_dir_with_asset(monkeypatch, tmp_path, ("long.mp4", 50))

        for videos in ("gather:window", None, {"a": 1}):
            definition = expanded(3, videos=videos)
            assert window_count_errors(definition, base_dir=base_dir) == []

    def test_a_float_literal_num_frames_still_checks(self, monkeypatch, tmp_path):
        base_dir = workflow_dir_with_asset(monkeypatch, tmp_path, ("long.mp4", 50))

        problems = window_count_errors(
            expanded(3, num_frames=17.0, overlap=4.0), base_dir=base_dir
        )

        assert len(problems) == 1
        assert "needs 4 windows" in problems[0]["message"]

    def test_a_caller_supplied_probe_is_used(self, monkeypatch, tmp_path):
        base_dir = workflow_dir_with_asset(monkeypatch, tmp_path, ("long.mp4", 50))
        calls = []

        def counting_probe(path):
            calls.append(path)
            return probe_metadata(path)

        window_count_errors(expanded(3), base_dir=base_dir, probe=counting_probe)

        assert len(calls) == 1

        # a stub answering 999 frames needs far more than 3 windows
        stub = window_count_errors(
            expanded(3),
            base_dir=base_dir,
            probe=lambda path: {"kind": "video", "frame_count": 999},
        )
        assert "needs 77 windows" in stub[0]["message"]
