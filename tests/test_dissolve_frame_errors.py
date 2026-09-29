"""A dissolve_videos overlap wider than a statically-resolvable input's real
frame count, refused before the run - #400.

`dissolve_videos` itself only discovers a too-short input after decoding
every video in the step; these tests exercise the real decode path
(`probe_media` against a genuine mp4) rather than a mock of it, so a fixture
short of its declared dissolve_frames is the same file the run itself would
have failed on.
"""

import os
import tempfile

import numpy

from dw.dissolve_frame_errors import dissolve_frame_errors
from dw.media_info import probe_metadata
from dw.runs import activate_output_root, deactivate_output_root
from dw.workflow import workflow_from_definition


def write_mp4(path, frames=12, fps=6, width=32, height=16):
    import av

    container = av.open(str(path), "w")
    video = container.add_stream("libx264", rate=fps)
    video.width, video.height, video.pix_fmt = width, height, "yuv420p"
    for _ in range(frames):
        frame = av.VideoFrame.from_ndarray(
            numpy.zeros((height, width, 3), numpy.uint8), format="rgb24"
        )
        for packet in video.encode(frame):
            container.mux(packet)
    for packet in video.encode():
        container.mux(packet)
    container.close()


def workflow_dir_with_asset(monkeypatch, *names_and_frames):
    """A base_dir whose assets/ subfolder holds the given fixtures, pinned
    via DW_ASSET_DIR so it is found ahead of this checkout's own assets/
    (discover_library's './assets' in the working directory would otherwise
    shadow it)."""
    base_dir = tempfile.mkdtemp()
    asset_dir = os.path.join(base_dir, "assets")
    os.makedirs(asset_dir)
    for name, frames in names_and_frames:
        write_mp4(os.path.join(asset_dir, name), frames=frames)
    monkeypatch.setenv("DW_ASSET_DIR", asset_dir)
    return base_dir


def dissolve_workflow(videos, dissolve_frames=12):
    return {
        "id": "dissolving",
        "steps": [
            {
                "name": "join",
                "task": {
                    "command": "dissolve_videos",
                    "arguments": {"videos": videos, "dissolve_frames": dissolve_frames},
                },
                "result": {"content_type": "video/mp4", "fps": 6},
            }
        ],
    }


class TestTheCheck:
    def test_an_asset_too_short_for_its_dissolve_is_refused(self, monkeypatch):
        base_dir = workflow_dir_with_asset(monkeypatch, ("a.mp4", 124), ("b.mp4", 124))
        definition = dissolve_workflow(
            ["asset:a.mp4", "asset:b.mp4"], dissolve_frames=130
        )

        problems = dissolve_frame_errors(definition, base_dir=base_dir)

        assert len(problems) == 1
        assert problems[0]["path"] == "steps[0].task.arguments.dissolve_frames"
        assert "124 frames" in problems[0]["message"]
        assert "130 frames" in problems[0]["message"]

    def test_a_location_dict_too_short_for_its_dissolve_is_refused(self, monkeypatch):
        # dissolve_videos accepts a {"location": ...} entry (#510), so the
        # check must probe what it wraps rather than skip it
        base_dir = workflow_dir_with_asset(monkeypatch, ("a.mp4", 12), ("b.mp4", 4))
        definition = dissolve_workflow(
            ["asset:a.mp4", {"location": "asset:b.mp4"}], dissolve_frames=10
        )

        problems = dissolve_frame_errors(definition, base_dir=base_dir)

        assert len(problems) == 1
        assert "video 1 has 4 frames" in problems[0]["message"]

    def test_a_location_dict_is_confined_like_a_plain_path(self, monkeypatch):
        from dw.probe_paths import resolve_probe_path

        base_dir = workflow_dir_with_asset(monkeypatch, ("a.mp4", 12))
        outside = os.path.join(os.path.dirname(base_dir), "outside.mp4")
        for value in (
            "asset:a.mp4",
            "assets/a.mp4",
            "../outside.mp4",
            outside,
            "previous_result:make_a",
            "asset:missing.mp4",
        ):
            plain = resolve_probe_path(value, base_dir)
            assert resolve_probe_path({"location": value}, base_dir) == plain
        assert resolve_probe_path({"location": "asset:a.mp4"}, base_dir) is not None
        assert resolve_probe_path({"location": {"location": "a"}}, base_dir) is None
        assert resolve_probe_path({"path": "asset:a.mp4"}, base_dir) is None

    def test_enough_frames_validates_clean(self, monkeypatch):
        base_dir = workflow_dir_with_asset(monkeypatch, ("a.mp4", 124), ("b.mp4", 124))
        definition = dissolve_workflow(
            ["asset:a.mp4", "asset:b.mp4"], dissolve_frames=12
        )

        assert dissolve_frame_errors(definition, base_dir=base_dir) == []

    def test_a_previous_result_video_is_left_to_the_run(self):
        definition = dissolve_workflow(
            ["previous_result:make_a", "previous_result:make_b"],
            dissolve_frames=130,
        )

        assert dissolve_frame_errors(definition) == []

    def test_an_output_reference_too_short_for_its_dissolve_is_refused(self):
        output_root = tempfile.mkdtemp()
        run_dir = os.path.join(output_root, "clip", "20260101-000000-abc")
        os.makedirs(run_dir)
        write_mp4(os.path.join(run_dir, "a.mp4"), frames=124)
        write_mp4(os.path.join(run_dir, "b.mp4"), frames=124)
        definition = dissolve_workflow(
            [
                "output:clip/20260101-000000-abc/a.mp4",
                "output:clip/20260101-000000-abc/b.mp4",
            ],
            dissolve_frames=130,
        )

        token = activate_output_root(output_root)
        try:
            problems = dissolve_frame_errors(definition)
        finally:
            deactivate_output_root(token)

        assert len(problems) == 1
        assert "124 frames" in problems[0]["message"]

    def test_nothing_is_reported_for_a_definition_with_no_dissolve_step(self):
        assert dissolve_frame_errors({"steps": [{"name": "a", "task": {}}]}) == []

    def test_a_single_video_needs_no_dissolve(self, monkeypatch):
        base_dir = workflow_dir_with_asset(monkeypatch, ("a.mp4", 5))
        definition = dissolve_workflow(["asset:a.mp4"], dissolve_frames=130)

        assert dissolve_frame_errors(definition, base_dir=base_dir) == []

    def test_a_caller_supplied_probe_is_used_instead_of_the_default(self, monkeypatch):
        base_dir = workflow_dir_with_asset(monkeypatch, ("a.mp4", 124), ("b.mp4", 124))
        definition = dissolve_workflow(
            ["asset:a.mp4", "asset:b.mp4"], dissolve_frames=130
        )
        # A stub probe answering a frame count nothing on disk holds proves
        # the check reads the `probe` argument rather than always reaching
        # for probe_metadata itself.
        problems = dissolve_frame_errors(
            definition,
            base_dir=base_dir,
            probe=lambda path: {"kind": "video", "frame_count": 999},
        )

        assert problems == []

    def test_a_shared_cache_probes_each_file_once_across_two_calls(self, monkeypatch):
        # B9: a memoizing `probe` passed in by the caller (a per-validation
        # cache in a later task) must be genuinely consulted - two calls to
        # the check sharing one cache probe each distinct file only once,
        # not once per call.
        base_dir = workflow_dir_with_asset(monkeypatch, ("a.mp4", 124), ("b.mp4", 124))
        definition = dissolve_workflow(
            ["asset:a.mp4", "asset:b.mp4"], dissolve_frames=130
        )
        calls = []
        cache = {}

        def counting_cache(path):
            if path not in cache:
                calls.append(path)
                cache[path] = probe_metadata(path)
            return cache[path]

        dissolve_frame_errors(definition, base_dir=base_dir, probe=counting_cache)
        dissolve_frame_errors(definition, base_dir=base_dir, probe=counting_cache)

        assert len(calls) == 2


class TestTheValidationPass:
    def test_wired_into_validation_errors(self, monkeypatch):
        base_dir = workflow_dir_with_asset(monkeypatch, ("a.mp4", 124), ("b.mp4", 124))
        definition = dissolve_workflow(
            ["asset:a.mp4", "asset:b.mp4"], dissolve_frames=130
        )
        workflow = workflow_from_definition(
            definition, os.path.join(base_dir, "workflow.json")
        )

        problems = workflow.validation_errors()

        assert any(
            problem["path"] == "steps[0].task.arguments.dissolve_frames"
            for problem in problems
        )
