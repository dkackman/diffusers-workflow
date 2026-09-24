"""
Unit tests for shot boundaries (#385): where each input landed on a video a
step joined from several, recorded on the `AudioVideo` a join returns
(`dw/shots.py`), carried into the step's manifest entry and read back by
`dw.runs.recorded_shots`.

Covers: every `AudioVideo(` constructor site in `dw/` is accounted for by an
explicit decision (populates / carries / rescales / remeasures / none);
`concat_videos`, `dissolve_videos` and `run_chain` populate shots that
partition their output; a track measured longer than the frames imply is
recorded as measured, not derived; `pair_audio` re-measures the sample side
against a new track; `slice_audio` drops shots entirely (its output is audio,
which has no picture to partition); `rescaled_shots` is exercised directly for
`interpolate_frames`; and a join's shots survive a save/manifest/
`recorded_shots` round trip.
"""

import ast
import json
import os
from unittest.mock import patch

import numpy
from PIL import Image

from dw.pipeline_processors.chain import run_chain
from dw.result import AudioVideo, Result
from dw.runs import MANIFEST_FILE_NAME, recorded_shots
from dw.shots import (
    carried_shots,
    named_shots,
    remeasured_shots,
    rescaled_shots,
    shot_reference_names,
    shot_record,
    shots_for_file,
    step_shots,
)
from dw.tasks.audio_utils import slice_audio
from dw.tasks.concat_videos import concat_videos
from dw.tasks.dissolve_videos import dissolve_videos
from dw.tasks.pair_audio import pair_audio

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DW_ROOT = os.path.join(REPO_ROOT, "dw")


def frames(count, color=(0, 0, 0)):
    return [Image.new("RGB", (4, 4), color) for _ in range(count)]


def audio_video(num_frames, level, fps=4, sample_rate=100):
    samples = int(num_frames / fps * sample_rate)
    audio = numpy.full((2, samples), float(level), dtype=numpy.float32)
    return AudioVideo(frames(num_frames), audio, sample_rate, fps=fps)


# ---------------------------------------------------------------------------
# 1. Every AudioVideo(...) construction site in dw/ is accounted for
# ---------------------------------------------------------------------------

# (relative path, enclosing function name) -> (decision, expected call count)
#
# A decision is one of:
#   "populates" - builds a fresh shots list for a video it joined
#   "carries"   - copies an unchanged input's shots across (carried_shots)
#   "rescales"  - stretches an input's shots to a new frame count (rescaled_shots)
#   "remeasures" - keeps the frame side, re-measures the sample side (remeasured_shots)
#   "none"      - the video is not joined from named inputs; no shots kwarg at all
EXPECTED_SITES = {
    ("dw/tasks/concat_videos.py", "concat_videos"): ("populates", 1),
    ("dw/tasks/dissolve_videos.py", "dissolve_videos"): ("populates", 1),
    ("dw/pipeline_processors/chain.py", "run_chain"): ("populates", 2),
    ("dw/tasks/task.py", "_per_frame"): ("carries", 1),
    ("dw/tasks/stabilize.py", "stabilize_video"): ("carries", 1),
    ("dw/tasks/interpolate_frames.py", "interpolate_frames"): ("rescales", 1),
    ("dw/tasks/pair_audio.py", "pair_audio"): ("remeasures", 1),
    ("dw/tasks/video_utils.py", "_decode_audio_video"): ("none", 1),
    ("dw/result.py", "pair_audio_with_frames"): ("none", 1),
}

_SHOTS_HELPER_BY_DECISION = {
    "populates": None,  # builds its own list - no single shared helper
    "carries": "carried_shots",
    "rescales": "rescaled_shots",
    "remeasures": "remeasured_shots",
}


def _iter_python_files(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def _enclosing_function(tree, call_node):
    """The innermost function/method def that contains `call_node`, or None."""
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= call_node.lineno and getattr(
                node, "end_lineno", node.lineno
            ) >= getattr(call_node, "end_lineno", call_node.lineno):
                if best is None or node.lineno > best.lineno:
                    best = node
    return best.name if best else None


def _is_audio_video_call(call_node):
    func = call_node.func
    if isinstance(func, ast.Name):
        return func.id == "AudioVideo"
    if isinstance(func, ast.Attribute):
        return func.attr == "AudioVideo"
    return False


def _find_audio_video_calls():
    """Every `AudioVideo(...)` call in dw/, as (relative path, function, node)."""
    found = []
    for path in _iter_python_files(DW_ROOT):
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            continue
        relative = os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_audio_video_call(node):
                function_name = _enclosing_function(tree, node)
                found.append((relative, function_name, node))
    return found


def test_every_audio_video_constructor_site_is_decided():
    """Every `AudioVideo(` call site in dw/ is one of the sites this module
    decided for (dw/shots.py's own module docstring: "tests/test_shots.py
    fails on a constructor site nobody decided for.")."""
    calls = _find_audio_video_calls()

    counts = {}
    for relative, function_name, _node in calls:
        counts[(relative, function_name)] = counts.get((relative, function_name), 0) + 1

    unexpected = sorted(set(counts) - set(EXPECTED_SITES))
    assert not unexpected, (
        "AudioVideo(...) constructed at a site with no recorded decision - "
        "decide for it in tests/test_shots.py and dw/shots.py: "
        f"{unexpected}"
    )

    missing = sorted(set(EXPECTED_SITES) - set(counts))
    assert not missing, (
        "A previously-decided AudioVideo(...) site has disappeared - update "
        f"EXPECTED_SITES in tests/test_shots.py: {missing}"
    )

    for site, (_decision, expected_count) in EXPECTED_SITES.items():
        assert counts[site] == expected_count, (
            f"{site} now constructs AudioVideo {counts[site]} time(s), "
            f"expected {expected_count}"
        )


def test_every_non_none_site_passes_shots_explicitly():
    """A site that decided its video is (or is not) joined from named inputs
    says so with an explicit `shots=` keyword - "none" is the one decision
    that is allowed to omit it entirely."""
    calls = _find_audio_video_calls()
    for relative, function_name, node in calls:
        decision, _count = EXPECTED_SITES[(relative, function_name)]
        keyword_names = {kw.arg for kw in node.keywords if kw.arg is not None}
        if decision == "none":
            continue
        assert "shots" in keyword_names, (
            f"{relative}:{node.lineno} ({function_name}, decision={decision!r}) "
            "does not pass shots= explicitly"
        )


# ---------------------------------------------------------------------------
# 2 & 3. concat_videos populates shots that partition its output, and an
# overrun is measured rather than derived
# ---------------------------------------------------------------------------


class TestConcatVideosShots:
    def test_shots_partition_frames_and_samples(self):
        videos = [audio_video(4, 1), audio_video(6, 2), audio_video(5, 3)]

        result = concat_videos(videos, fps=4)

        assert [shot["name"] for shot in result.shots] == [
            "video 1",
            "video 2",
            "video 3",
        ]
        starts = [shot["start_frame"] for shot in result.shots]
        counts = [shot["num_frames"] for shot in result.shots]
        assert starts == [0, 4, 10]
        assert sum(counts) == len(result.frames)

        sample_starts = [shot["start_sample"] for shot in result.shots]
        sample_counts = [shot["num_samples"] for shot in result.shots]
        assert sample_starts[0] == 0
        assert sum(sample_counts) == result.audio.shape[1]

    def test_shot_reference_names_name_shot_at_members(self):
        videos = [audio_video(4, 1), audio_video(4, 2)]

        result = concat_videos(videos, fps=4)
        named = named_shots(
            result.shots,
            shot_reference_names(
                ["previous_result:shot@a.frames", "previous_result:shot@b.frames"]
            ),
        )

        # The prefix strips only "previous_result:" - the member keeps its
        # "shot@" marker, which is how a for_each member is named elsewhere
        assert [shot["name"] for shot in named] == ["shot@a", "shot@b"]

    def test_no_audio_input_leaves_sample_fields_none(self):
        result = concat_videos([frames(4), frames(3)])

        assert result.shots is not None
        for shot in result.shots:
            assert shot["start_sample"] is None
            assert shot["num_samples"] is None

    def test_an_overrun_track_is_measured_not_derived(self):
        """One input's audio runs 267 samples longer than its frames alone
        would imply - concat_videos records the measured length of the
        joined track, not a count derived from frame/fps arithmetic."""
        fps, sample_rate = 4, 100
        first = audio_video(4, 1, fps=fps, sample_rate=sample_rate)
        second = audio_video(4, 2, fps=fps, sample_rate=sample_rate)
        overrun = 267
        second.audio = numpy.concatenate(
            [
                second.audio,
                numpy.full((2, overrun), 2.0, dtype=numpy.float32),
            ],
            axis=1,
        )

        result = concat_videos([first, second], fps=fps)

        frame_derived_samples = int(4 / fps * sample_rate)
        second_shot = result.shots[1]
        assert second_shot["num_samples"] == frame_derived_samples + overrun
        assert second_shot["num_samples"] == result.audio.shape[1] - int(
            4 / fps * sample_rate
        )


# ---------------------------------------------------------------------------
# 4. dissolve_videos
# ---------------------------------------------------------------------------


class TestDissolveVideosShots:
    def test_shots_partition_frames_with_overlap_recorded(self):
        result = dissolve_videos([frames(10), frames(10), frames(10)], 3)

        starts = [shot["start_frame"] for shot in result.shots]
        counts = [shot["num_frames"] for shot in result.shots]
        assert sum(counts) == len(result.frames)
        assert starts[0] == 0
        assert starts[1] == starts[0] + counts[0]
        assert starts[2] == starts[1] + counts[1]

        assert "overlap_frames" not in result.shots[0]
        assert result.shots[1]["overlap_frames"] == 3
        assert result.shots[2]["overlap_frames"] == 3

    def test_sample_fields_partition_the_track_when_audio_is_present(self):
        def clip(level):
            samples = 250
            audio = numpy.full((2, samples), float(level), dtype=numpy.float32)
            return AudioVideo(frames(10), audio, 100, fps=4)

        result = dissolve_videos([clip(1), clip(2), clip(3)], 3, fps=4)

        sample_starts = [shot["start_sample"] for shot in result.shots]
        sample_counts = [shot["num_samples"] for shot in result.shots]
        assert sample_starts[0] == 0
        assert sum(sample_counts) == result.audio.shape[1]


# ---------------------------------------------------------------------------
# 5. run_chain
# ---------------------------------------------------------------------------


class _FakePipeline:
    def __init__(self, output_factory):
        self.output_factory = output_factory
        self.calls = []

    def _run_once(self, arguments):
        self.calls.append(arguments)
        return self.output_factory(arguments, len(self.calls) - 1)


def _video_output(arguments, index, num_frames=4):
    color = (50 * index % 256, 100, 150)
    made = frames(num_frames, color)
    if "image" in arguments:
        made[0] = arguments["image"]
    from types import SimpleNamespace

    return SimpleNamespace(frames=[made])


class TestRunChainShots:
    def test_segments_are_named_and_partition_the_output(self):
        pipeline = _FakePipeline(_video_output)

        result = run_chain(pipeline, {"segments": 3, "trim_frames": 1}, {})

        assert [shot["name"] for shot in result.shots] == [
            "segment 1",
            "segment 2",
            "segment 3",
        ]
        counts = [shot["num_frames"] for shot in result.shots]
        starts = [shot["start_frame"] for shot in result.shots]
        assert counts == [4, 3, 3]
        assert starts == [0, 4, 7]
        assert sum(counts) == len(result.frames)

    def test_video_only_chain_has_no_sample_fields(self):
        pipeline = _FakePipeline(_video_output)

        result = run_chain(pipeline, {"segments": 2}, {})

        for shot in result.shots:
            assert shot["start_sample"] is None
            assert shot["num_samples"] is None


# ---------------------------------------------------------------------------
# 6. pair_audio recomputes the sample fields
# ---------------------------------------------------------------------------


class TestPairAudioShots:
    def test_recomputes_sample_fields_against_the_new_track(self):
        fps, rate = 4, 100
        shots = [
            shot_record("a", 0, 4, start_sample=999, num_samples=999),
            shot_record("b", 4, 4, start_sample=999, num_samples=999),
        ]
        video = AudioVideo(frames(8), None, None, fps=fps, shots=shots)
        new_track = numpy.zeros((2, 250), dtype=numpy.float32)

        paired = pair_audio(video, new_track, sample_rate=rate)

        # Frame side is untouched
        assert [shot["start_frame"] for shot in paired.shots] == [0, 4]
        assert [shot["num_frames"] for shot in paired.shots] == [4, 4]

        assert paired.shots[0]["start_sample"] == round(0 / fps * rate)
        assert paired.shots[1]["start_sample"] == round(4 / fps * rate)
        # The last shot ends at the waveform's own length
        last = paired.shots[-1]
        assert last["start_sample"] + last["num_samples"] == new_track.shape[1]
        assert sum(shot["num_samples"] for shot in paired.shots) == new_track.shape[1]

    def test_no_frame_rate_clears_the_sample_side(self):
        shots = [shot_record("a", 0, 4, start_sample=1, num_samples=2)]
        video = AudioVideo(frames(4), None, None, fps=None, shots=shots)
        new_track = numpy.zeros((2, 100), dtype=numpy.float32)

        paired = pair_audio(video, new_track, sample_rate=100)

        assert paired.shots[0]["start_sample"] is None
        assert paired.shots[0]["num_samples"] is None

    def test_remeasured_shots_directly(self):
        """dw.shots.remeasured_shots in isolation, the function pair_audio calls."""
        shots = [shot_record("a", 0, 5), shot_record("b", 5, 5)]

        remeasured = remeasured_shots(shots, fps=5, sample_rate=10, total_samples=20)

        assert remeasured[0]["start_sample"] == 0
        assert remeasured[1]["start_sample"] == 10
        assert remeasured[1]["num_samples"] == 10

        assert remeasured_shots(shots, fps=None, sample_rate=10, total_samples=20) == [
            {**shot, "start_sample": None, "num_samples": None} for shot in shots
        ]
        assert remeasured_shots(None, fps=5, sample_rate=10, total_samples=20) is None


# ---------------------------------------------------------------------------
# 7. slice_audio drops shots
# ---------------------------------------------------------------------------


class TestSliceAudioDropsShots:
    def test_slicing_a_joined_video_returns_a_track_with_no_shots(self):
        shots = [shot_record("a", 0, 4, 0, 100), shot_record("b", 4, 4, 100, 100)]
        video = AudioVideo(
            frames(8),
            numpy.zeros((2, 200), dtype=numpy.float32),
            100,
            fps=4,
            shots=shots,
        )

        sliced = slice_audio(video, start_frame=0, num_frames=4, fps=4)

        assert getattr(sliced, "shots", None) is None


# ---------------------------------------------------------------------------
# 8. interpolate_frames / rescaled_shots
# ---------------------------------------------------------------------------


class TestRescaledShots:
    """interpolate_frames calls rescaled_shots(source_shots, multiplier)
    directly (dw/tasks/interpolate_frames.py); driving it through the real
    task needs a loaded RIFE model, so the function is exercised here."""

    def test_frame_counts_and_partition_after_doubling(self):
        shots = [shot_record("a", 0, 5), shot_record("b", 5, 5)]

        rescaled = rescaled_shots(shots, multiplier=2)

        # (10 - 1) * 2 + 1 = 19 total frames
        assert rescaled[0]["start_frame"] == 0
        assert rescaled[1]["start_frame"] == 10
        last_end = rescaled[1]["start_frame"] + rescaled[1]["num_frames"]
        assert last_end == 19

    def test_sample_fields_are_cleared(self):
        shots = [shot_record("a", 0, 5, start_sample=0, num_samples=50)]

        rescaled = rescaled_shots(shots, multiplier=2)

        assert rescaled[0]["start_sample"] is None
        assert rescaled[0]["num_samples"] is None

    def test_overlap_frames_scales_with_the_multiplier(self):
        shots = [
            shot_record("a", 0, 5),
            {**shot_record("b", 5, 5), "overlap_frames": 3},
        ]

        rescaled = rescaled_shots(shots, multiplier=2)

        assert rescaled[1]["overlap_frames"] == 6

    def test_empty_input_returns_none(self):
        assert rescaled_shots(None, multiplier=2) is None
        assert rescaled_shots([], multiplier=2) is None


class TestCarriedShots:
    def test_deep_copies_so_the_source_is_unaffected(self):
        source = AudioVideo(
            frames(4), None, None, shots=[shot_record("a", 0, 4, 0, 10)]
        )

        copied = carried_shots(source)
        copied[0]["num_frames"] = 999

        assert source.shots[0]["num_frames"] == 4

    def test_no_shots_returns_none(self):
        source = AudioVideo(frames(4), None, None)

        assert carried_shots(source) is None


# ---------------------------------------------------------------------------
# 9. Round trip: join -> Result.save -> manifest -> recorded_shots
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Exercises the same functions Workflow.run does (Result.save,
    dw.shots.step_shots, and dw.runs.recorded_shots) with a real join's
    output, without spinning up a full model-backed Workflow.run - the
    workflow's own step loop (dw/workflow.py) glues these together with no
    logic of its own beyond what is called here."""

    def test_shots_survive_save_and_recorded_shots(self, tmp_path):
        videos = [audio_video(4, 1), audio_video(6, 2)]
        joined = concat_videos(videos, fps=4)

        result = Result({"content_type": "video/mp4", "save": True})
        result.add_result(joined)

        run_dir = tmp_path / "concat-demo" / "20260923-120000-abcdef01"
        run_dir.mkdir(parents=True)
        # The actual mux is exercised by dw's own result-saving tests
        # (tests/test_concat_videos.py's TestPreviousResultChainAudioFit
        # patches the same three names); what this test pins is what
        # Result.save records in saved_shots and how the manifest step reads
        # it back, not the codec.
        with (
            patch("dw.result.encode_video"),
            patch("dw.result.export_to_video"),
            patch("dw.result.is_av_available", return_value=True),
        ):
            saved_files = result.save(str(run_dir), "concat-demo-join.0")

        assert saved_files
        assert result.saved_shots

        relative_files = [os.path.relpath(path, run_dir) for path in saved_files]
        manifest_shots = step_shots(result.saved_shots, saved_files)
        assert manifest_shots is not None
        assert [shot["name"] for shot in manifest_shots] == ["video 1", "video 2"]
        assert sum(shot["num_frames"] for shot in manifest_shots) == len(joined.frames)

        manifest = {
            "steps": [
                {
                    "step": "join",
                    "files": relative_files,
                    "shots": manifest_shots,
                }
            ]
        }
        with open(run_dir / MANIFEST_FILE_NAME, "w") as handle:
            json.dump(manifest, handle)

        relative_path = f"concat-demo/20260923-120000-abcdef01/{relative_files[0]}"
        read_back = recorded_shots(str(tmp_path), relative_path)

        assert read_back == manifest_shots

    def test_recorded_shots_is_none_outside_a_run_directory(self, tmp_path):
        # The flat layout - no run id segment - has no manifest to read shots from
        assert recorded_shots(str(tmp_path), "workflow/still.png") is None

    def test_recorded_shots_is_none_when_the_step_was_reused(self, tmp_path):
        run_dir = tmp_path / "wf" / "20260923-120000-abcdef01"
        run_dir.mkdir(parents=True)
        manifest = {
            "steps": [
                {
                    "step": "join",
                    "files": ["out.mp4"],
                    "shots": [shot_record("a", 0, 4, 0, 10)],
                    "reused": True,
                }
            ]
        }
        with open(run_dir / MANIFEST_FILE_NAME, "w") as handle:
            json.dump(manifest, handle)

        assert (
            recorded_shots(str(tmp_path), "wf/20260923-120000-abcdef01/out.mp4") is None
        )


# ---------------------------------------------------------------------------
# 10. shots_for_file with multi-file entries
# ---------------------------------------------------------------------------


class TestShotsForFile:
    def test_multi_file_entry_returns_only_that_files_shots_without_file_key(self):
        shots = [
            {**shot_record("a", 0, 4, 0, 10), "file": "one.mp4"},
            {**shot_record("b", 0, 5, 0, 12), "file": "two.mp4"},
        ]

        own = shots_for_file(shots, "two.mp4", ["one.mp4", "two.mp4"])

        assert len(own) == 1
        assert own[0]["name"] == "b"
        assert "file" not in own[0]

    def test_single_file_entry_matches_by_the_step_files_list(self):
        shots = [shot_record("a", 0, 4, 0, 10)]

        assert shots_for_file(shots, "solo.mp4", ["solo.mp4"]) == shots
        assert shots_for_file(shots, "other.mp4", ["solo.mp4"]) is None

    def test_no_shots_returns_none(self):
        assert shots_for_file(None, "solo.mp4", ["solo.mp4"]) is None
        assert shots_for_file([], "solo.mp4", ["solo.mp4"]) is None
