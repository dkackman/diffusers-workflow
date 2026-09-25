"""Tests for the assessment probes (dw/tasks/assess.py, #387).

Every probe call here is the real function - no mocking - against synthetic
media built in-process (PIL frames + numpy sine tones) or real mp4 files
written with PyAV, following the fixture style of tests/test_shots.py and
tests/test_media_info.py.
"""

import json

import numpy
import pytest
from PIL import Image

from dw.assessment_rules import RULES_BY_NAME
from dw.result import AudioVideo
from dw.runs import MANIFEST_FILE_NAME, shots_beside
from dw.shots import shot_record
from dw.tasks.assess import (
    THUMB_HEIGHT,
    THUMB_WIDTH,
    analyze_seams,
    analyze_shots,
    analyze_sync_drift,
    read_media,
    resolve_shots,
)
from dw.tasks.dissolve_videos import dissolve_videos


# ---------------------------------------------------------------------------
# Synthetic media helpers
# ---------------------------------------------------------------------------


def make_frames(num_frames, base_grey, noise=2.0, size=(64, 64), seed=0):
    """`num_frames` RGB frames around one grey level, with per-pixel noise so
    consecutive frames have a nonzero typical delta."""
    rng = numpy.random.default_rng(seed)
    width, height = size
    frames = []
    for _ in range(num_frames):
        arr = base_grey + rng.normal(0, noise, size=(height, width, 3))
        arr = numpy.clip(arr, 0, 255).astype(numpy.uint8)
        frames.append(Image.fromarray(arr, mode="RGB"))
    return frames


def make_tone(num_samples, sample_rate=48000, freq=440.0, amplitude=0.2, channels=2):
    t = numpy.arange(num_samples) / sample_rate
    tone = (amplitude * numpy.sin(2 * numpy.pi * freq * t)).astype(numpy.float32)
    return numpy.tile(tone, (channels, 1))


def write_mp4(
    path, frames=48, fps=24, width=64, height=64, sample_rate=48000, seconds=None
):
    """A real mp4 with a moving picture and a sine soundtrack, for the tests
    that exercise read_media on an actual encoded file. Mirrors
    tests/test_media_info.py's write_mp4 helper, with a picture that moves
    frame to frame instead of staying flat."""
    import av

    container = av.open(str(path), "w")
    video = container.add_stream("libx264", rate=fps)
    video.width, video.height, video.pix_fmt = width, height, "yuv420p"
    audio = container.add_stream("aac", rate=sample_rate)
    audio.layout = "stereo"

    rng = numpy.random.default_rng(1)
    for _ in range(frames):
        arr = rng.integers(0, 255, size=(height, width, 3), dtype=numpy.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in video.encode(frame):
            container.mux(packet)

    total_samples = (
        seconds and int(seconds * sample_rate) or int(sample_rate * frames / fps)
    )
    tone = make_tone(total_samples, sample_rate=sample_rate, amplitude=0.2)
    for start in range(0, total_samples, 1024):
        chunk = av.AudioFrame.from_ndarray(
            numpy.ascontiguousarray(tone[:, start : start + 1024]),
            format="fltp",
            layout="stereo",
        )
        chunk.sample_rate = sample_rate
        chunk.pts = start
        for packet in audio.encode(chunk):
            container.mux(packet)
    for packet in audio.encode():
        container.mux(packet)
    for packet in video.encode():
        container.mux(packet)
    container.close()


def cut_shots(names, frames_per_shot, samples_per_shot):
    """Shot records for a hard-cut concatenation: contiguous, no overlap."""
    shots = []
    for index, name in enumerate(names):
        shots.append(
            shot_record(
                name,
                index * frames_per_shot,
                frames_per_shot,
                index * samples_per_shot,
                samples_per_shot,
            )
        )
    return shots


# ---------------------------------------------------------------------------
# 1. A 6 dB step flags seam_level_step at one seam only, and shot_level_spread
# ---------------------------------------------------------------------------


class TestLevelStep:
    def test_step_flags_one_seam_and_shot_spread(self):
        fps = 24
        sample_rate = 48000
        frames_per_shot = 48
        samples_per_shot = sample_rate * frames_per_shot // fps

        names = ["s0", "s1", "s2"]
        amplitudes = [0.2, 0.2, 0.5]  # s2 is ~8 dB louder

        frames = []
        tracks = []
        for index, amp in enumerate(amplitudes):
            frames.extend(
                make_frames(frames_per_shot, base_grey=120, noise=3.0, seed=index)
            )
            tracks.append(make_tone(samples_per_shot, sample_rate, amplitude=amp))
        audio = numpy.concatenate(tracks, axis=1)

        shots = cut_shots(names, frames_per_shot, samples_per_shot)
        video = AudioVideo(frames, audio, sample_rate, fps=fps, shots=shots)

        seams = analyze_seams(video)
        assert seams["shots_source"] == "artifact"
        step_findings = [f for f in seams["findings"] if f["rule"] == "seam_level_step"]
        assert len(step_findings) == 1
        assert step_findings[0]["at"]["seam"] == 2

        seam1_rules = {f["rule"] for f in seams["findings"] if f["at"]["seam"] == 1}
        assert "seam_level_step" not in seam1_rules

        shots_answer = analyze_shots(video)
        assert shots_answer["has_audio"] is True
        spread_findings = [
            f for f in shots_answer["findings"] if f["rule"] == "shot_level_spread"
        ]
        assert len(spread_findings) == 1
        assert shots_answer["rms_range_db"] >= 6.0

        # JSON-serializable, per requirement 10
        json.dumps(seams)
        json.dumps(shots_answer)

    @staticmethod
    def _one_clip_three_times(duck_db=0.0):
        """One take cut after itself three times, as C-F099 builds it: the
        take trails off (-46 dBFS tail) and opens near silence (-66 dBFS
        head) around a voiced body, so its own edges sit 20 dB apart. The
        third shot is optionally ducked by `duck_db`."""
        fps, sample_rate, frames_per_shot = 24, 48000, 48
        samples_per_shot = sample_rate * frames_per_shot // fps
        edge = sample_rate // 4
        take = make_tone(samples_per_shot, sample_rate, amplitude=0.2)
        take[:, :edge] *= 10 ** (-66 / 20) / (0.2 / numpy.sqrt(2))
        take[:, -edge:] *= 10 ** (-46 / 20) / (0.2 / numpy.sqrt(2))
        ducked = take * 10 ** (-duck_db / 20)
        audio = numpy.concatenate([take, take, ducked], axis=1)
        frames = []
        for index in range(3):
            frames.extend(make_frames(frames_per_shot, 120, noise=3.0, seed=index))
        shots = cut_shots(["a", "b", "c"], frames_per_shot, samples_per_shot)
        return AudioVideo(frames, audio, sample_rate, fps=fps, shots=shots)

    def test_a_take_with_quiet_edges_cut_after_itself_is_clean(self):
        seams = analyze_seams(self._one_clip_three_times())
        assert all(seam["before_rms_dbfs"] < -40 for seam in seams["seams"])
        assert all(seam["level_step_db"] == 0.0 for seam in seams["seams"])
        assert not [f for f in seams["findings"] if f["rule"] == "seam_level_step"]

    def test_a_duck_reports_its_own_size_at_its_seam_only(self):
        seams = analyze_seams(self._one_clip_three_times(duck_db=12.0))
        steps = [f for f in seams["findings"] if f["rule"] == "seam_level_step"]
        assert [f["at"]["seam"] for f in steps] == [2]
        assert steps[0]["value"] == pytest.approx(12.0, abs=0.1)

    def test_shots_echo_their_frame_range(self):
        answer = analyze_shots(self._one_clip_three_times())
        assert [(s["start_frame"], s["num_frames"]) for s in answer["shots"]] == [
            (0, 48),
            (48, 48),
            (96, 48),
        ]


# ---------------------------------------------------------------------------
# 2. A dissolve does not flag a level step, click, hole or frame jump
# ---------------------------------------------------------------------------


class TestDissolveDoesNotFlag:
    def test_dissolve_seams_are_clean(self):
        fps = 24
        sample_rate = 48000
        frames_per_clip = 48
        samples_per_clip = sample_rate * frames_per_clip // fps

        clips = []
        for index, base in enumerate((100, 130, 160)):
            clip_frames = make_frames(
                frames_per_clip, base_grey=base, noise=3.0, seed=10 + index
            )
            audio = make_tone(samples_per_clip, sample_rate, amplitude=0.3)
            clips.append(AudioVideo(clip_frames, audio, sample_rate, fps=fps))

        joined = dissolve_videos(clips, dissolve_frames=12, fps=fps)
        assert joined.shots is not None

        seams = analyze_seams(joined)
        assert seams["shots_source"] == "artifact"
        assert len(seams["seams"]) == 2
        for seam in seams["seams"]:
            assert seam["kind"] == "dissolve"

        offending = [
            f
            for f in seams["findings"]
            if f["rule"] in ("seam_level_step", "seam_click", "seam_hole")
        ]
        assert offending == []
        assert seams["findings"] == [], seams["findings"]

        json.dumps(seams)


# ---------------------------------------------------------------------------
# 3. hard_cut suppresses seam_frame_jump; the same media without it fires
# ---------------------------------------------------------------------------


class TestHardCut:
    def _build(self, hard_cut):
        fps = 24
        frames_per_shot = 24
        frames = make_frames(frames_per_shot, base_grey=0.3 * 255, noise=2.0, seed=1)
        frames += make_frames(frames_per_shot, base_grey=0.8 * 255, noise=2.0, seed=2)
        video = AudioVideo(frames, None, None, fps=fps)
        extra = {"hard_cut": True} if hard_cut else {}
        shots = [
            shot_record("s0", 0, frames_per_shot),
            shot_record("s1", frames_per_shot, frames_per_shot, **extra),
        ]
        return video, shots

    def test_hard_cut_suppresses_the_finding(self):
        video, shots = self._build(hard_cut=True)
        answer = analyze_seams(video, shots=shots)
        assert answer["shots_source"] == "argument"
        assert answer["seams"][0]["hard_cut"] is True
        jump_findings = [
            f for f in answer["findings"] if f["rule"] == "seam_frame_jump"
        ]
        assert jump_findings == []

    def test_without_hard_cut_the_jump_fires(self):
        video, shots = self._build(hard_cut=False)
        answer = analyze_seams(video, shots=shots)
        assert answer["seams"][0]["hard_cut"] is False
        jump_findings = [
            f for f in answer["findings"] if f["rule"] == "seam_frame_jump"
        ]
        assert len(jump_findings) == 1
        assert jump_findings[0]["severity"] == "info"
        assert jump_findings[0]["at"]["seam"] == 1


# ---------------------------------------------------------------------------
# 4. A static shot into a modest cut does not flag seam_frame_jump
# ---------------------------------------------------------------------------


class TestStaticShotThenModestCut:
    def test_no_jump_for_a_modest_change_after_a_static_shot(self):
        fps = 24
        frames_per_shot = 24
        # shot 0: nearly static (tiny noise)
        frames = make_frames(frames_per_shot, base_grey=128, noise=0.3, seed=5)
        # shot 1: moderate, ordinary frame-to-frame motion
        frames += make_frames(frames_per_shot, base_grey=140, noise=5.0, seed=6)
        video = AudioVideo(frames, None, None, fps=fps)
        shots = [
            shot_record("s0", 0, frames_per_shot),
            shot_record("s1", frames_per_shot, frames_per_shot),
        ]

        answer = analyze_seams(video, shots=shots)
        seam = answer["seams"][0]
        assert seam["typical_delta"] is not None
        assert seam["jump_ratio"] is not None
        assert seam["jump_ratio"] <= RULES_BY_NAME["seam_frame_jump"]["threshold"]
        jump_findings = [
            f for f in answer["findings"] if f["rule"] == "seam_frame_jump"
        ]
        assert jump_findings == []


# ---------------------------------------------------------------------------
# 5. Drift accumulates and crosses sync_drift once it passes 40 ms
# ---------------------------------------------------------------------------


class TestSyncDrift:
    def test_drift_accumulates_and_flags(self):
        fps = 24
        sample_rate = 48000
        frames_per_shot = 48  # 2s
        nominal_samples = sample_rate * frames_per_shot // fps  # 96000
        overrun = 267
        num_shots = 9

        names = [f"s{i}" for i in range(num_shots)]
        start_sample = 0
        shots = []
        expected_offsets = []
        for i, name in enumerate(names):
            num_samples = nominal_samples + overrun
            shots.append(
                shot_record(
                    name,
                    i * frames_per_shot,
                    frames_per_shot,
                    start_sample,
                    num_samples,
                )
            )
            end_frame = (i + 1) * frames_per_shot
            offset_ms = (
                (start_sample + num_samples) / sample_rate - end_frame / fps
            ) * 1000.0
            expected_offsets.append(offset_ms)
            start_sample += num_samples

        total_frames = num_shots * frames_per_shot
        total_samples = start_sample
        frames = make_frames(total_frames, base_grey=100, noise=2.0, seed=7)
        audio = make_tone(total_samples, sample_rate, amplitude=0.2)
        video = AudioVideo(frames, audio, sample_rate, fps=fps)

        answer = analyze_sync_drift(video, shots=shots)
        assert answer["shots_source"] == "argument"
        measured_offsets = [shot["end_offset_ms"] for shot in answer["shots"]]

        # growing, per shot, and matching the arithmetic above
        assert measured_offsets == sorted(measured_offsets)
        for measured, expected in zip(measured_offsets, expected_offsets):
            assert measured == pytest.approx(expected, abs=0.01)

        assert answer["max_offset_ms"] == pytest.approx(expected_offsets[-1], abs=0.01)
        assert answer["max_offset_ms"] > 40.0

        drift_findings = [f for f in answer["findings"] if f["rule"] == "sync_drift"]
        assert drift_findings
        assert all(f["severity"] == "warn" for f in drift_findings)

        json.dumps(answer)

    def test_a_clean_cut_reports_no_findings(self):
        fps = 24
        sample_rate = 48000
        frames_per_shot = 48
        samples_per_shot = sample_rate * frames_per_shot // fps
        names = ["a", "b", "c"]

        shots = cut_shots(names, frames_per_shot, samples_per_shot)
        total_frames = len(names) * frames_per_shot
        total_samples = len(names) * samples_per_shot
        frames = make_frames(total_frames, base_grey=100, noise=2.0, seed=8)
        audio = make_tone(total_samples, sample_rate, amplitude=0.2)
        video = AudioVideo(frames, audio, sample_rate, fps=fps)

        answer = analyze_sync_drift(video, shots=shots)
        assert answer["findings"] == []
        for shot in answer["shots"]:
            assert shot["end_offset_ms"] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# 6. Memory bound: read_media never holds the full decoded frame list
# ---------------------------------------------------------------------------


class TestMemoryBound:
    def test_read_media_stays_far_below_the_full_frame_list(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
        import tracemalloc

        path = tmp_path / "big.mp4"
        frames, width, height = 240, 256, 256
        write_mp4(path, frames=frames, fps=24, width=width, height=height)

        tracemalloc.start()
        media = read_media(str(path))
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        full_frame_list_bytes = frames * height * width * 3
        assert peak < full_frame_list_bytes / 4

        assert media.thumbs is not None
        assert media.thumbs.shape[1] == THUMB_HEIGHT
        assert media.thumbs.shape[2] == THUMB_WIDTH
        assert media.thumbs.shape[0] == frames


# ---------------------------------------------------------------------------
# 7. read_media + probes on a real encoded file; AAC priming doesn't drift
# ---------------------------------------------------------------------------


class TestRealFile:
    def test_probes_on_an_encoded_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
        path = tmp_path / "cut.mp4"
        write_mp4(path, frames=48, fps=24, width=64, height=64, sample_rate=48000)

        media = read_media(str(path))
        assert media.thumbs is not None
        assert media.audio is not None

        shots_answer = analyze_shots(str(path))
        assert shots_answer["shots_source"] == "none"

        seams_answer = analyze_seams(str(path))
        assert seams_answer["shots_source"] == "none"
        assert seams_answer["seams"] == []

        drift_answer = analyze_sync_drift(str(path))
        assert drift_answer["shots_source"] == "none"
        assert drift_answer["length_delta_ms"] is not None
        assert abs(drift_answer["length_delta_ms"]) < 40.0

        json.dumps(shots_answer)
        json.dumps(seams_answer)
        json.dumps(drift_answer)

    def test_shotless_file_reports_skipped_rules_not_a_clean_pass(
        self, tmp_path, monkeypatch
    ):
        """#394: shots_source "none" used to report every rule as applied
        with findings: [] - a false clean, since no seam or shot spread was
        actually measured. It must say the rules were skipped instead, and
        warn that shots= would supply the missing boundaries."""
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
        import dw.tasks.assess as assess_module

        warnings = []
        monkeypatch.setattr(
            assess_module,
            "emit_warning",
            lambda message, **data: warnings.append((message, data)),
        )

        path = tmp_path / "two-shots.mp4"
        write_mp4(path, frames=48, fps=24, width=64, height=64, sample_rate=48000)

        seams_answer = analyze_seams(str(path))
        assert seams_answer["shots_source"] == "none"
        assert seams_answer["rules_applied"] == []
        assert seams_answer["rules_skipped"] == [
            {"rule": name, "reason": "no shot boundaries"}
            for name in (
                "seam_level_step",
                "seam_click",
                "seam_hole",
                "seam_frame_jump",
            )
        ]
        assert any("no shot boundaries" in w[0].lower() for w in warnings)
        assert any(w[1].get("kind") == "no_shot_boundaries" for w in warnings)
        warnings.clear()

        shots_answer = analyze_shots(str(path))
        assert shots_answer["shots_source"] == "none"
        assert "shot_level_spread" not in shots_answer["rules_applied"]
        assert shots_answer["rules_skipped"] == [
            {"rule": "shot_level_spread", "reason": "no shot boundaries"}
        ]
        assert any("no shot boundaries" in w[0].lower() for w in warnings)

        json.dumps(seams_answer)
        json.dumps(shots_answer)


# ---------------------------------------------------------------------------
# 8. shots_beside reads a manifest, and a probe reports shots_source manifest
# ---------------------------------------------------------------------------


class TestShotsBeside:
    def test_manifest_shots_are_read_back(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
        run_dir = tmp_path / "demo-workflow" / "20260923-120000-abcdef01"
        run_dir.mkdir(parents=True)
        media_path = run_dir / "final" / "cut.mp4"
        media_path.parent.mkdir(parents=True)
        write_mp4(media_path, frames=48, fps=24, width=64, height=64, sample_rate=48000)

        manifest_shots = [
            shot_record("a", 0, 24, 0, 48000),
            shot_record("b", 24, 24, 48000, 48000),
        ]
        own = "final/cut.mp4"
        manifest = {
            "steps": [{"step": "join", "files": [own], "shots": manifest_shots}]
        }
        with open(run_dir / MANIFEST_FILE_NAME, "w") as handle:
            json.dump(manifest, handle)

        read_back = shots_beside(str(media_path))
        assert read_back == manifest_shots

        answer = analyze_seams(str(media_path))
        assert answer["shots_source"] == "manifest"
        assert len(answer["seams"]) == 1


# ---------------------------------------------------------------------------
# 9. resolve_shots order: argument > artifact > manifest > none
# ---------------------------------------------------------------------------


class TestResolveShotsOrder:
    def _media(self, video):
        from dw.tasks.assess import media_from

        return media_from(video)

    def test_argument_wins_over_artifact(self):
        frames = make_frames(8, base_grey=100, noise=1.0, seed=20)
        artifact_shots = [shot_record("artifact", 0, 8)]
        video = AudioVideo(frames, None, None, fps=4, shots=artifact_shots)
        media = self._media(video)
        argument_shots = [shot_record("argument", 0, 8)]

        records, source = resolve_shots(video, media, shots=argument_shots)
        assert source == "argument"
        assert records[0]["name"] == "argument"

    def test_artifact_wins_over_manifest(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
        run_dir = tmp_path / "wf" / "20260923-120000-abcdef02"
        run_dir.mkdir(parents=True)
        media_path = run_dir / "cut.mp4"
        write_mp4(media_path, frames=8, fps=4, width=32, height=32, sample_rate=8000)
        manifest = {
            "steps": [
                {
                    "step": "join",
                    "files": ["cut.mp4"],
                    "shots": [shot_record("manifest", 0, 8)],
                }
            ]
        }
        with open(run_dir / MANIFEST_FILE_NAME, "w") as handle:
            json.dump(manifest, handle)

        from dw.tasks.assess import media_from

        media = media_from(str(media_path))
        media.shots = [shot_record("artifact", 0, 8)]

        records, source = resolve_shots(str(media_path), media, shots=None)
        assert source == "artifact"
        assert records[0]["name"] == "artifact"

    def test_manifest_wins_over_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
        run_dir = tmp_path / "wf" / "20260923-120000-abcdef03"
        run_dir.mkdir(parents=True)
        media_path = run_dir / "cut.mp4"
        write_mp4(media_path, frames=8, fps=4, width=32, height=32, sample_rate=8000)
        manifest = {
            "steps": [
                {
                    "step": "join",
                    "files": ["cut.mp4"],
                    "shots": [shot_record("manifest", 0, 8)],
                }
            ]
        }
        with open(run_dir / MANIFEST_FILE_NAME, "w") as handle:
            json.dump(manifest, handle)

        from dw.tasks.assess import media_from

        media = media_from(str(media_path))
        records, source = resolve_shots(str(media_path), media, shots=None)
        assert source == "manifest"
        assert records[0]["name"] == "manifest"

    def test_none_when_nothing_resolves(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
        media_path = tmp_path / "solo.mp4"
        write_mp4(media_path, frames=8, fps=4, width=32, height=32, sample_rate=8000)

        from dw.tasks.assess import media_from

        media = media_from(str(media_path))
        records, source = resolve_shots(str(media_path), media, shots=None)
        assert records is None
        assert source == "none"


# ---------------------------------------------------------------------------
# 10. Answers are JSON-serializable and carry the common fields
# ---------------------------------------------------------------------------


class TestAnswerShape:
    def test_answers_carry_the_common_fields(self):
        fps = 24
        sample_rate = 48000
        frames_per_shot = 24
        samples_per_shot = sample_rate * frames_per_shot // fps
        shots = cut_shots(["a", "b"], frames_per_shot, samples_per_shot)
        frames = make_frames(frames_per_shot * 2, base_grey=100, noise=2.0, seed=30)
        audio = make_tone(samples_per_shot * 2, sample_rate, amplitude=0.2)
        video = AudioVideo(frames, audio, sample_rate, fps=fps, shots=shots)

        for answer in (
            analyze_shots(video),
            analyze_seams(video),
            analyze_sync_drift(video),
        ):
            json.dumps(answer)
            assert "findings" in answer
            assert "rules_applied" in answer
            assert "shots_source" in answer
            assert answer["shots_source"] == "artifact"


# ---------------------------------------------------------------------------
# 10. A probe step on a stored video: asset: and output: stream the file (#387)
# ---------------------------------------------------------------------------


class TestStoredMediaInAWorkflow:
    """A probe's 'video' naming a stored file used to be decoded to a frame
    list first - dropping the soundtrack - and every probe then refused it
    ("not FrameList"). It is now read by reference, like get_frame's."""

    def probe_workflow(self, video, command="analyze_shots"):
        return {
            "id": "probe-stored",
            "steps": [
                {
                    "name": "probe",
                    "task": {"command": command, "arguments": {"video": video}},
                    "result": {"content_type": "application/json"},
                }
            ],
        }

    def run(self, definition, output_dir):
        from dw.workflow import Workflow

        results = Workflow(definition, str(output_dir), "").run({})
        answers = list(results.values()) if isinstance(results, dict) else results
        return answers

    def saved_answer(self, output_dir):
        saved = [
            p
            for p in output_dir.rglob("*.json")
            if p.name not in ("manifest.json", "workflow.json")
        ]
        assert len(saved) == 1, saved
        return json.loads(saved[0].read_text())

    def test_an_asset_video_is_probed_from_the_file(self, tmp_path, monkeypatch):
        import dw.arguments as arguments_module

        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
        assets = tmp_path / "assets" / "cast"
        assets.mkdir(parents=True)
        write_mp4(assets / "clip.mp4", frames=48, fps=24)
        monkeypatch.setenv("DW_ASSET_DIR", str(tmp_path / "assets"))

        def _boom(*args, **kwargs):
            raise AssertionError("a probe's video must not be decoded eagerly")

        monkeypatch.setattr(arguments_module, "load_video", _boom)

        outputs = tmp_path / "outputs"
        self.run(self.probe_workflow("asset:cast/clip.mp4"), outputs)

        answer = self.saved_answer(outputs)
        assert answer["shots_source"] == "none"
        assert len(answer["shots"]) == 1
        # The soundtrack was read, not dropped with a frame-list decode
        assert answer["shots"][0]["rms_dbfs"] is not None

    def test_an_output_video_is_probed_with_its_manifest_shots(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
        outputs = tmp_path / "outputs"
        run_dir = outputs / "cutter" / "20260923-120000-abcdef01"
        (run_dir / "final").mkdir(parents=True)
        write_mp4(run_dir / "final" / "cut.mp4", frames=48, fps=24)
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "steps": [
                        {
                            "step": "cut",
                            "files": ["final/cut.mp4"],
                            "shots": [
                                shot_record("shot@a", 0, 24, 0, 48000),
                                shot_record("shot@b", 24, 24, 48000, 48000),
                            ],
                        }
                    ],
                }
            )
        )

        self.run(
            self.probe_workflow(
                "output:cutter/20260923-120000-abcdef01/final/cut.mp4",
                command="analyze_seams",
            ),
            outputs,
        )

        answer = self.saved_answer(outputs / "probe-stored")
        assert answer["shots_source"] == "manifest"
        assert len(answer["seams"]) == 1


# ---------------------------------------------------------------------------
# 8. A shots= record reaching past the file's end is warned, not silently
# clipped (#425)
# ---------------------------------------------------------------------------


class TestShotSpanOverrun:
    def _overrunning_shots(self):
        # Mirrors the issue's own repro: a 248-frame video, second shot
        # declared 124..424 - 176 frames past the real end.
        return [
            shot_record("a", 0, 124, 0, 124 * 2000),
            shot_record("b", 124, 300, 124 * 2000, 300 * 2000, hard_cut=True),
        ]

    def _video(self):
        frames = make_frames(248, base_grey=120, noise=3.0)
        audio = make_tone(248 * 2000, sample_rate=248 * 2000 * 24 // 248)
        return AudioVideo(frames, audio, 2000 * 24, fps=24)

    def _capture(self, monkeypatch):
        import dw.tasks.assess as assess_module

        warnings = []
        monkeypatch.setattr(
            assess_module,
            "emit_warning",
            lambda message, **data: warnings.append((message, data)),
        )
        return warnings

    def test_analyze_seams_warns_and_reports_a_finding(self, monkeypatch):
        warnings = self._capture(monkeypatch)
        answer = analyze_seams(self._video(), shots=self._overrunning_shots())

        overrun = [f for f in answer["findings"] if f["rule"] == "shot_span_overrun"]
        assert len(overrun) == 1
        assert overrun[0]["at"] == {"shot": "b"}
        assert overrun[0]["value"]["frames"] == 176

        assert any(w[1].get("kind") == "shot_span_overrun" for w in warnings)
        assert any(w[1].get("shots") == ["b"] for w in warnings)
        json.dumps(answer)

    def test_analyze_shots_warns_and_reports_a_finding(self, monkeypatch):
        warnings = self._capture(monkeypatch)
        answer = analyze_shots(self._video(), shots=self._overrunning_shots())

        overrun = [f for f in answer["findings"] if f["rule"] == "shot_span_overrun"]
        assert len(overrun) == 1
        assert overrun[0]["at"] == {"shot": "b"}
        assert any(w[1].get("kind") == "shot_span_overrun" for w in warnings)
        json.dumps(answer)

    def test_analyze_sync_drift_warns_and_reports_a_finding(self, monkeypatch):
        warnings = self._capture(monkeypatch)
        answer = analyze_sync_drift(self._video(), shots=self._overrunning_shots())

        overrun = [f for f in answer["findings"] if f["rule"] == "shot_span_overrun"]
        assert len(overrun) == 1
        assert overrun[0]["at"] == {"shot": "b"}
        assert any(w[1].get("kind") == "shot_span_overrun" for w in warnings)
        json.dumps(answer)

    def test_shots_that_fit_report_no_overrun_finding(self, monkeypatch):
        warnings = self._capture(monkeypatch)
        fitting = [
            shot_record("a", 0, 124, 0, 124 * 2000),
            shot_record("b", 124, 124, 124 * 2000, 124 * 2000, hard_cut=True),
        ]

        answer = analyze_seams(self._video(), shots=fitting)

        assert not [f for f in answer["findings"] if f["rule"] == "shot_span_overrun"]
        assert not any(w[1].get("kind") == "shot_span_overrun" for w in warnings)
