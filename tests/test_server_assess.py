"""Tests for GET /api/gallery/{name}/assess (#388).

The route runs the real probes over real mp4 files written with PyAV - no
mocked probe - with the shot boundaries in a run manifest (an output) or in
the sidecar keep_output writes (an asset), the two places the route reads
them from.
"""

import json

import numpy
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dw.runs import MANIFEST_FILE_NAME, record_kept_shots
from dw.server.app import create_app
from dw.server.jobs import JobManager
from dw.shots import shot_record
from tests.test_server import (
    ScriptedWorkerManager,
    hanging_script,
    success_script,
    valid_workflow,
    wait_for_status,
)

FPS = 24
SAMPLE_RATE = 48000
FRAMES_PER_SHOT = 24
SAMPLES_PER_SHOT = SAMPLE_RATE * FRAMES_PER_SHOT // FPS
RUN = "cut/20260923-120000-abcdef01"
CUT = f"{RUN}/final/cut.mp4"


def write_cut(path, amplitudes):
    """A real mp4 of len(amplitudes) shots: one grey level and one tone
    level per shot, hard cuts between them."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), "w")
    video = container.add_stream("libx264", rate=FPS)
    video.width, video.height, video.pix_fmt = 64, 64, "yuv420p"
    audio = container.add_stream("aac", rate=SAMPLE_RATE)
    audio.layout = "stereo"

    rng = numpy.random.default_rng(0)
    for shot in range(len(amplitudes)):
        for _ in range(FRAMES_PER_SHOT):
            grey = 60 + 60 * shot + rng.normal(0, 3, size=(64, 64, 3))
            arr = numpy.clip(grey, 0, 255).astype(numpy.uint8)
            for packet in video.encode(av.VideoFrame.from_ndarray(arr, "rgb24")):
                container.mux(packet)

    t = numpy.arange(SAMPLES_PER_SHOT) / SAMPLE_RATE
    tone = numpy.sin(2 * numpy.pi * 440.0 * t)
    track = numpy.concatenate([amp * tone for amp in amplitudes]).astype(numpy.float32)
    track = numpy.tile(track, (2, 1))
    for start in range(0, track.shape[1], 1024):
        chunk = av.AudioFrame.from_ndarray(
            numpy.ascontiguousarray(track[:, start : start + 1024]),
            format="fltp",
            layout="stereo",
        )
        chunk.sample_rate = SAMPLE_RATE
        chunk.pts = start
        for packet in audio.encode(chunk):
            container.mux(packet)
    for packet in audio.encode():
        container.mux(packet)
    for packet in video.encode():
        container.mux(packet)
    container.close()


def shots_of(count):
    return [
        shot_record(
            f"s{index}",
            index * FRAMES_PER_SHOT,
            FRAMES_PER_SHOT,
            index * SAMPLES_PER_SHOT,
            SAMPLES_PER_SHOT,
        )
        for index in range(count)
    ]


def write_run(outputs, amplitudes=(0.1, 0.1, 0.4)):
    """A stepped cut in a run directory, its shots in the run's manifest."""
    run_dir = outputs / RUN
    write_cut(run_dir / "final" / "cut.mp4", amplitudes)
    manifest = {
        "steps": [
            {
                "step": "join",
                "files": ["final/cut.mp4"],
                "shots": shots_of(len(amplitudes)),
            }
        ]
    }
    (run_dir / MANIFEST_FILE_NAME).write_text(json.dumps(manifest))


@pytest.fixture
def server(tmp_path):
    workflow_dir = tmp_path / "workflows"
    workflow_dir.mkdir()
    (tmp_path / "assets").mkdir()

    def make(script=success_script):
        manager = JobManager(
            str(tmp_path / "outputs"),
            worker_manager=ScriptedWorkerManager(script),
            history_path=str(tmp_path / "jobs.sqlite"),
        )
        app = create_app(
            workflow_dir=str(workflow_dir),
            output_dir=str(tmp_path / "outputs"),
            job_manager=manager,
            asset_dir=str(tmp_path / "assets"),
        )
        return TestClient(app, base_url="http://localhost")

    return make


def test_a_stepped_cut_answers_the_seam_finding(server, tmp_path):
    with server() as client:
        write_run(tmp_path / "outputs")
        answer = client.get(f"/api/gallery/{CUT}/assess")

    assert answer.status_code == 200
    body = answer.json()
    assert body["name"] == CUT
    assert body["source"] == "output"
    assert body["kind"] == "video"
    assert body["shots_source"] == "manifest"
    steps = [f for f in body["findings"] if f["rule"] == "seam_level_step"]
    assert [f["at"]["seam"] for f in steps] == [2]
    for found in body["findings"]:
        assert set(found) == {"rule", "severity", "at", "value", "threshold", "says"}
    assert {"seam_level_step", "shot_level_spread", "sync_drift"} <= set(
        body["rules_applied"]
    )
    assert body["not_applicable"] == {}
    assert "probes" not in body


def test_a_hot_cut_raises_full_scale_naming_relevel_clip(server, tmp_path):
    with server() as client:
        write_run(tmp_path / "outputs", amplitudes=(1.0, 1.2, 1.0))
        body = client.get(f"/api/gallery/{CUT}/assess").json()

    hot = [f for f in body["findings"] if f["rule"] == "full_scale"]
    assert len(hot) == 1
    assert "relevel-clip" in hot[0]["says"]
    assert {"full_scale", "near_silent"} <= set(body["rules_applied"])


def test_a_cut_below_full_scale_applies_the_rule_without_a_finding(server, tmp_path):
    with server() as client:
        write_run(tmp_path / "outputs")
        body = client.get(f"/api/gallery/{CUT}/assess").json()

    assert "full_scale" in body["rules_applied"]
    assert not [f for f in body["findings"] if f["rule"] == "full_scale"]


def test_detail_adds_every_probes_full_body(server, tmp_path):
    with server() as client:
        write_run(tmp_path / "outputs")
        body = client.get(f"/api/gallery/{CUT}/assess?detail=true").json()

    assert set(body["probes"]) == {
        "analyze_shots",
        "analyze_seams",
        "analyze_sync_drift",
        "analyze_held_frames",
    }
    assert len(body["probes"]["analyze_seams"]["seams"]) == 2


def test_one_probe_answers_its_full_body(server, tmp_path):
    with server() as client:
        write_run(tmp_path / "outputs")
        body = client.get(f"/api/gallery/{CUT}/assess?probe=analyze_seams").json()

    assert body["probe"] == "analyze_seams"
    assert len(body["seams"]) == 2
    assert body["seams"][1]["level_step_db"] > 3.0
    assert "findings" in body and "rules_applied" in body


def test_an_unknown_probe_is_refused_before_the_name_is_read(server):
    with server() as client:
        # the file does not exist: the whitelist answers first
        answer = client.get("/api/gallery/missing.mp4/assess?probe=analyze_vibes")

    assert answer.status_code == 400
    detail = answer.json()["detail"]
    for probe in (
        "analyze_shots",
        "analyze_seams",
        "analyze_sync_drift",
        "analyze_held_frames",
    ):
        assert probe in detail


def test_a_still_is_not_applicable_to_any_probe(server, tmp_path):
    with server() as client:
        outputs = tmp_path / "outputs"
        outputs.mkdir(exist_ok=True)
        Image.new("RGB", (8, 8)).save(outputs / "still.png")
        body = client.get("/api/gallery/still.png/assess").json()

    assert body["kind"] == "image"
    assert body["findings"] == []
    assert set(body["not_applicable"]) == {
        "analyze_shots",
        "analyze_seams",
        "analyze_sync_drift",
        "analyze_held_frames",
    }
    assert "still" in body["not_applicable"]["analyze_seams"]
    assert "still" in body["not_applicable"]["analyze_held_frames"]


def test_a_file_with_no_recorded_shots_says_seams_do_not_apply(server, tmp_path):
    with server() as client:
        write_cut(tmp_path / "outputs" / "loose.mp4", (0.1, 0.4))
        body = client.get("/api/gallery/loose.mp4/assess").json()

    assert body["shots_source"] == "none"
    assert "analyze_seams" in body["not_applicable"]
    assert "analyze_shots" not in body["not_applicable"]


def test_an_asset_is_assessed_with_the_shots_kept_beside_it(server, tmp_path):
    with server() as client:
        assets = tmp_path / "assets"
        write_cut(assets / "episode.mp4", (0.1, 0.1, 0.4))
        record_kept_shots(str(assets), "episode.mp4", shots_of(3))
        body = client.get("/api/gallery/asset:episode.mp4/assess").json()

    assert body["source"] == "asset"
    assert body["shots_source"] == "manifest"
    assert [
        f["at"]["seam"] for f in body["findings"] if f["rule"] == "seam_level_step"
    ] == [2]


def test_an_output_reference_resolves(server, tmp_path):
    with server() as client:
        write_run(tmp_path / "outputs")
        answer = client.get(f"/api/gallery/output:{CUT}/assess")

    assert answer.status_code == 200
    assert answer.json()["name"] == CUT


def test_an_overrunning_shot_is_reported_once_not_per_probe(server, tmp_path):
    # #427: analyze_shots, analyze_seams and analyze_sync_drift all raise the
    # same shot_span_overrun finding (via the shared _shot_span_findings), so
    # the compact merge listed one real overrun three times.
    with server() as client:
        run_dir = tmp_path / "outputs" / RUN
        write_cut(run_dir / "final" / "cut.mp4", (0.1, 0.1, 0.4))
        overrunning = shots_of(3)
        overrunning[-1]["num_samples"] += 1000
        manifest = {
            "steps": [
                {"step": "join", "files": ["final/cut.mp4"], "shots": overrunning}
            ]
        }
        (run_dir / MANIFEST_FILE_NAME).write_text(json.dumps(manifest))

        body = client.get(f"/api/gallery/{CUT}/assess").json()

    overrun = [f for f in body["findings"] if f["rule"] == "shot_span_overrun"]
    assert len(overrun) == 1
    assert overrun[0]["at"]["shot"] == "s2"


def test_a_path_outside_the_outputs_is_refused(server):
    with server() as client:
        answer = client.get("/api/gallery/..%2Fsecret.mp4/assess")

    assert answer.status_code == 404


def test_it_answers_while_a_job_holds_the_worker(server, tmp_path):
    """The route is not a job: it answers while one is running, and queues
    nothing behind it."""
    with server(hanging_script) as client:
        write_run(tmp_path / "outputs")
        running = client.post("/api/jobs", json={"workflow": valid_workflow()}).json()
        wait_for_status(client, running["id"], ["running"])

        answer = client.get(f"/api/gallery/{CUT}/assess")
        jobs = client.get("/api/jobs").json()

        client.post(f"/api/jobs/{running['id']}/cancel")

    assert answer.status_code == 200
    assert answer.json()["findings"]
    listed = jobs["jobs"] if isinstance(jobs, dict) else jobs
    assert len(listed) == 1


class TestMetadataLevelFindings:
    """The metadata route reports a level problem itself, from the
    thresholds dw/audio_qc.py owns, so no client restates them."""

    def test_a_full_scale_peak_is_a_warn_finding(self):
        from dw.server.assess import level_findings

        found = level_findings({"kind": "audio", "peak_dbfs": 0.0, "mean_dbfs": -12.0})
        assert [f["rule"] for f in found] == ["full_scale"]
        assert found[0]["severity"] == "warn"
        assert "normalize_audio" in found[0]["says"]
        # the gallery reads the written file: past full scale it clips on
        # playback, and the fix is audio_qc's - -3 dB, more for a mux
        assert "clips on playback" in found[0]["says"]
        assert "peak_dbfs: -3" in found[0]["says"]
        # and for the file already written, the template that re-levels it
        assert "relevel-clip" in found[0]["says"]

    def test_near_silent_with_real_peaks_is_info(self):
        from dw.server.assess import level_findings

        quiet = level_findings(
            {"kind": "audio", "peak_dbfs": -18.0, "mean_dbfs": -54.0}
        )
        empty = level_findings(
            {"kind": "audio", "peak_dbfs": -60.0, "mean_dbfs": -70.0}
        )
        assert [(f["rule"], f["severity"]) for f in quiet] == [("near_silent", "info")]
        assert [(f["rule"], f["severity"]) for f in empty] == [("near_silent", "warn")]

    def test_no_levels_no_findings(self):
        from dw.server.assess import level_findings

        assert level_findings(None) == []
        assert level_findings({"kind": "video"}) == []
        assert level_findings({"peak_dbfs": -6.0, "mean_dbfs": -20.0}) == []

    def test_level_findings_read_audio_qc_thresholds(self, monkeypatch):
        from dw import audio_qc
        from dw.server.assess import level_findings

        monkeypatch.setattr(audio_qc, "CLIPPED_WARN_DBFS", -3.0)
        found = level_findings({"peak_dbfs": -2.0, "mean_dbfs": -12.0})
        assert [f["rule"] for f in found] == ["full_scale"]

    def test_the_metadata_route_carries_findings(self, server, tmp_path):
        import wave

        path = tmp_path / "outputs" / "loud" / "20260923-120000-abcdef01" / "a.wav"
        path.parent.mkdir(parents=True)
        t = numpy.arange(SAMPLE_RATE // 4) / SAMPLE_RATE
        samples = (numpy.sin(2 * numpy.pi * 440.0 * t) * 32767).astype(numpy.int16)
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(SAMPLE_RATE)
            out.writeframes(samples.tobytes())
        with server() as client:
            name = "loud/20260923-120000-abcdef01/a.wav"
            body = client.get(f"/api/gallery/{name}/metadata").json()
        assert [f["rule"] for f in body["findings"]] == ["full_scale"]


def write_picture_only(path, frames=48):
    """A real mp4 with a panning picture, a repeat every 4th frame and no
    audio track."""
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), "w")
    video = container.add_stream(
        "libx264", rate=FPS, options={"crf": "8", "preset": "ultrafast"}
    )
    video.width, video.height, video.pix_fmt = 128, 72, "yuv420p"
    rng = numpy.random.default_rng(3)
    field = rng.normal(0, 1, (72, 256))
    kx = numpy.fft.fftfreq(256)[None, :]
    ky = numpy.fft.fftfreq(72)[:, None]
    field = numpy.real(
        numpy.fft.ifft2(numpy.fft.fft2(field) * numpy.exp(-(kx**2 + ky**2) * 300))
    )
    field = 128 + 50 * (field - field.mean()) / field.std()
    previous = None
    for index in range(frames):
        grey = numpy.roll(field, -index * 3, axis=1)[:, :128]
        if index % 4 == 2 and previous is not None:
            grey = previous
        previous = grey
        arr = numpy.stack([numpy.clip(grey, 0, 255).astype(numpy.uint8)] * 3, -1)
        for packet in video.encode(av.VideoFrame.from_ndarray(arr, "rgb24")):
            container.mux(packet)
    for packet in video.encode():
        container.mux(packet)
    container.close()


def write_audio_only(path, seconds=1.0):
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    t = numpy.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    samples = (0.2 * 32767 * numpy.sin(2 * numpy.pi * 440.0 * t)).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(samples.tobytes())


def test_one_probe_held_frames_answers_its_full_body(server, tmp_path):
    with server() as client:
        write_picture_only(tmp_path / "outputs" / "held.mp4", frames=96)
        body = client.get(
            "/api/gallery/held.mp4/assess?probe=analyze_held_frames"
        ).json()

    assert body["probe"] == "analyze_held_frames"
    assert body["held_frames"] == 24
    assert body["clusters"]
    assert [f["rule"] for f in body["findings"]][0] == "held_frames"
    assert "series" not in body


def test_a_default_call_applies_held_frames_to_a_video(server, tmp_path):
    with server() as client:
        write_cut(tmp_path / "outputs" / "loose.mp4", (0.1, 0.4))
        body = client.get("/api/gallery/loose.mp4/assess").json()

    assert "held_frames" in body["rules_applied"]
    assert "analyze_held_frames" not in body["not_applicable"]


def test_held_frames_detail_includes_the_series(server, tmp_path):
    with server() as client:
        write_picture_only(tmp_path / "outputs" / "held.mp4", frames=48)
        body = client.get(
            "/api/gallery/held.mp4/assess?probe=analyze_held_frames&detail=true"
        ).json()

    assert body["series"]
    assert set(body["series"][0]) == {"frame", "d"}


def test_an_audio_only_file_is_not_applicable_to_held_frames(server, tmp_path):
    with server() as client:
        write_audio_only(tmp_path / "outputs" / "tone.wav")
        answer = client.get("/api/gallery/tone.wav/assess")

    assert answer.status_code == 200
    body = answer.json()
    assert "error" not in body
    assert "audio-only" in body["not_applicable"]["analyze_held_frames"]
    assert "held_frames" not in body["rules_applied"]


def test_a_video_without_audio_is_still_assessed_for_held_frames(server, tmp_path):
    with server() as client:
        write_picture_only(tmp_path / "outputs" / "silent.mp4", frames=96)
        body = client.get("/api/gallery/silent.mp4/assess").json()

    assert "analyze_held_frames" not in body["not_applicable"]
    assert "held_frames" in body["rules_applied"]
    assert [f for f in body["findings"] if f["rule"] == "held_frames"]
