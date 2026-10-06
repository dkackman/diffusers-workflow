"""analyze_beats: onset tracking, the loudness fallback, anchors, validation."""

import json

import numpy
import pytest

from dw import dsp
from dw.task_domains import beats_problems, task_argument_errors
from dw.tasks.beats import analyze_beats

SR = 44100


def click(freq=1000.0, amplitude=1.0):
    t = numpy.arange(int(0.01 * SR)) / SR
    return amplitude * numpy.sin(2 * numpy.pi * freq * t) * numpy.exp(-t / 0.002)


def place(signal, seconds, burst):
    start = int(round(seconds * SR))
    end = min(len(signal), start + len(burst))
    signal[start:end] += burst[: end - start]


def click_track(bpm, seconds=30.0, first=0.5):
    signal = numpy.zeros(int(seconds * SR))
    period = 60.0 / bpm
    times = numpy.arange(first, seconds - 0.02, period)
    for time in times:
        place(signal, time, click())
    return signal.astype(numpy.float32)[None, :], times


def run(waveform, **kwargs):
    return analyze_beats(waveform, sample_rate=SR, **kwargs)


def assert_beats_match(beats, truth, tolerance=0.02):
    beats = numpy.asarray(beats)
    truth = numpy.asarray(truth)
    for time in truth:
        assert numpy.abs(beats - time).min() <= tolerance, f"true click {time}"
    for time in beats:
        assert numpy.abs(truth - time).min() <= tolerance, f"beat {time}"


@pytest.fixture(scope="module")
def track128():
    waveform, times = click_track(128)
    return waveform, times, run(waveform)


class TestOnsetTracking:
    @pytest.mark.parametrize("bpm", [90, 128])
    def test_click_tracks(self, bpm):
        waveform, times = click_track(bpm)
        result = run(waveform)
        assert result["method"] == "onset"
        assert abs(result["bpm"] - bpm) <= 1
        assert_beats_match(result["beats"], times)
        assert all(b > a for a, b in zip(result["beats"], result["beats"][1:]))

    def test_swung_track_keeps_the_main_tempo(self):
        bpm, seconds = 100, 30.0
        period = 60.0 / bpm
        signal = numpy.zeros(int(seconds * SR))
        mains = numpy.arange(0.5, seconds - 0.1, period)
        for time in mains:
            place(signal, time, click())
            place(signal, time + period * 2 / 3, click(2000.0, 0.4))
        result = run(signal.astype(numpy.float32)[None, :])
        assert abs(result["bpm"] - 100) <= 3
        for time in result["beats"]:
            assert numpy.abs(mains - time).min() <= 0.02


class TestFallback:
    def test_silence(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            "dw.events.emit_warning", lambda message, **data: seen.append(message)
        )
        result = run(numpy.zeros((1, SR * 5), dtype=numpy.float32))
        assert result["beats"] == []
        assert result["bpm"] is None
        assert result["method"] == "rms_peaks"
        assert result["warnings"]
        assert seen == result["warnings"]

    def test_swells_fall_back_to_loudness_peaks(self):
        rng = numpy.random.default_rng(0)
        t = numpy.arange(SR * 20) / SR
        signal = rng.standard_normal(t.shape[0]) * (
            0.5 - 0.5 * numpy.cos(2 * numpy.pi * t)
        )
        result = run((0.3 * signal).astype(numpy.float32)[None, :])
        assert result["method"] == "rms_peaks"
        assert abs(result["bpm"] - 60) <= 3
        assert result["warnings"]


class TestAnchors:
    def test_one_anchor_and_tempo_lay_a_grid(self):
        result = run(
            numpy.zeros((1, SR * 10), dtype=numpy.float32),
            tempo_bpm=128,
            anchors=[{"beat_index": 0, "seconds": 0.5}],
        )
        assert result["method"] == "grid"
        assert result["bpm"] == 128
        beats = result["beats"]
        assert beats[0] == 0.5
        assert numpy.allclose(numpy.diff(beats), 60 / 128, atol=1e-3)

    def test_a_bare_anchor_grid_runs_both_ways(self):
        result = run(
            numpy.zeros((1, SR * 10), dtype=numpy.float32),
            tempo_bpm=120,
            anchors=[3.0],
        )
        assert result["method"] == "grid"
        assert result["beats"][0] == 0.0
        assert 3.0 in result["beats"]

    def test_a_beat_index_putting_beat_zero_before_the_start_is_refused(self):
        with pytest.raises(ValueError, match="before the song starts"):
            run(
                numpy.zeros((1, SR * 10), dtype=numpy.float32),
                tempo_bpm=120,
                anchors=[{"beat_index": 4, "seconds": 0.5}],
            )

    def _truth(self, track128):
        _, _, base = track128
        truth = numpy.array(base["beats"]) * 1.005 + 0.05
        return base, truth

    def test_two_bare_anchors_correct_drift(self, track128):
        waveform, _, _ = track128
        base, truth = self._truth(track128)
        result = run(waveform, anchors=[float(truth[2]), float(truth[-3])])
        assert result["calibration"]["anchors_used"] == 2
        kept = numpy.array(result["beats"])
        assert numpy.abs(kept - truth[: len(kept)]).max() < 0.002
        expected = (truth[-3] - base["beats"][-3]) - (truth[2] - base["beats"][2])
        assert result["calibration"]["drift"] > 0
        assert result["calibration"]["drift"] == pytest.approx(expected, abs=2e-3)

    def test_two_indexed_anchors_correct_drift(self, track128):
        waveform, _, _ = track128
        base, truth = self._truth(track128)
        last = len(truth) - 3
        result = run(
            waveform,
            anchors=[
                {"beat_index": 2, "seconds": float(truth[2])},
                {"beat_index": last, "seconds": float(truth[last])},
            ],
        )
        assert result["calibration"]["anchors_used"] == 2
        kept = numpy.array(result["beats"])
        assert numpy.abs(kept - truth[: len(kept)]).max() < 0.002
        assert result["calibration"]["drift"] > 0

    def test_one_bare_anchor_shifts_every_beat(self, track128):
        waveform, _, base = track128
        shifted = run(waveform, anchors=[base["beats"][4] + 0.1])
        assert shifted["calibration"]["anchors_used"] == 1
        deltas = numpy.array(shifted["beats"]) - numpy.array(base["beats"])
        assert numpy.allclose(deltas, 0.1, atol=2e-4)

    def test_an_anchor_past_the_end_is_refused(self):
        waveform, _ = click_track(128, seconds=10)
        with pytest.raises(ValueError, match="past the song's end"):
            run(waveform, anchors=[11.0])

    def test_a_beat_index_past_the_detected_count_is_refused(self):
        waveform, _ = click_track(128, seconds=10)
        with pytest.raises(ValueError, match="only"):
            run(waveform, anchors=[{"beat_index": 500, "seconds": 5.0}])

    def test_an_empty_range_is_refused(self):
        with pytest.raises(ValueError, match="min_bpm"):
            run(numpy.zeros((1, SR), dtype=numpy.float32), min_bpm=120, max_bpm=100)

    def test_mixed_and_unordered_anchors_are_refused(self):
        silent = numpy.zeros((1, SR * 10), dtype=numpy.float32)
        with pytest.raises(ValueError, match="mixes"):
            run(silent, anchors=[1.0, {"beat_index": 2, "seconds": 3.0}])
        with pytest.raises(ValueError, match="not after"):
            run(silent, anchors=[3.0, 1.0])


def workflow_errors(arguments):
    return task_argument_errors(
        {
            "id": "beats",
            "steps": [
                {
                    "name": "beats",
                    "task": {"command": "analyze_beats", "arguments": arguments},
                    "result": {"content_type": "application/json"},
                }
            ],
        }
    )


class TestStaticValidation:
    @pytest.mark.parametrize(
        "arguments",
        [
            {"audio": "asset:a.wav", "min_bpm": 120, "max_bpm": 100},
            {"audio": "asset:a.wav", "anchors": [1.0, {"beat_index": 1, "seconds": 2}]},
            {"audio": "asset:a.wav", "anchors": [3.0, 1.0]},
            {"audio": "asset:a.wav", "anchors": [{"beat_index": -1, "seconds": 1}]},
            {"audio": "asset:a.wav", "anchors": [{"beat_index": 1.5, "seconds": 1}]},
            {"audio": "asset:a.wav", "tempo_bpm": 0},
            {"audio": "asset:a.wav", "tempo_bpm": -90},
        ],
    )
    def test_bad_literals_are_reported(self, arguments):
        assert workflow_errors(arguments)

    def test_references_are_not_complained_about(self):
        assert (
            workflow_errors(
                {
                    "audio": "asset:a.wav",
                    "anchors": "variable:marks",
                    "min_bpm": "variable:low",
                    "max_bpm": 100,
                }
            )
            == []
        )

    def test_problems_names_each_argument(self):
        problems = beats_problems(120, 100, [3.0, 1.0])
        assert {name for name, _ in problems} == {"min_bpm", "anchors"}
        assert beats_problems(60, 200, [1.0, 2.0]) == []
        assert beats_problems("variable:x", 100, "variable:y") == []


class TestRealPath:
    def _file(self, tmp_path):
        import soundfile

        waveform, times = click_track(128, seconds=12)
        path = tmp_path / "clicks.wav"
        soundfile.write(path, waveform[0], SR)
        return path, times

    def test_a_wav_file(self, tmp_path):
        path, times = self._file(tmp_path)
        result = analyze_beats(str(path))
        assert result["method"] == "onset"
        assert_beats_match(result["beats"], times)

    def test_the_registered_handler(self, tmp_path):
        from dw.tasks.task import _COMMAND_REGISTRY

        path, _ = self._file(tmp_path)
        result = _COMMAND_REGISTRY["analyze_beats"](
            None, {"audio": str(path), "device": "cpu"}, {}
        )
        assert result["method"] == "onset"
        assert abs(result["bpm"] - 128) <= 1


class TestResultShape:
    def test_json_serializable_with_plain_floats(self, track128):
        _, _, result = track128
        json.dumps(result)
        waveform, _, base = track128
        calibrated = run(waveform, anchors=[base["beats"][2], base["beats"][-3]])
        json.dumps(calibrated)
        for value in calibrated["calibration"].values():
            assert type(value) in (float, int)
        assert type(calibrated["calibration"]["offset_s"]) is float
        assert type(calibrated["bpm"]) is float
        assert all(type(b) is float for b in calibrated["beats"])


class TestWarpTimes:
    def test_two_knots_interpolate_and_extrapolate(self):
        warped = dsp.warp_times([0.0, 1.0, 2.0, 3.0], [1.0, 2.0], [2.0, 4.0])
        assert warped.tolist() == pytest.approx([0.0, 2.0, 4.0, 6.0])

    def test_one_knot_shifts(self):
        warped = dsp.warp_times([0.0, 1.0, 2.0], [1.0], [1.5])
        assert warped.tolist() == pytest.approx([0.5, 1.5, 2.5])
