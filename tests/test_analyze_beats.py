"""analyze_beats: onset tracking, the loudness fallback, anchors, validation."""

import json

import numpy
import pytest

from dw import dsp
from dw.task_domains import beats_problems, task_argument_errors
from dw.tasks.beats import QUIET_DBFS, analyze_beats

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


class TestPulseConfidence:
    def test_detrending_removes_drift_and_keeps_a_pulse(self):
        rate = 100.0
        frames = numpy.arange(1000)
        pulse = (frames % 50 == 0).astype(float)
        detrended = dsp.detrended(pulse + numpy.linspace(0, 5, 1000), rate)
        assert abs(detrended[100:900].mean()) < 0.05
        peaks = numpy.nonzero(detrended > 0.5)[0]
        assert set(peaks) <= set(frames[frames % 50 == 0])

    def test_salience_of_beats_on_clicks_and_on_noise(self):
        rng = numpy.random.default_rng(0)
        envelope = numpy.abs(rng.standard_normal(1000))
        clicks = numpy.arange(25, 1000, 50)
        envelope[clicks] += 20
        assert dsp.beat_salience(envelope, clicks) > 10
        noise = numpy.arange(10, 1000, 50)
        assert dsp.beat_salience(envelope, noise) < dsp.CONFIDENT_SALIENCE
        assert dsp.is_confident(envelope, noise, dsp.CONFIDENT_PERIODICITY)
        assert not dsp.is_confident(envelope, noise, 0.1)


class TestFaintPulse:
    def test_a_faint_pulse_is_kept_and_warned_about(self, monkeypatch, track128):
        # Periodicity under CONFIDENT_PERIODICITY and salience under
        # CONFIDENT_SALIENCE together mean faint, not absent (a song with a
        # long quiet build, 2026-10-06): the beats stay, the caller is told
        monkeypatch.setattr(dsp, "is_confident", lambda *_: False)
        waveform, times, confident = track128
        result = run(waveform)
        assert result["method"] == "onset"
        assert result["beats"] == confident["beats"]
        assert len(result["warnings"]) == 1
        assert "faint" in result["warnings"][0]
        assert "anchors" in result["warnings"][0]


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


def burst(name):
    t = numpy.arange(int(0.05 * SR)) / SR
    if name == "tone":
        # A hard-gated tone: its release splatters a second, later onset
        return 0.5 * numpy.sin(2 * numpy.pi * 880 * t)
    return click()


def backbeat(bpm=85.71, seconds=30.0, first=0.3):
    """Kick and snare alternating on the beat over a noise bed from 0 s - a
    pulse strongest at the bar, with a weak kick at each end."""
    rng = numpy.random.default_rng(2)
    signal = 0.05 * rng.standard_normal(int(seconds * SR))
    t = numpy.arange(int(0.2 * SR)) / SR
    kick = numpy.sin(2 * numpy.pi * 60 * t) * numpy.exp(-t / 0.05)
    snare = (
        0.4 * rng.standard_normal(int(0.1 * SR)) * numpy.exp(-t[: int(0.1 * SR)] / 0.03)
    )
    times = numpy.arange(first, seconds - 0.3, 60.0 / bpm)
    for index, time in enumerate(times):
        place(signal, time, kick if index % 2 == 0 else snare)
    return signal.astype(numpy.float32)[None, :], times


class TestEdges:
    """The first and last beats, which the programme sees from one side."""

    @pytest.mark.parametrize("bpm", [100, 120, 128])
    @pytest.mark.parametrize("shape", ["click", "tone"])
    @pytest.mark.parametrize("first", [0.0, 0.5])
    def test_every_click_to_the_ends(self, bpm, shape, first):
        signal = numpy.zeros(SR * 20)
        times = numpy.arange(first, 20.0 - 0.02, 60.0 / bpm)
        for time in times:
            place(signal, time, burst(shape))
        result = run(signal.astype(numpy.float32)[None, :])
        assert result["method"] == "onset"
        assert len(result["beats"]) == len(times)
        assert_beats_match(result["beats"], times)

    def test_a_bed_from_zero_puts_no_beat_at_zero(self):
        waveform, times = backbeat()
        result = run(waveform)
        assert result["beats"][0] == pytest.approx(times[0], abs=0.02)
        assert_beats_match(result["beats"], times)


class TestRange:
    def test_a_range_above_the_pulse_folds_it_up(self):
        waveform, times = backbeat()
        result = run(waveform, min_bpm=140, max_bpm=200)
        assert result["method"] == "onset"
        assert result["bpm"] == pytest.approx(171.4, abs=1)
        assert result["warnings"] == []
        for time in times:
            assert numpy.abs(numpy.array(result["beats"]) - time).min() <= 0.02

    def test_a_weak_tempo_in_the_range_yields_to_a_strong_pulse_outside_it(self):
        # A half-time song: its strongest pulse is 50 BPM, under the range,
        # and a soft non-octave train inside the range repeats just well
        # enough to track on its own (periodicity about 0.22). The pulse's
        # octave (100 BPM) wins, not the train (acorn-wars, 2026-10-06)
        waveform, strong = click_track(50, seconds=40.0, first=0.5)
        signal = waveform[0].astype(numpy.float64)
        for time in numpy.arange(0.65, 39.9, 60.0 / 190.0):
            place(signal, time, click(amplitude=0.15))
        result = run(signal.astype(numpy.float32)[None, :])
        assert result["method"] == "onset"
        assert result["bpm"] == pytest.approx(100, abs=1)
        for time in strong:
            assert numpy.abs(numpy.array(result["beats"]) - time).min() <= 0.02

    def test_a_range_missing_every_octave_keeps_to_the_range_and_says_so(self):
        waveform, _ = backbeat()
        result = run(waveform, min_bpm=119, max_bpm=120)
        assert 119 <= result["bpm"] <= 120
        assert any("no octave" in warning for warning in result["warnings"])

    def test_the_fallback_bpm_is_in_the_range(self):
        rng = numpy.random.default_rng(0)
        t = numpy.arange(SR * 20) / SR
        signal = rng.standard_normal(t.shape[0]) * (
            0.5 - 0.5 * numpy.cos(2 * numpy.pi * t)
        )
        result = run(
            (0.3 * signal).astype(numpy.float32)[None, :], min_bpm=100, max_bpm=140
        )
        assert result["method"] == "rms_peaks"
        assert result["bpm"] == pytest.approx(120, abs=6)
        result = run(
            (0.3 * signal).astype(numpy.float32)[None, :], min_bpm=70, max_bpm=80
        )
        assert result["bpm"] is None
        assert any("no octave" in warning for warning in result["warnings"])

    def test_fold_bpm(self):
        assert dsp.fold_bpm(85.7, 140, 200) == pytest.approx(171.4)
        assert dsp.fold_bpm(85.7, 119, 120) is None
        assert dsp.fold_bpm(240, 60, 200, centre=120) == 120


class TestFallback:
    @pytest.mark.parametrize("gain_db", [-120, -50])
    def test_a_quiet_song_is_not_tracked(self, gain_db):
        waveform, _ = backbeat()
        scale = 10 ** (gain_db / 20) / numpy.abs(waveform).max()
        result = run(waveform * scale)
        assert result["beats"] == []
        assert result["bpm"] is None
        assert result["method"] == "rms_peaks"
        assert result["warnings"]

    def test_room_tone_is_not_tracked(self):
        rng = numpy.random.default_rng(3)
        tone = 10 ** (-50 / 20) * rng.standard_normal(SR * 20)
        result = run(tone.astype(numpy.float32)[None, :])
        assert result["beats"] == []
        assert "near-silent" in result["warnings"][0]

    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize(
        "shape", ["loud_head", "fade", "swell", "full_scale", "transient"]
    )
    def test_a_noise_bed_above_the_level_gate_is_not_a_pulse(self, seed, shape):
        # Room tone loud enough to pass the near-silent gate: a drifting level
        # once read as a confident grid (C-F208, #625)
        rng = numpy.random.default_rng(seed)
        count = int(4.96 * SR)
        spectrum = numpy.fft.rfft(rng.standard_normal(count))
        spectrum /= numpy.sqrt(numpy.maximum(numpy.arange(spectrum.shape[0]), 1))
        bed = numpy.fft.irfft(spectrum, count)
        bed *= 10 ** (-50 / 20) / numpy.sqrt(numpy.mean(bed**2))
        if shape == "loud_head":
            bed[: int(0.3 * SR)] *= 4
        elif shape == "fade":
            bed *= numpy.linspace(1.6, 0.6, count)
        elif shape == "swell":
            bed *= 1 + 0.5 * numpy.sin(2 * numpy.pi * 0.3 * numpy.arange(count) / SR)
        elif shape == "full_scale":
            bed *= 10 ** (35 / 20)
        else:
            bed[SR : SR + 400] += 0.05 * rng.standard_normal(400)
        result = run(bed.astype(numpy.float32)[None, :])
        assert result["method"] == "rms_peaks"
        assert result["warnings"]
        if dsp.rms_peaks(bed, SR, 200)[1] >= QUIET_DBFS:
            assert "no reliable beat" in result["warnings"][0]

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
