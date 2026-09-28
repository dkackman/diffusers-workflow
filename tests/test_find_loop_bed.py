"""Tests for find_loop_bed (#218): ranking the quiet windows of a recording
worth looping into a room-tone bed, measured as they will sound looped.

Fixtures are synthetic noise built with a seeded numpy.random.Generator at
48 kHz, so a run is deterministic; each dBFS target is converted to an
amplitude via 10**(dbfs/20) (root-mean-square for noise, amplitude/sqrt(2)
for a sine), matching the engine's own `10*log10(mean square)` bin reading.
"""

import math

import numpy
import pytest
import torch

from dw.introspection import list_tasks
from dw.media_audio import decode_soundtrack
from dw.result import AudioTrack
from dw.scalar_result_validation import scalar_result_errors
from dw.task_domains import task_argument_errors
from dw.tasks.audio_utils import _PERIODICITY_MAX_HZ, _PERIODICITY_MIN_HZ, _harmonicity
from dw.tasks.loop_bed import _occupied_rate, find_loop_bed
from dw.tasks.task import task_command_info
from dw.workflow import Workflow

SR = 48000
BIN_SECONDS = 0.05
BIN_LENGTH = 2400  # round(SR * BIN_SECONDS)


def rms_amplitude(dbfs):
    """The RMS a noise segment needs to read `dbfs` on the engine's own
    10*log10(mean square) scale."""
    return 10.0 ** (dbfs / 20.0)


def track(waveform, sample_rate=SR):
    return AudioTrack(waveform[None, :].astype(numpy.float32), sample_rate)


def noise(rng, num_samples, dbfs):
    return rng.normal(0.0, rms_amplitude(dbfs), num_samples)


CANDIDATE_KEYS = {
    "rank",
    "start_seconds",
    "duration_seconds",
    "end_seconds",
    "shot",
    "mean_dbfs",
    "max_bin_dbfs",
    "spike_db",
    "flatness",
    "harmonicity",
    "looped",
    "gain_db",
    "gain",
    "warnings",
}
LOOPED_KEYS = {
    "ripple_db",
    "envelope_peak_db",
    "envelope_peak_hz",
    "lap_hz",
    "lap_component_db",
}


class TestCleanNoiseWins:
    """A quiet stretch inside a loud track is the only place a candidate can
    come from, and the answer's shape is exactly what the plan promises."""

    @pytest.fixture
    def result(self):
        rng = numpy.random.default_rng(1)
        total = 12 * SR  # bin-aligned: 240 bins
        waveform = noise(rng, total, -50.0)
        quiet_start, quiet_end = 3 * SR, 8 * SR  # bins 60..160
        waveform[quiet_start:quiet_end] = noise(rng, quiet_end - quiet_start, -78.0)
        return find_loop_bed(track(waveform))

    def test_top_level_keys(self, result):
        assert set(result.keys()) == {
            "source",
            "criteria",
            "candidates",
            "rejected",
            "findings",
        }

    def test_source_carries_no_shots(self, result):
        assert result["source"]["shots_source"] is None

    def test_no_findings_when_something_was_found(self, result):
        assert result["findings"] == []

    def test_every_candidate_lies_inside_the_quiet_stretch(self, result):
        candidates = result["candidates"]
        assert candidates
        for candidate in candidates:
            assert candidate["start_seconds"] >= 3.0 - 1e-6
            assert candidate["end_seconds"] <= 8.0 + 1e-6

    def test_candidates_are_sorted_by_looped_ripple_ascending(self, result):
        ripples = [c["looped"]["ripple_db"] for c in result["candidates"]]
        assert ripples == sorted(ripples)

    def test_ranks_are_1_through_n(self, result):
        assert [c["rank"] for c in result["candidates"]] == list(
            range(1, len(result["candidates"]) + 1)
        )

    def test_every_candidate_has_every_plan_key(self, result):
        for candidate in result["candidates"]:
            assert set(candidate.keys()) == CANDIDATE_KEYS
            assert set(candidate["looped"].keys()) == LOOPED_KEYS

    def test_gain_matches_gain_db(self, result):
        for candidate in result["candidates"]:
            expected = 10.0 ** (candidate["gain_db"] / 20.0)
            assert math.isclose(candidate["gain"], expected, rel_tol=1e-3)

    def test_gain_db_targets_the_default_bed_level(self, result):
        for candidate in result["candidates"]:
            expected = -60.0 - candidate["mean_dbfs"]
            assert math.isclose(candidate["gain_db"], expected, abs_tol=0.05)


class TestATick:
    """A single-sample spike is invisible to 50 ms RMS but not to the 1 ms
    peak search, and only a window that covers it should be rejected."""

    def _fixture(self):
        # A loud (-50 dBFS) recording with one quiet (-78 dBFS) stretch from
        # 5s to 15s: the loud material bounds how many non-overlapping quiet
        # windows there are to tile around the tick with, so a window that
        # actually covers the tick is eventually reached once the spike
        # check stops rejecting it. The tick itself is loud enough to
        # dominate its own 1 ms peak (spike_db well above the 12 dB default)
        # but small enough - spread over one 2400 sample bin's mean square -
        # that it barely moves that bin's own level.
        rng = numpy.random.default_rng(2)
        duration = 20 * SR
        waveform = noise(rng, duration, -50.0)
        quiet_start, quiet_end = 5 * SR, 15 * SR
        waveform[quiet_start:quiet_end] = noise(rng, quiet_end - quiet_start, -78.0)
        tick_sample = 10 * SR
        waveform[tick_sample] = 10.0 * rms_amplitude(-78.0)
        return waveform, tick_sample / SR

    def test_no_candidate_covers_the_tick(self):
        waveform, tick_time = self._fixture()
        result = find_loop_bed(track(waveform))

        for candidate in result["candidates"]:
            assert not (
                candidate["start_seconds"] <= tick_time < candidate["end_seconds"]
            )
        assert result["rejected"]["spike"] >= 1

    def test_raising_max_spike_db_lets_the_tick_survive(self):
        waveform, tick_time = self._fixture()
        result = find_loop_bed(track(waveform), max_spike_db=30.0, max_candidates=10)

        assert any(
            candidate["start_seconds"] <= tick_time < candidate["end_seconds"]
            for candidate in result["candidates"]
        )
        assert result["rejected"]["spike"] == 0

    def test_lowering_max_bin_dbfs_below_the_bed_gives_no_candidates(self):
        waveform, _tick_time = self._fixture()
        result = find_loop_bed(track(waveform), max_bin_dbfs=-90.0)

        assert result["candidates"] == []
        assert result["rejected"]["too_loud"] > 0


class TestTonal:
    """A faint pure tone is as quiet as the noise around it, but its
    periodicity should still get it thrown out."""

    def test_no_candidate_covers_the_tone(self):
        rng = numpy.random.default_rng(3)
        duration = 15 * SR
        waveform = noise(rng, duration, -78.0)

        tone_start, tone_end = 6 * SR, 9 * SR
        t = numpy.arange(tone_end - tone_start) / SR
        amplitude = rms_amplitude(-78.0) * math.sqrt(2)
        waveform[tone_start:tone_end] = amplitude * numpy.sin(2 * math.pi * 220.0 * t)

        result = find_loop_bed(track(waveform))

        for candidate in result["candidates"]:
            assert (
                candidate["end_seconds"] <= 6.0 + 1e-6
                or candidate["start_seconds"] >= 9.0 - 1e-6
            )
        assert result["rejected"]["tonal"] >= 1


class TestIntermittentVoice:
    """Speech under a bed comes and goes: 250 ms harmonic syllables every
    400 ms, at the bed's own level. Measured over a whole 0.5-2 s window the
    pauses dilute the voice below the harmonicity threshold (about 0.37-0.40
    here, against 0.45); each syllable is tonal in the block it sits in, so
    the voiced stretch yields no candidate and the tonal tally rises against
    the same bed without it (#544)."""

    VOICED = (5.0, 9.65)  # first onset, last syllable's end

    @staticmethod
    def _bed(voiced):
        rng = numpy.random.default_rng(11)
        waveform = noise(rng, 15 * SR, -78.0)
        if voiced:
            t = numpy.arange(int(0.25 * SR)) / SR
            syllable = sum(
                numpy.sin(2 * math.pi * 150.0 * harmonic * t) / harmonic
                for harmonic in range(1, 6)
            )
            syllable *= rms_amplitude(-78.0) / numpy.sqrt(numpy.mean(syllable**2))
            for onset in numpy.arange(5.0, 9.5, 0.4):
                at = int(round(onset * SR))
                waveform[at : at + syllable.size] += syllable
        return waveform

    def test_no_candidate_covers_the_voice(self):
        result = find_loop_bed(track(self._bed(voiced=True)), max_candidates=10)

        first, last = self.VOICED
        assert result["candidates"]
        for candidate in result["candidates"]:
            assert (
                candidate["end_seconds"] <= first + 1e-6
                or candidate["start_seconds"] >= last - 1e-6
            )

    def test_the_voice_raises_the_tonal_tally(self):
        voiced = find_loop_bed(track(self._bed(voiced=True)))["rejected"]
        control = find_loop_bed(track(self._bed(voiced=False)))["rejected"]

        assert voiced["tonal"] > control["tonal"]


class TestAnUpsampledBed:
    """A bed resampled up to a mix's rate has an empty band above its own
    Nyquist, which spectral flatness reads as tonal whatever the material:
    a 16 kHz bed mixed at 24 kHz read tonal in every window, and a voice
    under it filled the band and read as noise (#544). Flatness is measured
    over the band the material occupies, as bleed_join's is (#198)."""

    @staticmethod
    def _upsampled(seed=21, seconds=4.0, native=SR // 2, dbfs=-78.0):
        """Noise made at `native` and band-limited up to SR by zero-padding
        its spectrum - what a resampler leaves above the old Nyquist."""
        rng = numpy.random.default_rng(seed)
        low = noise(rng, int(seconds * native), dbfs)
        spectrum = numpy.fft.rfft(low)
        length = int(seconds * SR)
        padded = numpy.zeros(length // 2 + 1, dtype=complex)
        padded[: spectrum.shape[0]] = spectrum
        return numpy.fft.irfft(padded, n=length) * (SR / native)

    def test_the_occupied_band_is_measured(self):
        rate = _occupied_rate(self._upsampled(), SR)
        assert rate is not None
        assert SR // 2 <= rate < 0.6 * SR

    def test_a_full_band_source_is_left_alone(self):
        rng = numpy.random.default_rng(22)
        assert _occupied_rate(noise(rng, 4 * SR, -78.0), SR) is None

    def test_upsampled_noise_is_still_a_bed(self):
        result = find_loop_bed(audio=track(self._upsampled()))
        assert result["rejected"]["tonal"] == 0
        assert result["candidates"]
        assert all(c["flatness"] >= 0.3 for c in result["candidates"])


class TestRejectedIsATallyOfEveryWindow:
    """Every window on the grid is counted once, under the first rule it
    fails, so the tally and the survivors together cover the grid."""

    def test_the_counts_cover_the_grid(self):
        rng = numpy.random.default_rng(12)
        waveform = noise(rng, 4 * SR, -78.0)
        waveform[: 2 * SR] = noise(rng, 2 * SR, -40.0)

        result = find_loop_bed(track(waveform), min_seconds=0.5, max_seconds=1.0)

        bins = 4 * SR // BIN_LENGTH
        windows = sum(bins - length + 1 for length in range(10, 21))
        rejected = sum(result["rejected"].values())
        assert 0 < rejected < windows
        assert result["rejected"]["too_loud"] > 0


class TestDigitalSilenceIsSilent:
    """A window that is mostly exact zeros is silence with something in it,
    not a tick in room tone."""

    def test_a_mostly_zero_window_counts_as_silent(self):
        waveform = numpy.zeros(4 * SR)
        rng = numpy.random.default_rng(13)
        # a little noise in the first 10 ms of every bin, zeros after it
        for start in range(0, waveform.size, BIN_LENGTH):
            waveform[start : start + 480] = noise(rng, 480, -78.0)

        result = find_loop_bed(track(waveform))

        assert result["rejected"]["silent"] > 0
        assert result["rejected"]["spike"] == 0
        assert result["candidates"] == []


class TestATickAtTheEdge:
    """A window ending exactly where a click starts would put the click's
    onset at the loop's seam, so the click counts against it."""

    def test_no_candidate_ends_on_the_click(self):
        rng = numpy.random.default_rng(14)
        waveform = noise(rng, 10 * SR, -78.0)
        click = 5 * SR  # the first sample of a bin
        waveform[click : click + 24] = 30.0 * rms_amplitude(-78.0)

        result = find_loop_bed(track(waveform), max_candidates=50)

        assert result["candidates"]
        for candidate in result["candidates"]:
            assert not (candidate["start_seconds"] <= 5.0 <= candidate["end_seconds"])


class TestMaxCandidatesOnlyTruncates:
    """Asking for fewer candidates returns the head of the same ranking,
    not the ranking of a smaller pool (#544)."""

    @staticmethod
    def _unranked(candidate):
        return {key: value for key, value in candidate.items() if key != "rank"}

    def test_fewer_candidates_are_the_head_of_the_default_run(self):
        rng = numpy.random.default_rng(15)
        waveform = noise(rng, 30 * SR, -78.0)
        # uneven level second to second, so the pre-rank and the looped
        # ranking disagree
        for second in range(30):
            waveform[second * SR : (second + 1) * SR] *= 1.0 + 0.4 * rng.random()

        full = find_loop_bed(track(waveform), max_candidates=10)["candidates"]
        for count in (1, 3):
            head = find_loop_bed(track(waveform), max_candidates=count)["candidates"]
            assert [self._unranked(c) for c in head] == [
                self._unranked(c) for c in full[:count]
            ]


class TestLapModulation:
    """A candidate whose level ramps across the window becomes a sawtooth
    once looped, and that sawtooth should be visible in the measurement."""

    @staticmethod
    def _ramped(rng, num_bins=40, low_dbfs=-73.0, high_dbfs=-65.0):
        waveform = numpy.empty(num_bins * BIN_LENGTH)
        for i in range(num_bins):
            dbfs = low_dbfs + (high_dbfs - low_dbfs) * i / (num_bins - 1)
            waveform[i * BIN_LENGTH : (i + 1) * BIN_LENGTH] = noise(
                rng, BIN_LENGTH, dbfs
            )
        return waveform

    @staticmethod
    def _flat(rng, num_bins=40, dbfs=-70.0):
        return noise(rng, num_bins * BIN_LENGTH, dbfs)

    def test_a_ramp_reads_as_lap_modulation(self):
        rng = numpy.random.default_rng(4)
        waveform = self._ramped(rng)

        result = find_loop_bed(track(waveform), min_seconds=2.0, max_seconds=2.0)

        assert len(result["candidates"]) == 1
        looped = result["candidates"][0]["looped"]
        assert looped["envelope_peak_db"] > -15.0
        assert "lap_modulation" in result["candidates"][0]["warnings"]

    def test_a_flat_stretch_does_not(self):
        rng = numpy.random.default_rng(5)
        waveform = self._flat(rng)

        result = find_loop_bed(track(waveform), min_seconds=2.0, max_seconds=2.0)

        assert len(result["candidates"]) == 1
        candidate = result["candidates"][0]
        assert candidate["warnings"] == []

        ramped = find_loop_bed(
            track(self._ramped(numpy.random.default_rng(4))),
            min_seconds=2.0,
            max_seconds=2.0,
        )["candidates"][0]
        assert (
            candidate["looped"]["lap_component_db"]
            < ramped["looped"]["lap_component_db"]
        )


class TestNoCandidates:
    def test_an_all_loud_track_reports_why(self):
        rng = numpy.random.default_rng(6)
        waveform = noise(rng, 5 * SR, -20.0)

        result = find_loop_bed(track(waveform))

        assert result["candidates"] == []
        assert result["rejected"]["too_loud"] > 0
        assert len(result["findings"]) == 1
        assert result["findings"][0]["rule"] == "no_loop_bed"
        assert result["findings"][0]["rejected_by"] == "too_loud"
        assert "max_bin_dbfs" in result["findings"][0]["message"]

    def test_digital_silence_is_not_a_bed(self):
        waveform = numpy.zeros(5 * SR)

        result = find_loop_bed(track(waveform))

        assert result["candidates"] == []
        assert result["rejected"]["silent"] > 0


class TestCandidatesDoNotOverlap:
    def test_pairwise_disjoint(self):
        rng = numpy.random.default_rng(7)
        waveform = noise(rng, 20 * SR, -78.0)

        result = find_loop_bed(track(waveform), max_candidates=5)
        candidates = result["candidates"]
        assert len(candidates) > 1

        for i, a in enumerate(candidates):
            for b in candidates[i + 1 :]:
                assert (
                    a["end_seconds"] <= b["start_seconds"]
                    or b["end_seconds"] <= a["start_seconds"]
                )


class TestRunTimeRefusals:
    @staticmethod
    def _quiet(seconds=5.0, seed=8, dbfs=-78.0):
        rng = numpy.random.default_rng(seed)
        return track(noise(rng, int(seconds * SR), dbfs))

    def test_end_at_or_before_start(self):
        with pytest.raises(ValueError, match="after 'start_seconds'"):
            find_loop_bed(self._quiet(), start_seconds=3.0, end_seconds=2.0)

    def test_min_above_max(self):
        with pytest.raises(ValueError, match="at or below 'max_seconds'"):
            find_loop_bed(self._quiet(), min_seconds=2.0, max_seconds=1.0)

    def test_start_past_the_end(self):
        with pytest.raises(ValueError, match="'start_seconds'.*past the end"):
            find_loop_bed(self._quiet(seconds=2.0), start_seconds=5.0)

    def test_end_past_the_end(self):
        with pytest.raises(ValueError, match="'end_seconds'.*past the end"):
            find_loop_bed(self._quiet(seconds=2.0), end_seconds=5.0)

    def test_source_shorter_than_min_seconds(self):
        with pytest.raises(ValueError, match="shorter than 'min_seconds'"):
            find_loop_bed(self._quiet(seconds=0.2), min_seconds=0.5)

    def test_range_shorter_than_min_seconds(self):
        with pytest.raises(ValueError, match="the range .* is shorter than"):
            find_loop_bed(
                self._quiet(seconds=5.0),
                start_seconds=1.0,
                end_seconds=1.2,
                min_seconds=0.5,
            )

    def test_min_seconds_zero_is_a_domain_violation(self):
        with pytest.raises(ValueError, match="min_seconds"):
            find_loop_bed(self._quiet(), min_seconds=0)

    def test_max_candidates_zero_is_a_domain_violation(self):
        with pytest.raises(ValueError, match="max_candidates"):
            find_loop_bed(self._quiet(), max_candidates=0)

    def test_negative_crossfade_ms_is_a_domain_violation(self):
        with pytest.raises(ValueError, match="crossfade_ms"):
            find_loop_bed(self._quiet(), crossfade_ms=-1)

    def test_string_numerics_are_coerced(self):
        result = find_loop_bed(self._quiet(), min_seconds="0.5")

        assert "candidates" in result


def _write_video(path, num_frames=8, fps=4, sample_rate=8000, audio=True):
    from diffusers.utils.export_utils import encode_video
    from PIL import Image

    frames = [Image.new("RGB", (16, 16), (i, 0, 0)) for i in range(num_frames)]
    clip = torch.full((2, int(num_frames / fps * sample_rate)), 0.25) if audio else None
    encode_video(
        frames,
        fps=fps,
        output_path=str(path),
        audio=clip,
        audio_sample_rate=sample_rate if audio else None,
    )
    return str(path)


class TestVideoSource:
    """A video file's soundtrack is read without decoding its pictures."""

    def test_an_mp4_soundtrack_is_read_without_the_frame_decoder(
        self, tmp_path, monkeypatch
    ):
        def _explode(*args, **kwargs):
            raise AssertionError("load_audio_video should not run for find_loop_bed")

        monkeypatch.setattr("dw.tasks.video_utils.load_audio_video", _explode)
        path = _write_video(tmp_path / "cut.mp4")

        result = find_loop_bed(path)

        assert "candidates" in result

    def test_a_silent_video_is_an_error(self, tmp_path):
        path = _write_video(tmp_path / "mute.mp4", audio=False)

        with pytest.raises(ValueError, match="no audio track"):
            find_loop_bed(path)


class TestDecodeSoundtrack:
    def test_a_wav(self, tmp_path):
        import soundfile

        path = tmp_path / "tone.wav"
        soundfile.write(path, numpy.zeros((100, 2), dtype=numpy.float32), 16000)

        waveform, sample_rate = decode_soundtrack(str(path))

        assert waveform.shape == (2, 100)
        assert waveform.dtype == numpy.float32
        assert sample_rate == 16000

    def test_an_mp4(self, tmp_path):
        path = _write_video(tmp_path / "cut.mp4")

        waveform, sample_rate = decode_soundtrack(path)

        assert waveform.shape[0] == 2
        assert waveform.dtype == numpy.float32
        assert sample_rate == 8000


def _old_harmonicity(waveform, sample_rate):
    """The pre-#218 O(n^2) implementation, kept here only to pin the FFT
    replacement's numbers against it."""
    min_lag = max(int(sample_rate / _PERIODICITY_MAX_HZ), 1)
    max_lag = min(int(sample_rate / _PERIODICITY_MIN_HZ), waveform.shape[1] - 1)
    if max_lag <= min_lag:
        return 0.0

    scores = []
    for channel in waveform:
        centered = channel - channel.mean()
        energy = float(numpy.dot(centered, centered))
        if energy <= 1e-12:
            continue
        n = centered.shape[0]
        correlation = numpy.correlate(centered, centered, "full")
        zero_lag = n - 1
        window = correlation[zero_lag + min_lag : zero_lag + max_lag + 1]
        if window.size == 0:
            continue
        scores.append(float(numpy.max(window) / energy))
    return max(scores) if scores else 0.0


class TestHarmonicityMatchesTheOldImplementation:
    def _assert_matches(self, waveform, sample_rate=SR):
        old = _old_harmonicity(waveform, sample_rate)
        new = _harmonicity(waveform, sample_rate)
        assert numpy.isclose(old, new, rtol=1e-9, atol=1e-12)

    def test_noise(self):
        rng = numpy.random.default_rng(9)
        self._assert_matches(rng.normal(0, 1.0, (1, 4096)))

    def test_a_sine_tone(self):
        t = numpy.arange(4096) / SR
        waveform = numpy.sin(2 * math.pi * 220.0 * t)[numpy.newaxis, :]
        self._assert_matches(waveform)

    def test_two_channels_with_different_content(self):
        rng = numpy.random.default_rng(10)
        t = numpy.arange(4096) / SR
        tone = numpy.sin(2 * math.pi * 300.0 * t)
        waveform = numpy.stack([tone, rng.normal(0, 1.0, 4096)])
        self._assert_matches(waveform)

    def test_a_very_short_waveform(self):
        rng = numpy.random.default_rng(11)
        self._assert_matches(rng.normal(0, 1.0, (1, 50)))

    def test_silence(self):
        self._assert_matches(numpy.zeros((1, 4096)))


class TestRegistry:
    def test_returns_json(self):
        assert task_command_info("find_loop_bed")["returns"] == "json"

    def test_is_a_registered_command(self):
        assert "find_loop_bed" in list_tasks()["commands"]

    def test_is_not_an_assessment_probe(self):
        assert "find_loop_bed" not in list_tasks()["assessment"]


def _step(name, command, result=None, arguments=None):
    step = {"name": name, "task": {"command": command, "arguments": arguments or {}}}
    if result is not None:
        step["result"] = result
    return step


class TestContentType:
    def test_application_json_is_fine(self):
        definition = {
            "steps": [
                _step(
                    "bed",
                    "find_loop_bed",
                    result={"content_type": "application/json"},
                    arguments={"audio": "asset:bed.wav"},
                )
            ]
        }

        assert scalar_result_errors(definition, source_indices=[0]) == []

    def test_audio_wav_is_an_error_naming_application_json(self):
        definition = {
            "steps": [
                _step(
                    "bed",
                    "find_loop_bed",
                    result={"content_type": "audio/wav"},
                    arguments={"audio": "asset:bed.wav"},
                )
            ]
        }

        errors = scalar_result_errors(definition, source_indices=[0])

        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].result"
        assert "application/json" in errors[0]["message"]
        assert "find_loop_bed" in errors[0]["message"]


class TestDomainInAWorkflow:
    def _definition(self, arguments):
        return {
            "id": "loop-bed",
            "steps": [
                _step(
                    "bed",
                    "find_loop_bed",
                    result={"content_type": "application/json"},
                    arguments=arguments,
                )
            ],
        }

    def test_task_argument_errors_reports_a_literal_min_seconds_zero(self):
        errors = task_argument_errors(
            self._definition({"audio": "asset:bed.wav", "min_seconds": 0})
        )

        assert [e["path"] for e in errors] == ["steps[0].task.arguments.min_seconds"]

    def test_validation_errors_reports_it_too(self):
        workflow = Workflow(
            self._definition({"audio": "asset:bed.wav", "min_seconds": 0}),
            "outputs",
            None,
        )

        errors = workflow.validation_errors()

        assert any(e["path"] == "steps[0].task.arguments.min_seconds" for e in errors)
