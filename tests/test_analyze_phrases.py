"""analyze_phrases: sung phrases from voiced energy, centre-weighted (#1095)."""

import numpy
import pytest
import soundfile

import dw.tasks.task  # noqa: F401 - registers every command, as validation does
from dw.task_domains import task_argument_errors
from dw.task_problems import phrases_errors
from dw.tasks.phrases import analyze_phrases, voiced_runs
from dw.workflow import Workflow

SR = 22050
BURSTS = [(0.5, 1.5), (2.0, 3.2), (4.0, 5.5)]
SECONDS = 6.0


def tone_bursts(bursts=BURSTS, seconds=SECONDS, level=0.3):
    """A 440 Hz tone, on during each (start, end) and silent between."""
    t = numpy.arange(int(seconds * SR)) / SR
    tone = numpy.zeros_like(t)
    for start, end in bursts:
        on = (t >= start) & (t < end)
        tone[on] = level * numpy.sin(2 * numpy.pi * 440 * t[on])
    return tone


def noise(seconds=SECONDS, level=0.3, seed=0):
    return level * numpy.random.default_rng(seed).standard_normal(int(seconds * SR))


def run(wave, **kwargs):
    return analyze_phrases(
        numpy.asarray(wave, dtype=numpy.float32), sample_rate=SR, **kwargs
    )


def spans(result):
    return [(p["start"], p["end"]) for p in result["phrases"]]


def assert_found(result, bursts=BURSTS):
    found = spans(result)
    assert len(found) == len(bursts), found
    for (start, end), (want_start, want_end) in zip(found, bursts):
        assert start == pytest.approx(want_start, abs=0.05)
        assert end == pytest.approx(want_end, abs=0.05)


class TestCentre:
    def test_centred_bursts_under_hard_panned_noise_are_found(self):
        tone = tone_bursts()
        result = run(numpy.stack([tone + noise(), tone]))
        assert result["method"] == "centre"
        assert_found(result)
        assert result["warnings"] == []
        assert result["duration_seconds"] == SECONDS

    def test_noise_panned_to_either_side_cancels(self):
        tone = tone_bursts()
        result = run(numpy.stack([tone, tone + noise(seed=3)]))
        assert_found(result)

    def test_bursts_panned_hard_to_one_side_are_not_centre(self):
        tone = tone_bursts()
        hard_left = numpy.stack([tone, numpy.zeros_like(tone)])
        result = run(hard_left)
        assert result["method"] == "centre"
        assert result["phrases"] == []
        assert any("no phrases" in w for w in result["warnings"])
        assert any("'centre': false" in w for w in result["warnings"])

    def test_bursts_panned_hard_to_one_side_are_found_without_centre(self):
        tone = tone_bursts()
        result = run(numpy.stack([tone, numpy.zeros_like(tone)]), centre=False)
        assert result["method"] == "band"
        assert_found(result)
        assert result["warnings"] == []

    def test_identical_channels_fall_back_to_band_without_a_warning(self):
        tone = tone_bursts()
        result = run(numpy.stack([tone, tone]))
        assert result["method"] == "band"
        assert_found(result)
        assert result["warnings"] == []


class TestBand:
    def test_mono_input_is_band_with_a_warning(self):
        result = run(tone_bursts()[None, :])
        assert result["method"] == "band"
        assert_found(result)
        (warning,) = result["warnings"]
        assert "mono" in warning

    def test_a_one_dimensional_waveform_is_mono(self):
        result = run(tone_bursts())
        assert result["method"] == "band"
        assert any("mono" in w for w in result["warnings"])

    def test_a_dense_track_is_one_phrase_with_a_warning(self):
        wave = noise()
        result = run(numpy.stack([wave, wave]))
        assert len(result["phrases"]) == 1
        assert any("stems" in w and "90%" in w for w in result["warnings"])


class TestSilence:
    def test_silence_gives_no_phrases_and_a_warning(self):
        result = run(numpy.zeros((2, int(3 * SR))))
        assert result["phrases"] == []
        assert result["duration_seconds"] == 3.0
        assert any("no phrases" in w for w in result["warnings"])

    def test_near_silent_hiss_gives_no_phrases(self):
        hiss = noise(level=1e-4)
        result = run(numpy.stack([hiss, noise(level=1e-4, seed=1)]))
        assert result["phrases"] == []
        assert any("near-silent" in w for w in result["warnings"])

    def test_a_given_floor_skips_the_near_silence_gate(self):
        quiet = tone_bursts(level=1e-3)
        result = run(quiet[None, :], floor_db=-75)
        assert_found(result)


class TestPhraseRules:
    def test_a_short_gap_is_bridged(self):
        result = run(tone_bursts([(0.5, 1.5), (1.7, 2.5)], seconds=3.0)[None, :])
        assert spans(result) == [pytest.approx((0.5, 2.5), abs=0.05)]

    def test_a_gap_at_min_gap_splits(self):
        result = run(
            tone_bursts([(0.5, 1.5), (1.7, 2.5)], seconds=3.0)[None, :],
            min_gap_seconds=0.1,
        )
        assert len(result["phrases"]) == 2

    def test_a_short_run_is_dropped(self):
        result = run(tone_bursts([(0.5, 1.5), (2.2, 2.35)], seconds=3.0)[None, :])
        assert spans(result) == [pytest.approx((0.5, 1.5), abs=0.05)]

    def test_phrases_are_ordered_and_inside_the_track(self):
        result = run(tone_bursts([(0.0, 0.8), (1.5, 3.0)], seconds=3.0)[None, :])
        found = spans(result)
        assert found == sorted(found)
        assert all(0 <= s < e <= result["duration_seconds"] for s, e in found)
        assert all(a[1] <= b[0] for a, b in zip(found, found[1:]))

    def test_voiced_runs_bridges_then_drops(self):
        voiced = numpy.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1], dtype=bool)
        assert voiced_runs(voiced, min_gap_frames=2, min_phrase_frames=2) == [
            (0, 5),
            (12, 15),
        ]


class TestFromAFile:
    def test_a_stereo_wav_keeps_its_channels(self, tmp_path):
        tone = tone_bursts()
        path = tmp_path / "song.wav"
        soundfile.write(path, numpy.stack([tone + noise(), tone]).T, SR)
        result = analyze_phrases(str(path))
        assert result["method"] == "centre"
        assert_found(result)


class TestRefusals:
    def errors(self, arguments):
        step = {
            "name": "phrases",
            "task": {"command": "analyze_phrases", "arguments": arguments},
        }
        return task_argument_errors({"id": "p", "steps": [step]})

    @pytest.mark.parametrize(
        "name, value",
        [
            ("min_gap_seconds", -1),
            ("min_phrase_seconds", 0),
            ("min_phrase_seconds", -0.5),
            ("floor_db", 6),
        ],
    )
    def test_an_out_of_range_value_is_refused_at_its_path(self, name, value):
        errors = self.errors({"audio": "asset:song.wav", name: value})
        assert [e["path"] for e in errors] == [f"steps[0].task.arguments.{name}"]

    def test_a_zero_gap_is_allowed(self):
        assert self.errors({"audio": "asset:song.wav", "min_gap_seconds": 0}) == []

    def test_a_string_centre_is_refused(self):
        assert phrases_errors({"centre": "false"})[0][0] == "centre"
        errors = self.errors({"audio": "asset:song.wav", "centre": "false"})
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.centre"]

    def test_a_referenced_centre_is_left_to_the_run(self):
        assert phrases_errors({"centre": "variable:centre"}) == []

    def test_the_run_refuses_a_string_centre(self):
        with pytest.raises(ValueError, match="centre"):
            run(tone_bursts()[None, :], centre="false")

    def test_the_run_refuses_a_negative_gap(self):
        with pytest.raises(ValueError, match="min_gap_seconds"):
            run(tone_bursts()[None, :], min_gap_seconds=-1)

    def test_validate_refuses_a_missing_audio_and_a_negative_gap(self):
        workflow = Workflow(
            {
                "id": "p",
                "steps": [
                    {
                        "name": "phrases",
                        "task": {
                            "command": "analyze_phrases",
                            "arguments": {"min_gap_seconds": -1},
                        },
                    }
                ],
            },
            "outputs",
            None,
        )
        messages = " ".join(e["message"] for e in workflow.validation_errors())
        assert "audio" in messages
        assert "min_gap_seconds" in messages
