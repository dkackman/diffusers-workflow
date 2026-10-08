"""Tests for #485: attribute_voices - which reference voice sings each line
of a song, by timbre.

No network access and no real demucs/speechbrain models: separation and
embedding are mocked throughout, following the pattern in
test_speech_generation.py (patch.dict("sys.modules", ...) for the "package
missing" paths, direct patches of the module's own helpers everywhere else).
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy

from dw.introspection import describe_task, list_tasks
from dw.scalar_result_validation import scalar_result_errors
from dw.tasks import model_cache
from dw.tasks.task import _COMMAND_INFO
from dw.tasks.voice_attribution import (
    MIN_VOICED_SECONDS,
    MIXED_LINE_SHARE,
    UNCERTAIN_MARGIN,
    UNCERTAIN_SHARE_MARGIN,
    VOICES_TOO_SIMILAR,
    attribute_voices,
    cosine,
    mixed_reason,
    reference_similarity,
    roll_up,
    score_line,
    voiced_floor_dbfs,
    voiced_mask,
)
from dw.locations import location_errors
from dw.task_domains import task_argument_errors
from dw.trust import TRUST_WORKFLOWS_ENV_VAR
from dw.workflow import Workflow


def tone(seconds, sample_rate=16000, amplitude=0.3, channels=1):
    """A loud sine tone - counts as voiced (well above any voiced floor)."""
    samples = int(round(seconds * sample_rate))
    t = numpy.arange(samples, dtype=numpy.float32) / sample_rate
    wave = (amplitude * numpy.sin(2 * numpy.pi * 220 * t)).astype(numpy.float32)
    return numpy.tile(wave, (channels, 1))


def silence(seconds, sample_rate=16000, channels=1):
    samples = int(round(seconds * sample_rate))
    return numpy.zeros((channels, samples), dtype=numpy.float32)


class FakeAudio:
    """Matches `waveform_and_rate`'s `hasattr(audio, "audio")` branch, so no
    file-path/security validation is triggered."""

    def __init__(self, waveform, sample_rate=16000):
        self.audio = waveform
        self.sample_rate = sample_rate


def song_audio(seconds=10.0, sample_rate=16000):
    return FakeAudio(tone(seconds, sample_rate), sample_rate)


def setUpModule():
    # Guard against a mock leaking into another test through the process-wide
    # model cache keyed on (COMMAND, model name, device, dtype)
    model_cache.clear_model_cache()


def tearDownModule():
    model_cache.clear_model_cache()


# ---------------------------------------------------------------------------
# 1. Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors(unittest.TestCase):
    def test_fewer_than_two_voices_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "at least 2 voices"):
            attribute_voices(
                song_audio(),
                voices={"a": [{"start_seconds": 0, "duration_seconds": 4}]},
            )

    def test_a_voices_reference_under_the_minimum_names_the_voice(self):
        with self.assertRaisesRegex(ValueError, "voice 'a' has"):
            attribute_voices(
                song_audio(),
                voices={
                    "a": [{"start_seconds": 0, "duration_seconds": 1}],
                    "b": [{"start_seconds": 1, "duration_seconds": 4}],
                },
            )

    def test_a_span_past_the_end_of_the_audio_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "past the end"):
            attribute_voices(
                song_audio(seconds=5.0),
                voices={
                    "a": [{"start_seconds": 0, "duration_seconds": 4}],
                    "b": [{"start_seconds": 20, "duration_seconds": 4}],
                },
            )

    def test_a_span_ending_before_it_starts_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "not after its start"):
            attribute_voices(
                song_audio(),
                voices={
                    "a": [{"start": 4, "end": 4}],
                    "b": [{"start_seconds": 4, "duration_seconds": 4}],
                },
            )

    def test_invalid_voice_name_is_named(self):
        for bad in ("1bad", "a b"):
            with self.assertRaisesRegex(ValueError, "voice name .* is not allowed"):
                attribute_voices(
                    song_audio(),
                    voices={
                        bad: [{"start_seconds": 0, "duration_seconds": 4}],
                        "b": [{"start_seconds": 4, "duration_seconds": 4}],
                    },
                )

    def test_a_span_dict_mixing_both_shapes_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "needs exactly one of"):
            attribute_voices(
                song_audio(),
                voices={
                    "a": [{"start": 0, "end": 4, "start_seconds": 0}],
                    "b": [{"start_seconds": 4, "duration_seconds": 4}],
                },
            )

    def test_window_seconds_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "window_seconds"):
            attribute_voices(
                song_audio(),
                voices={
                    "a": [{"start_seconds": 0, "duration_seconds": 4}],
                    "b": [{"start_seconds": 4, "duration_seconds": 4}],
                },
                window_seconds=0,
            )

    def test_min_reference_seconds_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "min_reference_seconds"):
            attribute_voices(
                song_audio(),
                voices={
                    "a": [{"start_seconds": 0, "duration_seconds": 4}],
                    "b": [{"start_seconds": 4, "duration_seconds": 4}],
                },
                min_reference_seconds=0,
            )


# ---------------------------------------------------------------------------
# 2. Both `lines` shapes, chunks dict, text carried through
# ---------------------------------------------------------------------------


class TestLinesShapes(unittest.TestCase):
    def _run(self, lines):
        with (
            patch("dw.tasks.voice_attribution._load_embedder") as mock_load_embedder,
            patch("dw.tasks.voice_attribution.embed") as mock_embed,
        ):
            mock_load_embedder.return_value = MagicMock()

            # Fixed embedding regardless of input: the line embedding matches
            # 'a' exactly, so scoring is deterministic
            def fake_embed(encoder, samples):
                return numpy.array([1.0, 0.0])

            mock_embed.side_effect = fake_embed
            return attribute_voices(
                song_audio(seconds=10.0),
                voices={
                    "a": [{"start_seconds": 0, "duration_seconds": 4}],
                    "b": [{"start_seconds": 4, "duration_seconds": 4}],
                },
                lines=lines,
                separate=False,
            )

    def test_start_end_shape_and_start_seconds_shape_give_identical_spans(self):
        result_a = self._run([{"start": 1, "end": 3, "text": "hi"}])
        result_b = self._run(
            [{"start_seconds": 1, "duration_seconds": 2, "text": "hi"}]
        )
        self.assertEqual(result_a["lines"][0]["start"], result_b["lines"][0]["start"])
        self.assertEqual(result_a["lines"][0]["end"], result_b["lines"][0]["end"])

    def test_text_is_carried_through(self):
        result = self._run([{"start": 1, "end": 3, "text": "hello there"}])
        self.assertEqual(result["lines"][0]["text"], "hello there")

    def test_transcript_dict_with_chunks_is_accepted(self):
        transcript = {"text": "hello", "chunks": [{"start": 1, "end": 3, "text": "hi"}]}
        result = self._run(transcript)
        self.assertEqual(len(result["lines"]), 1)
        self.assertEqual(result["lines"][0]["text"], "hi")


# ---------------------------------------------------------------------------
# 3. Scoring
# ---------------------------------------------------------------------------


class TestScoring(unittest.TestCase):
    def test_cosine_of_identical_vectors_is_one(self):
        self.assertAlmostEqual(cosine([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_cosine_of_orthogonal_vectors_is_zero(self):
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_cosine_zero_vector_is_zero_not_a_division_error(self):
        self.assertEqual(cosine([0.0, 0.0], [1.0, 0.0]), 0.0)

    def test_reference_similarity_reports_every_pair(self):
        embeddings = {
            "a": numpy.array([1.0, 0.0]),
            "b": numpy.array([0.0, 1.0]),
            "c": numpy.array([1.0, 0.0]),
        }
        pairs = reference_similarity(embeddings)
        self.assertEqual(len(pairs), 3)
        too_similar = {frozenset(p["voices"]) for p in pairs if p["too_similar"]}
        self.assertEqual(too_similar, {frozenset(["a", "c"])})

    def test_score_line_picks_the_argmax_and_reports_every_voice(self):
        references = {
            "a": numpy.array([1.0, 0.0]),
            "b": numpy.array([0.0, 1.0]),
        }
        result = score_line(numpy.array([0.9, 0.1]), references, voiced_seconds=2.0)
        self.assertEqual(result["voice"], "a")
        self.assertEqual(set(result["scores"]), {"a", "b"})
        self.assertAlmostEqual(
            result["margin"], result["scores"]["a"] - result["scores"]["b"], places=3
        )

    def test_roll_up_shares_and_voice_directly(self):
        windows = [{"name": "w1", "start": 0.0, "end": 4.0}]
        lines = [
            {"start": 0.0, "end": 2.0, "voice": "a"},
            {"start": 2.0, "end": 4.0, "voice": "b"},
        ]

        def voiced_seconds(start, end):
            return end - start

        rolled = roll_up(windows, lines, voiced_seconds, ["a", "b"])
        self.assertEqual(rolled[0]["share"], {"a": 0.5, "b": 0.5})

    def test_attribute_voices_end_to_end_scores_each_voice(self):
        with (
            patch("dw.tasks.voice_attribution._load_embedder") as mock_load_embedder,
            patch("dw.tasks.voice_attribution.embed") as mock_embed,
        ):
            mock_load_embedder.return_value = MagicMock()
            mock_embed.return_value = numpy.array([1.0, 0.0])
            result = attribute_voices(
                song_audio(seconds=8.0),
                voices={
                    "a": [{"start_seconds": 0, "duration_seconds": 4}],
                    "b": [{"start_seconds": 4, "duration_seconds": 4}],
                },
                lines=[{"start": 0, "end": 4, "text": None}],
                separate=False,
            )
        line = result["lines"][0]
        self.assertEqual(set(line["scores"]), {"a", "b"})
        self.assertEqual(line["voice"], "a")


# ---------------------------------------------------------------------------
# 4. Uncertain paths
# ---------------------------------------------------------------------------


class TestUncertainPaths(unittest.TestCase):
    def test_a_thin_margin_is_uncertain_but_still_reports_the_argmax(self):
        references = {
            "a": numpy.array([1.0, 0.001]),
            "b": numpy.array([0.999, 0.0]),
        }
        result = score_line(numpy.array([1.0, 0.0005]), references, voiced_seconds=2.0)
        self.assertLess(result["margin"], UNCERTAIN_MARGIN)
        self.assertTrue(result["uncertain"])
        self.assertIn("margin", result["reason"])
        self.assertIsNotNone(result["voice"])

    def test_a_silent_line_has_no_voice_and_empty_scores(self):
        result = score_line(None, {"a": numpy.array([1.0, 0.0])}, voiced_seconds=0.1)
        self.assertIsNone(result["voice"])
        self.assertTrue(result["uncertain"])
        self.assertIn("voiced", result["reason"])
        self.assertEqual(result["scores"], {})
        self.assertLess(0.1, MIN_VOICED_SECONDS)
        self.assertIn("after separation", result["reason"])

    def test_an_unseparated_silent_line_does_not_claim_separation(self):
        result = score_line(
            None, {"a": numpy.array([1.0, 0.0])}, voiced_seconds=0.1, separated=False
        )
        self.assertIsNone(result["voice"])
        self.assertNotIn("separation", result["reason"])

    def test_a_too_similar_top_two_pair_is_uncertain(self):
        references = {
            "a": numpy.array([1.0, 0.0]),
            "b": numpy.array([1.0, 0.0]),
        }
        result = score_line(
            numpy.array([1.0, 0.0]),
            references,
            voiced_seconds=2.0,
            too_similar_pairs=[["a", "b"]],
        )
        self.assertTrue(result["uncertain"])
        self.assertIn("too similar", result["reason"])


# ---------------------------------------------------------------------------
# 4b. Transcript lines that are zero-length or hold two voices (#617)
# ---------------------------------------------------------------------------


def duet_audio():
    """4 s of a quiet tone ('a') then 4 s of a loud one ('b'). Both sit well
    above the relative voiced floor, so all 8 s is voiced."""
    quiet = tone(4.0, amplitude=0.1)
    loud = tone(4.0, amplitude=0.6)
    return FakeAudio(numpy.concatenate([quiet, loud], axis=1))


def loudness_embed(encoder, samples):
    """'b' for any stretch holding the loud tone, else 'a' - so a whole-line
    embedding over both singers names 'b' confidently, as ECAPA did on the
    tester's duet."""
    if len(samples) and float(numpy.max(numpy.abs(samples))) > 0.3:
        return numpy.array([0.0, 1.0])
    return numpy.array([1.0, 0.0])


class TestTranscriptLines(unittest.TestCase):
    def _run(self, audio, lines):
        with (
            patch("dw.tasks.voice_attribution._load_embedder") as mock_load_embedder,
            patch("dw.tasks.voice_attribution.embed") as mock_embed,
        ):
            mock_load_embedder.return_value = MagicMock()
            mock_embed.side_effect = loudness_embed
            return attribute_voices(
                audio,
                voices={
                    "a": [{"start_seconds": 0, "duration_seconds": 4}],
                    "b": [{"start_seconds": 4, "duration_seconds": 4}],
                },
                lines=lines,
                separate=False,
            )

    def test_a_zero_length_word_chunk_has_no_voice_rather_than_refusing(self):
        # Whisper word timestamps on the tester's duet: chunks [8] and [9]
        # both {start 3.96, end 3.96} refused the whole template run
        transcript = {
            "text": "la la la",
            "chunks": [
                {"start": 1.0, "end": 2.0, "text": "la"},
                {"start": 3.96, "end": 3.96, "text": "la"},
                {"start": 5.0, "end": 6.5, "text": "la"},
            ],
        }
        result = self._run(duet_audio(), transcript)
        lines = result["lines"]
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0]["voice"], "a")
        self.assertIsNone(lines[1]["voice"])
        self.assertTrue(lines[1]["uncertain"])
        self.assertIn("voiced", lines[1]["reason"])
        self.assertEqual((lines[1]["start"], lines[1]["end"]), (3.96, 3.96))
        self.assertEqual(lines[2]["voice"], "b")

    def test_a_zero_length_chunk_at_the_very_end_is_accepted(self):
        result = self._run(duet_audio(), [{"start": 8.0, "end": 8.0}])
        self.assertIsNone(result["lines"][0]["voice"])

    def test_a_line_ending_before_it_starts_is_still_refused(self):
        with self.assertRaisesRegex(ValueError, "lines\\[0\\] ends at 2.0 s"):
            self._run(duet_audio(), [{"start": 3.0, "end": 2.0}])

    def test_a_zero_length_window_is_still_refused(self):
        with (
            patch("dw.tasks.voice_attribution._load_embedder"),
            self.assertRaisesRegex(ValueError, "windows\\[0\\] .*not after"),
        ):
            attribute_voices(
                duet_audio(),
                voices={
                    "a": [{"start_seconds": 0, "duration_seconds": 4}],
                    "b": [{"start_seconds": 4, "duration_seconds": 4}],
                },
                lines=[{"start": 0, "end": 2}],
                windows=[{"name": "w", "start": 2, "end": 2}],
                separate=False,
            )

    def test_one_segment_over_both_singers_is_uncertain_and_names_the_split(self):
        # The tester's segment-mode run: one chunk over the whole duet came
        # back as one confident voice. Its pieces disagree, so it is uncertain
        result = self._run(duet_audio(), [{"start": 0.0, "end": 8.0}])
        line = result["lines"][0]
        self.assertEqual(line["voice"], "b")  # the whole-line argmax, still reported
        self.assertTrue(line["uncertain"])
        self.assertIn("more than one voice", line["reason"])
        self.assertIn("'a' 4.00 s", line["reason"])
        self.assertIn("'b' 4.00 s", line["reason"])
        self.assertEqual(result["thresholds"]["mixed_line_share"], MIXED_LINE_SHARE)

    def test_a_long_line_by_one_singer_stays_confident(self):
        result = self._run(duet_audio(), [{"start": 0.0, "end": 4.0}])
        line = result["lines"][0]
        self.assertEqual(line["voice"], "a")
        self.assertFalse(line["uncertain"])
        self.assertIsNone(line["reason"])

    def test_a_short_line_is_not_split(self):
        # Inside 2 s there is one piece: nothing to disagree with
        result = self._run(duet_audio(), [{"start": 3.0, "end": 5.0}])
        self.assertFalse(result["lines"][0]["uncertain"])

    def test_mixed_reason_needs_the_second_voice_at_the_share(self):
        self.assertIsNone(mixed_reason({"a": 9.0, "b": 1.0}))
        self.assertIsNone(mixed_reason({"a": 4.0}))
        self.assertIsNone(mixed_reason({}))
        self.assertIsNotNone(mixed_reason({"a": 6.0, "b": 2.0}))


# ---------------------------------------------------------------------------
# 5. voices_too_similar warnings
# ---------------------------------------------------------------------------


class TestVoicesTooSimilarWarning(unittest.TestCase):
    def test_too_similar_references_emit_a_warning(self):
        with (
            patch("dw.tasks.voice_attribution._load_embedder") as mock_load_embedder,
            patch("dw.tasks.voice_attribution.embed") as mock_embed,
            patch("dw.tasks.voice_attribution.emit_warning") as mock_emit_warning,
        ):
            mock_load_embedder.return_value = MagicMock()
            # Both voices' references embed identically -> cosine 1.0 > 0.8
            mock_embed.return_value = numpy.array([1.0, 0.0])
            result = attribute_voices(
                song_audio(seconds=8.0),
                voices={
                    "a": [{"start_seconds": 0, "duration_seconds": 4}],
                    "b": [{"start_seconds": 4, "duration_seconds": 4}],
                },
                lines=[{"start": 0, "end": 4}],
                separate=False,
            )

        kinds = [w["kind"] for w in result["warnings"]]
        self.assertIn("voices_too_similar", kinds)
        self.assertTrue(
            any(
                call.kwargs.get("kind") == "voices_too_similar"
                for call in mock_emit_warning.call_args_list
            )
        )
        self.assertGreater(VOICES_TOO_SIMILAR, 0)


# ---------------------------------------------------------------------------
# 6. Windows roll-up and defaults
# ---------------------------------------------------------------------------


class TestWindowsRollUp(unittest.TestCase):
    def test_split_window_under_share_margin_is_uncertain(self):
        windows = [{"name": "w1", "start": 0.0, "end": 4.0}]
        lines = [
            {"start": 0.0, "end": 2.1, "voice": "a"},
            {"start": 2.1, "end": 4.0, "voice": "b"},
        ]

        def voiced_seconds(start, end):
            return end - start

        rolled = roll_up(windows, lines, voiced_seconds, ["a", "b"])
        lead = abs(rolled[0]["share"]["a"] - rolled[0]["share"]["b"])
        self.assertLess(lead, UNCERTAIN_SHARE_MARGIN)
        self.assertTrue(rolled[0]["uncertain"])

    def test_window_with_no_attributed_voice_has_voice_none(self):
        windows = [{"name": "w1", "start": 0.0, "end": 4.0}]
        lines = [{"start": 0.0, "end": 4.0, "voice": None}]

        def voiced_seconds(start, end):
            return end - start

        rolled = roll_up(windows, lines, voiced_seconds, ["a", "b"])
        self.assertIsNone(rolled[0]["voice"])
        self.assertTrue(rolled[0]["uncertain"])

    def test_no_lines_no_windows_mirrors_the_fixed_grid(self):
        with (
            patch("dw.tasks.voice_attribution._load_embedder") as mock_load_embedder,
            patch("dw.tasks.voice_attribution.embed") as mock_embed,
        ):
            mock_load_embedder.return_value = MagicMock()
            mock_embed.return_value = numpy.array([1.0, 0.0])
            result = attribute_voices(
                song_audio(seconds=10.0),
                voices={
                    "a": [{"start_seconds": 0, "duration_seconds": 4}],
                    "b": [{"start_seconds": 4, "duration_seconds": 4}],
                },
                window_seconds=4.0,
                separate=False,
            )
        # 10s / 4s windows -> 3 windows, the last one shorter (2s)
        self.assertEqual(len(result["windows"]), 3)
        self.assertEqual(len(result["lines"]), 3)
        self.assertAlmostEqual(
            result["windows"][-1]["end"] - result["windows"][-1]["start"], 2.0
        )

    def test_lines_given_no_windows_gives_empty_windows(self):
        with (
            patch("dw.tasks.voice_attribution._load_embedder") as mock_load_embedder,
            patch("dw.tasks.voice_attribution.embed") as mock_embed,
        ):
            mock_load_embedder.return_value = MagicMock()
            mock_embed.return_value = numpy.array([1.0, 0.0])
            result = attribute_voices(
                song_audio(seconds=8.0),
                voices={
                    "a": [{"start_seconds": 0, "duration_seconds": 4}],
                    "b": [{"start_seconds": 4, "duration_seconds": 4}],
                },
                lines=[{"start": 0, "end": 4}],
                separate=False,
            )
        self.assertEqual(result["windows"], [])


# ---------------------------------------------------------------------------
# 7. Clip-path reference
# ---------------------------------------------------------------------------


class TestClipPathReference(unittest.TestCase):
    def test_clip_path_reference_is_loaded_and_length_checked(self):
        with (
            patch("dw.tasks.voice_attribution.load_audio") as mock_load_audio,
            patch("dw.tasks.voice_attribution._load_embedder") as mock_load_embedder,
            patch("dw.tasks.voice_attribution.embed") as mock_embed,
        ):
            mock_load_audio.return_value = (tone(4.0), 16000)
            mock_load_embedder.return_value = MagicMock()
            mock_embed.return_value = numpy.array([1.0, 0.0])
            attribute_voices(
                song_audio(seconds=8.0),
                voices={
                    "a": "asset:voices/a.wav",
                    "b": [{"start_seconds": 4, "duration_seconds": 4}],
                },
                lines=[{"start": 0, "end": 4}],
                separate=False,
            )
        mock_load_audio.assert_called_once_with("asset:voices/a.wav")

    def test_clip_path_reference_under_minimum_is_refused(self):
        with patch("dw.tasks.voice_attribution.load_audio") as mock_load_audio:
            mock_load_audio.return_value = (tone(1.0), 16000)
            with self.assertRaisesRegex(ValueError, "voice 'a' has"):
                attribute_voices(
                    song_audio(seconds=8.0),
                    voices={
                        "a": "asset:voices/a.wav",
                        "b": [{"start_seconds": 4, "duration_seconds": 4}],
                    },
                    lines=[{"start": 0, "end": 4}],
                    separate=False,
                )


# ---------------------------------------------------------------------------
# 8. MPS fallback
# ---------------------------------------------------------------------------


class TestSeparationMpsFallback(unittest.TestCase):
    def test_mps_retries_on_cpu_and_warns(self):
        from dw.tasks.voice_attribution import separate_vocals

        calls = []

        def fake_run_separator(waveform, sample_rate, device, dtype):
            calls.append(device)
            if len(calls) == 1:
                raise RuntimeError("boom")
            return tone(1.0, sample_rate=16000)[0], 16000

        with (
            patch(
                "dw.tasks.voice_attribution._run_separator",
                side_effect=fake_run_separator,
            ),
            patch("dw.get_device_type", return_value="mps"),
            patch("dw.tasks.voice_attribution.emit_warning") as mock_emit_warning,
        ):
            separate_vocals(tone(1.0, channels=2), 16000, "mps", None)

        self.assertEqual(calls, ["mps", "cpu"])
        self.assertTrue(
            any(
                call.kwargs.get("kind") == "separation_cpu_fallback"
                for call in mock_emit_warning.call_args_list
            )
        )

    def test_non_mps_device_propagates_the_error(self):
        from dw.tasks.voice_attribution import separate_vocals

        with (
            patch(
                "dw.tasks.voice_attribution._run_separator",
                side_effect=RuntimeError("boom"),
            ),
            patch("dw.get_device_type", return_value="cuda"),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                separate_vocals(tone(1.0, channels=2), 16000, "cuda", None)


# ---------------------------------------------------------------------------
# 9. Missing package
# ---------------------------------------------------------------------------


class TestMissingPackages(unittest.TestCase):
    def test_missing_demucs_raises_runtime_error_naming_it(self):
        from dw.tasks.voice_attribution import _load_separator

        with patch.dict(sys.modules, {"demucs": None, "demucs.pretrained": None}):
            with self.assertRaisesRegex(RuntimeError, "demucs"):
                _load_separator("cpu", None)

    def test_missing_speechbrain_raises_runtime_error_naming_it(self):
        from dw.tasks.voice_attribution import _load_embedder

        with patch.dict(
            sys.modules,
            {
                "speechbrain": None,
                "speechbrain.inference": None,
                "speechbrain.inference.speaker": None,
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "speechbrain"):
                _load_embedder("cpu", None)


# ---------------------------------------------------------------------------
# 10. Registration
# ---------------------------------------------------------------------------


class TestRegistration(unittest.TestCase):
    def test_assessment_is_still_exactly_the_three_probes(self):
        self.assertEqual(
            list_tasks()["assessment"],
            sorted(["analyze_shots", "analyze_seams", "analyze_sync_drift"]),
        )

    def test_attribute_voices_is_a_command_but_not_an_assessment_probe(self):
        tasks = list_tasks()
        self.assertIn("attribute_voices", tasks["commands"])
        self.assertNotIn("attribute_voices", tasks["assessment"])

    def test_attribute_voices_returns_json_with_no_assessment_flag(self):
        info = _COMMAND_INFO["attribute_voices"]
        self.assertEqual(info["returns"], "json")
        self.assertNotIn("assessment", info)

    def test_argument_schema_lists_the_documented_arguments_only(self):
        schema = describe_task("attribute_voices")
        names = {p["name"] for p in schema["parameters"]}
        for expected in (
            "audio",
            "voices",
            "lines",
            "windows",
            "window_seconds",
            "min_reference_seconds",
            "separate",
        ):
            self.assertIn(expected, names)
        # The models are fixed - nothing the caller writes reaches a loader
        self.assertFalse(any("model" in name for name in names))


# ---------------------------------------------------------------------------
# 11. Step validation: result.content_type must be application/json
# ---------------------------------------------------------------------------


class TestStepValidation(unittest.TestCase):
    def _step(self, content_type):
        return {
            "name": "voices",
            "task": {
                "command": "attribute_voices",
                "arguments": {
                    "audio": "asset:song.wav",
                    "voices": {"a": "asset:a.wav", "b": "asset:b.wav"},
                },
            },
            "result": {"content_type": content_type},
        }

    def test_application_json_is_fine(self):
        definition = {"steps": [self._step("application/json")]}
        errors = scalar_result_errors(definition, source_indices=[0])
        self.assertEqual(errors, [])

    def test_a_non_json_content_type_is_refused(self):
        definition = {"steps": [self._step("text/plain")]}
        errors = scalar_result_errors(definition, source_indices=[0])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["path"], "steps[0].result")
        self.assertIn("application/json", errors[0]["message"])
        self.assertIn("attribute_voices", errors[0]["message"])

    def test_through_validation_errors_on_a_full_workflow(self):
        workflow = Workflow(
            {"id": "attribute", "steps": [self._step("image/png")]},
            "outputs",
            None,
        )
        errors = workflow.validation_errors()
        messages = [e["message"] for e in errors if e["path"] == "steps[0].result"]
        self.assertTrue(messages)
        self.assertTrue(any("application/json" in m for m in messages))


# ---------------------------------------------------------------------------
# Bounce 1 (#494): a floor relative to the stem, voices checked statically
# ---------------------------------------------------------------------------


class TestRelativeVoicedFloor(unittest.TestCase):
    """C-F138: an absolute -40 dBFS floor dropped a quiet sung section."""

    def test_a_quiet_section_under_a_loud_one_is_voiced(self):
        # -20 dBFS and ~-43 dBFS sine sections: the quiet one sat under the
        # old absolute floor, and is 23 dB under the loud one
        loud = tone(4.0, amplitude=0.14)
        quiet = tone(4.0, amplitude=0.01)
        waveform = numpy.concatenate([quiet, loud, silence(2.0)], axis=1)[0]

        mask, frame, floor = voiced_mask(waveform, 16000)

        quiet_frames = int(4.0 * 16000) // frame
        self.assertTrue(mask[:quiet_frames].all())
        self.assertFalse(mask[-(int(2.0 * 16000) // frame) :].any())
        self.assertLess(floor, -40.0)

    def test_the_floor_follows_the_stem_level(self):
        loud = numpy.full(100, 0.5)
        quiet = numpy.full(100, 0.005)
        self.assertGreater(voiced_floor_dbfs(loud), voiced_floor_dbfs(quiet))

    def test_the_floor_never_drops_below_its_minimum(self):
        self.assertEqual(voiced_floor_dbfs(numpy.full(100, 1e-7)), -60.0)
        self.assertEqual(voiced_floor_dbfs(numpy.zeros(0)), -60.0)

    def test_silence_is_never_voiced(self):
        mask, _, _ = voiced_mask(silence(2.0)[0], 16000)
        self.assertFalse(mask.any())


def _voices_step(voices, **extra):
    return {
        "name": "who",
        "task": {
            "command": "attribute_voices",
            "arguments": {"audio": "asset:song.wav", "voices": voices, **extra},
        },
        "result": {"content_type": "application/json"},
    }


SPAN = {"start_seconds": 0, "duration_seconds": 4}


class TestVoicesErrors(unittest.TestCase):
    """C-F141: what parse_voices refuses without the audio is refused at
    validation, at the path the author wrote - through attribute_voices'
    registered static check (#773)."""

    def errors(self, *steps, source_indices=None):
        return task_argument_errors({"steps": list(steps)}, source_indices)

    def test_a_good_voices_map_is_clean(self):
        self.assertEqual(self.errors(_voices_step({"a": SPAN, "b": [SPAN]})), [])

    def test_one_voice_is_refused(self):
        errors = self.errors(_voices_step({"a": SPAN}))
        self.assertEqual(errors[0]["path"], "steps[0].task.arguments.voices")
        self.assertIn("at least 2 voices", errors[0]["message"])

    def test_a_bad_name_is_refused(self):
        errors = self.errors(_voices_step({"1bad": SPAN, "b": SPAN}))
        self.assertIn("1bad", errors[0]["message"])

    def test_a_short_reference_is_refused_against_the_literal_minimum(self):
        short = {"start_seconds": 0, "duration_seconds": 2}
        self.assertTrue(self.errors(_voices_step({"a": short, "b": SPAN})))
        self.assertEqual(
            self.errors(_voices_step({"a": short, "b": SPAN}, min_reference_seconds=1)),
            [],
        )

    def test_a_span_past_the_end_is_left_to_the_run(self):
        late = {"start_seconds": 9000, "duration_seconds": 4}
        self.assertEqual(self.errors(_voices_step({"a": late, "b": SPAN})), [])

    def test_references_are_left_to_the_run(self):
        self.assertEqual(self.errors(_voices_step("variable:voices")), [])
        self.assertEqual(
            self.errors(_voices_step({"a": "asset:a.wav", "b": "asset:b.wav"})), []
        )
        held = {"start_seconds": "variable:start", "duration_seconds": 4}
        self.assertEqual(self.errors(_voices_step({"a": held, "b": SPAN})), [])

    def test_a_member_is_reported_at_its_source_step(self):
        step = _voices_step({"a": SPAN})
        step["name"] = "who@verse"
        errors = self.errors({"name": "x", "task": {}}, step, source_indices=[0, 0])
        self.assertEqual(errors[0]["path"], "steps[0].task.arguments.voices")
        self.assertIn("in member 'who@verse'", errors[0]["message"])

    def test_through_validation_errors(self):
        workflow = Workflow(
            {"id": "attribute", "steps": [_voices_step({"a": SPAN})]},
            "outputs",
            None,
        )
        paths = [e["path"] for e in workflow.validation_errors()]
        self.assertIn("steps[0].task.arguments.voices", paths)


class TestVoicePathsAreLocations(unittest.TestCase):
    """SE-F038: a bare path as a voice is a location like `audio` is, in
    the posture a server runs on (workflow files untrusted)."""

    def setUp(self):
        patcher = patch.dict("os.environ", {TRUST_WORKFLOWS_ENV_VAR: "0"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_an_absolute_voice_path_is_refused(self):
        definition = {
            "steps": [
                _voices_step(
                    {"a": "/usr/share/sounds/alsa/Front_Center.wav", "b": SPAN}
                )
            ]
        }
        errors = location_errors(definition, [0], base_dir="/tmp/wf")
        self.assertEqual(
            [e["path"] for e in errors], ["steps[0].task.arguments.voices.a"]
        )

    def test_a_traversing_voice_path_is_refused(self):
        definition = {
            "steps": [_voices_step({"a": "../../../../../etc/x.wav", "b": SPAN})]
        }
        errors = location_errors(definition, [0], base_dir="/tmp/wf")
        self.assertEqual(
            [e["path"] for e in errors], ["steps[0].task.arguments.voices.a"]
        )

    def test_asset_voices_and_spans_are_not_locations(self):
        definition = {"steps": [_voices_step({"a": "asset:a.wav", "b": SPAN})]}
        self.assertEqual(location_errors(definition, [0], base_dir="/tmp/wf"), [])


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# The attribute-lines template: transcribe_audio -> attribute_voices (#488)
# ---------------------------------------------------------------------------


class TestAttributeLinesTemplate(unittest.TestCase):
    TEMPLATE = "workflows/templates/attribute-lines.json"

    def _workflow(self, output_dir):
        import json

        with open(self.TEMPLATE) as f:
            return Workflow(json.load(f), output_dir, "")

    def test_the_template_validates_and_needs_its_voices(self):
        import tempfile

        with tempfile.TemporaryDirectory() as output_dir:
            workflow = self._workflow(output_dir)
            self.assertEqual(workflow.validation_errors(), [])
            errors = workflow.validation_errors({"audio": "asset:song.wav"})
            self.assertEqual(
                [error["path"] for error in errors],
                ["steps[1].task.arguments.voices"],
            )

    def test_the_transcript_reaches_attribute_voices_as_its_lines(self):
        """The whole {text, chunks} transcript is attribute_voices' lines, and
        a last chunk Whisper left open-ended (end None) is attributed with the
        clip's duration as its end rather than refusing the transcript."""
        import tempfile

        pipe = MagicMock(
            return_value={
                "text": "one two",
                "chunks": [
                    {"timestamp": (0.0, 4.0), "text": " one"},
                    {"timestamp": (4.0, None), "text": " two"},
                ],
            }
        )
        pipe.type = "seq2seq_whisper"
        model_cache.clear_model_cache()
        with (
            tempfile.TemporaryDirectory() as output_dir,
            patch("dw.tasks.audio_transcription.hf_pipeline", return_value=pipe),
            patch("dw.tasks.voice_attribution._load_embedder"),
            patch(
                "dw.tasks.voice_attribution.embed",
                return_value=numpy.array([1.0, 0.0]),
            ),
            patch(
                "dw.tasks.voice_attribution.separate_vocals",
                side_effect=lambda mix, rate, device, dtype: mix.mean(axis=0),
            ),
        ):
            result = self._workflow(output_dir).run(
                {
                    "audio": song_audio(10.0),
                    "voices": {
                        "a": [{"start_seconds": 0.0, "duration_seconds": 3.0}],
                        "b": [{"start_seconds": 5.0, "duration_seconds": 3.0}],
                    },
                }
            )
        model_cache.clear_model_cache()

        self.assertEqual(len(result), 1)
        lines = result[0]["lines"]
        self.assertEqual([line["text"] for line in lines], ["one", "two"])
        self.assertEqual(
            [(line["start"], line["end"]) for line in lines],
            [(0.0, 4.0), (4.0, 10.0)],
        )
