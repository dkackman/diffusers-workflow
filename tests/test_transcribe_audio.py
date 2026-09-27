"""Tests for transcribe_audio task."""

import unittest
from unittest.mock import patch, MagicMock

import numpy

from dw.tasks.audio_transcription import (
    transcribe_audio,
    _DEFAULT_ASR_MODEL,
    _ASR_SAMPLE_RATE,
)


class TestTranscribeAudio(unittest.TestCase):
    """Tests for the transcribe_audio function.

    hf_pipeline is a module-level import in audio_transcription.py (like
    text_generation.py's), so it is patched as that module's attribute.
    """

    def _waveform(self, channels=1, seconds=1.0, sample_rate=16000):
        samples = int(seconds * sample_rate)
        tone = numpy.zeros((channels, samples), dtype=numpy.float32)
        return tone, sample_rate

    def _mock_pipe(
        self, mock_pipeline, text="hello world", kind="seq2seq_whisper", chunks=None
    ):
        pipe = MagicMock()
        pipe.type = kind
        result = {"text": text}
        if chunks is not None:
            result["chunks"] = chunks
        pipe.return_value = result
        mock_pipeline.return_value = pipe
        return pipe

    @patch("dw.tasks.audio_transcription.hf_pipeline")
    def test_returns_transcript_string(self, mock_pipeline):
        self._mock_pipe(mock_pipeline, "testing 1 2 3")
        waveform, rate = self._waveform()

        result = transcribe_audio(waveform, device="cpu", sample_rate=rate)

        self.assertEqual(result, "testing 1 2 3")
        mock_pipeline.assert_called_once()

    @patch("dw.tasks.audio_transcription.hf_pipeline")
    def test_builds_an_asr_pipeline_with_default_model(self, mock_pipeline):
        self._mock_pipe(mock_pipeline)
        waveform, rate = self._waveform()

        transcribe_audio(waveform, device="cpu", sample_rate=rate)

        self.assertEqual(mock_pipeline.call_args[0][0], "automatic-speech-recognition")
        self.assertEqual(mock_pipeline.call_args[1]["model"], _DEFAULT_ASR_MODEL)

    @patch("dw.tasks.audio_transcription.hf_pipeline")
    def test_custom_model_name(self, mock_pipeline):
        self._mock_pipe(mock_pipeline)
        waveform, rate = self._waveform()

        transcribe_audio(
            waveform, device="cpu", sample_rate=rate, model_name="openai/whisper-tiny"
        )

        self.assertEqual(mock_pipeline.call_args[1]["model"], "openai/whisper-tiny")

    @patch("dw.tasks.audio_transcription.hf_pipeline")
    def test_downmixes_stereo_to_mono(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)
        waveform, rate = self._waveform(channels=2)

        transcribe_audio(waveform, device="cpu", sample_rate=rate)

        fed = pipe.call_args[0][0]
        self.assertEqual(fed["raw"].ndim, 1)

    @patch("dw.tasks.audio_transcription.hf_pipeline")
    def test_a_clip_over_thirty_seconds_asks_for_timestamps(self, mock_pipeline):
        # Whisper refuses more than 30 s of audio ("more than 3000 mel input
        # features") unless it predicts timestamps - so a 110 s song failed
        pipe = self._mock_pipe(mock_pipeline)
        waveform, rate = self._waveform(seconds=31.0)

        transcribe_audio(waveform, device="cpu", sample_rate=rate)

        self.assertIs(pipe.call_args.kwargs.get("return_timestamps"), True)

    @patch("dw.tasks.audio_transcription.hf_pipeline")
    def test_a_long_clip_on_a_ctc_model_does_not_ask_for_timestamps(
        self, mock_pipeline
    ):
        # transformers raises for a CTC model unless return_timestamps is
        # "char" or "word"; a CTC model has no 30 s window to stitch anyway
        pipe = self._mock_pipe(mock_pipeline, kind="ctc")
        waveform, rate = self._waveform(seconds=31.0)

        transcribe_audio(waveform, device="cpu", sample_rate=rate)

        self.assertNotIn("return_timestamps", pipe.call_args.kwargs)

    @patch("dw.tasks.audio_transcription.hf_pipeline")
    def test_a_short_clip_does_not_ask_for_timestamps(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)
        waveform, rate = self._waveform(seconds=5.0)

        transcribe_audio(waveform, device="cpu", sample_rate=rate)

        self.assertNotIn("return_timestamps", pipe.call_args.kwargs)

    @patch("dw.tasks.audio_transcription.hf_pipeline")
    def test_resamples_to_16khz(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)
        waveform, rate = self._waveform(sample_rate=44100)

        transcribe_audio(waveform, device="cpu", sample_rate=rate)

        fed = pipe.call_args[0][0]
        self.assertEqual(fed["sampling_rate"], _ASR_SAMPLE_RATE)

    @patch("dw.tasks.audio_transcription.hf_pipeline")
    def test_strips_whitespace(self, mock_pipeline):
        self._mock_pipe(mock_pipeline, "  spaced out  ")
        waveform, rate = self._waveform()

        result = transcribe_audio(waveform, device="cpu", sample_rate=rate)

        self.assertEqual(result, "spaced out")

    @patch("dw.tasks.audio_transcription.hf_pipeline")
    def test_timestamps_word_requests_word_level_and_returns_chunks(
        self, mock_pipeline
    ):
        pipe = self._mock_pipe(
            mock_pipeline,
            "hello world",
            chunks=[
                {"text": " hello", "timestamp": (0.0, 0.5)},
                {"text": " world", "timestamp": (0.5, 1.0)},
            ],
        )
        waveform, rate = self._waveform()

        result = transcribe_audio(
            waveform, device="cpu", sample_rate=rate, timestamps="word"
        )

        self.assertEqual(pipe.call_args.kwargs.get("return_timestamps"), "word")
        self.assertEqual(result["text"], "hello world")
        self.assertEqual(
            result["chunks"],
            [
                {"start": 0.0, "end": 0.5, "text": "hello"},
                {"start": 0.5, "end": 1.0, "text": "world"},
            ],
        )

    @patch("dw.tasks.audio_transcription.hf_pipeline")
    def test_timestamps_segment_requests_true_and_returns_chunks(self, mock_pipeline):
        pipe = self._mock_pipe(
            mock_pipeline,
            "hello world",
            chunks=[{"text": "hello world", "timestamp": (0.0, 1.0)}],
        )
        waveform, rate = self._waveform()

        result = transcribe_audio(
            waveform, device="cpu", sample_rate=rate, timestamps="segment"
        )

        self.assertIs(pipe.call_args.kwargs.get("return_timestamps"), True)
        self.assertEqual(
            result["chunks"], [{"start": 0.0, "end": 1.0, "text": "hello world"}]
        )

    @patch("dw.tasks.audio_transcription.hf_pipeline")
    def test_timestamps_requested_under_thirty_seconds_too(self, mock_pipeline):
        # The 30 s long-form branch is a separate reason to ask for
        # timestamps; an explicit request must not depend on clip length
        pipe = self._mock_pipe(mock_pipeline, chunks=[])
        waveform, rate = self._waveform(seconds=5.0)

        transcribe_audio(waveform, device="cpu", sample_rate=rate, timestamps="word")

        self.assertEqual(pipe.call_args.kwargs.get("return_timestamps"), "word")

    @patch("dw.tasks.audio_transcription.hf_pipeline")
    def test_timestamps_unset_still_returns_plain_text(self, mock_pipeline):
        self._mock_pipe(mock_pipeline, "unchanged")
        waveform, rate = self._waveform()

        result = transcribe_audio(waveform, device="cpu", sample_rate=rate)

        self.assertEqual(result, "unchanged")

    @patch("dw.tasks.audio_transcription.hf_pipeline")
    def test_invalid_timestamps_value_is_rejected(self, mock_pipeline):
        self._mock_pipe(mock_pipeline)
        waveform, rate = self._waveform()

        with self.assertRaises(ValueError):
            transcribe_audio(
                waveform, device="cpu", sample_rate=rate, timestamps="paragraph"
            )


class TestTranscribeAudioRegistration(unittest.TestCase):
    """Test that transcribe_audio is registered as a task command."""

    def test_command_registered(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        self.assertIn("transcribe_audio", _COMMAND_REGISTRY)


if __name__ == "__main__":
    unittest.main()
