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

    def _mock_pipe(self, mock_pipeline, text="hello world"):
        pipe = MagicMock()
        pipe.return_value = {"text": text}
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

        self.assertEqual(
            mock_pipeline.call_args[0][0], "automatic-speech-recognition"
        )
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


class TestTranscribeAudioRegistration(unittest.TestCase):
    """Test that transcribe_audio is registered as a task command."""

    def test_command_registered(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        self.assertIn("transcribe_audio", _COMMAND_REGISTRY)


if __name__ == "__main__":
    unittest.main()
