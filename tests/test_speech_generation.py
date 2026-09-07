"""Tests for the speech_generation task."""

import unittest
from unittest.mock import patch, MagicMock

import numpy

from dw.tasks.speech_generation import generate_speech, _DEFAULT_MODEL


def spoken(sample_rate=24000, samples=480):
    """What a HuggingFace text-to-audio pipeline hands back."""
    return {
        "audio": numpy.zeros((1, samples), dtype=numpy.float32),
        "sampling_rate": sample_rate,
    }


class TestGenerateSpeech(unittest.TestCase):
    """Tests for the generate_speech function."""

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_returns_the_waveform_channels_first(self, mock_pipeline):
        pipe = MagicMock(return_value=spoken(samples=480))
        mock_pipeline.return_value = pipe

        track = generate_speech("hello", device="cpu")

        self.assertEqual(track.audio.shape, (1, 480))

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_carries_the_rate_the_model_generated_at(self, mock_pipeline):
        # The whole point of returning a track rather than a bare waveform: every
        # TTS model has its own rate, and a workflow declaring the wrong one plays
        # the speech at the wrong speed without ever failing
        mock_pipeline.return_value = MagicMock(return_value=spoken(sample_rate=16000))

        track = generate_speech("hello", device="cpu")

        self.assertEqual(track.sample_rate, 16000)

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_uses_the_default_model(self, mock_pipeline):
        mock_pipeline.return_value = MagicMock(return_value=spoken())

        generate_speech("hello", device="cpu")

        self.assertEqual(mock_pipeline.call_args[1]["model"], _DEFAULT_MODEL)

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_custom_model_name(self, mock_pipeline):
        mock_pipeline.return_value = MagicMock(return_value=spoken())

        generate_speech("hello", device="cpu", model_name="facebook/mms-tts-eng")

        self.assertEqual(mock_pipeline.call_args[1]["model"], "facebook/mms-tts-eng")

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_the_text_to_speech_task_is_what_is_loaded(self, mock_pipeline):
        mock_pipeline.return_value = MagicMock(return_value=spoken())

        generate_speech("hello", device="cpu")

        self.assertEqual(mock_pipeline.call_args[0][0], "text-to-speech")

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_voice_preset_reaches_the_processor(self, mock_pipeline):
        # A preset is a preprocessing argument - it selects the speaker before
        # generation rather than parameterizing it. Passed as a forward argument
        # it would be silently dropped and every character would sound the same
        pipe = MagicMock(return_value=spoken())
        mock_pipeline.return_value = pipe

        generate_speech("hello", device="cpu", voice_preset="v2/en_speaker_6")

        self.assertEqual(
            pipe.call_args[1]["preprocess_params"], {"voice_preset": "v2/en_speaker_6"}
        )

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_no_preprocessing_arguments_without_a_preset(self, mock_pipeline):
        pipe = MagicMock(return_value=spoken())
        mock_pipeline.return_value = pipe

        generate_speech("hello", device="cpu")

        self.assertEqual(pipe.call_args[1]["preprocess_params"], {})

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_forward_params_are_passed_through(self, mock_pipeline):
        pipe = MagicMock(return_value=spoken())
        mock_pipeline.return_value = pipe

        generate_speech("hello", device="cpu", forward_params={"do_sample": True})

        self.assertEqual(pipe.call_args[1]["forward_params"], {"do_sample": True})

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_generate_kwargs_are_passed_through(self, mock_pipeline):
        pipe = MagicMock(return_value=spoken())
        mock_pipeline.return_value = pipe

        generate_speech("hello", device="cpu", generate_kwargs={"temperature": 0.7})

        self.assertEqual(pipe.call_args[1]["generate_kwargs"], {"temperature": 0.7})

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_a_batched_waveform_is_flattened_to_channels_and_samples(
        self, mock_pipeline
    ):
        # Bark returns (1, samples); as_channels_samples is what the rest of the
        # audio plumbing expects everything to arrive in
        mock_pipeline.return_value = MagicMock(
            return_value={
                "audio": numpy.zeros((1, 1, 200), dtype=numpy.float32),
                "sampling_rate": 24000,
            }
        )

        track = generate_speech("hello", device="cpu")

        self.assertEqual(track.audio.shape, (1, 200))

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_a_model_reporting_no_rate_is_an_error(self, mock_pipeline):
        # Better to say so than to save at a default rate that is quietly wrong
        mock_pipeline.return_value = MagicMock(
            return_value={"audio": numpy.zeros((1, 200), dtype=numpy.float32)}
        )

        with self.assertRaisesRegex(ValueError, "sample rate"):
            generate_speech("hello", device="cpu")

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_a_preset_the_model_cannot_take_is_an_error(self, mock_pipeline):
        # facebook/mms-tts-eng has no processor: transformers would log the
        # preset as an unrecognised kwarg and speak in its one voice
        mock_pipeline.return_value.processor = None

        with self.assertRaises(ValueError) as raised:
            generate_speech(
                "hi", model_name="facebook/mms-tts-eng", voice_preset="v2/en_speaker_6"
            )

        self.assertIn("voice_preset", str(raised.exception))
        self.assertIn("facebook/mms-tts-eng", str(raised.exception))


class TestHandleSpeechGeneration(unittest.TestCase):
    """Covers the task.py dispatch handler directly, since
    test_task_discovery.py only introspects the command registry rather than
    running handlers."""

    def test_generate_speech_without_text_names_the_argument(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        class FakeTask:
            def device_for(self, arguments):
                return "cpu"

        handler = _COMMAND_REGISTRY["generate_speech"]
        with self.assertRaisesRegex(ValueError, "generate_speech needs 'text'"):
            handler(FakeTask(), {"voice_preset": "v2/en_speaker_6"}, {})
