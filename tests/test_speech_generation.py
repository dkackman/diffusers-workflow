"""Tests for the speech_generation task."""

import unittest
from unittest.mock import patch, MagicMock

import numpy
import torch

from dw.tasks.speech_generation import (
    generate_speech,
    _DEFAULT_MODEL,
    _speaker_embedding_tensor,
)


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

    @patch("dw.tasks.speech_generation._speaker_embedding_tensor")
    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_speaker_embedding_reaches_forward_params_for_speecht5(
        self, mock_pipeline, mock_tensor
    ):
        pipe = MagicMock(return_value=spoken())
        pipe.model.config.model_type = "speecht5"
        mock_pipeline.return_value = pipe
        mock_tensor.return_value = "the-x-vector"

        generate_speech(
            "hi",
            device="cpu",
            model_name="microsoft/speecht5_tts",
            speaker_embedding="asset:voices/iris.wav",
        )

        mock_tensor.assert_called_once_with("asset:voices/iris.wav", "cpu")
        self.assertEqual(
            pipe.call_args[1]["forward_params"],
            {"speaker_embeddings": "the-x-vector"},
        )

    @patch("dw.tasks.speech_generation._speaker_embedding_tensor")
    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_speaker_embedding_merges_with_other_forward_params(
        self, mock_pipeline, mock_tensor
    ):
        pipe = MagicMock(return_value=spoken())
        pipe.model.config.model_type = "speecht5"
        mock_pipeline.return_value = pipe
        mock_tensor.return_value = "the-x-vector"

        generate_speech(
            "hi",
            device="cpu",
            model_name="microsoft/speecht5_tts",
            speaker_embedding="asset:voices/iris.wav",
            forward_params={"do_sample": True},
        )

        self.assertEqual(
            pipe.call_args[1]["forward_params"],
            {"do_sample": True, "speaker_embeddings": "the-x-vector"},
        )

    @patch("dw.tasks.speech_generation._speaker_embedding_tensor")
    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_speaker_embedding_on_a_non_speecht5_model_is_an_error(
        self, mock_pipeline, mock_tensor
    ):
        # Only SpeechT5 conditions on an x-vector; a VITS model would drop
        # 'speaker_embeddings' as an unrecognised forward kwarg and generate
        # in its own voice
        pipe = MagicMock(return_value=spoken())
        pipe.model.config.model_type = "vits"
        mock_pipeline.return_value = pipe

        with self.assertRaises(ValueError) as raised:
            generate_speech(
                "hi",
                model_name="facebook/mms-tts-vits",
                speaker_embedding="asset:voices/iris.wav",
            )

        self.assertIn("speaker_embedding", str(raised.exception))
        self.assertIn("facebook/mms-tts-vits", str(raised.exception))
        mock_tensor.assert_not_called()

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_vits_speaker_id_passes_through_forward_params_unchanged(
        self, mock_pipeline
    ):
        # The other half of speaker conditioning (#223): a VITS model's
        # speaker_id is a plain int forward kwarg that already reached the
        # model through forward_params before speaker_embedding existed, and
        # needs no code of its own - confirmed here by never touching
        # speaker_embedding and still seeing speaker_id pass through intact
        pipe = MagicMock(return_value=spoken())
        pipe.model.config.model_type = "vits"
        mock_pipeline.return_value = pipe

        generate_speech(
            "hi",
            device="cpu",
            model_name="facebook/mms-tts-vits",
            forward_params={"speaker_id": 3},
        )

        self.assertEqual(pipe.call_args[1]["forward_params"], {"speaker_id": 3})


class TestSpeakerEmbeddingTensor(unittest.TestCase):
    """#223 regression: SpeechT5's generate() rejects a bare (512,) vector -
    it wants (batch, 512). These exercise the real squeeze/unsqueeze logic
    rather than mocking it away, unlike the forward_params tests above."""

    @patch("dw.tasks.speech_generation.load_audio")
    @patch("dw.tasks.speech_generation.cached_model")
    def test_output_shape_is_batch_of_one_by_512(self, mock_cached_model, mock_load_audio):
        mock_load_audio.return_value = (numpy.zeros((1, 16000), dtype=numpy.float32), 16000)
        encoder = MagicMock()
        # speechbrain's raw encode_batch output: (1, 1, 512)
        encoder.encode_batch.return_value = torch.zeros((1, 1, 512))
        mock_cached_model.return_value = encoder

        # speechbrain isn't a hard dependency of the test env; the helper
        # imports it lazily, so stand in a fake module rather than requiring
        # the real package just to exercise the tensor-shape logic
        fake_module = MagicMock()
        fake_module.EncoderClassifier = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "speechbrain": MagicMock(),
                "speechbrain.inference": MagicMock(),
                "speechbrain.inference.speaker": fake_module,
            },
        ):
            embedding = _speaker_embedding_tensor("asset:voices/iris.wav", "cpu")

        self.assertEqual(tuple(embedding.shape), (1, 512))
        self.assertEqual(embedding.dtype, torch.float32)


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
