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

        mock_tensor.assert_called_once_with(
            "asset:voices/iris.wav", "cpu", torch.float32
        )
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
    def test_messages_are_passed_as_text_inputs_instead_of_text(self, mock_pipeline):
        # VibeVoice and other chat-templated models take a conversation rather
        # than a bare string; transformers applies the model's chat template
        # when text_inputs is a list of {role, content} dicts
        pipe = MagicMock(return_value=spoken())
        mock_pipeline.return_value = pipe
        messages = [{"role": "user", "content": "Say hello."}]

        generate_speech(device="cpu", messages=messages)

        self.assertEqual(pipe.call_args[0][0], messages)

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_both_text_and_messages_is_an_error(self, mock_pipeline):
        mock_pipeline.return_value = MagicMock(return_value=spoken())

        with self.assertRaisesRegex(ValueError, "exactly one of 'text' or 'messages'"):
            generate_speech(
                "hi", device="cpu", messages=[{"role": "user", "content": "hi"}]
            )

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_neither_text_nor_messages_is_an_error(self, mock_pipeline):
        mock_pipeline.return_value = MagicMock(return_value=spoken())

        with self.assertRaisesRegex(ValueError, "exactly one of 'text' or 'messages'"):
            generate_speech(device="cpu")

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_a_bare_string_messages_is_an_error(self, mock_pipeline):
        # A bare string is itself a valid iterable of characters and would
        # otherwise sail through the exactly-one check as "messages given"
        mock_pipeline.return_value = MagicMock(return_value=spoken())

        with self.assertRaisesRegex(ValueError, "non-empty list"):
            generate_speech(device="cpu", messages="hello")

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_an_empty_messages_list_is_an_error(self, mock_pipeline):
        mock_pipeline.return_value = MagicMock(return_value=spoken())

        with self.assertRaisesRegex(ValueError, "non-empty list"):
            generate_speech(device="cpu", messages=[])

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_a_wrong_keyed_message_dict_is_an_error(self, mock_pipeline):
        # Caught here rather than left to the tokenizer, whose error for this
        # shape names neither 'messages' nor what it expected instead
        mock_pipeline.return_value = MagicMock(return_value=spoken())

        with self.assertRaisesRegex(ValueError, "non-empty list"):
            generate_speech(device="cpu", messages=[{"speaker": "user", "text": "hi"}])

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_a_non_dict_message_item_is_an_error(self, mock_pipeline):
        mock_pipeline.return_value = MagicMock(return_value=spoken())

        with self.assertRaisesRegex(ValueError, "non-empty list"):
            generate_speech(device="cpu", messages=["hello"])

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_messages_that_pass_the_shape_guard_still_reach_the_pipeline(
        self, mock_pipeline
    ):
        # #233 only guards shape (role/content present and strings) - a role
        # value the guard cannot judge (not a recognised chat role) is left
        # to transformers' own Chat wrapper, and its error surfaces unchanged
        pipe = MagicMock(
            side_effect=ValueError(
                "Chat role 'narrator' is not one of 'system', 'user', 'assistant'"
            )
        )
        mock_pipeline.return_value = pipe

        with self.assertRaisesRegex(ValueError, "Chat role 'narrator'"):
            generate_speech(
                device="cpu", messages=[{"role": "narrator", "content": "hi"}]
            )

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_a_seed_seeds_the_global_rng_before_generating(self, mock_pipeline):
        # transformers' generate() takes no generator= kwarg, unlike a diffusers
        # pipeline - reproducing Bark means seeding torch's global RNG (#261)
        mock_pipeline.return_value = MagicMock(return_value=spoken())

        with patch("dw.tasks.speech_generation.torch.manual_seed") as mock_seed:
            generate_speech("hello", device="cpu", seed=7)

        mock_seed.assert_called_once_with(7)

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_no_seed_does_not_touch_the_global_rng(self, mock_pipeline):
        mock_pipeline.return_value = MagicMock(return_value=spoken())

        with patch("dw.tasks.speech_generation.torch.manual_seed") as mock_seed:
            generate_speech("hello", device="cpu")

        mock_seed.assert_not_called()

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
    def test_output_shape_is_batch_of_one_by_512(
        self, mock_cached_model, mock_load_audio
    ):
        mock_load_audio.return_value = (
            numpy.zeros((1, 16000), dtype=numpy.float32),
            16000,
        )
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
            embedding = _speaker_embedding_tensor(
                "asset:voices/iris.wav", "cpu", torch.float16
            )

        self.assertEqual(tuple(embedding.shape), (1, 512))
        self.assertEqual(embedding.dtype, torch.float16)


class TestHandleSpeechGeneration(unittest.TestCase):
    """Covers the model_handlers.py dispatch handler directly, since
    test_task_discovery.py only introspects the command registry rather than
    running handlers."""

    def test_generate_speech_without_text_names_the_argument(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        class FakeTask:
            def device_for(self, arguments):
                return "cpu"

            def seed_for(self, arguments):
                return arguments.pop("seed", None)

        handler = _COMMAND_REGISTRY["generate_speech"]
        with self.assertRaisesRegex(ValueError, "generate_speech needs exactly one of"):
            handler(FakeTask(), {"voice_preset": "v2/en_speaker_6"}, {})

    def test_generate_speech_with_both_text_and_messages_is_an_error(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        class FakeTask:
            def device_for(self, arguments):
                return "cpu"

            def seed_for(self, arguments):
                return arguments.pop("seed", None)

        handler = _COMMAND_REGISTRY["generate_speech"]
        with self.assertRaisesRegex(ValueError, "generate_speech needs exactly one of"):
            handler(
                FakeTask(),
                {"text": "hi", "messages": [{"role": "user", "content": "hi"}]},
                {},
            )

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_messages_dispatched_through_the_handler_reach_the_pipeline(
        self, mock_pipeline
    ):
        from dw.tasks.task import _COMMAND_REGISTRY

        class FakeTask:
            def device_for(self, arguments):
                return "cpu"

            def seed_for(self, arguments):
                return arguments.pop("seed", None)

        pipe = MagicMock(return_value=spoken())
        mock_pipeline.return_value = pipe

        handler = _COMMAND_REGISTRY["generate_speech"]
        messages = [{"role": "user", "content": "hi"}]
        handler(FakeTask(), {"messages": messages}, {})

        self.assertEqual(pipe.call_args[0][0], messages)

    @patch("dw.tasks.speech_generation.hf_pipeline")
    def test_seed_reaches_generate_speech_from_the_task(self, mock_pipeline):
        from dw.tasks.task import _COMMAND_REGISTRY

        class FakeTask:
            def device_for(self, arguments):
                return "cpu"

            def seed_for(self, arguments):
                return arguments.pop("seed", 7)

        pipe = MagicMock(return_value=spoken())
        mock_pipeline.return_value = pipe

        with patch("dw.tasks.speech_generation.torch.manual_seed") as mock_seed:
            handler = _COMMAND_REGISTRY["generate_speech"]
            handler(FakeTask(), {"text": "hi"}, {})

        mock_seed.assert_called_once_with(7)


# --- Audio8-TTS (#1121) -------------------------------------------------------

import json  # noqa: E402
import wave  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from dw.tasks import audio8_tts_model as a8  # noqa: E402
from dw.tasks import speech_generation as sg  # noqa: E402
from dw.tasks.model_cache import clear_model_cache  # noqa: E402
from tests.test_audio8_tts_model import TINY, legacy_codec_state  # noqa: E402

AUDIO8 = a8.REPO_ID


def write_wav(path, seconds, rate=22050):
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        tone = (numpy.sin(numpy.linspace(0, 400 * seconds, int(rate * seconds))) * 8000)
        handle.writeframes(tone.astype("<i2").tobytes())
    return str(path)


@pytest.fixture
def never_loads(monkeypatch):
    """Every way the task could load a model, each failing the test if hit."""
    loaders = {
        "_load_audio8": MagicMock(side_effect=AssertionError("Audio8 loaded")),
        "hf_pipeline": MagicMock(side_effect=AssertionError("pipeline loaded")),
    }
    for name, mock in loaders.items():
        monkeypatch.setattr(sg, name, mock)
    yield loaders
    for mock in loaders.values():
        mock.assert_not_called()


class TestAudio8Refusals:
    """Each refusal is the task's own, raised before anything loads."""

    @pytest.mark.parametrize("argument", ["reference_audio", "reference_text"])
    def test_reference_arguments_are_refused_on_another_model(
        self, never_loads, argument
    ):
        with pytest.raises(ValueError, match=f"takes no '{argument}'.*{AUDIO8}"):
            generate_speech(
                "hello", model_name="facebook/mms-tts-eng", **{argument: "x"}
            )

    def test_reference_arguments_are_refused_on_the_default_model(self, never_loads):
        with pytest.raises(ValueError, match="takes no 'reference_audio'"):
            generate_speech("hello", reference_audio="voice.wav", reference_text="hi")

    @pytest.mark.parametrize("argument", ["speaker_embedding", "voice_preset"])
    def test_other_voice_controls_are_refused_on_audio8(self, never_loads, argument):
        with pytest.raises(ValueError, match=f"{AUDIO8} takes no '{argument}'"):
            generate_speech("hello", model_name=AUDIO8, **{argument: "x"})

    def test_messages_are_refused_on_audio8(self, never_loads):
        with pytest.raises(ValueError, match=f"{AUDIO8} takes no 'messages'"):
            generate_speech(
                model_name=AUDIO8, messages=[{"role": "user", "content": "hi"}]
            )

    def test_reference_audio_needs_reference_text(self, never_loads):
        with pytest.raises(ValueError, match="'reference_audio' needs 'reference_text'"):
            generate_speech("hello", model_name=AUDIO8, reference_audio="voice.wav")

    def test_an_empty_reference_text_counts_as_missing(self, never_loads):
        with pytest.raises(ValueError, match="'reference_audio' needs 'reference_text'"):
            generate_speech(
                "hello", model_name=AUDIO8, reference_audio="v.wav", reference_text=""
            )

    def test_reference_text_needs_reference_audio(self, never_loads):
        with pytest.raises(ValueError, match="'reference_text' needs 'reference_audio'"):
            generate_speech("hello", model_name=AUDIO8, reference_text="hi there")

    def test_a_reference_over_twenty_seconds_is_refused(self, never_loads, tmp_path):
        clip = write_wav(tmp_path / "long.wav", 21.0)
        with pytest.raises(ValueError, match=r"21\.0 s.*at most 20 s"):
            generate_speech(
                "hello", model_name=AUDIO8, reference_audio=clip, reference_text="hi"
            )

    def test_text_that_cannot_fit_the_context_is_refused(self, never_loads):
        line = "This sentence is spoken at a steady pace. " * 40
        with pytest.raises(ValueError, match="too long.*2048 context positions"):
            generate_speech(line, model_name=AUDIO8)

    def test_a_long_reference_shrinks_what_fits(self, never_loads, tmp_path):
        # A line that fits alone is refused beside a long reference: the clip's
        # frames come out of the same 2,048 positions
        line = "This sentence is spoken at a steady pace. " * 24
        assert sg._audio8_estimated_positions(line, "", 0.0) <= 2048
        clip = write_wav(tmp_path / "ref.wav", 19.0)
        with pytest.raises(ValueError, match="too long.*shorter reference"):
            generate_speech(
                line, model_name=AUDIO8, reference_audio=clip, reference_text="hi"
            )

    @pytest.mark.parametrize("where", ["generate_kwargs", "forward_params"])
    def test_an_unknown_generation_setting_is_refused_by_name(self, never_loads, where):
        with pytest.raises(ValueError, match=r"generation setting\(s\) \['speaker_id'\]"):
            generate_speech("hello", model_name=AUDIO8, **{where: {"speaker_id": 1}})

    def test_audio8_needs_text(self, never_loads):
        with pytest.raises(ValueError, match="generate_speech needs exactly one of"):
            generate_speech(model_name=AUDIO8)


class FakeOutput:
    def __init__(self, finished=True):
        self.codes = torch.zeros((1, 10, 3), dtype=torch.long)
        self.finished = torch.tensor([finished])


class TestAudio8Dispatch:
    """The branch around a stand-in model: what reaches it, what comes back."""

    @pytest.fixture
    def fake_model(self, monkeypatch):
        model = MagicMock()
        model.config.codec_sample_rate = 44100
        model.device = torch.device("cpu")
        model.generate.return_value = FakeOutput()
        model.decode_audio.return_value = (torch.ones((1, 900)), torch.tensor([700]))
        processor = MagicMock(return_value={"prefix_input_ids": torch.zeros(1)})
        loader = MagicMock(return_value=(model, processor))
        monkeypatch.setattr(sg, "_load_audio8", loader)
        monkeypatch.setattr(
            sg, "hf_pipeline", MagicMock(side_effect=AssertionError("pipeline"))
        )
        return model, processor, loader

    def test_returns_44k_mono_trimmed_to_its_length(self, fake_model):
        track = generate_speech("hello", model_name=AUDIO8)
        assert track.sample_rate == 44100
        assert track.audio.shape == (1, 700)

    def test_no_reference_speaks_in_the_default_voice(self, fake_model):
        _, processor, _ = fake_model
        generate_speech("hello", model_name=AUDIO8)
        processor.assert_called_once_with("hello")

    def test_the_reference_reaches_the_processor_as_mono(self, fake_model, tmp_path):
        _, processor, _ = fake_model
        clip = write_wav(tmp_path / "ref.wav", 2.0, rate=22050)
        generate_speech(
            "hello", model_name=AUDIO8, reference_audio=clip, reference_text="hi"
        )
        kwargs = processor.call_args.kwargs
        assert kwargs["reference_text"] == "hi"
        assert kwargs["sampling_rate"] == 22050
        assert kwargs["reference_audio"].shape == (44100,)

    def test_the_context_is_filled_by_default(self, fake_model):
        model, _, _ = fake_model
        generate_speech("hello", model_name=AUDIO8, generate_kwargs={"top_k": 20})
        kwargs = model.generate.call_args.kwargs
        assert kwargs["max_new_tokens"] == 2048
        assert kwargs["top_k"] == 20
        assert kwargs["return_dict_in_generate"] is True

    def test_a_caller_max_new_tokens_wins(self, fake_model):
        model, _, _ = fake_model
        generate_speech("hello", model_name=AUDIO8, forward_params={"max_new_tokens": 9})
        assert model.generate.call_args.kwargs["max_new_tokens"] == 9

    def test_an_unfinished_line_is_refused_not_truncated(self, fake_model):
        model, _, _ = fake_model
        model.generate.return_value = FakeOutput(finished=False)
        with pytest.raises(ValueError, match="ran out of room"):
            generate_speech("hello", model_name=AUDIO8)
        model.decode_audio.assert_not_called()

    def test_the_seed_becomes_the_generator(self, fake_model):
        model, _, _ = fake_model
        generate_speech("hello", model_name=AUDIO8, seed=7)
        generator = model.generate.call_args.kwargs["generator"]
        assert generator.initial_seed() == 7

    def test_no_seed_samples_unseeded(self, fake_model):
        model, _, _ = fake_model
        generate_speech("hello", model_name=AUDIO8)
        assert model.generate.call_args.kwargs["generator"] is None

    @pytest.mark.parametrize(
        "device, dtype",
        [("cpu", torch.float32), ("mps", torch.float32), ("cuda:1", torch.bfloat16)],
    )
    def test_dtype_by_backend(self, fake_model, device, dtype):
        _, _, loader = fake_model
        generate_speech("hello", model_name=AUDIO8, device=device)
        assert loader.call_args.args == (device, dtype)


class TestAudio8Loader:
    def test_fetches_each_file_at_the_pinned_revision(self, monkeypatch, tmp_path):
        clear_model_cache()
        fetched = []

        def fake_download(repo_id, filename, revision):
            fetched.append((repo_id, filename, revision))
            return str(tmp_path / filename)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
        monkeypatch.setattr(a8, "load_model", MagicMock(return_value=MagicMock()))
        monkeypatch.setattr(a8, "load_tokenizer", MagicMock())
        monkeypatch.setattr(a8.ArkttsProcessor, "for_model", MagicMock())
        try:
            sg._load_audio8("cpu", torch.float32)
        finally:
            clear_model_cache()
        assert fetched == [(AUDIO8, f, a8.REVISION) for f in a8.MODEL_FILES]
        args, kwargs = a8.load_model.call_args
        assert args == tuple(
            str(tmp_path / f) for f in (a8.CONFIG_FILE, a8.WEIGHTS_FILE, a8.CODEC_FILE)
        )
        assert kwargs["generation_config_path"] == str(
            tmp_path / a8.GENERATION_CONFIG_FILE
        )

    def test_the_codec_is_read_with_weights_only(self):
        # what the branch relies on to load codec.pth from a Hub repo
        assert "weights_only=True" in Path(a8.__file__).read_text()


class SmallTokenizer:
    """Characters into the tiny model's text range, below its semantic ids."""

    def encode(self, text, add_special_tokens=True):
        return [3 + ord(ch) % 30 for ch in text]


@pytest.fixture
def tiny_hub(tmp_path, monkeypatch):
    """The real branch over a tiny random checkpoint: the files a hub fetch
    would return, written locally. The slow transformer's output norm is
    zeroed so every allowed token is equally likely - a random model then
    reaches eos within a short line rather than never."""
    from safetensors.torch import save_file

    torch.manual_seed(0)
    overrides = {**TINY, "max_seq_len": 512}
    config = a8.ArkttsConfig(**overrides)
    model = a8.ArkttsModel(config)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state["norm.weight"] = torch.zeros_like(state["norm.weight"])
    files = tmp_path / "hub"
    files.mkdir()
    save_file(state, files / a8.WEIGHTS_FILE)
    torch.save(legacy_codec_state(a8.ArkttsCodec(config)), files / a8.CODEC_FILE)
    (files / a8.CONFIG_FILE).write_text(json.dumps(overrides))
    (files / a8.GENERATION_CONFIG_FILE).write_text(json.dumps({"top_p": 1.0}))
    (files / a8.TOKENIZER_FILE).write_text("{}")

    def fake_download(repo_id, filename, revision):
        assert (repo_id, revision) == (AUDIO8, a8.REVISION)
        return str(files / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr(a8, "load_tokenizer", lambda path: SmallTokenizer())
    clear_model_cache()
    yield tmp_path
    clear_model_cache()


class TestAudio8RealPath:
    """generate_speech through the vendored model, processor and codec - no
    stand-in between the task and the architecture."""

    def test_default_voice_speaks_44k_mono(self, tiny_hub):
        track = generate_speech("hello there", model_name=AUDIO8, seed=3)
        assert track.sample_rate == 44100
        assert track.audio.ndim == 2 and track.audio.shape[0] == 1
        assert track.audio.shape[1] > 0
        assert track.audio.shape[1] % a8.ArkttsCodec.frame_length == 0
        assert numpy.isfinite(track.audio).all()

    def test_a_reference_clone_runs_and_the_seed_repeats_it(self, tiny_hub):
        clip = write_wav(tiny_hub / "voice.wav", 1.0)
        arguments = dict(
            model_name=AUDIO8, reference_audio=clip, reference_text="hi", seed=5
        )
        first = generate_speech("hello there", **arguments)
        second = generate_speech("hello there", **arguments)
        assert first.sample_rate == 44100
        assert first.audio.shape[1] > 0
        assert first.audio.shape == second.audio.shape
        numpy.testing.assert_array_equal(first.audio, second.audio)
