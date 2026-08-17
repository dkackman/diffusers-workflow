"""Tests for text_generation task."""

import unittest
from unittest.mock import patch, MagicMock

from dw.tasks.text_generation import (
    generate_text,
    _DEFAULT_MODEL,
    _DEFAULT_VISION_MODEL,
    _VISION_REPETITION_PENALTY,
)


class TestTextGeneration(unittest.TestCase):
    """Tests for the generate_text function."""

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_returns_generated_string(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "an expanded detailed prompt"}]
        mock_pipeline.return_value = pipe

        result = generate_text("a cat", device="cpu")

        self.assertEqual(result, "an expanded detailed prompt")
        mock_pipeline.assert_called_once()

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_uses_default_model(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "output"}]
        mock_pipeline.return_value = pipe

        generate_text("test", device="cpu")

        call_kwargs = mock_pipeline.call_args
        self.assertEqual(call_kwargs[1]["model"], _DEFAULT_MODEL)

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_custom_model_name(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "output"}]
        mock_pipeline.return_value = pipe

        generate_text(
            "test", device="cpu", model_name="meta-llama/Llama-3.2-1B-Instruct"
        )

        call_kwargs = mock_pipeline.call_args
        self.assertEqual(call_kwargs[1]["model"], "meta-llama/Llama-3.2-1B-Instruct")

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_system_prompt_included(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "output"}]
        mock_pipeline.return_value = pipe

        generate_text(
            "a cat",
            device="cpu",
            system_prompt="You expand prompts for image generation.",
        )

        call_args = pipe.call_args
        messages = call_args[0][0]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(
            messages[0]["content"], "You expand prompts for image generation."
        )
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "a cat")

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_no_system_prompt(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "output"}]
        mock_pipeline.return_value = pipe

        generate_text("a cat", device="cpu")

        call_args = pipe.call_args
        messages = call_args[0][0]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_max_new_tokens(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "output"}]
        mock_pipeline.return_value = pipe

        generate_text("test", device="cpu", max_new_tokens=200)

        call_args = pipe.call_args
        self.assertEqual(call_args[1]["max_new_tokens"], 200)

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_strips_whitespace(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "  some text with spaces  \n"}]
        mock_pipeline.return_value = pipe

        result = generate_text("test", device="cpu")
        self.assertEqual(result, "some text with spaces")


class TestTextGenerationWithImage(unittest.TestCase):
    """An image switches generation onto a vision-language pipeline."""

    def _make_image(self):
        from PIL import Image

        return Image.new("RGB", (32, 32), color="blue")

    def _mock_pipe(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "a blue square on a plain field"}]
        mock_pipeline.return_value = pipe
        return pipe

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_image_selects_the_vision_pipeline(self, mock_pipeline):
        self._mock_pipe(mock_pipeline)

        generate_text("describe it", device="cpu", image=self._make_image())

        self.assertEqual(mock_pipeline.call_args[0][0], "image-text-to-text")

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_no_image_stays_on_text_generation(self, mock_pipeline):
        self._mock_pipe(mock_pipeline)

        generate_text("a cat", device="cpu")

        self.assertEqual(mock_pipeline.call_args[0][0], "text-generation")

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_image_uses_the_vision_default_model(self, mock_pipeline):
        self._mock_pipe(mock_pipeline)

        generate_text("describe it", device="cpu", image=self._make_image())

        self.assertEqual(mock_pipeline.call_args[1]["model"], _DEFAULT_VISION_MODEL)

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_vision_content_is_typed_parts(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)
        image = self._make_image()

        generate_text(
            "describe it",
            device="cpu",
            image=image,
            system_prompt="You are terse.",
        )

        messages = pipe.call_args[1]["text"]
        self.assertEqual(
            messages[0]["content"], [{"type": "text", "text": "You are terse."}]
        )
        self.assertEqual(
            messages[1]["content"],
            [
                {"type": "image", "image": image},
                {"type": "text", "text": "describe it"},
            ],
        )

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_string_image_is_passed_as_a_url(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)

        generate_text("describe it", device="cpu", image="https://example.com/cat.jpg")

        content = pipe.call_args[1]["text"][-1]["content"]
        self.assertEqual(
            content[0], {"type": "image", "url": "https://example.com/cat.jpg"}
        )

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_sampling_is_disabled_through_generate_kwargs(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)

        generate_text("describe it", device="cpu", image=self._make_image())

        # This pipeline forwards unrecognised kwargs to the processor, which
        # drops them - a bare do_sample=False would silently leave sampling on
        self.assertIs(pipe.call_args[1]["generate_kwargs"]["do_sample"], False)
        self.assertNotIn("do_sample", pipe.call_args[1])

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_vision_defaults_to_a_repetition_penalty(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)

        generate_text("describe it", device="cpu", image=self._make_image())

        # Greedy decoding loops on long format specs; the penalty is what stops
        # that without giving up reproducible output
        self.assertEqual(
            pipe.call_args[1]["generate_kwargs"]["repetition_penalty"],
            _VISION_REPETITION_PENALTY,
        )

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_repetition_penalty_is_overridable(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)

        generate_text(
            "describe it",
            device="cpu",
            image=self._make_image(),
            repetition_penalty=1.2,
        )

        self.assertEqual(
            pipe.call_args[1]["generate_kwargs"]["repetition_penalty"], 1.2
        )

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_text_path_takes_no_repetition_penalty(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)

        generate_text("a cat", device="cpu")

        # The text models do not loop, and changing their decoding would change
        # the output of every workflow already using them
        self.assertNotIn("generate_kwargs", pipe.call_args[1])

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_text_and_vision_variants_cache_separately(self, mock_pipeline):
        self._mock_pipe(mock_pipeline)

        # Same model name under both pipelines - they are not interchangeable,
        # so the cache must not hand one back for the other
        generate_text("a cat", device="cpu", model_name="some/model")
        generate_text(
            "a cat", device="cpu", model_name="some/model", image=self._make_image()
        )

        self.assertEqual(mock_pipeline.call_count, 2)
        tasks = [call[0][0] for call in mock_pipeline.call_args_list]
        self.assertEqual(tasks, ["text-generation", "image-text-to-text"])


class TestGenerateKwargsPassThrough(unittest.TestCase):
    """Anything generate() understands can be set from a workflow."""

    def _make_image(self):
        from PIL import Image

        return Image.new("RGB", (32, 32), color="blue")

    def _mock_pipe(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "output"}]
        mock_pipeline.return_value = pipe
        return pipe

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_reaches_the_text_pipeline(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)

        generate_text(
            "a cat", device="cpu", generate_kwargs={"no_repeat_ngram_size": 25}
        )

        self.assertEqual(pipe.call_args[1]["no_repeat_ngram_size"], 25)

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_reaches_the_vision_pipeline(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)

        generate_text(
            "describe it",
            device="cpu",
            image=self._make_image(),
            generate_kwargs={"no_repeat_ngram_size": 25},
        )

        self.assertEqual(
            pipe.call_args[1]["generate_kwargs"]["no_repeat_ngram_size"], 25
        )

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_overrides_what_the_task_decided(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline)

        generate_text(
            "describe it",
            device="cpu",
            image=self._make_image(),
            generate_kwargs={"repetition_penalty": 1.3},
        )

        self.assertEqual(
            pipe.call_args[1]["generate_kwargs"]["repetition_penalty"], 1.3
        )


class TestTextGenerationRegistration(unittest.TestCase):
    """Test that text_generation is registered as a task command."""

    def test_command_registered(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        self.assertIn("text_generation", _COMMAND_REGISTRY)


if __name__ == "__main__":
    unittest.main()
