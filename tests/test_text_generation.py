"""Tests for text_generation task."""

import unittest
from unittest.mock import patch, MagicMock

from dw.tasks.text_generation import generate_text, _DEFAULT_MODEL


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

        generate_text("test", device="cpu", model_name="meta-llama/Llama-3.2-1B-Instruct")

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
        self.assertEqual(messages[0]["content"], "You expand prompts for image generation.")
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


class TestTextGenerationRegistration(unittest.TestCase):
    """Test that text_generation is registered as a task command."""

    def test_command_registered(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        self.assertIn("text_generation", _COMMAND_REGISTRY)


if __name__ == "__main__":
    unittest.main()
