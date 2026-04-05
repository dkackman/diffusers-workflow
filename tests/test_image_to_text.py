"""Tests for image_to_text task."""

import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

from dw.tasks.image_to_text import image_to_text, _DEFAULT_MODEL


class TestImageToText(unittest.TestCase):
    """Tests for the image_to_text function."""

    def _make_image(self):
        return Image.new("RGB", (64, 64), color="red")

    @patch("dw.tasks.image_to_text.hf_pipeline")
    def test_returns_caption_string(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "a red square"}]
        mock_pipeline.return_value = pipe

        result = image_to_text(self._make_image(), device="cpu")

        self.assertEqual(result, "a red square")
        mock_pipeline.assert_called_once()

    @patch("dw.tasks.image_to_text.hf_pipeline")
    def test_uses_default_model(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "caption"}]
        mock_pipeline.return_value = pipe

        image_to_text(self._make_image(), device="cpu")

        call_kwargs = mock_pipeline.call_args
        self.assertEqual(call_kwargs[1]["model"], _DEFAULT_MODEL)

    @patch("dw.tasks.image_to_text.hf_pipeline")
    def test_custom_model_name(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "detailed caption"}]
        mock_pipeline.return_value = pipe

        image_to_text(
            self._make_image(),
            device="cpu",
            model_name="Salesforce/blip2-opt-2.7b",
        )

        call_kwargs = mock_pipeline.call_args
        self.assertEqual(call_kwargs[1]["model"], "Salesforce/blip2-opt-2.7b")

    @patch("dw.tasks.image_to_text.hf_pipeline")
    def test_prompt_passed_as_generate_kwarg(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "a photo of a dog"}]
        mock_pipeline.return_value = pipe

        image_to_text(
            self._make_image(),
            device="cpu",
            prompt="Question: what is this? Answer:",
        )

        call_args = pipe.call_args
        self.assertIn("prompt", call_args[1]["generate_kwargs"])

    @patch("dw.tasks.image_to_text.hf_pipeline")
    def test_max_new_tokens(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "caption"}]
        mock_pipeline.return_value = pipe

        image_to_text(self._make_image(), device="cpu", max_new_tokens=100)

        call_args = pipe.call_args
        self.assertEqual(call_args[1]["generate_kwargs"]["max_new_tokens"], 100)

    @patch("dw.tasks.image_to_text.hf_pipeline")
    def test_strips_whitespace(self, mock_pipeline):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "  a caption with spaces  "}]
        mock_pipeline.return_value = pipe

        result = image_to_text(self._make_image(), device="cpu")
        self.assertEqual(result, "a caption with spaces")


class TestImageToTextRegistration(unittest.TestCase):
    """Test that image_to_text is registered as a task command."""

    def test_command_registered(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        self.assertIn("image_to_text", _COMMAND_REGISTRY)


if __name__ == "__main__":
    unittest.main()
