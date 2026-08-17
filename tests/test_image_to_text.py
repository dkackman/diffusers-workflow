"""Tests for image_to_text task."""

import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

from dw.tasks.image_to_text import image_to_text, _DEFAULT_MODEL, _DEFAULT_PROMPT


class TestImageToText(unittest.TestCase):
    """Tests for the image_to_text function.

    Captioning runs through the text_generation task's vision path, so the
    pipeline is patched at its source there.
    """

    def _make_image(self):
        return Image.new("RGB", (64, 64), color="red")

    def _mock_pipe(self, mock_pipeline, generated="a red square"):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": generated}]
        mock_pipeline.return_value = pipe
        return pipe

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_returns_caption_string(self, mock_pipeline):
        self._mock_pipe(mock_pipeline)

        result = image_to_text(self._make_image(), device="cpu")

        self.assertEqual(result, "a red square")
        mock_pipeline.assert_called_once()

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_builds_a_vision_pipeline(self, mock_pipeline):
        self._mock_pipe(mock_pipeline)

        image_to_text(self._make_image(), device="cpu")

        # Transformers 5 removed "image-to-text"; captioning is a VLM task now
        self.assertEqual(mock_pipeline.call_args[0][0], "image-text-to-text")

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_uses_default_model(self, mock_pipeline):
        self._mock_pipe(mock_pipeline, "caption")

        image_to_text(self._make_image(), device="cpu")

        self.assertEqual(mock_pipeline.call_args[1]["model"], _DEFAULT_MODEL)

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_custom_model_name(self, mock_pipeline):
        self._mock_pipe(mock_pipeline, "detailed caption")

        image_to_text(
            self._make_image(),
            device="cpu",
            model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        )

        self.assertEqual(
            mock_pipeline.call_args[1]["model"], "Qwen/Qwen2.5-VL-3B-Instruct"
        )

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_image_and_default_prompt_are_in_the_message(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline, "caption")
        image = self._make_image()

        image_to_text(image, device="cpu")

        content = pipe.call_args[1]["text"][-1]["content"]
        self.assertEqual(content[0], {"type": "image", "image": image})
        self.assertEqual(content[1], {"type": "text", "text": _DEFAULT_PROMPT})

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_prompt_becomes_the_question(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline, "a photo of a dog")

        image_to_text(
            self._make_image(),
            device="cpu",
            prompt="What breed is this?",
        )

        content = pipe.call_args[1]["text"][-1]["content"]
        self.assertEqual(content[1], {"type": "text", "text": "What breed is this?"})

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_max_new_tokens(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline, "caption")

        image_to_text(self._make_image(), device="cpu", max_new_tokens=100)

        self.assertEqual(pipe.call_args[1]["max_new_tokens"], 100)

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_strips_whitespace(self, mock_pipeline):
        self._mock_pipe(mock_pipeline, "  a caption with spaces  ")

        result = image_to_text(self._make_image(), device="cpu")
        self.assertEqual(result, "a caption with spaces")


class TestImageToTextRegistration(unittest.TestCase):
    """Test that image_to_text is registered as a task command."""

    def test_command_registered(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        self.assertIn("image_to_text", _COMMAND_REGISTRY)


if __name__ == "__main__":
    unittest.main()
