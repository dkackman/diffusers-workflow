"""Tests for the judge task - scoring a candidate against a rubric."""

import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

from dw.tasks.judge import judge, _DEFAULT_MODEL


class TestJudge(unittest.TestCase):
    """Judge prompts a vision-language model with a rubric and scale, and
    parses the reply to one number. Mocked at the same seam image_to_text's
    tests use, since judge runs through the same text_generation vision
    path."""

    def _make_image(self):
        return Image.new("RGB", (64, 64), color="blue")

    def _mock_pipe(self, mock_pipeline, generated):
        pipe = MagicMock()
        pipe.return_value = [{"generated_text": generated}]
        mock_pipeline.return_value = pipe
        return pipe

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_parses_a_clean_number(self, mock_pipeline):
        self._mock_pipe(mock_pipeline, "7")

        score = judge(
            self._make_image(), rubric="Sharpest image", scale=[0, 10], device="cpu"
        )

        self.assertEqual(score, 7.0)

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_parses_a_number_in_prose(self, mock_pipeline):
        self._mock_pipe(mock_pipeline, "I would rate this a 8 out of 10 for sharpness.")

        score = judge(
            self._make_image(), rubric="Sharpest image", scale=[0, 10], device="cpu"
        )

        self.assertEqual(score, 8.0)

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_no_number_in_reply_is_an_error(self, mock_pipeline):
        self._mock_pipe(mock_pipeline, "I cannot judge this image.")

        with self.assertRaises(ValueError) as ctx:
            judge(
                self._make_image(), rubric="Sharpest image", scale=[0, 10], device="cpu"
            )

        message = str(ctx.exception)
        self.assertIn("judge", message)
        self.assertIn("I cannot judge this image.", message)

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_uses_default_model(self, mock_pipeline):
        self._mock_pipe(mock_pipeline, "5")

        judge(self._make_image(), rubric="Sharpest image", scale=[0, 10], device="cpu")

        self.assertEqual(mock_pipeline.call_args[1]["model"], _DEFAULT_MODEL)

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_custom_model_name(self, mock_pipeline):
        self._mock_pipe(mock_pipeline, "5")

        judge(
            self._make_image(),
            rubric="Sharpest image",
            scale=[0, 10],
            device="cpu",
            model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        )

        self.assertEqual(
            mock_pipeline.call_args[1]["model"], "Qwen/Qwen2.5-VL-3B-Instruct"
        )

    @patch("dw.tasks.text_generation.hf_pipeline")
    def test_rubric_and_scale_are_in_the_prompt(self, mock_pipeline):
        pipe = self._mock_pipe(mock_pipeline, "5")

        judge(self._make_image(), rubric="Sharpest image", scale=[0, 10], device="cpu")

        content = pipe.call_args[1]["text"][-1]["content"]
        prompt_text = content[1]["text"]
        self.assertIn("Sharpest image", prompt_text)
        self.assertIn("0", prompt_text)
        self.assertIn("10", prompt_text)


class TestJudgeRegistration(unittest.TestCase):
    """Test that judge is registered as a task command."""

    def test_command_registered(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        self.assertIn("judge", _COMMAND_REGISTRY)


if __name__ == "__main__":
    unittest.main()
