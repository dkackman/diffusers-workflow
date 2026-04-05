"""Tests for diffusion_upscale task."""

import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

from dw.tasks.diffusion_upscale import diffusion_upscale, _MODELS


class TestDiffusionUpscale(unittest.TestCase):
    """Tests for the diffusion_upscale function."""

    def _make_image(self):
        return Image.new("RGB", (128, 128), color="red")

    def _make_mock_pipeline(self):
        """Create a mock pipeline that returns a PIL image."""
        output_image = Image.new("RGB", (512, 512), color="blue")
        mock_result = MagicMock()
        mock_result.images = [output_image]

        mock_pipe = MagicMock()
        mock_pipe.return_value = mock_result
        return mock_pipe

    @patch("dw.tasks.diffusion_upscale.diffusers")
    def test_returns_pil_image(self, mock_diffusers):
        mock_pipe = self._make_mock_pipeline()
        mock_diffusers.StableDiffusionUpscalePipeline.from_pretrained.return_value = (
            mock_pipe
        )

        result = diffusion_upscale(self._make_image(), device="cpu")

        self.assertIsInstance(result, Image.Image)

    @patch("dw.tasks.diffusion_upscale.diffusers")
    def test_default_mode_is_x4(self, mock_diffusers):
        mock_pipe = self._make_mock_pipeline()
        mock_diffusers.StableDiffusionUpscalePipeline.from_pretrained.return_value = (
            mock_pipe
        )

        diffusion_upscale(self._make_image(), device="cpu")

        mock_diffusers.StableDiffusionUpscalePipeline.from_pretrained.assert_called_once()

    @patch("dw.tasks.diffusion_upscale.diffusers")
    def test_x2_mode_uses_latent_pipeline(self, mock_diffusers):
        mock_pipe = self._make_mock_pipeline()
        mock_diffusers.StableDiffusionLatentUpscalePipeline.from_pretrained.return_value = (
            mock_pipe
        )

        diffusion_upscale(self._make_image(), device="cpu", mode="x2")

        mock_diffusers.StableDiffusionLatentUpscalePipeline.from_pretrained.assert_called_once()

    @patch("dw.tasks.diffusion_upscale.diffusers")
    def test_x4_includes_noise_level(self, mock_diffusers):
        mock_pipe = self._make_mock_pipeline()
        mock_diffusers.StableDiffusionUpscalePipeline.from_pretrained.return_value = (
            mock_pipe
        )

        diffusion_upscale(self._make_image(), device="cpu")

        call_kwargs = mock_pipe.call_args[1]
        self.assertIn("noise_level", call_kwargs)
        self.assertEqual(call_kwargs["noise_level"], 20)

    @patch("dw.tasks.diffusion_upscale.diffusers")
    def test_x2_excludes_noise_level(self, mock_diffusers):
        mock_pipe = self._make_mock_pipeline()
        mock_diffusers.StableDiffusionLatentUpscalePipeline.from_pretrained.return_value = (
            mock_pipe
        )

        diffusion_upscale(self._make_image(), device="cpu", mode="x2")

        call_kwargs = mock_pipe.call_args[1]
        self.assertNotIn("noise_level", call_kwargs)

    @patch("dw.tasks.diffusion_upscale.diffusers")
    def test_custom_prompt(self, mock_diffusers):
        mock_pipe = self._make_mock_pipeline()
        mock_diffusers.StableDiffusionUpscalePipeline.from_pretrained.return_value = (
            mock_pipe
        )

        diffusion_upscale(self._make_image(), device="cpu", prompt="a photo of a cat")

        call_kwargs = mock_pipe.call_args[1]
        self.assertEqual(call_kwargs["prompt"], "a photo of a cat")

    @patch("dw.tasks.diffusion_upscale.diffusers")
    def test_negative_prompt(self, mock_diffusers):
        mock_pipe = self._make_mock_pipeline()
        mock_diffusers.StableDiffusionUpscalePipeline.from_pretrained.return_value = (
            mock_pipe
        )

        diffusion_upscale(
            self._make_image(), device="cpu", negative_prompt="blurry, low quality"
        )

        call_kwargs = mock_pipe.call_args[1]
        self.assertEqual(call_kwargs["negative_prompt"], "blurry, low quality")

    @patch("dw.tasks.diffusion_upscale.diffusers")
    def test_no_negative_prompt_by_default(self, mock_diffusers):
        mock_pipe = self._make_mock_pipeline()
        mock_diffusers.StableDiffusionUpscalePipeline.from_pretrained.return_value = (
            mock_pipe
        )

        diffusion_upscale(self._make_image(), device="cpu")

        call_kwargs = mock_pipe.call_args[1]
        self.assertNotIn("negative_prompt", call_kwargs)

    @patch("dw.tasks.diffusion_upscale.diffusers")
    def test_custom_inference_steps(self, mock_diffusers):
        mock_pipe = self._make_mock_pipeline()
        mock_diffusers.StableDiffusionUpscalePipeline.from_pretrained.return_value = (
            mock_pipe
        )

        diffusion_upscale(self._make_image(), device="cpu", num_inference_steps=50)

        call_kwargs = mock_pipe.call_args[1]
        self.assertEqual(call_kwargs["num_inference_steps"], 50)

    @patch("dw.tasks.diffusion_upscale.diffusers")
    def test_custom_guidance_scale(self, mock_diffusers):
        mock_pipe = self._make_mock_pipeline()
        mock_diffusers.StableDiffusionUpscalePipeline.from_pretrained.return_value = (
            mock_pipe
        )

        diffusion_upscale(self._make_image(), device="cpu", guidance_scale=7.5)

        call_kwargs = mock_pipe.call_args[1]
        self.assertEqual(call_kwargs["guidance_scale"], 7.5)

    @patch("dw.tasks.diffusion_upscale.diffusers")
    def test_custom_model_name(self, mock_diffusers):
        mock_pipe = self._make_mock_pipeline()
        mock_diffusers.StableDiffusionUpscalePipeline.from_pretrained.return_value = (
            mock_pipe
        )

        diffusion_upscale(
            self._make_image(), device="cpu", model_name="my-org/my-upscaler"
        )

        call_args = (
            mock_diffusers.StableDiffusionUpscalePipeline.from_pretrained.call_args
        )
        self.assertEqual(call_args[0][0], "my-org/my-upscaler")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError) as ctx:
            diffusion_upscale(self._make_image(), device="cpu", mode="x8")
        self.assertIn("x8", str(ctx.exception))

    def test_models_config_has_expected_modes(self):
        self.assertIn("x4", _MODELS)
        self.assertIn("x2", _MODELS)
        self.assertEqual(
            _MODELS["x4"]["pipeline_class"], "StableDiffusionUpscalePipeline"
        )
        self.assertEqual(
            _MODELS["x2"]["pipeline_class"], "StableDiffusionLatentUpscalePipeline"
        )


class TestDiffusionUpscaleRegistration(unittest.TestCase):
    """Test that diffusion_upscale is registered as a task command."""

    def test_command_registered(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        self.assertIn("diffusion_upscale", _COMMAND_REGISTRY)


if __name__ == "__main__":
    unittest.main()
