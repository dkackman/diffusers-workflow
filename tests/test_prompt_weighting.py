"""
Unit tests for prompt weighting device handling
"""

import torch
from unittest.mock import MagicMock, patch

from dw.prompt_weighting import _get_device, _hook_managed, apply_prompt_weighting


class TestDeviceResolution:
    """Embeddings must land on the device the pipeline runs on"""

    def pipe_on(self, device_type):
        pipe = MagicMock()
        pipe.device = torch.device(device_type)
        return pipe

    def test_pipeline_device_wins_when_not_offloaded(self):
        assert _get_device(self.pipe_on("cuda")) == torch.device("cuda")

    def test_cpu_pipeline_falls_back_to_the_dw_device(self):
        # Model offloading parks the pipeline on cpu - the configured dw device
        # (DW_DEVICE / settings) decides, never a hardcoded accelerator
        with patch("dw.get_device", return_value="cpu") as get_device:
            assert _get_device(self.pipe_on("cpu")) == torch.device("cpu")
        get_device.assert_called_once()

    def test_meta_pipeline_falls_back_to_the_dw_device(self):
        # Sequential offloading can leave the pipeline reporting meta
        with patch("dw.get_device", return_value="cpu"):
            assert _get_device(self.pipe_on("meta")) == torch.device("cpu")

    def test_hook_managed_detects_accelerate_hooks(self):
        hooked = MagicMock()
        hooked._hf_hook = object()
        plain = object()
        assert _hook_managed(hooked)
        assert not _hook_managed(plain)


class TestEmbeddingFunctionSelection:
    """Flux-family pipelines with the same encoder stack are supported"""

    def flux_like(self, name, missing=None):
        pipeline = MagicMock(
            spec=["tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2"]
        )
        pipeline.__class__ = type(name, (), {})
        if missing:
            setattr(pipeline, missing, None)
        return pipeline

    def test_a_new_flux_variant_is_supported(self):
        from dw.prompt_weighting import (
            _select_embedding_function,
            get_weighted_text_embeddings_flux,
        )

        pipeline = self.flux_like("FluxKontextPipeline")
        assert _select_embedding_function(pipeline) is get_weighted_text_embeddings_flux

    def test_a_flux_variant_missing_an_encoder_is_not(self):
        from dw.prompt_weighting import _select_embedding_function

        pipeline = self.flux_like("FluxKontextPipeline", missing="text_encoder_2")
        assert _select_embedding_function(pipeline) is None

    def test_non_flux_pipelines_are_not_supported(self):
        from dw.prompt_weighting import _select_embedding_function

        pipeline = self.flux_like("StableDiffusion3Pipeline")
        assert _select_embedding_function(pipeline) is None


class TestApplyPromptWeighting:
    def test_device_is_threaded_to_the_embedding_function(self):
        pipeline = MagicMock()
        embed_fn = MagicMock(return_value=(torch.zeros(1), torch.zeros(1)))
        arguments = {"prompt": "a (weighted:1.2) prompt"}

        with patch.dict(
            "dw.prompt_weighting._PIPELINE_FUNCTIONS",
            {type(pipeline).__name__: embed_fn},
        ):
            applied = apply_prompt_weighting(pipeline, arguments, device="cuda:1")

        assert applied
        assert embed_fn.call_args.kwargs["device"] == "cuda:1"
        assert "prompt_embeds" in arguments
