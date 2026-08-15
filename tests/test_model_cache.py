"""Tests for dw.tasks.model_cache and its use by task modules.

Covers the cache primitive itself (cached_model / clear_model_cache) plus an
integration-style check that a real task module (image_to_text) only invokes
its underlying loader once for repeated calls with the same key.
"""

from unittest.mock import patch, MagicMock

from PIL import Image

from dw.tasks.model_cache import cached_model, clear_model_cache


class TestCachedModel:
    def test_factory_called_once_for_same_key(self):
        factory = MagicMock(return_value="the-model")

        first = cached_model(("task", "model-a", "cpu"), factory)
        second = cached_model(("task", "model-a", "cpu"), factory)

        assert first == "the-model"
        assert second == "the-model"
        factory.assert_called_once()

    def test_distinct_keys_get_distinct_loads(self):
        factory_a = MagicMock(return_value="model-a-instance")
        factory_b = MagicMock(return_value="model-b-instance")

        result_a = cached_model(("task", "model-a", "cpu"), factory_a)
        result_b = cached_model(("task", "model-b", "cpu"), factory_b)

        assert result_a == "model-a-instance"
        assert result_b == "model-b-instance"
        factory_a.assert_called_once()
        factory_b.assert_called_once()

    def test_device_is_part_of_the_key(self):
        """Same model name on two different devices must load independently."""
        factory_cpu = MagicMock(return_value="on-cpu")
        factory_cuda = MagicMock(return_value="on-cuda")

        result_cpu = cached_model(("task", "model-a", "cpu"), factory_cpu)
        result_cuda = cached_model(("task", "model-a", "cuda"), factory_cuda)

        assert result_cpu == "on-cpu"
        assert result_cuda == "on-cuda"
        factory_cpu.assert_called_once()
        factory_cuda.assert_called_once()

    def test_returns_the_same_object_instance(self):
        """Callers get the identical cached object back, not a copy."""
        loaded = object()
        factory = MagicMock(return_value=loaded)

        first = cached_model(("task", "model-a", "cpu"), factory)
        second = cached_model(("task", "model-a", "cpu"), factory)

        assert first is loaded
        assert second is loaded

    def test_factory_not_called_until_needed(self):
        """A cache hit must not invoke factory at all, even zero times isn't assumed."""
        cached_model(("task", "shared-key", "cpu"), MagicMock(return_value="v1"))

        never_called = MagicMock(return_value="v2")
        result = cached_model(("task", "shared-key", "cpu"), never_called)

        assert result == "v1"
        never_called.assert_not_called()

    def test_clear_model_cache_releases_entries(self):
        factory = MagicMock(return_value="the-model")
        cached_model(("task", "model-a", "cpu"), factory)

        clear_model_cache()

        # After clearing, the same key must trigger a fresh factory call.
        factory2 = MagicMock(return_value="the-model-again")
        result = cached_model(("task", "model-a", "cpu"), factory2)

        assert result == "the-model-again"
        factory2.assert_called_once()

    def test_clear_model_cache_is_safe_when_empty(self):
        clear_model_cache()
        clear_model_cache()  # must not raise


class TestImageToTextIntegration:
    """Cheap integration check: a real task module reuses a cached load."""

    def _make_image(self):
        return Image.new("RGB", (32, 32), color="green")

    @patch("dw.tasks.image_to_text.hf_pipeline")
    def test_second_call_does_not_reload_pipeline(self, mock_hf_pipeline):
        from dw.tasks.image_to_text import image_to_text

        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "a caption"}]
        mock_hf_pipeline.return_value = pipe

        image_to_text(self._make_image(), device="cpu")
        image_to_text(self._make_image(), device="cpu")

        # The heavy loader (transformers pipeline()) should only run once;
        # the second call must reuse the cached pipe object.
        mock_hf_pipeline.assert_called_once()
        assert pipe.call_count == 2

    @patch("dw.tasks.image_to_text.hf_pipeline")
    def test_different_model_name_reloads(self, mock_hf_pipeline):
        from dw.tasks.image_to_text import image_to_text

        pipe = MagicMock()
        pipe.return_value = [{"generated_text": "a caption"}]
        mock_hf_pipeline.return_value = pipe

        image_to_text(self._make_image(), device="cpu", model_name="model-a")
        image_to_text(self._make_image(), device="cpu", model_name="model-b")

        assert mock_hf_pipeline.call_count == 2


class TestZoeDepthIntegration:
    """zoe_depth.load_zoe used to re-run torch.hub.help (a fresh download)
    on every call; verify caching stops that."""

    @patch("dw.tasks.zoe_depth.torch")
    def test_second_call_does_not_rerun_hub_help(self, mock_torch):
        from dw.tasks.zoe_depth import load_zoe

        model_instance = MagicMock()
        model_instance.eval.return_value = model_instance
        model_instance.to.return_value = model_instance
        mock_torch.hub.load.return_value = model_instance

        load_zoe(device="cpu")
        load_zoe(device="cpu")

        mock_torch.hub.help.assert_called_once()
        mock_torch.hub.load.assert_called_once()
