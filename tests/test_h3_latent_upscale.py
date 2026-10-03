"""Unit tests for MiniMax-H3 latent upscale/decode (#499).

No network access and no real weights: `_load_upscaler`/`_load_decoder` are
monkeypatched at their call sites, and the model cache is cleared around any
test that touches it so a stub from one test cannot leak into another.
"""

import re

import pytest
import torch
from PIL import Image

from dw.tasks import h3_latent_upscale as hlu
from dw.tasks.h3_latent_upscale import (
    DEFAULT_H3_REPO,
    DEFAULT_UPSCALER_REPO,
    UPSCALER_REVISION,
    decode_h3_latents,
    target_latent_size,
    upscale_h3_latents,
)
from dw.tasks.h3_latent_upscaler_model import (
    LATENT_CHANNELS,
    LatentResizer3D,
    architecture_mismatch,
    build_upscaler,
    strip_prefix,
)
from dw.tasks.model_cache import clear_model_cache
from dw.result import AudioVideo


def h3_latents(batch=1, frames=5, height=34, width=60, dtype=torch.float32):
    return torch.randn(batch, LATENT_CHANNELS, frames, height, width, dtype=dtype)


@pytest.fixture(autouse=True)
def _clean_model_cache():
    clear_model_cache()
    yield
    clear_model_cache()


class TestTargetLatentSize:
    def test_non_uniform_scale_and_the_conditioning_scale_is_their_mean(self):
        latents = h3_latents(height=34, width=60)
        size, (scale_h, scale_w) = target_latent_size(latents, width=1344, height=768)

        assert size == (5, 48, 84)
        assert scale_h == pytest.approx(48 / 34)
        assert scale_w == pytest.approx(84 / 60)
        assert scale_h != scale_w
        mean_scale = (scale_h + scale_w) / 2
        assert mean_scale == pytest.approx((48 / 34 + 84 / 60) / 2)

    def test_uniform_scale(self):
        # 16x16 latents (256x256 px) uniformly doubled to 512x512
        latents = h3_latents(height=16, width=16)
        size, (scale_h, scale_w) = target_latent_size(latents, width=512, height=512)

        assert size == (5, 32, 32)
        assert scale_h == pytest.approx(scale_w) == pytest.approx(2.0)

    def test_frames_are_unchanged(self):
        latents = h3_latents(frames=9, height=34, width=60)
        (frames, _, _), _ = target_latent_size(latents, width=960, height=544)
        assert frames == 9


class TestTargetLatentSizeRefusals:
    def test_not_a_tensor_pil_image(self):
        with pytest.raises(ValueError, match="latents"):
            target_latent_size(Image.new("RGB", (8, 8)), width=960, height=544)

    def test_not_a_tensor_list(self):
        with pytest.raises(ValueError, match="latents"):
            target_latent_size([1, 2, 3], width=960, height=544)

    def test_not_5d(self):
        with pytest.raises(ValueError, match="latents"):
            target_latent_size(torch.randn(1, 24, 5, 34), width=960, height=544)

    def test_wrong_channel_count(self):
        with pytest.raises(ValueError, match="latents"):
            target_latent_size(torch.randn(1, 3, 5, 34, 60), width=960, height=544)

    def test_width_not_a_multiple_of_16(self):
        latents = h3_latents()
        with pytest.raises(ValueError, match="width"):
            target_latent_size(latents, width=1350, height=768)

    def test_height_not_a_multiple_of_16(self):
        latents = h3_latents()
        with pytest.raises(ValueError, match="height"):
            target_latent_size(latents, width=960, height=550)

    def test_over_the_canvas_long_edge(self):
        latents = h3_latents()
        with pytest.raises(ValueError, match="canvas"):
            target_latent_size(latents, width=4000, height=768)

    def test_over_the_canvas_short_edge(self):
        latents = h3_latents()
        with pytest.raises(ValueError, match="canvas"):
            target_latent_size(latents, width=1344, height=800)

    def test_portrait_canvas_is_accepted(self):
        latents = h3_latents(height=60, width=34)
        # 768x1344 portrait is within the canvas (short edge <= 768, long <= 1344)
        size, _ = target_latent_size(latents, width=768, height=1344)
        assert size == (5, 1344 // 16, 768 // 16)

    def test_downscale_is_refused(self):
        # 34x60 latents (544x960) downscaled to 528x944 - just under 1x on both axes
        latents = h3_latents(height=34, width=60)
        with pytest.raises(ValueError, match="upscaler takes"):
            target_latent_size(latents, width=944, height=528)

    def test_over_4x_scale_is_refused(self):
        # 4x4 latents upscaled to 768x768 is a 12x scale on both axes
        latents = h3_latents(height=4, width=4)
        with pytest.raises(ValueError, match="upscaler takes"):
            target_latent_size(latents, width=768, height=768)

    def test_width_not_an_int(self):
        latents = h3_latents()
        with pytest.raises(ValueError, match="width"):
            target_latent_size(latents, width="960", height=544)

    def test_width_not_positive(self):
        latents = h3_latents()
        with pytest.raises(ValueError, match="width"):
            target_latent_size(latents, width=0, height=544)


class TestDecodeH3LatentsRefusesNonH3Input:
    def test_not_a_tensor(self):
        with pytest.raises(ValueError, match="latents"):
            decode_h3_latents([1, 2, 3])

    def test_wrong_ndim(self):
        with pytest.raises(ValueError, match="latents"):
            decode_h3_latents(torch.randn(1, 24, 5, 34))

    def test_wrong_channels(self):
        with pytest.raises(ValueError, match="latents"):
            decode_h3_latents(torch.randn(1, 3, 5, 34, 60))


class TestUpscaleH3LatentsCarriesTheNodesNormalization:
    """The network was trained behind the ComfyUI node's (x - mean) / std ...
    x * std + mean wrapper, on latents the VAE had already normalized - the
    space a diffusers pipeline hands over - so the wrapper is applied here
    too (#499, M-F041: without it the decode was garbage)."""

    @staticmethod
    def _stats():
        shape = (1, LATENT_CHANNELS, 1, 1, 1)
        return (
            torch.tensor(hlu.LATENTS_MEAN).view(shape),
            torch.tensor(hlu.LATENTS_STD).view(shape),
        )

    def test_the_stats_are_the_h3_vaes(self):
        assert len(hlu.LATENTS_MEAN) == len(hlu.LATENTS_STD) == LATENT_CHANNELS
        assert all(s > 0 for s in hlu.LATENTS_STD)

    def test_stub_network_receives_the_normalized_input(self, monkeypatch):
        received = {}

        def fake_load_upscaler(device, model_name, weight_name):
            class _StubNet:
                def __call__(self, x, scale, target_size):
                    received["x"] = x
                    received["scale"] = scale
                    received["target_size"] = target_size
                    return torch.zeros(
                        x.shape[0], LATENT_CHANNELS, *target_size, dtype=x.dtype
                    )

            return _StubNet()

        monkeypatch.setattr(hlu, "_load_upscaler", fake_load_upscaler)

        latents = h3_latents(batch=1, frames=5, height=34, width=60)
        out = upscale_h3_latents(latents, width=1344, height=768, device="cpu")

        mean, std = self._stats()
        assert torch.allclose(received["x"], (latents - mean) / std, atol=1e-6)
        assert received["target_size"] == (5, 48, 84)
        assert received["scale"] == pytest.approx(((48 / 34) + (84 / 60)) / 2)

        assert out.shape == (1, LATENT_CHANNELS, 5, 768 // 16, 1344 // 16)
        # A network answering zeros answers the mean once denormalized
        assert torch.allclose(out, mean.expand_as(out), atol=1e-6)

    def test_a_network_returning_its_input_round_trips(self, monkeypatch):
        class _Identity:
            def __call__(self, x, scale, target_size):
                return x

        monkeypatch.setattr(hlu, "_load_upscaler", lambda *a: _Identity())
        monkeypatch.setattr(
            hlu,
            "target_latent_size",
            lambda latents, w, h: (tuple(latents.shape[-3:]), (1.0, 1.0)),
        )
        latents = h3_latents(batch=1, frames=3, height=34, width=60)
        out = upscale_h3_latents(latents, width=960, height=544, device="cpu")
        assert torch.allclose(out, latents, atol=1e-5)

    def test_output_dtype_and_device_match_the_input(self, monkeypatch):
        def fake_load_upscaler(device, model_name, weight_name):
            class _StubNet:
                def __call__(self, x, scale, target_size):
                    return torch.zeros(
                        x.shape[0], LATENT_CHANNELS, *target_size, dtype=x.dtype
                    )

            return _StubNet()

        monkeypatch.setattr(hlu, "_load_upscaler", fake_load_upscaler)

        latents = h3_latents(height=34, width=60, dtype=torch.float32)
        out = upscale_h3_latents(latents, width=960, height=544, device="cpu")
        assert out.dtype == latents.dtype
        assert out.device == latents.device


class TestUpscalerRevisionPin:
    def test_revision_is_a_40_hex_sha(self):
        assert re.fullmatch(r"[0-9a-f]{40}", UPSCALER_REVISION)

    def test_load_upscaler_pins_the_default_repo_to_the_revision(
        self, monkeypatch, tmp_path
    ):
        from dw.tasks import h3_latent_upscaler_model

        captured = {}

        # Write a tiny, architecturally-correct checkpoint so build_upscaler
        # succeeds against a small (64-channel) model rather than the real
        # 512-wide one
        small_model = LatentResizer3D(channels=64)
        weights_path = tmp_path / "weights.safetensors"
        from safetensors.torch import save_file

        save_file(small_model.state_dict(), str(weights_path))

        monkeypatch.setattr(
            h3_latent_upscaler_model,
            "LatentResizer3D",
            lambda: LatentResizer3D(channels=64),
        )

        def fake_hf_hub_download(repo_id, filename, subfolder, revision):
            captured["repo_id"] = repo_id
            captured["filename"] = filename
            captured["subfolder"] = subfolder
            captured["revision"] = revision
            return str(weights_path)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

        net = hlu._load_upscaler("cpu", DEFAULT_UPSCALER_REPO, "weights.safetensors")

        assert captured["revision"] == UPSCALER_REVISION
        assert captured["subfolder"] == hlu.DEFAULT_UPSCALER_SUBFOLDER
        assert captured["repo_id"] == DEFAULT_UPSCALER_REPO
        assert isinstance(net, LatentResizer3D)

    def test_a_non_default_repo_is_read_at_its_own_main(self, monkeypatch, tmp_path):
        from dw.tasks import h3_latent_upscaler_model

        captured = {}
        small_model = LatentResizer3D(channels=64)
        weights_path = tmp_path / "weights.safetensors"
        from safetensors.torch import save_file

        save_file(small_model.state_dict(), str(weights_path))

        monkeypatch.setattr(
            h3_latent_upscaler_model,
            "LatentResizer3D",
            lambda: LatentResizer3D(channels=64),
        )

        def fake_hf_hub_download(repo_id, filename, subfolder, revision):
            captured["subfolder"] = subfolder
            captured["revision"] = revision
            return str(weights_path)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

        hlu._load_upscaler("cpu", "someone/other-repo", "weights.safetensors")
        assert captured["revision"] is None
        assert captured["subfolder"] is None


class TestArchitectureMismatch:
    def test_matching_state_dict_returns_none(self):
        model = LatentResizer3D(channels=64)
        other = LatentResizer3D(channels=64)
        assert architecture_mismatch(other.state_dict(), model) is None

    def test_missing_keys_are_reported(self):
        model = LatentResizer3D(channels=64)
        state_dict = dict(model.state_dict())
        del state_dict["conv_in.weight"]
        message = architecture_mismatch(state_dict, model)
        assert message is not None
        assert "missing" in message
        assert "conv_in.weight" in message

    def test_unexpected_keys_are_reported(self):
        model = LatentResizer3D(channels=64)
        state_dict = dict(model.state_dict())
        state_dict["not_a_real_key"] = torch.zeros(1)
        message = architecture_mismatch(state_dict, model)
        assert message is not None
        assert "unexpected" in message
        assert "not_a_real_key" in message

    def test_reshaped_keys_are_reported(self):
        model = LatentResizer3D(channels=64)
        state_dict = dict(model.state_dict())
        state_dict["conv_in.weight"] = state_dict["conv_in.weight"][:1]
        message = architecture_mismatch(state_dict, model)
        assert message is not None
        assert "another shape" in message
        assert "conv_in.weight" in message

    def test_strip_prefix_removes_the_wrapper_name(self):
        state_dict = {
            "upscaler.conv_in.weight": torch.zeros(1),
            "no_prefix": torch.zeros(1),
        }
        stripped = strip_prefix(state_dict)
        assert set(stripped) == {"conv_in.weight", "no_prefix"}

    def test_build_upscaler_refuses_a_mismatched_state_dict(self, monkeypatch):
        from dw.tasks import h3_latent_upscaler_model

        # Swap the full 512-wide architecture for a small one so constructing
        # the "expected" model inside build_upscaler is cheap
        monkeypatch.setattr(
            h3_latent_upscaler_model,
            "LatentResizer3D",
            lambda: LatentResizer3D(channels=64),
        )

        wrong_state_dict = {"upscaler.totally_wrong_key": torch.zeros(1)}
        with pytest.raises(ValueError, match="weight_name"):
            build_upscaler(wrong_state_dict)

    def test_build_upscaler_succeeds_with_a_matching_state_dict(self, monkeypatch):
        from dw.tasks import h3_latent_upscaler_model

        monkeypatch.setattr(
            h3_latent_upscaler_model,
            "LatentResizer3D",
            lambda: LatentResizer3D(channels=64),
        )
        good_state_dict = LatentResizer3D(channels=64).state_dict()
        net = build_upscaler(good_state_dict)
        assert isinstance(net, LatentResizer3D)
        assert not net.training


class TestLatentResizer3DNetwork:
    def test_short_clip_upscales_spatially(self):
        net = LatentResizer3D(channels=64).eval()
        x = torch.randn(1, LATENT_CHANNELS, 8, 4, 6)
        with torch.no_grad():
            out = net(x, scale=1.5, target_size=(8, 6, 9))
        assert out.shape == (1, LATENT_CHANNELS, 8, 6, 9)
        assert torch.isfinite(out).all()

    def test_long_clip_uses_the_chunked_path(self):
        net = LatentResizer3D(channels=64).eval()
        x = torch.randn(1, LATENT_CHANNELS, 40, 4, 6)
        with torch.no_grad():
            out = net(x, scale=1.5, target_size=(40, 6, 9))
        assert out.shape == (1, LATENT_CHANNELS, 40, 6, 9)
        assert torch.isfinite(out).all()

    def test_same_size_returns_the_input_unchanged(self):
        net = LatentResizer3D(channels=64).eval()
        x = torch.randn(1, LATENT_CHANNELS, 8, 4, 6)
        out = net(x, scale=1.0, target_size=(8, 4, 6))
        assert out is x


class TestDecodeH3Latents:
    def test_calls_the_decode_block_with_the_expected_arguments(self, monkeypatch):
        recorded = {}
        fake_frames = [[Image.new("RGB", (16, 16))] * 3]

        class FakePipeline:
            _execution_device = torch.device("cpu")

            def __call__(self, **kwargs):
                recorded.update(kwargs)
                return fake_frames

        monkeypatch.setattr(
            hlu, "_load_decoder", lambda device, model_name: FakePipeline()
        )

        latents = h3_latents(frames=3, height=34, width=60)
        result = decode_h3_latents(latents, device="cpu")

        assert recorded["output_type"] == "pil"
        assert recorded["output"] == "videos"
        assert recorded["latents"].dtype == torch.float32
        assert torch.equal(recorded["latents"], latents.to(torch.float32))

        assert isinstance(result, AudioVideo)
        assert result.frames == fake_frames[0]
        assert result.fps == 24
        assert result.audio is None

    def test_load_decoder_builds_from_the_h3_video_decode_step(self, monkeypatch):
        import diffusers.modular_pipelines.minimax_h3.decoders as decoders_module

        recorded = {}

        class FakePipeline:
            def load_components(self, names, torch_dtype):
                recorded["names"] = names
                recorded["torch_dtype"] = torch_dtype

            def to(self, device):
                recorded["device"] = device
                return self

        class FakeDecodeStep:
            def init_pipeline(self, name):
                recorded["repo"] = name
                return FakePipeline()

        monkeypatch.setattr(decoders_module, "MiniMaxH3VideoDecodeStep", FakeDecodeStep)

        pipeline = hlu._load_decoder("cpu", DEFAULT_H3_REPO)

        assert recorded["repo"] == DEFAULT_H3_REPO == "MiniMaxAI/MiniMax-H3"
        assert recorded["names"] == ["vae"]
        assert recorded["torch_dtype"] == torch.float32
        assert recorded["device"] == "cpu"
        assert isinstance(pipeline, FakePipeline)


class TestRegistration:
    def test_both_commands_are_registered_with_summaries(self):
        from dw.tasks.task import _COMMAND_INFO

        for command in ("upscale_h3_latents", "decode_h3_latents"):
            assert command in _COMMAND_INFO
            assert _COMMAND_INFO[command]["summary"]

    def test_upscale_h3_latents_domains_are_positive(self):
        from dw.task_domains import POSITIVE, TASK_ARGUMENT_DOMAINS

        domains = TASK_ARGUMENT_DOMAINS["upscale_h3_latents"]
        assert domains["width"] == POSITIVE
        assert domains["height"] == POSITIVE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
