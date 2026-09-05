import pytest
import torch
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np


def _make_test_frames(count=4, width=64, height=64):
    """Create a list of test frames with different colors."""
    frames = []
    for i in range(count):
        shade = int(255 * i / max(count - 1, 1))
        frames.append(Image.new("RGB", (width, height), color=(shade, shade, shade)))
    return frames


class TestInterpolateFrames:
    """Test interpolate_frames function."""

    @patch("dw.tasks.interpolate_frames._load_rife_model")
    def test_2x_doubles_frame_count(self, mock_load):
        """2x multiplier should produce 2N-1 frames from N input frames."""
        from dw.tasks.interpolate_frames import interpolate_frames

        mock_model = MagicMock()

        def fake_inference(img1, img2):
            arr1 = np.array(img1).astype(np.float32)
            arr2 = np.array(img2).astype(np.float32)
            mid = ((arr1 + arr2) / 2).astype(np.uint8)
            return Image.fromarray(mid)

        mock_model.side_effect = fake_inference
        mock_load.return_value = mock_model

        frames = _make_test_frames(4)
        result = interpolate_frames(frames, multiplier=2)

        # 4 frames with 2x: (4-1)*2 + 1 = 7
        assert len(result.frames) == 7
        assert all(isinstance(f, Image.Image) for f in result.frames)

    @patch("dw.tasks.interpolate_frames._load_rife_model")
    def test_an_audio_video_unwraps_to_its_frames(self, mock_load):
        """A concat or dissolve step's AudioVideo interpolates like a frame list."""
        from dw.result import AudioVideo
        from dw.tasks.interpolate_frames import interpolate_frames

        mock_model = MagicMock(side_effect=lambda a, b: a)
        mock_load.return_value = mock_model

        video = AudioVideo(_make_test_frames(3), np.zeros((2, 100)), 100)
        result = interpolate_frames(video, multiplier=2)

        assert len(result.frames) == 5
        assert all(isinstance(f, Image.Image) for f in result.frames)

    @patch("dw.tasks.interpolate_frames._load_rife_model")
    def test_4x_quadruples_frame_count(self, mock_load):
        """4x multiplier should run two passes of 2x."""
        from dw.tasks.interpolate_frames import interpolate_frames

        mock_model = MagicMock()

        def fake_inference(img1, img2):
            arr1 = np.array(img1).astype(np.float32)
            arr2 = np.array(img2).astype(np.float32)
            mid = ((arr1 + arr2) / 2).astype(np.uint8)
            return Image.fromarray(mid)

        mock_model.side_effect = fake_inference
        mock_load.return_value = mock_model

        frames = _make_test_frames(4)
        result = interpolate_frames(frames, multiplier=4)

        # Two passes of 2x: 4 -> 7 -> 13
        assert len(result.frames) == 13
        assert all(isinstance(f, Image.Image) for f in result.frames)

    def test_invalid_multiplier_raises(self):
        """multiplier must be 2, 4, or 8."""
        from dw.tasks.interpolate_frames import interpolate_frames

        with pytest.raises(ValueError, match="multiplier"):
            interpolate_frames(_make_test_frames(2), multiplier=3)

    def test_single_frame_raises(self):
        """Need at least 2 frames to interpolate."""
        from dw.tasks.interpolate_frames import interpolate_frames

        with pytest.raises(ValueError, match="at least 2"):
            interpolate_frames(_make_test_frames(1), multiplier=2)


class TestIFNetArchitecture:
    def test_forward_channel_widths_are_self_consistent(self):
        """The real IFNet must run a forward pass with randomly initialized
        weights — the declared IFBlock input widths must match what forward()
        actually concatenates (this once regressed: a mangled encode head made
        block0 receive 39 channels instead of its declared 23)."""
        from dw.tasks.rife_model import IFNet
        from dw.tasks.interpolate_frames import _make_flow_context

        net = IFNet().eval()
        h = w = 64
        img0 = torch.rand(1, 3, h, w)
        img1 = torch.rand(1, 3, h, w)
        tenFlow_div, backwarp_tenGrid, timestep = _make_flow_context(h, w, "cpu")

        with torch.inference_mode():
            flow_list, mask_list, merged = net(
                img0, img1, timestep, [8, 4, 2, 1], tenFlow_div, backwarp_tenGrid
            )

        assert len(merged) == 4
        assert merged[3].shape == (1, 3, h, w)

    def test_matches_v4_13_checkpoint_layout(self):
        """Parameter names and shapes must match the v4.13.2 checkpoint the
        loader downloads, or load_state_dict would reject the real weights."""
        from dw.tasks.rife_model import IFNet

        shapes = {k: tuple(v.shape) for k, v in IFNet().state_dict().items()}
        # Spot-check the shapes that pin the architecture version
        assert shapes["block0.conv0.0.0.weight"] == (96, 23, 3, 3)
        assert shapes["block1.conv0.0.0.weight"] == (64, 28, 3, 3)
        assert shapes["block0.lastconv.0.weight"] == (192, 24, 4, 4)
        assert shapes["encode.cnn0.weight"] == (32, 3, 3, 3)
        assert shapes["encode.cnn3.weight"] == (32, 8, 4, 4)


class TestInterpolateFramesRegistration:
    def test_interpolate_frames_command_registered(self):
        from dw.tasks.task import _COMMAND_REGISTRY

        assert "interpolate_frames" in _COMMAND_REGISTRY


class _FakeRifeNet:
    """Stand-in for the real IFNet with the same call signature.

    Exercises the real padding / flow-context / tensor-carry plumbing in
    `_build_inference` deterministically, without depending on a
    randomly-initialized IFNet's output.
    """

    def __call__(self, img0, img1, timestep, scale_list, tenFlow_div, backwarp_tenGrid):
        # A trivial deterministic function of the (already padded) inputs,
        # shaped like the real net's return: (flow_list, mask_list, merged).
        mid = (img0[:, :3] + img1[:, :3]) / 2.0 + timestep[:, 0:1] * 0
        merged = [mid, mid, mid, mid]
        return [], [], merged


class TestBuildInferenceEfficiency:
    """Cover the hoisted flow-context / carried-tensor optimization in
    dw.tasks.interpolate_frames._build_inference: it must be a pure
    perf change — deterministic, correctly shaped output — while avoiding
    a redundant PIL->tensor conversion for interior frames."""

    def test_output_is_deterministic_and_correctly_shaped(self):
        """Same net + same frames -> bit-identical output, correct size/type."""
        from dw.tasks.interpolate_frames import _build_inference, _interpolate_2x

        net = _FakeRifeNet()
        frames = _make_test_frames(4, width=64, height=64)

        inference_a = _build_inference(net, "cpu")
        result_a = _interpolate_2x(frames, inference_a)

        inference_b = _build_inference(net, "cpu")
        result_b = _interpolate_2x(frames, inference_b)

        # 4 frames, one pass of 2x -> 7 frames
        assert len(result_a) == 7 == len(result_b)
        assert all(isinstance(f, Image.Image) for f in result_a)
        assert all(f.size == (64, 64) for f in result_a)

        for fa, fb in zip(result_a, result_b):
            assert np.array_equal(np.array(fa), np.array(fb))

    def test_interior_frame_converted_once_not_twice(self):
        """Each interior frame is img2 of one pair and img1 of the next.

        Before the fix, _pil_to_tensor ran twice for such frames (once per
        pair). The carry-forward cache in _build_inference should collapse
        that to a single conversion per frame.
        """
        import dw.tasks.interpolate_frames as mod

        net = _FakeRifeNet()
        frames = _make_test_frames(4, width=64, height=64)

        real_pil_to_tensor = mod._pil_to_tensor
        with patch.object(mod, "_pil_to_tensor", side_effect=real_pil_to_tensor) as spy:
            inference = mod._build_inference(net, "cpu")
            mod._interpolate_2x(frames, inference)

        # 4 frames -> 3 pairs. Without the carry-forward optimization this
        # would be 2 conversions per pair == 6 calls. With it, each of the
        # 4 distinct frame objects is converted exactly once.
        assert spy.call_count == len(frames)
