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
        from dw.media_types import AudioVideo
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


# --- refill: "held" (#1527 stage C, #1553) ---------------------------------


def _pattern(phase, shot=0, width=128, height=72):
    """A frame of a sliding sinusoid: consecutive phases differ by a few grey
    levels (moving, no cut); shot 1 recolours it, a hard cut from shot 0."""
    x = np.arange(width)[None, :, None]
    y = np.arange(height)[:, None, None]
    base = 127 + 100 * np.sin((x + phase) / 9.0 + y / 23.0)
    if shot == 0:
        rgb = np.concatenate([base, base * 0.8, 255 - base], axis=2)
    else:
        rgb = np.concatenate([255 - base, 60 + 0 * base, base * 0.5], axis=2)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def _moving(count, shot=0):
    return [_pattern(3 * i, shot) for i in range(count)]


def _take(extra=0):
    """Two shots (a cut at 12) with a single hold at 5 and a frame shown four
    times at 15..18 - the measure flags 16 and 18, not the middle 17."""
    frames = _moving(12) + _moving(12 + extra, shot=1)
    frames[5] = frames[4]
    frames[16] = frames[17] = frames[18] = frames[15]
    return frames


class _Recorder:
    """A stand-in interpolator recording its (a, b, t) calls, answering a
    new frame that is the blend of a and b at t."""

    def __init__(self):
        self.calls = []
        self.loads = 0

    def load(self, *args, **kwargs):
        self.loads += 1
        return self

    def __call__(self, a, b, t=0.5):
        self.calls.append((a, b, t))
        return Image.blend(a, b, t)


def _refill(video, recorder):
    import dw.tasks.interpolate_frames as mod

    with patch.object(mod, "_load_rife_model", side_effect=recorder.load):
        return mod.interpolate_frames(video, refill="held")


def _same(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


class TestHeldRuns:
    def test_a_frame_shown_four_times_is_one_run_of_three(self):
        from dw.tasks.assess import frames_media
        from dw.tasks.held_frames import held_measure
        from dw.tasks.interpolate_frames import held_runs

        frames = _take()
        d, held, cuts = held_measure(frames_media(frames))
        assert held == [5, 16, 18] and cuts == [12]
        refill, left = held_runs(d, held, cuts, len(frames))
        assert refill == [(5, 5), (16, 18)]
        assert left == []


class TestRefillHeld:
    @pytest.mark.parametrize("extra", [0, 1])
    def test_frame_count_is_the_sources_for_even_and_odd_lengths(self, extra):
        frames = _take(extra)
        assert len(frames) % 2 == extra
        result = _refill(frames, _Recorder())
        assert len(result.frames) == len(frames)

    def test_single_hold_at_half_and_a_three_frame_run_at_quarters(self):
        frames = _take()
        recorder = _Recorder()
        _refill(frames, recorder)
        assert recorder.loads == 1
        got = [(id(a), id(b), t) for a, b, t in recorder.calls]
        assert got == [
            (id(frames[4]), id(frames[6]), 0.5),
            (id(frames[15]), id(frames[19]), 0.25),
            (id(frames[15]), id(frames[19]), 0.5),
            (id(frames[15]), id(frames[19]), 0.75),
        ]

    def test_every_frame_but_the_runs_passes_through_byte_equal(self):
        frames = _take()
        source = [np.asarray(f).copy() for f in frames]
        result = _refill(frames, _Recorder())
        refilled = {5, 16, 17, 18}
        for i, frame in enumerate(result.frames):
            if i in refilled:
                assert not _same(frame, source[i])
            else:
                # The cut frame and its neighbours included
                assert _same(frame, source[i]), i
        assert result.held_before == 3
        assert result.held_after == 0

    def test_a_run_whose_next_moving_frame_is_across_a_cut_stays_held(self):
        frames = _moving(12) + _moving(12, shot=1)
        frames[10] = frames[11] = frames[9]
        recorder = _Recorder()
        result = _refill(frames, recorder)
        assert recorder.calls == []
        assert all(_same(r, f) for r, f in zip(result.frames, frames))
        assert result.held_before == result.held_after == 1

    def test_a_run_at_the_clips_end_stays_held_and_counts(self):
        frames = _moving(12)
        frames[5] = frames[4]
        frames[11] = frames[10]
        recorder = _Recorder()
        result = _refill(frames, recorder)
        assert [t for _, _, t in recorder.calls] == [0.5]
        assert _same(result.frames[11], frames[10])
        assert result.held_before == 2
        assert result.held_after == 1

    def test_no_holds_comes_back_unchanged_without_loading_rife(self):
        frames = _moving(10)
        recorder = _Recorder()
        result = _refill(frames, recorder)
        assert recorder.loads == 0
        assert all(r is f for r, f in zip(result.frames, frames))
        assert result.held_before == result.held_after == 0

    def test_audio_fps_and_shots_are_the_sources(self):
        from dw.media_types import AudioVideo

        audio = np.zeros((2, 48000), dtype=np.float32)
        shots = [{"start_frame": 0, "num_frames": 12}]
        video = AudioVideo(_take(), audio, 48000, fps=24, shots=shots)
        result = _refill(video, _Recorder())
        assert result.audio is audio
        assert result.sample_rate == 48000
        assert result.fps == 24
        assert result.shots == shots

    def test_held_counts_go_out_as_a_log_event(self):
        import dw.events

        with patch.object(dw.events, "emit_log") as emit:
            _refill(_take(), _Recorder())
        data = emit.call_args.kwargs
        assert data["command"] == "interpolate_frames"
        assert data["held_before"] == 3 and data["held_after"] == 0
        assert data["refilled_frames"] == 4 and data["refilled_runs"] == 2

    def test_real_path_with_the_net_stubbed_at_the_load_seam(self):
        """Through interpolate_frames, _load_rife_model and _build_inference,
        with only the IFNet stood in for: the refilled frame is the net's
        answer at the run's timestep."""
        import dw.tasks.interpolate_frames as mod
        from dw.tasks import model_cache

        class BlendNet:
            def __call__(self, img0, img1, timestep, *rest):
                t = timestep[:, 0:1]
                mid = img0[:, :3] * (1 - t) + img1[:, :3] * t
                return [], [], [mid, mid, mid, mid]

        frames = _take()
        with (
            patch.object(model_cache, "cached_model", return_value=BlendNet()),
            patch("huggingface_hub.hf_hub_download") as download,
        ):
            result = mod.interpolate_frames(frames, refill="held")
        download.assert_not_called()
        expected = Image.blend(frames[15], frames[19], 0.25)
        diff = np.abs(
            np.asarray(result.frames[16], dtype=int) - np.asarray(expected, dtype=int)
        )
        assert diff.max() <= 1
        assert result.held_before > result.held_after


class TestRefillArguments:
    def test_refill_with_multiplier_is_refused_at_run_time(self):
        from dw.tasks.interpolate_frames import interpolate_frames

        with pytest.raises(ValueError, match="'refill' and 'multiplier'"):
            interpolate_frames(_moving(4), refill="held", multiplier=2)

    @staticmethod
    def _errors(arguments):
        from dw.task_domains import task_argument_errors

        return task_argument_errors(
            {
                "id": "refill",
                "steps": [
                    {
                        "name": "refill",
                        "task": {
                            "command": "interpolate_frames",
                            "arguments": arguments,
                        },
                    }
                ],
            }
        )

    def test_refill_with_multiplier_is_refused_at_validate(self):
        errors = self._errors(
            {"video": "asset:take.mp4", "refill": "held", "multiplier": 2}
        )
        assert any(
            "refill" in e["message"] and "multiplier" in e["message"] for e in errors
        ), errors

    def test_refill_alone_and_multiplier_alone_validate(self):
        assert self._errors({"video": "asset:take.mp4", "refill": "held"}) == []
        assert self._errors({"video": "asset:take.mp4", "multiplier": 4}) == []

    def test_an_unknown_refill_mode_is_refused_at_validate(self):
        errors = self._errors({"video": "asset:take.mp4", "refill": "all"})
        assert any("refill" in e["message"] for e in errors), errors

    def test_a_file_video_loads_with_its_soundtrack_for_a_refill(self):
        import dw.arguments
        from dw.arguments import realize_args

        loaded = object()
        task = {
            "command": "interpolate_frames",
            "arguments": {"video": "take.mp4", "refill": "held"},
        }
        with patch.object(
            dw.arguments, "fetch_video_with_context", return_value=loaded
        ) as fetch:
            realize_args(task, "/tmp")
        assert fetch.call_args.kwargs.get("with_audio") is True
        assert task["arguments"]["video"] is loaded

    @pytest.mark.parametrize(
        "reference, resolver",
        [
            ("asset:take.mp4", "fetch_asset"),
            ("output:job/run/take.mp4", "fetch_output"),
            ({"location": "asset:take.mp4"}, "fetch_asset"),
        ],
    )
    def test_a_path_reference_video_resolves_before_the_soundtrack_load(
        self, reference, resolver
    ):
        """#1553 bounce: an 'asset:'/'output:' video reached load_audio_video
        as the reference string and failed 'Path does not exist'."""
        import dw.arguments
        import dw.tasks.video_utils
        from dw.arguments import realize_args

        loaded = object()
        task = {
            "command": "interpolate_frames",
            "arguments": {"video": reference, "refill": "held"},
        }
        with (
            patch.object(
                dw.arguments, resolver, return_value="/srv/take.mp4"
            ) as resolve,
            patch.object(
                dw.tasks.video_utils, "load_audio_video", return_value=loaded
            ) as load,
        ):
            realize_args(task, "/tmp")
        resolve.assert_called_once()
        spec = load.call_args.args[0]
        location = spec["location"] if isinstance(spec, dict) else spec
        assert location == "/srv/take.mp4"
        assert task["arguments"]["video"] is loaded


class TestTimestep:
    def test_flow_context_timestep_defaults_to_the_midpoint(self):
        from dw.tasks.interpolate_frames import _make_flow_context

        _, _, plane = _make_flow_context(32, 32, "cpu")
        assert torch.all(plane == 0.5)
        _, _, plane = _make_flow_context(32, 32, "cpu", 0.25)
        assert torch.all(plane == 0.25)

    def test_multiplier_output_runs_the_net_at_half(self):
        from dw.tasks.interpolate_frames import _build_inference, _interpolate_2x

        seen = []

        class Net(_FakeRifeNet):
            def __call__(self, img0, img1, timestep, *rest):
                seen.append(float(timestep.flatten()[0]))
                return super().__call__(img0, img1, timestep, *rest)

        _interpolate_2x(_make_test_frames(4), _build_inference(Net(), "cpu"))
        assert seen == [0.5, 0.5, 0.5]


class TestFlowContextCache:
    def test_many_timesteps_build_one_grid(self, monkeypatch):
        """A long hold is refilled at a timestep per frame; the warp grid is
        the same for all of them, and a full-resolution grid per timestep
        was gigabytes on a freeze of a few seconds."""
        from dw.tasks import interpolate_frames

        built, seen = [], []
        real = interpolate_frames._make_flow_context

        def counting(*args, **kwargs):
            built.append(args)
            return real(*args, **kwargs)

        class Net(_FakeRifeNet):
            def __call__(self, img0, img1, timestep, *rest):
                seen.append(round(float(timestep.flatten()[0]), 3))
                return super().__call__(img0, img1, timestep, *rest)

        monkeypatch.setattr(interpolate_frames, "_make_flow_context", counting)
        inference = interpolate_frames._build_inference(Net(), "cpu")
        first, second = _make_test_frames(2)
        steps = [(j + 1) / 40 for j in range(39)]
        for timestep in steps:
            inference(first, second, timestep)

        assert len(built) == 1
        assert seen == [round(step, 3) for step in steps]
