"""The experimental time-windowed H3 refine (#1199 Task 17 Step 4): the window
plan, the fuse weights, the windowed denoiser against the stock one, and the
refusals. dw/pipeline_processors/h3_windows.py."""

import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

minimax = pytest.importorskip("diffusers.modular_pipelines.minimax_h3")
from diffusers.modular_pipelines.minimax_h3.before_denoise import (  # noqa: E402
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
)
from diffusers.modular_pipelines.minimax_h3.denoise import (  # noqa: E402
    MiniMaxH3LoopDenoiser,
    MiniMaxH3LoopSchedulerStep,
)
from diffusers.schedulers import MiniMaxH3Scheduler  # noqa: E402

from dw.pipeline_processors import h3_rules, h3_windows  # noqa: E402
from dw.pipeline_processors.h3_hold import insert_audio_hold  # noqa: E402
from dw.pipeline_processors.h3_mask import insert_mask_blocks  # noqa: E402
from dw.pipeline_processors.h3_windows import (  # noqa: E402
    DENOISER,
    ENABLE_ENV_VAR,
    REFINE_WINDOWS_INPUT,
    DwH3WindowedLoopDenoiser,
    fuse_weights,
    insert_refine_windows,
    window_plan,
    window_problems,
    window_rows,
    windows_refine,
    windows_refusal,
)

# 1. The plan

PLANS = [(37, 12, 4), (37, 16, 6), (12, 6, 2), (11, 6, 2), (10, 6, 2), (52, 17, 5)]


@pytest.mark.parametrize("frames, window, overlap", PLANS)
class TestWindowPlan:
    def test_every_latent_is_covered(self, frames, window, overlap):
        covered = set()
        for start, stop in window_plan(frames, window, overlap):
            covered.update(range(start, stop))
        assert covered == set(range(frames))

    def test_windows_are_whole_latents_of_the_window_size(
        self, frames, window, overlap
    ):
        for start, stop in window_plan(frames, window, overlap):
            assert isinstance(start, int) and isinstance(stop, int)
            assert stop - start == window

    def test_the_overlaps_are_the_overlap_and_the_last_at_least_it(
        self, frames, window, overlap
    ):
        plan = window_plan(frames, window, overlap)
        shared = [a[1] - b[0] for a, b in zip(plan, plan[1:])]
        assert all(value == overlap for value in shared[:-1])
        assert shared[-1] >= overlap

    def test_the_last_window_ends_on_the_last_latent(self, frames, window, overlap):
        assert window_plan(frames, window, overlap)[-1][1] == frames


def test_a_take_no_longer_than_a_window_is_one_window():
    assert window_plan(12, 12, 4) == [(0, 12)]
    assert window_plan(12, 40, 4) == [(0, 12)]


def test_an_exact_fit_does_not_repeat_the_last_window():
    assert window_plan(10, 6, 2) == [(0, 6), (4, 10)]


@pytest.mark.parametrize(
    "window, overlap", [(0, 0), (6, 6), (6, -1), (6.0, 2), (True, 0), (6, None)]
)
def test_bad_sizes_are_refused(window, overlap):
    with pytest.raises(ValueError):
        window_plan(20, window, overlap)


# 2. The weights


@pytest.mark.parametrize("blend", ["pyramid", "cosine"])
@pytest.mark.parametrize("frames, window, overlap", PLANS + [(20, 6, 0)])
def test_the_weights_sum_to_one_at_every_latent(frames, window, overlap, blend):
    plan = window_plan(frames, window, overlap)
    total = torch.zeros(frames, dtype=torch.float64)
    for (start, stop), weights in zip(plan, fuse_weights(plan, frames, blend)):
        assert weights.shape == (stop - start,)
        assert bool((weights > 0).all())
        total[start:stop] += weights
    assert torch.allclose(total, torch.ones(frames, dtype=torch.float64), atol=1e-12)


def test_a_regular_overlap_is_a_complementary_linear_ramp():
    plan = [(0, 6), (4, 10)]
    first, second = fuse_weights(plan, 10)
    assert first.tolist() == pytest.approx([1, 1, 1, 1, 2 / 3, 1 / 3])
    assert second.tolist() == pytest.approx([1 / 3, 2 / 3, 1, 1, 1, 1])


def test_two_windows_weigh_the_literal_ramp_the_blend_test_uses():
    """The plan the hand-computed blend runs on, by value: 2 shared latents,
    so the ramp is 1/3, 2/3 - pinned here so the blend test does not only
    check fuse_weights against itself."""
    first, second = fuse_weights([(0, 7), (5, 12)], 12)
    assert first.tolist() == pytest.approx([1, 1, 1, 1, 1, 2 / 3, 1 / 3])
    assert second.tolist() == pytest.approx([1 / 3, 2 / 3, 1, 1, 1, 1, 1])


def test_the_cosine_ramp_is_complementary_too():
    first, second = fuse_weights([(0, 6), (3, 9)], 9, "cosine")
    assert (first[3:] + second[:3]).tolist() == pytest.approx([1, 1, 1])
    assert first[3] > first[4] > first[5]


def test_one_window_weighs_one_everywhere():
    (weights,) = fuse_weights([(0, 7)], 7)
    assert torch.equal(weights, torch.ones(7, dtype=torch.float64))


def test_an_unknown_blend_is_refused():
    with pytest.raises(ValueError, match="blend"):
        fuse_weights([(0, 4)], 4, "gaussian")


# 3. The windowed denoiser on a real layout

TEXT_ROWS = 3
# 17 * 2 + 5 frames -> 5 * 2 + 2 latent frames, 65 audio latents a channel
NUM_FRAMES, LATENT_FRAMES, AUDIO_LATENTS = 39, 12, 65
VIDEO_WIDTH, AUDIO_WIDTH, TEXT_WIDTH = 96, 32, 16


def layout(latent_height=2, latent_width=4):
    """The real t2va packed layout, `(fields, num_video_rows, num_audio_rows)`:
    2 rows a frame at 2x4 latents under the (1, 2, 2) patch."""
    (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        _,
        _,
    ) = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.ones(TEXT_ROWS, dtype=torch.long),
        LATENT_FRAMES,
        latent_height,
        latent_width,
        AUDIO_LATENTS,
        (1, 2, 2),
        2,
        2,
        0,
    )
    fields = {
        "position_ids": position_ids,
        "token_tags": token_tags,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
    }
    return fields, video_indices.numel(), audio_indices.numel()


def block_state(settings=None, steps=3, seed=0, **layout_size):
    """What the loop hands its denoiser: a t2va refine with every audio row
    held, its row plan built by the stock rule."""
    fields, video_rows, audio_rows = layout(**layout_size)
    generator = torch.Generator().manual_seed(seed)
    scheduler = MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(steps)
    plan = [
        MiniMaxH3SetTimestepsStep.build_row_timesteps(
            fields["video_indices"],
            fields["audio_indices"],
            0,
            audio_rows,
            TEXT_ROWS,
            float(timestep),
            float(timestep),
            float(timestep),
            1.0,
        )
        for timestep in scheduler.timesteps
    ]
    state = SimpleNamespace(
        latents=torch.randn(video_rows, VIDEO_WIDTH, generator=generator),
        audio_latents=torch.randn(audio_rows, AUDIO_WIDTH, generator=generator),
        prompt_embeds=torch.randn(1, TEXT_ROWS, TEXT_WIDTH, generator=generator),
        row_timestep_plan=plan,
        denoiser_input_fields=fields,
        attention_kwargs=None,
        num_latent_frames=LATENT_FRAMES,
        num_condition_video_rows=0,
        num_condition_audio_rows=audio_rows,
        dw_held_audio_rows=audio_rows,
        audio_timesteps=scheduler.timesteps,
    )
    setattr(state, REFINE_WINDOWS_INPUT, settings)
    return state, scheduler


def tiny_transformer():
    from diffusers import MiniMaxH3Transformer3DModel

    torch.manual_seed(0)
    return MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=32,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=32,
        text_dim=TEXT_WIDTH,
        freq_dim=16,
        time_embed_hidden_dim=16,
        time_embed_dim=16,
        rope_freq_dim=2,
    ).eval()


def denoise_loop(denoiser, settings, steps=3):
    """Every step of the loop - `denoiser` then the stock `update` - from the
    same start, on the tiny transformer: the final `(latents, audio_latents)`."""
    state, scheduler = block_state(settings, steps)
    components = SimpleNamespace(
        transformer=tiny_transformer(),
        scheduler=scheduler,
        audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
    )
    components.audio_scheduler.set_timesteps(steps)
    update = MiniMaxH3LoopSchedulerStep()
    for i, t in enumerate(scheduler.timesteps):
        denoiser(components, state, i, t)
        update(components, state, i, t)
    return state.latents, state.audio_latents


@pytest.mark.parametrize("frames", [LATENT_FRAMES, LATENT_FRAMES + 9])
def test_one_window_over_the_take_is_bit_identical_to_the_stock_step(frames):
    stock = denoise_loop(MiniMaxH3LoopDenoiser(), None)
    windowed = denoise_loop(
        DwH3WindowedLoopDenoiser(), {"frames": frames, "overlap": 2}
    )
    assert torch.equal(stock[0], windowed[0])
    assert torch.equal(stock[1], windowed[1])


def test_without_the_argument_the_windowed_denoiser_is_the_stock_one():
    stock = denoise_loop(MiniMaxH3LoopDenoiser(), None)
    windowed = denoise_loop(DwH3WindowedLoopDenoiser(), None)
    assert torch.equal(stock[0], windowed[0])


def test_two_windows_change_the_refine_and_leave_the_audio_held():
    stock = denoise_loop(MiniMaxH3LoopDenoiser(), None)
    windowed = denoise_loop(DwH3WindowedLoopDenoiser(), {"frames": 7, "overlap": 2})
    assert not torch.equal(stock[0], windowed[0])
    assert torch.isfinite(windowed[0]).all()
    assert torch.equal(stock[1], windowed[1])


class StubTransformer(torch.nn.Module):
    """A transformer whose video velocity is a known function of what it is
    shown: each row's own value plus the mean of every video row in the
    window, plus its rotary time - so a window's context and positions both
    show in what it predicts. Records every call."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
        attention_kwargs=None,
        return_dict=True,
    ):
        self.calls.append(
            dict(
                position_ids=position_ids,
                token_tags=token_tags,
                video_indices=video_indices,
                audio_indices=audio_indices,
                text_indices=text_indices,
                timestep_indices=timestep_indices,
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
            )
        )
        times = position_ids[video_indices, 0].float()[None, :, None]
        video = hidden_states + hidden_states.mean(dim=1, keepdim=True) + times
        return video, torch.full_like(audio_hidden_states, float("nan"))


def stub_step(settings):
    state, scheduler = block_state(settings)
    stub = StubTransformer()
    DwH3WindowedLoopDenoiser()(
        SimpleNamespace(transformer=stub), state, 0, scheduler.timesteps[0]
    )
    return state, stub


def frame_rows(per_frame=2):
    """The latents rows of each latent frame (2 rows a frame)."""
    return lambda frame: slice(frame * per_frame, (frame + 1) * per_frame)


@pytest.mark.parametrize("positions", ["true", "rebased"])
def test_two_windows_fuse_to_the_hand_computed_blend(positions):
    settings = {"frames": 7, "overlap": 2, "positions": positions}
    state, stub = stub_step(settings)
    plan = window_plan(LATENT_FRAMES, 7, 2)
    assert plan == [(0, 7), (5, 12)]
    assert len(stub.calls) == 2

    x = state.latents
    times = state.denoiser_input_fields["position_ids"][
        state.denoiser_input_fields["video_indices"], 0
    ].float()
    rows = frame_rows()
    origin = times[0]
    expected = torch.empty_like(x)
    w_first, w_second = fuse_weights(plan, LATENT_FRAMES)
    for frame in range(LATENT_FRAMES):
        r = rows(frame)
        predictions, weights = [], []
        for (start, stop), w in zip(plan, (w_first, w_second)):
            if start <= frame < stop:
                window = x[start * 2 : stop * 2]
                shift = times[start * 2] - origin if positions == "rebased" else 0
                t = (times[r] - shift)[:, None]
                predictions.append(x[r] + window.mean(dim=0, keepdim=True) + t)
                weights.append(float(w[frame - start]))
        expected[r] = sum(p * w for p, w in zip(predictions, weights)) / sum(weights)

    assert torch.allclose(state.noise_pred[0], expected, atol=1e-5)
    # The audio is held: nothing of the audio prediction reaches the update
    assert torch.equal(
        state.audio_noise_pred, torch.zeros(1, *state.audio_latents.shape)
    )


def test_true_positions_are_the_takes_own_and_rebased_start_at_its_origin():
    state, true_stub = stub_step({"frames": 7, "overlap": 2})
    _, rebased_stub = stub_step({"frames": 7, "overlap": 2, "positions": "rebased"})
    fields = state.denoiser_input_fields
    global_times = fields["position_ids"][fields["video_indices"], 0]
    late_true = true_stub.calls[1]
    late_rebased = rebased_stub.calls[1]
    true_times = late_true["position_ids"][late_true["video_indices"], 0]
    assert torch.equal(true_times, global_times[10:])
    rebased_times = late_rebased["position_ids"][late_rebased["video_indices"], 0]
    assert rebased_times[0] == global_times[0]
    assert torch.allclose(rebased_times - rebased_times[0], true_times - true_times[0])
    # Text never moves
    text = late_rebased["position_ids"][late_rebased["text_indices"]]
    assert torch.equal(text, fields["position_ids"][fields["text_indices"]])


def test_a_window_sees_the_text_its_video_and_the_audio_of_its_span():
    state, stub = stub_step({"frames": 7, "overlap": 2})
    fields = state.denoiser_input_fields
    times = fields["position_ids"][:, 0]
    seen_audio = set()
    for call, (start, stop) in zip(stub.calls, [(0, 7), (5, 12)]):
        assert call["text_indices"].numel() == TEXT_ROWS
        assert call["hidden_states"].shape[1] == (stop - start) * 2
        audio_times = call["position_ids"][call["audio_indices"], 0]
        video_times = call["position_ids"][call["video_indices"], 0]
        assert audio_times.min() >= video_times.min()
        if stop < LATENT_FRAMES:
            next_frame = times[fields["video_indices"][stop * 2]]
            assert audio_times.max() < next_frame
        # Both channels of every latent in the span
        assert call["audio_indices"].numel() % 2 == 0
        # The global order, kept: the target video is the trailing run
        tags = call["token_tags"]
        assert bool((tags[-call["video_indices"].numel() :] == 0).all())
        # Each row keeps its own timestep
        assert call["timestep_indices"].numel() == tags.numel()
        _, audio, _ = window_rows(fields, state.latents.shape[0], 0, 12, (start, stop))
        seen_audio.update(fields["audio_indices"][audio].tolist())
    assert seen_audio == set(fields["audio_indices"].tolist())


def test_an_unheld_audio_row_or_a_conditioning_video_row_is_refused():
    state, scheduler = block_state({"frames": 7, "overlap": 2})
    state.dw_held_audio_rows -= 1
    with pytest.raises(ValueError, match="held"):
        DwH3WindowedLoopDenoiser()(
            SimpleNamespace(transformer=StubTransformer()), state, 0, 0
        )


# 4. The graph change


def pipelines():
    blocks = minimax.MiniMaxH3Blocks()
    result = [blocks.init_pipeline()]
    result += [
        blocks.get_workflow(name).init_pipeline()
        for name in ("t2va", "fl2va", "ref2va")
    ]
    for pipeline in result:
        insert_audio_hold(pipeline)
    return result


def loops(pipeline):
    return list(h3_windows._loops(pipeline))


def test_the_windowed_denoiser_replaces_every_loops_in_place():
    for pipeline in pipelines():
        assert not windows_refine(pipeline)
        before = [
            (list(loop.sub_blocks), loop.sub_blocks[DENOISER].transformer_name)
            for loop in loops(pipeline)
        ]
        assert insert_refine_windows(pipeline)
        assert insert_refine_windows(pipeline)  # idempotent
        assert windows_refine(pipeline)
        after = [
            (list(loop.sub_blocks), loop.sub_blocks[DENOISER].transformer_name)
            for loop in loops(pipeline)
        ]
        assert after == before
        assert REFINE_WINDOWS_INPUT in {p.name for p in pipeline._blocks.inputs}


def test_a_pipeline_that_is_not_h3_takes_no_windows():
    assert not insert_refine_windows(object())


# 5. The refusals


def refine(**overrides):
    arguments = {
        "refine_strength": 0.3,
        "latents": torch.zeros(1),
        "hold_audio": object(),
        REFINE_WINDOWS_INPUT: {"frames": 16, "overlap": 4},
    }
    arguments.update(overrides)
    return {k: v for k, v in arguments.items() if v is not None}


def test_a_valid_call_has_no_problems():
    assert window_problems(refine()) == []
    assert window_problems(refine(latents=None, source_video="take.mp4")) == []
    assert window_problems(refine(**{REFINE_WINDOWS_INPUT: None})) == []


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({REFINE_WINDOWS_INPUT: [16, 4]}, "frames, overlap"),
        ({REFINE_WINDOWS_INPUT: {"frames": 16, "overlap": 16}}, "overlap"),
        ({REFINE_WINDOWS_INPUT: {"overlap": 4}}, "whole number"),
        ({REFINE_WINDOWS_INPUT: {"frames": 16, "size": 4}}, "not size"),
        ({REFINE_WINDOWS_INPUT: {"frames": 16, "blend": "box"}}, "blend"),
        ({REFINE_WINDOWS_INPUT: {"frames": 16, "positions": "x"}}, "positions"),
        ({"refine_strength": None}, "refine_strength"),
        ({"hold_audio": None}, "hold_audio"),
        ({"refine_tiles": {"width": 1344}}, "refine_tiles"),
        ({"latents": None}, "source_video"),
        ({"source_video": "a.mp4", "mask_video": "m.mp4"}, "mask_video"),
        ({"image": "first.png"}, "image"),
        ({"last_image": "last.png"}, "last_image"),
        ({"keyframes": [object()]}, "keyframes"),
        ({"guides": [object()]}, "guides"),
        ({"latent_anchor": "take.mp4"}, "latent_anchor"),
        ({"references": [object()]}, "references"),
        (
            {
                REFINE_WINDOWS_INPUT: {
                    "frames": 12,
                    "overlap": 4,
                    "positions": "rebased",
                }
            },
            "multiple of 5",
        ),
        (
            {
                REFINE_WINDOWS_INPUT: {
                    "frames": 15,
                    "overlap": 5,
                    "positions": "rebased",
                },
                "num_frames": 124,
            },
            "multiple of 5",
        ),
    ],
)
def test_a_call_the_windows_cannot_run_is_refused(overrides, match):
    problems = window_problems(refine(**overrides))
    assert problems and any(match in problem for problem in problems), problems


def test_rebased_positions_on_the_clip_grid_are_taken():
    windows = {"frames": 12, "overlap": 2, "positions": "rebased"}
    arguments = refine(**{REFINE_WINDOWS_INPUT: windows})
    # Stride 10 and no take length known yet: the stride is on the grid
    assert window_problems(arguments) == []
    # 124 frames -> 37 latents: starts 0, 10, 20 and the pulled-back 25
    assert window_problems({**arguments, "num_frames": 124}) == []
    # ...and the same read off a 5-d latents tensor
    latents = torch.zeros(1, 24, 37, 2, 2)
    assert window_problems({**arguments, "latents": latents}) == []
    # 39 frames -> 12 latents: one window
    assert window_problems({**arguments, "num_frames": 39}) == []
    # 107 frames -> 32 latents: starts 0, 10 and the pulled-back 20
    assert window_problems({**arguments, "num_frames": 107}) == []
    # 90 frames -> 27 latents: the pulled-back start is 15
    assert window_problems({**arguments, "num_frames": 90}) == []
    # A stride off the grid with true positions is fine
    assert (
        window_problems(refine(**{REFINE_WINDOWS_INPUT: {"frames": 12, "overlap": 4}}))
        == []
    )


def test_the_rebased_rule_also_holds_in_the_loop():
    state, _ = block_state({"frames": 6, "overlap": 2, "positions": "rebased"})
    with pytest.raises(ValueError, match="multiple of 5"):
        DwH3WindowedLoopDenoiser()(
            SimpleNamespace(transformer=StubTransformer()), state, 0, 0
        )


def test_a_step_cache_is_refused():
    assert any("cache" in p for p in window_problems(refine(), has_cache=True))


def test_off_in_this_process_names_the_flag():
    pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
    insert_audio_hold(pipeline)
    assert ENABLE_ENV_VAR in windows_refusal(pipeline, refine())
    insert_refine_windows(pipeline)
    assert windows_refusal(pipeline, refine()) is None
    assert windows_refusal(pipeline, refine(**{REFINE_WINDOWS_INPUT: None})) is None


# 6. End to end: the real H3 graph on tiny weights, through dw's blocks


@pytest.fixture(scope="module")
def tiny_h3():
    from tests.test_chain_h3_native import _tiny_h3

    pipeline = _tiny_h3()
    insert_audio_hold(pipeline)
    assert insert_refine_windows(pipeline)
    return pipeline


def tiny_refine(
    pipeline,
    windows,
    source_video=None,
    num_frames=124,
    output="latents",
    refine_strength=0.3,
):
    from diffusers.modular_pipelines.minimax_h3.references import (
        MiniMaxH3AudioReference,
    )

    # 124 frames: 37 latent frames of 2x2 latents (one row a frame at 32x32)
    latent_frames = h3_rules.guide_latent_frames(num_frames)
    latents = torch.randn(
        1, 24, latent_frames, 2, 2, generator=torch.Generator().manual_seed(3)
    )
    track = torch.randn(
        2, num_frames * 16000 // 24, generator=torch.Generator().manual_seed(4)
    )
    arguments = dict(
        prompt="a cat sat",
        height=32,
        width=32,
        num_frames=num_frames,
        num_inference_steps=4,
        latents=latents,
        hold_audio=MiniMaxH3AudioReference(audio=track * 0.1, sample_rate=16000),
        refine_strength=refine_strength,
        generator=torch.Generator("cpu").manual_seed(1234),
        output=output,
    )
    if source_video is not None:
        del arguments["latents"]
        arguments["source_video"] = source_video
    if windows is not None:
        arguments[REFINE_WINDOWS_INPUT] = windows
    return pipeline(**arguments)


def test_end_to_end_one_window_is_the_plain_refine(tiny_h3):
    plain = tiny_refine(tiny_h3, None)
    one = tiny_refine(tiny_h3, {"frames": 37, "overlap": 4})
    assert torch.equal(plain, one)


def test_end_to_end_windows_run_and_differ(tiny_h3):
    plain = tiny_refine(tiny_h3, None)
    windowed = tiny_refine(tiny_h3, {"frames": 12, "overlap": 4, "blend": "cosine"})
    assert windowed.shape == plain.shape
    assert torch.isfinite(windowed).all()
    assert not torch.equal(plain, windowed)


# 6b. Past 15 s: the layout's duration ceiling, lifted for a windowed refine only

# 17 * 22 + 5 frames: 15.8 s at 24 fps, over diffusers' 15 s ceiling
LONG_TAKE = 379


def test_a_windowed_refine_lays_out_a_take_past_fifteen_seconds(tiny_h3):
    """The whole long take - 112 latents and its 15.8 s held track - through
    the real graph, windowed. The held audio is the whole track: its own
    encode has no ceiling of its own."""
    result = tiny_refine(
        tiny_h3,
        {"frames": 40, "overlap": 10},
        num_frames=LONG_TAKE,
        output=["latents", "audio", "sampling_rate"],
    )
    latents = result["latents"]
    assert latents.shape[2] == h3_rules.guide_latent_frames(LONG_TAKE) == 112
    assert torch.isfinite(latents).all()
    # The decoded soundtrack covers the whole take, to within one 40 Hz latent
    audio, rate = result["audio"], result["sampling_rate"]
    assert abs(audio.shape[-1] - LONG_TAKE / 24 * rate) < rate / 40


def test_the_ceiling_lifts_inside_the_guides_layout_too():
    """The loader wraps the layout in the guides' one (h3_guides.py) before the
    windows go in: the long refine runs through that graph as well."""
    from dw.pipeline_processors.h3_guides import insert_guides
    from tests.test_chain_h3_native import _tiny_h3

    pipeline = _tiny_h3()
    insert_audio_hold(pipeline)
    assert insert_guides(pipeline)
    assert insert_refine_windows(pipeline)
    latents = tiny_refine(pipeline, {"frames": 40, "overlap": 10}, num_frames=LONG_TAKE)
    assert latents.shape[2] == 112
    with pytest.raises(ValueError, match="between 5.0 and 15.0 seconds"):
        tiny_refine(pipeline, None, num_frames=LONG_TAKE)


def test_a_plain_refine_past_fifteen_seconds_keeps_the_ceiling(tiny_h3):
    with pytest.raises(ValueError, match="between 5.0 and 15.0 seconds"):
        tiny_refine(tiny_h3, None, num_frames=LONG_TAKE)


def test_windows_without_a_refine_keep_the_ceiling(tiny_h3):
    """A generation over 15 s is another model regime: the windows argument
    alone does not lift the ceiling, only a windowed refine does."""
    with pytest.raises(ValueError, match="between 5.0 and 15.0 seconds"):
        tiny_refine(
            tiny_h3,
            {"frames": 40, "overlap": 10},
            num_frames=LONG_TAKE,
            refine_strength=None,
        )


def test_the_layout_is_replaced_where_the_windows_go_and_only_there():
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3Ref2VAPrepareLayoutStep,
    )

    from dw.pipeline_processors.h3_guides import insert_guides

    def sequences(pipeline):
        return h3_windows.core_denoise_sequences(pipeline)

    for guided in (False, True):
        for pipeline in pipelines():
            if guided:
                insert_guides(pipeline)
            before = [list(s.sub_blocks) for _, s in sequences(pipeline)]
            stock = h3_windows.layout_steps(pipeline)
            ref2va = [
                b
                for prefix, s in sequences(pipeline)
                for b in [s.sub_blocks.get(prefix + h3_windows.LAYOUT)]
                if isinstance(b, MiniMaxH3Ref2VAPrepareLayoutStep)
            ]
            insert_refine_windows(pipeline)
            insert_refine_windows(pipeline)  # idempotent
            layouts = h3_windows.layout_steps(pipeline)
            assert len(layouts) == len(stock)
            for block in layouts:
                assert isinstance(block, h3_windows.DwH3LongRefineLayoutStep)
            # The block order is kept, and Ref2VA's own layout is untouched
            assert before == [list(s.sub_blocks) for _, s in sequences(pipeline)]
            for block in ref2va:
                assert type(block) is MiniMaxH3Ref2VAPrepareLayoutStep
        assert any(h3_windows.layout_steps(p) for p in pipelines())


def test_a_source_video_past_fifteen_seconds_is_sent_to_the_chunked_encode():
    problems = window_problems(
        refine(latents=None, source_video=object(), num_frames=LONG_TAKE)
    )
    assert any("encode_h3_latents" in p for p in problems), problems
    assert (
        window_problems(refine(latents=None, source_video=object(), num_frames=124))
        == []
    )


def test_a_source_video_path_is_not_counted_as_frames():
    long_path = "/data/" + "a" * 411 + ".mp4"
    assert window_problems(refine(latents=None, source_video=long_path)) == []


def test_the_run_time_check_refuses_with_the_flag_off():
    from dw.pipeline_processors.pipeline import Pipeline

    pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
    insert_audio_hold(pipeline)
    step = SimpleNamespace(name="refine", pipeline=pipeline, configuration={})
    with pytest.raises(ValueError, match=f"Step 'refine': .*{ENABLE_ENV_VAR}=1"):
        Pipeline._check_refine(step, refine())
    insert_refine_windows(pipeline)
    assert Pipeline._check_refine(step, refine()) is None
    step.configuration = {"cache": {"type": "first_block"}}
    with pytest.raises(ValueError, match="cache"):
        Pipeline._check_refine(step, refine())


@pytest.fixture(scope="module")
def tiny_h3_source():
    """The tiny graph as the loader finishes it for a `source_video`: the hold,
    the regional edit's source and blend blocks (h3_mask.py), then the
    windows - the order `_assemble_loaded` puts them in."""
    from tests.test_chain_h3_native import _tiny_h3

    pipeline = _tiny_h3()
    insert_audio_hold(pipeline)
    assert insert_mask_blocks(pipeline)
    assert insert_refine_windows(pipeline)
    return pipeline


def source_frames(seed):
    import numpy

    return numpy.random.default_rng(seed).integers(0, 255, (124, 32, 32, 3), "uint8")


def strict_refine(pipeline, windows, source_video, caplog):
    """`tiny_refine` with an ignored input made an error: diffusers drops an
    argument no block declares with a warning (both `warnings.warn` and its
    logger), and a dropped `source_video` would leave nothing to compare."""
    caplog.clear()
    with caplog.at_level("WARNING"), warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*Unexpected input")
        result = tiny_refine(pipeline, windows, source_video=source_video)
    unexpected = [r for r in caplog.records if "Unexpected input" in r.getMessage()]
    assert not unexpected, unexpected[0].getMessage()
    return result


def test_the_source_video_reaches_the_graph(tiny_h3_source, caplog):
    """The source is encoded, not dropped: another source refines to another
    take."""
    plain = strict_refine(tiny_h3_source, None, source_frames(5), caplog)
    other = strict_refine(tiny_h3_source, None, source_frames(6), caplog)
    assert not torch.equal(plain, other)


def test_a_graph_without_the_source_block_drops_the_source_video(tiny_h3, caplog):
    """Why the fixture above exists: on the hold-only graph diffusers ignores
    `source_video`, and the check in `strict_refine` says so."""
    with pytest.raises((AssertionError, UserWarning), match="Unexpected input"):
        strict_refine(tiny_h3, None, source_frames(5), caplog)


def test_end_to_end_one_window_is_the_plain_refine_of_a_source_video(
    tiny_h3_source, caplog
):
    """A finished take as pixels: the regional edit's encode with no mask, and
    its blend block in the loop after `update` (a no-op without a mask)."""
    frames = source_frames(5)
    plain = strict_refine(tiny_h3_source, None, frames, caplog)
    one = strict_refine(tiny_h3_source, {"frames": 40}, frames, caplog)
    two = strict_refine(tiny_h3_source, {"frames": 20, "overlap": 6}, frames, caplog)
    assert torch.equal(plain, one)
    assert torch.isfinite(two).all()
    assert not torch.equal(plain, two)


# 7. The flag: one owner, and the loader gate


def test_the_flag_has_one_owner():
    assert h3_windows.ENABLE_ENV_VAR is h3_rules.REFINE_WINDOWS_ENV_VAR
    from dw.pipeline_processors import components

    source = Path(components.__file__).read_text()
    assert h3_rules.REFINE_WINDOWS_ENV_VAR not in source


@pytest.mark.parametrize("value, on", [(None, False), ("1", True), ("0", False)])
def test_the_flag_reads_the_environment(monkeypatch, value, on):
    if value is None:
        monkeypatch.delenv(h3_rules.REFINE_WINDOWS_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(h3_rules.REFINE_WINDOWS_ENV_VAR, value)
    assert h3_rules.refine_windows_enabled() is on


@pytest.mark.parametrize("value, windowed", [(None, False), ("1", True)])
def test_the_loader_inserts_the_windows_only_behind_the_flag(
    monkeypatch, value, windowed
):
    from dw.pipeline_processors.components import _assemble_loaded

    if value is None:
        monkeypatch.delenv(h3_rules.REFINE_WINDOWS_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(h3_rules.REFINE_WINDOWS_ENV_VAR, value)
    pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
    _assemble_loaded("pipeline", type(pipeline), {}, pipeline, {})
    denoisers = [loop.sub_blocks[DENOISER] for loop in loops(pipeline)]
    assert denoisers
    for denoiser in denoisers:
        assert isinstance(denoiser, DwH3WindowedLoopDenoiser) is windowed
        if not windowed:
            assert type(denoiser) is MiniMaxH3LoopDenoiser
    # The loader's graph wraps the layout in the guides' one: the step that
    # runs is the one it holds
    layouts = h3_windows.layout_steps(pipeline)
    assert layouts
    for block in layouts:
        assert isinstance(block, h3_windows.DwH3LongRefineLayoutStep) is windowed
        if not windowed:
            assert type(block) is MiniMaxH3PrepareLayoutStep
    assert windows_refine(pipeline) is windowed
