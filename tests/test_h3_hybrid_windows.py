"""The experimental windowed generation (#1188 S1): `experimental_hybrid_windows`
generates a held-audio t2va take in blended time windows with its state in host
RAM. The flag, the refusals, the host loop against the stock one, and the
estimates. dw/pipeline_processors/h3_windows.py."""

from types import SimpleNamespace

import pytest
import torch

minimax = pytest.importorskip("diffusers.modular_pipelines.minimax_h3")
from diffusers.modular_pipelines.minimax_h3.denoise import (  # noqa: E402
    MiniMaxH3LoopDenoiser,
    MiniMaxH3LoopSchedulerStep,
)
from diffusers.schedulers import MiniMaxH3Scheduler  # noqa: E402

from dw.pipeline_processors import h3_rules, h3_windows  # noqa: E402
from dw.pipeline_processors.h3_hold import insert_audio_hold  # noqa: E402
from dw.pipeline_processors.h3_rules import (  # noqa: E402
    HYBRID_DEFAULT_WINDOW_LATENTS,
    HYBRID_WINDOWS_ENV_VAR,
    HYBRID_WINDOWS_INPUT,
    hybrid_window_frames,
    lifts_duration_ceiling,
)
from dw.pipeline_processors.h3_windows import (  # noqa: E402
    DENOISER,
    HYBRID_DEVICE,
    REFINE_WINDOWS_INPUT,
    UPDATE,
    WINDOW_DEFAULTS,
    DwH3HostLoopSchedulerStep,
    DwH3WindowedLoopDenoiser,
    fuse_weights,
    hybrid_problems,
    hybrid_refusal,
    insert_refine_windows,
    window_plan,
)
from tests.test_h3_windows import (  # noqa: E402
    LATENT_FRAMES,
    StubTransformer,
    block_state,
    tiny_transformer,
)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(HYBRID_WINDOWS_ENV_VAR, "1")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv(HYBRID_WINDOWS_ENV_VAR, raising=False)


def hybrid_state(settings, steps=3):
    state, scheduler = block_state(None, steps)
    setattr(state, HYBRID_WINDOWS_INPUT, settings)
    state.timesteps = scheduler.timesteps
    return state, scheduler


def run_loop(denoiser, update, settings, steps=3, hybrid=True):
    """Every step of the loop from the same start on the tiny transformer:
    the final `(latents, audio_latents)` and the state."""
    if hybrid:
        state, scheduler = hybrid_state(settings, steps)
    else:
        state, scheduler = block_state(None, steps)
    components = SimpleNamespace(
        transformer=tiny_transformer(),
        scheduler=scheduler,
        audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
    )
    components.audio_scheduler.set_timesteps(steps)
    for i, t in enumerate(scheduler.timesteps):
        denoiser(components, state, i, t)
        update(components, state, i, t)
    return state.latents, state.audio_latents, state


# 1. The loop


@pytest.mark.parametrize("frames", [LATENT_FRAMES, LATENT_FRAMES + 9])
def test_one_window_over_the_take_is_bit_identical_to_the_stock_loop(flag_on, frames):
    stock = run_loop(
        MiniMaxH3LoopDenoiser(), MiniMaxH3LoopSchedulerStep(), None, hybrid=False
    )
    hybrid = run_loop(
        DwH3WindowedLoopDenoiser(),
        DwH3HostLoopSchedulerStep(),
        {"frames": frames, "overlap": 2, "switch_step": 0},
    )
    assert torch.equal(stock[0], hybrid[0])
    assert torch.equal(stock[1], hybrid[1])


def test_two_windows_run_change_the_video_and_leave_the_audio_held(flag_on):
    stock = run_loop(
        MiniMaxH3LoopDenoiser(), MiniMaxH3LoopSchedulerStep(), None, hybrid=False
    )
    hybrid = run_loop(
        DwH3WindowedLoopDenoiser(),
        DwH3HostLoopSchedulerStep(),
        {"frames": 7, "overlap": 2, "switch_step": 0},
    )
    assert not torch.equal(stock[0], hybrid[0])
    assert torch.isfinite(hybrid[0]).all()
    assert torch.equal(stock[1], hybrid[1])


def test_the_fused_step_is_the_weighted_sum_of_each_windows_own_step(flag_on):
    """The flow-matching step is affine in the velocity, so stepping the
    fused velocity is stepping each window alone and blending the results."""
    settings = {"frames": 7, "overlap": 2, "switch_step": 0, "positions": "true"}
    state, scheduler = hybrid_state(settings, steps=3)
    stub = StubTransformer()
    start = state.latents.clone()
    DwH3WindowedLoopDenoiser()(
        SimpleNamespace(transformer=stub), state, 0, scheduler.timesteps[0]
    )
    fused_next = scheduler.step(
        state.noise_pred[0], scheduler.timesteps[0], start, return_dict=False
    )[0]
    scheduler._step_index = None

    plan = window_plan(LATENT_FRAMES, 7, 2)
    weights = fuse_weights(plan, LATENT_FRAMES, "pyramid")
    assert len(plan) == len(stub.calls) > 1
    per_frame = start.shape[0] // LATENT_FRAMES
    blended = torch.zeros_like(start)
    for (lo, hi), w, call in zip(plan, weights, stub.calls):
        rows = slice(lo * per_frame, hi * per_frame)
        velocity = torch.zeros(1, *start.shape)
        video = torch.arange(start.shape[0])[rows]
        velocity[0, video] = stub_velocity(call)[0]
        alone = scheduler.step(
            velocity[0], scheduler.timesteps[0], start, return_dict=False
        )[0]
        scheduler._step_index = None  # each window steps from the same step
        blended[rows] += alone[rows] * w.repeat_interleave(per_frame)[:, None]
    torch.testing.assert_close(fused_next, blended, rtol=1e-5, atol=1e-5)


def stub_velocity(call):
    hidden = call["hidden_states"]
    times = call["position_ids"][call["video_indices"], 0].float()[None, :, None]
    return hidden + hidden.mean(dim=1, keepdim=True) + times


def test_the_device_sees_one_window_and_the_take_lives_on_the_host(flag_on):
    settings = {"frames": 5, "overlap": 2, "switch_step": 0, "positions": "true"}
    state, scheduler = hybrid_state(settings, steps=2)
    rows = state.latents.shape[0]
    stub = StubTransformer()
    components = SimpleNamespace(
        transformer=stub,
        scheduler=scheduler,
        audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
    )
    components.audio_scheduler.set_timesteps(2)
    denoiser, update = DwH3WindowedLoopDenoiser(), DwH3HostLoopSchedulerStep()
    timesteps = list(scheduler.timesteps)
    for i, t in enumerate(timesteps):
        denoiser(components, state, i, t)
        assert getattr(state, HYBRID_DEVICE) is not None
        assert state.latents.device.type == "cpu"
        assert state.noise_pred.device.type == "cpu"
        update(components, state, i, t)
        if i < len(timesteps) - 1:
            assert getattr(state, HYBRID_DEVICE) is not None
    # after the last step the take is back where it came from
    assert getattr(state, HYBRID_DEVICE) is None
    window_rows = rows * 5 // LATENT_FRAMES
    assert stub.calls
    assert all(c["hidden_states"].shape[1] <= window_rows for c in stub.calls)
    assert all(c["hidden_states"].shape[1] < rows for c in stub.calls)


def test_a_nonzero_switch_step_warns_and_runs_blended(flag_on, monkeypatch):
    seen = []
    monkeypatch.setattr(
        h3_windows, "emit_warning", lambda message, **data: seen.append(data)
    )
    blended = run_loop(
        DwH3WindowedLoopDenoiser(),
        DwH3HostLoopSchedulerStep(),
        {"frames": 7, "overlap": 2, "switch_step": 0},
    )
    assert seen == []
    asked = run_loop(
        DwH3WindowedLoopDenoiser(),
        DwH3HostLoopSchedulerStep(),
        {"frames": 7, "overlap": 2, "switch_step": 2},
    )
    assert seen == [{"switch_step": 2}]
    assert torch.equal(blended[0], asked[0])


def test_the_loop_refuses_when_the_flag_is_off(flag_off):
    state, scheduler = hybrid_state({"switch_step": 0})
    components = SimpleNamespace(transformer=tiny_transformer(), scheduler=scheduler)
    with pytest.raises(ValueError, match=HYBRID_WINDOWS_ENV_VAR):
        DwH3WindowedLoopDenoiser()(components, state, 0, scheduler.timesteps[0])


def test_the_host_update_is_the_stock_one_without_a_windowed_generation():
    stock = run_loop(
        MiniMaxH3LoopDenoiser(), MiniMaxH3LoopSchedulerStep(), None, hybrid=False
    )
    ours = run_loop(
        MiniMaxH3LoopDenoiser(), DwH3HostLoopSchedulerStep(), None, hybrid=False
    )
    assert torch.equal(stock[0], ours[0])
    assert torch.equal(stock[1], ours[1])


# 2. The graph


def test_the_windows_insert_the_host_update_idempotently():
    pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
    insert_audio_hold(pipeline)
    assert insert_refine_windows(pipeline)
    assert insert_refine_windows(pipeline)
    loops = list(h3_windows._loops(pipeline))
    assert loops
    for loop in loops:
        assert type(loop.sub_blocks[UPDATE]) is DwH3HostLoopSchedulerStep
        assert isinstance(loop.sub_blocks[DENOISER], DwH3WindowedLoopDenoiser)
    assert HYBRID_WINDOWS_INPUT in {p.name for p in pipeline._blocks.inputs}


# 3. The refusals


def generate(**overrides):
    arguments = {
        "hold_audio": object(),
        HYBRID_WINDOWS_INPUT: {"frames": 42, "overlap": 7, "switch_step": 0},
    }
    arguments.update(overrides)
    return {k: v for k, v in arguments.items() if v is not None}


def test_a_valid_call_has_no_problems(flag_on):
    assert hybrid_problems(generate()) == []
    assert hybrid_problems(generate(**{HYBRID_WINDOWS_INPUT: {}})) == []
    assert hybrid_problems(generate(**{HYBRID_WINDOWS_INPUT: None})) == []


def test_the_flag_off_refusal_names_the_flag(flag_off):
    problems = hybrid_problems(generate(**{HYBRID_WINDOWS_INPUT: {}}))
    assert len(problems) == 1
    assert HYBRID_WINDOWS_ENV_VAR in problems[0]
    assert "experimental" in problems[0]


@pytest.mark.parametrize("value", ["0", "true", ""])
def test_only_one_turns_the_flag_on(monkeypatch, value):
    monkeypatch.setenv(HYBRID_WINDOWS_ENV_VAR, value)
    assert any(HYBRID_WINDOWS_ENV_VAR in p for p in hybrid_problems(generate()))


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"hold_audio": None}, "hold_audio"),
        ({"references": [object()]}, "references"),
        ({"refine_strength": 0.3}, "refine_strength"),
        ({REFINE_WINDOWS_INPUT: {}}, REFINE_WINDOWS_INPUT),
        ({"latents": torch.zeros(1)}, "latents"),
        ({"source_video": "take.mp4"}, "source_video"),
        ({HYBRID_WINDOWS_INPUT: [42, 8]}, "every key optional"),
        ({HYBRID_WINDOWS_INPUT: {"switch_step": -1}}, "switch_step"),
        ({HYBRID_WINDOWS_INPUT: {"switch_step": True}}, "switch_step"),
        ({HYBRID_WINDOWS_INPUT: {"handoff": "sideways"}}, "handoff"),
        ({HYBRID_WINDOWS_INPUT: {"frames": 8, "overlap": 8}}, "overlap"),
        ({HYBRID_WINDOWS_INPUT: {"window": 8}}, "window"),
    ],
)
def test_a_call_the_windows_cannot_run_is_refused(flag_on, overrides, match):
    problems = hybrid_problems(generate(**overrides))
    assert any(match in p for p in problems), problems


def test_a_step_cache_is_refused(flag_on):
    assert any("cache" in p for p in hybrid_problems(generate(), has_cache=True))


def test_the_refusal_needs_the_windowed_graph(flag_on):
    pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
    insert_audio_hold(pipeline)
    assert "no denoise loop that windows" in hybrid_refusal(pipeline, generate())
    insert_refine_windows(pipeline)
    assert hybrid_refusal(pipeline, generate()) is None
    assert hybrid_refusal(pipeline, generate(**{HYBRID_WINDOWS_INPUT: None})) is None


def test_the_run_time_check_refuses_with_the_flag_off(flag_off):
    from dw.pipeline_processors.pipeline import Pipeline

    pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
    insert_audio_hold(pipeline)
    insert_refine_windows(pipeline)
    step = SimpleNamespace(name="gen", pipeline=pipeline, configuration={})
    with pytest.raises(ValueError, match=f"Step 'gen': .*{HYBRID_WINDOWS_ENV_VAR}"):
        Pipeline._check_refine(step, generate())


def test_the_run_time_check_refuses_a_cache(flag_on):
    from dw.pipeline_processors.pipeline import Pipeline

    pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
    insert_audio_hold(pipeline)
    insert_refine_windows(pipeline)
    step = SimpleNamespace(name="gen", pipeline=pipeline, configuration={})
    assert Pipeline._check_refine(step, generate()) is None
    step.configuration = {"cache": {"type": "first_block"}}
    with pytest.raises(ValueError, match="cache"):
        Pipeline._check_refine(step, generate())


# 4. The ceiling and the estimates


def test_the_key_lifts_the_duration_ceiling():
    assert lifts_duration_ceiling({HYBRID_WINDOWS_INPUT: {}})
    assert not lifts_duration_ceiling({HYBRID_WINDOWS_INPUT: None})


def test_the_default_window_is_the_refines_default():
    assert HYBRID_DEFAULT_WINDOW_LATENTS == WINDOW_DEFAULTS["frames"]


@pytest.mark.parametrize(
    "num_frames, settings, expected",
    [
        (690, {}, 17 * 40 // 5 + 5),  # 42 latents -> 141 frames
        (690, {"frames": 12}, 17 * 10 // 5 + 5),
        (100, {}, 100),  # a take shorter than a window is the take
        ("variable:n", {}, None),
        (690, {"frames": "x"}, None),
    ],
)
def test_one_window_in_frames(num_frames, settings, expected):
    assert hybrid_window_frames(num_frames, settings) == expected


def test_window_frames_land_on_the_render_grid():
    for latents in range(2, 60):
        frames = hybrid_window_frames(10_000, {"frames": latents})
        assert isinstance(frames, int)
        assert (frames - 5) % 17 == 0
        held = h3_rules.guide_latent_frames(frames)
        assert latents <= held < latents + 5


def test_the_vram_estimate_prices_one_window():
    from dw import vram_estimate

    values = {"num_frames": 690}
    assert vram_estimate.hybrid_frames({}, values) is None
    assert vram_estimate.hybrid_frames(
        {HYBRID_WINDOWS_INPUT: {"frames": 12}}, values
    ) == hybrid_window_frames(690, {"frames": 12})


def test_the_time_estimate_is_left_unpriced_with_a_reason():
    from dw.cost_scaling import UNKNOWN
    from dw.plan import _hybrid_unpriced

    estimate = {"minutes": 4.0, "cached_minutes": 1.0, "basis": "history"}
    plain = {"steps": [{"name": "s", "pipeline": {"arguments": {"num_frames": 9}}}]}
    assert _hybrid_unpriced(plain, estimate) is estimate
    hybrid = {
        "steps": [{"name": "s", "pipeline": {"arguments": {HYBRID_WINDOWS_INPUT: {}}}}]
    }
    unpriced = _hybrid_unpriced(hybrid, estimate)
    assert unpriced["minutes"] is None and unpriced["cached_minutes"] is None
    assert unpriced["basis"] == UNKNOWN
    assert HYBRID_WINDOWS_INPUT in unpriced["unpriced_reason"]
