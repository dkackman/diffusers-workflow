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
    window_inputs,
    window_plan,
    window_rows,
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


# 1b. The ordered phase (#1188 S2)

STEPS = 5  # num_inference_steps 5: four denoise steps
ORDERED = {"frames": 7, "overlap": 2, "positions": "true"}  # (0, 7), (5, 12)


class CopyingStub(StubTransformer):
    """The stub, recording a copy of what it is shown: the ordered phase
    steps a window's rows in place after its forward."""

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
        out = super().forward(
            hidden_states.clone(),
            audio_hidden_states.clone(),
            encoder_hidden_states,
            timestep,
            timestep_indices,
            token_tags,
            position_ids,
            video_indices,
            audio_indices,
            text_indices,
        )
        self.calls[-1]["timestep"] = timestep.clone()
        return out


def run_stub(settings, steps=STEPS):
    state, scheduler = hybrid_state(settings, steps)
    start = state.latents.clone()
    stub = CopyingStub()
    components = SimpleNamespace(
        transformer=stub,
        scheduler=scheduler,
        audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
    )
    components.audio_scheduler.set_timesteps(steps)
    sigmas = scheduler.sigmas.clone()
    denoiser, update = DwH3WindowedLoopDenoiser(), DwH3HostLoopSchedulerStep()
    for i, t in enumerate(scheduler.timesteps):
        denoiser(components, state, i, t)
        update(components, state, i, t)
    return state, stub, start, sigmas


def window_video(state, span):
    fields = state.denoiser_input_fields
    return window_rows(fields, state.latents.shape[0], 0, LATENT_FRAMES, span)[0]


def sequential_windows(settings, steps=STEPS):
    """Plain sequential windows, written out: each window denoised alone
    over every step from the start noise, with the rows an earlier window
    finished held at its finished values and the clean timestep, and the
    first window's value kept for every row."""
    state, _ = hybrid_state(settings, steps)
    fields = state.denoiser_input_fields
    transformer = tiny_transformer()
    result = state.latents.clone()
    done = torch.zeros(result.shape[0], dtype=torch.bool)
    for span in window_plan(LATENT_FRAMES, settings["frames"], settings["overlap"]):
        rows = window_rows(fields, result.shape[0], 0, LATENT_FRAMES, span)
        video = rows[0]
        frozen = done[video]
        x = result[video].clone()
        scheduler = MiniMaxH3Scheduler(shift=12.0)
        scheduler.set_timesteps(steps)
        for t, entry in zip(scheduler.timesteps, state.row_timestep_plan):
            layout, (unique, indices) = window_inputs(fields, entry, rows)
            indices = indices.clone()
            indices[layout["video_indices"][frozen]] = int(
                (unique == 1.0).nonzero()[0, 0]
            )
            velocity = transformer(
                hidden_states=x[None],
                audio_hidden_states=state.audio_latents[rows[1]][None],
                encoder_hidden_states=state.prompt_embeds,
                timestep=unique,
                timestep_indices=indices,
                return_dict=False,
                **layout,
            )[0][0]
            stepped = scheduler.step(velocity.float(), t, x, return_dict=False)[0]
            x = torch.where(frozen[:, None], x, stepped)
        result[video] = x
        done[video] = True
    return result


def test_switch_step_at_the_last_step_is_plain_sequential_windows(flag_on):
    settings = {**ORDERED, "switch_step": STEPS - 1, "handoff": "finished"}
    ours = run_loop(
        DwH3WindowedLoopDenoiser(), DwH3HostLoopSchedulerStep(), settings, STEPS
    )
    torch.testing.assert_close(ours[0], sequential_windows(settings))
    stock_audio = run_loop(
        MiniMaxH3LoopDenoiser(), MiniMaxH3LoopSchedulerStep(), None, STEPS, False
    )[1]
    assert torch.equal(ours[1], stock_audio)


def test_switch_step_zero_is_the_blended_windows_of_s1(flag_on):
    """k = 0 has no ordered phase: every step is the fused one, and the
    default (3) is a different take."""
    blended = run_loop(
        DwH3WindowedLoopDenoiser(),
        DwH3HostLoopSchedulerStep(),
        {**ORDERED, "switch_step": 0},
        STEPS,
    )
    ordered = run_loop(
        DwH3WindowedLoopDenoiser(), DwH3HostLoopSchedulerStep(), ORDERED, STEPS
    )
    assert not torch.equal(blended[0], ordered[0])
    assert torch.isfinite(ordered[0]).all()


def split_calls(stub, state, k, steps=STEPS - 1):
    """The stub's calls by phase for the two-window plan: window 1's
    `steps`, window 2's k ordered ones, then the blended finish's."""
    calls = stub.calls
    assert len(calls) == steps + k + (steps - k)
    return calls[:steps], calls[steps : steps + k], calls[steps + k :]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_the_overlap_is_pinned_to_the_window_befores_finished_frames(flag_on, k):
    state, stub, _, _ = run_stub({**ORDERED, "switch_step": k, "handoff": "finished"})
    first, second, _ = split_calls(stub, state, k)
    video1, video2 = window_video(state, (0, 7)), window_video(state, (5, 12))
    overlap = torch.isin(video2, video1)
    assert overlap.any() and not overlap.all()
    # window 1 is kept finished, so its finished rows are the take's
    finished = state.latents[video2[overlap]]
    for call in second:
        assert torch.equal(call["hidden_states"][0, overlap], finished)
        assert_clean(call, overlap)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_the_overlap_is_pinned_to_the_window_befores_x0_at_step_k(flag_on, k):
    state, stub, _, sigmas = run_stub({**ORDERED, "switch_step": k, "handoff": "x0"})
    first, second, _ = split_calls(stub, state, k)
    video1, video2 = window_video(state, (0, 7)), window_video(state, (5, 12))
    call = first[k - 1]  # window 1's forward at step k
    x0 = call["hidden_states"][0] + sigmas[k - 1] * stub_velocity(call)[0]
    in_first = torch.isin(video1, video2)
    overlap = torch.isin(video2, video1)
    for call in second:
        torch.testing.assert_close(call["hidden_states"][0, overlap], x0[in_first])
        assert_clean(call, overlap)
    # the take keeps window 1's finished rows there, not the x0
    assert not torch.allclose(state.latents[video2[overlap]], x0[in_first])


def assert_clean(call, rows):
    """`rows` of the window's video are read at t = 1, the rest not."""
    unique = call["timestep"]
    times = unique[call["timestep_indices"][call["video_indices"]]]
    assert (times[rows] == 1.0).all()
    assert (times[~rows] < 1.0).all()


@pytest.mark.parametrize("handoff", ["finished", "x0"])
def test_every_row_keeps_the_first_windows_state(flag_on, handoff):
    """Window 1's rows are its finished ones - the last step of its own
    loop - through the ordered phase and the blended finish alike."""
    k = 2
    state, stub, _, sigmas = run_stub({**ORDERED, "switch_step": k, "handoff": handoff})
    first, _, blended = split_calls(stub, state, k)
    video1, video2 = window_video(state, (0, 7)), window_video(state, (5, 12))
    last = first[-1]
    finished = last["hidden_states"][0] + sigmas[STEPS - 2] * stub_velocity(last)[0]
    torch.testing.assert_close(state.latents[video1], finished)
    # the blended finish runs window 2 only, its overlap held at t = 1
    overlap = torch.isin(video2, video1)
    for call in blended:
        assert call["hidden_states"].shape[1] == video2.numel()
        assert_clean(call, overlap)


def test_a_later_window_pins_to_the_latest_window_that_covers_a_row(flag_on):
    """Four windows of 5 over 12 frames: (0, 5), (3, 8), (6, 11), (7, 12).
    The last window's rows 7-10 are pinned to window 3's finished frames."""
    settings = {"frames": 5, "overlap": 2, "positions": "true", "switch_step": 4}
    state, stub, _, _ = run_stub(settings)
    plan = window_plan(LATENT_FRAMES, 5, 2)
    assert plan == [(0, 5), (3, 8), (6, 11), (7, 12)]
    videos = [window_video(state, span) for span in plan]
    calls = stub.calls
    assert len(calls) == 4 * (STEPS - 1)
    for w in range(1, 4):
        covered = torch.isin(videos[w], torch.cat(videos[:w]))
        for call in calls[w * (STEPS - 1) : (w + 1) * (STEPS - 1)]:
            assert_clean(call, covered)
    # window 4 is pinned to window 3's frames: the ones it stepped finished,
    # the one it held itself (frame 7, from window 2) as it held it
    third_end = calls[3 * (STEPS - 1) - 1]
    held3 = torch.isin(videos[2], torch.cat(videos[:2]))
    third = torch.where(
        held3[:, None],
        third_end["hidden_states"][0],
        third_end["hidden_states"][0] + _last_sigma() * stub_velocity(third_end)[0],
    )
    shared3 = torch.isin(videos[2], videos[3])
    shared4 = torch.isin(videos[3], videos[2])
    torch.testing.assert_close(
        calls[3 * (STEPS - 1)]["hidden_states"][0, shared4], third[shared3]
    )


def _last_sigma(steps=STEPS):
    scheduler = MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(steps)
    return scheduler.sigmas[-2]


def test_a_switch_step_past_the_runs_steps_is_refused_in_the_loop(flag_on):
    state, scheduler = hybrid_state({**ORDERED, "switch_step": STEPS}, STEPS)
    components = SimpleNamespace(transformer=tiny_transformer(), scheduler=scheduler)
    with pytest.raises(ValueError, match="switch_step"):
        DwH3WindowedLoopDenoiser()(components, state, 0, scheduler.timesteps[0])


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


@pytest.mark.parametrize(
    "value",
    [{"frames": 42, "windw": 8}, "yes", [42, 12]],
)
@pytest.mark.parametrize("flag", ["flag_on", "flag_off"])
def test_a_key_list_refusal_does_not_name_the_unpromoted_anchor(request, flag, value):
    """#1572 bounce (M-F495): `anchor` is accepted, but until promotion no
    refusal of a caller who did not write it lists it as a field of the key."""
    request.getfixturevalue(flag)
    problems = hybrid_problems(generate(**{HYBRID_WINDOWS_INPUT: value}))
    assert problems
    assert any("handoff" in p for p in problems), problems
    assert not any("anchor" in p for p in problems), problems


def test_the_anchor_is_still_accepted_and_its_bad_value_named(flag_on):
    assert (
        hybrid_problems(generate(**{HYBRID_WINDOWS_INPUT: {"anchor": "keystone"}}))
        == []
    )
    problems = hybrid_problems(generate(**{HYBRID_WINDOWS_INPUT: {"anchor": "bogus"}}))
    assert len(problems) == 1 and "anchor is one of none, keystone" in problems[0]


def ordered(switch, steps=31, **settings):
    return generate(
        num_inference_steps=steps,
        **{HYBRID_WINDOWS_INPUT: {"switch_step": switch, **settings}},
    )


def test_a_switch_step_past_the_runs_denoise_steps_is_refused(flag_on):
    """num_inference_steps 31 runs 30 denoise steps: k = 30 is every step
    ordered (plain sequential windows), 31 has no step to be."""
    assert hybrid_problems(ordered(30)) == []
    assert hybrid_problems(ordered(3)) == []
    problems = hybrid_problems(ordered(31))
    assert len(problems) == 1 and "switch_step" in problems[0], problems
    assert "num_inference_steps 31 runs 30" in problems[0]


def test_switch_step_range_is_left_to_the_run_without_a_literal_step_count(
    flag_on,
):
    assert (
        hybrid_problems(generate(**{HYBRID_WINDOWS_INPUT: {"switch_step": 99}})) == []
    )
    assert hybrid_problems(ordered(99, steps="variable:steps")) == []


@pytest.mark.parametrize(
    "settings, match",
    [
        ({"switch_step": 31}, "switch_step"),
        ({"switch_step": -1}, "switch_step"),
        ({"handoff": "sideways"}, "handoff"),
        ({"anchor": "bogus"}, "anchor"),
    ],
)
def test_a_bad_switch_step_or_handoff_is_refused_before_the_flag(
    flag_off, settings, match
):
    problems = hybrid_problems(
        generate(num_inference_steps=31, **{HYBRID_WINDOWS_INPUT: settings})
    )
    assert len(problems) == 2, problems
    assert match in problems[0]
    assert HYBRID_WINDOWS_ENV_VAR in problems[1]


def test_the_keystone_anchor_is_refused_with_the_flag_off_naming_the_flag(flag_off):
    settings = {"frames": 42, "overlap": 7, "anchor": "keystone"}
    problems = hybrid_problems(generate(**{HYBRID_WINDOWS_INPUT: settings}))
    assert len(problems) == 1, problems
    assert HYBRID_WINDOWS_ENV_VAR in problems[0]


@pytest.mark.parametrize("anchor", ["none", "keystone"])
def test_both_anchors_are_accepted(flag_on, anchor):
    settings = {"frames": 42, "overlap": 7, "anchor": anchor}
    assert hybrid_problems(generate(**{HYBRID_WINDOWS_INPUT: settings})) == []


@pytest.mark.parametrize("handoff", ["finished", "x0"])
def test_both_handoffs_are_accepted(flag_on, handoff):
    assert hybrid_problems(ordered(3, handoff=handoff)) == []


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
    with pytest.raises(ValueError, match=f".*{HYBRID_WINDOWS_ENV_VAR}"):
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


# 1c. The keystone anchor (#1188 S4)

KEYSTONE = {**ORDERED, "anchor": "keystone"}


def run_keystone(settings, transformer, steps=STEPS):
    """The loop on `transformer`; returns the state and the keystone the
    first step published (it is cleared after the last)."""
    state, scheduler = hybrid_state(settings, steps)
    components = SimpleNamespace(
        transformer=transformer,
        scheduler=scheduler,
        audio_scheduler=MiniMaxH3Scheduler(shift=3.0),
    )
    components.audio_scheduler.set_timesteps(steps)
    denoiser, update = DwH3WindowedLoopDenoiser(), DwH3HostLoopSchedulerStep()
    keystone = None
    for i, t in enumerate(scheduler.timesteps):
        denoiser(components, state, i, t)
        if i == 0:
            keystone = getattr(state, h3_windows.HYBRID_KEYSTONE)
        update(components, state, i, t)
    assert getattr(state, h3_windows.HYBRID_KEYSTONE) is None
    return state, keystone


@pytest.mark.parametrize("switch", [0, 3])
def test_anchor_none_is_the_s2_path_bit_for_bit(flag_on, switch):
    loop = DwH3WindowedLoopDenoiser, DwH3HostLoopSchedulerStep
    before = run_loop(*(c() for c in loop), {**ORDERED, "switch_step": switch}, STEPS)
    none = run_loop(
        *(c() for c in loop),
        {**ORDERED, "switch_step": switch, "anchor": "none"},
        STEPS,
    )
    assert torch.equal(before[0], none[0])
    assert torch.equal(before[1], none[1])


def test_the_prepass_keystone_is_a_single_window_runs_first_frame(flag_on):
    settings = {**KEYSTONE, "switch_step": 0}
    state, keystone = run_keystone(settings, tiny_transformer())
    per_frame = state.latents.shape[0] // LATENT_FRAMES
    assert keystone is not None and keystone.shape[0] == per_frame
    alone = sequential_windows({**ORDERED, "switch_step": STEPS - 1})
    torch.testing.assert_close(keystone, alone[:per_frame])


@pytest.mark.parametrize("switch", [0, 2])
def test_every_window_sees_the_keystone_rows_and_the_take_has_none(flag_on, switch):
    stub = CopyingStub()
    state, keystone = run_keystone({**KEYSTONE, "switch_step": switch}, stub)
    fields = state.denoiser_input_fields
    rows = state.latents.shape[0]
    per_frame = rows // LATENT_FRAMES
    assert keystone.shape[0] == per_frame
    window1 = window_video(state, (0, 7)).numel()
    # window 1 runs every step on its own rows: the pre-pass, or the
    # ordered phase's first window
    first, rest = stub.calls[: STEPS - 1], stub.calls[STEPS - 1 :]
    assert all(c["hidden_states"].shape[1] == window1 for c in first)
    assert rest
    frame0 = fields["position_ids"][fields["video_indices"][:per_frame]]
    for call in rest:
        hidden = call["hidden_states"]
        assert torch.equal(
            hidden[:, :per_frame], keystone[None].expand(hidden.shape[0], -1, -1)
        )
        lead = call["video_indices"][:per_frame]
        assert torch.equal(call["position_ids"][lead], frame0)
        times = call["timestep"][call["timestep_indices"][lead]]
        assert (times == 1.0).all()
        assert (call["token_tags"].reshape(-1)[lead] == 0).all()
    # the fused take has the take's rows, no keystone rows
    assert state.latents.shape[0] == rows
    assert torch.isfinite(state.latents).all()


def test_keystone_attention_is_on_for_ordered_steps_and_off_for_blended(
    flag_on, monkeypatch
):
    from dw.pipeline_processors import h3_keystone_attention

    monkeypatch.setattr(
        h3_windows,
        "KEYSTONE_ATTENTION",
        h3_keystone_attention.KeystoneAttentionSettings(blocks="0-1"),
    )
    transformer = tiny_transformer()
    forwards, resplits = [], []
    transformer.register_forward_pre_hook(lambda *_: forwards.append(1))
    real = h3_keystone_attention.resplit_attention

    def counting(*args, **kwargs):
        resplits.append(len(forwards) - 1)
        return real(*args, **kwargs)

    monkeypatch.setattr(h3_keystone_attention, "resplit_attention", counting)
    published = []
    publish = h3_keystone_attention.publish_window_rows

    def recording(component, keystone_rows, held_rows, frame_rows):
        state = getattr(component, h3_keystone_attention._STATE)
        published.append((state.per_forward, keystone_rows, held_rows, frame_rows))
        return publish(component, keystone_rows, held_rows, frame_rows)

    monkeypatch.setattr(h3_keystone_attention, "publish_window_rows", recording)
    originals = [b.attn.processor for b in transformer.transformer_blocks]
    k = 2
    run_keystone({**KEYSTONE, "switch_step": k}, transformer)
    # the owner's per-forward install took each ordered forward's rows: K the
    # keystone, C the pinned overlap right after it, one latent frame each
    assert len(published) == k
    assert all(per_forward for per_forward, *_ in published)
    for _, (k0, kn), (c0, cn), frame_rows in published:
        assert kn == frame_rows and cn % frame_rows == 0 and c0 > k0
    # window 1's STEPS - 1 forwards, window 2's k ordered ones, then blended
    ordered = range(STEPS - 1, STEPS - 1 + k)
    assert len(forwards) > STEPS - 1 + k
    assert sorted(set(resplits)) == list(ordered)
    assert len(resplits) == k * 2  # two blocks per forward
    # the original processors are back
    assert [b.attn.processor for b in transformer.transformer_blocks] == originals
    assert not h3_keystone_attention.installed(transformer)


def test_keystone_attention_blocks_past_the_transformer_raise(flag_on, monkeypatch):
    from dw.pipeline_processors import h3_keystone_attention

    monkeypatch.setattr(
        h3_windows,
        "KEYSTONE_ATTENTION",
        h3_keystone_attention.KeystoneAttentionSettings(blocks="0-5"),
    )
    transformer = tiny_transformer()
    originals = [b.attn.processor for b in transformer.transformer_blocks]
    with pytest.raises(ValueError, match="anchor keystone"):
        run_keystone({**KEYSTONE, "switch_step": 2}, transformer)
    assert [b.attn.processor for b in transformer.transformer_blocks] == originals
    assert not h3_keystone_attention.installed(transformer)
