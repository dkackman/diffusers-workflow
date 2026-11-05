"""The experimental SelfLift (#1195, #1199 Task 19): the flag gate, the rules,
the half-canvas layout against diffusers' own, the sigma-consistent lift, and
the real H3 graph on tiny weights. dw/pipeline_processors/h3_selflift.py and
h3_selflift_steps.py."""

from types import SimpleNamespace

import pytest
import torch

minimax = pytest.importorskip("diffusers.modular_pipelines.minimax_h3")
from diffusers.modular_pipelines.minimax_h3.before_denoise import (  # noqa: E402
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
    _frame_position_grid,
    patchify_video_latents,
)

from dw.guides import selflift_errors  # noqa: E402
from dw.pipeline_processors import h3_rules, h3_selflift  # noqa: E402
from dw.pipeline_processors.h3_hold import insert_audio_hold  # noqa: E402
from dw.pipeline_processors.h3_selflift import (  # noqa: E402
    SELFLIFT_BLOCK,
    SELFLIFT_INPUT,
    detail_noise,
    insert_selflift,
    lift,
    lifts,
    patchify,
    selflift_problems,
    selflift_refusal,
    shrink_clean,
    shrink_layout,
    shrink_noise,
    unpatchify,
)

FLAG = h3_rules.REFINE_WINDOWS_ENV_VAR
PATCH = (1, 2, 2)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)


def call(**overrides):
    arguments = {
        "prompt": "a cat",
        "width": 1024,
        "height": 576,
        "num_frames": 124,
        "num_inference_steps": 5,
        SELFLIFT_INPUT: {"lift_step": 2, "start_scale": 0.5},
    }
    arguments.update(overrides)
    return {k: v for k, v in arguments.items() if v is not None}


def h3_workflow(arguments=None, workflow="fl2va", configuration=None, chain=None):
    pipeline = {
        "configuration": {"component_type": "ModularPipeline", **(configuration or {})},
        "from_pretrained_arguments": {
            "model_name": "MiniMaxAI/MiniMax-H3",
            "workflow": workflow,
        },
        "arguments": call(**(arguments or {})),
    }
    if chain is not None:
        pipeline["chain"] = chain
    return {"steps": [{"name": "render", "pipeline": pipeline}]}


PATH = f"steps[0].pipeline.arguments.{SELFLIFT_INPUT}"


# 1. The flag gate, at every entry point


def test_validation_refuses_the_argument_with_the_flag_off(flag_off):
    errors = selflift_errors(h3_workflow())
    assert [error["path"] for error in errors] == [PATH]
    assert FLAG in errors[0]["message"]


def test_the_validator_runs_the_check(flag_off, tmp_path):
    import json

    from dw.validation import workflow_errors
    from dw.workflow import workflow_from_file

    definition = {"id": "SelfLift", **h3_workflow()}
    path = tmp_path / "selflift.json"
    path.write_text(json.dumps(definition))
    errors = workflow_errors(workflow_from_file(str(path), str(tmp_path)))
    assert [error["path"] for error in errors if SELFLIFT_INPUT in error["path"]]


def stock_t2va():
    pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
    insert_audio_hold(pipeline)
    return pipeline


def test_the_run_refuses_with_the_flag_off(flag_off):
    from dw.pipeline_processors.pipeline import Pipeline

    pipeline = stock_t2va()
    insert_selflift(pipeline)
    step = SimpleNamespace(name="render", pipeline=pipeline, configuration={})
    with pytest.raises(ValueError, match=f"Step 'render': .*{FLAG}=1"):
        Pipeline._check_selflift(step, call())


def test_the_run_refuses_a_graph_without_the_block(flag_on):
    assert "no loop that lifts" in selflift_refusal(stock_t2va(), call())


def test_the_run_takes_a_valid_call_and_refuses_a_cache(flag_on):
    from dw.pipeline_processors.pipeline import Pipeline

    pipeline = stock_t2va()
    insert_selflift(pipeline)
    step = SimpleNamespace(name="render", pipeline=pipeline, configuration={})
    assert Pipeline._check_selflift(step, call()) is None
    step.configuration = {"cache": {"type": "first_block"}}
    with pytest.raises(ValueError, match="cache"):
        Pipeline._check_selflift(step, call())


def test_absent_or_null_is_never_refused(flag_off):
    assert selflift_errors(h3_workflow({SELFLIFT_INPUT: None})) == []
    assert selflift_refusal(stock_t2va(), call(**{SELFLIFT_INPUT: None})) is None


@pytest.mark.parametrize("value, lifted", [(None, False), ("1", True)])
def test_the_loader_inserts_the_block_only_behind_the_flag(monkeypatch, value, lifted):
    from dw.pipeline_processors.components import _assemble_loaded

    if value is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, value)
    pipeline = minimax.MiniMaxH3Blocks().get_workflow("t2va").init_pipeline()
    _assemble_loaded("pipeline", type(pipeline), {}, pipeline, {})
    assert lifts(pipeline) is lifted


def test_the_block_goes_in_front_of_the_denoiser_and_not_on_ref2va():
    pipeline = minimax.MiniMaxH3Blocks().init_pipeline()
    assert insert_selflift(pipeline)
    assert insert_selflift(pipeline)  # idempotent
    loops = {
        name: sequence.sub_blocks["denoise"]
        for name, sequence in pipeline._blocks.sub_blocks["denoise"].sub_blocks.items()
    }
    for name in ("t2va", "fl2va"):
        assert list(loops[name].sub_blocks)[:2] == [SELFLIFT_BLOCK, "denoiser"]
    assert SELFLIFT_BLOCK not in loops["ref2va"].sub_blocks


# 2. The rules, at validate time and the run's own


def test_a_valid_call_has_no_problems(flag_on):
    assert selflift_problems(call()) == []
    assert selflift_errors(h3_workflow()) == []


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({SELFLIFT_INPUT: 2}, "lift_step, start_scale"),
        ({SELFLIFT_INPUT: {"lift_step": 2, "blend": 1}}, "not blend"),
        ({SELFLIFT_INPUT: {"lift_step": 0}}, "1 or more"),
        ({SELFLIFT_INPUT: {"lift_step": 1.5}}, "whole number"),
        ({SELFLIFT_INPUT: {"lift_step": 2, "start_scale": 0.75}}, "start_scale"),
        ({SELFLIFT_INPUT: {"lift_step": 2, "upsample": "bicubic"}}, "upsample"),
        ({SELFLIFT_INPUT: {"lift_step": 4}}, "no evaluation on the full canvas"),
        ({"height": 544, "width": 960}, "multiple of 64"),
        ({"latents": "previous_result:x"}, "latents"),
        ({"refine_strength": 0.3}, "refine_strength"),
        ({"source_video": "asset:take.mp4"}, "source_video"),
        ({"mask_video": "asset:m.mp4"}, "mask_video"),
        ({"references": [{"image": "asset:a.png"}]}, "references"),
        ({h3_rules.REFINE_WINDOWS_INPUT: {"frames": 12}}, "experimental_refine"),
    ],
)
def test_a_call_that_cannot_lift_is_refused(flag_on, overrides, match):
    problems = selflift_problems(call(**overrides))
    assert any(match in problem for problem in problems), problems
    errors = selflift_errors(h3_workflow(overrides))
    assert errors and all(error["path"] == PATH for error in errors)


def test_validation_refuses_off_h3_on_ref2va_with_a_cache_or_a_capstone(flag_on):
    assert "ref2va" in selflift_errors(h3_workflow(workflow="ref2va"))[0]["message"]
    cache = {"cache": {"type": "first_block"}}
    assert "cache" in selflift_errors(h3_workflow(configuration=cache))[0]["message"]
    chain = {"segments": 3, "experimental_capstone": {"enter_below_sigma": 0.5}}
    assert "capstone" in selflift_errors(h3_workflow(chain=chain))[0]["message"]
    workflow = h3_workflow()
    workflow["steps"][0]["pipeline"]["configuration"]["component_type"] = "FluxPipeline"
    assert "MiniMax-H3" in selflift_errors(workflow)[0]["message"]


def test_unresolved_values_are_left_to_the_run(flag_on):
    assert selflift_problems(call(**{SELFLIFT_INPUT: "variable:lift"})) == []
    lifted = {"lift_step": "variable:k", "start_scale": 0.5}
    assert selflift_problems(call(**{SELFLIFT_INPUT: lifted})) == []
    assert selflift_problems(call(width="variable:w")) == []


# 3. The packing and the half canvas


def test_patchify_is_diffusers_and_unpatchify_inverts_it():
    latents = torch.randn(1, 24, 5, 6, 8)
    rows = patchify(latents, PATCH)
    assert torch.equal(rows, patchify_video_latents(latents, PATCH))
    assert torch.equal(unpatchify(rows, 5, 6, 8, PATCH), latents)


def test_shrunk_noise_is_unit_noise_and_clean_latents_are_block_means():
    noise = torch.randn(1, 24, 9, 64, 64, generator=torch.Generator().manual_seed(1))
    small = shrink_noise(noise, 2)
    assert small.shape == (1, 24, 9, 32, 32)
    assert abs(float(small.std()) - 1) < 0.01 and abs(float(small.mean())) < 0.01
    clean = torch.arange(16.0).reshape(1, 1, 1, 4, 4)
    assert shrink_clean(clean, 2).flatten().tolist() == [2.5, 4.5, 10.5, 12.5]


def layout(height, width, anchors=(), frames=7):
    return MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.ones(11, dtype=torch.long),
        frames,
        height,
        width,
        12,
        PATCH,
        2,
        2,
        0,
        anchors,
    )


def fields_of(packed):
    names = (
        "position_ids",
        "token_tags",
        "video_indices",
        "audio_indices",
        "text_indices",
    )
    return dict(zip(names, packed[:5])), packed[5]


def plan_of(packed, steps=3):
    fields, conditions = fields_of(packed)
    return [
        MiniMaxH3SetTimestepsStep.build_row_timesteps(
            fields["video_indices"],
            fields["audio_indices"],
            conditions,
            0,
            fields["text_indices"].numel(),
            0.1 * step,
            0.2 * step,
            0.999,
            1.0,
        )
        for step in range(steps)
    ]


@pytest.mark.parametrize("anchors", [(), ("first",), ("first", "last")])
@pytest.mark.parametrize("height, width", [(8, 12), (12, 8), (36, 64)])
def test_the_shrunk_layout_is_diffusers_own_at_the_half_canvas(anchors, height, width):
    full, half = (
        layout(height, width, anchors),
        layout(height // 2, width // 2, anchors),
    )
    full_fields, _ = fields_of(full)
    half_fields, half_conditions = fields_of(half)
    full_grid = _frame_position_grid(height, width, 2, 2)[0]
    half_grid = _frame_position_grid(height // 2, width // 2, 2, 2)[0]
    shrunk, plan = shrink_layout(
        full_fields, plan_of(full), full_grid.shape[0], full_grid, half_grid
    )
    for name, value in half_fields.items():
        assert torch.equal(shrunk[name], value), name
    for (timesteps, indices), (expected_t, expected_i) in zip(plan, plan_of(half)):
        assert torch.equal(timesteps[indices], expected_t[expected_i])
    assert half_conditions == len(anchors) * half_grid.shape[0]


def test_a_layout_off_the_canvas_grid_is_refused():
    full_fields, _ = fields_of(layout(8, 12, ("first",)))
    grid = _frame_position_grid(8, 12, 2, 2)[0]
    moved = dict(full_fields)
    moved["position_ids"] = full_fields["position_ids"].clone()
    moved["position_ids"][full_fields["video_indices"][0], 1] += 1
    with pytest.raises(ValueError, match="canvas grid"):
        shrink_layout(moved, [], grid.shape[0], grid, grid[: grid.shape[0] // 4])


# 4. The lift: shape, and sigma-consistent noise


def test_the_lift_is_the_full_canvas_shape():
    small = torch.randn(1, 24, 7, 4, 6)
    noise = torch.randn(1, 24, 7, 8, 12)
    for mode in ("bilinear", "nearest"):
        assert lift(small, small, 0.8, noise, 2, mode).shape == (1, 24, 7, 8, 12)


def test_a_lift_at_sigma_one_with_nothing_denoised_is_the_draw_itself():
    noise = torch.randn(1, 24, 7, 8, 12, generator=torch.Generator().manual_seed(2))
    small = shrink_noise(noise, 2)
    lifted = lift(small, torch.zeros_like(small), 1.0, noise, 2)
    assert torch.allclose(lifted, noise, atol=1e-6)


def test_the_lifted_noise_is_unit_noise_whose_low_band_is_the_trajectorys():
    generator = torch.Generator().manual_seed(3)
    eps = torch.randn(1, 24, 5, 32, 32, generator=generator)
    noise = torch.randn(1, 24, 5, 64, 64, generator=generator)
    # x0 = 0, so x = sigma * eps and v = x0 - eps: the lift is sigma * E
    sigma = 0.86
    lifted = lift(sigma * eps, -eps, sigma, noise, 2) / sigma
    assert abs(float(lifted.std()) - 1) < 0.01
    assert torch.allclose(shrink_noise(lifted, 2), eps, atol=1e-5)
    # ... and the high band is the call's own draw, not a fresh one
    assert torch.allclose(detail_noise(lifted, 2), detail_noise(noise, 2), atol=1e-5)
    # neighbours in a block are uncorrelated, as in fresh noise
    a, b = lifted[..., 0::2, 0::2].flatten(), lifted[..., 0::2, 1::2].flatten()
    assert abs(float(torch.corrcoef(torch.stack([a, b]))[0, 1])) < 0.02


def test_the_lifted_signal_is_the_upsampled_denoised_estimate():
    small = torch.randn(1, 24, 3, 4, 4)
    velocity = torch.randn(1, 24, 3, 4, 4)
    sigma = 0.5
    lifted = lift(small, velocity, sigma, torch.zeros(1, 24, 3, 8, 8), 2, "nearest")
    denoised = small + sigma * velocity
    expected = (1 - sigma) * denoised + sigma * (denoised - velocity) / 2
    assert torch.allclose(shrink_clean(lifted, 2), expected, atol=1e-6)


# 5. The real H3 graph on tiny weights


@pytest.fixture(scope="module")
def tiny_h3():
    from dw.pipeline_processors.h3_guides import insert_guides
    from tests.test_chain_h3_native import _tiny_h3

    pipeline = _tiny_h3()
    insert_guides(pipeline)
    insert_audio_hold(pipeline)
    assert insert_selflift(pipeline)
    return pipeline


class Forwards:
    """What every transformer forward of a call saw."""

    def __init__(self, transformer):
        self.calls = []
        self.handle = transformer.register_forward_pre_hook(self.hook, with_kwargs=True)

    def hook(self, module, args, kwargs):
        self.calls.append(
            {
                "rows": kwargs["hidden_states"].shape[1],
                "audio": kwargs["audio_hidden_states"].clone(),
                "position_ids": kwargs["position_ids"].clone(),
            }
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.handle.remove()


def render(pipeline, selflift=None, steps=4, **extra):
    arguments = dict(
        prompt="a cat sat",
        height=64,
        width=64,
        num_frames=124,
        num_inference_steps=steps,
        generator=torch.Generator("cpu").manual_seed(1234),
        output="latents",
        **extra,
    )
    if selflift is not None:
        arguments[SELFLIFT_INPUT] = selflift
    with Forwards(pipeline.transformer) as forwards:
        latents = pipeline(**arguments)
    return latents, forwards.calls


# 124 frames: 37 latent frames of 4x4 latents, 4 rows a frame (1 at half)
FULL_ROWS, HALF_ROWS = 37 * 4, 37


@pytest.mark.parametrize("lift_step", [1, 2])
def test_end_to_end_the_early_forwards_run_small_and_the_take_is_full_size(
    tiny_h3, lift_step
):
    stock, stock_calls = render(tiny_h3)
    lifted, calls = render(tiny_h3, {"lift_step": lift_step, "start_scale": 0.5})
    assert [c["rows"] for c in calls] == [HALF_ROWS] * lift_step + [FULL_ROWS] * (
        3 - lift_step
    )
    assert lifted.shape == stock.shape == (1, 24, 37, 4, 4)
    assert torch.isfinite(lifted).all() and not torch.equal(lifted, stock)
    # positions rebuilt: after the lift every forward is the stock layout
    for call_, stock_call in zip(calls[lift_step:], stock_calls[lift_step:]):
        assert torch.equal(call_["position_ids"], stock_call["position_ids"])
    # audio rows: the same count in every forward, and the first forward's
    # audio is the stock call's (the draw is the same, the canvas is not)
    assert len({c["audio"].shape for c in calls}) == 1
    assert torch.equal(calls[0]["audio"], stock_calls[0]["audio"])


def test_end_to_end_the_lift_leaves_the_audio_rows_alone(tiny_h3, monkeypatch):
    from dw.pipeline_processors import h3_selflift_steps

    seen = {}
    original = h3_selflift_steps.DwH3SelfLiftStep._lift

    def spy(self, components, block_state, i, settings):
        before = block_state.audio_latents
        seen["values"] = before.clone()
        result = original(self, components, block_state, i, settings)
        seen["same"] = block_state.audio_latents is before
        seen["equal"] = torch.equal(block_state.audio_latents, seen["values"])
        return result

    monkeypatch.setattr(h3_selflift_steps.DwH3SelfLiftStep, "_lift", spy)
    render(tiny_h3, {"lift_step": 1})
    assert seen == {**seen, "same": True, "equal": True}


def test_end_to_end_a_guides_condition_rows_shrink_and_come_back(tiny_h3):
    """A guide's frames are condition rows on the call's canvas, as a fl2va
    keyframe's frame is (the tiny text encoder takes no image, so the
    keyframe's own layout is pinned against diffusers' above)."""
    import numpy as np

    pixels = np.random.default_rng(5).integers(0, 255, (22, 64, 64, 3), "uint8")
    guide = {"video": pixels, "frame": 51, "audio": None, "sample_rate": None}
    stock, stock_calls = render(tiny_h3, guides=[guide])
    lifted, calls = render(tiny_h3, {"lift_step": 1}, guides=[guide])
    # the guide's 7 latent frames of condition rows ride along
    rows = 37 + 7
    assert stock_calls[0]["rows"] == rows * 4
    assert [c["rows"] for c in calls] == [rows, rows * 4, rows * 4]
    assert lifted.shape == stock.shape and torch.isfinite(lifted).all()
    assert not torch.equal(lifted, stock)
    for call_, stock_call in zip(calls[1:], stock_calls[1:]):
        assert torch.equal(call_["position_ids"], stock_call["position_ids"])


def test_end_to_end_a_lift_past_the_schedule_is_refused_in_the_loop(tiny_h3):
    with pytest.raises(ValueError, match="no evaluation on the full canvas"):
        render(tiny_h3, {"lift_step": 3})


def test_end_to_end_a_canvas_that_does_not_halve_is_refused_in_the_loop(tiny_h3):
    with pytest.raises(ValueError, match="does not divide"):
        tiny_h3(
            prompt="a cat sat",
            height=32,
            width=32,
            num_frames=124,
            num_inference_steps=4,
            generator=torch.Generator("cpu").manual_seed(1),
            output="latents",
            **{SELFLIFT_INPUT: {"lift_step": 1}},
        )


def test_the_saved_fraction_is_the_small_forwards_share():
    assert h3_selflift.saved_fraction(2, 4) == pytest.approx(0.375)
    assert h3_selflift.saved_fraction(4, 8) == pytest.approx(0.375)
    assert h3_selflift.saved_fraction(1, 4) == pytest.approx(0.1875)
