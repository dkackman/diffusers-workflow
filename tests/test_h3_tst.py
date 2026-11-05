"""The experimental TST temporal-attention correction (#1196, #1199 Task 19):
`experimental_tst` on an H3 call puts a per-call attention processor on every
transformer block that measures the target video's frame-to-frame attention
(Temporal State Transport) and scales each head's attention output by
`1 + tau_effective * q`.

The flag gate and the validate-time refusals, the processor's install and
restore (no processor, hook or diffusers global left behind), the gain against
a port of the MIT reference formula on small tensors, the processor against
the stock one, and the real H3 graph on tiny weights."""

import math

import pytest
import torch

pytest.importorskip("diffusers.modular_pipelines.minimax_h3")

from dw.pipeline_processors import h3_rules, h3_tst  # noqa: E402
from dw.pipeline_processors.h3_tst import (  # noqa: E402
    TST_INPUT,
    TstSettings,
    layer_weight,
    step_weight,
    temporal_quality,
    tst_call,
    tst_errors,
    tst_refusal,
    tst_settings,
)

FLAG = h3_rules.REFINE_WINDOWS_ENV_VAR


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)


# The oracle: lytang63/temporal-state-transport tst/core.py `tst_modulation`
# (MIT, Copyright (c) 2025 TST Authors), ported with its globals passed in and
# the matmul taken in float32. The module under test is a chunked, per-head
# vectorization of it; this copy stays as close to the source as it can.
def reference_tst_scale(
    query_image, key_image, head_dim, num_frames, tau_peak, layer_index,
    total_layers, step_index, total_steps, eps=1e-8,
):  # fmt: skip
    scale = head_dim**-0.5
    query_image = query_image.float() * scale
    attn_logits = query_image @ key_image.float().transpose(-2, -1)
    attn_temp = attn_logits.softmax(dim=-1)
    attn_temp = attn_temp.reshape(-1, num_frames, num_frames)
    mean_attn = attn_temp.mean(dim=0)
    gram_all = attn_temp @ attn_temp.transpose(-2, -1)
    trace_all = torch.diagonal(gram_all, dim1=-2, dim2=-1).sum(dim=-1).clamp(min=eps)
    rho = (gram_all / trace_all[:, None, None]).mean(dim=0)
    rho = rho / torch.diagonal(rho, dim1=-2, dim2=-1).sum().clamp(min=eps)
    eigvals = torch.linalg.eigvalsh(rho).clamp(min=eps)
    eigvals = eigvals / eigvals.sum().clamp(min=eps)
    entropy = -(eigvals * eigvals.log()).sum()
    log_f = torch.log(torch.tensor(float(num_frames)))
    entropy_norm = (entropy / log_f.clamp(min=eps)).clamp(min=0.0, max=1.0)
    diag_mask = torch.eye(num_frames).bool()
    diag_mean = mean_attn.masked_select(diag_mask).mean()
    temporal_alpha = (1.0 - diag_mean).clamp(min=0.0, max=1.0)
    spectral_q = entropy_norm * (1.0 - entropy_norm)
    q_tst = temporal_alpha * spectral_q
    lw = (
        1.0
        if total_layers == 1
        else 0.5 - 0.5 * math.cos(math.pi * layer_index / (total_layers - 1))
    )
    sw = (
        1.0
        if total_steps == 1
        else 0.5 + 0.5 * math.cos(math.pi * step_index / (total_steps - 1))
    )
    return 1.0 + tau_peak * lw * sw * q_tst


# 1. The flag gate, and what validation refuses


def h3_workflow(arguments=None, configuration=None, workflow="fl2va"):
    return {
        "steps": [
            {
                "name": "clip",
                "pipeline": {
                    "configuration": {
                        "component_type": "ModularPipeline",
                        **(configuration or {}),
                    },
                    "from_pretrained_arguments": {
                        "model_name": "MiniMaxAI/MiniMax-H3",
                        "workflow": workflow,
                    },
                    "arguments": {"num_frames": 124, **(arguments or {})},
                },
            }
        ]
    }


PATH = f"steps[0].pipeline.arguments.{TST_INPUT}"


def test_validation_refuses_the_argument_with_the_flag_off(flag_off):
    errors = tst_errors(h3_workflow({TST_INPUT: {"tau": 0.2}}))
    assert [e["path"] for e in errors] == [PATH]
    assert FLAG in errors[0]["message"]


def test_the_run_refuses_the_argument_with_the_flag_off(flag_off):
    with pytest.raises(ValueError, match=FLAG):
        tst_settings({"tau": 0.2})


@pytest.mark.parametrize("value", [None, False])
def test_off_or_absent_is_never_refused(flag_off, value):
    assert tst_errors(h3_workflow({TST_INPUT: value})) == []
    assert tst_errors(h3_workflow()) == []
    assert tst_settings(value) is None


def test_the_flag_on_accepts_a_dense_h3_step(flag_on):
    assert tst_errors(h3_workflow({TST_INPUT: {"tau": 0.2}})) == []
    assert tst_settings({"tau": 0.2}) == TstSettings(tau=0.2, per_head=True)
    assert tst_settings({"tau": 0, "per_head": False}) == TstSettings(0.0, False)


@pytest.mark.parametrize(
    "value, match",
    [
        (0.2, "an object"),
        ({}, "tau"),
        ({"tau": True}, "tau"),
        ({"tau": -0.1}, "tau"),
        ({"tau": 1.5}, "tau"),
        ({"tau": float("nan")}, "tau"),
        ({"tau": 0.2, "per_head": "yes"}, "per_head"),
        ({"tau": 0.2, "strength": 1}, "strength"),
    ],
)
def test_a_bad_value_is_refused_at_validate_and_at_run(flag_on, value, match):
    errors = tst_errors(h3_workflow({TST_INPUT: value}))
    assert [e["path"] for e in errors] == [PATH]
    assert match in errors[0]["message"]
    with pytest.raises(ValueError, match=match):
        tst_settings(value)


@pytest.mark.parametrize(
    "configuration",
    [
        {"attention_backend": "sparse"},
        {"components": {"transformer": {"attention_backend": "sparse"}}},
    ],
)
def test_sparse_attention_is_refused_at_validate_time(flag_on, configuration):
    errors = tst_errors(h3_workflow({TST_INPUT: {"tau": 0.2}}, configuration))
    assert [e["path"] for e in errors] == [PATH]
    assert "dense" in errors[0]["message"] and "sparse" in errors[0]["message"]


def test_another_attention_backend_is_not_a_conflict(flag_on):
    configuration = {"components": {"transformer": {"attention_backend": "native"}}}
    assert tst_errors(h3_workflow({TST_INPUT: {"tau": 0.2}}, configuration)) == []


def test_a_step_off_h3_is_refused(flag_on):
    workflow = h3_workflow({TST_INPUT: {"tau": 0.2}})
    workflow["steps"][0]["pipeline"]["configuration"]["component_type"] = "FluxPipeline"
    errors = tst_errors(workflow)
    assert [e["path"] for e in errors] == [PATH]
    assert "MiniMax-H3" in errors[0]["message"]


def test_unresolved_values_are_left_to_the_run(flag_on):
    workflow = h3_workflow(
        {TST_INPUT: "variable:tst"},
        {"components": {"transformer": {"attention_backend": "variable:attn"}}},
    )
    assert tst_errors(workflow) == []
    workflow = h3_workflow({TST_INPUT: {"tau": "variable:tau"}})
    assert tst_errors(workflow) == []


def test_validation_runs_the_check(flag_off):
    from dw.workflow import Workflow

    definition = h3_workflow({TST_INPUT: {"tau": 0.2}})
    definition["steps"][0]["pipeline"]["arguments"]["prompt"] = "a cat"
    definition["id"] = "tst"
    errors = Workflow(definition, None, "tst.json").validation_errors({})
    assert any(TST_INPUT in str(e) and FLAG in str(e) for e in errors), errors


# 2. The schedule and the gain against the reference formula


def test_the_schedules_are_the_references():
    assert layer_weight(0, 1) == 1.0 and step_weight(0, 1) == 1.0
    assert layer_weight(0, 5) == pytest.approx(0.0)
    assert layer_weight(4, 5) == pytest.approx(1.0)
    assert layer_weight(2, 5) == pytest.approx(0.5)
    assert step_weight(0, 5) == pytest.approx(1.0)
    assert step_weight(4, 5) == pytest.approx(0.0)
    # a step past the plan (a refine, a re-roll) clamps to the last
    assert step_weight(9, 5) == pytest.approx(0.0)


def _qk(spatial=6, heads=3, frames=5, dim=8, seed=0, spread=None):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(spatial, heads, frames, dim, generator=generator)
    k = torch.randn(spatial, heads, frames, dim, generator=generator)
    if spread is not None:
        # heads that differ: one sharp, one flat, one in between
        q = q * torch.tensor(spread)[None, :, None, None]
    return q, k


@pytest.mark.parametrize("chunk", [None, 1, 4])
def test_the_whole_layer_gain_is_the_reference(chunk):
    q, k = _qk(spread=[4.0, 0.1, 1.0])
    quality = temporal_quality(q, k, per_head=False, chunk=chunk)
    expected = reference_tst_scale(q, k, 8, 5, 1.0, 0, 1, 0, 1) - 1.0
    assert quality.shape == (1,)
    torch.testing.assert_close(quality[0], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("chunk", [None, 2])
def test_each_head_gain_is_the_reference_on_that_head_alone(chunk):
    q, k = _qk(spread=[4.0, 0.1, 1.0])
    quality = temporal_quality(q, k, per_head=True, chunk=chunk)
    assert quality.shape == (3,)
    for h in range(3):
        one = reference_tst_scale(
            q[:, h : h + 1], k[:, h : h + 1], 8, 5, 1.0, 0, 1, 0, 1
        )
        torch.testing.assert_close(quality[h], one - 1.0, rtol=1e-5, atol=1e-6)
    # the heads really differ, so per-head is not the layer's gain repeated
    assert quality.max() - quality.min() > 1e-3


def test_the_scheduled_gain_is_the_reference():
    from dw.pipeline_processors.h3_tst import scheduled_gain

    q, k = _qk(seed=3)
    for layer, step in ((1, 0), (2, 1), (3, 3)):
        gain = scheduled_gain(
            temporal_quality(q, k, per_head=False), 0.2, layer, 4, step, 4
        )
        expected = reference_tst_scale(q, k, 8, 5, 0.2, layer, 4, step, 4)
        torch.testing.assert_close(gain[0], expected, rtol=1e-5, atol=1e-6)


def test_the_gain_is_never_below_one_and_at_most_one_plus_a_quarter_tau():
    # The released code's q is alpha * E * (1 - E), in [0, 1/4]: it only ever
    # amplifies, so the strongest correction at tau 0.2 is a gain of 1.05
    for seed in range(5):
        q, k = _qk(seed=seed)
        quality = temporal_quality(q, k, per_head=True)
        assert (quality >= 0).all() and (quality <= 0.25 + 1e-6).all()


def test_one_frame_has_no_temporal_attention_to_correct():
    q, k = _qk(frames=1)
    assert torch.equal(temporal_quality(q, k, per_head=True), torch.zeros(3))


# 3. The processor on a real H3 block, and its install and restore


def tiny_transformer(layers=3):
    from diffusers import MiniMaxH3Transformer3DModel

    torch.manual_seed(0)
    return MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=32,
        num_layers=layers,
        num_refiner_layers=1,
        ffn_dim=32,
        text_dim=16,
        freq_dim=16,
        time_embed_hidden_dim=16,
        time_embed_dim=16,
        rope_freq_dim=2,
    ).eval()


def packed_layout(text=3, audio=2, frames=4, height=2, width=3):
    """text rows, audio rows, then the target video frame-major, as H3 packs
    one t2va call; the video rows' position ids are a (t, h, w) grid."""
    video = frames * height * width
    tags = torch.tensor([1] * text + [2] * audio + [0] * video)
    positions = torch.zeros(len(tags), 3)
    positions[: text + audio, 0] = torch.arange(text + audio).float()
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(frames).float() * 0.7,
            torch.arange(height).float() / height,
            torch.arange(width).float() / width,
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    positions[text + audio :] = grid
    return positions, tags, text + audio


class TinyPipeline:
    """A pipeline holding one H3 transformer, as `named_pipeline_components`
    reads a modular pipeline."""

    def __init__(self, transformer):
        self.components = {"transformer": transformer}


def stock_processors(transformer):
    return [block.attn.processor for block in transformer.transformer_blocks]


def test_the_processors_are_installed_for_the_call_and_restored(flag_on):
    transformer = tiny_transformer()
    before = stock_processors(transformer)
    pre, post = (
        dict(transformer._forward_pre_hooks),
        dict(transformer._forward_hooks),
    )
    with tst_call(TinyPipeline(transformer), TstSettings(0.2, True), 4, "clip"):
        during = stock_processors(transformer)
        assert all(isinstance(p, h3_tst.TstH3AttnProcessor) for p in during)
        assert [p._original for p in during] == before
    assert stock_processors(transformer) == before
    assert dict(transformer._forward_pre_hooks) == pre
    assert dict(transformer._forward_hooks) == post
    assert not hasattr(transformer, "_dw_tst_state")


def test_the_processors_are_restored_when_the_call_raises(flag_on):
    transformer = tiny_transformer()
    before = stock_processors(transformer)
    with pytest.raises(RuntimeError, match="boom"):
        with tst_call(TinyPipeline(transformer), TstSettings(0.2, True), 4, "clip"):
            raise RuntimeError("boom")
    assert stock_processors(transformer) == before
    assert not transformer._forward_pre_hooks


def test_the_step_ramp_follows_h3s_evaluation_count(flag_on):
    """H3's num_inference_steps N counts sigma grid points: N - 1 evaluations.
    At N = 5 the four steps weigh 1, .75, .25, 0 - the reference's ramp."""
    transformer = tiny_transformer()
    with tst_call(TinyPipeline(transformer), TstSettings(0.2, True), 5, "clip"):
        state = transformer._dw_tst_state
        assert state.total_steps == 4
        weights = [step_weight(i, state.total_steps) for i in range(4)]
    assert weights == pytest.approx([1.0, 0.75, 0.25, 0.0])


def test_no_step_count_uses_the_pipelines_default_so_the_ramp_still_applies(flag_on):
    transformer = tiny_transformer()
    with tst_call(TinyPipeline(transformer), TstSettings(0.2, True), None, "clip"):
        total = transformer._dw_tst_state.total_steps
    assert total == h3_tst.DEFAULT_STEPS - 1 > 1
    assert step_weight(total - 1, total) == pytest.approx(0.0)
    assert step_weight(0, total) == pytest.approx(1.0)


def test_a_failed_install_partway_is_still_restored(flag_on, monkeypatch):
    transformer = tiny_transformer()
    before = stock_processors(transformer)
    blocks = list(transformer.transformer_blocks)
    real = blocks[1].attn.set_processor

    def boom(processor):
        monkeypatch.setattr(blocks[1].attn, "set_processor", real)
        raise RuntimeError("second block")

    monkeypatch.setattr(blocks[1].attn, "set_processor", boom)
    with pytest.raises(RuntimeError, match="second block"):
        with tst_call(TinyPipeline(transformer), TstSettings(0.2, True), 4, "clip"):
            pass  # pragma: no cover
    assert stock_processors(transformer) == before
    assert not hasattr(transformer, "_dw_tst_state")
    assert not transformer._forward_pre_hooks


def test_no_diffusers_attention_global_is_touched(flag_on):
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    transformer = tiny_transformer()
    active = _AttentionBackendRegistry.get_active_backend()
    backends = [p._attention_backend for p in stock_processors(transformer)]
    with tst_call(TinyPipeline(transformer), TstSettings(0.2, True), 4, "clip"):
        assert _AttentionBackendRegistry.get_active_backend() == active
    assert _AttentionBackendRegistry.get_active_backend() == active
    assert [p._attention_backend for p in stock_processors(transformer)] == backends


def test_a_component_backend_set_during_the_call_lands_on_the_original(flag_on):
    transformer = tiny_transformer()
    original = stock_processors(transformer)[0]
    with tst_call(TinyPipeline(transformer), TstSettings(0.2, True), 4, "clip"):
        stock_processors(transformer)[0]._attention_backend = "native"
    assert original._attention_backend == "native"


def test_the_run_refuses_a_component_running_sparse(flag_on):
    from dw.sparse_attention import install_sparse

    transformer = tiny_transformer()
    assert install_sparse(transformer, 0.5)
    refusal = tst_refusal(TinyPipeline(transformer))
    assert "dense" in refusal and "sparse" in refusal


def test_the_run_refuses_a_pipeline_with_no_h3_transformer(flag_on):
    refusal = tst_refusal(TinyPipeline(torch.nn.Linear(2, 2)))
    assert "MiniMax-H3 transformer" in refusal


def _forward(transformer, positions, tags, timestep=0.5, seed=1):
    generator = torch.Generator().manual_seed(seed)
    video = int((tags == 0).sum())
    audio = int((tags == 2).sum())
    text = int((tags == 1).sum())
    channels = transformer.config.in_channels * math.prod(transformer.config.patch_size)
    return transformer(
        hidden_states=torch.randn(1, video, channels, generator=generator),
        audio_hidden_states=torch.randn(
            1, audio, transformer.config.audio_in_channels, generator=generator
        ),
        encoder_hidden_states=torch.randn(
            1, text, transformer.config.text_dim, generator=generator
        ),
        timestep=torch.tensor([timestep]),
        timestep_indices=torch.zeros(len(tags), dtype=torch.long),
        token_tags=tags,
        position_ids=positions,
        video_indices=(tags == 0).nonzero().squeeze(1),
        audio_indices=(tags == 2).nonzero().squeeze(1),
        text_indices=(tags == 1).nonzero().squeeze(1),
        return_dict=False,
    )


def test_tau_zero_is_the_stock_forward_bit_exact(flag_on):
    transformer = tiny_transformer()
    positions, tags, _ = packed_layout()
    with torch.no_grad():
        stock = _forward(transformer, positions, tags)
        with tst_call(TinyPipeline(transformer), TstSettings(0.0, True), 4, "clip"):
            corrected = _forward(transformer, positions, tags)
    assert torch.equal(stock[0], corrected[0])
    assert torch.equal(stock[1], corrected[1])


def test_the_correction_scales_only_the_target_video_rows():
    """One block's processor against the stock one: the prefix rows are
    untouched and each target row is the stock attention times the head's
    gain, before the bias-free output projection."""
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3AttnProcessor,
    )

    transformer = tiny_transformer(layers=2)
    attn = transformer.transformer_blocks[1].attn  # layer weight 1 of 2
    positions, tags, prefix = packed_layout()
    rotary = transformer.rope(positions)
    hidden = torch.randn(1, len(tags), 32, generator=torch.Generator().manual_seed(4))

    state = h3_tst._TstState(TstSettings(0.5, True), total_steps=1)
    state.observe(positions, tags, torch.tensor([0.5]), torch.zeros(len(tags)))
    processor = h3_tst.TstH3AttnProcessor(
        MiniMaxH3AttnProcessor(), state, layer_index=1, total_layers=2
    )
    with torch.no_grad():
        stock = MiniMaxH3AttnProcessor()(attn, hidden, rotary)
        corrected = processor(attn, hidden, rotary)
        gains = state.last_gain
    assert gains.shape == (2,) and (gains > 1).all()
    assert torch.equal(corrected[:, :prefix], stock[:, :prefix])

    # the stock attention per head, rescaled and projected the same way
    captured = {}
    to_out = attn.to_out[0]
    handle = to_out.register_forward_pre_hook(
        lambda module, args: captured.setdefault("x", args[0])
    )
    with torch.no_grad():
        MiniMaxH3AttnProcessor()(attn, hidden, rotary)
    handle.remove()
    heads = captured["x"].unflatten(-1, (2, -1)).clone()
    heads[:, prefix:] = heads[:, prefix:] * gains[None, None, :, None]
    expected = to_out(heads.flatten(2, 3))
    torch.testing.assert_close(corrected, expected, rtol=1e-5, atol=1e-6)


def test_the_step_index_follows_the_target_timestep(flag_on):
    positions, tags, _ = packed_layout()
    state = h3_tst._TstState(TstSettings(0.2, True), total_steps=3)
    indices = torch.zeros(len(tags), dtype=torch.long)
    for t, step in ((0.9, 0), (0.9, 0), (0.5, 1), (0.5, 1), (0.1, 2), (0.05, 3)):
        state.observe(positions, tags, torch.tensor([t]), indices)
        assert state.step == step


def test_a_layout_with_no_full_video_grid_is_refused():
    positions, tags, _ = packed_layout()
    state = h3_tst._TstState(TstSettings(0.2, True), total_steps=1)
    with pytest.raises(ValueError, match="grid"):
        state.observe(
            positions[:-1], tags[:-1], torch.tensor([0.5]), torch.zeros(len(tags) - 1)
        )


def test_the_call_reports_its_gains(flag_on, monkeypatch):
    logged = []
    monkeypatch.setattr(
        h3_tst, "emit_log", lambda message, **fields: logged.append((message, fields))
    )
    transformer = tiny_transformer()
    positions, tags, _ = packed_layout()
    with torch.no_grad():
        with tst_call(TinyPipeline(transformer), TstSettings(0.5, True), 1, "clip"):
            _forward(transformer, positions, tags)
    ((message, fields),) = logged
    assert "TST" in message and fields["forwards"] == 1
    assert fields["max_gain"] >= fields["mean_gain"] >= 1.0


# 4. End to end: the real H3 graph on tiny weights


@pytest.fixture(scope="module")
def tiny_h3():
    from tests.test_chain_h3_native import _tiny_h3

    return _tiny_h3()


def _render(pipeline, steps=3):
    return pipeline(
        prompt="a cat sat",
        height=32,
        width=32,
        num_frames=124,
        num_inference_steps=steps,
        generator=torch.Generator("cpu").manual_seed(1234),
        output="latents",
    )


def test_end_to_end_tau_zero_is_the_stock_render_and_tau_half_is_not(tiny_h3, flag_on):
    transformer = tiny_h3.transformer
    before = stock_processors(transformer)
    stock = _render(tiny_h3)
    with tst_call(tiny_h3, TstSettings(0.0, True), 3, "clip"):
        zero = _render(tiny_h3)
    with tst_call(tiny_h3, TstSettings(0.5, True), 3, "clip"):
        corrected = _render(tiny_h3)
    after = _render(tiny_h3)
    assert torch.equal(stock, zero)
    assert not torch.equal(stock, corrected)
    assert torch.equal(stock, after)
    assert stock_processors(transformer) == before


def test_end_to_end_through_the_step_argument(tiny_h3, flag_on):
    """`_take_tst` (the H3 argument wrapper) takes the argument off the call
    and hands back the settings the call installs."""
    from dw.pipeline_processors.pipeline_h3 import H3Arguments

    class Step(H3Arguments):
        name = "clip"
        pipeline = tiny_h3
        configuration = {}

    arguments, settings = Step()._take_tst(
        {"num_inference_steps": 3, TST_INPUT: {"tau": 0.2}}
    )
    assert TST_INPUT not in arguments and settings == TstSettings(0.2, True)
    arguments, settings = Step()._take_tst({TST_INPUT: None})
    assert arguments == {} and settings is None
