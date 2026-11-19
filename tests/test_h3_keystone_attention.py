"""The experimental keystone attention re-split (#1391 stage A, #1393): the
re-split math against a dense softmax reference, the block and value parsing,
the flag gate and validate-time refusals, the rows check, the install and
restore on a fake transformer, and the processor against H3's stock one."""

import math

import pytest
import torch
from torch import nn

pytest.importorskip("diffusers.modular_pipelines.minimax_h3")

from dw.pipeline_processors import h3_keystone_attention as ka  # noqa: E402
from dw.pipeline_processors import h3_rules  # noqa: E402
from dw.pipeline_processors.h3_keystone_attention import (  # noqa: E402
    KEYSTONE_ATTENTION_INPUT,
    KeystoneAttentionH3Processor,
    KeystoneAttentionSettings,
    _cpu_flash,
    check_rows,
    install,
    keystone_attention_errors,
    keystone_attention_notes,
    keystone_attention_settings,
    keystone_attention_value_problem,
    parse_blocks,
    publish_rows,
    resplit,
    resplit_attention,
    uninstall,
)

FLAG = h3_rules.REFINE_WINDOWS_ENV_VAR
PATH = f"steps[0].pipeline.arguments.{KEYSTONE_ATTENTION_INPUT}"


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)


# 1. The re-split math


def _qkv(s=12, n=2, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(1, n, s, d, generator=g) for _ in range(3))


def _dense_weights(q, k):
    return (q @ k.transpose(-2, -1) * q.shape[-1] ** -0.5).softmax(-1)


ROWS_K, ROWS_C = (2, 3), (5, 4)


def test_balance_one_is_the_identity():
    q, k, v = _qkv()
    out = resplit_attention(q, k, v, ROWS_K, ROWS_C, 1.0, _cpu_flash)
    torch.testing.assert_close(out, _dense_weights(q, k) @ v, atol=1e-5, rtol=1e-5)


def test_balance_one_returns_the_main_pass_bit_for_bit():
    # #1393's bounce: balance 1 must reproduce a run without the argument
    gen = torch.Generator().manual_seed(7)
    out = torch.randn(1, 2, 4, 8, generator=gen).to(torch.bfloat16)
    out_k, out_c = torch.randn(2, 1, 2, 4, 8, generator=gen).to(torch.bfloat16)
    lse = torch.randn(1, 2, 4, generator=gen)
    lse_k = lse - torch.rand(1, 2, 4, generator=gen) - 0.1
    lse_c = lse - torch.rand(1, 2, 4, generator=gen) - 0.1
    assert torch.equal(resplit(out, lse, out_k, lse_k, out_c, lse_c, 1.0), out)
    assert not torch.equal(resplit(out, lse, out_k, lse_k, out_c, lse_c, 20.0), out)


def test_cuda_lse_ops_follow_the_sdpa_priority():
    names = lambda ops: [name for name, _ in ops]  # noqa: E731
    torch_default = [
        "FLASH_ATTENTION",
        "EFFICIENT_ATTENTION",
        "MATH",
        "CUDNN_ATTENTION",
    ]
    assert names(ka.lse_candidates("cuda", torch_default)) == ["native flash", "cuDNN"]
    cudnn_first = ["CUDNN_ATTENTION", "FLASH_ATTENTION", "MATH"]
    assert names(ka.lse_candidates("cuda", cudnn_first)) == ["cuDNN", "native flash"]
    assert names(ka.lse_candidates("cuda", ["MATH"])) == ["native flash", "cuDNN"]
    assert names(ka.lse_candidates("cuda")) and len(ka.lse_candidates("cuda")) == 2
    assert names(ka.lse_candidates("cpu")) == ["CPU flash"]
    assert ka.lse_candidates("mps") == []


def test_balance_twenty_matches_the_dense_expected_weights():
    q, k, v = _qkv(seed=1)
    balance = 20.0
    w = _dense_weights(q, k)
    (k0, kn), (c0, cn) = ROWS_K, ROWS_C
    p_k = w[..., k0 : k0 + kn].sum(-1, keepdim=True)
    p_c = w[..., c0 : c0 + cn].sum(-1, keepdim=True)
    total = p_k + p_c
    p_k_new = total * balance * p_k / (balance * p_k + p_c)
    p_c_new = total - p_k_new
    expected_w = w.clone()
    expected_w[..., k0 : k0 + kn] = w[..., k0 : k0 + kn] * p_k_new / p_k
    expected_w[..., c0 : c0 + cn] = w[..., c0 : c0 + cn] * p_c_new / p_c
    # keys outside K and C keep their weight; the total is preserved
    outside = torch.ones(w.shape[-1], dtype=torch.bool)
    outside[k0 : k0 + kn] = False
    outside[c0 : c0 + cn] = False
    assert torch.equal(expected_w[..., outside], w[..., outside])
    torch.testing.assert_close(expected_w.sum(-1), torch.ones_like(expected_w.sum(-1)))
    out = resplit_attention(q, k, v, ROWS_K, ROWS_C, balance, _cpu_flash)
    torch.testing.assert_close(out, expected_w @ v, atol=1e-5, rtol=1e-4)
    assert not torch.allclose(out, w @ v, atol=1e-3)


def test_condition_row_queries_are_left_untouched():
    # #1494: queries at or before the guide rows keep the main pass's output
    q, k, v = _qkv(seed=3)
    first = ROWS_C[0] + ROWS_C[1]
    plain = resplit_attention(q, k, v, ROWS_K, ROWS_C, 1.0, _cpu_flash)
    out = resplit_attention(
        q, k, v, ROWS_K, ROWS_C, 20.0, _cpu_flash, first_query=first
    )
    full = resplit_attention(q, k, v, ROWS_K, ROWS_C, 20.0, _cpu_flash)
    assert torch.equal(out[:, :, :first], plain[:, :, :first])
    torch.testing.assert_close(out[:, :, first:], full[:, :, first:])
    assert not torch.allclose(out[:, :, first:], plain[:, :, first:], atol=1e-4)


def test_resplit_shares_and_the_no_attention_query():
    q, k, v = _qkv(seed=2)
    scale = q.shape[-1] ** -0.5
    out, lse = _cpu_flash(q, k, v, scale)
    (k0, kn), (c0, cn) = ROWS_K, ROWS_C
    out_k, lse_k = _cpu_flash(q, k[:, :, k0 : k0 + kn], v[:, :, k0 : k0 + kn], scale)
    out_c, lse_c = _cpu_flash(q, k[:, :, c0 : c0 + cn], v[:, :, c0 : c0 + cn], scale)
    # query position 0 pays K and C nothing
    lse_k = lse_k.clone()
    lse_c = lse_c.clone()
    lse_k[..., 0] = -math.inf
    lse_c[..., 0] = -math.inf
    balance = 20.0
    result = resplit(out, lse, out_k, lse_k, out_c, lse_c, balance)
    assert torch.equal(result[:, :, 0], out[:, :, 0])
    p_k = torch.exp(lse_k - lse)
    p_c = torch.exp(lse_c - lse)
    total = p_k + p_c
    p_k_new = total * balance * p_k / (balance * p_k + p_c)
    p_c_new = total - p_k_new
    expected = (
        out + (p_k_new - p_k)[..., None] * out_k + (p_c_new - p_c)[..., None] * out_c
    )
    torch.testing.assert_close(result[:, :, 1:], expected[:, :, 1:])
    torch.testing.assert_close(p_k_new[..., 1:] + p_c_new[..., 1:], total[..., 1:])
    assert (p_k_new[..., 1:] > p_k[..., 1:]).all()
    # balance 1 changes nothing
    same = resplit(out, lse, out_k, lse_k, out_c, lse_c, 1.0)
    torch.testing.assert_close(same, out, atol=1e-6, rtol=1e-6)


# 2. parse_blocks


def test_parse_blocks_default_and_whitespace():
    assert parse_blocks(ka.DEFAULT_BLOCKS) == (31, 32, 33, *range(36, 45), 47)
    assert parse_blocks(" 1 - 3 , 5 ") == (1, 2, 3, 5)
    assert parse_blocks("0-49", 50) == tuple(range(50))


@pytest.mark.parametrize(
    "text, total",
    [
        ("", None),
        ("  ", None),
        ("a-b", None),
        ("3-1", None),
        ("1,1", None),
        ("1-3,2", None),
        ("31-50", 50),
        (None, None),
        (5, None),
        (["1"], None),
    ],
)
def test_parse_blocks_refusals(text, total):
    with pytest.raises(ValueError):
        parse_blocks(text, total)


# 3. The value


def test_value_problem_accepts_the_empty_object():
    assert keystone_attention_value_problem({}) is None
    assert keystone_attention_value_problem({"balance": 2, "blocks": "1-2"}) is None


@pytest.mark.parametrize(
    "value",
    [
        {"balance": 0},
        {"balance": -1},
        {"balance": float("nan")},
        {"balance": float("inf")},
        {"balance": True},
        {"balance": "x"},
        {"strength": 1},
        "yes",
        None,
    ],
)
def test_value_problem_refuses(value):
    assert keystone_attention_value_problem(value)


def test_value_problem_blocks_past_the_total():
    assert keystone_attention_value_problem({"blocks": "31-50"}, 50)
    assert keystone_attention_value_problem({"blocks": "31-49"}, 50) is None


def test_value_problem_leaves_unresolved_references_alone():
    assert keystone_attention_value_problem({"balance": "variable:b"}) is None
    assert keystone_attention_value_problem({"blocks": "variable:b"}, 50) is None


# 4. The settings


@pytest.mark.parametrize("value", [None, False])
def test_settings_off_or_absent(flag_off, value):
    assert keystone_attention_settings(value) is None


def test_settings_flag_off_is_refused(flag_off):
    with pytest.raises(ValueError, match=FLAG):
        keystone_attention_settings({})


def test_settings_defaults_with_the_flag_on(flag_on):
    settings = keystone_attention_settings({})
    assert settings == KeystoneAttentionSettings(20.0, "31-33,36-44,47")
    assert keystone_attention_settings({"balance": 3, "blocks": "1"}) == (
        KeystoneAttentionSettings(3.0, "1")
    )
    with pytest.raises(ValueError, match="balance"):
        keystone_attention_settings({"balance": 0})


# 5. Validate


def h3_workflow(arguments=None, configuration=None, pipeline=None, workflow="fl2va"):
    args = {
        "image": "x.png",
        "guides": [{"video": "g.mp4", "frame": 0}],
        KEYSTONE_ATTENTION_INPUT: {},
        **(arguments or {}),
    }
    return {
        "steps": [
            {
                "name": "s",
                "pipeline": {
                    "configuration": {
                        "component_type": "ModularPipeline",
                        **(configuration or {}),
                    },
                    "from_pretrained_arguments": {
                        "model_name": "MiniMaxAI/MiniMax-H3",
                        "workflow": workflow,
                    },
                    "arguments": args,
                    **(pipeline or {}),
                },
            }
        ]
    }


def _drop(key):
    workflow = h3_workflow()
    del workflow["steps"][0]["pipeline"]["arguments"][key]
    return workflow


def test_a_clean_step_with_the_flag_on(flag_on):
    assert keystone_attention_errors(h3_workflow()) == []


def test_absent_or_false_is_never_refused(flag_off):
    assert (
        keystone_attention_errors(h3_workflow({KEYSTONE_ATTENTION_INPUT: None})) == []
    )
    assert (
        keystone_attention_errors(h3_workflow({KEYSTONE_ATTENTION_INPUT: False})) == []
    )


def test_the_flag_off_is_one_finding(flag_off):
    errors = keystone_attention_errors(h3_workflow())
    assert [e["path"] for e in errors] == [PATH]
    assert FLAG in errors[0]["message"]


def _refused(workflow, match=None):
    errors = keystone_attention_errors(workflow)
    assert [e["path"] for e in errors] == [PATH], errors
    if match:
        assert match in errors[0]["message"]


def test_other_workflows_are_refused(flag_on):
    _refused(h3_workflow(workflow="t2va"), "fl2va")


def test_no_image_or_no_guides_is_refused(flag_on):
    _refused(_drop("image"), "image")
    _refused(_drop("guides"), "guides")
    _refused(h3_workflow({"guides": []}), "guides")


@pytest.mark.parametrize(
    "value, match",
    [
        ({"balance": 0}, "balance"),
        ({"balance": -1}, "balance"),
        ({"blocks": "31-50"}, "blocks"),
        ({"blocks": "a-b"}, "blocks"),
        ({"blocks": ""}, "blocks"),
    ],
)
def test_bad_values_are_refused(flag_on, monkeypatch, value, match):
    monkeypatch.setattr(ka, "declared_block_count", lambda: 50)
    _refused(h3_workflow({KEYSTONE_ATTENTION_INPUT: value}), match)


def test_tst_is_refused(flag_on):
    _refused(h3_workflow({"experimental_tst": {"tau": 0.2}}), "experimental_tst")


def test_sparse_is_refused(flag_on):
    _refused(h3_workflow(configuration={"attention_backend": "sparse"}), "sparse")


def test_a_chain_capstone_or_gated_slot_is_refused(flag_on):
    _refused(
        h3_workflow(pipeline={"chain": {"experimental_capstone": {"x": 1}}}),
        "capstone",
    )
    slot = {"frame": 34, "enter_below_sigma": 0.5}
    _refused(h3_workflow(pipeline={"chain": {"keystone_slot": slot}}), "keystone_slot")


def test_a_gated_guide_is_refused(flag_on):
    guides = [{"video": "g.mp4", "frame": 0, "enter_below_sigma": 0.5}]
    _refused(h3_workflow({"guides": guides}), "enter_below_sigma")


def test_a_step_off_h3_is_refused(flag_on):
    workflow = h3_workflow()
    workflow["steps"][0]["pipeline"]["configuration"]["component_type"] = "FluxPipeline"
    _refused(workflow, "MiniMax-H3")


def test_notes_name_the_defaults(flag_on, flag_off=None):
    (note,) = keystone_attention_notes(h3_workflow())
    assert "balance 20" in note and "31-33,36-44,47" in note


def test_notes_are_empty_with_the_flag_off(flag_off):
    assert keystone_attention_notes(h3_workflow()) == []


# 6. check_rows


def _tags(text=3, cond=7, audio=2, target=10):
    return torch.tensor([1] * text + [0] * cond + [2] * audio + [0] * target)


def _indices(tags, cond_value=1):
    indices = torch.zeros(len(tags), dtype=torch.long)
    indices[3:10] = cond_value
    return indices


def test_check_rows_passes():
    tags = _tags()
    check_rows(((3, 3), (6, 4)), tags, _indices(tags))
    check_rows(((3, 3), (6, 4)), tags, None)


@pytest.mark.parametrize(
    "rows",
    [
        None,
        ((3, 0), (3, 4)),
        ((3, 3), (6, 0)),
        ((3, 2), (6, 4)),  # C not adjacent to K
        ((3, 3), (6, 8)),  # C runs into the audio and target
        ((3, 3), (15, 4)),  # C in the target
        ((3, 3), (6, 100)),  # past the end
    ],
)
def test_check_rows_refuses(rows):
    tags = _tags()
    with pytest.raises(ValueError):
        check_rows(rows, tags, _indices(tags))


def test_check_rows_refuses_a_non_video_tag_inside():
    tags = _tags()
    tags[7] = 2
    with pytest.raises(ValueError):
        check_rows(((3, 3), (6, 4)), tags, None)


def test_check_rows_refuses_mismatched_timestep_indices():
    tags = _tags()
    indices = _indices(tags)
    indices[7] = 5
    with pytest.raises(ValueError, match="timestep"):
        check_rows(((3, 3), (6, 4)), tags, indices)


# 7. Install and uninstall


class FakeAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = object()

    def set_processor(self, processor):
        self.processor = processor


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = FakeAttn()


class FakeTransformer(nn.Module):
    def __init__(self, blocks=5):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(FakeBlock() for _ in range(blocks))

    def forward(self, *args, **kwargs):
        return None


def test_install_replaces_only_the_named_blocks_and_uninstall_restores():
    fake = FakeTransformer()
    before = [b.attn.processor for b in fake.transformer_blocks]
    state = install(fake, KeystoneAttentionSettings(20.0, "1,3"))
    assert ka.installed(fake) and state.blocks == (1, 3)
    for i, block in enumerate(fake.transformer_blocks):
        if i in (1, 3):
            assert isinstance(block.attn.processor, KeystoneAttentionH3Processor)
            assert block.attn.processor._original is before[i]
            assert block.attn.processor.layer_index == i
        else:
            assert block.attn.processor is before[i]
    assert len(fake._forward_pre_hooks) == 1
    uninstall(fake)
    assert [b.attn.processor for b in fake.transformer_blocks] == before
    assert not hasattr(fake, ka._STATE)
    assert not fake._forward_pre_hooks
    assert not ka.installed(fake)
    uninstall(fake)  # a second one is a no-op


def test_install_past_the_last_block_raises_and_is_restorable():
    fake = FakeTransformer()
    before = [b.attn.processor for b in fake.transformer_blocks]
    with pytest.raises(ValueError):
        install(fake, KeystoneAttentionSettings(20.0, "5"))
    uninstall(fake)
    assert [b.attn.processor for b in fake.transformer_blocks] == before


def test_publish_rows():
    fake = FakeTransformer()
    assert publish_rows(fake, 3, 2, 4) is None
    state = install(fake, KeystoneAttentionSettings(20.0, "1"))
    assert publish_rows(fake, 3, 2, 4) == ((3, 2), (5, 4))
    assert state.rows == ((3, 2), (5, 4))
    uninstall(fake)


def test_the_pre_hook_checks_the_rows_once_per_publish():
    fake = FakeTransformer()
    state = install(fake, KeystoneAttentionSettings(20.0, "1"))
    tags = _tags()
    with pytest.raises(ValueError, match="published no"):
        fake(token_tags=tags, timestep_indices=None)
    publish_rows(fake, 3, 3, 4)
    fake(token_tags=tags, timestep_indices=_indices(tags))
    fake(token_tags=tags, timestep_indices=_indices(tags))
    assert state.forwards == 2
    uninstall(fake)


# 8. The processor against H3's stock one


def _tiny_attention():
    from tests.test_h3_tst import packed_layout, tiny_transformer

    transformer = tiny_transformer(layers=2)
    attn = transformer.transformer_blocks[1].attn
    positions, tags, _ = packed_layout()
    rotary = transformer.rope(positions)
    hidden = torch.randn(1, len(tags), 32, generator=torch.Generator().manual_seed(4))
    return attn, rotary, hidden, len(tags)


def _state(balance, rows):
    state = ka._KeystoneAttentionState(KeystoneAttentionSettings(balance, "1"))
    state.rows = rows
    return state


def test_the_processor_matches_the_stock_one_at_balance_one_and_differs_at_twenty():
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3AttnProcessor,
    )

    attn, rotary, hidden, _ = _tiny_attention()
    rows = ((2, 3), (5, 4))
    original = MiniMaxH3AttnProcessor()
    with torch.no_grad():
        stock = MiniMaxH3AttnProcessor()(attn, hidden, rotary)
        one = KeystoneAttentionH3Processor(original, _state(1.0, rows), 0)(
            attn, hidden, rotary
        )
        twenty = KeystoneAttentionH3Processor(original, _state(20.0, rows), 0)(
            attn, hidden, rotary
        )
    torch.testing.assert_close(one, stock, atol=1e-5, rtol=1e-4)
    assert not torch.allclose(twenty, stock, atol=1e-4)


def test_the_processor_refuses_without_rows_or_with_a_mask():
    attn, rotary, hidden, _ = _tiny_attention()
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3AttnProcessor,
    )

    original = MiniMaxH3AttnProcessor()
    with pytest.raises(ValueError, match="published no"):
        KeystoneAttentionH3Processor(original, _state(20.0, None), 0)(
            attn, hidden, rotary
        )
    with pytest.raises(ValueError, match="mask"):
        KeystoneAttentionH3Processor(original, _state(20.0, ((2, 3), (5, 4))), 0)(
            attn, hidden, rotary, attention_mask=torch.ones(1)
        )
