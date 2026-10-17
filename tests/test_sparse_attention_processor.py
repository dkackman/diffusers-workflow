"""The H3 sparse processor, its installation and the per-call decision (#931).

A tiny real MiniMaxH3Transformer3DModel runs on CPU in float32. The sparse path
needs a CUDA kernel, so the tests that take it swap `sparse_unavailable_reason`
for one that says yes and `_sol_attn` for a dense stand-in over the tiled
tensors: the output then equals the stock processor's exactly when the
gather into the padded tile order and back is a true permutation.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

from dw import sparse_attention
from dw.sparse_attention import (
    DEFAULT_TAU,
    SPARSE_BACKEND,
    SparseLayoutError,
    install_sparse,
    is_sparse_installed,
    set_sparse_tau,
    sparse_call,
    sparse_unavailable_reason,
    uninstall_sparse,
)
from tests.sparse_layouts import _layout, dense_sol_attn_fake

CONFIG = dict(
    num_attention_heads=2,
    attention_head_dim=128,
    hidden_size=256,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=64,
    in_channels=4,
    audio_in_channels=4,
    patch_size=(1, 2, 2),
    text_dim=32,
    freq_dim=32,
    time_embed_hidden_dim=32,
    time_embed_dim=32,
    rope_freq_dim=16,
)


@pytest.fixture
def tiny_h3():
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )

    torch.manual_seed(0)
    return MiniMaxH3Transformer3DModel(**CONFIG).eval()


def _inputs(text=70, ref_rows=20, audio=70, t=5, h=6, w=5, batch=2):
    """Forward kwargs for one packed layout. h=6, w=5 leaves partial tiles, so
    the padding mask is exercised; batch=2 stands in for CFG."""
    tags, pos = _layout(text, ref_rows, audio, t, h, w)
    target = len(tags) - t * h * w
    timestep_indices = torch.zeros(len(tags), dtype=torch.long)
    timestep_indices[text + ref_rows :] = 1  # target audio and video are noised
    g = torch.Generator().manual_seed(1)
    video = (tags == 0).nonzero().squeeze(1)
    audio_idx = (tags == 2).nonzero().squeeze(1)
    text_idx = (tags == 1).nonzero().squeeze(1)
    assert video[-1] == len(tags) - 1 and target > 0
    patch = CONFIG["in_channels"] * 4
    return dict(
        hidden_states=torch.randn(batch, len(video), patch, generator=g),
        audio_hidden_states=torch.randn(batch, len(audio_idx), 4, generator=g),
        encoder_hidden_states=torch.randn(batch, len(text_idx), 32, generator=g),
        timestep=torch.tensor([0.0, 0.6]),
        timestep_indices=timestep_indices,
        token_tags=tags,
        position_ids=pos,
        video_indices=video,
        audio_indices=audio_idx,
        text_indices=text_idx,
    )


def _pipe(component):
    return SimpleNamespace(transformer=component)


@pytest.fixture
def kernel_calls(monkeypatch):
    """Say the kernel can run, and record each call to the dense stand-in."""
    calls = []

    def fake(q, k, v, **kw):
        calls.append(dict(dtype=q.dtype, rows=q.shape[1], **kw))
        return dense_sol_attn_fake(q, k, v, **kw)

    monkeypatch.setattr(sparse_attention, "_sol_attn", fake)
    monkeypatch.setattr(sparse_attention, "sparse_unavailable_reason", lambda *a: None)
    return calls


@pytest.fixture
def warnings(monkeypatch):
    seen = []
    monkeypatch.setattr(
        sparse_attention,
        "emit_warning",
        lambda message, **data: seen.append((message, data)),
    )
    return seen


def _refiner_processors(model):
    return [b.attn.processor for b in model.token_refiner.refiner_blocks]


def test_constants():
    assert SPARSE_BACKEND == "sparse"
    assert DEFAULT_TAU == 0.5


def test_install_swaps_only_block_processors(tiny_h3):
    refiner_before = _refiner_processors(tiny_h3)
    assert install_sparse(tiny_h3, 0.5) is True
    assert is_sparse_installed(tiny_h3)
    assert all(
        type(b.attn.processor).__name__ == "SparseH3AttnProcessor"
        for b in tiny_h3.transformer_blocks
    )
    assert _refiner_processors(tiny_h3) == refiner_before


def test_refiner_processors_untouched(tiny_h3, kernel_calls):
    refiner_before = _refiner_processors(tiny_h3)
    install_sparse(tiny_h3, 0.5)
    with sparse_call(_pipe(tiny_h3), "t"):
        tiny_h3(**_inputs())
    assert _refiner_processors(tiny_h3) == refiner_before
    assert all(type(p).__name__ == "MiniMaxH3AttnProcessor" for p in refiner_before)
    # One kernel call per block, none for the refiner
    assert len(kernel_calls) == CONFIG["num_layers"]


def test_install_on_non_h3_returns_false():
    linear = torch.nn.Linear(2, 2)
    assert install_sparse(linear, 0.5) is False
    assert not is_sparse_installed(linear)
    assert not linear._forward_pre_hooks and not linear._forward_hooks


def test_install_is_idempotent(tiny_h3, kernel_calls):
    install_sparse(tiny_h3, 0.5)
    processors = [b.attn.processor for b in tiny_h3.transformer_blocks]
    assert install_sparse(tiny_h3, 0.25) is True
    assert [b.attn.processor for b in tiny_h3.transformer_blocks] == processors
    assert len(tiny_h3._forward_pre_hooks) == 1 and len(tiny_h3._forward_hooks) == 1
    with sparse_call(_pipe(tiny_h3), "t"):
        tiny_h3(**_inputs())
    assert {c["tau"] for c in kernel_calls} == {0.25}


def test_set_sparse_tau_reaches_the_kernel(tiny_h3, kernel_calls):
    install_sparse(tiny_h3, 0.5)
    set_sparse_tau(tiny_h3, 0.8)
    with sparse_call(_pipe(tiny_h3), "t"):
        tiny_h3(**_inputs())
    assert {c["tau"] for c in kernel_calls} == {0.8}


def test_signature_unchanged(tiny_h3):
    before = inspect.signature(tiny_h3.forward)
    install_sparse(tiny_h3, 0.5)
    assert inspect.signature(tiny_h3.forward) == before
    assert "token_tags" in before.parameters and "position_ids" in before.parameters


def test_permutation_is_exact(tiny_h3, kernel_calls):
    inputs = _inputs()
    with torch.no_grad():
        ref = tiny_h3(**inputs)
    install_sparse(tiny_h3, 0.5)
    with torch.no_grad(), sparse_call(_pipe(tiny_h3), "t") as report:
        out = tiny_h3(**inputs)
    assert report.ran_sparse and report.reason is None
    assert len(kernel_calls) == CONFIG["num_layers"]
    # 70 text -> 2 tiles, 20 refs -> 1, 70 audio -> 2: the prefix is 5 tiles
    assert kernel_calls[0]["sink_blocks"] == [0, 5]
    assert kernel_calls[0]["sink_q"] == [0, 5]
    torch.testing.assert_close(out.sample, ref.sample, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out.audio_sample, ref.audio_sample, rtol=1e-4, atol=1e-4)


def test_permutation_test_is_sensitive(tiny_h3, monkeypatch):
    """The exactness check above means something only if attention moves the
    output: a kernel that drops a row's attention must not pass it."""
    inputs = _inputs()
    with torch.no_grad():
        ref = tiny_h3(**inputs).sample

    def shifted(q, k, v, **kw):
        out = dense_sol_attn_fake(q, k, v, **kw)
        return out.roll(1, dims=1)

    monkeypatch.setattr(sparse_attention, "_sol_attn", shifted)
    monkeypatch.setattr(sparse_attention, "sparse_unavailable_reason", lambda *a: None)
    install_sparse(tiny_h3, 0.5)
    with torch.no_grad(), sparse_call(_pipe(tiny_h3), "t"):
        out = tiny_h3(**inputs).sample
    assert not torch.allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_fallback_on_non_cuda_device(tiny_h3, warnings):
    inputs = _inputs()
    with torch.no_grad():
        ref_free = tiny_h3(**inputs).sample
    install_sparse(tiny_h3, 0.5)
    with torch.no_grad(), sparse_call(_pipe(tiny_h3), "step") as report:
        out = tiny_h3(**inputs).sample
        tiny_h3(**inputs)  # a second forward in the same call warns no more
    assert not report.ran_sparse and "is not CUDA" in report.reason
    assert len(warnings) == 1
    message, data = warnings[0]
    assert message == f"step: sparse attention ran dense - {report.reason}"
    assert data == {"attention_backend": "sparse", "reason": report.reason}
    torch.testing.assert_close(out, ref_free)


def test_layout_error_at_decision_falls_back(tiny_h3, kernel_calls, warnings):
    inputs = _inputs()
    # A layout whose target video is not a full grid
    inputs["position_ids"] = inputs["position_ids"].clone()
    inputs["position_ids"][-1, 2] += 0.5
    with torch.no_grad():
        ref = tiny_h3(**inputs).sample
    install_sparse(tiny_h3, 0.5)
    with torch.no_grad(), sparse_call(_pipe(tiny_h3), "step") as report:
        out = tiny_h3(**inputs).sample
    assert not report.ran_sparse and "not a full grid" in report.reason
    assert not kernel_calls and len(warnings) == 1
    torch.testing.assert_close(out, ref)


def test_failure_after_decision_raises(tiny_h3, kernel_calls, warnings):
    install_sparse(tiny_h3, 0.5)
    good = _inputs()
    # The same layout one target row short: a new shape, so the map is
    # rebuilt, and no longer a full grid
    bad = dict(good)
    for name in ("timestep_indices", "token_tags", "position_ids", "video_indices"):
        bad[name] = good[name][:-1]
    bad["hidden_states"] = good["hidden_states"][:, :-1]
    with torch.no_grad(), sparse_call(_pipe(tiny_h3), "t"):
        tiny_h3(**good)
        with pytest.raises(SparseLayoutError):
            tiny_h3(**bad)


def test_each_call_decides_afresh(tiny_h3, kernel_calls, monkeypatch, warnings):
    answers = iter(["device cpu is not CUDA", None])
    monkeypatch.setattr(
        sparse_attention, "sparse_unavailable_reason", lambda *a: next(answers)
    )
    install_sparse(tiny_h3, 0.5)
    inputs = _inputs()
    with torch.no_grad(), sparse_call(_pipe(tiny_h3), "t") as first:
        tiny_h3(**inputs)
        tiny_h3(**inputs)  # decided once: no second question
    assert not first.ran_sparse and not kernel_calls
    with torch.no_grad(), sparse_call(_pipe(tiny_h3), "t") as second:
        tiny_h3(**inputs)
    assert second.ran_sparse and len(kernel_calls) == CONFIG["num_layers"]
    assert len(warnings) == 1


def test_tile_map_rebuilt_when_layout_changes(tiny_h3, kernel_calls, monkeypatch):
    built = []
    real = sparse_attention.build_tile_map

    def counting(*a, **kw):
        built.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(sparse_attention, "build_tile_map", counting)
    small, large = _inputs(t=5, h=6, w=5), _inputs(t=3, h=8, w=9, ref_rows=0)
    with torch.no_grad():
        refs = [tiny_h3(**small).sample, tiny_h3(**large).sample]
    install_sparse(tiny_h3, 0.5)
    with torch.no_grad(), sparse_call(_pipe(tiny_h3), "t") as report:
        outs = [
            tiny_h3(**small).sample,
            tiny_h3(**small).sample,
            tiny_h3(**large).sample,
        ]
    assert report.ran_sparse
    assert len(built) == 2  # the repeated layout reuses its map
    assert {c["rows"] for c in kernel_calls} == {13 * 64, 10 * 64}
    torch.testing.assert_close(outs[0], refs[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(outs[1], refs[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(outs[2], refs[1], rtol=1e-4, atol=1e-4)


def test_tile_map_rebuilt_for_a_transposed_grid(tiny_h3, kernel_calls, monkeypatch):
    """h=6, w=5 and h=5, w=6 share row count, tag sum and position sum; only
    a rebuilt map gathers the second one correctly."""
    built = []
    real = sparse_attention.build_tile_map

    def counting(*a, **kw):
        tm = real(*a, **kw)
        built.append(tm.video_grid)
        return tm

    monkeypatch.setattr(sparse_attention, "build_tile_map", counting)
    first, second = _inputs(h=6, w=5), _inputs(h=5, w=6)
    assert first["position_ids"].shape == second["position_ids"].shape
    assert first["token_tags"].sum() == second["token_tags"].sum()
    with torch.no_grad():
        refs = [tiny_h3(**first).sample, tiny_h3(**second).sample]
    install_sparse(tiny_h3, 0.5)
    with torch.no_grad(), sparse_call(_pipe(tiny_h3), "t") as report:
        outs = [tiny_h3(**first).sample, tiny_h3(**second).sample]
    assert report.ran_sparse
    assert built == [(5, 6, 5), (5, 5, 6)]
    torch.testing.assert_close(outs[0], refs[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(outs[1], refs[1], rtol=1e-4, atol=1e-4)


def test_any_error_while_deciding_falls_back(
    tiny_h3, kernel_calls, monkeypatch, warnings
):
    def boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(sparse_attention, "build_tile_map", boom)
    inputs = _inputs()
    with torch.no_grad():
        ref = tiny_h3(**inputs).sample
    install_sparse(tiny_h3, 0.5)
    with torch.no_grad(), sparse_call(_pipe(tiny_h3), "t") as report:
        out = tiny_h3(**inputs).sample
    assert not report.ran_sparse and "RuntimeError: boom" in report.reason
    assert not kernel_calls and len(warnings) == 1
    torch.testing.assert_close(out, ref)


def test_uninstall_restores_original_processors(tiny_h3):
    originals = [b.attn.processor for b in tiny_h3.transformer_blocks]
    install_sparse(tiny_h3, 0.5)
    uninstall_sparse(tiny_h3)
    assert [b.attn.processor for b in tiny_h3.transformer_blocks] == originals
    assert not tiny_h3._forward_pre_hooks and not tiny_h3._forward_hooks
    assert not is_sparse_installed(tiny_h3)
    uninstall_sparse(tiny_h3)  # a no-op the second time
    uninstall_sparse(torch.nn.Linear(2, 2))


def test_backend_setting_reaches_the_original_processor(tiny_h3):
    """diffusers' set_attention_backend writes `_attention_backend` onto each
    processor; the original, which the dense path runs, must see it."""
    install_sparse(tiny_h3, 0.5)
    block = tiny_h3.transformer_blocks[0]
    block.attn.processor._attention_backend = "native"
    uninstall_sparse(tiny_h3)
    assert block.attn.processor._attention_backend == "native"


def test_active_backend_unchanged(tiny_h3):
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    before = _AttentionBackendRegistry._active_backend
    install_sparse(tiny_h3, 0.5)
    uninstall_sparse(tiny_h3)
    assert _AttentionBackendRegistry._active_backend == before


def test_sparse_call_reads_a_components_mapping(tiny_h3, kernel_calls):
    install_sparse(tiny_h3, 0.5)
    pipeline = SimpleNamespace(components={"transformer": tiny_h3, "vae": None})
    with torch.no_grad(), sparse_call(pipeline, "t") as report:
        tiny_h3(**_inputs())
    assert report.ran_sparse


def test_sparse_call_with_nothing_installed(tiny_h3, warnings):
    with sparse_call(_pipe(tiny_h3), "t") as report:
        pass
    assert not report.ran_sparse and report.reason is None and not warnings


def test_unavailable_reason_off_cuda():
    assert sparse_unavailable_reason(torch.device("cpu"), torch.bfloat16, 128) == (
        "device cpu is not CUDA"
    )
    assert (
        sparse_unavailable_reason("mps", torch.bfloat16, 128)
        == "device mps is not CUDA"
    )


def test_unavailable_reason_on_cuda(monkeypatch):
    import comfy_kitchen

    cuda = torch.device("cuda", 0)
    capability = [(8, 0)]
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability[0]
    )
    monkeypatch.setattr(
        comfy_kitchen, "sol_attn_is_available", lambda device=None: True
    )
    assert sparse_unavailable_reason(cuda, torch.bfloat16, 128) is None
    assert sparse_unavailable_reason(cuda, torch.float16, 128) is None
    assert sparse_unavailable_reason(cuda, torch.float32, 128) == (
        "dtype float32 is not bfloat16/float16"
    )
    assert (
        sparse_unavailable_reason(cuda, torch.bfloat16, 64) == "head_dim 64 is not 128"
    )

    def broken(device=None):
        raise RuntimeError("no sm_80 build")

    monkeypatch.setattr(comfy_kitchen, "sol_attn_is_available", broken)
    assert sparse_unavailable_reason(cuda, torch.bfloat16, 128) == (
        "comfy-kitchen kernels unavailable: no sm_80 build"
    )
    capability[0] = (7, 5)
    assert sparse_unavailable_reason(cuda, torch.bfloat16, 128) == (
        "compute capability 7.5 is below 8.0"
    )
