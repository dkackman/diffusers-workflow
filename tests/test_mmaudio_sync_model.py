"""The vendored Synchformer visual encoder (dw/vendored/mmaudio_sync_model.py)."""

import re
from pathlib import Path

import pytest
import torch

from dw.vendored import mmaudio_sync_model as sm

MODULE_PATH = Path(sm.__file__)
DW = MODULE_PATH.parents[1]  # dw/, with the port under dw/vendored/
TINY = sm.SyncConfig(
    image_size=32,
    patch_size=16,
    patch_size_temp=2,
    embed_dim=32,
    depth=2,
    num_heads=4,
    mlp_ratio=2,
    temporal_resolution=8,
    segment_frames=16,
)
GRID = 2  # patches per side of a tiny frame
T_OUT = TINY.temporal_resolution


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return sm.Synchformer(TINY).eval()


# --- shapes ----------------------------------------------------------------------


def test_forward_feature_shape(tiny_model):
    vis = torch.randn(2, 3, 16, 3, 32, 32)
    with torch.no_grad():
        out = tiny_model(vis)
    assert out.shape == (2, 3, 16 // TINY.patch_size_temp, TINY.embed_dim)
    assert torch.isfinite(out).all()
    # segments are independent
    with torch.no_grad():
        single = tiny_model(vis[:, 1:2])
    torch.testing.assert_close(out[:, 1:2], single, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("batch_size", [-1, 1, 2])
def test_encode_sync_frames_segments(tiny_model, batch_size):
    frames = torch.randn(2, 32, 3, 32, 32)
    out = sm.encode_sync_frames(tiny_model, frames, batch_size=batch_size)
    num_segments = (32 - 16) // 8 + 1
    assert out.shape == (2, num_segments * T_OUT, TINY.embed_dim)
    # the first segment's features come first, in order
    first = tiny_model(frames[:, :16].unsqueeze(1))[:, 0]
    torch.testing.assert_close(out[:, :T_OUT], first, atol=1e-5, rtol=1e-4)
    with pytest.raises(ValueError):
        sm.encode_sync_frames(tiny_model, frames[:, :8])
    with pytest.raises(ValueError):
        sm.encode_sync_frames(tiny_model, torch.randn(1, 16, 3, 64, 64))


# --- the default configuration and the checkpoint's keys -------------------------


def test_default_config_matches_upstream():
    cfg = sm.SyncConfig()
    assert (cfg.embed_dim, cfg.depth, cfg.num_heads, cfg.mlp_ratio) == (768, 12, 12, 4)
    assert (cfg.patch_size, cfg.patch_size_temp, cfg.image_size) == (16, 2, 224)
    assert (cfg.segment_frames, cfg.temporal_resolution, cfg.in_chans) == (16, 8, 3)
    assert cfg.qkv_bias is True


def _expected_keys(depth):
    """Upstream's state-dict keys, written out from its attribute names."""
    keys = {"vfeat_extractor.cls_token", "vfeat_extractor.pos_embed"}
    keys.add("vfeat_extractor.temp_embed")
    for name in ("patch_embed", "patch_embed_3d"):
        keys |= {f"vfeat_extractor.{name}.proj.{p}" for p in ("weight", "bias")}
    keys |= {f"vfeat_extractor.norm.{p}" for p in ("weight", "bias")}
    for i in range(depth):
        b = f"vfeat_extractor.blocks.{i}"
        for norm in ("norm1", "norm2", "norm3"):
            keys |= {f"{b}.{norm}.{p}" for p in ("weight", "bias")}
        for attn in ("attn", "timeattn"):
            for lin in ("qkv", "proj"):
                keys |= {f"{b}.{attn}.{lin}.{p}" for p in ("weight", "bias")}
        for fc in ("fc1", "fc2"):
            keys |= {f"{b}.mlp.{fc}.{p}" for p in ("weight", "bias")}
    s = "vfeat_extractor.spatial_attn_agg"
    keys.add(f"{s}.cls_token")
    keys |= {f"{s}.self_attn.in_proj_{p}" for p in ("weight", "bias")}
    keys |= {f"{s}.self_attn.out_proj.{p}" for p in ("weight", "bias")}
    for name in ("linear1", "linear2", "norm1", "norm2"):
        keys |= {f"{s}.{name}.{p}" for p in ("weight", "bias")}
    return keys


def test_default_model_state_dict_keys():
    with torch.device("meta"):
        model = sm.Synchformer()
    keys = set(model.state_dict())
    assert all(k.startswith("vfeat_extractor.") for k in keys)
    assert keys == _expected_keys(12)
    sd = model.state_dict()
    assert sd["vfeat_extractor.patch_embed_3d.proj.weight"].shape == (768, 3, 2, 16, 16)
    assert sd["vfeat_extractor.patch_embed.proj.weight"].shape == (768, 3, 16, 16)
    assert sd["vfeat_extractor.blocks.0.attn.qkv.weight"].shape == (2304, 768)
    assert sd["vfeat_extractor.blocks.0.attn.qkv.bias"].shape == (2304,)
    assert sd["vfeat_extractor.pos_embed"].shape == (1, 197, 768)
    assert sd["vfeat_extractor.temp_embed"].shape == (1, 8, 768)
    assert sd["vfeat_extractor.spatial_attn_agg.linear1.weight"].shape == (3072, 768)
    # the head, pre-logits and temporal aggregation own no keys
    assert not any(".head" in k or "pre_logits" in k or "temp_attn" in k for k in keys)


# --- loading ---------------------------------------------------------------------


def test_load_synchformer_reads_weights_only_and_strictly(
    tiny_model, tmp_path, monkeypatch
):
    state = dict(tiny_model.state_dict())
    state["afeat_extractor.x"] = torch.zeros(1)
    path = tmp_path / "sync.pth"
    torch.save(state, path)

    calls = []
    real_load = torch.load

    def spy(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    model = sm.load_synchformer(path, config=TINY)
    assert calls == [True]
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
    for key, value in tiny_model.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value)
    vis = torch.randn(1, 1, 16, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(model(vis), tiny_model(vis), atol=1e-5, rtol=1e-4)

    # strict: a missing key raises
    del state["vfeat_extractor.norm.weight"]
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="norm.weight"):
        sm.load_synchformer(path, config=TINY)


# --- the module's rules ----------------------------------------------------------


def test_module_has_no_hub_io_or_forbidden_imports():
    source = MODULE_PATH.read_text(encoding="utf-8")
    for name in (
        "from_pretrained",
        "hf_hub_download",
        "cached_file",
        "snapshot_download",
    ):
        assert name not in source, name
    forbidden = (
        r"open_clip|timm|einops|omegaconf|hydra|tensordict|torchdiffeq|torchaudio"
    )
    assert not re.search(rf"^\s*(import|from)\s+({forbidden})\b", source, re.M)
    assert "weights_only=True" in source
    assert "weights_only=False" not in source
    assert "save_file" not in source and "torch.save" not in source
    assert not re.search(r"(?<![.\w])(eval|exec)\(", source)
    assert "974010a026c731054592d8f777218bd9d85a6c24" in source
    assert "MIT License" in source and "Vladimir Iashin" in source
    assert len(source.splitlines()) <= 1100


def test_weights_are_pinned():
    assert sm.REPO_ID == "hkchengrex/MMAudio"
    assert sm.REVISION == "eb13a1a98fdbec91753775c57b074ccdfc60587c"
    assert sm.SYNC_FILE == "ext_weights/synchformer_state_dict.pth"


def test_module_is_imported_only_by_the_sound_effects_task():
    # stage D (#1133): generate_sound_effects is its one consumer
    own = {
        MODULE_PATH,
        DW / "tasks" / "sound_effects.py",
        DW / "vendored" / "__init__.py",
    }
    for path in DW.rglob("*.py"):
        if path not in own:
            text = path.read_text(encoding="utf-8")
            assert "mmaudio_sync_model" not in text, path
