"""The vendored MMAudio network and Euler sampler (dw/tasks/mmaudio_model.py)
and its DFN5B CLIP key map and towers (dw/tasks/mmaudio_clip.py), on tiny random configurations on CPU."""

import math
import re
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import dw.tasks.mmaudio_clip as mc
import dw.tasks.mmaudio_model as mm

MODULE_PATH = Path(mm.__file__)
CLIP_MODULE_PATH = Path(mc.__file__)

TINY_NET = mm.MMAudioConfig(
    latent_dim=4,
    clip_dim=8,
    sync_dim=6,
    text_dim=8,
    hidden_dim=32,
    depth=3,
    fused_depth=1,
    num_heads=2,
    mlp_ratio=2.0,
    text_seq_len=5,
)
# one second: 44 latent frames, 8 CLIP frames, 16 sync frames
DURATION = 1.0

TINY_CLIP = mc.ClipConfig(
    embed_dim=16,
    image_size=28,
    patch_size=14,
    vision_width=32,
    vision_layers=2,
    vision_heads=4,
    vision_mlp=64,
    context_length=8,
    vocab_size=50,
    text_width=24,
    text_layers=2,
    text_heads=4,
    text_mlp=48,
    sot_token_id=48,
    eot_token_id=49,
)


def _randomize(module, seed=0):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.2)
        if hasattr(module, "latent_std"):
            module.latent_std.abs_().add_(0.5)
    return module


@pytest.fixture
def tiny_net():
    return _randomize(mm.MMAudioNetwork(TINY_NET, duration=DURATION)).eval()


def _features(net, bs, seed=1):
    gen = torch.Generator().manual_seed(seed)
    clip = torch.randn(bs, net.clip_seq_len, TINY_NET.clip_dim, generator=gen)
    sync = torch.randn(bs, net.sync_seq_len, TINY_NET.sync_dim, generator=gen)
    text = torch.randn(bs, TINY_NET.text_seq_len, TINY_NET.text_dim, generator=gen)
    return clip, sync, text


# --- sequence lengths and the released configuration -------------------------


def test_sequence_lengths_match_upstream():
    seq = mm.SequenceConfig(duration=8.0)
    assert (seq.latent_seq_len, seq.clip_seq_len, seq.sync_seq_len) == (345, 64, 192)
    assert seq.num_audio_frames == 345 * 1024
    tiny = mm.SequenceConfig(duration=DURATION)
    assert (tiny.latent_seq_len, tiny.clip_seq_len, tiny.sync_seq_len) == (44, 8, 16)


def test_default_config_is_large_44k_v2():
    with torch.device("meta"):
        net = mm.MMAudioNetwork()
    assert (net.latent_seq_len, net.clip_seq_len, net.sync_seq_len) == (345, 64, 192)
    assert len(net.joint_blocks) == 7 and len(net.fused_blocks) == 14
    assert net.joint_blocks[-1].pre_only and not net.joint_blocks[0].pre_only
    shapes = {k: tuple(v.shape) for k, v in net.state_dict().items()}
    assert shapes["audio_input_proj.0.weight"] == (896, 40, 7)
    assert shapes["clip_input_proj.0.weight"] == (896, 1024)
    assert shapes["sync_input_proj.0.weight"] == (896, 768, 7)
    assert shapes["text_input_proj.0.weight"] == (896, 1024)
    assert shapes["sync_pos_emb"] == (1, 1, 8, 768)
    assert shapes["empty_string_feat"] == (77, 1024)
    assert shapes["t_embed.mlp.0.weight"] == (896, 896)
    assert shapes["final_layer.conv.weight"] == (40, 896, 7)
    # SwiGLU width: 256 * ceil(int(2 * 3584 / 3) / 256)
    assert shapes["global_cond_mlp.w1.weight"] == (2560, 896)
    for key in mm.NON_PERSISTENT_KEYS:
        assert key not in shapes


def test_v2_timestep_frequencies():
    net = mm.MMAudioNetwork(TINY_NET, duration=DURATION)
    size = TINY_NET.hidden_dim
    expected = 10000 / (10000 ** (torch.arange(0, size, 2).float() / size))
    torch.testing.assert_close(net.t_embed.freqs, expected)


# --- the network samples ---------------------------------------------------


def test_tiny_network_samples_a_latent_of_the_right_shape(tiny_net):
    clip, sync, text = _features(tiny_net, bs=2)
    gen = torch.Generator().manual_seed(7)
    latent = mm.sample_latent(
        tiny_net,
        clip_features=clip,
        sync_features=sync,
        text_features=text,
        steps=4,
        cfg_strength=4.5,
        generator=gen,
    )
    assert latent.shape == (2, 44, 4)
    assert torch.isfinite(latent).all()
    again = mm.sample_latent(
        tiny_net,
        clip_features=clip,
        sync_features=sync,
        text_features=text,
        steps=4,
        cfg_strength=4.5,
        generator=torch.Generator().manual_seed(7),
    )
    torch.testing.assert_close(latent, again)


def test_text_only_and_unconditioned_sampling(tiny_net):
    _, _, text = _features(tiny_net, bs=1)
    negative = torch.randn_like(text)
    latent = mm.sample_latent(
        tiny_net, text_features=text, negative_text_features=negative, steps=2
    )
    assert latent.shape == (1, 44, 4)
    empty = mm.sample_latent(tiny_net, steps=2, batch_size=3)
    assert empty.shape == (3, 44, 4)
    assert torch.isfinite(empty).all()


def test_sampling_is_the_euler_loop_over_guided_flow(tiny_net):
    clip, sync, text = _features(tiny_net, bs=1)
    gen = torch.Generator().manual_seed(3)
    latent = mm.sample_latent(
        tiny_net,
        clip_features=clip,
        sync_features=sync,
        text_features=text,
        steps=3,
        cfg_strength=2.0,
        generator=gen,
    )
    x = torch.randn(1, 44, 4, generator=torch.Generator().manual_seed(3))
    cond = tiny_net.preprocess_conditions(clip, sync, text)
    empty = tiny_net.get_empty_conditions(1)
    with torch.no_grad():
        times = torch.linspace(0, 1, 4)
        for i in range(3):
            t = times[i].expand(1)
            c = tiny_net.predict_flow(x, t, cond)
            e = tiny_net.predict_flow(x, t, empty)
            x = x + (times[i + 1] - times[i]) * (2.0 * c + (1 - 2.0) * e)
    torch.testing.assert_close(latent, tiny_net.unnormalize(x), rtol=1e-4, atol=1e-5)


def test_guidance_combines_conditional_and_empty_flows(tiny_net):
    clip, sync, text = _features(tiny_net, bs=2)
    x = torch.randn(2, 44, 4)
    t = torch.tensor(0.25)
    with torch.no_grad():
        cond = tiny_net.preprocess_conditions(clip, sync, text)
        empty = tiny_net.get_empty_conditions(2)
        tt = torch.full((2,), 0.25)
        c = tiny_net.predict_flow(x, tt, cond)
        e = tiny_net.predict_flow(x, tt, empty)
        torch.testing.assert_close(tiny_net.guided_flow(t, x, cond, empty, 0.5), c)
        torch.testing.assert_close(
            tiny_net.guided_flow(t, x, cond, empty, 3.0),
            3.0 * c - 2.0 * e,
            rtol=1e-5,
            atol=1e-5,
        )
    assert not torch.allclose(c, e)


def test_preprocess_rejects_wrong_lengths(tiny_net):
    clip, sync, text = _features(tiny_net, bs=1)
    with pytest.raises(ValueError, match="clip"):
        tiny_net.preprocess_conditions(clip[:, :-1], sync, text)
    with pytest.raises(ValueError, match="sync"):
        tiny_net.preprocess_conditions(clip, sync[:, :-8], text)


def test_set_duration_changes_the_lengths(tiny_net):
    seq = tiny_net.set_duration(2.0)
    assert tiny_net.latent_seq_len == seq.latent_seq_len == 87
    assert tiny_net.latent_rot.shape[1] == 87 and tiny_net.clip_rot.shape[1] == 16
    assert tiny_net.latent_rot.dtype == torch.float32
    assert mm.sample_latent(tiny_net, steps=1).shape == (1, 87, 4)


# --- the Euler loop ------------------------------------------------------------


@pytest.mark.parametrize("steps", [1, 4, 25])
def test_euler_matches_the_closed_form_linear_ode(steps):
    # dx/dt = x: Euler with N equal steps gives x0 * (1 + 1/N)^N exactly
    x0 = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    seen = []

    def flow(t, x):
        seen.append(float(t))
        return x

    out = mm.euler_sample(flow, x0, steps)
    torch.testing.assert_close(out, x0 * (1 + 1 / steps) ** steps)
    assert seen == pytest.approx([i / steps for i in range(steps)])
    if steps == 25:
        # and converges toward e^1
        assert abs(out[0].item() - math.e) < 0.06


def test_euler_lands_a_straight_flow_exactly():
    # the rectified-flow ideal: a constant velocity reaches x0 + v at t = 1
    x0 = torch.randn(2, 5)
    v = torch.randn(2, 5)
    torch.testing.assert_close(mm.euler_sample(lambda t, x: v, x0, 7), x0 + v)
    # dx/dt = 2t integrates to 1 only in the limit; Euler under-shoots by 1/N
    out = mm.euler_sample(lambda t, x: 2 * t * torch.ones_like(x), torch.zeros(1), 10)
    torch.testing.assert_close(out, torch.tensor([1 - 1 / 10]))
    with pytest.raises(ValueError):
        mm.euler_sample(lambda t, x: x, x0, 0)


# --- loading the network ---------------------------------------------------------


def test_load_network_reads_weights_only_and_strictly(tiny_net, tmp_path, monkeypatch):
    state = tiny_net.state_dict()
    # the released checkpoint carries the buffers upstream drops on load
    state["t_embed.freqs"] = torch.zeros(1)
    state["latent_rot"] = torch.zeros(1)
    state["clip_rot"] = torch.zeros(1)
    path = tmp_path / "net.pth"
    torch.save(state, path)

    calls = []
    real_load = torch.load

    def spy(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    net = mm.load_network(path, config=TINY_NET, duration=DURATION)
    assert calls == [True]
    assert not net.training
    assert not any(p.requires_grad for p in net.parameters())
    for key, value in tiny_net.state_dict().items():
        torch.testing.assert_close(net.state_dict()[key], value)
    torch.testing.assert_close(net.t_embed.freqs, tiny_net.t_embed.freqs)
    torch.testing.assert_close(net.latent_rot, tiny_net.latent_rot)

    clip, sync, text = _features(net, bs=1)
    x, t = torch.randn(1, 44, 4), torch.tensor([0.5])
    with torch.no_grad():
        torch.testing.assert_close(
            net(x, clip, sync, text, t), tiny_net(x, clip, sync, text, t)
        )

    del state["sync_pos_emb"]
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="sync_pos_emb"):
        mm.load_network(path, config=TINY_NET, duration=DURATION)


def test_load_network_in_bfloat16_keeps_rotations_float32(tiny_net, tmp_path):
    path = tmp_path / "net.pth"
    torch.save(tiny_net.state_dict(), path)
    net = mm.load_network(
        path, config=TINY_NET, duration=DURATION, dtype=torch.bfloat16
    )
    assert net.dtype == torch.bfloat16
    assert net.latent_rot.dtype == torch.float32
    assert mm.sample_latent(net, steps=2).dtype == torch.bfloat16


# --- DFN5B CLIP: key map and towers ---------------------------------------------


def _block_state(prefix, width, mlp, gen):
    def r(*shape):
        return torch.randn(*shape, generator=gen) * 0.2

    return {
        f"{prefix}ln_1.weight": 1 + r(width),
        f"{prefix}ln_1.bias": r(width),
        f"{prefix}attn.in_proj_weight": r(3 * width, width),
        f"{prefix}attn.in_proj_bias": r(3 * width),
        f"{prefix}attn.out_proj.weight": r(width, width),
        f"{prefix}attn.out_proj.bias": r(width),
        f"{prefix}ln_2.weight": 1 + r(width),
        f"{prefix}ln_2.bias": r(width),
        f"{prefix}mlp.c_fc.weight": r(mlp, width),
        f"{prefix}mlp.c_fc.bias": r(mlp),
        f"{prefix}mlp.c_proj.weight": r(width, mlp),
        f"{prefix}mlp.c_proj.bias": r(width),
    }


def tiny_open_clip_state(c=TINY_CLIP, seed=0):
    """An open_clip CLIP state dict at the tiny sizes, random."""
    gen = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=gen) * 0.2

    vw, tw = c.vision_width, c.text_width
    grid = c.image_size // c.patch_size
    state = {
        "visual.class_embedding": r(vw),
        "visual.positional_embedding": r(grid * grid + 1, vw),
        "visual.proj": r(vw, c.embed_dim),
        "visual.conv1.weight": r(vw, 3, c.patch_size, c.patch_size),
        "visual.ln_pre.weight": 1 + r(vw),
        "visual.ln_pre.bias": r(vw),
        "visual.ln_post.weight": 1 + r(vw),
        "visual.ln_post.bias": r(vw),
        "positional_embedding": r(c.context_length, tw),
        "text_projection": r(tw, c.embed_dim),
        "logit_scale": torch.tensor(4.6),
        "token_embedding.weight": r(c.vocab_size, tw),
        "ln_final.weight": 1 + r(tw),
        "ln_final.bias": r(tw),
    }
    for i in range(c.vision_layers):
        state.update(
            _block_state(f"visual.transformer.resblocks.{i}.", vw, c.vision_mlp, gen)
        )
    for i in range(c.text_layers):
        state.update(_block_state(f"transformer.resblocks.{i}.", tw, c.text_mlp, gen))
    return state


def _ref_block(x, s, p, heads, mask):
    """open_clip's ResidualAttentionBlock, in plain torch (quick_gelu)."""
    h = F.layer_norm(x, x.shape[-1:], s[p + "ln_1.weight"], s[p + "ln_1.bias"], 1e-5)
    q, k, v = F.linear(
        h, s[p + "attn.in_proj_weight"], s[p + "attn.in_proj_bias"]
    ).chunk(3, -1)
    b, n, w = q.shape

    def heads_of(t):
        return t.view(b, n, heads, w // heads).transpose(1, 2)

    a = F.scaled_dot_product_attention(
        heads_of(q), heads_of(k), heads_of(v), attn_mask=mask
    )
    a = a.transpose(1, 2).reshape(b, n, w)
    x = x + F.linear(a, s[p + "attn.out_proj.weight"], s[p + "attn.out_proj.bias"])
    h = F.layer_norm(x, x.shape[-1:], s[p + "ln_2.weight"], s[p + "ln_2.bias"], 1e-5)
    h = F.linear(h, s[p + "mlp.c_fc.weight"], s[p + "mlp.c_fc.bias"])
    h = h * torch.sigmoid(1.702 * h)
    return x + F.linear(h, s[p + "mlp.c_proj.weight"], s[p + "mlp.c_proj.bias"])


def ref_encode_text(s, tokens, c=TINY_CLIP):
    """upstream patch_clip's encode_text: normalised ln_final states."""
    x = s["token_embedding.weight"][tokens] + s["positional_embedding"]
    n = tokens.shape[1]
    mask = torch.full((n, n), float("-inf")).triu(1)
    for i in range(c.text_layers):
        x = _ref_block(x, s, f"transformer.resblocks.{i}.", c.text_heads, mask)
    x = F.layer_norm(x, x.shape[-1:], s["ln_final.weight"], s["ln_final.bias"], 1e-5)
    return F.normalize(x, dim=-1)


def ref_encode_image(s, images, c=TINY_CLIP):
    """open_clip encode_image(normalize=True) on already-normalised pixels."""
    x = F.conv2d(images, s["visual.conv1.weight"], stride=c.patch_size)
    x = x.flatten(2).transpose(1, 2)
    cls = s["visual.class_embedding"].expand(x.shape[0], 1, -1)
    x = torch.cat([cls, x], dim=1) + s["visual.positional_embedding"]
    w = x.shape[-1]
    x = F.layer_norm(x, (w,), s["visual.ln_pre.weight"], s["visual.ln_pre.bias"], 1e-5)
    for i in range(c.vision_layers):
        x = _ref_block(x, s, f"visual.transformer.resblocks.{i}.", c.vision_heads, None)
    x = F.layer_norm(
        x[:, 0], (w,), s["visual.ln_post.weight"], s["visual.ln_post.bias"], 1e-5
    )
    return F.normalize(x @ s["visual.proj"], dim=-1)


def test_clip_key_map_round_trips():
    state = tiny_open_clip_state()
    vision, text = mc.open_clip_to_transformers(state)
    back = mc.transformers_to_open_clip(vision, text)
    expected = {k: v for k, v in state.items() if k not in mc.UNUSED_CLIP_KEYS}
    assert set(back) == set(expected)
    for key, value in expected.items():
        torch.testing.assert_close(back[key], value, msg=key)


def test_clip_key_map_fills_the_transformers_towers_exactly():
    from transformers import CLIPTextModel, CLIPVisionModelWithProjection

    vision, text = mc.open_clip_to_transformers(tiny_open_clip_state())
    v_model = CLIPVisionModelWithProjection(mc.clip_vision_config(TINY_CLIP))
    t_model = CLIPTextModel(mc.clip_text_config(TINY_CLIP))
    mc._load_into(v_model, vision)  # strict: no key missing or left over
    mc._load_into(t_model, text)
    with pytest.raises(KeyError, match="unknown"):
        mc.open_clip_to_transformers({"unknown.weight": torch.zeros(1)})


def test_load_clip_reproduces_open_clip(tmp_path, monkeypatch):
    state = tiny_open_clip_state()
    path = tmp_path / "open_clip_pytorch_model.bin"
    torch.save(state, path)

    calls = []
    real_load = torch.load

    def spy(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    clip = mc.load_clip(path, config=TINY_CLIP)
    assert calls == [True]
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]  # nothing written

    # the text path: per-token, normalised, as patch_clip returns it
    tokens = torch.tensor([[48, 5, 7, 49, 0, 0, 0, 0], [48, 9, 49, 0, 0, 0, 0, 0]])
    got = clip.encode_text(tokens)
    assert got.shape == (2, TINY_CLIP.context_length, TINY_CLIP.text_width)
    torch.testing.assert_close(got.norm(dim=-1), torch.ones(2, 8))
    torch.testing.assert_close(
        got, ref_encode_text(state, tokens), rtol=1e-4, atol=1e-5
    )

    # the image path: [0, 1] frames, CLIP-normalised, one embedding per frame
    frames = torch.rand(1, 3, 3, 28, 28, generator=torch.Generator().manual_seed(2))
    mean = torch.tensor(mc.CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(mc.CLIP_STD).view(1, 3, 1, 1)
    want = ref_encode_image(state, (frames[0] - mean) / std).view(1, 3, -1)
    got = clip.encode_frames(frames, batch_size=2)
    assert got.shape == (1, 3, TINY_CLIP.embed_dim)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError, match="CLIP frames"):
        clip.encode_frames(torch.rand(1, 1, 3, 32, 32))


def test_default_clip_config_is_dfn5b():
    c = mc.ClipConfig()
    v = mc.clip_vision_config(c)
    t = mc.clip_text_config(c)
    assert (v.hidden_size, v.num_hidden_layers, v.num_attention_heads) == (1280, 32, 16)
    assert (v.image_size, v.patch_size, v.projection_dim) == (384, 14, 1024)
    assert (t.hidden_size, t.num_hidden_layers, t.vocab_size) == (1024, 24, 49408)
    assert t.max_position_embeddings == 77
    assert v.hidden_act == t.hidden_act == "quick_gelu"
    # 384 / 14 -> a 27 x 27 grid, as upstream feeds it
    assert (v.image_size // v.patch_size) ** 2 + 1 == 730


def test_tokenize_uses_open_clip_layout(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(
        models.WordLevel({"[UNK]": 1, "a": 2, "cat": 3, "&": 4}, unk_token="[UNK]")
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path / "tokenizer.json"
    tok.save(str(path))
    tokenizer = mc.load_clip_tokenizer(path)

    out = mc.tokenize(
        tokenizer, ["A  Cat", "a &amp; cat cat cat cat cat cat"], TINY_CLIP
    )
    assert out.dtype == torch.long and out.shape == (2, 8)
    assert out[0].tolist() == [48, 2, 3, 49, 0, 0, 0, 0]
    # truncated to the context length, the end token kept last
    assert out[1].tolist() == [48, 2, 4, 3, 3, 3, 3, 49]


# --- the module's rules --------------------------------------------------------


@pytest.mark.parametrize("path", [MODULE_PATH, CLIP_MODULE_PATH], ids=lambda p: p.name)
def test_modules_have_no_hub_io_or_forbidden_imports(path):
    source = path.read_text(encoding="utf-8")
    for name in (
        "from_pretrained",
        "hf_hub_download",
        "cached_file",
        "snapshot_download",
    ):
        assert name not in source, name
    forbidden = r"open_clip|timm|einops|hydra|tensordict|torchdiffeq|torchaudio"
    assert not re.search(rf"^\s*(import|from)\s+({forbidden})\b", source, re.M)
    assert "weights_only=True" in source
    assert "weights_only=False" not in source
    # converted DFN5B tensors are never written
    assert "save_file" not in source and "torch.save" not in source
    assert "974010a026c731054592d8f777218bd9d85a6c24" in source
    assert "MIT License" in source and "Sony Research Inc." in source
    assert len(source.splitlines()) <= 1100


def test_weights_are_pinned():
    assert mm.REPO_ID == "hkchengrex/MMAudio"
    assert mm.REVISION == "eb13a1a98fdbec91753775c57b074ccdfc60587c"
    assert mm.NETWORK_FILE == "weights/mmaudio_large_44k_v2.pth"
    assert mc.CLIP_REPO_ID == "apple/DFN5B-CLIP-ViT-H-14-384"
    assert mc.CLIP_REVISION == "01b771ed0d1395ca5ffdd279897d665ebe00dfd2"


def test_modules_are_not_imported_by_dw():
    # stage B is inert: nothing in dw imports either module
    own = {MODULE_PATH, CLIP_MODULE_PATH}
    for path in MODULE_PATH.parents[1].rglob("*.py"):
        if path not in own:
            text = path.read_text(encoding="utf-8")
            assert "mmaudio_model" not in text and "mmaudio_clip" not in text, path
