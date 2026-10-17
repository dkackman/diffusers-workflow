"""Shared helpers for the H3 sparse-attention tests (#931): packed layouts the
way diffusers builds them, and a dense stand-in for the comfy-kitchen kernel."""

import numpy as np
import torch
import torch.nn.functional as F

# A tiny MiniMaxH3Transformer3DModel: head_dim 128, the one the kernel takes,
# and two blocks, so a per-call decision is seen across more than one
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


def _layout(text, ref_rows, audio, t, h, w):
    """token_tags and fractional position_ids packed the way diffusers does:
    text (tag 1), reference-image rows (tag 0), target audio (tag 2), then the
    target video (tag 0) last."""
    tags = [1] * text + [0] * ref_rows + [2] * audio + [0] * (t * h * w)
    pos = np.zeros((len(tags), 3), dtype=np.float64)
    # Rows before the target video carry positions the tile map never reads
    pos[:, 0] = np.arange(len(tags))

    def spatial(n):
        ratio = n / np.sqrt(h * w)
        left = (1 - ratio) / 2
        return np.linspace(left, left + ratio, n, endpoint=False) * 32

    spans = 5 / 3 * np.array([1] + [4] * (t - 1))
    times = text + np.cumsum(spans)
    grid = np.stack(np.meshgrid(times, spatial(h), spatial(w), indexing="ij"), axis=-1)
    pos[len(tags) - t * h * w :] = grid.reshape(-1, 3)
    return torch.tensor(tags, dtype=torch.long), torch.from_numpy(pos)


def dense_sol_attn_fake(q, k, v, *, sink_blocks, sink_q, block_len, tau, **_):
    """Full attention over the padded, tiled tensors the sparse processor hands
    the kernel. Within each 64-row tile, rows at or past `block_len[tile]` are
    padding: they are masked as keys, and their query rows are discarded by the
    caller. Equal to the stock processor exactly when the permutation is."""
    within = torch.arange(q.shape[1], device=q.device) % 64
    live = within < block_len.long().repeat_interleave(64)
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=live[None, None, None, :],
    )
    return out.transpose(1, 2)
