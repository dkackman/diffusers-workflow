"""
MMAudio - the flow-matching network that turns video and text features into
an audio latent, and its Euler sampler, v1.

Vendored from https://github.com/hkchengrex/MMAudio at commit
974010a026c731054592d8f777218bd9d85a6c24 (mmaudio/model/networks.py,
transformer_layers.py, low_level.py, embeddings.py, sequence_config.py,
flow_matching.py and mmaudio/ext/rotary_embeddings.py, merged into this one
file). Its DFN5B CLIP conditioning is dw/tasks/mmaudio_clip.py. The weights
are hkchengrex/MMAudio's weights/mmaudio_large_44k_v2.pth (REPO_ID /
REVISION below), under their own licence, stricter than this code's:
CC-BY-NC 4.0, non-commercial use only.

Carried over as written: the MMAudio network (the joint and fused MM-DiT
blocks, the input projections, the timestep embedder, the rotary position
rotations, the synchformer position embedding and the sync upsampling, and
the final layer's global conditioning, upstream's issue #55, as trained),
the sequence-length arithmetic of SequenceConfig, and the empty-condition
sequences and classifier-free guidance (upstream's ode_wrapper).

Changed in the port, so nothing here touches the Hub, a network, timm,
einops, hydra, tensordict or torchdiffeq:
- Only the large_44k_v2 configuration is carried (MMAudioConfig's defaults),
  so only the v2 input projections and timestep embedder; the v1 branches,
  the other model sizes, training (loss, conditional flow) and the random
  weight initialisation are not, since the weights always load from the
  checkpoint. load_network() reads a local .pth path its caller gives, with
  torch.load(weights_only=True), strictly.
- The flow-matching sampler is our own Euler loop (euler_sample), the same
  fixed-step scheme as upstream's 'euler' mode over t in [0, 1]; upstream's
  adaptive torchdiffeq mode is not carried.
- einops rearranges are plain view/permute/reshape calls, nn.Buffer is
  register_buffer, and the autocast-disabled regions compute in float32
  directly.

MIT License

Copyright (c) 2024 Sony Research Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

REPO_ID = "hkchengrex/MMAudio"
REVISION = "eb13a1a98fdbec91753775c57b074ccdfc60587c"
NETWORK_FILE = "weights/mmaudio_large_44k_v2.pth"

# buffers upstream's load_weights drops from a checkpoint: they are rebuilt
NON_PERSISTENT_KEYS = ("t_embed.freqs", "latent_rot", "clip_rot")


# --- configuration -----------------------------------------------------------


@dataclass
class SequenceConfig:
    """Upstream's SequenceConfig: how a duration in seconds becomes the latent,
    CLIP and Synchformer sequence lengths (44.1 kHz defaults)."""

    duration: float = 8.0
    sampling_rate: int = 44100
    spectrogram_frame_rate: int = 512
    latent_downsample_rate: int = 2
    clip_frame_rate: int = 8
    sync_frame_rate: int = 25
    sync_num_frames_per_segment: int = 16
    sync_step_size: int = 8
    sync_downsample_rate: int = 2

    @property
    def num_audio_frames(self) -> int:
        return (
            self.latent_seq_len
            * self.spectrogram_frame_rate
            * self.latent_downsample_rate
        )

    @property
    def latent_seq_len(self) -> int:
        return int(
            math.ceil(
                self.duration
                * self.sampling_rate
                / self.spectrogram_frame_rate
                / self.latent_downsample_rate
            )
        )

    @property
    def clip_seq_len(self) -> int:
        return int(self.duration * self.clip_frame_rate)

    @property
    def sync_seq_len(self) -> int:
        num_frames = self.duration * self.sync_frame_rate
        num_segments = (
            num_frames - self.sync_num_frames_per_segment
        ) // self.sync_step_size + 1
        return int(
            num_segments * self.sync_num_frames_per_segment / self.sync_downsample_rate
        )


@dataclass
class MMAudioConfig:
    """The network's sizes; the defaults are upstream's large_44k_v2."""

    latent_dim: int = 40
    clip_dim: int = 1024
    sync_dim: int = 768
    text_dim: int = 1024
    hidden_dim: int = 64 * 14
    depth: int = 21
    fused_depth: int = 14
    num_heads: int = 14
    mlp_ratio: float = 4.0
    text_seq_len: int = 77


# --- low-level layers --------------------------------------------------------


class ChannelLastConv1d(nn.Conv1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 1)).permute(0, 2, 1)


def _swiglu_width(hidden_dim: int, multiple_of: int = 256) -> int:
    hidden_dim = int(2 * hidden_dim / 3)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


class MLP(nn.Module):
    """The SwiGLU feed-forward from Stability-AI/sd3-ref."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = _swiglu_width(hidden_dim)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ConvMLP(nn.Module):
    """MLP with channel-last 1-D convolutions in place of the linears."""

    def __init__(
        self, dim: int, hidden_dim: int, kernel_size: int = 3, padding: int = 1
    ):
        super().__init__()
        hidden_dim = _swiglu_width(hidden_dim)

        def conv(i, o):
            return ChannelLastConv1d(
                i, o, bias=False, kernel_size=kernel_size, padding=padding
            )

        self.w1 = conv(dim, hidden_dim)
        self.w2 = conv(hidden_dim, dim)
        self.w3 = conv(dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# --- rotary positions and timestep embedding ---------------------------------


def compute_rope_rotations(
    length: int, dim: int, theta: int, *, freq_scaling: float = 1.0, device=None
) -> torch.Tensor:
    """(1, length, dim/2, 2, 2) rotation matrices, in float32."""
    assert dim % 2 == 0
    pos = torch.arange(length, dtype=torch.float32, device=device)
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    freqs = freqs * freq_scaling
    rot = pos[:, None] * freqs[None, :]
    rot = torch.stack(
        [torch.cos(rot), -torch.sin(rot), torch.sin(rot), torch.cos(rot)], dim=-1
    )
    return rot.view(1, length, dim // 2, 2, 2)


def apply_rope(x: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    _x = x.float()
    _x = _x.view(*_x.shape[:-1], -1, 1, 2)
    x_out = rot[..., 0] * _x[..., 0] + rot[..., 1] * _x[..., 1]
    return x_out.reshape(*x.shape).to(dtype=x.dtype)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep features through a two-layer MLP (DiT), with the v2
    frequencies: max_period 1 over hidden_dim features."""

    def __init__(self, dim: int, frequency_embedding_size: int, max_period: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.register_buffer("freqs", self._frequencies(), persistent=False)

    def _frequencies(self, device=None) -> torch.Tensor:
        size = self.frequency_embedding_size
        freqs = 1.0 / (
            10000
            ** (torch.arange(0, size, 2, dtype=torch.float32, device=device) / size)
        )
        return (10000 / self.max_period) * freqs

    def reset_frequencies(self, device=None) -> None:
        self.freqs = self._frequencies(device)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t[:, None].float() * self.freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(t_freq.to(t.dtype))


# --- transformer blocks ------------------------------------------------------


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
    b, h, n, d = out.shape
    return out.transpose(1, 2).reshape(b, n, h * d).contiguous()


class SelfAttention(nn.Module):
    def __init__(self, dim: int, nheads: int):
        super().__init__()
        self.dim = dim
        self.nheads = nheads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.RMSNorm(dim // nheads)
        self.k_norm = nn.RMSNorm(dim // nheads)

    def pre_attention(self, x: torch.Tensor, rot: Optional[torch.Tensor]):
        b, n, _ = x.shape
        # 'b n (h d j) -> b h n d j' with j = 3 (q, k, v interleaved last)
        qkv = self.qkv(x).view(b, n, self.nheads, self.dim // self.nheads, 3)
        qkv = qkv.permute(0, 2, 1, 3, 4)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rot is not None:
            q = apply_rope(q, rot)
            k = apply_rope(k, rot)
        return q, k, v


class MMDitSingleBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        nhead: int,
        mlp_ratio: float = 4.0,
        pre_only: bool = False,
        kernel_size: int = 7,
        padding: int = 3,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(dim, nhead)
        self.pre_only = pre_only
        if pre_only:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True)
            )
            return
        if kernel_size == 1:
            self.linear1 = nn.Linear(dim, dim)
            self.ffn = MLP(dim, int(dim * mlp_ratio))
        else:
            self.linear1 = ChannelLastConv1d(
                dim, dim, kernel_size=kernel_size, padding=padding
            )
            self.ffn = ConvMLP(
                dim, int(dim * mlp_ratio), kernel_size=kernel_size, padding=padding
            )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

    def pre_attention(
        self, x: torch.Tensor, c: torch.Tensor, rot: Optional[torch.Tensor]
    ):
        modulation = self.adaLN_modulation(c)
        if self.pre_only:
            shift_msa, scale_msa = modulation.chunk(2, dim=-1)
            gates = (None, None, None, None)
        else:
            shift_msa, scale_msa, *gates = modulation.chunk(6, dim=-1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        return self.attn.pre_attention(x, rot), tuple(gates)

    def post_attention(self, x: torch.Tensor, attn_out: torch.Tensor, c: tuple):
        if self.pre_only:
            return x
        gate_msa, shift_mlp, scale_mlp, gate_mlp = c
        x = x + self.linear1(attn_out) * gate_msa
        r = modulate(self.norm2(x), shift_mlp, scale_mlp)
        return x + self.ffn(r) * gate_mlp

    def forward(self, x: torch.Tensor, cond: torch.Tensor, rot: Optional[torch.Tensor]):
        x_qkv, x_conditions = self.pre_attention(x, cond, rot)
        return self.post_attention(x, attention(*x_qkv), x_conditions)


class JointBlock(nn.Module):
    """One joint-attention block over the latent, CLIP and text streams."""

    def __init__(
        self, dim: int, nhead: int, mlp_ratio: float = 4.0, pre_only: bool = False
    ):
        super().__init__()
        self.pre_only = pre_only
        self.latent_block = MMDitSingleBlock(
            dim, nhead, mlp_ratio, pre_only=False, kernel_size=3, padding=1
        )
        self.clip_block = MMDitSingleBlock(
            dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=3, padding=1
        )
        self.text_block = MMDitSingleBlock(
            dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=1
        )

    def forward(
        self, latent, clip_f, text_f, global_c, extended_c, latent_rot, clip_rot
    ):
        x_qkv, x_mod = self.latent_block.pre_attention(latent, extended_c, latent_rot)
        c_qkv, c_mod = self.clip_block.pre_attention(clip_f, global_c, clip_rot)
        t_qkv, t_mod = self.text_block.pre_attention(text_f, global_c, rot=None)

        latent_len = latent.shape[1]
        clip_len = clip_f.shape[1]
        joint_qkv = [torch.cat([x_qkv[i], c_qkv[i], t_qkv[i]], dim=2) for i in range(3)]
        attn_out = attention(*joint_qkv)
        x_attn_out = attn_out[:, :latent_len]
        c_attn_out = attn_out[:, latent_len : latent_len + clip_len]
        t_attn_out = attn_out[:, latent_len + clip_len :]

        latent = self.latent_block.post_attention(latent, x_attn_out, x_mod)
        if not self.pre_only:
            clip_f = self.clip_block.post_attention(clip_f, c_attn_out, c_mod)
            text_f = self.text_block.post_attention(text_f, t_attn_out, t_mod)
        return latent, clip_f, text_f


class FinalBlock(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True)
        )
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.conv = ChannelLastConv1d(dim, out_dim, kernel_size=7, padding=3)

    def forward(self, latent: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.conv(modulate(self.norm(latent), shift, scale))


# --- the network -------------------------------------------------------------


@dataclass
class PreprocessedConditions:
    clip_f: torch.Tensor
    sync_f: torch.Tensor
    text_f: torch.Tensor
    clip_f_c: torch.Tensor
    text_f_c: torch.Tensor


class MMAudioNetwork(nn.Module):
    """Upstream's MMAudio class (v2): predicts the flow from noise toward an
    audio latent given CLIP frame features, Synchformer features and text."""

    def __init__(self, config: Optional[MMAudioConfig] = None, duration: float = 8.0):
        super().__init__()
        config = config or MMAudioConfig()
        self.config = config
        self.latent_dim = config.latent_dim
        self.hidden_dim = hidden = config.hidden_dim
        self.num_heads = config.num_heads
        self._text_seq_len = config.text_seq_len

        self.audio_input_proj = nn.Sequential(
            ChannelLastConv1d(config.latent_dim, hidden, kernel_size=7, padding=3),
            nn.SiLU(),
            ConvMLP(hidden, hidden * 4, kernel_size=7, padding=3),
        )
        self.clip_input_proj = nn.Sequential(
            nn.Linear(config.clip_dim, hidden),
            nn.SiLU(),
            ConvMLP(hidden, hidden * 4, kernel_size=3, padding=1),
        )
        self.sync_input_proj = nn.Sequential(
            ChannelLastConv1d(config.sync_dim, hidden, kernel_size=7, padding=3),
            nn.SiLU(),
            ConvMLP(hidden, hidden * 4, kernel_size=3, padding=1),
        )
        self.text_input_proj = nn.Sequential(
            nn.Linear(config.text_dim, hidden), nn.SiLU(), MLP(hidden, hidden * 4)
        )
        self.clip_cond_proj = nn.Linear(hidden, hidden)
        self.text_cond_proj = nn.Linear(hidden, hidden)
        self.global_cond_mlp = MLP(hidden, hidden * 4)
        # each synchformer output segment has 8 feature frames
        self.sync_pos_emb = nn.Parameter(torch.zeros((1, 1, 8, config.sync_dim)))
        self.final_layer = FinalBlock(hidden, config.latent_dim)
        self.t_embed = TimestepEmbedder(
            hidden, frequency_embedding_size=hidden, max_period=1
        )

        joint_depth = config.depth - config.fused_depth
        self.joint_blocks = nn.ModuleList(
            JointBlock(
                hidden,
                config.num_heads,
                config.mlp_ratio,
                pre_only=(i == joint_depth - 1),
            )
            for i in range(joint_depth)
        )
        self.fused_blocks = nn.ModuleList(
            MMDitSingleBlock(
                hidden, config.num_heads, config.mlp_ratio, kernel_size=3, padding=1
            )
            for _ in range(config.fused_depth)
        )

        # loaded from the checkpoint; identity statistics until then
        frozen = dict(requires_grad=False)
        self.latent_mean = nn.Parameter(torch.zeros(1, 1, config.latent_dim), **frozen)
        self.latent_std = nn.Parameter(torch.ones(1, 1, config.latent_dim), **frozen)
        self.empty_string_feat = nn.Parameter(
            torch.zeros(config.text_seq_len, config.text_dim), **frozen
        )
        self.empty_clip_feat = nn.Parameter(torch.zeros(1, config.clip_dim))
        self.empty_sync_feat = nn.Parameter(torch.zeros(1, config.sync_dim))

        self.set_duration(duration)

    # sequence lengths -------------------------------------------------------

    def set_duration(self, duration: float) -> SequenceConfig:
        """Upstream's update_seq_lengths for a duration in seconds; rebuilds the
        rotations (float32) on the network's device. Returns the sequence
        config, whose num_audio_frames is the decoded length."""
        seq = SequenceConfig(duration=duration)
        self.seq_cfg = seq
        self._latent_seq_len = seq.latent_seq_len
        self._clip_seq_len = seq.clip_seq_len
        self._sync_seq_len = seq.sync_seq_len
        head_dim = self.hidden_dim // self.num_heads
        device = None if self.latent_mean.is_meta else self.latent_mean.device
        latent_rot = compute_rope_rotations(
            self._latent_seq_len, head_dim, 10000, freq_scaling=1.0, device=device
        )
        clip_rot = compute_rope_rotations(
            self._clip_seq_len,
            head_dim,
            10000,
            freq_scaling=self._latent_seq_len / self._clip_seq_len,
            device=device,
        )
        self.register_buffer("latent_rot", latent_rot, persistent=False)
        self.register_buffer("clip_rot", clip_rot, persistent=False)
        return seq

    @property
    def device(self) -> torch.device:
        return self.latent_mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self.latent_mean.dtype

    @property
    def latent_seq_len(self) -> int:
        return self._latent_seq_len

    @property
    def clip_seq_len(self) -> int:
        return self._clip_seq_len

    @property
    def sync_seq_len(self) -> int:
        return self._sync_seq_len

    # latents ----------------------------------------------------------------

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.latent_mean) / self.latent_std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.latent_std + self.latent_mean

    # conditions -------------------------------------------------------------

    def preprocess_conditions(
        self, clip_f: torch.Tensor, sync_f: torch.Tensor, text_f: torch.Tensor
    ) -> PreprocessedConditions:
        """The work that doesn't depend on the latent or the step, done once."""
        if clip_f.shape[1] != self._clip_seq_len:
            raise ValueError(
                f"clip features {tuple(clip_f.shape)}: want {self._clip_seq_len} frames"
            )
        if sync_f.shape[1] != self._sync_seq_len:
            raise ValueError(
                f"sync features {tuple(sync_f.shape)}: want {self._sync_seq_len}"
            )
        if text_f.shape[1] != self._text_seq_len:
            raise ValueError(
                f"text features {tuple(text_f.shape)}: want {self._text_seq_len}"
            )

        bs = clip_f.shape[0]
        num_sync_segments = self._sync_seq_len // 8
        sync_f = sync_f.view(bs, num_sync_segments, 8, -1) + self.sync_pos_emb
        sync_f = sync_f.flatten(1, 2)

        clip_f = self.clip_input_proj(clip_f)
        sync_f = self.sync_input_proj(sync_f)
        text_f = self.text_input_proj(text_f)

        # upsample the sync features to the latent's length
        sync_f = F.interpolate(
            sync_f.transpose(1, 2), size=self._latent_seq_len, mode="nearest-exact"
        ).transpose(1, 2)

        clip_f_c = self.clip_cond_proj(clip_f.mean(dim=1))
        text_f_c = self.text_cond_proj(text_f.mean(dim=1))
        return PreprocessedConditions(clip_f, sync_f, text_f, clip_f_c, text_f_c)

    def predict_flow(
        self, latent: torch.Tensor, t: torch.Tensor, conditions: PreprocessedConditions
    ) -> torch.Tensor:
        if latent.shape[1] != self._latent_seq_len:
            raise ValueError(
                f"latent {tuple(latent.shape)}: want {self._latent_seq_len} frames"
            )
        clip_f = conditions.clip_f
        text_f = conditions.text_f

        latent = self.audio_input_proj(latent)
        global_c = self.global_cond_mlp(conditions.clip_f_c + conditions.text_f_c)
        global_c = self.t_embed(t).unsqueeze(1) + global_c.unsqueeze(1)
        extended_c = global_c + conditions.sync_f

        for block in self.joint_blocks:
            latent, clip_f, text_f = block(
                latent,
                clip_f,
                text_f,
                global_c,
                extended_c,
                self.latent_rot,
                self.clip_rot,
            )
        for block in self.fused_blocks:
            latent = block(latent, extended_c, self.latent_rot)

        # upstream notes this should be extended_c (its issue #55); the
        # checkpoint was trained with global_c, so global_c it stays
        return self.final_layer(latent, global_c)

    def forward(self, latent, clip_f, sync_f, text_f, t) -> torch.Tensor:
        return self.predict_flow(
            latent, t, self.preprocess_conditions(clip_f, sync_f, text_f)
        )

    def get_empty_string_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_string_feat.unsqueeze(0).expand(bs, -1, -1)

    def get_empty_clip_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_clip_feat.unsqueeze(0).expand(bs, self._clip_seq_len, -1)

    def get_empty_sync_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_sync_feat.unsqueeze(0).expand(bs, self._sync_seq_len, -1)

    def get_empty_conditions(
        self, bs: int, *, negative_text_features: Optional[torch.Tensor] = None
    ) -> PreprocessedConditions:
        """The unconditional branch of guidance: learned empty CLIP and sync
        sequences, and the empty string's text features or the negative
        prompt's."""
        if negative_text_features is not None:
            empty_text = negative_text_features
        else:
            empty_text = self.get_empty_string_sequence(1)
        conditions = self.preprocess_conditions(
            self.get_empty_clip_sequence(1), self.get_empty_sync_sequence(1), empty_text
        )
        conditions.clip_f = conditions.clip_f.expand(bs, -1, -1)
        conditions.sync_f = conditions.sync_f.expand(bs, -1, -1)
        conditions.clip_f_c = conditions.clip_f_c.expand(bs, -1)
        if negative_text_features is None:
            conditions.text_f = conditions.text_f.expand(bs, -1, -1)
            conditions.text_f_c = conditions.text_f_c.expand(bs, -1)
        return conditions

    def guided_flow(
        self,
        t: torch.Tensor,
        latent: torch.Tensor,
        conditions: PreprocessedConditions,
        empty_conditions: PreprocessedConditions,
        cfg_strength: float,
    ) -> torch.Tensor:
        """Upstream's ode_wrapper: classifier-free guidance, or the conditional
        flow alone when cfg_strength is below 1."""
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)
        if cfg_strength < 1.0:
            return self.predict_flow(latent, t, conditions)
        return cfg_strength * self.predict_flow(latent, t, conditions) + (
            1 - cfg_strength
        ) * self.predict_flow(latent, t, empty_conditions)


def load_network(
    path,
    *,
    config: Optional[MMAudioConfig] = None,
    duration: float = 8.0,
    device="cpu",
    dtype: torch.dtype = torch.float32,
) -> MMAudioNetwork:
    """Build the network from a local checkpoint (a plain state dict), loaded
    with weights_only=True and strictly. The rotations are rebuilt in float32
    after the cast, as upstream's update_seq_lengths leaves them."""
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    for key in NON_PERSISTENT_KEYS:
        state.pop(key, None)
    with torch.device("meta"):
        net = MMAudioNetwork(config, duration=duration)
    net.load_state_dict(state, strict=True, assign=True)
    # the non-persistent buffers are still meta: build them, cast with the
    # rest (upstream casts freqs too), then rebuild the rotations in float32
    net.t_embed.reset_frequencies()
    net.set_duration(duration)
    net.to(device=device, dtype=dtype)
    net.set_duration(duration)
    return net.eval().requires_grad_(False)


# --- sampling ----------------------------------------------------------------


def euler_sample(
    flow: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    steps: int,
    t0: float = 0.0,
    t1: float = 1.0,
) -> torch.Tensor:
    """Integrate dx/dt = flow(t, x) from t0 to t1 with `steps` fixed Euler
    steps: upstream FlowMatching's 'euler' mode (min_sigma 0)."""
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    times = torch.linspace(t0, t1, steps + 1)
    x = x0
    for i in range(steps):
        t = times[i]
        x = x + (times[i + 1] - t) * flow(t, x)
    return x


@torch.inference_mode()
def sample_latent(
    net: MMAudioNetwork,
    *,
    clip_features: Optional[torch.Tensor] = None,
    sync_features: Optional[torch.Tensor] = None,
    text_features: Optional[torch.Tensor] = None,
    negative_text_features: Optional[torch.Tensor] = None,
    steps: int = 25,
    cfg_strength: float = 4.5,
    generator: Optional[torch.Generator] = None,
    batch_size: int = 1,
) -> torch.Tensor:
    """Upstream's generate() up to the decoder: noise to an unnormalised audio
    latent (batch, latent_seq_len, latent_dim). A missing condition takes the
    network's empty sequence, so text-only (no video features) and video-only
    (no text features) both run; negative text replaces the empty string in
    guidance's unconditional branch."""
    given = [f for f in (clip_features, sync_features, text_features) if f is not None]
    bs = given[0].shape[0] if given else batch_size
    device, dtype = net.device, net.dtype

    def cast(features):
        return None if features is None else features.to(device, dtype)

    clip_f = cast(clip_features)
    sync_f = cast(sync_features)
    text_f = cast(text_features)
    negative = cast(negative_text_features)
    if clip_f is None:
        clip_f = net.get_empty_clip_sequence(bs)
    if sync_f is None:
        sync_f = net.get_empty_sync_sequence(bs)
    if text_f is None:
        text_f = net.get_empty_string_sequence(bs)

    x0 = torch.randn(
        bs,
        net.latent_seq_len,
        net.latent_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    conditions = net.preprocess_conditions(clip_f, sync_f, text_f)
    empty_conditions = net.get_empty_conditions(bs, negative_text_features=negative)

    def flow(t, x):
        return net.guided_flow(t, x, conditions, empty_conditions, cfg_strength)

    return net.unnormalize(euler_sample(flow, x0, steps))


__all__ = [
    "MMAudioConfig",
    "MMAudioNetwork",
    "NETWORK_FILE",
    "PreprocessedConditions",
    "REPO_ID",
    "REVISION",
    "SequenceConfig",
    "euler_sample",
    "load_network",
    "sample_latent",
]
