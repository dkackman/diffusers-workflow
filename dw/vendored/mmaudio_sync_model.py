"""
MMAudio - the Synchformer visual encoder (a MotionFormer feature extractor)
that turns video frames into the synchronisation features the network
conditions on, v1.

Vendored from https://github.com/hkchengrex/MMAudio at commit
974010a026c731054592d8f777218bd9d85a6c24 (mmaudio/ext/synchformer/
synchformer.py, motionformer.py, video_model_builder.py, vit_helper.py and
divided_224_16x4.yaml, merged into this one file, plus the segmenting of
mmaudio/model/utils/features_utils.py's encode_video_with_sync). The weights
are hkchengrex/MMAudio's ext_weights/synchformer_state_dict.pth (REPO_ID /
REVISION / SYNC_FILE below), under their own licence, stricter than this
code's: CC-BY-NC 4.0, non-commercial use only.

Carried over as written: the divided space-time vision transformer (the 3D
patch embedding, the class token and separate spatial / temporal position
embeddings, the divided attention in which the class token attends to every
patch and the patches attend across time or across space, the MLP blocks),
the spatial aggregation layer (a class-token transformer encoder layer) and
the identity temporal aggregation of Synchformer(), the 16-frame segments
with a step of 8, and every parameter and submodule name, so the checkpoint
loads strictly: the keys are upstream's, under the `vfeat_extractor.` prefix.
The patch embedding of the 2D stem and the projection of the 3D one are kept
as parameters although the forward pass only uses the 3D one, as upstream.

Changed in the port, so nothing here touches the Hub, a network, timm,
einops, omegaconf or hydra:
- The configuration of divided_224_16x4.yaml and the patches of
  MotionFormer.__init__ (APPROX_ATTN_DIM and the others) are the SyncConfig
  dataclass; only the divided, separate-position, extract-features,
  factorized path of Synchformer() is carried. The checkpoint and config
  download, the ckpt_path branches, the joint and plain attention blocks, the
  other aggregation modules, the global representation, the classification
  heads, token masks, drop-path and training initialisation are not: the
  head, pre-logits and dropouts are identity (so own no keys) and the
  weights always load from the checkpoint. load_synchformer() reads a local
  .pth path its caller gives, with torch.load(weights_only=True), strictly.
- einops rearranges and reductions are plain view/permute/reshape calls;
  the attention uses torch's scaled_dot_product_attention with upstream's
  scale; trunc_normal_ is torch.nn.init's. No float64 anywhere.
- The upstream sources carry Facebook (Motionformer) and Ross Wightman
  copyright headers; the licence below is the one upstream's synchformer
  directory ships.

MIT License

Copyright (c) 2024 Vladimir Iashin

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

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

REPO_ID = "hkchengrex/MMAudio"
REVISION = "eb13a1a98fdbec91753775c57b074ccdfc60587c"
SYNC_FILE = "ext_weights/synchformer_state_dict.pth"

__all__ = [
    "REPO_ID",
    "REVISION",
    "SYNC_FILE",
    "SyncConfig",
    "Synchformer",
    "encode_sync_frames",
    "load_synchformer",
]

# upstream's segmenting of a video: windows 8 frames apart (16 frames each by default)
STEP_SIZE = 8


# --- configuration -----------------------------------------------------------


@dataclass
class SyncConfig:
    """divided_224_16x4.yaml's VIT / DATA values with the patches of
    MotionFormer.__init__ (the defaults), the shape of the vision transformer."""

    image_size: int = 224
    patch_size: int = 16
    patch_size_temp: int = 2
    in_chans: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: int = 4
    qkv_bias: bool = True
    temporal_resolution: int = 8
    segment_frames: int = 16

    def __post_init__(self) -> None:
        if self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be a multiple of num_heads")
        if self.segment_frames != self.temporal_resolution * self.patch_size_temp:
            raise ValueError(
                "segment_frames must be temporal_resolution * patch_size_temp"
            )

    @property
    def grid(self) -> int:
        """Patches along each side of a frame."""
        return self.image_size // self.patch_size


# --- layers (vit_helper.py) --------------------------------------------------


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class PatchEmbed(nn.Module):
    """Image to patch embedding; only its geometry is used (the 3D one embeds)."""

    def __init__(
        self, img_size: int, patch_size: int, in_chans: int, embed_dim: int
    ) -> None:
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )


class PatchEmbed3D(nn.Module):
    """Video to patch embedding: (B, C, T, H, W) -> (B, t*h*w, D)."""

    def __init__(
        self,
        img_size: int,
        in_chans: int,
        patch_size: int,
        z_block_size: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.height = img_size // patch_size
        self.width = img_size // patch_size
        self.z_block_size = z_block_size
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(z_block_size, patch_size, patch_size),
            stride=(z_block_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class DividedAttention(nn.Module):
    """Attention in which the class token sees every token and the patches see
    their own time or space group plus the class token (upstream's, no masks)."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def _attend(self, q, k, v) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, scale=self.scale)

    def forward(self, x: torch.Tensor, mode: str, size: int) -> torch.Tensor:
        """mode "time": size is n, patches per frame, and time is the sequence;
        mode "space": size is f, frames, and each frame's patches are."""
        b, n_tok, dim = x.shape
        h = self.num_heads
        d = dim // h
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (b, n, h*d) -> (b*h, n, d)
        q, k, v = (
            t.reshape(b, n_tok, h, d).permute(0, 2, 1, 3).reshape(b * h, n_tok, d)
            for t in (q, k, v)
        )
        bh = b * h

        # the class token attends to the keys / values of every token
        cls_out = self._attend(q[:, :1], k, v)

        q_, k_, v_ = q[:, 1:], k[:, 1:], v[:, 1:]
        if mode == "time":
            n = size
            f = (n_tok - 1) // n
            # 'b (f n) d -> (b n) f d'
            q_, k_, v_ = (
                t.reshape(bh, f, n, d).transpose(1, 2).reshape(bh * n, f, d)
                for t in (q_, k_, v_)
            )
        else:
            f = size
            n = (n_tok - 1) // f
            # 'b (f n) d -> (b f) n d'
            q_, k_, v_ = (t.reshape(bh * f, n, d) for t in (q_, k_, v_))

        # the class key / value joins every group
        r = q_.shape[0] // bh
        cls_k = k[:, :1].repeat_interleave(r, dim=0)
        cls_v = v[:, :1].repeat_interleave(r, dim=0)
        out = self._attend(
            q_, torch.cat((cls_k, k_), dim=1), torch.cat((cls_v, v_), dim=1)
        )

        if mode == "time":
            # '(b n) f d -> b (f n) d'
            out = out.reshape(bh, n, f, d).transpose(1, 2).reshape(bh, f * n, d)
        else:
            out = out.reshape(bh, f * n, d)
        out = torch.cat((cls_out, out), dim=1)
        # '(b h) n d -> b n (h d)'
        out = out.reshape(b, h, n_tok, d).permute(0, 2, 1, 3).reshape(b, n_tok, dim)
        return self.proj(out)


class DividedSpaceTimeBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int,
        qkv_bias: bool,
        norm_layer,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DividedAttention(dim, num_heads, qkv_bias)
        self.timeattn = DividedAttention(dim, num_heads, qkv_bias)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.norm3 = norm_layer(dim)

    def forward(self, x: torch.Tensor, seq_len: int, num_frames: int) -> torch.Tensor:
        x = x + self.timeattn(self.norm3(x), "time", seq_len)
        x = x + self.attn(self.norm1(x), "space", num_frames)
        return x + self.mlp(self.norm2(x))


# --- spatial aggregation (motionformer.py) -----------------------------------


class SpatialTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Aggregates the patches of each frame into the class token's output: an
    encoder layer (its keys are torch's) with a class token of its own."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.self_attn.embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """x is (B*S*t, h*w, D); returns (B*S*t, D)."""
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=-2)
        return super().forward(x)[:, 0, :]


# --- the encoder (video_model_builder.py, motionformer.py) -------------------


class MotionFormer(nn.Module):
    """VisionTransformer + MotionFormer with extract_features, factorized
    space and time, a transformer-layer spatial aggregation and an identity
    temporal one: (B, S, C, T, H, W) -> (B, S, t, D)."""

    def __init__(self, cfg: SyncConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.temporal_resolution = cfg.temporal_resolution
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # the 2D stem is only used for its geometry, its weights are kept
        self.patch_embed = PatchEmbed(
            cfg.image_size, cfg.patch_size, cfg.in_chans, cfg.embed_dim
        )
        self.patch_embed_3d = PatchEmbed3D(
            cfg.image_size,
            cfg.in_chans,
            cfg.patch_size,
            cfg.patch_size_temp,
            cfg.embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, cfg.embed_dim)
        )
        self.temp_embed = nn.Parameter(
            torch.zeros(1, cfg.temporal_resolution, cfg.embed_dim)
        )
        self.blocks = nn.ModuleList(
            DividedSpaceTimeBlock(
                cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.qkv_bias, norm_layer
            )
            for _ in range(cfg.depth)
        )
        self.norm = norm_layer(cfg.embed_dim)
        # no classification head, pre-logits or dropout: identity, no keys
        self.spatial_attn_agg = SpatialTransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            activation=nn.GELU(),
            batch_first=True,
            dim_feedforward=cfg.mlp_ratio * cfg.embed_dim,
            dropout=0.0,
            layer_norm_eps=1e-6,
            norm_first=True,
        )
        self.temp_attn_agg = nn.Identity()
        self._init_weights()

    def _init_weights(self) -> None:
        for p in (self.cls_token, self.pos_embed):
            nn.init.trunc_normal_(p, std=0.02)
        nn.init.trunc_normal_(self.temp_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x is (B*S, C, T, H, W); returns (B*S, 1 + t*h*w, D)."""
        x = self.patch_embed_3d(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        # separate position embeddings: space tiled over time, time over space
        npatch = self.patch_embed.num_patches
        cls_embed = self.pos_embed[:, :1, :]
        tile_pos = self.pos_embed[:, 1:, :].repeat(1, self.temporal_resolution, 1)
        tile_temp = self.temp_embed.repeat_interleave(npatch, 1)
        x = x + torch.cat([cls_embed, tile_pos + tile_temp], dim=1)

        for blk in self.blocks:
            x = blk(x, seq_len=npatch, num_frames=self.temporal_resolution)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C, T, H, W = x.shape
        x = self.forward_features(x.reshape(B * S, C, T, H, W))
        x = self.norm(x[:, 1:, :])  # without the class token
        # (B*S, t*h*w, D) -> (B*S*t, h*w, D): the tokens are ordered (t, h, w)
        t = T // self.patch_embed_3d.z_block_size
        hw = self.patch_embed_3d.height * self.patch_embed_3d.width
        x = x.reshape(B * S * t, hw, self.embed_dim)
        x = self.spatial_attn_agg(x).reshape(B * S, t, self.embed_dim)
        x = self.temp_attn_agg(x)
        return x.view(B, S, *x.shape[1:])


class Synchformer(nn.Module):
    """Upstream's Synchformer(): only the visual feature extractor."""

    def __init__(self, config: SyncConfig | None = None) -> None:
        super().__init__()
        self.config = config or SyncConfig()
        self.vfeat_extractor = MotionFormer(self.config)

    def forward(self, vis: torch.Tensor) -> torch.Tensor:
        """vis is (B, S, Tv, C, H, W); returns (B, S, t, D), e.g. (B, 7, 8, 768)."""
        return self.vfeat_extractor(vis.permute(0, 1, 3, 2, 4, 5))  # (B, S, C, Tv, ..)

    def load_state_dict(  # type: ignore[override]
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        # discard all entries except vfeat_extractor, as upstream
        state_dict = {
            k: v for k, v in state_dict.items() if k.startswith("vfeat_extractor")
        }
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def load_synchformer(
    path,
    device="cpu",
    dtype: torch.dtype = torch.float32,
    config: SyncConfig | None = None,
) -> Synchformer:
    """Build the encoder from a local checkpoint (a plain state dict), loaded
    with weights_only=True and strictly; entries outside vfeat_extractor are
    dropped, as upstream."""
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    with torch.device("meta"):
        model = Synchformer(config)
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


@torch.inference_mode()
def encode_sync_frames(
    model: Synchformer, frames: torch.Tensor, batch_size: int = -1
) -> torch.Tensor:
    """Upstream's encode_video_with_sync: frames (B, T, 3, H, W) in 16-frame
    segments 8 apart -> (B, num_segments * t, D)."""
    b, t, c, h, w = frames.shape
    size = model.config.image_size
    if c != model.config.in_chans or h != size or w != size:
        raise ValueError(
            f"frames must be (B, T, 3, {size}, {size}), got {tuple(frames.shape)}"
        )
    segment = model.config.segment_frames
    step = STEP_SIZE
    if t < segment:
        raise ValueError(f"need at least {segment} frames, got {t}")
    num_segments = (t - segment) // step + 1
    x = torch.stack(
        [frames[:, i * step : i * step + segment] for i in range(num_segments)], dim=1
    )  # (B, S, T, C, H, W)
    x = x.reshape(b * num_segments, 1, segment, c, h, w)
    if batch_size < 0:
        batch_size = b
    outputs = [
        model(x[i : i + batch_size]) for i in range(0, b * num_segments, batch_size)
    ]
    x = torch.cat(outputs, dim=0)  # (B*S, 1, t, D)
    return x.reshape(b, num_segments * x.shape[2], x.shape[3])
