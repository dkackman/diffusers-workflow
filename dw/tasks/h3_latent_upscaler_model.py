"""
MiniMax-H3 latent upscaler - the pure 3D-convolution network, v1.

Vendored from https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
(nodes/minimax_h3_latent_upscaler_3d.py), weights at
https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler. Fixed to the v1
checkpoint's layout: 24 latent channels, 512 wide, twelve residual blocks
either side of the resize with a temporal convolution after every second one,
and no attention (the upstream node forces it off). The ComfyUI plumbing,
the latent normalization wrapper and the other resize modes are not carried:
diffusers hands MiniMax-H3 latents over already in the space the network was
trained on, so the caller runs it directly.

MIT License - Copyright (c) LBH-123-AI

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

import torch
import torch.nn as nn
import torch.nn.functional as F

# The v1 architecture, as the checkpoint's config.json declares it
LATENT_CHANNELS = 24
CHANNELS = 512
EMBED_DIM = 64
BLOCKS = 12
TEMPORAL_EVERY = 2
TEMPORAL_KERNEL = 5
# Temporal chunking: a segment of latent frames the network sees at once, and
# how many frames of context each side it is padded with and blended over
CHUNK_FRAMES = 32

# The checkpoint on the Hub was saved from a wrapper module
_STATE_DICT_PREFIX = "upscaler."


def _normalization(channels):
    return nn.GroupNorm(32, channels)


class ResBlockEmb3D(nn.Module):
    """A residual block modulated by the scale embedding."""

    def __init__(self, channels, emb_channels):
        super().__init__()
        self.in_layers = nn.Sequential(
            _normalization(channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * channels),
        )
        self.out_norm = _normalization(channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=0.0),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return x + h


class TemporalConv(nn.Module):
    """A depthwise convolution along time, then a pointwise one, residual."""

    def __init__(self, channels, kernel_size=TEMPORAL_KERNEL):
        super().__init__()
        padding = kernel_size // 2
        self.norm = _normalization(channels)
        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0),
            groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        h = F.silu(self.norm(x))
        return x + self.pwconv(self.dwconv(h))


def _block_stack(channels):
    blocks = nn.ModuleList()
    for b in range(BLOCKS):
        blocks.append(ResBlockEmb3D(channels, EMBED_DIM))
        if b % TEMPORAL_EVERY == 0:
            blocks.append(TemporalConv(channels))
    return blocks


class LatentResizer3D(nn.Module):
    """Resize MiniMax-H3 video latents (B, 24, T, H, W) to a target size.

    `scale` conditions the blocks - the upstream node passes the mean of the
    two spatial scale factors - and `target_size` is the (T, H, W) the
    latents are interpolated to between the two block stacks.
    """

    def __init__(self, channels=CHANNELS):
        super().__init__()
        self.conv_in = nn.Conv3d(LATENT_CHANNELS, channels, 3, padding=1)
        self.embed = nn.Sequential(
            nn.Linear(1, EMBED_DIM), nn.SiLU(), nn.Linear(EMBED_DIM, EMBED_DIM)
        )
        self.in_blocks = _block_stack(channels)
        self.out_blocks = _block_stack(channels)
        self.norm_out = _normalization(channels)
        self.conv_out = nn.Conv3d(channels, LATENT_CHANNELS, 3, padding=1)

    def forward(self, x, scale, target_size):
        size = tuple(int(s) for s in target_size)
        if size == tuple(x.shape[-3:]):
            return x
        T = x.shape[2]
        if T <= CHUNK_FRAMES:
            return self._forward_segment(x, scale, size)
        return self._forward_chunked(x, scale, size)

    def _forward_chunked(self, x, scale, size):
        """Run a long clip in overlapping temporal segments and blend them.

        Upstream's scheme exactly: replicate-pad the clip by one temporal
        kernel each side, run each 32-frame segment with that much context,
        and cross-fade the overlaps with linear weights.
        """
        B, C, T = x.shape[:3]
        overlap = TEMPORAL_KERNEL
        chunk = CHUNK_FRAMES
        x_padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode="replicate")

        out_full = torch.zeros(
            B, C, T, size[-2], size[-1], device=x.device, dtype=x.dtype
        )
        weight_full = torch.zeros(1, 1, T, 1, 1, device=x.device, dtype=x.dtype)

        for seg_start in range(0, T, chunk):
            seg_end = min(T, seg_start + chunk)
            out_start = max(0, seg_start - overlap)
            out_end = min(T, seg_end + overlap)
            lo = max(0, out_start - overlap)
            hi = min(T + 2 * overlap, out_end + overlap)

            seg = x_padded[:, :, lo:hi].contiguous()
            seg_out = self._forward_segment(seg, scale, (hi - lo, size[-2], size[-1]))

            s0 = (out_start + overlap) - lo
            n_valid = out_end - out_start
            valid_out = seg_out[:, :, s0 : s0 + n_valid]

            weight = torch.ones(n_valid, device=x.device, dtype=x.dtype)
            if seg_start > out_start:
                blend = seg_start - out_start
                weight[:blend] = torch.arange(
                    1, blend + 1, device=x.device, dtype=x.dtype
                ) / (blend + 1)
            if out_end > seg_end:
                blend = out_end - seg_end
                weight[-blend:] = torch.arange(
                    blend, 0, -1, device=x.device, dtype=x.dtype
                ) / (blend + 1)

            weight = weight.view(1, 1, n_valid, 1, 1)
            out_full[:, :, out_start:out_end] += valid_out * weight
            weight_full[:, :, out_start:out_end] += weight
            del seg, seg_out, valid_out

        return out_full / weight_full.clamp(min=1e-8)

    def _forward_segment(self, x, scale, size):
        scale_emb = torch.tensor([[scale - 1.0]], dtype=x.dtype, device=x.device)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for block in self.in_blocks:
            x = self._run_block(block, x, emb)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for block in self.out_blocks:
            x = self._run_block(block, x, emb)
        x = F.silu(self.norm_out(x))
        return self.conv_out(x)

    @staticmethod
    def _run_block(block, x, emb):
        if isinstance(block, ResBlockEmb3D):
            return block(x, emb.expand(x.shape[0], -1))
        return block(x)


def architecture_mismatch(state_dict, model):
    """What keeps a state dict from being the v1 architecture, or None.

    The checkpoint is loaded strictly, but a strict-load failure lists every
    key; this names the difference in a sentence a caller can act on - the
    usual cause is pointing `weight_name` at a later checkpoint whose layout
    this vendored network does not have.
    """
    expected = {name: tuple(t.shape) for name, t in model.state_dict().items()}
    found = {name: tuple(t.shape) for name, t in state_dict.items()}
    missing = sorted(set(expected) - set(found))
    unexpected = sorted(set(found) - set(expected))
    reshaped = sorted(
        name for name in set(expected) & set(found) if expected[name] != found[name]
    )
    if not (missing or unexpected or reshaped):
        return None
    parts = []
    if missing:
        parts.append(f"{len(missing)} missing (first: {missing[0]})")
    if unexpected:
        parts.append(f"{len(unexpected)} unexpected (first: {unexpected[0]})")
    if reshaped:
        name = reshaped[0]
        parts.append(
            f"{len(reshaped)} of another shape (first: {name} is "
            f"{list(found[name])}, v1 has {list(expected[name])})"
        )
    return "; ".join(parts)


def strip_prefix(state_dict):
    """The state dict with the checkpoint's wrapper prefix removed."""
    return {
        name.removeprefix(_STATE_DICT_PREFIX): tensor
        for name, tensor in state_dict.items()
    }


def build_upscaler(state_dict):
    """The v1 network with the state dict loaded, or a directed refusal.

    Raises:
        ValueError: The state dict is not the v1 architecture's
    """
    model = LatentResizer3D()
    state_dict = strip_prefix(state_dict)
    mismatch = architecture_mismatch(state_dict, model)
    if mismatch:
        raise ValueError(
            "The upscaler weights are not the v1 MiniMax-H3 latent upscaler "
            f"this task runs: {mismatch}. Point weight_name at a v1 checkpoint"
        )
    model.load_state_dict(state_dict, strict=True)
    return model.eval()
