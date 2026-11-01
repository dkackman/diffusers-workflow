"""
DFN5B CLIP conditioning for MMAudio (dw/tasks/mmaudio_model.py), v1.

Ported from https://github.com/hkchengrex/MMAudio at commit
974010a026c731054592d8f777218bd9d85a6c24: the CLIP parts of
mmaudio/model/utils/features_utils.py (with its patch_clip). The
weights are apple/DFN5B-CLIP-ViT-H-14-384's open_clip checkpoint
(CLIP_REPO_ID / CLIP_REVISION), under the Apple Machine Learning Research
Model License (research purposes only). "Apple Machine Learning Research
Model is licensed under the Apple Machine Learning Research Model License
Agreement."

Carried over as written: what upstream reads from the two towers - the image
tower's normalised projected embedding per frame (encode_video_with_clip),
and, through upstream's patch_clip, the text tower's normalised last hidden
states for every one of the 77 token positions rather than the pooled
embedding; and the CLIP pixel statistics.

Changed in the port, so nothing here touches the Hub or imports open_clip:
- The towers are transformers' CLIPVisionModelWithProjection and
  CLIPTextModel, built from DFN5B's open_clip dimensions (ClipConfig, the
  image side at upstream's 384 pixels) and filled by a key map from the
  open_clip state dict at load time (open_clip_to_transformers). The mapped
  tensors are never written to disk. load_clip() reads a local path its
  caller gives, with torch.load(weights_only=True) or safetensors.
- The tokenizer is the repository's tokenizer.json read with tokenizers,
  laid out as open_clip lays it out: start token, text, end token, padded
  with 0 to 77 tokens. open_clip's ftfy text repair is not carried.

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

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

CLIP_REPO_ID = "apple/DFN5B-CLIP-ViT-H-14-384"
CLIP_REVISION = "01b771ed0d1395ca5ffdd279897d665ebe00dfd2"
CLIP_WEIGHTS_FILE = "open_clip_pytorch_model.bin"
CLIP_TOKENIZER_FILE = "tokenizer.json"

# upstream's clip_preprocess (the OpenAI CLIP statistics, as DFN5B's config)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# --- configuration -------------------------------------------------


@dataclass
class ClipConfig:
    """DFN5B-CLIP-ViT-H-14-384's open_clip dimensions. The image side is 384
    pixels (the open_clip config's 378 gives the same 27x27 patch grid)."""

    embed_dim: int = 1024
    image_size: int = 384
    patch_size: int = 14
    vision_width: int = 1280
    vision_layers: int = 32
    vision_heads: int = 16
    vision_mlp: int = 5120
    context_length: int = 77
    vocab_size: int = 49408
    text_width: int = 1024
    text_layers: int = 24
    text_heads: int = 16
    text_mlp: int = 4096
    sot_token_id: int = 49406
    eot_token_id: int = 49407
    pad_token_id: int = 0
    layer_norm_eps: float = 1e-5


def clip_vision_config(config: ClipConfig):
    from transformers import CLIPVisionConfig

    return CLIPVisionConfig(
        hidden_size=config.vision_width,
        intermediate_size=config.vision_mlp,
        num_hidden_layers=config.vision_layers,
        num_attention_heads=config.vision_heads,
        image_size=config.image_size,
        patch_size=config.patch_size,
        projection_dim=config.embed_dim,
        hidden_act="quick_gelu",
        layer_norm_eps=config.layer_norm_eps,
    )


def clip_text_config(config: ClipConfig):
    from transformers import CLIPTextConfig

    return CLIPTextConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.text_width,
        intermediate_size=config.text_mlp,
        num_hidden_layers=config.text_layers,
        num_attention_heads=config.text_heads,
        max_position_embeddings=config.context_length,
        projection_dim=config.embed_dim,
        hidden_act="quick_gelu",
        layer_norm_eps=config.layer_norm_eps,
        bos_token_id=config.sot_token_id,
        eos_token_id=config.eot_token_id,
        pad_token_id=config.pad_token_id,
    )


# open_clip residual-block names -> transformers CLIP encoder-layer names
_BLOCK_NAMES = (
    ("ln_1.", "layer_norm1."),
    ("ln_2.", "layer_norm2."),
    ("mlp.c_fc.", "mlp.fc1."),
    ("mlp.c_proj.", "mlp.fc2."),
    ("attn.out_proj.", "self_attn.out_proj."),
)
# the tower-level names, (open_clip, transformers)
_VISION_NAMES = (
    ("visual.conv1.weight", "vision_model.embeddings.patch_embedding.weight"),
    ("visual.class_embedding", "vision_model.embeddings.class_embedding"),
    (
        "visual.positional_embedding",
        "vision_model.embeddings.position_embedding.weight",
    ),
    ("visual.ln_pre.weight", "vision_model.pre_layrnorm.weight"),
    ("visual.ln_pre.bias", "vision_model.pre_layrnorm.bias"),
    ("visual.ln_post.weight", "vision_model.post_layernorm.weight"),
    ("visual.ln_post.bias", "vision_model.post_layernorm.bias"),
)
_TEXT_NAMES = (
    ("token_embedding.weight", "embeddings.token_embedding.weight"),
    ("positional_embedding", "embeddings.position_embedding.weight"),
    ("ln_final.weight", "final_layer_norm.weight"),
    ("ln_final.bias", "final_layer_norm.bias"),
)
# open_clip tensors the conditioning doesn't use: the text projection and
# pooling (patch_clip reads the hidden states) and the contrastive scale
UNUSED_CLIP_KEYS = ("text_projection", "logit_scale", "logit_bias", "attn_mask")
_VISION_BLOCKS = ("visual.transformer.resblocks.", "vision_model.encoder.layers.")
_TEXT_BLOCKS = ("transformer.resblocks.", "encoder.layers.")


def _map_block(rest: str, value: torch.Tensor, dst: str, out: dict) -> None:
    index, _, name = rest.partition(".")
    prefix = f"{dst}{index}."
    if name in ("attn.in_proj_weight", "attn.in_proj_bias"):
        kind = name.rsplit("_", 1)[1]
        for proj, part in zip(("q", "k", "v"), value.chunk(3, dim=0)):
            out[f"{prefix}self_attn.{proj}_proj.{kind}"] = part
        return
    for src_name, dst_name in _BLOCK_NAMES:
        if name.startswith(src_name):
            out[prefix + dst_name + name[len(src_name) :]] = value
            return
    raise KeyError(f"unmapped open_clip block tensor: {name}")


def open_clip_to_transformers(state: dict) -> tuple[dict, dict]:
    """Split an open_clip CLIP state dict into (vision, text) state dicts for
    CLIPVisionModelWithProjection and CLIPTextModel. The fused attention
    in-projection splits into q/k/v; the image projection is transposed into
    a Linear weight. Any key it doesn't know raises."""
    vision, text = {}, {}
    names = dict(_VISION_NAMES)
    text_names = dict(_TEXT_NAMES)
    for key, value in state.items():
        if key in UNUSED_CLIP_KEYS:
            continue
        if key == "visual.proj":
            vision["visual_projection.weight"] = value.t().contiguous()
        elif key in names:
            vision[names[key]] = value
        elif key in text_names:
            text[text_names[key]] = value
        elif key.startswith(_VISION_BLOCKS[0]):
            _map_block(key[len(_VISION_BLOCKS[0]) :], value, _VISION_BLOCKS[1], vision)
        elif key.startswith(_TEXT_BLOCKS[0]):
            _map_block(key[len(_TEXT_BLOCKS[0]) :], value, _TEXT_BLOCKS[1], text)
        else:
            raise KeyError(f"unmapped open_clip tensor: {key}")
    return vision, text


def _unmap_blocks(state: dict, src: str, dst: str, out: dict) -> None:
    qkv: dict = {}
    reverse = [(t, o) for o, t in _BLOCK_NAMES]
    for key, value in state.items():
        if not key.startswith(src):
            continue
        index, _, name = key[len(src) :].partition(".")
        if name.startswith("self_attn.") and not name.startswith("self_attn.out_proj."):
            proj, kind = name[len("self_attn.") :].split("_proj.")
            qkv.setdefault((index, kind), {})[proj] = value
            continue
        for t_name, o_name in reverse:
            if name.startswith(t_name):
                out[f"{dst}{index}.{o_name}{name[len(t_name) :]}"] = value
                break
        else:
            raise KeyError(f"unmapped transformers block tensor: {key}")
    for (index, kind), parts in qkv.items():
        out[f"{dst}{index}.attn.in_proj_{kind}"] = torch.cat(
            [parts["q"], parts["k"], parts["v"]], dim=0
        )


def transformers_to_open_clip(vision: dict, text: dict) -> dict:
    """The inverse of open_clip_to_transformers, less the unused keys."""
    out = {}
    for o_name, t_name in _VISION_NAMES:
        out[o_name] = vision[t_name]
    for o_name, t_name in _TEXT_NAMES:
        out[o_name] = text[t_name]
    out["visual.proj"] = vision["visual_projection.weight"].t().contiguous()
    _unmap_blocks(vision, _VISION_BLOCKS[1], _VISION_BLOCKS[0], out)
    _unmap_blocks(text, _TEXT_BLOCKS[1], _TEXT_BLOCKS[0], out)
    return out


def _load_into(model: nn.Module, state: dict) -> None:
    """Load a mapped state dict strictly, whether or not this transformers
    version prefixes the text model's keys with 'text_model.'."""
    own = model.state_dict().keys()
    if any(k.startswith("text_model.") for k in own) and not any(
        k.startswith("text_model.") for k in state
    ):
        state = {f"text_model.{k}": v for k, v in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    # non-persistent position ids are not in a state dict: rebuild them
    for name, buffer in list(model.named_buffers()):
        if not buffer.is_meta:
            continue
        if not name.endswith("position_ids"):
            raise RuntimeError(f"no value for buffer {name}")
        owner_name, _, attr = name.rpartition(".")
        owner = model.get_submodule(owner_name)
        ids = torch.arange(buffer.shape[-1]).expand(buffer.shape)
        owner.register_buffer(attr, ids, persistent=False)


class DFN5BClip(nn.Module):
    """The two DFN5B towers, with upstream's encode_video_with_clip and
    patch_clip'd encode_text outputs."""

    def __init__(self, config: Optional[ClipConfig] = None):
        super().__init__()
        from transformers import CLIPTextModel, CLIPVisionModelWithProjection

        self.config = config = config or ClipConfig()
        self.vision = CLIPVisionModelWithProjection(clip_vision_config(config))
        self.text = CLIPTextModel(clip_text_config(config))
        self.register_buffer(
            "mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False
        )

    @property
    def device(self) -> torch.device:
        return self.text.get_input_embeddings().weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.text.get_input_embeddings().weight.dtype

    @torch.inference_mode()
    def encode_frames(self, frames: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        """frames (B, T, 3, H, W) in [0, 1] at image_size square -> the
        normalised image embedding per frame, (B, T, embed_dim)."""
        b, t, c, h, w = frames.shape
        size = self.config.image_size
        if c != 3 or h != size or w != size:
            raise ValueError(
                f"CLIP frames {tuple(frames.shape)}: want (B, T, 3, {size}, {size})"
            )
        x = frames.to(self.device, self.dtype).reshape(b * t, c, h, w)
        x = (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)
        step = b * t if batch_size < 0 else batch_size
        outputs = [
            self.vision(pixel_values=x[i : i + step]).image_embeds
            for i in range(0, b * t, step)
        ]
        return F.normalize(torch.cat(outputs, dim=0), dim=-1).view(b, t, -1)

    @torch.inference_mode()
    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, context_length) -> the normalised last hidden state of
        every token position, (B, context_length, text_width)."""
        hidden = self.text(input_ids=tokens.to(self.device)).last_hidden_state
        return F.normalize(hidden, dim=-1)


def load_clip(
    weights_path,
    *,
    config: Optional[ClipConfig] = None,
    device="cpu",
    dtype: torch.dtype = torch.float32,
) -> DFN5BClip:
    """Build the towers from a local open_clip checkpoint: safetensors if the
    path is one, otherwise torch.load(weights_only=True). The key map runs in
    memory; nothing converted is written."""
    path = Path(weights_path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    vision, text = open_clip_to_transformers(state)
    del state
    with torch.device("meta"):
        clip = DFN5BClip(config)
    _load_into(clip.vision, vision)
    _load_into(clip.text, text)
    clip.register_buffer(
        "mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False
    )
    clip.register_buffer(
        "std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False
    )
    clip.to(device=device, dtype=dtype)
    return clip.eval().requires_grad_(False)


def load_clip_tokenizer(path):
    """The CLIP BPE tokenizer from a local tokenizer.json."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(Path(path)))


def tokenize(
    tokenizer, texts: list[str], config: Optional[ClipConfig] = None
) -> torch.Tensor:
    """open_clip's layout: start token, the text's BPE ids, end token,
    truncated to context_length with the end token kept last, padded with
    pad_token_id (0). Text is html-unescaped, whitespace-collapsed and
    lower-cased first, as open_clip's 'lower' cleaning does."""
    config = config or ClipConfig()
    length = config.context_length
    rows = []
    for text in texts:
        clean = " ".join(html.unescape(html.unescape(text)).split()).lower()
        ids = tokenizer.encode(clean, add_special_tokens=False).ids
        row = [config.sot_token_id, *ids, config.eot_token_id]
        if len(row) > length:
            row = row[:length]
            row[-1] = config.eot_token_id
        rows.append(row + [config.pad_token_id] * (length - len(row)))
    return torch.tensor(rows, dtype=torch.long)


__all__ = [
    "CLIP_MEAN",
    "CLIP_REPO_ID",
    "CLIP_REVISION",
    "CLIP_STD",
    "CLIP_TOKENIZER_FILE",
    "CLIP_WEIGHTS_FILE",
    "ClipConfig",
    "DFN5BClip",
    "UNUSED_CLIP_KEYS",
    "clip_text_config",
    "clip_vision_config",
    "load_clip",
    "load_clip_tokenizer",
    "open_clip_to_transformers",
    "tokenize",
    "transformers_to_open_clip",
]
