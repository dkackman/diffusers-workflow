"""
Audio8-TTS (ArkTTS) - the dual-AR text-to-speech model, its audio codec and
its prompt processor, v1.

Vendored from https://huggingface.co/Edge0/Audio8-TTS-Realistic-0.6b at
revision 1e857568bfd4f02931aca3988ee773a44d2d26d3 (configuration_arktts.py,
modeling_arktts.py, modeling_arktts_codec.py, processing_arktts.py, merged
into this one file). The weights are that repository's model.safetensors and
codec.pth at the same revision (REPO_ID / REVISION below).

Carried over as written: the slow and fast transformers, the hand-written
generate() loop with its own KV cache, top-k/top-p and RAS sampling and the
fast-AR codebook loop, the codec's encoder, residual quantizers and decoder,
and the processor's prompt layout.

Changed in the port, so nothing here touches the Hub, a network, torchaudio
or soundfile, and nothing depends on transformers' model-loading machinery:
- The config, model and processor are plain classes, not PretrainedConfig,
  PreTrainedModel or ProcessorMixin subclasses. load_model() builds the model
  from the local config.json, model.safetensors, generation_config.json and
  codec.pth paths its caller gives; load_tokenizer() reads a local
  tokenizer.json. Upstream's Hub loading and saving entry points (and the
  saver's copy of the codec) are not carried. The transformers logits/stopping
  list classes become plain lists of callables, ModelOutput/BatchFeature plain
  dataclasses and dicts.
- codec.pth loads with torch.load(weights_only=True); the deprecated legacy
  weight_norm on the quantizer projections is folded into plain weights at
  load, and the TorchScript snake activation is plain eager code.
- Reference audio given as a path is read with audio_utils.load_audio and
  resampled with dw.dsp.resample_waveform. The processor's precomputed
  reference-codes input (np.load of a file) is not carried.
- The codec's layer widths are config fields whose defaults are the released
  checkpoint's, so a test can build a tiny one; codec.pth loads strictly, so
  any other size fails to load.

Apache License 2.0 - upstream NOTICE:

    audio8_tts

    The model architecture is inspired by the DualAR design used in Fish
    Audio S2 Pro.

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy of
the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.utils.parametrizations import weight_norm

REPO_ID = "Edge0/Audio8-TTS-Realistic-0.6b"
REVISION = "1e857568bfd4f02931aca3988ee773a44d2d26d3"
CONFIG_FILE = "config.json"
GENERATION_CONFIG_FILE = "generation_config.json"
WEIGHTS_FILE = "model.safetensors"
TOKENIZER_FILE = "tokenizer.json"
CODEC_FILE = "codec.pth"
MODEL_FILES = (
    CONFIG_FILE,
    GENERATION_CONFIG_FILE,
    WEIGHTS_FILE,
    TOKENIZER_FILE,
    CODEC_FILE,
)


# --- configuration (upstream configuration_arktts.py) -----------------------


class ArkttsConfig:
    model_type = "arktts"

    def __init__(
        self,
        vocab_size: int = 155776,
        dim: int = 896,
        n_layer: int = 24,
        n_head: int = 14,
        n_local_heads: int = 2,
        head_dim: int = 64,
        intermediate_size: int = 4864,
        max_seq_len: int = 2048,
        rope_base: float = 1_000_000,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        attention_qkv_bias: bool = True,
        attention_qk_norm: bool = False,
        attention_o_bias: bool = False,
        tie_word_embeddings: bool = True,
        codebook_size: int = 4096,
        num_codebooks: int = 10,
        semantic_begin_id: int = 151678,
        semantic_end_id: int = 155773,
        n_fast_layer: int = 4,
        fast_dim: int = 896,
        fast_n_head: int = 14,
        fast_n_local_heads: int = 2,
        fast_head_dim: int = 64,
        fast_intermediate_size: int = 4864,
        fast_attention_qkv_bias: bool = False,
        fast_attention_qk_norm: bool = False,
        fast_attention_o_bias: bool = False,
        norm_fastlayer_input: bool = True,
        initializer_range: float = 0.02,
        use_gradient_checkpointing: bool = False,
        codec_filename: str = CODEC_FILE,
        codec_sample_rate: int = 44100,
        codec_frame_size: int = 2048,
        codec_post_n_layer: int = 8,
        codec_post_n_head: int = 16,
        codec_post_n_local_heads: int = 8,
        codec_post_intermediate_size: int = 1216,
        codec_encoder_dim: int = 64,
        codec_encoder_transformer_layers: int = 4,
        codec_latent_dim: int = 1024,
        codec_decoder_dim: int = 1536,
        codec_pre_n_layer: int = 8,
        codec_pre_n_head: int = 16,
        codec_pre_intermediate_size: int = 3072,
        codec_residual_codebook_size: int = 1024,
        codec_codebook_dim: int = 8,
        ras_window_size: int = 10,
        ras_temperature: float = 1.0,
        ras_top_p: float = 0.9,
        eos_token_id: int = 151645,
        pad_token_id: int = 151643,
        **kwargs,
    ):
        # config.json also carries transformers bookkeeping (architectures,
        # auto_map, dtype, the hidden_size-style aliases set below) - none of
        # it configures the network
        del kwargs
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.hidden_size = self.dim
        self.n_layer = int(n_layer)
        self.num_hidden_layers = self.n_layer
        self.n_head = int(n_head)
        self.num_attention_heads = self.n_head
        self.n_local_heads = int(n_local_heads)
        self.num_key_value_heads = self.n_local_heads
        self.head_dim = int(head_dim)
        self.intermediate_size = int(intermediate_size)
        self.max_seq_len = int(max_seq_len)
        self.max_position_embeddings = self.max_seq_len
        self.rope_base = float(rope_base)
        self.rope_theta = self.rope_base
        self.norm_eps = float(norm_eps)
        self.rms_norm_eps = self.norm_eps
        self.dropout = float(dropout)
        self.attention_qkv_bias = bool(attention_qkv_bias)
        self.attention_qk_norm = bool(attention_qk_norm)
        self.attention_o_bias = bool(attention_o_bias)
        self.tie_word_embeddings = bool(tie_word_embeddings)
        self.codebook_size = int(codebook_size)
        self.num_codebooks = int(num_codebooks)
        self.semantic_begin_id = int(semantic_begin_id)
        self.semantic_end_id = int(semantic_end_id)
        self.n_fast_layer = int(n_fast_layer)
        self.fast_dim = int(fast_dim)
        self.fast_n_head = int(fast_n_head)
        self.fast_n_local_heads = int(fast_n_local_heads)
        self.fast_head_dim = int(fast_head_dim)
        self.fast_intermediate_size = int(fast_intermediate_size)
        self.fast_attention_qkv_bias = bool(fast_attention_qkv_bias)
        self.fast_attention_qk_norm = bool(fast_attention_qk_norm)
        self.fast_attention_o_bias = bool(fast_attention_o_bias)
        self.norm_fastlayer_input = bool(norm_fastlayer_input)
        self.initializer_range = float(initializer_range)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.codec_filename = str(codec_filename)
        self.codec_sample_rate = int(codec_sample_rate)
        self.codec_frame_size = int(codec_frame_size)
        self.codec_post_n_layer = int(codec_post_n_layer)
        self.codec_post_n_head = int(codec_post_n_head)
        self.codec_post_n_local_heads = int(codec_post_n_local_heads)
        self.codec_post_intermediate_size = int(codec_post_intermediate_size)
        self.codec_encoder_dim = int(codec_encoder_dim)
        self.codec_encoder_transformer_layers = int(codec_encoder_transformer_layers)
        self.codec_latent_dim = int(codec_latent_dim)
        self.codec_decoder_dim = int(codec_decoder_dim)
        self.codec_pre_n_layer = int(codec_pre_n_layer)
        self.codec_pre_n_head = int(codec_pre_n_head)
        self.codec_pre_intermediate_size = int(codec_pre_intermediate_size)
        self.codec_residual_codebook_size = int(codec_residual_codebook_size)
        self.codec_codebook_dim = int(codec_codebook_dim)
        self.ras_window_size = int(ras_window_size)
        self.ras_temperature = float(ras_temperature)
        self.ras_top_p = float(ras_top_p)
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)

    @classmethod
    def from_json_file(cls, path) -> "ArkttsConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(**json.load(handle))


# --- model (upstream modeling_arktts.py) ------------------------------------


@dataclass
class ArkttsModelOutput:
    logits: Optional[Tensor] = None
    hidden_states: Optional[Tensor] = None
    codebook_logits: Optional[Tensor] = None


@dataclass
class ArkttsGenerateOutput:
    codes: Optional[Tensor] = None
    code_lengths: Optional[Tensor] = None
    finished: Optional[Tensor] = None


class ArkttsSemanticLogitsProcessor:
    def __init__(self, semantic_begin_id: int, semantic_end_id: int, eos_token_id: int):
        self.semantic_begin_id = int(semantic_begin_id)
        self.semantic_end_id = int(semantic_end_id)
        self.eos_token_id = int(eos_token_id)

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        filtered = torch.full_like(scores, float("-inf"))
        filtered[:, self.semantic_begin_id : self.semantic_end_id + 1] = scores[
            :, self.semantic_begin_id : self.semantic_end_id + 1
        ]
        filtered[:, self.eos_token_id] = scores[:, self.eos_token_id]
        return filtered


class ArkttsLegacyTopKTopPLogitsProcessor:
    """Matches the candidate filtering order used by the original inference code."""

    def __init__(self, top_k: int, top_p: float):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        self.top_k = int(top_k)
        self.top_p = float(top_p)

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative = torch.cumsum(torch.softmax(sorted_scores, dim=-1), dim=-1)
        positions = torch.arange(sorted_scores.shape[-1], device=scores.device)
        threshold = torch.tensor(
            self.top_p, dtype=cumulative.dtype, device=cumulative.device
        )
        remove_sorted = (cumulative > threshold) | (positions >= self.top_k)
        remove_sorted[..., 0] = False
        remove = torch.zeros_like(remove_sorted).scatter(
            1, sorted_indices, remove_sorted
        )
        return scores.masked_fill(remove, float("-inf"))


def _apply_processors(processors, input_ids: Tensor, scores: Tensor) -> Tensor:
    for processor in processors:
        scores = processor(input_ids, scores)
    return scores


class ArkttsKVCache(nn.Module):
    def __init__(
        self,
        batch_size: int,
        max_length: int,
        heads: int,
        head_dim: int,
        dtype,
        return_full: bool = False,
    ):
        super().__init__()
        shape = (batch_size, heads, max_length, head_dim)
        self.register_buffer("keys", torch.zeros(shape, dtype=dtype), persistent=False)
        self.register_buffer(
            "values", torch.zeros(shape, dtype=dtype), persistent=False
        )
        self.return_full = bool(return_full)

    def update(self, cache_position: Tensor, keys: Tensor, values: Tensor):
        self.keys[:, :, cache_position] = keys
        self.values[:, :, cache_position] = values
        end = (
            self.keys.shape[2]
            if self.return_full
            else int(cache_position[-1].item()) + 1
        )
        return self.keys[:, :, :end], self.values[:, :, :end]


class ArkttsRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return normalized.to(x.dtype) * self.weight


def _precompute_rope(length: int, head_dim: int, base: float, device=None) -> Tensor:
    frequencies = 1.0 / (
        base
        ** (
            torch.arange(0, head_dim, 2, device=device).float()[: head_dim // 2]
            / head_dim
        )
    )
    phases = torch.outer(torch.arange(length, device=device), frequencies)
    values = torch.polar(torch.ones_like(phases), phases)
    return torch.stack((values.real, values.imag), dim=-1).to(torch.bfloat16)


def _apply_rope(x: Tensor, rope: Tensor) -> Tensor:
    shaped = x.float().reshape(*x.shape[:-1], -1, 2)
    if rope.ndim == 3:
        rope = rope[None, :, None]
    elif rope.ndim == 4:
        rope = rope[:, :, None]
    else:
        raise ValueError(f"Unexpected RoPE shape: {tuple(rope.shape)}")
    output = torch.stack(
        (
            shaped[..., 0] * rope[..., 0] - shaped[..., 1] * rope[..., 1],
            shaped[..., 1] * rope[..., 0] + shaped[..., 0] * rope[..., 1],
        ),
        dim=-1,
    )
    return output.flatten(3).to(x.dtype)


class ArkttsAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        n_local_heads: int,
        head_dim: int,
        qkv_bias: bool,
        output_bias: bool,
        qk_norm: bool,
        norm_eps: float,
        dropout: float,
        use_sdpa: bool,
    ):
        super().__init__()
        total = (n_head + 2 * n_local_heads) * head_dim
        self.wqkv = nn.Linear(dim, total, bias=qkv_bias)
        self.wo = nn.Linear(n_head * head_dim, dim, bias=output_bias)
        self.n_head = int(n_head)
        self.n_local_heads = int(n_local_heads)
        self.head_dim = int(head_dim)
        self.dropout = float(dropout)
        self.use_sdpa = bool(use_sdpa)
        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            self.q_norm = ArkttsRMSNorm(head_dim, norm_eps)
            self.k_norm = ArkttsRMSNorm(head_dim, norm_eps)
        self.kv_cache: Optional[ArkttsKVCache] = None

    def forward(
        self,
        x: Tensor,
        rope: Tensor,
        attention_mask: Optional[Tensor],
        cache_position: Optional[Tensor] = None,
    ) -> Tensor:
        batch, length, _ = x.shape
        query_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        query, key, value = self.wqkv(x).split((query_size, kv_size, kv_size), dim=-1)
        query = query.view(batch, length, self.n_head, self.head_dim)
        key = key.view(batch, length, self.n_local_heads, self.head_dim)
        value = value.view(batch, length, self.n_local_heads, self.head_dim)
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query = _apply_rope(query, rope).transpose(1, 2)
        key = _apply_rope(key, rope).transpose(1, 2)
        value = value.transpose(1, 2)
        if self.kv_cache is not None:
            if cache_position is None:
                raise ValueError("cache_position is required when KV cache is enabled")
            key, value = self.kv_cache.update(cache_position, key, value)
        repeats = self.n_head // self.n_local_heads
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        if self.use_sdpa:
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                scores = scores.masked_fill(~attention_mask, float("-inf"))
            probabilities = torch.softmax(scores, dim=-1)
            if self.training and self.dropout:
                probabilities = F.dropout(probabilities, p=self.dropout)
            output = probabilities @ value
        output = output.transpose(1, 2).contiguous().view(batch, length, query_size)
        return self.wo(output)


class ArkttsFeedForward(nn.Module):
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ArkttsTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_size: int,
        n_head: int,
        n_local_heads: int,
        head_dim: int,
        qkv_bias: bool,
        output_bias: bool,
        qk_norm: bool,
        norm_eps: float,
        dropout: float,
        use_sdpa: bool,
    ):
        super().__init__()
        self.attention = ArkttsAttention(
            dim,
            n_head,
            n_local_heads,
            head_dim,
            qkv_bias,
            output_bias,
            qk_norm,
            norm_eps,
            dropout,
            use_sdpa,
        )
        self.feed_forward = ArkttsFeedForward(dim, intermediate_size)
        self.ffn_norm = ArkttsRMSNorm(dim, norm_eps)
        self.attention_norm = ArkttsRMSNorm(dim, norm_eps)

    def forward(self, x, rope, attention_mask, cache_position=None):
        hidden = x + self.attention(
            self.attention_norm(x), rope, attention_mask, cache_position
        )
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class ArkttsModel(nn.Module):
    def __init__(self, config: ArkttsConfig, codec_path=None):
        super().__init__()
        self.config = config
        self.codec_path = str(codec_path) if codec_path is not None else None
        self.generation_config: dict = {}
        self.embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.codebook_embeddings = nn.Embedding(
            config.codebook_size * config.num_codebooks, config.dim
        )
        self.layers = nn.ModuleList(
            [
                ArkttsTransformerBlock(
                    config.dim,
                    config.intermediate_size,
                    config.n_head,
                    config.n_local_heads,
                    config.head_dim,
                    config.attention_qkv_bias,
                    config.attention_o_bias,
                    config.attention_qk_norm,
                    config.norm_eps,
                    config.dropout,
                    True,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.norm = ArkttsRMSNorm(config.dim, config.norm_eps)
        self.fast_project_in = (
            nn.Linear(config.dim, config.fast_dim)
            if config.fast_dim != config.dim
            else nn.Identity()
        )
        self.fast_embeddings = nn.Embedding(config.codebook_size, config.fast_dim)
        self.fast_layers = nn.ModuleList(
            [
                ArkttsTransformerBlock(
                    config.fast_dim,
                    config.fast_intermediate_size,
                    config.fast_n_head,
                    config.fast_n_local_heads,
                    config.fast_head_dim,
                    config.fast_attention_qkv_bias,
                    config.fast_attention_o_bias,
                    config.fast_attention_qk_norm,
                    config.norm_eps,
                    config.dropout,
                    False,
                )
                for _ in range(config.n_fast_layer)
            ]
        )
        self.fast_norm = ArkttsRMSNorm(config.fast_dim, config.norm_eps)
        self.fast_output = nn.Linear(config.fast_dim, config.codebook_size, bias=False)
        self.register_buffer(
            "freqs_cis",
            _precompute_rope(config.max_seq_len, config.head_dim, config.rope_base),
            persistent=False,
        )
        self.register_buffer(
            "fast_freqs_cis",
            _precompute_rope(
                config.num_codebooks, config.fast_head_dim, config.rope_base
            ),
            persistent=False,
        )
        # kept out of the module tree: the codec loads lazily from its own
        # file and is not part of model.safetensors
        self.__dict__["_arktts_codec"] = None

    def reset_rope_buffers(self, device=None):
        """Recompute the RoPE tables - after load_model built the model on meta."""
        config = self.config
        self.freqs_cis = _precompute_rope(
            config.max_seq_len, config.head_dim, config.rope_base, device=device
        )
        self.fast_freqs_cis = _precompute_rope(
            config.num_codebooks, config.fast_head_dim, config.rope_base, device=device
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def get_input_embeddings(self):
        return self.embeddings

    def _embed(self, input_ids: Tensor) -> Tensor:
        codebook_embeds = []
        for index in range(self.config.num_codebooks):
            codebook_embeds.append(
                self.codebook_embeddings(
                    input_ids[:, index + 1] + index * self.config.codebook_size
                )
            )
        codebook_sum = torch.stack(codebook_embeds, dim=1).sum(dim=1)
        semantic = (input_ids[:, 0] >= self.config.semantic_begin_id) & (
            input_ids[:, 0] <= self.config.semantic_end_id
        )
        codebook_sum = torch.where(semantic.unsqueeze(-1), codebook_sum, 0.0)
        return self.embeddings(input_ids[:, 0]) + codebook_sum

    @staticmethod
    def _causal_mask(
        attention_mask: Tensor, query_positions: Tensor, key_length: int
    ) -> Tensor:
        if attention_mask.shape[1] < key_length:
            attention_mask = F.pad(
                attention_mask,
                (0, key_length - attention_mask.shape[1]),
                value=0,
            )
        key_positions = torch.arange(key_length, device=attention_mask.device)
        causal = key_positions[None, :] <= query_positions[:, None]
        return causal[None, None] & attention_mask[:, None, None, :key_length].bool()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> ArkttsModelOutput:
        if input_ids.ndim != 3 or input_ids.shape[1] != self.config.num_codebooks + 1:
            raise ValueError(
                f"input_ids must have shape [B, {self.config.num_codebooks + 1}, T]"
            )
        batch, _, length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch, length), dtype=torch.long, device=input_ids.device
            )
        position_ids = attention_mask.long().cumsum(-1).sub(1).clamp_min(0)
        rope = self.freqs_cis[position_ids]
        mask = self._causal_mask(
            attention_mask,
            torch.arange(length, device=input_ids.device),
            length,
        )
        hidden = self._embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, rope, mask)
        normalized = self.norm(hidden)
        logits = F.linear(normalized, self.embeddings.weight)
        return ArkttsModelOutput(logits=logits, hidden_states=hidden)

    def _setup_generation_caches(self, batch_size: int, dtype):
        for layer in self.layers:
            layer.attention.kv_cache = ArkttsKVCache(
                batch_size,
                self.config.max_seq_len,
                self.config.n_local_heads,
                self.config.head_dim,
                dtype,
                return_full=True,
            ).to(self.device)
        for layer in self.fast_layers:
            layer.attention.kv_cache = ArkttsKVCache(
                batch_size,
                self.config.num_codebooks,
                self.config.fast_n_local_heads,
                self.config.fast_head_dim,
                dtype,
                return_full=True,
            ).to(self.device)

    def _clear_generation_caches(self):
        # the slow cache is max_seq_len long per layer; a cached model keeps
        # no buffers from its last run between jobs
        for layer in (*self.layers, *self.fast_layers):
            layer.attention.kv_cache = None

    def _slow_step(
        self,
        input_ids: Tensor,
        cache_position: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden = self._embed(input_ids)
        rope = self.freqs_cis[position_ids]
        mask = self._causal_mask(
            attention_mask,
            cache_position,
            self.config.max_seq_len,
        )
        for layer in self.layers:
            hidden = layer(hidden, rope, mask, cache_position)
        hidden = hidden[:, -1:]
        normalized = self.norm(hidden)
        logits = F.linear(normalized, self.embeddings.weight)[:, -1]
        fast_hidden = normalized if self.config.norm_fastlayer_input else hidden
        return logits, fast_hidden

    def _fast_step(self, hidden: Tensor, position: int) -> Tensor:
        cache_position = torch.tensor(
            [position], device=hidden.device, dtype=torch.long
        )
        rope = self.fast_freqs_cis[cache_position]
        key_mask = torch.ones(
            (hidden.shape[0], self.config.num_codebooks),
            device=hidden.device,
            dtype=torch.bool,
        )
        mask = self._causal_mask(key_mask, cache_position, self.config.num_codebooks)
        for layer in self.fast_layers:
            hidden = layer(hidden, rope, mask, cache_position)
        return self.fast_output(self.fast_norm(hidden))[:, -1]

    @staticmethod
    def _sample(scores: Tensor, generator=None) -> Tensor:
        probabilities = torch.softmax(scores, dim=-1)
        random = torch.rand(
            probabilities.shape,
            dtype=probabilities.dtype,
            device=probabilities.device,
            generator=generator,
        )
        noise = -torch.log(random)
        return torch.argmax(probabilities / noise, dim=-1)

    def _processed_scores(
        self,
        input_ids: Tensor,
        scores: Tensor,
        processors: list,
        top_k: int,
        top_p: float,
        temperature: float,
    ) -> Tensor:
        scores = _apply_processors(processors, input_ids, scores)
        scores = ArkttsLegacyTopKTopPLogitsProcessor(top_k, top_p)(input_ids, scores)
        temperature_value = torch.tensor(
            temperature, dtype=scores.dtype, device=scores.device
        ).clamp_min(1e-5)
        return scores / temperature_value

    def _sample_semantic(
        self,
        history: Tensor,
        logits: Tensor,
        custom_processors: list,
        top_k: int,
        top_p: float,
        temperature: float,
        previous: Optional[Tensor],
        do_sample: bool,
        generator=None,
    ) -> Tensor:
        processors = [
            ArkttsSemanticLogitsProcessor(
                self.config.semantic_begin_id,
                self.config.semantic_end_id,
                self.config.eos_token_id,
            ),
            *custom_processors,
        ]
        regular_scores = self._processed_scores(
            history, logits, processors, top_k, top_p, temperature
        )
        if not do_sample:
            return regular_scores.argmax(dim=-1)
        normal = self._sample(regular_scores, generator=generator)
        high_scores = self._processed_scores(
            history,
            logits,
            processors,
            top_k,
            self.config.ras_top_p,
            self.config.ras_temperature,
        )
        high = self._sample(high_scores, generator=generator)
        if previous is None:
            return normal
        repeated = (previous == normal[:, None]).any(dim=1)
        semantic = (normal >= self.config.semantic_begin_id) & (
            normal <= self.config.semantic_end_id
        )
        return torch.where(repeated & semantic, high, normal)

    def _generate_codebooks(
        self,
        slow_hidden: Tensor,
        semantic: Tensor,
        processors: list,
        top_k: int,
        top_p: float,
        temperature: float,
        do_sample: bool,
        generator=None,
    ) -> Tensor:
        hidden = self.fast_project_in(slow_hidden)
        self._fast_step(hidden, 0)
        current = (semantic - self.config.semantic_begin_id).clamp(
            0, self.config.codebook_size - 1
        )
        codebooks = [current]
        fast_history = current[:, None]
        hidden = self.fast_embeddings(current)[:, None]
        for position in range(1, self.config.num_codebooks):
            scores = self._fast_step(hidden, position)
            scores = self._processed_scores(
                fast_history, scores, processors, top_k, top_p, temperature
            )
            current = (
                self._sample(scores, generator=generator)
                if do_sample
                else scores.argmax(dim=-1)
            )
            codebooks.append(current)
            fast_history = torch.cat((fast_history, current[:, None]), dim=1)
            hidden = self.fast_embeddings(current)[:, None]
        return torch.stack(codebooks, dim=1)

    def _prepare_prompt(
        self,
        input_ids=None,
        attention_mask=None,
        prefix_input_ids=None,
        prefix_attention_mask=None,
        suffix_input_ids=None,
        suffix_attention_mask=None,
        reference_codes=None,
        reference_code_lengths=None,
        reference_audio_values=None,
        reference_audio_lengths=None,
    ):
        if input_ids is not None:
            if (
                input_ids.ndim != 3
                or input_ids.shape[1] != self.config.num_codebooks + 1
            ):
                raise ValueError(
                    "Direct input_ids must have shape [B, num_codebooks + 1, T]"
                )
            if attention_mask is None:
                attention_mask = torch.ones(
                    input_ids.shape[0],
                    input_ids.shape[-1],
                    dtype=torch.long,
                    device=input_ids.device,
                )
            return input_ids.to(self.device), attention_mask.to(self.device)
        if prefix_input_ids is None or suffix_input_ids is None:
            raise ValueError("Processor output or direct input_ids is required")
        prefix_input_ids = prefix_input_ids.to(self.device)
        suffix_input_ids = suffix_input_ids.to(self.device)
        prefix_attention_mask = prefix_attention_mask.to(self.device)
        suffix_attention_mask = suffix_attention_mask.to(self.device)
        if reference_audio_values is not None:
            if reference_codes is not None:
                raise ValueError("Provide reference audio or reference codes, not both")
            reference_codes, reference_code_lengths = self.encode_audio(
                reference_audio_values.to(self.device),
                reference_audio_lengths.to(self.device),
            )
        if reference_codes is not None:
            reference_codes = reference_codes.to(self.device)
            reference_code_lengths = reference_code_lengths.to(self.device)
        batch_size = prefix_input_ids.shape[0]
        rows = []
        for batch_index in range(batch_size):
            prefix = prefix_input_ids[
                batch_index, prefix_attention_mask[batch_index].bool()
            ]
            suffix = suffix_input_ids[
                batch_index, suffix_attention_mask[batch_index].bool()
            ]
            if reference_codes is None:
                semantic_row = torch.cat((prefix, suffix))
                values = torch.zeros(
                    (self.config.num_codebooks + 1, semantic_row.numel()),
                    dtype=torch.long,
                    device=self.device,
                )
                values[0] = semantic_row
            else:
                length = int(reference_code_lengths[batch_index])
                codes = reference_codes[batch_index, :, :length].long()
                semantic_codes = codes[0] + self.config.semantic_begin_id
                semantic_row = torch.cat((prefix, semantic_codes, suffix))
                values = torch.zeros(
                    (self.config.num_codebooks + 1, semantic_row.numel()),
                    dtype=torch.long,
                    device=self.device,
                )
                values[0] = semantic_row
                values[1:, prefix.numel() : prefix.numel() + length] = codes
            rows.append(values)
        max_length = max(row.shape[1] for row in rows)
        prompt = torch.zeros(
            (batch_size, self.config.num_codebooks + 1, max_length),
            dtype=torch.long,
            device=self.device,
        )
        prompt[:, 0] = self.config.pad_token_id
        prompt_mask = torch.zeros(
            (batch_size, max_length), dtype=torch.long, device=self.device
        )
        for batch_index, row in enumerate(rows):
            start = max_length - row.shape[1]
            prompt[batch_index, :, start:] = row
            prompt_mask[batch_index, start:] = 1
        return prompt, prompt_mask

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        prefix_input_ids: Optional[Tensor] = None,
        prefix_attention_mask: Optional[Tensor] = None,
        suffix_input_ids: Optional[Tensor] = None,
        suffix_attention_mask: Optional[Tensor] = None,
        reference_codes: Optional[Tensor] = None,
        reference_code_lengths: Optional[Tensor] = None,
        reference_audio_values: Optional[Tensor] = None,
        reference_audio_lengths: Optional[Tensor] = None,
        generation_config: Optional[dict] = None,
        logits_processor=None,
        codebook_logits_processor=None,
        stopping_criteria=None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        return_dict_in_generate: bool = False,
        generator=None,
        **kwargs,
    ):
        """Codes [B, num_codebooks, T] for the prompt; -1 marks frames past a row's end.

        generation_config is a mapping (generation_config.json's keys); the
        model's own, loaded by load_model, is used when none is given.
        logits_processor, codebook_logits_processor and stopping_criteria are
        lists of callables taking (history, scores).
        """
        if kwargs:
            raise TypeError(f"Unexpected generation arguments: {sorted(kwargs)}")
        config = (
            generation_config
            if generation_config is not None
            else self.generation_config
        ) or {}
        config_max_new = config.get("max_new_tokens")
        config_temperature = config.get("temperature")
        config_top_p = config.get("top_p")
        config_top_k = config.get("top_k")
        max_new_tokens = int(
            max_new_tokens if max_new_tokens is not None else (config_max_new or 512)
        )
        temperature = float(
            temperature if temperature is not None else (config_temperature or 0.7)
        )
        top_p = float(top_p if top_p is not None else (config_top_p or 0.9))
        top_k = int(top_k if top_k is not None else (config_top_k or 50))
        do_sample = bool(
            do_sample if do_sample is not None else config.get("do_sample", True)
        )
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        prompt, prompt_mask = self._prepare_prompt(
            input_ids,
            attention_mask,
            prefix_input_ids,
            prefix_attention_mask,
            suffix_input_ids,
            suffix_attention_mask,
            reference_codes,
            reference_code_lengths,
            reference_audio_values,
            reference_audio_lengths,
        )
        batch_size, _, prompt_width = prompt.shape
        if prompt_width >= self.config.max_seq_len:
            raise ValueError(
                f"Prompt length {prompt_width} must be smaller than "
                f"{self.config.max_seq_len}"
            )
        max_new_tokens = min(max_new_tokens, self.config.max_seq_len - prompt_width)
        self._setup_generation_caches(batch_size, self.dtype)
        try:
            return self._generate_loop(
                prompt,
                prompt_mask,
                list(logits_processor or []),
                list(codebook_logits_processor or []),
                list(stopping_criteria or []),
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                do_sample,
                return_dict_in_generate,
                generator,
            )
        finally:
            self._clear_generation_caches()

    def _generate_loop(
        self,
        prompt,
        prompt_mask,
        semantic_processors,
        codebook_processors,
        criteria,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        do_sample,
        return_dict_in_generate,
        generator,
    ):
        batch_size, _, prompt_width = prompt.shape
        cache_position = torch.arange(
            prompt_width, device=self.device, dtype=torch.long
        )
        position_ids = prompt_mask.cumsum(-1).sub(1).clamp_min(0)
        logits, slow_hidden = self._slow_step(
            prompt, cache_position, position_ids, prompt_mask
        )
        semantic_history = prompt[:, 0]
        prompt_lengths = prompt_mask.sum(-1)
        previous = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        code_lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        generated_frames = []

        for step in range(max_new_tokens):
            active_before = ~finished
            semantic = self._sample_semantic(
                semantic_history,
                logits,
                semantic_processors,
                top_k,
                top_p,
                temperature,
                previous,
                do_sample,
                generator,
            )
            codebooks = self._generate_codebooks(
                slow_hidden,
                semantic,
                codebook_processors,
                top_k,
                top_p,
                temperature,
                do_sample,
                generator,
            )
            emitted = active_before & (semantic != self.config.eos_token_id)
            frame = torch.where(emitted[:, None], codebooks, -1)
            generated_frames.append(frame)
            code_lengths += emitted.long()
            semantic_history = torch.cat((semantic_history, semantic[:, None]), dim=1)

            if previous is None:
                previous = torch.zeros(
                    (batch_size, self.config.ras_window_size),
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                previous = previous.roll(-1, dims=1)
                previous[:, -1] = semantic
            finished |= semantic.eq(self.config.eos_token_id)
            for criterion in criteria:
                stopped = criterion(semantic_history, logits)
                if not isinstance(stopped, Tensor):
                    stopped = torch.full_like(finished, bool(stopped))
                finished |= stopped.to(device=self.device, dtype=torch.bool)
            if finished.all():
                break

            next_column = torch.cat((semantic[:, None], codebooks), dim=1).unsqueeze(-1)
            new_valid = active_before.long()[:, None]
            prompt_mask = torch.cat((prompt_mask, new_valid), dim=1)
            physical_position = torch.tensor([prompt_width + step], device=self.device)
            token_position = (prompt_lengths + step)[:, None]
            with sdpa_kernel(SDPBackend.MATH):
                logits, slow_hidden = self._slow_step(
                    next_column, physical_position, token_position, prompt_mask
                )

        if generated_frames:
            codes = torch.stack(generated_frames, dim=2)
            max_valid = int(code_lengths.max().item()) if code_lengths.numel() else 0
            codes = codes[:, :, :max_valid]
        else:
            codes = torch.empty(
                (batch_size, self.config.num_codebooks, 0),
                dtype=torch.long,
                device=self.device,
            )
        if return_dict_in_generate:
            return ArkttsGenerateOutput(
                codes=codes, code_lengths=code_lengths, finished=finished
            )
        return codes

    def load_codec(self, device=None, dtype=None):
        codec = self.__dict__.get("_arktts_codec")
        target_device = torch.device(device) if device is not None else self.device
        target_dtype = dtype or self.dtype
        if target_device.type == "cpu":
            target_dtype = torch.float32
        if codec is None:
            if self.codec_path is None:
                raise FileNotFoundError(
                    f"No {self.config.codec_filename} path was given for this model"
                )
            codec = ArkttsCodec(self.config)
            codec.load_state_dict(load_codec_state(self.codec_path), strict=True)
            codec.eval()
            self.__dict__["_arktts_codec"] = codec
        codec.to(device=target_device, dtype=target_dtype)
        return codec

    @torch.inference_mode()
    def encode_audio(
        self, audio_values: Tensor, audio_lengths: Optional[Tensor] = None
    ):
        codec = self.load_codec(device=audio_values.device)
        audio_values = audio_values.to(dtype=next(codec.parameters()).dtype)
        return codec.encode(audio_values, audio_lengths)

    @torch.inference_mode()
    def decode_audio(self, codes: Tensor):
        """(waveforms [B, samples] float32, lengths [B]) at config.codec_sample_rate."""
        if codes.ndim == 2:
            codes = codes.unsqueeze(0)
        if codes.ndim != 3 or codes.shape[1] != self.config.num_codebooks:
            raise ValueError(
                f"codes must have shape [B, {self.config.num_codebooks}, T]"
            )
        codec = self.load_codec(device=codes.device)
        waveforms = []
        lengths = []
        for item in codes:
            valid = (item >= 0).all(dim=0)
            length = int(valid.sum().item())
            if length == 0:
                waveform = torch.empty(0, device=codes.device, dtype=torch.float32)
            else:
                waveform = codec.decode(item[:, :length].unsqueeze(0))[0, 0].float()
            waveforms.append(waveform)
            lengths.append(waveform.numel())
        max_length = max(lengths, default=0)
        padded = torch.zeros(
            (len(waveforms), max_length), dtype=torch.float32, device=codes.device
        )
        for index, waveform in enumerate(waveforms):
            padded[index, : waveform.numel()] = waveform
        return padded, torch.tensor(lengths, dtype=torch.long, device=codes.device)

    @torch.inference_mode()
    def generate_audio(self, **kwargs):
        codes = self.generate(**kwargs)
        waveforms, lengths = self.decode_audio(codes)
        return waveforms, lengths, codes


def _fold_legacy_weight_norm(state: dict) -> dict:
    """Plain '.weight' tensors for the checkpoint's legacy weight_g/weight_v pairs.

    The quantizer projections were saved through torch.nn.utils.weight_norm,
    which is deprecated; at inference the norm is a constant, so it is folded
    into the weight once here (w = g * v / ||v||, norm over every dim but 0).
    """
    folded = {}
    for key, value in state.items():
        if key.endswith(".weight_v"):
            continue
        if key.endswith(".weight_g"):
            stem = key[: -len("_g")]
            direction = state[f"{stem}_v"]
            folded[stem] = torch._weight_norm(direction, value, 0)
            continue
        folded[key] = value
    return folded


def load_codec_state(path) -> dict:
    """The codec state dict from a local codec.pth, read with weights_only=True."""
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    if any("generator." in key for key in state):
        state = {
            key.replace("generator.", ""): value
            for key, value in state.items()
            if "generator." in key
        }
    state = {
        key: value
        for key, value in state.items()
        if not key.endswith(("freqs_cis", "causal_mask"))
    }
    return _fold_legacy_weight_norm(state)


def load_model(
    config_path,
    weights_path,
    codec_path,
    generation_config_path=None,
    device="cpu",
    dtype=torch.bfloat16,
) -> ArkttsModel:
    """An ArkttsModel from local files; the codec loads on first encode/decode."""
    from safetensors.torch import load_file

    config = ArkttsConfig.from_json_file(config_path)
    # built on meta so the 0.6B parameters are not randomly initialised only
    # to be overwritten; assign=True takes the loaded tensors as they are
    with torch.device("meta"):
        model = ArkttsModel(config, codec_path=codec_path)
    model.load_state_dict(load_file(str(weights_path)), strict=True, assign=True)
    model.reset_rope_buffers(device="cpu")
    if generation_config_path is not None:
        with open(generation_config_path, "r", encoding="utf-8") as handle:
            model.generation_config = json.load(handle)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


# --- codec (upstream modeling_arktts_codec.py) ------------------------------


def _rope(length: int, head_dim: int, base: float, device=None) -> Tensor:
    frequencies = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    phases = torch.outer(torch.arange(length, device=device), frequencies)
    values = torch.polar(torch.ones_like(phases), phases)
    return torch.stack((values.real, values.imag), dim=-1).to(torch.bfloat16)


def _apply_codec_rope(x: Tensor, values: Tensor) -> Tensor:
    shaped = x.float().reshape(*x.shape[:-1], -1, 2)
    values = values.view(1, shaped.shape[1], 1, shaped.shape[3], 2)
    output = torch.stack(
        (
            shaped[..., 0] * values[..., 0] - shaped[..., 1] * values[..., 1],
            shaped[..., 1] * values[..., 0] + shaped[..., 0] * values[..., 1],
        ),
        dim=-1,
    )
    return output.flatten(3).to(x.dtype)


class ArkttsCodecRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return output.to(x.dtype) * self.weight


class ArkttsCodecLayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-2, inplace: bool = False):
        super().__init__()
        self.inplace = bool(inplace)
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


@dataclass
class ArkttsCodecTransformerConfig:
    n_layer: int
    n_head: int
    dim: int
    intermediate_size: int
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    dropout_rate: float = 0.1
    attn_dropout_rate: float = 0.1
    channels_first: bool = True

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head


class ArkttsCodecAttention(nn.Module):
    def __init__(self, config: ArkttsCodecTransformerConfig):
        super().__init__()
        total = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)
        self.n_head = config.n_head
        self.n_local_heads = config.n_local_heads
        self.head_dim = config.head_dim
        self.attn_dropout_rate = config.attn_dropout_rate

    def forward(self, x, rope_values, mask):
        batch, length, _ = x.shape
        query_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        query, key, value = self.wqkv(x).split((query_size, kv_size, kv_size), dim=-1)
        query = query.view(batch, length, self.n_head, self.head_dim)
        key = key.view(batch, length, self.n_local_heads, self.head_dim)
        value = value.view(batch, length, self.n_local_heads, self.head_dim)
        query = _apply_codec_rope(query, rope_values).transpose(1, 2)
        key = _apply_codec_rope(key, rope_values).transpose(1, 2)
        value = value.transpose(1, 2)
        repeat = self.n_head // self.n_local_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.attn_dropout_rate if self.training else 0.0,
        )
        output = output.transpose(1, 2).contiguous().view(batch, length, query_size)
        return self.wo(output)


class ArkttsCodecFeedForward(nn.Module):
    def __init__(self, config: ArkttsCodecTransformerConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class ArkttsCodecTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ArkttsCodecAttention(config)
        self.feed_forward = ArkttsCodecFeedForward(config)
        self.ffn_norm = ArkttsCodecRMSNorm(config.dim, config.norm_eps)
        self.attention_norm = ArkttsCodecRMSNorm(config.dim, config.norm_eps)
        self.attention_layer_scale = ArkttsCodecLayerScale(config.dim, inplace=True)
        self.ffn_layer_scale = ArkttsCodecLayerScale(config.dim, inplace=True)

    def forward(self, x, rope_values, mask):
        hidden = x + self.attention_layer_scale(
            self.attention(self.attention_norm(x), rope_values, mask)
        )
        return hidden + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(hidden)))


class ArkttsCodecWindowTransformer(nn.Module):
    def __init__(
        self, config, input_dim: int, window_size: int | None, causal: bool = True
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [ArkttsCodecTransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.norm = ArkttsCodecRMSNorm(config.dim, config.norm_eps)
        self.window_size = window_size
        self.causal = causal
        self.channels_first = config.channels_first
        self.input_proj = (
            nn.Linear(input_dim, config.dim)
            if input_dim != config.dim
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(config.dim, input_dim)
            if input_dim != config.dim
            else nn.Identity()
        )
        self.look_ahead_conv = nn.Identity()
        self.head_dim = config.head_dim
        self.rope_base = config.rope_base

    def forward(self, x, x_lens=None):
        del x_lens
        if self.channels_first:
            x = x.transpose(1, 2)
        x = self.look_ahead_conv(self.input_proj(x))
        length = x.shape[1]
        row = torch.arange(length, device=x.device)[:, None]
        column = torch.arange(length, device=x.device)[None, :]
        mask = column <= row
        if self.window_size is not None:
            mask &= column >= (row - self.window_size + 1).clamp_min(0)
        mask = mask[None, None]
        rope_values = _rope(length, self.head_dim, self.rope_base, x.device)
        for layer in self.layers:
            x = layer(x, rope_values, mask)
        x = self.output_proj(self.norm(x))
        return x.transpose(1, 2) if self.channels_first else x


def _extra_padding(x, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    length = x.shape[-1]
    frames = (length - kernel_size + padding_total) / stride + 1
    ideal = (math.ceil(frames) - 1) * stride + kernel_size - padding_total
    return ideal - length


class ArkttsCausalConv1d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation=1, stride=1, groups=1
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.kernel_size - self.stride

    def forward(self, x):
        right = _extra_padding(x, self.kernel_size, self.stride, self.padding)
        return self.conv(F.pad(x, (self.padding, right))).contiguous()

    def apply_weight_norm(self):
        self.conv = weight_norm(self.conv)
        return self


class ArkttsCausalConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.conv(x)
        crop = self.kernel_size - self.stride
        return x[..., : x.shape[-1] - crop].contiguous() if crop else x.contiguous()

    def apply_weight_norm(self):
        self.conv = weight_norm(self.conv)
        return self


def _causal_wn_conv(*args, **kwargs):
    return ArkttsCausalConv1d(*args, **kwargs).apply_weight_norm()


def _causal_wn_transpose(*args, **kwargs):
    return ArkttsCausalConvTranspose1d(*args, **kwargs).apply_weight_norm()


def _arktts_snake(x: Tensor, alpha: Tensor) -> Tensor:
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    return x.reshape(shape)


class ArkttsSnake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return _arktts_snake(x, self.alpha)


class ArkttsResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            ArkttsSnake1d(dim),
            _causal_wn_conv(dim, dim, kernel_size=7, dilation=dilation),
            ArkttsSnake1d(dim),
            _causal_wn_conv(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        output = self.block(x)
        difference = x.shape[-1] - output.shape[-1]
        if difference > 0:
            x = x[..., :-difference]
        return x + output


class ArkttsEncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int, transformer_layers: int):
        super().__init__()
        modules = [
            ArkttsResidualUnit(dim // 2, 1),
            ArkttsResidualUnit(dim // 2, 3),
            ArkttsResidualUnit(dim // 2, 9),
            ArkttsSnake1d(dim // 2),
            _causal_wn_conv(dim // 2, dim, kernel_size=2 * stride, stride=stride),
        ]
        if transformer_layers:
            config = ArkttsCodecTransformerConfig(
                n_layer=transformer_layers,
                n_head=max(dim // 64, 1),
                dim=dim,
                intermediate_size=dim * 3,
            )
            modules.append(ArkttsCodecWindowTransformer(config, dim, window_size=512))
        else:
            modules.append(nn.Identity())
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x)


class ArkttsEncoder(nn.Module):
    def __init__(
        self, base_dim: int = 64, transformer_layers: int = 4, latent_dim: int = 1024
    ):
        super().__init__()
        dim = base_dim
        modules = [_causal_wn_conv(1, dim, kernel_size=7)]
        for stride, layers in zip((2, 4, 8, 8), (0, 0, 0, transformer_layers)):
            dim *= 2
            modules.append(ArkttsEncoderBlock(dim, stride, layers))
        modules.extend(
            (ArkttsSnake1d(dim), _causal_wn_conv(dim, latent_dim, kernel_size=3))
        )
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x)


class ArkttsDecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            ArkttsSnake1d(input_dim),
            _causal_wn_transpose(
                input_dim, output_dim, kernel_size=2 * stride, stride=stride
            ),
            ArkttsResidualUnit(output_dim, 1),
            ArkttsResidualUnit(output_dim, 3),
            ArkttsResidualUnit(output_dim, 9),
        )

    def forward(self, x):
        return self.block(x)


class ArkttsDecoder(nn.Module):
    def __init__(self, channels: int = 1536, latent_dim: int = 1024):
        super().__init__()
        modules = [_causal_wn_conv(latent_dim, channels, kernel_size=7)]
        for index, stride in enumerate((8, 8, 4, 2)):
            input_dim = channels // (2**index)
            output_dim = channels // (2 ** (index + 1))
            modules.append(ArkttsDecoderBlock(input_dim, output_dim, stride))
        modules.extend(
            (
                ArkttsSnake1d(output_dim),
                _causal_wn_conv(output_dim, 1, kernel_size=7),
                nn.Tanh(),
            )
        )
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class ArkttsVectorQuantizer(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        # upstream wraps both projections in the deprecated legacy
        # weight_norm; load_codec_state folds the checkpoint's g/v pairs
        self.in_proj = nn.Conv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = nn.Conv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def decode_code(self, indices):
        return F.embedding(indices, self.codebook.weight).transpose(1, 2)

    def decode_latents(self, latents):
        batch, _, length = latents.shape
        flattened = latents.transpose(1, 2).reshape(batch * length, -1)
        flattened = F.normalize(flattened)
        codebook = F.normalize(self.codebook.weight)
        distances = (
            flattened.pow(2).sum(1, keepdim=True)
            - 2 * flattened @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-distances).argmax(1).view(batch, length)
        return self.decode_code(indices), indices

    def forward(self, z):
        projected = self.in_proj(z)
        quantized, indices = self.decode_latents(projected)
        quantized_st = projected + (quantized - projected).detach()
        return self.out_proj(quantized_st), indices, projected


class ArkttsResidualQuantizer(nn.Module):
    def __init__(
        self, input_dim: int, n_codebooks: int, codebook_size: int, codebook_dim: int
    ):
        super().__init__()
        self.n_codebooks = int(n_codebooks)
        self.codebook_size = int(codebook_size)
        self.quantizers = nn.ModuleList(
            [
                ArkttsVectorQuantizer(input_dim, codebook_size, codebook_dim)
                for _ in range(n_codebooks)
            ]
        )

    def forward(self, z):
        quantized_sum = 0.0
        residual = z
        codes = []
        for quantizer in self.quantizers:
            quantized, indices, _ = quantizer(residual)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized
            codes.append(indices)
        return quantized_sum, torch.stack(codes, dim=1)

    def from_codes(self, codes):
        output = 0.0
        for index in range(codes.shape[1]):
            projected = self.quantizers[index].decode_code(codes[:, index])
            output = output + self.quantizers[index].out_proj(projected)
        return output


class ArkttsConvNeXtBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = ArkttsCausalConv1d(dim, dim, kernel_size=7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x):
        residual = x
        x = self.dwconv(x).permute(0, 2, 1)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        x = (self.gamma * x).permute(0, 2, 1)
        return residual + x


class ArkttsDownsampleQuantizer(nn.Module):
    def __init__(self, config: ArkttsConfig):
        super().__init__()
        latent = config.codec_latent_dim
        self.semantic_quantizer = ArkttsResidualQuantizer(
            latent, 1, config.codebook_size, config.codec_codebook_dim
        )
        self.quantizer = ArkttsResidualQuantizer(
            latent,
            config.num_codebooks - 1,
            config.codec_residual_codebook_size,
            config.codec_codebook_dim,
        )
        self.downsample = nn.Sequential(
            nn.Sequential(
                ArkttsCausalConv1d(latent, latent, kernel_size=2, stride=2),
                ArkttsConvNeXtBlock(latent),
            ),
            nn.Sequential(
                ArkttsCausalConv1d(latent, latent, kernel_size=2, stride=2),
                ArkttsConvNeXtBlock(latent),
            ),
        )
        self.upsample = nn.Sequential(
            nn.Sequential(
                ArkttsCausalConvTranspose1d(latent, latent, kernel_size=2, stride=2),
                ArkttsConvNeXtBlock(latent),
            ),
            nn.Sequential(
                ArkttsCausalConvTranspose1d(latent, latent, kernel_size=2, stride=2),
                ArkttsConvNeXtBlock(latent),
            ),
        )
        pre_transformer_config = ArkttsCodecTransformerConfig(
            n_layer=config.codec_pre_n_layer,
            n_head=config.codec_pre_n_head,
            dim=latent,
            intermediate_size=config.codec_pre_intermediate_size,
        )
        post_transformer_config = ArkttsCodecTransformerConfig(
            n_layer=config.codec_post_n_layer,
            n_head=config.codec_post_n_head,
            n_local_heads=config.codec_post_n_local_heads,
            dim=latent,
            intermediate_size=config.codec_post_intermediate_size,
        )
        self.pre_module = ArkttsCodecWindowTransformer(
            pre_transformer_config, latent, window_size=128
        )
        self.post_module = ArkttsCodecWindowTransformer(
            post_transformer_config, latent, window_size=128
        )
        self.semantic_predictor_module = nn.Identity()

    def forward(self, z):
        original_length = z.shape[-1]
        z = self.pre_module(self.downsample(z))
        semantic, semantic_codes = self.semantic_quantizer(z)
        residual, residual_codes = self.quantizer(z - semantic)
        z = self.upsample(self.post_module(semantic + residual))
        difference = original_length - z.shape[-1]
        if difference > 0:
            z = F.pad(z, (difference, 0))
        elif difference < 0:
            z = z[..., -difference:]
        return z, torch.cat((semantic_codes, residual_codes), dim=1)

    def decode(self, indices):
        indices = indices.clone()
        indices[:, 0].clamp_(0, self.semantic_quantizer.codebook_size - 1)
        indices[:, 1:].clamp_(0, self.quantizer.codebook_size - 1)
        semantic = self.semantic_quantizer.from_codes(indices[:, :1])
        residual = self.quantizer.from_codes(indices[:, 1:])
        return self.upsample(self.post_module(semantic + residual))


class ArkttsCodec(nn.Module):
    sample_rate = 44100
    hop_length = 512
    frame_length = 2048

    def __init__(self, config: ArkttsConfig | None = None):
        super().__init__()
        config = config or ArkttsConfig()
        self.encoder = ArkttsEncoder(
            config.codec_encoder_dim,
            config.codec_encoder_transformer_layers,
            config.codec_latent_dim,
        )
        self.quantizer = ArkttsDownsampleQuantizer(config)
        self.decoder = ArkttsDecoder(config.codec_decoder_dim, config.codec_latent_dim)

    @torch.inference_mode()
    def encode(self, audio, audio_lengths=None):
        if audio.ndim == 2:
            audio = audio[:, None]
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise ValueError("audio must have shape [B, 1, samples]")
        original_length = audio.shape[-1]
        right = (
            math.ceil(original_length / self.frame_length) * self.frame_length
            - original_length
        )
        if right:
            audio = F.pad(audio, (0, right))
        if audio_lengths is None:
            audio_lengths = torch.full(
                (audio.shape[0],),
                original_length,
                device=audio.device,
                dtype=torch.long,
            )
        encoded = self.encoder(audio)
        _, codes = self.quantizer(encoded)
        code_lengths = torch.ceil(audio_lengths.float() / self.frame_length).long()
        max_codes = codes.shape[-1]
        padded = torch.full_like(codes, -1)
        for index, length in enumerate(code_lengths.tolist()):
            padded[index, :, : min(length, max_codes)] = codes[
                index, :, : min(length, max_codes)
            ]
        return padded, code_lengths.clamp_max(max_codes)

    @torch.inference_mode()
    def decode(self, codes):
        return self.decoder(self.quantizer.decode(codes.long()))


# --- processor (upstream processing_arktts.py) ------------------------------


def load_tokenizer(path):
    """The model's stock fast tokenizer, from a local tokenizer.json."""
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(tokenizer_file=str(path))


def _clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _as_list(value: Any, batch_size: int, name: str) -> list[Any]:
    if isinstance(value, (str, Path)) or value is None or np.isscalar(value):
        return [value] * batch_size
    if isinstance(value, torch.Tensor) and value.ndim <= 2:
        return [value] if batch_size == 1 else list(value)
    if isinstance(value, np.ndarray) and value.ndim <= 2:
        return [value] if batch_size == 1 else list(value)
    values = list(value)
    if len(values) != batch_size:
        raise ValueError(f"{name} must contain {batch_size} items, got {len(values)}")
    return values


def _pad_1d(rows: list[Tensor], pad_value: int) -> tuple[Tensor, Tensor]:
    max_len = max((row.numel() for row in rows), default=0)
    values = torch.full((len(rows), max_len), pad_value, dtype=torch.long)
    mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    for idx, row in enumerate(rows):
        length = row.numel()
        values[idx, :length] = row
        mask[idx, :length] = 1
    return values, mask


class ArkttsProcessor:
    """Builds generate()'s prompt inputs from text and an optional reference voice.

    The tokenizer is anything with encode(text, add_special_tokens=False)
    returning token ids - load_tokenizer() gives the model's own.
    """

    def __init__(
        self,
        tokenizer,
        pad_token_id: int = 151643,
        num_codebooks: int = 10,
        semantic_begin_id: int = 151678,
        audio_sampling_rate: int = 44100,
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = int(pad_token_id)
        self.num_codebooks = int(num_codebooks)
        self.semantic_begin_id = int(semantic_begin_id)
        self.audio_sampling_rate = int(audio_sampling_rate)

    @classmethod
    def for_model(cls, tokenizer, config: ArkttsConfig) -> "ArkttsProcessor":
        return cls(
            tokenizer,
            pad_token_id=config.pad_token_id,
            num_codebooks=config.num_codebooks,
            semantic_begin_id=config.semantic_begin_id,
            audio_sampling_rate=config.codec_sample_rate,
        )

    def _encode(self, text: str) -> Tensor:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(ids, dtype=torch.long)

    @staticmethod
    def _format_reference_text(text: str) -> str:
        cleaned = _clean_text(text)
        if re.search(r"<\|speaker:\d+\|>", cleaned):
            return cleaned
        return f"<|speaker:0|>{cleaned}"

    def _prompt_segments(
        self, text: str, reference_text: str | None, has_reference: bool
    ):
        target = _clean_text(text)
        if not target:
            raise ValueError("text must not be empty")

        def encode_parts(parts: list[str]) -> Tensor:
            return torch.cat([self._encode(part) for part in parts])

        if not has_reference:
            full = encode_parts(
                [
                    "<|im_start|>system\n",
                    "convert the provided text to speech",
                    "<|im_end|>\n",
                    "<|im_start|>user\n",
                    target,
                    "<|im_end|>\n",
                    "<|im_start|>assistant\n<|voice|>",
                ]
            )
            return full, self._encode("")
        if not reference_text:
            raise ValueError(
                "reference_text is required when a reference voice is provided"
            )
        prefix = encode_parts(
            [
                "<|im_start|>system\n",
                "convert the provided text to speech reference to the following:"
                "\n\nText:\n",
                self._format_reference_text(reference_text),
                "\n\nSpeech:\n",
            ]
        )
        suffix = encode_parts(
            [
                "<|im_end|>\n",
                "<|im_start|>user\n",
                target,
                "<|im_end|>\n",
                "<|im_start|>assistant\n<|voice|>",
            ]
        )
        return prefix, suffix

    def _load_audio(
        self, value: Any, sampling_rate: int | None, base_dir=None
    ) -> Tensor:
        source_rate = sampling_rate
        if isinstance(value, (str, Path)):
            from .audio_utils import load_audio

            if str(value).lower().startswith(("http://", "https://")):
                # load_audio would fetch it; this module reads local files only
                raise ValueError("reference audio must be a local file, not a URL")

            array, source_rate = load_audio(str(value), base_dir=base_dir)
            audio = torch.from_numpy(np.asarray(array, dtype=np.float32).mean(axis=0))
        else:
            if isinstance(value, dict):
                source_rate = value.get("sampling_rate", source_rate)
                value = value.get("array")
            if (
                isinstance(value, (tuple, list))
                and len(value) == 2
                and np.isscalar(value[1])
            ):
                value, source_rate = value
            audio = torch.as_tensor(np.asarray(value), dtype=torch.float32)
            if audio.ndim == 2:
                audio = audio.mean(dim=0)
            if audio.ndim != 1:
                raise ValueError(
                    "reference audio must be mono or channels-first, got "
                    f"{tuple(audio.shape)}"
                )
        if audio.numel() == 0:
            raise ValueError("reference audio must not be empty")
        if source_rate is None:
            raise ValueError("sampling_rate is required for reference audio arrays")
        if int(source_rate) != self.audio_sampling_rate:
            from ..dsp import resample_waveform

            resampled = resample_waveform(
                audio.numpy()[None, :], int(source_rate), self.audio_sampling_rate
            )
            audio = torch.from_numpy(np.asarray(resampled, dtype=np.float32)[0])
        return audio.contiguous()

    def __call__(
        self,
        text: str | Sequence[str],
        reference_text: str | Sequence[str] | None = None,
        reference_audio: Any = None,
        sampling_rate: int | Sequence[int] | None = None,
        base_dir=None,
    ) -> dict[str, Tensor]:
        """generate()'s keyword inputs, as tensors.

        reference_audio is a path (read with audio_utils.load_audio, checked
        against base_dir), a (samples,) or (channels, samples) array with
        sampling_rate, an (array, rate) pair or a {"array", "sampling_rate"}
        dict; it is mixed to mono and resampled to the codec's rate.
        """
        texts = [text] if isinstance(text, str) else list(text)
        if not texts:
            raise ValueError("text batch must not be empty")
        batch_size = len(texts)
        ref_texts = _as_list(reference_text, batch_size, "reference_text")
        has_reference = reference_audio is not None

        prefix_rows, suffix_rows = zip(
            *[
                self._prompt_segments(item, ref_texts[idx], has_reference)
                for idx, item in enumerate(texts)
            ]
        )
        prefix_ids, prefix_mask = _pad_1d(list(prefix_rows), self.pad_token_id)
        suffix_ids, suffix_mask = _pad_1d(list(suffix_rows), self.pad_token_id)
        data: dict[str, Tensor] = {
            "prefix_input_ids": prefix_ids,
            "prefix_attention_mask": prefix_mask,
            "suffix_input_ids": suffix_ids,
            "suffix_attention_mask": suffix_mask,
        }

        if reference_audio is not None:
            audio_items = _as_list(reference_audio, batch_size, "reference_audio")
            rate_items = _as_list(sampling_rate, batch_size, "sampling_rate")
            loaded_audio = [
                self._load_audio(item, rate_items[idx], base_dir)
                for idx, item in enumerate(audio_items)
            ]
            max_samples = max(item.numel() for item in loaded_audio)
            padded_audio = torch.zeros(
                (batch_size, 1, max_samples), dtype=torch.float32
            )
            lengths = torch.empty(batch_size, dtype=torch.long)
            for idx, audio in enumerate(loaded_audio):
                lengths[idx] = audio.numel()
                padded_audio[idx, 0, : audio.numel()] = audio
            data["reference_audio_values"] = padded_audio
            data["reference_audio_lengths"] = lengths

        return data


__all__ = [
    "CODEC_FILE",
    "CONFIG_FILE",
    "GENERATION_CONFIG_FILE",
    "MODEL_FILES",
    "REPO_ID",
    "REVISION",
    "TOKENIZER_FILE",
    "WEIGHTS_FILE",
    "ArkttsCodec",
    "ArkttsConfig",
    "ArkttsGenerateOutput",
    "ArkttsModel",
    "ArkttsModelOutput",
    "ArkttsProcessor",
    "load_codec_state",
    "load_model",
    "load_tokenizer",
]
