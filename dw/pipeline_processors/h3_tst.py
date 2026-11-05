"""MiniMax-H3 Temporal State Transport (TST, #1196; #1199 Task 19): EXPERIMENTAL.

A spike, not a public surface. `experimental_tst: {"tau": 0.2, "per_head":
true}` on an H3 call puts `TstH3AttnProcessor` on every transformer block for
that one call, and takes it off when the call ends (`tst_call`). Off unless
`DW_EXPERIMENTAL_REFINE_WINDOWS=1` (this plan's one scratch flag) is set where
the step runs; without it the argument is refused at validate time
(`tst_errors`) and at the call (`tst_settings`).

The method. TST (Temporal State Transport in Video Generation, arXiv
2609.08505) reads each block's *temporal* attention - at every spatial cell of
the target video, the softmax of that cell's query against the same cell's key
in every latent frame, a frames x frames matrix - and from it two numbers:

- `alpha`, the share of attention that leaves its own frame (1 - mean diagonal);
- `E`, the von Neumann entropy of the mean normalized Gram matrix of those
  matrices, over log(frames).

`q = alpha * E * (1 - E)`, and the block's attention output is scaled by
`1 + tau * layer_weight * step_weight * q`, with a cosine ramp up the layers
(0 at the first block, 1 at the last) and down the steps (1 at the first, 0 at
the last). The formula, the schedules and their constants are ported from the
reference code, lytang63/temporal-state-transport `tst/core.py`
(`tst_modulation`), MIT licence, Copyright (c) 2025 TST Authors; the port is
vectorized and chunked, and tests/test_h3_tst.py checks it against a near-
verbatim copy. What the released code does, read off it rather than the README:

- `q` is in [0, 1/4] and never negative, so the correction only ever amplifies
  (at tau 0.2 a gain of at most 1.05). The README's signed "spectral tension"
  that softens fragmented heads is not in the released code; the SelfLift node
  that describes one has no licence and is not reproduced here.
- It is one number per layer, averaged over every head. `per_head: true` (the
  default here, the plan's "per-head correction") evaluates the same formula on
  each head alone and scales each head's output by its own gain; `false` is the
  reference's single gain.
- It needs only the query and key, never the attention output, so it costs a
  frames x frames attention per spatial cell (about T / rows of the full one)
  plus one small eigendecomposition per group per block. No calibration pass and
  no stored statistics: everything is measured on the forward it corrects.

Where dw departs from the reference. The reference scales every row of Wan's
self-attention, all video; H3 packs text and audio into the same attention, so
only the target video's rows are scaled (the trailing video run, as
dw/sparse_attention.py reads it). H3's output projection has no bias, so a gain
before it is the reference's gain after it. The step index is the count of
distinct target timesteps the call has seen, not the reference's attention-call
counter (which assumes two forwards per step); `num_inference_steps` sets the
step schedule.

Dense only, for this round. The statistic is operator-independent - it is read
from Q and K - so it could sit on the sparse processor too, but both want to own
the block's processor, and the plan asks for dense first. A step whose
transformer runs dw's 'sparse' backend is refused (validate and run); any
diffusers backend is fine, since the processor dispatches through the original
processor's own `_attention_backend`. Nothing here calls
`set_attention_backend` or touches diffusers' process-wide backend registry
(the a9a1298c leak): the processors and hooks are the only state, and they come
off with the call.

Under torch.compile the step index is a Python int the compiled forward
specialises on, so each step recompiles, up to the compile cache limit.

Imports diffusers only inside the processor's call, so validation can import
this module.

The ported formula and schedules (lytang63/temporal-state-transport):

    MIT License

    Copyright (c) 2025 TST Authors

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
"""

import logging
import math
import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from numbers import Real

import torch

from .. import references
from ..events import emit_log
from ..for_each import render_path
from .h3_rules import REFINE_WINDOWS_ENV_VAR, not_h3

logger = logging.getLogger("dw")

TST_INPUT = "experimental_tst"
TST_KEYS = ("tau", "per_head")
SPARSE = "sparse"
_H3_CLASS = "MiniMaxH3Transformer3DModel"
_EPS = 1e-8
# Elements of one frames x frames float32 chunk (spatial cells x heads x T x T)
# before the statistic splits the cells into chunks: about 64 MiB per buffer
_CHUNK_ELEMENTS = 16 * 1024 * 1024
# A gain this far above 1 counts as a corrected head in the call's report
_CORRECTED = 1e-3


@dataclass(frozen=True)
class TstSettings:
    tau: float
    per_head: bool = True


def tst_enabled():
    return os.environ.get(REFINE_WINDOWS_ENV_VAR) == "1"


def _flag_problem():
    return (
        f"{TST_INPUT} is an experimental spike (#1196), off unless "
        f"{REFINE_WINDOWS_ENV_VAR}=1 is set where the server runs"
    )


def tst_value_problem(value):
    """Why `value` cannot be an `experimental_tst`, or None. An unresolved
    reference, anywhere in it, is left to the run."""
    if not isinstance(value, Mapping):
        return (
            f"{TST_INPUT} is an object {{'tau': 0.2, 'per_head': true}}, got {value!r}"
        )
    unknown = sorted(set(value) - set(TST_KEYS))
    if unknown:
        return f"{TST_INPUT} takes only {', '.join(TST_KEYS)}, not {', '.join(unknown)}"
    tau = value.get("tau")
    if isinstance(tau, str) and references.is_ref(references.UNRESOLVED, tau):
        tau = 0.0
    if (
        isinstance(tau, bool)
        or not isinstance(tau, Real)
        or not 0 <= tau <= 1  # NaN fails it too
    ):
        return (
            f"{TST_INPUT} needs 'tau', the correction strength, a number in "
            f"[0, 1] - 0.2 is the reference's, 0 measures only - got {tau!r}"
        )
    per_head = value.get("per_head", True)
    if not isinstance(per_head, bool) and not (
        isinstance(per_head, str) and references.is_ref(references.UNRESOLVED, per_head)
    ):
        return f"{TST_INPUT} 'per_head' must be true or false, got {per_head!r}"
    return None


def tst_settings(value):
    """The call's settings, or None when the argument is absent, null or false.

    Raises:
        ValueError: If the flag is off, or the value is not a TST object
    """
    if value is None or value is False:
        return None
    if not tst_enabled():
        raise ValueError(_flag_problem())
    problem = tst_value_problem(value)
    if problem:
        raise ValueError(problem)
    return TstSettings(float(value["tau"]), bool(value.get("per_head", True)))


def _sparse_configured(configuration):
    """Whether a pipeline configuration runs dw's 'sparse' on the transformer:
    pipeline level, or the transformer component's own pin."""
    if not isinstance(configuration, Mapping):
        return False
    if configuration.get("attention_backend") == SPARSE:
        return True
    components = configuration.get("components")
    transformer = (
        components.get("transformer") if isinstance(components, Mapping) else None
    )
    return (
        isinstance(transformer, Mapping)
        and transformer.get("attention_backend") == SPARSE
    )


def _dense_problem():
    return (
        f"{TST_INPUT} runs on dense attention for this round (#1196), and this "
        f"step's transformer runs attention_backend 'sparse' - set it to null"
    )


def tst_errors(workflow_definition, source_indices=None):
    """Every `experimental_tst` refused before the run, as [{path, message}]:
    the flag off, a step off MiniMax-H3, a value that is no TST object, or a
    transformer on dw's 'sparse' backend."""
    errors = []
    for _, _, pipeline, source, where in references.iter_steps(
        workflow_definition.get("steps"), source_indices, "pipeline"
    ):
        arguments = pipeline.get("arguments")
        if not isinstance(arguments, dict):
            continue
        value = arguments.get(TST_INPUT)
        if value is None or value is False:
            continue
        path = render_path(("steps", source, "pipeline", "arguments", TST_INPUT))
        if isinstance(value, str) and references.is_ref(references.UNRESOLVED, value):
            problem = None if tst_enabled() else _flag_problem()
        else:
            problem = (
                (None if tst_enabled() else _flag_problem())
                or not_h3(pipeline, TST_INPUT)
                or tst_value_problem(value)
                or (
                    _dense_problem()
                    if _sparse_configured(pipeline.get("configuration"))
                    else None
                )
            )
        if problem:
            errors.append({"path": path, "message": f"{problem}{where}"})
    return errors


def _h3_transformers(pipeline):
    from ..sparse_attention import named_pipeline_components

    return [
        (name, component)
        for name, component in named_pipeline_components(pipeline)
        if type(component).__name__ == _H3_CLASS
    ]


def tst_refusal(pipeline, configuration=None):
    """Why this call cannot run TST, or None: no H3 transformer, or one that
    runs dw's 'sparse' processor (installed, or configured for this call)."""
    from ..sparse_attention import is_sparse_installed

    transformers = _h3_transformers(pipeline)
    if not transformers:
        return f"{TST_INPUT} found no MiniMax-H3 transformer on this pipeline"
    if _sparse_configured(configuration) or any(
        is_sparse_installed(component) for _, component in transformers
    ):
        return _dense_problem()
    return None


# 1. The schedules and the statistic - pure


def layer_weight(layer_index, total_layers):
    """The reference's layer ramp: 0 at the first block, 1 at the last."""
    if total_layers <= 1:
        return 1.0
    return 0.5 - 0.5 * math.cos(math.pi * layer_index / (total_layers - 1))


def step_weight(step_index, total_steps):
    """The reference's step ramp: 1 at the first step, 0 at the last; a step
    past the plan is the last."""
    if total_steps <= 1:
        return 1.0
    step_index = min(step_index, total_steps - 1)
    return 0.5 + 0.5 * math.cos(math.pi * step_index / (total_steps - 1))


def temporal_quality(query, key, per_head=True, chunk=None):
    """TST's `q = alpha * E * (1 - E)` from temporal query and key.

    `query`, `key`: `[cells, heads, frames, head_dim]` - one row per spatial
    cell (and batch entry), the cell's query in every latent frame. Returns
    `[heads]` with `per_head`, else `[1]` (the reference: every head pooled).
    The frames x frames attention is taken `chunk` cells at a time, so its
    float32 buffers stay bounded; every pooled statistic is a mean, so the
    chunks sum exactly.
    """
    cells, heads, frames, dim = query.shape
    groups = heads if per_head else 1
    if frames < 2:
        return torch.zeros(groups, device=query.device)
    if chunk is None:
        chunk = max(1, _CHUNK_ELEMENTS // max(1, heads * frames * frames))
    scale = dim**-0.5
    attn_sum = torch.zeros(heads, frames, frames, device=query.device)
    rho_sum = torch.zeros(heads, frames, frames, device=query.device)
    for start in range(0, cells, chunk):
        q = query[start : start + chunk].float() * scale
        k = key[start : start + chunk].float()
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)  # [c, H, T, T]
        attn_sum += attn.sum(0)
        gram = attn @ attn.transpose(-2, -1)
        trace = torch.diagonal(gram, dim1=-2, dim2=-1).sum(-1).clamp(min=_EPS)
        rho_sum += (gram / trace[..., None, None]).sum(0)
    if not per_head:
        attn_sum = attn_sum.sum(0, keepdim=True)
        rho_sum = rho_sum.sum(0, keepdim=True)
    pooled = cells * (heads // groups)
    mean_attn = attn_sum / pooled
    rho = rho_sum / pooled
    rho = (
        rho
        / torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1).clamp(min=_EPS)[:, None, None]
    )
    eigvals = torch.linalg.eigvalsh(rho).clamp(min=_EPS)
    eigvals = eigvals / eigvals.sum(-1, keepdim=True).clamp(min=_EPS)
    entropy = -(eigvals * eigvals.log()).sum(-1)
    log_f = max(math.log(frames), _EPS)
    entropy_norm = (entropy / log_f).clamp(0.0, 1.0)
    diag_mean = torch.diagonal(mean_attn, dim1=-2, dim2=-1).mean(-1)
    alpha = (1.0 - diag_mean).clamp(0.0, 1.0)
    # The reference also computes a row entropy and never uses it; it is left out
    return alpha * entropy_norm * (1.0 - entropy_norm)


def scheduled_gain(quality, tau, layer_index, total_layers, step_index, total_steps):
    """`1 + tau * layer_weight * step_weight * q`, one gain per group."""
    effective = (
        tau
        * layer_weight(layer_index, total_layers)
        * step_weight(step_index, total_steps)
    )
    return 1.0 + effective * quality


# 2. The per-call state, kept current by a forward pre-hook


def _ranked(values):
    return torch.unique(values, return_inverse=True)[1]


@dataclass
class _TstState:
    """One installed transformer's settings, the current forward's target
    video layout, and the call's running report."""

    settings: TstSettings
    total_steps: int
    step: int = -1
    last_timestep: float | None = None
    active: bool = False
    order: torch.Tensor | None = None
    grid: tuple | None = None
    layout: tuple | None = None
    last_gain: torch.Tensor | None = None
    forwards: int = 0
    gains: list = field(default_factory=list)
    handles: list = field(default_factory=list)
    originals: list = field(default_factory=list)

    def observe(self, position_ids, token_tags, timestep, timestep_indices):
        """Read one forward's layout: the target video rows in (t, h, w)
        order, and the step its target timestep is.

        Raises:
            ValueError: If the target video is not a full (t, h, w) grid
        """
        layout = self.layout
        if not (
            layout is not None and layout[0] is position_ids and layout[1] is token_tags
        ):
            self.order, self.grid = _target_order(position_ids, token_tags)
            self.layout = (position_ids, token_tags)
        first = int(self.order[0])
        value = float(timestep[int(timestep_indices[first])])
        if value != self.last_timestep:
            self.step += 1
            self.last_timestep = value
        self.forwards += 1


def _target_order(position_ids, token_tags):
    """The target video's packed rows ordered (t, h, w), and (T, H, W).

    The target is the trailing run of video rows (dw/sparse_attention.py's
    rule); H3's rotary coordinates are fractional, so each axis is ranked."""
    tags = token_tags.cpu()
    pos = position_ids.cpu()
    non_video = (tags != 0).nonzero()
    start = int(non_video[-1]) + 1 if len(non_video) else 0
    rows = torch.arange(start, tags.numel())
    if not len(rows) or pos.shape != (tags.numel(), 3):
        raise ValueError(f"{TST_INPUT}: no target video rows to read a grid from")
    ranks = torch.stack([_ranked(pos[rows, a]) for a in range(3)], dim=1)
    t, h, w = (ranks.max(0).values + 1).tolist()
    if t * h * w != len(rows):
        raise ValueError(
            f"{TST_INPUT}: the target video is not a full grid "
            f"({t}x{h}x{w} != {len(rows)} rows)"
        )
    key = (ranks[:, 0] * h + ranks[:, 1]) * w + ranks[:, 2]
    if torch.unique(key).numel() != len(rows):
        raise ValueError(f"{TST_INPUT}: two target rows share a grid cell")
    order = rows[torch.argsort(key)]
    return order.to(position_ids.device), (t, h, w)


_FORWARD_ARGS = (
    "hidden_states",
    "audio_hidden_states",
    "encoder_hidden_states",
    "timestep",
    "timestep_indices",
    "token_tags",
    "position_ids",
)


def _forward_arg(args, kwargs, name):
    if name in kwargs:
        return kwargs[name]
    index = _FORWARD_ARGS.index(name)
    return args[index] if index < len(args) else None


def _pre_hook(component, args, kwargs):
    state = component._dw_tst_state
    state.observe(
        _forward_arg(args, kwargs, "position_ids"),
        _forward_arg(args, kwargs, "token_tags"),
        _forward_arg(args, kwargs, "timestep"),
        _forward_arg(args, kwargs, "timestep_indices"),
    )
    state.active = True
    return None


def _post_hook(component, args, output):
    component._dw_tst_state.active = False
    return None


# 3. The processor


class TstH3AttnProcessor:
    """The stock H3 attention with TST's gain on the target video's head
    outputs. The original processor's backend and parallel config are the
    ones it dispatches with, and setting them sets the original's."""

    def __init__(self, original, state, layer_index, total_layers):
        self._original = original
        self._state = state
        self.layer_index = layer_index
        self.total_layers = total_layers

    @property
    def _attention_backend(self):
        return getattr(self._original, "_attention_backend", None)

    @_attention_backend.setter
    def _attention_backend(self, value):
        self._original._attention_backend = value

    @property
    def _parallel_config(self):
        return getattr(self._original, "_parallel_config", None)

    @_parallel_config.setter
    def _parallel_config(self, value):
        self._original._parallel_config = value

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        state = self._state
        if state.order is None:
            return self._original(attn, hidden_states, rotary_emb, attention_mask)

        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_minimax_h3 import (
            _apply_rotary_emb,
        )

        # The stock processor's path, op for op (MiniMaxH3AttnProcessor)
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        settings = state.settings
        order = state.order.to(query.device)
        t, h, w = state.grid
        b, _, heads, dim = query.shape

        def temporal(x):
            # [B, T*H*W, N, D] -> [B*H*W, N, T, D]
            x = x[:, order].view(b, t, h * w, heads, dim)
            return x.permute(0, 2, 3, 1, 4).reshape(b * h * w, heads, t, dim)

        quality = temporal_quality(
            temporal(query), temporal(key), per_head=settings.per_head
        )
        gain = scheduled_gain(
            quality,
            settings.tau,
            self.layer_index,
            self.total_layers,
            state.step,
            state.total_steps,
        )
        state.last_gain = gain.detach()
        state.gains.append(gain.detach().cpu())

        out = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        if settings.tau != 0:
            # [N] (or [1]) gains over the target rows' heads; the bias-free
            # output projection makes this the reference's gain on the output
            target = out[:, order]
            scaled = (target * gain.to(target.device)[None, None, :, None]).type_as(
                target
            )
            out = out.index_copy(1, order, scaled)
        hidden_states = out.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


# 4. One call's install


def _install(component, settings, total_steps):
    state = _TstState(settings=settings, total_steps=max(1, int(total_steps or 1)))
    # First, so a raise partway through still leaves `_uninstall` its record
    component._dw_tst_state = state
    blocks = list(component.transformer_blocks)
    for index, block in enumerate(blocks):
        state.originals.append(block.attn.processor)
        block.attn.set_processor(
            TstH3AttnProcessor(block.attn.processor, state, index, len(blocks))
        )
    state.handles = [
        component.register_forward_pre_hook(_pre_hook, with_kwargs=True),
        component.register_forward_hook(_post_hook, always_call=True),
    ]
    return state


def _uninstall(component):
    state = getattr(component, "_dw_tst_state", None)
    if state is None:
        return
    for block, original in zip(component.transformer_blocks, state.originals):
        block.attn.set_processor(original)
    for handle in state.handles:
        handle.remove()
    del component._dw_tst_state


def _report(label, settings, states):
    gains = [g for state in states for g in state.gains]
    forwards = sum(state.forwards for state in states)
    if not gains:
        mean_gain = max_gain = 1.0
        corrected = 0.0
    else:
        stacked = torch.cat([g.flatten().float() for g in gains])
        mean_gain = float(stacked.mean())
        max_gain = float(stacked.max())
        corrected = float((stacked > 1 + _CORRECTED).float().mean())
    mode = "per head" if settings.per_head else "per layer"
    message = (
        f"{label}: TST (experimental, #1196) tau {settings.tau:g} {mode}: "
        f"{forwards} forwards, gain mean {mean_gain:.4f} max {max_gain:.4f}, "
        f"{corrected:.0%} of head gains above {1 + _CORRECTED:g}"
    )
    logger.info(message)
    emit_log(
        message,
        tau=settings.tau,
        per_head=settings.per_head,
        forwards=forwards,
        mean_gain=mean_gain,
        max_gain=max_gain,
        corrected_share=corrected,
    )


# H3's `num_inference_steps` default (diffusers' input template), used when a
# call names none
DEFAULT_STEPS = 50


def _evaluations(num_inference_steps):
    """The transformer evaluations a call runs. H3's `num_inference_steps` N
    counts sigma grid points (MiniMaxH3Scheduler), so it drives N - 1; a call
    that names none (or not a whole number) runs the default."""
    steps = num_inference_steps
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        steps = DEFAULT_STEPS
    return steps - 1


@contextmanager
def tst_call(pipeline, settings, num_inference_steps, label):
    """Put the TST processor on every block of the pipeline's H3 transformers
    for one call, and the originals back when it ends - also when it raises.
    The call's gains are reported once, when it completes. The step ramp
    spans the call's N - 1 evaluations (`_evaluations`).

    Under torch.compile the step index is a Python int the compiled forward
    specialises on, so each step recompiles - up to the compile cache limit.

    Raises:
        ValueError: If the pipeline has no H3 transformer or runs sparse
    """
    refusal = tst_refusal(pipeline)
    if refusal:
        raise ValueError(f"{label}: {refusal}")
    installed = []
    try:
        for _, component in _h3_transformers(pipeline):
            installed.append(component)
            _install(component, settings, _evaluations(num_inference_steps))
        states = [component._dw_tst_state for component in installed]
        yield states
    finally:
        for component in installed:
            _uninstall(component)
    _report(label, settings, states)
