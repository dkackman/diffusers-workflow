"""MiniMax-H3 keystone attention re-split (#1391 stage A, #1393): EXPERIMENTAL.

A spike, not a public surface. `experimental_keystone_attention: {"balance":
20, "blocks": "31-33,36-44,47"}` on an H3 fl2va call that passes `image` and
`guides` re-splits, on the named transformer blocks, the attention every query after
the guide rows (the target; the held condition rows' own queries are left
alone, nor the target's first latent frame under them, #1494) pays to the call's condition rows between the keyframe rows K (laid out from
`image`) and the guide rows C (from `guides`). Off unless
`DW_EXPERIMENTAL_REFINE_WINDOWS=1` is set where the step runs; without it the
argument is refused at validate time (`keystone_attention_errors`) and at the
call (`keystone_attention_settings`).

The method. For one query, the attention it pays to K is pK = exp(lse_K - lse)
and to C is pC = exp(lse_C - lse), where lse is the log-sum-exp of its scores
over every key and lse_K, lse_C over the K and C keys alone. Their sum S is
kept; the balance b moves it toward K:

    pK' = S * b * pK / (b * pK + pC),   pC' = S - pK'
    O'  = O + (pK' - pK) * O_K + (pC' - pC) * O_C

with O the full attention output and O_K, O_C the outputs of attention over K
alone and C alone. Every other key's share is untouched, b = 1 is the identity
(to float rounding), and a query that pays K and C nothing is unchanged. The
concepts - re-weighting the attention a target pays to a keystone against the
rest of its conditioning, on a subset of blocks - are from NerdNotes' tests
19-20 and geiru_context_suite's Keystone Attention, and ckinpdx's Carry
Attention (docs/ATTRIBUTIONS.md); no code from any of them is used here.

The blocks are numbered 0..N-1 down diffusers' `transformer_blocks`, N read off
the loaded transformer; the default subset is NerdNotes' (ComfyUI's numbering,
taken as diffusers' - #1391 Q3). Unselected blocks keep H3's own processor.

The passes need each one's log-sum-exp, which `dispatch_attention_fn` does not
return, so the selected blocks run a torch-native op that does: native flash
or cuDNN on CUDA, in the order SDPA itself tries them (so the main pass runs
the kernel H3's dense processor runs); the CPU flash kernel on the CPU (the
unit tests). With
neither, the call is refused when the first selected block runs, naming why.

The rows. The guide layout step (h3_guide_steps.py) publishes K and C as packed
row ranges onto the installed state; before each forward the rows are checked
to be video condition rows - one condition timestep, ahead of the trailing
target video - and the forward raises rather than guess when they are not, or
were never published. Guide audio rows are not part of C.

Per forward (#1188 S4). The hybrid windows (h3_windows.py) install the same
processor with `per_forward=True` around their ordered phase and publish K and
C for each forward that re-splits (`publish_window_rows`, `clear_rows` after
it): K is the take's keystone rows, which lead the window's media, and C the
window's pinned overlap, the leading rows of its trailing video - so audio may
sit between them, and C is inside the target, not ahead of it. A forward with
no rows published runs the original processor. `check_window_rows` is that
layout's check, run before every forward that has rows.

Imports diffusers only inside the processor's call, so validation can import
this module.
"""

import functools
import logging
import math
import os
import re
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from numbers import Real

import torch

from .. import references
from ..events import emit_log
from ..for_each import render_path
from ..sparse_attention import forward_arg
from .h3_rules import (
    CAPSTONE_GATE,
    CAPSTONE_PAST_END,
    GUIDES_INPUT,
    REFINE_WINDOWS_ENV_VAR,
    not_h3,
)

logger = logging.getLogger("dw")

KEYSTONE_ATTENTION_INPUT = "experimental_keystone_attention"
KEYSTONE_ATTENTION_KEYS = ("balance", "blocks")
DEFAULT_BALANCE = 20.0
DEFAULT_BLOCKS = "31-33,36-44,47"
FL2VA = "fl2va"
IMAGE_INPUT = "image"
TST_INPUT = "experimental_tst"
SPARSE = "sparse"
_H3_CLASS = "MiniMaxH3Transformer3DModel"
_STATE = "_dw_keystone_attention_state"
_RANGE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$")


@dataclass(frozen=True)
class KeystoneAttentionSettings:
    balance: float = DEFAULT_BALANCE
    blocks: str = DEFAULT_BLOCKS


def keystone_attention_enabled():
    return os.environ.get(REFINE_WINDOWS_ENV_VAR) == "1"


def _flag_problem():
    return (
        f"{KEYSTONE_ATTENTION_INPUT} is an experimental spike (#1391), off "
        f"unless {REFINE_WINDOWS_ENV_VAR}=1 is set where the server runs"
    )


def _unresolved(value):
    return isinstance(value, str) and references.is_ref(references.UNRESOLVED, value)


def parse_blocks(text, total=None):
    """The block indices `text` names, in order: comma-separated ints and
    inclusive `a-b` ranges, each in 0..total-1 when `total` is given.

    Raises:
        ValueError: If `text` is not a string, is empty, has a malformed or
            reversed range, an index out of range, or names a block twice
    """
    if not isinstance(text, str):
        raise ValueError(f"'blocks' is a string like {DEFAULT_BLOCKS!r}, got {text!r}")
    if not text.strip():
        raise ValueError(
            f"'blocks' names no block - give ints and a-b ranges like {DEFAULT_BLOCKS!r}"
        )
    blocks = []
    for part in text.split(","):
        match = _RANGE.match(part)
        if not match:
            numbered = (
                ""
                if total is None
                else f"; blocks are numbered 0-{total - 1} on this transformer"
            )
            raise ValueError(
                f"'blocks' entry {part.strip()!r} is not an int or an a-b range "
                f"(like {DEFAULT_BLOCKS!r}){numbered}"
            )
        first = int(match.group(1))
        last = int(match.group(2)) if match.group(2) is not None else first
        if last < first:
            raise ValueError(f"'blocks' range {part.strip()!r} runs backwards")
        if total is not None and last >= total:
            raise ValueError(
                f"'blocks' entry {part.strip()!r} is past the transformer's last "
                f"block, {total - 1} (blocks are numbered 0-{total - 1})"
            )
        for index in range(first, last + 1):
            if index in blocks:
                raise ValueError(f"'blocks' names block {index} twice")
            blocks.append(index)
    return tuple(blocks)


@functools.lru_cache(maxsize=1)
def declared_block_count():
    """The H3 transformer's block count as diffusers declares it (the class's
    `num_layers` default), or None when diffusers cannot say - then a range is
    checked against the loaded transformer at the run."""
    try:
        import inspect

        from diffusers import MiniMaxH3Transformer3DModel

        default = (
            inspect.signature(MiniMaxH3Transformer3DModel.__init__)
            .parameters["num_layers"]
            .default
        )
    except Exception:  # an old diffusers, or no H3: left to the run
        return None
    return default if isinstance(default, int) and default > 0 else None


def keystone_attention_value_problem(value, total=None):
    """Why `value` cannot be an `experimental_keystone_attention`, or None. An
    unresolved reference, anywhere in it, is left to the run."""
    if not isinstance(value, Mapping):
        return (
            f"{KEYSTONE_ATTENTION_INPUT} is an object {{'balance': "
            f"{DEFAULT_BALANCE:g}, 'blocks': {DEFAULT_BLOCKS!r}}}, got {value!r}"
        )
    unknown = sorted(set(value) - set(KEYSTONE_ATTENTION_KEYS))
    if unknown:
        return (
            f"{KEYSTONE_ATTENTION_INPUT} takes only "
            f"{', '.join(KEYSTONE_ATTENTION_KEYS)}, not {', '.join(unknown)}"
        )
    balance = value.get("balance", DEFAULT_BALANCE)
    if not _unresolved(balance) and (
        isinstance(balance, bool)
        or not isinstance(balance, Real)
        or not math.isfinite(balance)
        or balance <= 0
    ):
        return (
            f"{KEYSTONE_ATTENTION_INPUT} 'balance' is a finite number above 0 - "
            f"{DEFAULT_BALANCE:g} is the default, 1 changes nothing - got {balance!r}"
        )
    blocks = value.get("blocks", DEFAULT_BLOCKS)
    if not _unresolved(blocks):
        try:
            parse_blocks(blocks, total)
        except ValueError as e:
            return f"{KEYSTONE_ATTENTION_INPUT} {e}"
    return None


def keystone_attention_settings(value):
    """The call's settings, or None when the argument is absent, null or false.

    Raises:
        ValueError: If the flag is off, or the value is not a keystone
            attention object
    """
    if value is None or value is False:
        return None
    if not keystone_attention_enabled():
        raise ValueError(_flag_problem())
    problem = keystone_attention_value_problem(value)
    if problem:
        raise ValueError(problem)
    return KeystoneAttentionSettings(
        float(value.get("balance", DEFAULT_BALANCE)),
        value.get("blocks", DEFAULT_BLOCKS),
    )


def describe(settings):
    return f"balance {settings.balance:g} on blocks {settings.blocks}"


# 1. Validate


def _sparse_configured(configuration):
    from .h3_tst import _sparse_configured as configured

    return configured(configuration)


def _step_problems(pipeline, arguments):
    """[message] for one step that passes the argument, the flag on."""
    from .chain_rules import CHAIN_CAPSTONE, CHAIN_KEYSTONE_SLOT
    from ..adapter_compatibility import FROM_PRETRAINED_KEY, WORKFLOW_KEY

    problems = []
    h3 = not_h3(pipeline, KEYSTONE_ATTENTION_INPUT, taken_by=(FL2VA,))
    if h3:
        return [h3]
    from_pretrained = pipeline.get(FROM_PRETRAINED_KEY)
    workflow = (
        from_pretrained.get(WORKFLOW_KEY) if isinstance(from_pretrained, dict) else None
    )
    if isinstance(workflow, str) and workflow != FL2VA and not _unresolved(workflow):
        problems.append(
            f"{KEYSTONE_ATTENTION_INPUT} re-splits attention between an fl2va "
            f"call's keyframe and guide rows, and this step loads the "
            f"'{workflow}' workflow - use fl2va"
        )
    chain = pipeline.get("chain")
    # A chain fills each segment's image and guides at the run, which checks
    if not isinstance(chain, dict):
        if arguments.get(IMAGE_INPUT) is None:
            problems.append(
                f"{KEYSTONE_ATTENTION_INPUT} re-splits attention toward the "
                f"keyframe rows, and this step passes no 'image'"
            )
        guides = arguments.get(GUIDES_INPUT)
        if guides is None or (isinstance(guides, (list, tuple)) and not guides):
            problems.append(
                f"{KEYSTONE_ATTENTION_INPUT} re-splits attention away from the "
                f"guide rows, and this step passes no 'guides'"
            )
    value = arguments.get(KEYSTONE_ATTENTION_INPUT)
    problem = (
        None
        if _unresolved(value)
        else keystone_attention_value_problem(value, declared_block_count())
    )
    if problem:
        problems.append(problem)
    tst = arguments.get(TST_INPUT)
    if tst is not None and tst is not False:
        problems.append(
            f"{KEYSTONE_ATTENTION_INPUT} does not combine with {TST_INPUT}: "
            f"both own the blocks' attention processor"
        )
    if _sparse_configured(pipeline.get("configuration")):
        problems.append(
            f"{KEYSTONE_ATTENTION_INPUT} runs on dense attention, and this "
            f"step's transformer runs attention_backend 'sparse' - set it to null"
        )
    if isinstance(chain, dict) and chain.get(CHAIN_CAPSTONE) is not None:
        problems.append(
            f"{KEYSTONE_ATTENTION_INPUT} does not combine with a chain's "
            f"{CHAIN_CAPSTONE}: its gated rows move the guide rows it re-splits"
        )
    slot = chain.get(CHAIN_KEYSTONE_SLOT) if isinstance(chain, dict) else None
    if isinstance(slot, dict) and slot.get(CAPSTONE_GATE) is not None:
        problems.append(
            f"{KEYSTONE_ATTENTION_INPUT} does not combine with a gated "
            f"{CHAIN_KEYSTONE_SLOT}: its gated rows move the guide rows it re-splits"
        )
    guides = arguments.get(GUIDES_INPUT)
    if isinstance(guides, (list, tuple)) and any(
        isinstance(g, dict)
        and (g.get(CAPSTONE_GATE) is not None or g.get(CAPSTONE_PAST_END) is not None)
        for g in guides
    ):
        problems.append(
            f"{KEYSTONE_ATTENTION_INPUT} does not combine with gated guide rows "
            f"('{CAPSTONE_GATE}' / '{CAPSTONE_PAST_END}' on a guide)"
        )
    return problems


def _passing_steps(workflow_definition, source_indices):
    for _, _, pipeline, source, where in references.iter_steps(
        workflow_definition.get("steps"), source_indices, "pipeline"
    ):
        arguments = pipeline.get("arguments")
        if not isinstance(arguments, dict):
            continue
        value = arguments.get(KEYSTONE_ATTENTION_INPUT)
        if value is None or value is False:
            continue
        path = render_path(
            ("steps", source, "pipeline", "arguments", KEYSTONE_ATTENTION_INPUT)
        )
        yield pipeline, arguments, value, path, where


def keystone_attention_errors(workflow_definition, source_indices=None):
    """Every `experimental_keystone_attention` refused before the run, as
    [{path, message}]: the flag off, a step that is not H3 fl2va or passes no
    image or no guides, a value that is no keystone attention object (a bad
    balance, a malformed, out-of-range, empty or repeated block), or a step
    that also asks for TST, sparse attention, a capstone or gated guide rows."""
    errors = []
    for pipeline, arguments, _, path, where in _passing_steps(
        workflow_definition, source_indices
    ):
        problems = (
            [_flag_problem()]
            if not keystone_attention_enabled()
            else _step_problems(pipeline, arguments)
        )
        errors += [{"path": path, "message": f"{p}{where}"} for p in problems]
    return errors


def keystone_attention_notes(workflow_definition, source_indices=None):
    """["path: message"] naming the settings each valid step's call runs, its
    defaults filled - what validate shows for `{}`."""
    if not keystone_attention_enabled():
        return []
    notes = []
    for _, _, value, path, where in _passing_steps(workflow_definition, source_indices):
        if not isinstance(value, Mapping) or keystone_attention_value_problem(value):
            continue
        balance = value.get("balance", DEFAULT_BALANCE)
        blocks = value.get("blocks", DEFAULT_BLOCKS)
        notes.append(
            f"{path}: {KEYSTONE_ATTENTION_INPUT} (experimental, #1391) runs "
            f"balance {balance if _unresolved(balance) else format(balance, 'g')} "
            f"on blocks {blocks}{where}"
        )
    return notes


# 2. The re-split - pure


def resplit(out, lse, out_k, lse_k, out_c, lse_c, balance):
    """The re-split output, from the three passes.

    `out*`: [B, N, S, D]; `lse*`: [B, N, S] natural-log log-sum-exps of the
    full, K-only and C-only passes. Taken in float32; returned in `out`'s
    dtype.

    The shift pK' - pK = S*b*pK/(b*pK + pC) - pK is taken in its reduced form,
    pK*pC*(b - 1)/(b*pK + pC), and pC' - pC is its negative (S is kept): the
    same formula, but exactly 0 at b = 1, so balance 1 returns `out` itself
    bit for bit."""
    lse = lse.float()
    p_k = torch.exp(lse_k.float() - lse)
    p_c = torch.exp(lse_c.float() - lse)
    denominator = balance * p_k + p_c
    shift = torch.where(
        denominator > 0,
        p_k
        * p_c
        * (balance - 1.0)
        / denominator.clamp(min=torch.finfo(torch.float32).tiny),
        torch.zeros_like(p_k),
    )
    result = out.float() + shift[..., None] * (out_k.float() - out_c.float())
    return result.to(out.dtype)


def _lse_shape(lse, q):
    b, n, s, _ = q.shape
    return lse.reshape(b, n, s)


def _cudnn(q, k, v, scale):
    out = torch.ops.aten._scaled_dot_product_cudnn_attention(
        q, k, v, None, True, 0.0, False, False, scale=scale
    )
    return out[0], _lse_shape(out[1], q)


def _flash(q, k, v, scale):
    out = torch.ops.aten._scaled_dot_product_flash_attention(
        q, k, v, 0.0, False, False, scale=scale
    )
    return out[0], _lse_shape(out[1], q)


def _cpu_flash(q, k, v, scale):
    out, lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        q, k, v, 0.0, False, scale=scale
    )
    return out, _lse_shape(lse, q)


_CUDA_LSE_OPS = {
    "FLASH_ATTENTION": ("native flash", _flash),
    "CUDNN_ATTENTION": ("cuDNN", _cudnn),
}


def sdpa_priority_names():
    """The SDPA backend names, in the order torch's dispatcher tries them, or
    None when this torch cannot say."""
    try:
        order = torch._C._get_sdp_priority_order()
        return [torch._C._SDPBackend(index).name for index in order]
    except Exception:  # an older or newer torch without the private call
        return None


def lse_candidates(device_type, priority=None):
    """[(name, op)] the LSE-returning attention ops to try on `device_type`,
    in order - none on mps.

    On CUDA the order is the one SDPA itself tries (`priority`, read off torch
    when not given): flash, then cuDNN on stock torch. H3's dense processor
    runs SDPA, so the main pass here runs the same kernel it does and balance
    1 reproduces a run without the argument (#1393's bounce: cuDNN ahead of
    flash gave a 2/255 drift)."""
    if device_type == "cuda":
        names = priority if priority is not None else sdpa_priority_names()
        ordered = [_CUDA_LSE_OPS[n] for n in names or () if n in _CUDA_LSE_OPS]
        return ordered + [op for op in _CUDA_LSE_OPS.values() if op not in ordered]
    if device_type == "cpu":
        return [("CPU flash", _cpu_flash)]
    return []


def pick_lse_op(q, k, v, scale):
    """(name, op) for the first candidate that runs on these tensors.

    Raises:
        RuntimeError: When none does, naming each one's reason
    """
    candidates = lse_candidates(q.device.type)
    reasons = []
    for name, op in candidates:
        try:
            op(q, k[:, :, :1], v[:, :, :1], scale)
        except Exception as e:  # each kernel names its own limit
            reasons.append(f"{name}: {str(e).splitlines()[0] if str(e) else e!r}")
            continue
        return name, op
    why = "; ".join(reasons) or f"no LSE attention op on {q.device.type}"
    raise RuntimeError(
        f"{KEYSTONE_ATTENTION_INPUT} needs an attention op that returns its "
        f"log-sum-exp (native flash or cuDNN on CUDA), and none runs here - {why}"
    )


def resplit_attention(q, k, v, rows_k, rows_c, balance, op, scale=None, first_query=0):
    """The re-split attention output, [B, N, S, D].

    `q`, `k`, `v`: [B, N, S, D]; `rows_k`, `rows_c`: (start, count) key row
    ranges; `op(q, k, v, scale) -> (out, lse)`. Only queries from
    `first_query` on are re-split; the earlier ones keep the main pass's
    output (#1494: re-splitting the held keyframe and guide rows' own
    queries moved the condition rows themselves, and the take opened on an
    unrelated frame)."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    out, lse = op(q, k, v, scale)
    if first_query >= q.shape[2]:
        return out
    qt = q[:, :, first_query:].contiguous()
    (k0, kn), (c0, cn) = rows_k, rows_c
    out_k, lse_k = op(
        qt,
        k[:, :, k0 : k0 + kn].contiguous(),
        v[:, :, k0 : k0 + kn].contiguous(),
        scale,
    )
    out_c, lse_c = op(
        qt,
        k[:, :, c0 : c0 + cn].contiguous(),
        v[:, :, c0 : c0 + cn].contiguous(),
        scale,
    )
    tail = resplit(
        out[:, :, first_query:],
        lse[..., first_query:],
        out_k,
        lse_k,
        out_c,
        lse_c,
        balance,
    )
    if first_query == 0:
        return tail
    return torch.cat([out[:, :, :first_query], tail], dim=2)


# 3. The state, the rows check and the processor


@dataclass
class _KeystoneAttentionState:
    settings: KeystoneAttentionSettings
    blocks: tuple = ()
    rows: tuple | None = None  # ((k start, k count), (c start, c count))
    checked: tuple | None = None
    frame_rows: int = 0  # rows of one latent frame; the target's first is held out
    target: int | None = None  # row the target video starts at, per forward
    op: tuple | None = None  # (name, op), picked on the first selected block
    forwards: int = 0
    per_forward: bool = False  # rows published per forward; none = the original
    originals: list = field(default_factory=list)
    handles: list = field(default_factory=list)


def publish_rows(transformer, num_text, keyframe_rows, guide_rows, frame_rows=0):
    """Hand the call's K and C row ranges to an installed re-split, if any;
    returns them, or None when no re-split is installed on `transformer`."""
    state = getattr(transformer, _STATE, None)
    if state is None:
        return None
    rows = (
        (int(num_text), int(keyframe_rows)),
        (int(num_text + keyframe_rows), int(guide_rows)),
    )
    state.rows = rows
    state.frame_rows = int(frame_rows)
    state.checked = None
    return rows


def publish_window_rows(transformer, keystone_rows, held_rows, frame_rows):
    """Hand one windowed forward's K (`keystone_rows`) and C (`held_rows`),
    each a (start, count) row range, to a `per_forward` install on
    `transformer`; returns them, or None when no re-split is installed.

    Raises:
        ValueError: If the install is not `per_forward`
    """
    state = getattr(transformer, _STATE, None)
    if state is None:
        return None
    if not state.per_forward:
        raise ValueError(
            f"{KEYSTONE_ATTENTION_INPUT}: this re-split takes its rows from the "
            f"guide layout, once per call, not per forward"
        )
    rows = tuple(
        (int(start), int(count)) for start, count in (keystone_rows, held_rows)
    )
    state.rows = rows
    state.frame_rows = int(frame_rows)
    state.checked = None
    return rows


def clear_rows(transformer):
    """Unpublish a `per_forward` install's rows: the next forward runs the
    original processor unless rows are published for it."""
    state = getattr(transformer, _STATE, None)
    if state is not None and state.per_forward:
        state.rows = None
        state.checked = None


def installed(transformer):
    return getattr(transformer, _STATE, None) is not None


def check_rows(rows, token_tags, timestep_indices):
    """Raise unless K and C are non-empty runs of video condition rows (tag 0,
    one condition timestep) that lie ahead of the trailing target video."""
    if rows is None:
        raise ValueError(
            f"{KEYSTONE_ATTENTION_INPUT}: the guide layout published no keyframe "
            f"or guide rows - the call has no guides, or its pipeline lays them "
            f"out some other way"
        )
    (k0, kn), (c0, cn) = rows
    if kn <= 0 or cn <= 0:
        raise ValueError(
            f"{KEYSTONE_ATTENTION_INPUT}: the call laid out {kn} keyframe and "
            f"{cn} guide rows; it needs both"
        )
    tags = token_tags.reshape(-1).cpu()
    end = c0 + cn
    non_video = (tags != 0).nonzero()
    target = int(non_video[-1]) + 1 if len(non_video) else 0
    if c0 != k0 + kn or end > tags.numel() or end > target:
        raise ValueError(
            f"{KEYSTONE_ATTENTION_INPUT}: the keyframe and guide rows "
            f"{k0}-{end - 1} are not one condition run ahead of the target video "
            f"(which starts at row {target})"
        )
    if bool((tags[k0:end] != 0).any()):
        raise ValueError(
            f"{KEYSTONE_ATTENTION_INPUT}: rows {k0}-{end - 1} are not all video "
            f"condition rows"
        )
    if timestep_indices is not None:
        indices = timestep_indices.reshape(-1)[k0:end].cpu()
        if indices.numel() and bool((indices != indices[0]).any()):
            raise ValueError(
                f"{KEYSTONE_ATTENTION_INPUT}: the keyframe and guide rows do not "
                f"share one condition timestep"
            )


def check_window_rows(rows, token_tags, timestep_indices):
    """Raise unless a windowed forward's K and C are non-empty runs of video
    rows (tag 0), K ahead of C, each at one timestep, and C the leading rows
    of the trailing video. Returns C's start, the target's first row."""
    (k0, kn), (c0, cn) = rows
    if kn <= 0 or cn <= 0:
        raise ValueError(
            f"{KEYSTONE_ATTENTION_INPUT}: the window published {kn} keystone and "
            f"{cn} held rows; it needs both"
        )
    tags = token_tags.reshape(-1).cpu()
    if c0 < k0 + kn or c0 + cn > tags.numel():
        raise ValueError(
            f"{KEYSTONE_ATTENTION_INPUT}: the window's keystone rows {k0}-"
            f"{k0 + kn - 1} and held rows {c0}-{c0 + cn - 1} do not lie in order "
            f"in its {tags.numel()} rows"
        )
    if bool((tags[k0 : k0 + kn] != 0).any()) or bool((tags[c0:] != 0).any()):
        raise ValueError(
            f"{KEYSTONE_ATTENTION_INPUT}: the window's keystone rows are not "
            f"video rows, or its held rows {c0}-{c0 + cn - 1} do not lead its "
            f"trailing video"
        )
    if timestep_indices is not None:
        indices = timestep_indices.reshape(-1).cpu()
        for start, count, what in ((k0, kn, "keystone"), (c0, cn, "held")):
            run = indices[start : start + count]
            if bool((run != run[0]).any()):
                raise ValueError(
                    f"{KEYSTONE_ATTENTION_INPUT}: the window's {what} rows do not "
                    f"share one timestep"
                )
    return c0


def _window_pre_hook(component, args, kwargs):
    state = getattr(component, _STATE)
    if state.rows is None:
        return None
    tags = forward_arg(args, kwargs, "token_tags")
    indices = forward_arg(args, kwargs, "timestep_indices")
    key = (state.rows, id(tags), id(indices))
    if state.checked != key:
        state.target = check_window_rows(state.rows, tags, indices)
        state.checked = key
    state.forwards += 1
    return None


def _pre_hook(component, args, kwargs):
    state = getattr(component, _STATE)
    tags = forward_arg(args, kwargs, "token_tags")
    indices = forward_arg(args, kwargs, "timestep_indices")
    key = (state.rows, id(tags), id(indices))
    if state.checked != key:
        check_rows(state.rows, tags, indices)
        non_video = (tags.reshape(-1).cpu() != 0).nonzero()
        state.target = int(non_video[-1]) + 1 if len(non_video) else 0
        state.checked = key
    state.forwards += 1
    return None


class KeystoneAttentionH3Processor:
    """H3's attention with the K/C re-split. The projections, norms and
    rotary embedding are the stock processor's (MiniMaxH3AttnProcessor); the
    attention runs on an LSE-returning op instead of the dispatcher. The
    original's backend and parallel config proxy through, as TST's do."""

    def __init__(self, original, state, layer_index):
        self._original = original
        self._state = state
        self.layer_index = layer_index

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

    @staticmethod
    def _first_query(state, rows_c):
        """The first query row that is re-split: the target video's second
        latent frame. The first one sits at the keyframe's and the guide's
        first position and is held to them (#1494), so it keeps the main
        pass; with no frame size or target known it is the row after the guide
        rows."""
        after_guides = rows_c[0] + rows_c[1]
        if state.target is None or state.frame_rows <= 0:
            return after_guides
        return max(after_guides, state.target + state.frame_rows)

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        from diffusers.models.transformers.transformer_minimax_h3 import (
            _apply_rotary_emb,
        )

        state = self._state
        if state.rows is None and state.per_forward:
            # A forward that publishes no rows keeps the original attention
            return self._original(
                attn,
                hidden_states,
                rotary_emb=rotary_emb,
                attention_mask=attention_mask,
            )
        if state.rows is None:
            # The pre-hook refuses this; a direct call never reaches it
            check_rows(None, None, None)
        if attention_mask is not None:
            raise ValueError(
                f"{KEYSTONE_ATTENTION_INPUT}: H3 passes no attention mask, and "
                f"this forward passed one"
            )
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        query = query.unflatten(-1, (-1, attn.head_dim))
        key = key.unflatten(-1, (-1, attn.head_dim))
        value = value.unflatten(-1, (-1, attn.head_dim))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        # [B, S, N, D] -> [B, N, S, D], the layout the aten ops take
        q, k, v = (x.transpose(1, 2).contiguous() for x in (query, key, value))
        scale = q.shape[-1] ** -0.5
        if state.op is None:
            state.op = pick_lse_op(q, k, v, scale)
        rows_k, rows_c = state.rows
        out = resplit_attention(
            q,
            k,
            v,
            rows_k,
            rows_c,
            state.settings.balance,
            state.op[1],
            scale,
            first_query=self._first_query(state, rows_c),
        )
        hidden_states = out.transpose(1, 2).flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


# 4. One call's install


def _h3_transformers(pipeline):
    from .h3_tst import _h3_transformers as transformers

    return transformers(pipeline)


def _sparse_problem():
    return (
        f"{KEYSTONE_ATTENTION_INPUT} runs on dense attention, and this "
        f"step's transformer runs attention_backend 'sparse' - set it to null"
    )


def keystone_attention_refusal(pipeline, configuration=None):
    """Why this call cannot run the re-split, or None: no H3 transformer, or
    one on dw's 'sparse' processor."""
    from ..sparse_attention import is_sparse_installed

    transformers = _h3_transformers(pipeline)
    if not transformers:
        return f"{KEYSTONE_ATTENTION_INPUT} found no MiniMax-H3 transformer on this pipeline"
    if _sparse_configured(configuration) or any(
        is_sparse_installed(component) for _, component in transformers
    ):
        return _sparse_problem()
    return None


def transformer_refusal(component):
    """Why the re-split cannot go on this one transformer, or None: it runs
    dw's 'sparse' processor, already has a re-split installed, or has no
    transformer blocks to put it on."""
    from ..sparse_attention import is_sparse_installed

    if not getattr(component, "transformer_blocks", None):
        return f"{KEYSTONE_ATTENTION_INPUT} found no transformer blocks on this transformer"
    if is_sparse_installed(component):
        return _sparse_problem()
    if installed(component):
        return f"{KEYSTONE_ATTENTION_INPUT} is already installed on this transformer"
    return None


def install(component, settings, per_forward=False):
    """Put the re-split processor on the settings' blocks of `component`;
    every other block keeps its own. `per_forward`: the rows are published
    for each forward that re-splits (`publish_window_rows`), and a forward
    with none runs the original attention.

    Raises:
        ValueError: If a block is past the transformer's last
    """
    blocks = list(component.transformer_blocks)
    selected = parse_blocks(settings.blocks, len(blocks))
    state = _KeystoneAttentionState(
        settings=settings, blocks=selected, per_forward=per_forward
    )
    # First, so a raise partway through still leaves `uninstall` its record
    setattr(component, _STATE, state)
    for index in selected:
        original = blocks[index].attn.processor
        state.originals.append((index, original))
        blocks[index].attn.set_processor(
            KeystoneAttentionH3Processor(original, state, index)
        )
    hook = _window_pre_hook if per_forward else _pre_hook
    state.handles = [component.register_forward_pre_hook(hook, with_kwargs=True)]
    return state


def uninstall(component):
    state = getattr(component, _STATE, None)
    if state is None:
        return
    blocks = list(component.transformer_blocks)
    for index, original in state.originals:
        blocks[index].attn.set_processor(original)
    for handle in state.handles:
        handle.remove()
    delattr(component, _STATE)


def _report(label, settings, states):
    forwards = sum(state.forwards for state in states)
    ops = sorted({state.op[0] for state in states if state.op is not None})
    rows = next((state.rows for state in states if state.rows is not None), None)
    counts = f", {rows[0][1]} keyframe / {rows[1][1]} guide rows" if rows else ""
    message = (
        f"{label}: keystone attention (experimental, #1391) {describe(settings)}: "
        f"{forwards} forwards on {', '.join(ops) or 'no op'}{counts}"
    )
    logger.info(message)
    emit_log(
        message,
        balance=settings.balance,
        blocks=settings.blocks,
        forwards=forwards,
        lse_op=", ".join(ops),
    )


@contextmanager
def keystone_attention_call(pipeline, settings, label):
    """Put the re-split on the settings' blocks of the pipeline's H3
    transformers for one call, and the originals back when it ends - also
    when it raises. Reported once, when the call completes.

    Raises:
        ValueError: If the pipeline has no dense H3 transformer, or a block
            is past its last
    """
    refusal = keystone_attention_refusal(pipeline)
    if refusal:
        raise ValueError(f"{label}: {refusal}")
    done = []
    try:
        for _, component in _h3_transformers(pipeline):
            done.append(component)
            try:
                install(component, settings)
            except ValueError as e:
                raise ValueError(f"{label}: {KEYSTONE_ATTENTION_INPUT} {e}") from e
        states = [getattr(component, _STATE) for component in done]
        yield states
    finally:
        for component in done:
            uninstall(component)
    _report(label, settings, states)
