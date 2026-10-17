"""Sparse attention for MiniMax H3's generated-video rows (#931).

Sparse here means comfy-kitchen's Sol-Attn: the generated video's rows are
grouped into 4x4x4 (t, h, w) tiles of 64 rows, and each query tile attends only
to the key tiles that matter, instead of every row. Everything that is not the
generated video (text, references, target audio) stays dense.

The layout rule: H3 packs text, then the references in request order, then the
target audio, then the target video last. A reference image carries the same
video tag as the target, so the target video is the trailing run of video rows,
not every video-tagged row. Rows before it go first, in segment-pure tiles
(contiguous runs of one tag, at most 64 rows each) that are always attended in
full; the video tiles follow.

The decided-once rule: whether a pipeline call runs sparse is decided once,
before the first forward, and is all or nothing. A layout check that fails
after that decision raises; it never drops a single forward back to dense,
because a run that mixes the two is worse than either.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from .events import emit_warning

SPARSE_BACKEND = "sparse"
DEFAULT_TAU = 0.5

# Rows per tile: 4 x 4 x 4 in (t, h, w)
TILE = 64
_SIDE = 4

# The one class sparse attention knows how to lay out
_H3_CLASS = "MiniMaxH3Transformer3DModel"

# The kernel seam: tests set this to a stand-in; None means comfy-kitchen's
_sol_attn = None


class SparseLayoutError(ValueError):
    """The packed rows are not a layout the tile map can describe."""


@dataclass(frozen=True)
class TileMap:
    """Where each packed row lives in the padded, tiled sequence.

    `dst` sends packed row -> padded row; `block_len` is the live row count of
    each tile; the first `num_prefix_tiles` tiles are the dense prefix and the
    rest are video tiles of the `video_grid` (t, h, w) latent grid.
    """

    dst: torch.Tensor
    block_len: torch.Tensor
    num_prefix_tiles: int
    padded_rows: int
    rows: int
    video_grid: tuple[int, int, int]


def build_tile_map(
    position_ids: torch.Tensor, token_tags: torch.Tensor, device
) -> TileMap:
    """Map each packed row to a padded tiled row.

    Every row that is not target video goes first, in segment-pure tiles; the
    target video follows in 4x4x4 tiles of its (t, h, w) position ids.
    """
    tags = token_tags.cpu()
    pos = position_ids.cpu()
    s = tags.numel()
    if pos.shape != (s, 3):
        raise SparseLayoutError(
            f"position_ids {tuple(pos.shape)} do not match {s} token tags"
        )

    # Target video = the trailing run of video rows
    non_video = (tags != 0).nonzero()
    target_start = int(non_video[-1]) + 1 if len(non_video) else 0
    if target_start >= s:
        raise SparseLayoutError("no target video rows at the end of the sequence")

    dst = torch.empty(s, dtype=torch.long)
    live: list[int] = []
    tile = 0

    # Prefix segments are contiguous runs of one tag
    prefix_tags = tags[:target_start]
    breaks = torch.ones(target_start, dtype=torch.bool)
    if target_start > 1:
        breaks[1:] = prefix_tags[1:] != prefix_tags[:-1]
    starts = breaks.nonzero().squeeze(1).tolist() + [target_start]
    for a, b in zip(starts[:-1], starts[1:]):
        for c in range(a, b, TILE):
            n = min(TILE, b - c)
            dst[c : c + n] = tile * TILE + torch.arange(n)
            live.append(n)
            tile += 1
    num_prefix_tiles = tile

    # H3's rotary coordinates are fractional (aspect-normalized space, uneven
    # time), so rank each axis's distinct values to get the integer grid index
    vrows = torch.arange(target_start, s)
    vp = torch.stack(
        [torch.unique(pos[vrows, a], return_inverse=True)[1] for a in range(3)], dim=1
    )
    gt, gh, gw = (vp.max(0).values + 1).tolist()
    if gt * gh * gw != len(vrows):
        raise SparseLayoutError(
            f"target video is not a full grid: {gt}x{gh}x{gw} != {len(vrows)} rows"
        )
    nh, nw = (gh + _SIDE - 1) // _SIDE, (gw + _SIDE - 1) // _SIDE
    tid = ((vp[:, 0] // _SIDE) * nh + vp[:, 1] // _SIDE) * nw + vp[:, 2] // _SIDE
    within = ((vp[:, 0] % _SIDE) * _SIDE + vp[:, 1] % _SIDE) * _SIDE + vp[:, 2] % _SIDE
    order = torch.argsort(tid * TILE + within, stable=True)
    _, counts = torch.unique_consecutive(tid[order], return_counts=True)
    offs = 0
    for n in counts.tolist():
        dst[vrows[order[offs : offs + n]]] = tile * TILE + torch.arange(n)
        live.append(n)
        tile += 1
        offs += n

    if sum(live) != s or torch.unique(dst).numel() != s:
        raise SparseLayoutError("tile map does not cover every row exactly once")
    return TileMap(
        dst=dst.to(device),
        block_len=torch.tensor(live, dtype=torch.int32, device=device),
        num_prefix_tiles=num_prefix_tiles,
        padded_rows=tile * TILE,
        rows=s,
        video_grid=(gt, gh, gw),
    )


def sparse_unavailable_reason(device, dtype, head_dim: int) -> str | None:
    """Why the Sol-Attn kernel cannot run here, or None when it can.

    Asked once per pipeline call, before the first forward: a reason makes the
    whole call run dense, with a warning, rather than fail.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return f"device {device.type} is not CUDA"
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) < (8, 0):
        return f"compute capability {major}.{minor} is below 8.0"
    if dtype not in (torch.bfloat16, torch.float16):
        return f"dtype {str(dtype).removeprefix('torch.')} is not bfloat16/float16"
    if head_dim != 128:
        return f"head_dim {head_dim} is not 128"
    try:
        import comfy_kitchen as ck

        if not ck.sol_attn_is_available(device):
            return "comfy-kitchen kernels unavailable: sol_attn_is_available() is False"
    except Exception as exc:  # an import or a kernel probe that fails is a no
        return f"comfy-kitchen kernels unavailable: {exc}"
    return None


def _kernel():
    """The sol_attn callable: the test seam when set, else comfy-kitchen's.

    The kernel takes bfloat16/float16 only. The decision refuses any other
    dtype, so the cast here is a guard a real run does not reach. It lives at
    this seam, not in the processor, so a stand-in gets the tensors as they
    are: a float32 test then checks the permutation exactly, not to bf16.
    """
    if _sol_attn is not None:
        return _sol_attn
    import comfy_kitchen as ck

    def run(q, k, v, **kwargs):
        if q.dtype not in (torch.bfloat16, torch.float16):
            q, k, v = (x.to(torch.bfloat16) for x in (q, k, v))
        with ck.use_backend("cuda"):
            return ck.sol_attn(q, k, v, **kwargs)

    return run


@dataclass
class _SparseState:
    """One installed component's sparse settings and its per-call decision.

    `decided` is reset by `sparse_call` on entry; the first forward after that
    sets it, and `reason` is None when that forward decided sparse. `active` is
    True only inside the component's forward, between the two hooks.
    """

    tau: float
    originals: list = field(default_factory=list)
    handles: list = field(default_factory=list)
    decided: bool = False
    reason: str | None = None
    tile_map: TileMap | None = None
    layout_key: tuple | None = None
    active: bool = False

    def reset(self):
        self.decided = False
        self.reason = None
        self.tile_map = None
        self.layout_key = None
        self.active = False


class SparseH3AttnProcessor:
    """Sol-Attn over one H3 block, or the original processor when the call
    decided dense.

    The sparse path is the stock processor's with the attention swapped: the
    same projections, q/k norms and rotary, then the rows are scattered into
    the padded tile order, attended there, and gathered back.
    """

    def __init__(self, original, state: _SparseState):
        self._original = original
        self._state = state

    # diffusers' set_attention_backend / context parallelism write these onto
    # every processor; they belong to the original, which the dense path runs
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
        if not (state.active and state.decided and state.reason is None):
            return self._original(attn, hidden_states, rotary_emb, attention_mask)

        tm = state.tile_map
        if attention_mask is not None:
            raise SparseLayoutError("sparse attention was given an attention mask")
        if hidden_states.shape[1] != tm.rows:
            raise SparseLayoutError(
                f"{hidden_states.shape[1]} rows reached a block whose tile map has {tm.rows}"
            )

        from diffusers.models.transformers.transformer_minimax_h3 import (
            _apply_rotary_emb,
        )

        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        b, _, h, d = query.shape
        dst = tm.dst.to(query.device)

        def tiled(x):
            out = x.new_zeros((b, tm.padded_rows, h, d))
            out[:, dst] = x
            return out

        pt = tm.num_prefix_tiles
        out = _kernel()(
            tiled(query),
            tiled(key),
            tiled(value),
            sink_blocks=[0, pt],
            sink_q=[0, pt],
            block_len=tm.block_len.to(query.device),
            tau=state.tau,
        )
        hidden_states = out[:, dst].flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


# The H3 forward's leading positional parameters, for a caller that passes
# them positionally; diffusers' denoiser passes them by name
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


def _block_dtype(component) -> torch.dtype:
    """The block stack's compute dtype: its first floating-point parameter,
    which skips the integer storage of a quantized weight."""
    for p in component.transformer_blocks[0].parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float32


def _pre_hook(component, args, kwargs):
    """Decide the call on its first forward; keep the tile map current after."""
    state = component._dw_sparse_state
    hidden_states = _forward_arg(args, kwargs, "hidden_states")
    position_ids = _forward_arg(args, kwargs, "position_ids")
    token_tags = _forward_arg(args, kwargs, "token_tags")
    if position_ids is None or token_tags is None:
        missing = "position_ids/token_tags were not passed"
        if state.decided and state.reason is None:
            raise SparseLayoutError(missing)
        state.decided, state.reason = True, state.reason or missing
        return None

    key = (tuple(position_ids.shape), int(token_tags.sum()))
    if not state.decided:
        state.decided = True
        device = hidden_states.device
        state.reason = sparse_unavailable_reason(
            device, _block_dtype(component), component.config.attention_head_dim
        )
        if state.reason is None:
            try:
                state.tile_map = build_tile_map(position_ids, token_tags, device)
                state.layout_key = key
            except SparseLayoutError as exc:
                state.reason = str(exc)
    elif state.reason is None and key != state.layout_key:
        # Decided sparse: a layout the map cannot describe now raises
        state.tile_map = build_tile_map(position_ids, token_tags, hidden_states.device)
        state.layout_key = key
    state.active = True
    return None


def _post_hook(component, args, output):
    component._dw_sparse_state.active = False
    return None


def install_sparse(component, tau: float) -> bool:
    """Put the sparse processor on every block of an H3 transformer.

    Returns False, changing nothing, for any other component. A second call
    only updates tau. The forward itself is not wrapped: H3's denoiser picks
    the layout kwargs it passes from the forward's signature, so the decision
    runs in forward hooks instead.
    """
    if type(component).__name__ != _H3_CLASS:
        return False
    state = getattr(component, "_dw_sparse_state", None)
    if state is not None:
        state.tau = tau
        return True
    state = _SparseState(tau=tau)
    for block in component.transformer_blocks:
        state.originals.append(block.attn.processor)
        block.attn.set_processor(SparseH3AttnProcessor(block.attn.processor, state))
    state.handles = [
        component.register_forward_pre_hook(_pre_hook, with_kwargs=True),
        component.register_forward_hook(_post_hook, always_call=True),
    ]
    component._dw_sparse_state = state
    return True


def uninstall_sparse(component) -> None:
    """Put the original processors back and remove both hooks."""
    state = getattr(component, "_dw_sparse_state", None)
    if state is None:
        return
    for block, original in zip(component.transformer_blocks, state.originals):
        block.attn.set_processor(original)
    for handle in state.handles:
        handle.remove()
    del component._dw_sparse_state


def is_sparse_installed(component) -> bool:
    return getattr(component, "_dw_sparse_state", None) is not None


def set_sparse_tau(component, tau: float) -> None:
    component._dw_sparse_state.tau = tau


@dataclass
class SparseCallReport:
    """How one pipeline call ran: sparse throughout, or dense and why."""

    ran_sparse: bool = False
    reason: str | None = None


def _installed_components(pipeline):
    """Every component of the pipeline with sparse installed.

    A ModularPipeline lists its components in a `components` mapping; anything
    else is read through its attributes.
    """
    components = getattr(pipeline, "components", None)
    if isinstance(components, Mapping):
        values = components.values()
    else:
        values = getattr(pipeline, "__dict__", {}).values()
    seen = set()
    for value in values:
        if id(value) not in seen and is_sparse_installed(value):
            seen.add(id(value))
            yield value


@contextmanager
def sparse_call(pipeline, label: str) -> Iterator[SparseCallReport]:
    """Scope one pipeline call's sparse decision.

    Every installed component decides afresh on its first forward inside the
    block. A component that fell back is reported once per call, through the
    run's warnings, since a run that silently ran dense is a speed regression
    nobody would see. The report is filled on exit, also when the call raises;
    the warning is emitted only when it completes.
    """
    states = [c._dw_sparse_state for c in _installed_components(pipeline)]
    for state in states:
        state.reset()
    report = SparseCallReport()
    try:
        yield report
    finally:
        decided = [s for s in states if s.decided]
        reasons = [s.reason for s in decided if s.reason is not None]
        report.reason = reasons[0] if reasons else None
        report.ran_sparse = bool(decided) and not reasons
    if report.reason is not None:
        emit_warning(
            f"{label}: sparse attention ran dense - {report.reason}",
            attention_backend=SPARSE_BACKEND,
            reason=report.reason,
        )
