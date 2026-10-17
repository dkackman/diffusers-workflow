"""Prototype (#931): run an H3 workflow in-process with comfy-kitchen sol_attn
swapped in for the generated-video rows. Not engine code.

usage: SOL_MODE=dense|tau|topk python h3_sol_proto.py WORKFLOW OUT_DIR [k=v ...]
"""

import functools
import json
import logging
import os
import sys
import time

sys.path.insert(0, "/tmp/ck-target")

import torch  # noqa: E402

MODE = os.environ.get("SOL_MODE", "dense")
TAU = float(os.environ.get("SOL_TAU", "1.0"))
TOPK = float(os.environ.get("SOL_TOPK", "0.25"))
# Run the final DENSE_LAST of TOTAL_FORWARDS transformer forwards dense:
# late steps set fine texture, which sparse attention smooths
DENSE_LAST = int(os.environ.get("SOL_DENSE_LAST", "0"))
TOTAL_FORWARDS = int(os.environ.get("SOL_TOTAL_FORWARDS", "0"))
TILE = 64
log = logging.getLogger("h3sol")

stats = {"forwards": 0, "fwd_s": 0.0, "sol_calls": 0, "fallback_calls": 0, "geometry": None}
_geo = {"current": None, "cache": {}}


def build_geometry(position_ids, token_tags, timestep_indices, device):
    """Map each packed row to a padded tiled row. Every row that is not
    target video goes first, in segment-pure tiles (dense); target video
    follows in 4x4x4 tiles of its (t, h, w) position ids (sparse)."""
    tags = token_tags.cpu()
    tsi = timestep_indices.cpu()
    pos = position_ids.cpu()
    s = tags.numel()
    video = tags == 0
    # Target video = the trailing run of video rows. H3 packs text, then the
    # references in request order, then target audio, then target video; a
    # reference image can share the target's noise level, so tsi can't tell them apart
    non_video = (~video).nonzero()
    target_start = int(non_video[-1]) + 1 if len(non_video) else 0
    target = torch.zeros(s, dtype=torch.bool)
    target[target_start:] = True

    dst = torch.empty(s, dtype=torch.long)
    live = []
    tile = 0
    prefix_rows = (~target).nonzero().squeeze(1)
    # Segments = contiguous runs of (tag, timestep) among prefix rows
    seg_key = tags[prefix_rows] * 1000 + tsi[prefix_rows]
    breaks = torch.ones(len(prefix_rows), dtype=torch.bool)
    if len(prefix_rows) > 1:
        breaks[1:] = (seg_key[1:] != seg_key[:-1]) | (prefix_rows[1:] != prefix_rows[:-1] + 1)
    starts = breaks.nonzero().squeeze(1).tolist() + [len(prefix_rows)]
    segments = []
    for a, b in zip(starts[:-1], starts[1:]):
        rows = prefix_rows[a:b]
        segments.append(len(rows))
        for c in range(0, len(rows), TILE):
            chunk = rows[c : c + TILE]
            dst[chunk] = tile * TILE + torch.arange(len(chunk))
            live.append(len(chunk))
            tile += 1
    num_prefix_tiles = tile

    vrows = target.nonzero().squeeze(1)
    # H3's rotary coordinates are fractional (aspect-normalized space, uneven
    # time), so rank each axis's distinct values to get the integer grid index
    vp = torch.stack(
        [torch.unique(pos[vrows, a], return_inverse=True)[1] for a in range(3)], dim=1
    )
    gt, gh, gw = (vp.max(0).values + 1).tolist()
    assert gt * gh * gw == len(vrows), f"target video is not a full grid: {gt}x{gh}x{gw} != {len(vrows)}"
    nh, nw = (gh + 3) // 4, (gw + 3) // 4
    tid = ((vp[:, 0] // 4) * nh + vp[:, 1] // 4) * nw + vp[:, 2] // 4
    within = ((vp[:, 0] % 4) * 4 + vp[:, 1] % 4) * 4 + vp[:, 2] % 4
    order = torch.argsort(tid * 64 + within, stable=True)
    tid_sorted = tid[order]
    uniq, counts = torch.unique_consecutive(tid_sorted, return_counts=True)
    offs = 0
    for n in counts.tolist():
        rows = vrows[order[offs : offs + n]]
        dst[rows] = tile * TILE + torch.arange(n)
        live.append(n)
        tile += 1
        offs += n

    assert sum(live) == s and torch.unique(dst).numel() == s
    geo = {
        "dst": dst.to(device),
        "block_len": torch.tensor(live, dtype=torch.int32, device=device),
        "num_prefix_tiles": num_prefix_tiles,
        "padded": tile * TILE,
        "rows": s,
    }
    stats["geometry"] = {
        "rows": s,
        "prefix_segments": segments,
        "prefix_tiles": num_prefix_tiles,
        "video_grid": [gt, gh, gw],
        "video_tiles": tile - num_prefix_tiles,
        "padded": tile * TILE,
    }
    return geo


def install():
    from diffusers.models.transformers import transformer_minimax_h3 as m
    import comfy_kitchen as ck

    original_call = m.MiniMaxH3AttnProcessor.__call__

    def sol_call(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        geo = _geo["current"]
        if MODE == "dense" or geo is None or geo["rows"] != hidden_states.shape[1] or attention_mask is not None:
            stats["fallback_calls"] += 1
            return original_call(self, attn, hidden_states, rotary_emb, attention_mask)

        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        if rotary_emb is not None:
            query = m._apply_rotary_emb(query, *rotary_emb)
            key = m._apply_rotary_emb(key, *rotary_emb)

        b, _, h, d = query.shape
        dst, sp = geo["dst"], geo["padded"]

        def tiled(x):
            out = x.new_zeros((b, sp, h, d), dtype=torch.bfloat16)
            out[:, dst] = x.to(torch.bfloat16)
            return out

        pt = geo["num_prefix_tiles"]
        kw = dict(topk_ratio=TOPK, tail=False) if MODE == "topk" else dict(tau=TAU)
        with ck.use_backend("cuda"):
            out = ck.sol_attn(
                tiled(query), tiled(key), tiled(value),
                sink_blocks=[0, pt], sink_q=[0, pt], block_len=geo["block_len"], **kw,
            )
        stats["sol_calls"] += 1
        hidden_states = out[:, dst].flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)

    m.MiniMaxH3AttnProcessor.__call__ = sol_call

    original_forward = m.MiniMaxH3Transformer3DModel.forward

    # wraps() matters: the H3 denoiser picks the layout kwargs it passes from
    # inspect.signature(transformer.forward), which follows __wrapped__
    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        pos, tags, tsi = kwargs.get("position_ids"), kwargs.get("token_tags"), kwargs.get("timestep_indices")
        dense_now = DENSE_LAST and stats["forwards"] >= TOTAL_FORWARDS - DENSE_LAST
        if MODE != "dense" and pos is not None and not dense_now:
            key = (tuple(pos.shape), int(tags.sum()), int(tsi.sum()), int(pos.sum()))
            if key not in _geo["cache"]:
                _geo["cache"][key] = build_geometry(pos, tags, tsi, pos.device)
                log.warning("sol geometry: %s", json.dumps(stats["geometry"]))
            _geo["current"] = _geo["cache"][key]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            return original_forward(self, *args, **kwargs)
        finally:
            torch.cuda.synchronize()
            stats["forwards"] += 1
            stats["fwd_s"] += time.perf_counter() - t0
            _geo["current"] = None

    m.MiniMaxH3Transformer3DModel.forward = forward


def main():
    wf_path, out_dir, *pairs = sys.argv[1:]
    variables = dict(p.split("=", 1) for p in pairs)
    for k, v in list(variables.items()):
        try:
            variables[k] = json.loads(v)
        except ValueError:
            pass
    install()
    from dw import startup
    from dw.workflow import workflow_from_file

    startup("INFO")
    os.makedirs(out_dir, exist_ok=True)
    wf = workflow_from_file(wf_path, out_dir)
    wf.validate()
    t0 = time.perf_counter()
    wf.run(variables)
    wall = time.perf_counter() - t0
    summary = {
        "mode": MODE, "tau": TAU if MODE == "tau" else None, "topk": TOPK if MODE == "topk" else None, "dense_last": DENSE_LAST,
        "wall_s": round(wall, 1), "transformer_forwards": stats["forwards"],
        "transformer_s": round(stats["fwd_s"], 1),
        "s_per_forward": round(stats["fwd_s"] / max(stats["forwards"], 1), 2),
        "sol_calls": stats["sol_calls"], "fallback_calls": stats["fallback_calls"],
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "geometry": stats["geometry"], "variables": variables,
    }
    print("SUMMARY " + json.dumps(summary))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
