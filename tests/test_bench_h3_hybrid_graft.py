"""bench/h3_hybrid_graft.py: tensor selection by block range on a toy state
dict, and a tiny two-shard graft round trip through safetensors."""

import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))
import h3_hybrid_graft as hg  # noqa: E402


def toy_shapes(blocks=4):
    shapes = {"proj_in.weight": (8, 4), "norm_out.weight": (8,)}
    for b in range(blocks):
        shapes[f"transformer_blocks.{b}.adaln_proj.linear.weight"] = (48, 8)
        shapes[f"transformer_blocks.{b}.adaln_proj.linear.bias"] = (48,)
        shapes[f"transformer_blocks.{b}.attn.to_q.weight"] = (8, 8)
    return shapes


def test_only_the_adaln_proj_of_the_range_is_grafted():
    shapes = toy_shapes(4)
    grafted, ref_only = hg.select(shapes, dict(shapes), 2, 3)
    assert grafted == [
        "transformer_blocks.2.adaln_proj.linear.bias",
        "transformer_blocks.2.adaln_proj.linear.weight",
        "transformer_blocks.3.adaln_proj.linear.bias",
        "transformer_blocks.3.adaln_proj.linear.weight",
    ]
    assert ref_only == []


def test_block_numbers_match_whole_not_by_prefix():
    # block 2 must not catch blocks 20..29, and vice versa
    shapes = toy_shapes(25)
    grafted, _ = hg.select(shapes, dict(shapes), 20, 24)
    blocks = {int(n.split(".")[1]) for n in grafted}
    assert blocks == {20, 21, 22, 23, 24}


def test_a_tensor_only_ref2va_has_is_taken_and_listed():
    fl = toy_shapes(2)
    ref = dict(fl, **{"ref_embedder.weight": (8, 8)})
    _, ref_only = hg.select(fl, ref, 0, 1)
    assert ref_only == ["ref_embedder.weight"]


def test_a_tensor_only_fl2va_has_is_refused():
    fl = toy_shapes(2)
    ref = dict(fl)
    del ref["proj_in.weight"]
    with pytest.raises(ValueError, match="only fl2va"):
        hg.select(fl, ref, 0, 1)


def test_a_shape_mismatch_is_refused():
    fl = toy_shapes(2)
    ref = dict(fl, **{"proj_in.weight": (8, 5)})
    with pytest.raises(ValueError, match="shapes differ"):
        hg.select(fl, ref, 0, 1)


def _write_parent(folder, fill, extra=()):
    torch = pytest.importorskip("torch")
    from safetensors.torch import save_file

    folder.mkdir()
    names = sorted(toy_shapes(3)) + list(extra)
    shapes = dict(toy_shapes(3), **{n: (2,) for n in extra})
    half = len(names) // 2
    shards = {
        "diffusion_pytorch_model-00001-of-00002.safetensors": names[:half],
        "diffusion_pytorch_model-00002-of-00002.safetensors": names[half:],
    }
    weight_map = {}
    for shard, members in shards.items():
        save_file(
            {n: torch.full(shapes[n], fill) for n in members}, str(folder / shard)
        )
        weight_map.update({n: shard for n in members})
    (folder / hg.INDEX).write_text(json.dumps({"weight_map": weight_map}))
    (folder / "config.json").write_text(json.dumps({"which": fill}))


def test_graft_round_trip(tmp_path):
    from safetensors import safe_open

    _write_parent(tmp_path / "fl", 1.0)
    _write_parent(tmp_path / "ref", 2.0, extra=["ref_only.weight"])
    report = hg.graft(tmp_path / "fl", tmp_path / "ref", tmp_path / "out", 1, 2)

    target = tmp_path / "out" / "transformer_ref"
    assert json.loads((target / "config.json").read_text()) == {"which": 2.0}
    weight_map = json.loads((target / hg.INDEX).read_text())["weight_map"]
    assert report["ref2va_only"] == ["ref_only.weight"]
    assert report["tensors"] == len(weight_map) == len(toy_shapes(3)) + 1
    for name, shard in weight_map.items():
        with safe_open(str(target / shard), framework="pt") as f:
            value = float(f.get_tensor(name).flatten()[0])
        expected = (
            2.0 if name in report["grafted"] or name == "ref_only.weight" else 1.0
        )
        assert value == expected, name
    assert {int(n.split(".")[1]) for n in report["grafted"]} == {1, 2}
