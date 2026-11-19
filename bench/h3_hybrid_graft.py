"""Graft ref2va's late-block adaln_proj onto fl2va's transformer (#1441, stage #1463).

The hybrid arm of bench/records/h3-hybrid.md. MiniMax-H3 ships two
transformers of one architecture: `transformer` (fl2va) and `transformer_ref`
(ref2va). This writes a third, in diffusers layout, that is fl2va everywhere
except `transformer_blocks.{FIRST..LAST}.adaln_proj.*`, which come from ref2va,
with ref2va's config.json (so the folder loads as a ref2va transformer_ref).

It reads both parents lazily with safetensors' `safe_open`, one fl2va shard
at a time, so it never holds either whole model in RAM. Before writing
anything it checks the two parents carry identical tensor names and shapes;
a tensor only ref2va has is taken from ref2va and listed in the report, and
one only fl2va has is an error (the hybrid would not load as transformer_ref).

The output folder is `<OUT>/transformer_ref/`: a dw step points
`load_components.pretrained_model_name_or_path.transformer_ref` at `<OUT>`
and diffusers appends the spec's `transformer_ref` subfolder. A local path
loads uncached, so it quantizes fresh every run.

usage: python bench/h3_hybrid_graft.py FL2VA_DIR REF2VA_DIR OUT [--first 20] [--last 49]
   FL2VA_DIR / REF2VA_DIR are the snapshot's `transformer` / `transformer_ref`
   folders; prints a JSON tensor report on stdout.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

INDEX = "diffusion_pytorch_model.safetensors.index.json"
BLOCK = re.compile(r"^transformer_blocks\.(\d+)\.adaln_proj\.")


def from_ref(name, first, last):
    """Whether tensor `name` is one of the grafted adaln_proj tensors."""
    match = BLOCK.match(name)
    return bool(match) and first <= int(match.group(1)) <= last


def select(fl2va_shapes, ref2va_shapes, first, last):
    """Which tensors the hybrid takes from ref2va.

    Both arguments map tensor name -> shape. Returns (grafted, ref_only):
    the adaln_proj tensors of blocks first..last, and the tensors only ref2va
    has, both sorted. Raises ValueError when a name only fl2va has, or a shape
    differs, since then the parents are not one architecture.
    """
    fl_only = sorted(set(fl2va_shapes) - set(ref2va_shapes))
    if fl_only:
        raise ValueError(f"tensors only fl2va has: {fl_only}")
    mismatched = sorted(
        name
        for name in fl2va_shapes
        if tuple(fl2va_shapes[name]) != tuple(ref2va_shapes[name])
    )
    if mismatched:
        raise ValueError(f"shapes differ: {mismatched}")
    grafted = sorted(n for n in fl2va_shapes if from_ref(n, first, last))
    ref_only = sorted(set(ref2va_shapes) - set(fl2va_shapes))
    return grafted, ref_only


def _weight_map(folder):
    return json.loads((folder / INDEX).read_text(encoding="utf-8"))["weight_map"]


def _shapes(folder, weight_map):
    from safetensors import safe_open

    shapes = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(folder / shard), framework="pt") as f:
            for name in f.keys():
                shapes[name] = tuple(f.get_slice(name).get_shape())
    return shapes


def graft(fl2va, ref2va, out, first, last):
    from safetensors import safe_open
    from safetensors.torch import save_file

    fl_map, ref_map = _weight_map(fl2va), _weight_map(ref2va)
    grafted, ref_only = select(
        _shapes(fl2va, fl_map), _shapes(ref2va, ref_map), first, last
    )
    take = set(grafted)
    target = out / "transformer_ref"
    target.mkdir(parents=True, exist_ok=False)

    shards = sorted(set(fl_map.values()))
    # tensors only ref2va has ride in the last shard
    extra = {shards[-1]: ref_only}
    weight_map, total = {}, 0
    for shard in shards:
        names = sorted(n for n, s in fl_map.items() if s == shard)
        tensors = {}
        with safe_open(str(fl2va / shard), framework="pt") as f:
            for name in names:
                if name not in take:
                    tensors[name] = f.get_tensor(name)
        for name in sorted(n for n in names if n in take) + extra.get(shard, []):
            with safe_open(str(ref2va / ref_map[name]), framework="pt") as f:
                tensors[name] = f.get_tensor(name)
        save_file(tensors, str(target / shard), metadata={"format": "pt"})
        for name, tensor in tensors.items():
            weight_map[name] = shard
            total += tensor.numel() * tensor.element_size()
        del tensors
        print(f"wrote {shard}", file=sys.stderr, flush=True)

    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (target / INDEX).write_text(json.dumps(index, indent=2), encoding="utf-8")
    shutil.copyfile(ref2va / "config.json", target / "config.json")
    return {
        "fl2va": str(fl2va),
        "ref2va": str(ref2va),
        "out": str(target),
        "blocks": [first, last],
        "tensors": len(weight_map),
        "from_ref2va": len(grafted) + len(ref_only),
        "grafted": grafted,
        "ref2va_only": ref_only,
        "config": "ref2va",
        "total_bytes": total,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fl2va", type=Path)
    parser.add_argument("ref2va", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--first", type=int, default=20)
    parser.add_argument("--last", type=int, default=49)
    args = parser.parse_args(argv)
    if args.first > args.last:
        parser.error("--first is past --last")
    report = graft(
        args.fl2va.resolve(),
        args.ref2va.resolve(),
        args.out.resolve(),
        args.first,
        args.last,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
