#!/usr/bin/env python
"""Measure what an on-disk cache of SDNQ on-the-fly-quantized components would save.

Stage 0 of #815 (#825): the gate for building the quantized cache. For the
MiniMax H3 default config (transformer, text encoder, VAE, as in
workflows/templates/minimax/video-with-audio.json) and the LTX-2.5 default
(transformer, text encoder, as in workflows/templates/ltx2/text-to-video.json)
it times, per component:

- fresh: `from_pretrained` with an on-the-fly `SDNQConfig`, the way dw loads
  it today (read bf16, quantize on `quantization_device`, land on
  `return_device`);
- cached: `load_sdnq_model` of the same component saved with
  `save_sdnq_model`, plus placement on `return_device`.

Each is timed with the page cache warm and cold. Cold means evicted per file,
without root: `os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` on every
source safetensors (fresh) or every saved file (cached) right before the
load. No `drop_caches`, no sudo. Eviction is advisory: a page another process
holds may stay resident, so cold numbers may read slightly warm.

It imports only torch, diffusers, transformers, huggingface_hub and sdnq - no
dw engine code - and reads the weights from the local Hugging Face cache only
(the models must already be downloaded; nothing is fetched).

**Stop `dw.serve` first.** A running worker's resident models distort the
page cache and the timings, and both cards share host RAM; the script refuses
to run while a process running `-m dw.serve` exists (this process and its
ancestors excepted).

Run it on lem from the repo root, with the venv active:

    cd ~/diffusers-workflow && source ./activate
    python scripts/measure_quantized_cache.py

Options: `--family h3|ltx|all` (default all), `--device cuda:0` (default
`cuda`; replaces "cuda" in the configs' quantization/return devices),
`--scratch DIR` (default `$HF_HOME/dw-quantized-measure`), `--keep` to leave
the saved entries in the scratch directory (by default each is deleted once
measured). It prints one table at the end: per component the fresh warm/cold
and cached warm/cold seconds and the bytes on disk, the eviction method,
per-family totals, and the H3 saving against the ~30 s gate.
"""

import argparse
import gc
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

GATE_SECONDS = 30.0
EVICTION_METHOD = "posix_fadvise DONTNEED (per file, no root)"


@dataclass
class ComponentSpec:
    family: str
    name: str
    repo: str
    subfolder: str
    cls_path: str  # "diffusers.X" or "transformers.X"
    sdnq_args: dict
    timings: dict = field(default_factory=dict)
    bytes_on_disk: int | None = None
    error: str | None = None


# The defaults as the templates ship them (variables resolved).
H3_REPO = "MiniMaxAI/MiniMax-H3"
LTX_REPO = "Lightricks/LTX-2.5-Diffusers"


def default_specs():
    return [
        ComponentSpec(
            "h3",
            "transformer",
            H3_REPO,
            "transformer",
            "diffusers.MiniMaxH3Transformer3DModel",
            {
                "weights_dtype": "int4",
                "quantization_device": "cuda",
                "return_device": "cpu",
                "use_quantized_matmul": True,
                "dequantize_fp32": False,
                "modules_to_not_convert": [
                    "proj_in",
                    "audio_proj_in",
                    "context_embedder",
                    "time_embedder",
                    "time_proj",
                    "token_refiner",
                    "norm_out",
                    "proj_out",
                    "audio_proj_out",
                ],
            },
        ),
        ComponentSpec(
            "h3",
            "text_encoder",
            H3_REPO,
            "text_encoder",
            "transformers.Qwen3VLForConditionalGeneration",
            {
                "weights_dtype": "int4",
                "quantization_device": "cuda",
                "return_device": "cpu",
                "dequantize_fp32": False,
                "modules_to_not_convert": [".model.visual", "lm_head"],
            },
        ),
        ComponentSpec(
            "h3",
            "vae",
            H3_REPO,
            "vae",
            "diffusers.AutoencoderKLMiniMaxH3",
            {
                "weights_dtype": "int8",
                "quant_conv": True,
                "use_quantized_matmul_conv": True,
                "quantization_device": "cuda",
                "return_device": "cpu",
                "dequantize_fp32": False,
            },
        ),
        ComponentSpec(
            "ltx",
            "transformer",
            LTX_REPO,
            "transformer",
            "diffusers.LTX2VideoTransformer3DModel",
            {
                "weights_dtype": "uint4",
                "quantization_device": "cuda",
                "return_device": "cuda",
                "use_quantized_matmul": True,
                "dequantize_fp32": False,
            },
        ),
        ComponentSpec(
            "ltx",
            "text_encoder",
            LTX_REPO,
            "text_encoder",
            "transformers.Gemma4UnifiedForConditionalGeneration",
            {
                "weights_dtype": "int8",
                "quantization_device": "cuda",
                "return_device": "cpu",
                "dequantize_fp32": False,
                "modules_to_not_convert": ["embed_vision", "embed_audio", "lm_head"],
            },
        ),
    ]


def ancestor_pids():
    """This process's pid and every ancestor's, so the launching shell never matches."""
    pids = set()
    pid = os.getpid()
    while pid > 0 and pid not in pids:
        pids.add(pid)
        result = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True
        )
        try:
            pid = int(result.stdout.strip())
        except ValueError:
            break
    return pids


def refuse_if_server_running():
    # the server's own invocation is `python -m dw.serve`; a bare "dw.serve" would
    # also match any unrelated command line that merely mentions it
    result = subprocess.run(
        ["pgrep", "-f", "--", r"-m dw\.serve"], capture_output=True, text=True
    )
    skip = ancestor_pids()
    pids = [p for p in result.stdout.split() if p.isdigit() and int(p) not in skip]
    if pids:
        sys.exit(
            f"dw.serve is running (pid {', '.join(pids)}). Stop it first: its resident models distort the page cache and timings."
        )


def resolve_class(cls_path):
    module_name, _, cls_name = cls_path.partition(".")
    module = __import__(module_name)
    cls = getattr(module, cls_name, None)
    if cls is None:
        raise RuntimeError(f"{cls_path} is not in the installed {module_name}")
    return cls


def source_dir(spec):
    from huggingface_hub import snapshot_download

    prefix = f"{spec.subfolder}/" if spec.subfolder else ""
    # only what a load reads: the weights, the config and the shard index
    patterns = [
        f"{prefix}*.safetensors",
        f"{prefix}config.json",
        f"{prefix}*.index.json",
    ]
    snapshot = snapshot_download(
        spec.repo, allow_patterns=patterns, local_files_only=True
    )
    path = Path(snapshot) / spec.subfolder
    if not any(path.glob("*.safetensors")):
        raise RuntimeError(f"no safetensors under {path}: download {spec.repo} first")
    return path


def files_under(path):
    # resolve() follows the HF cache's snapshot symlinks to the blobs
    return [p.resolve() for p in sorted(Path(path).rglob("*")) if p.is_file()]


def evict(paths):
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)  # DONTNEED does not drop dirty pages
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def warm(paths):
    # read every byte once, so the warm run starts with the files resident
    for path in paths:
        with open(path, "rb") as f:
            while f.read(64 * 1024 * 1024):
                pass


def sync(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def release(torch):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def device_args(sdnq_args, device):
    args = dict(sdnq_args)
    for key in ("quantization_device", "return_device"):
        if args.get(key) == "cuda":
            args[key] = device
    return args


def touch_pages(model, torch):
    """Read one byte per 4 KiB page of every CPU tensor, so lazily mmapped
    safetensors are actually read inside the timed region."""
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        try:
            if t.device.type != "cpu" or t.numel() == 0:
                continue
            flat = t.detach().reshape(-1).view(torch.uint8)
            total += int(flat[::4096].sum())
        except Exception:
            # a quantized tensor subclass may not expose a plain view: read its
            # inner tensors instead, else skip it
            flatten = getattr(t, "__tensor_flatten__", None)
            for name in flatten()[0] if flatten else ():
                inner = getattr(t, name, None)
                if isinstance(inner, torch.Tensor) and inner.device.type == "cpu":
                    try:
                        flat = inner.detach().reshape(-1).view(torch.uint8)
                        total += int(flat[::4096].sum())
                    except Exception:
                        pass
    return total


def load_fresh(spec, cls, sdnq_args, torch):
    from sdnq import SDNQConfig

    dtype_kwarg = (
        "dtype" if spec.cls_path.startswith("transformers.") else "torch_dtype"
    )
    kwargs = {
        dtype_kwarg: torch.bfloat16,
        "quantization_config": SDNQConfig(**sdnq_args),
    }
    model = cls.from_pretrained(
        spec.repo, subfolder=spec.subfolder, local_files_only=True, **kwargs
    )
    touch_pages(model, torch)
    return model


def load_cached(entry, cls, sdnq_args, torch):
    from sdnq import load_sdnq_model

    return_device = sdnq_args["return_device"]
    model = load_sdnq_model(
        str(entry),
        model_cls=cls,
        dtype=torch.bfloat16,
        device=return_device,
        dequantize_fp32=sdnq_args.get("dequantize_fp32"),
        use_quantized_matmul=sdnq_args.get("use_quantized_matmul"),
    )
    model = model.to(return_device)
    touch_pages(model, torch)
    return model


def timed(fn, torch):
    sync(torch)
    start = time.perf_counter()
    model = fn()
    sync(torch)
    return model, time.perf_counter() - start


def measure(spec, scratch, device, keep, torch):
    from sdnq import save_sdnq_model

    cls = resolve_class(spec.cls_path)
    sdnq_args = device_args(spec.sdnq_args, device)
    src_files = files_under(source_dir(spec))
    entry = scratch / f"{spec.family}-{spec.name}"
    if entry.exists():
        shutil.rmtree(entry)

    def step(label, fn, files, cold):
        (evict if cold else warm)(files)
        model, seconds = timed(fn, torch)
        spec.timings[label] = seconds
        print(f"  {spec.family}/{spec.name} {label}: {seconds:.1f} s", flush=True)
        return model

    print(
        f"{spec.family}/{spec.name}: {spec.repo} {spec.subfolder} ({spec.cls_path})",
        flush=True,
    )
    model = step(
        "fresh_cold",
        lambda: load_fresh(spec, cls, sdnq_args, torch),
        src_files,
        cold=True,
    )
    del model
    release(torch)
    model = step(
        "fresh_warm",
        lambda: load_fresh(spec, cls, sdnq_args, torch),
        src_files,
        cold=False,
    )
    save_sdnq_model(model, str(entry))
    del model
    release(torch)
    entry_files = files_under(entry)
    spec.bytes_on_disk = sum(p.stat().st_size for p in entry_files)
    print(
        f"  {spec.family}/{spec.name} saved: {spec.bytes_on_disk / 1e9:.2f} GB",
        flush=True,
    )
    # the fresh loads leave the source pages warm; evict them so they don't
    # crowd the cached runs out of RAM
    evict(src_files)

    model = step(
        "cached_cold",
        lambda: load_cached(entry, cls, sdnq_args, torch),
        entry_files,
        cold=True,
    )
    del model
    release(torch)
    model = step(
        "cached_warm",
        lambda: load_cached(entry, cls, sdnq_args, torch),
        entry_files,
        cold=False,
    )
    del model
    release(torch)
    evict(entry_files)
    if not keep:
        shutil.rmtree(entry)


def fmt_seconds(value):
    return "-" if value is None else f"{value:.1f}"


def print_table(specs, device):
    columns = ("fresh_warm", "fresh_cold", "cached_warm", "cached_cold")
    print()
    print(f"eviction: {EVICTION_METHOD}; device: {device}")
    versions = {
        pkg: importlib.metadata.version(pkg)
        for pkg in ("sdnq", "diffusers", "transformers", "torch")
    }
    print("versions: " + ", ".join(f"{k} {v}" for k, v in versions.items()))
    print()
    header = (
        f"| {'component':<20} | {'weights':<7} | "
        + " | ".join(f"{c:>11}" for c in columns)
        + f" | {'GB on disk':>10} |"
    )
    print(header)
    print(
        "|" + "|".join("-" * (len(cell) + 0) for cell in header.split("|")[1:-1]) + "|"
    )
    totals = {}
    for spec in specs:
        row_name = f"{spec.family}/{spec.name}"
        if spec.error:
            print(
                f"| {row_name:<20} | {spec.sdnq_args['weights_dtype']:<7} | error: {spec.error}"
            )
            continue
        cells = " | ".join(f"{fmt_seconds(spec.timings.get(c)):>11}" for c in columns)
        gb = (
            f"{spec.bytes_on_disk / 1e9:.2f}" if spec.bytes_on_disk is not None else "-"
        )
        print(
            f"| {row_name:<20} | {spec.sdnq_args['weights_dtype']:<7} | {cells} | {gb:>10} |"
        )
        family = totals.setdefault(
            spec.family, {c: 0.0 for c in columns} | {"bytes": 0}
        )
        for c in columns:
            family[c] += spec.timings.get(c, 0.0)
        family["bytes"] += spec.bytes_on_disk or 0
    for name, family in totals.items():
        complete = all(not s.error for s in specs if s.family == name)
        label = f"{name} total" + ("" if complete else " (partial)")
        cells = " | ".join(f"{family[c]:>11.1f}" for c in columns)
        print(f"| {label:<20} | {'':<7} | {cells} | {family['bytes'] / 1e9:>10.2f} |")
    print()
    h3 = totals.get("h3")
    if h3 is None:
        return
    if any(s.error for s in specs if s.family == "h3"):
        print("H3 gate: not decided - a component failed (see the table)")
        return
    saving_warm = h3["fresh_warm"] - h3["cached_warm"]
    saving_cold = h3["fresh_cold"] - h3["cached_cold"]
    for label, saving in (
        ("page cache warm", saving_warm),
        ("page cache cold", saving_cold),
    ):
        verdict = "clears" if saving >= GATE_SECONDS else "misses"
        print(
            f"H3 saving per cold load, {label}: {saving:.1f} s - {verdict} the ~{GATE_SECONDS:.0f} s gate"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--family", choices=("h3", "ltx", "all"), default="all")
    parser.add_argument(
        "--device",
        default="cuda",
        help="replaces 'cuda' in the configs' devices, e.g. cuda:1",
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        default=None,
        help="where saved entries go (default $HF_HOME/dw-quantized-measure)",
    )
    parser.add_argument("--keep", action="store_true", help="keep the saved entries")
    args = parser.parse_args()

    if not hasattr(os, "posix_fadvise"):
        sys.exit(
            "os.posix_fadvise is unavailable on this platform; run this on lem (Linux)."
        )
    refuse_if_server_running()

    import sdnq  # noqa: F401 - registers the SDNQ quantizer with diffusers and transformers
    import torch
    from huggingface_hub import constants

    scratch = args.scratch or Path(constants.HF_HOME) / "dw-quantized-measure"
    scratch.mkdir(parents=True, exist_ok=True)
    specs = [s for s in default_specs() if args.family in ("all", s.family)]

    if torch.cuda.is_available():
        # pay CUDA context and allocator setup before the first timed load
        torch.ones(1, device=args.device).add_(1)
        sync(torch)

    for spec in specs:
        try:
            measure(spec, scratch, args.device, args.keep, torch)
        except Exception as e:  # report and go on to the next component
            spec.error = f"{type(e).__name__}: {e}"
            print(f"  {spec.family}/{spec.name} failed: {spec.error}", flush=True)
            release(torch)

    print_table(specs, args.device)
    print()
    print(
        json.dumps(
            [
                {
                    "component": f"{s.family}/{s.name}",
                    **s.timings,
                    "bytes": s.bytes_on_disk,
                    "error": s.error,
                }
                for s in specs
            ]
        )
    )


if __name__ == "__main__":
    main()
