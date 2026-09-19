"""Runtime environment details - Python, torch, CUDA/driver and other
installed package versions - for diagnosing environment mismatches between
boxes (#222). Read-only, gathered fresh on every call rather than cached,
since an install can change underneath a long-running server (see
`update_diffusers`).
"""

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version

# Packages worth reporting beyond torch (which gets its own field): the
# ones most likely to cause a version-mismatch failure or be silently
# absent on a given box.
PACKAGES = (
    "diffusers",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "peft",
    "safetensors",
    "sentencepiece",
)


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _nvidia_driver_version():
    """The NVIDIA driver version, via nvidia-smi when it's on PATH. Distinct
    from torch's own `cuda_version`, which is the CUDA toolkit torch was
    built against, not the driver actually installed on this box."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.strip().splitlines()
    return lines[0].strip() if lines else None


def runtime_info():
    """Python, torch, CUDA/driver and other installed package versions -
    the detail `device`/`version` alone doesn't answer, like "is
    bitsandbytes even installed here" or "which CUDA build is torch"."""
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except ImportError:
        torch_version = None
        cuda_version = None

    return {
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "driver_version": _nvidia_driver_version(),
        "packages": {name: _package_version(name) for name in PACKAGES},
    }
