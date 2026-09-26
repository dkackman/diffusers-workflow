"""Parallel shard loading is on by default except on macOS.

diffusers' parallel loading runs one thread per shard, and each thread moves
its tensors to the device. On MPS, concurrent copies from those threads
segfaulted LTX-2.5's SDNQ transformer load (exit 139, four ThreadPoolExecutor
threads inside copy_ to mps); the same load with parallel loading off took
39s and succeeded. diffusers reads HF_ENABLE_PARALLEL_LOADING once at import,
so the default is chosen before torch is imported - by platform, since macOS
has no CUDA and its accelerator is always MPS. CUDA boxes keep the faster load.
"""

import os
import subprocess
import sys

import pytest

import dw


def test_off_on_macos():
    assert dw._parallel_loading_default("darwin") == "false"


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_on_elsewhere(platform):
    assert dw._parallel_loading_default(platform) == "true"


def _env_after_import(preset=None):
    code = "import os, dw; print(os.environ['HF_ENABLE_PARALLEL_LOADING'])"
    env = {k: v for k, v in os.environ.items() if k != "HF_ENABLE_PARALLEL_LOADING"}
    if preset is not None:
        env["HF_ENABLE_PARALLEL_LOADING"] = preset
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def test_import_applies_this_platforms_default():
    assert _env_after_import() == dw._parallel_loading_default(sys.platform)


def test_an_explicit_setting_wins():
    assert _env_after_import(preset="1") == "1"
