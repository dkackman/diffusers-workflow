"""Concurrent type resolution must not race diffusers' lazy imports (#787)."""

import subprocess
import sys
import textwrap


def test_parallel_get_type_in_fresh_interpreter():
    # A fresh process, so diffusers' lazy modules are not yet imported
    code = textwrap.dedent(
        """
        from concurrent.futures import ThreadPoolExecutor
        from dw.type_helpers import load_type_from_name
        names = ["DiffusionPipeline", "ModularPipeline", "AutoencoderKL",
                 "FluxPipeline", "StableDiffusionPipeline", "UNet2DConditionModel"] * 3
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda n: load_type_from_name(n, constructed=False), names))
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
