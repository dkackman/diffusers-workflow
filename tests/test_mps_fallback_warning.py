"""torch announces an op the MPS backend lacks, when PYTORCH_ENABLE_MPS_FALLBACK
is set, with a UserWarning - and dw ignored every UserWarning, so a step that
quietly ran part of its model on the CPU left no trace. The fallback variable is
set by the fp4-fp8-for-torch-mps autoload install.sh adds on macOS."""

import subprocess
import sys

PROBE = """
import warnings, dw
fallback = ("The operator 'aten::_linalg_eigvals' is not currently supported "
            "on the MPS backend and will fall back to run on the CPU. "
            "This may have performance implications.")
with warnings.catch_warnings(record=True) as caught:
    warnings.warn(fallback, UserWarning)
    warnings.warn("some other library chatter", UserWarning)
print(len(caught), caught[0].message if caught else "")
"""


def test_only_the_fallback_warning_gets_through():
    result = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    count, _, message = result.stdout.strip().splitlines()[-1].partition(" ")
    assert count == "1"
    assert "fall back to run on the CPU" in message
