"""torch.compile's on-disk cache survives a reboot by default.

Inductor caches compiled graphs and generated kernels under
/tmp/torchinductor_<user> unless TORCHINDUCTOR_CACHE_DIR says otherwise, and
/tmp is wiped on reboot. A fresh worker still has to trace, but a cache hit
skips codegen and kernel builds - the bulk of a cold compile, about 90s of
MiniMax-H3's first step on a 3090. dw points the cache at the user's cache
directory instead, so that cost is paid once per shape rather than once per
boot.
"""

import os
import subprocess
import sys

import dw


def test_follows_xdg_cache_home():
    environ = {"XDG_CACHE_HOME": os.path.join("/srv", "cache")}
    assert dw._inductor_cache_dir_default(environ) == os.path.join(
        "/srv", "cache", "dw", "torchinductor"
    )


def test_falls_back_to_home_cache():
    expected = os.path.join(os.path.expanduser("~"), ".cache", "dw", "torchinductor")
    assert dw._inductor_cache_dir_default({}) == expected


def test_an_empty_xdg_cache_home_is_ignored():
    # The XDG spec treats an empty value as unset
    expected = os.path.join(os.path.expanduser("~"), ".cache", "dw", "torchinductor")
    assert dw._inductor_cache_dir_default({"XDG_CACHE_HOME": ""}) == expected


def _env_after_import(preset=None):
    code = "import os, dw; print(os.environ['TORCHINDUCTOR_CACHE_DIR'])"
    env = {k: v for k, v in os.environ.items() if k != "TORCHINDUCTOR_CACHE_DIR"}
    if preset is not None:
        env["TORCHINDUCTOR_CACHE_DIR"] = preset
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def test_import_applies_the_default():
    assert _env_after_import() == dw._inductor_cache_dir_default(os.environ)


def test_an_explicit_setting_wins():
    assert _env_after_import(preset="/var/tmp/inductor") == "/var/tmp/inductor"
