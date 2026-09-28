"""scripts/arch_metrics.py - the numbers the stabilization gates and the
harness ratchet read."""

import importlib.util
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "arch_metrics.py"


def _load():
    spec = importlib.util.spec_from_file_location("arch_metrics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(root, files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def _long_function(lines):
    body = "".join(f"    x{i} = {i}\n" for i in range(lines))
    return f"def f():\n{body}"


def test_a_long_function_and_a_long_module_are_counted(tmp_path):
    metrics = _load().measure(
        _tree(
            tmp_path,
            {"dw/a.py": _long_function(151) + "\n" * 900, "dw/b.py": "x = 1\n"},
        )
    )
    assert metrics["functions_over_150_lines"] == 1
    assert metrics["modules_over_1000_lines"] == 1
    assert metrics["modules"] == 2


def test_community_pipelines_are_not_counted(tmp_path):
    metrics = _load().measure(
        _tree(tmp_path, {"dw/community_pipelines/p.py": _long_function(200)})
    )
    assert metrics["modules"] == 0
    assert metrics["functions_over_150_lines"] == 0


def test_a_reference_prefix_literal_is_counted_but_a_longer_string_is_not(tmp_path):
    metrics = _load().measure(
        _tree(tmp_path, {"dw/a.py": 'P = "asset:"\nQ = "asset:cat.png is a file"\n'})
    )
    assert metrics["prefix_literals"] == 1


def test_patches_of_dw_paths_and_claude_md_lines_are_counted(tmp_path):
    metrics = _load().measure(
        _tree(
            tmp_path,
            {
                "tests/test_x.py": 'patch("dw.a.b")\npatch("os.path")\n',
                "CLAUDE.md": "one\ntwo\n",
                "ui/CLAUDE.md": "three\n",
            },
        )
    )
    assert metrics["test_dw_patch_targets"] == 1
    assert metrics["claude_md_lines"] == 3


def test_regressions_names_only_the_metrics_that_got_worse():
    module = _load()
    baseline = {"modules": 10, "prefix_literals": 5, "duplicate_blocks": None}
    current = {"modules": 11, "prefix_literals": 4, "duplicate_blocks": 3}
    assert module.regressions(current, baseline) == ["modules: 10 -> 11"]


def test_check_mode_exits_nonzero_on_a_regression(tmp_path):
    # A small tree, never the real repo: duplicate analysis of the whole
    # codebase would run on every suite run
    root = _tree(tmp_path / "repo", {"dw/a.py": "x = 1\n"})
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"modules": 0}))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--check", str(baseline)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "modules: 0 ->" in result.stdout
