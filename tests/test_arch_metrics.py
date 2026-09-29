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


def _branches(count):
    body = "".join(f"    if x == {i}:\n        return {i}\n" for i in range(count))
    return f"def branchy(x):\n{body}    return -1\n"


def test_a_function_over_complexity_15_is_counted_and_one_at_15_is_not(tmp_path):
    metrics = _load().measure(
        _tree(
            tmp_path,
            {"dw/a.py": _branches(15), "dw_mcp/b.py": _branches(14)},
        )
    )
    # n ifs score n + 1: 16 is over the limit, 15 is not
    assert metrics["complex_functions"] == 1


def test_complexities_name_the_function_and_where_it_is(tmp_path):
    found = _load().complexities(_tree(tmp_path, {"dw/a.py": _branches(3)}))
    assert found == [(4, "dw/a.py:1", "branchy")]


def test_community_pipelines_do_not_count_toward_complexity(tmp_path):
    metrics = _load().measure(
        _tree(tmp_path, {"dw/community_pipelines/p.py": _branches(20)})
    )
    assert metrics["complex_functions"] == 0


def test_an_import_cycle_is_counted_with_its_modules(tmp_path):
    metrics = _load().measure(
        _tree(
            tmp_path,
            {
                "dw/__init__.py": "",
                "dw/a.py": "from . import b\n",
                "dw/b.py": "def f():\n    from . import a\n",
                "dw/c.py": "from . import a\n",
            },
        )
    )
    # a lazy import closes a cycle as surely as a top-level one; c only
    # depends on it
    assert metrics["import_cycles"] == 1
    assert metrics["modules_in_import_cycles"] == 2


def test_a_folder_without_an_init_is_still_in_the_graph(tmp_path):
    metrics = _load().measure(
        _tree(
            tmp_path,
            {
                "dw/__init__.py": "",
                "dw/a.py": "from .tasks import t\n",
                "dw/tasks/t.py": "from .. import a\n",
            },
        )
    )
    assert metrics["import_cycles"] == 1
    assert metrics["modules_in_import_cycles"] == 2


def test_a_type_checking_import_does_not_close_a_cycle(tmp_path):
    metrics = _load().measure(
        _tree(
            tmp_path,
            {
                "dw/__init__.py": "",
                "dw/a.py": "from . import b\n",
                "dw/b.py": (
                    "from typing import TYPE_CHECKING\n"
                    "if TYPE_CHECKING:\n    from . import a\n"
                ),
            },
        )
    )
    assert metrics["import_cycles"] == 0


def test_the_prefix_owner_may_spell_a_prefix(tmp_path):
    metrics = _load().measure(
        _tree(
            tmp_path,
            {"dw/references.py": 'ASSET = "asset:"\n', "dw/a.py": 'X = "asset:"\n'},
        )
    )
    assert metrics["prefix_literals"] == 1
