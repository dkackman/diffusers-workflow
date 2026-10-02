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
            {"dw/a.py": _long_function(151) + "\n" * 1000, "dw/b.py": "x = 1\n"},
        )
    )
    assert metrics["functions_over_150_lines"] == 1
    assert metrics["modules_over_size_ceiling"] == 1
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


def _module_of(lines):
    # distinct lines: the duplicate-block scan is slow on a repeated one
    return "".join(f"x{i} = {i}\n" for i in range(lines))


def _run_script(root, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *args],
        capture_output=True,
        text=True,
    )


def test_a_module_in_the_warn_band_warns_and_passes(tmp_path):
    root = _tree(tmp_path / "repo", {"dw/a.py": _module_of(1001)})
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"modules_over_size_ceiling": 0}))
    result = _run_script(root, "--check", str(baseline))
    assert result.returncode == 0
    assert (
        "warning: dw/a.py is 1,001 lines (warn above 1,000, fail above 1,100)"
        in result.stdout
    )
    # the warning follows the JSON, which stays parseable up to it
    assert result.stdout.index("{") < result.stdout.index("warning:")


def test_size_warnings_cover_the_band_and_nothing_else(tmp_path):
    root = _tree(
        tmp_path,
        {
            "dw/at_1000.py": _module_of(1000),
            "dw/at_1100.py": _module_of(1100),
            "dw/at_1101.py": _module_of(1101),
            "dw/x/small.py": "x = 1\n",
        },
    )
    assert _load().size_warnings(root) == [("dw/at_1100.py", 1100)]
    assert _load().measure(root)["modules_over_size_ceiling"] == 1


def test_a_module_at_1000_lines_does_not_warn(tmp_path):
    root = _tree(tmp_path / "repo", {"dw/a.py": _module_of(1000)})
    assert "warning:" not in _run_script(root).stdout


def test_a_module_over_the_ceiling_fails_the_check(tmp_path):
    root = _tree(tmp_path / "repo", {"dw/a.py": _module_of(1101)})
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"modules_over_size_ceiling": 0}))
    result = _run_script(root, "--check", str(baseline))
    assert result.returncode == 1
    assert "modules_over_size_ceiling: 0 -> 1" in result.stdout
    assert "warning:" not in result.stdout


_RULE = "raise it only in a commit whose message names the rise and why"


def test_a_failing_check_prints_the_rebaseline_rule_and_a_passing_one_does_not(
    tmp_path,
):
    root = _tree(tmp_path / "repo", {"dw/a.py": "x = 1\n"})
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"modules": 0}))
    failing = _run_script(root, "--check", str(baseline))
    assert failing.returncode == 1
    assert "lower baseline.json freely when a ratchet improves" in failing.stdout
    assert _RULE in failing.stdout
    assert "arch-approved" not in failing.stdout
    baseline.write_text(json.dumps({"modules": 5}))
    passing = _run_script(root, "--check", str(baseline))
    assert passing.returncode == 0
    assert _RULE not in passing.stdout


def test_the_check_fails_closed_when_the_import_graph_cannot_be_built(tmp_path):
    # the graph subprocess runs in the tree's root, so this shadows grimp
    root = _tree(
        tmp_path / "repo",
        {
            "dw/__init__.py": "",
            "dw/a.py": "x = 1\n",
            "grimp.py": "raise ImportError('grimp is gone')\n",
        },
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"modules": 99}))
    result = _run_script(root, "--check", str(baseline))
    assert result.returncode != 0
    assert "CalledProcessError" in result.stderr
    assert '"modules"' not in result.stdout  # no metrics were reported as a pass


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


_REFERENCES = 'ASSET = "asset:"\nVARIABLE = "variable:"\nSUBSTITUTED = (VARIABLE,)\n'


def test_each_form_of_prefix_handling_is_counted_once_and_prose_is_not(tmp_path):
    metrics = _load().measure(
        _tree(
            tmp_path,
            {
                "dw/references.py": _REFERENCES
                + "def f(x):\n    return x.startswith(ASSET) and x[len(ASSET):]\n",
                "dw/a.py": (
                    "from . import references\n"
                    "def a(x):\n    return x.startswith(references.ASSET)\n"  # (a)
                    "def b(x):\n    return x[len(references.ASSET) :]\n"  # (b)
                    "def c(x):\n    return references.ASSET + x\n"  # (c)
                    "ALIAS = references.VARIABLE\n"  # (d)
                    'LITERAL = "builtin:h3.json"\n'  # not a prefix here: ignored
                    'def e():\n    return "asset:cat.png"\n'  # (e)
                    "def prose():\n    return \"'asset:' reads a file\"\n"
                ),
            },
        )
    )
    assert metrics["prefix_handling"] == 5


def test_a_name_bound_to_the_references_module_resolves_in_every_import_form(
    tmp_path,
):
    forms = {
        "dw/a.py": "from . import references\nx.startswith(references.ASSET)\n",
        "dw/b.py": "from . import references as refs\nx.startswith(refs.ASSET)\n",
        "dw/c.py": "from dw import references\nx.startswith(references.ASSET)\n",
        "dw/d.py": "import dw.references as r\nx.startswith(r.ASSET)\n",
        "dw/e.py": "import dw.references\nx.startswith(dw.references.ASSET)\n",
        "dw/server/f.py": "from .. import references as p\nx.startswith(p.ASSET)\n",
        "dw/g.py": "from .references import ASSET\nx.startswith(ASSET)\n",
        "dw/h.py": "from .references import ASSET as A\nx.startswith(A)\n",
        "dw/i.py": "from . import references\nK = references.ASSET\nx.startswith(K)\n",
        "dw/j.py": "from .assets import ASSET_PREFIX\nx.startswith(ASSET_PREFIX)\n",
        "dw/k.py": "from . import references\nx.startswith((references.ASSET, 'z'))\n",
    }
    load = _load()
    for name, text in forms.items():
        root = _tree(tmp_path / name.replace("/", "_"), {name: text})
        _tree(root, {"dw/references.py": _REFERENCES})
        count = 2 if name == "dw/i.py" else 1  # the alias assignment and its use
        assert load.measure(root)["prefix_handling"] == count, name


def test_an_fstring_fragment_ending_in_a_prefix_is_counted(tmp_path):
    metrics = _load().measure(
        _tree(
            tmp_path,
            {
                "dw/references.py": _REFERENCES,
                "dw/a.py": 'def f(n):\n    return f"Kept {n} as asset:{n}"\n',
            },
        )
    )
    assert metrics["prefix_handling"] == 1


def test_a_fragment_ending_in_a_word_that_merely_ends_like_a_prefix_is_not_counted(
    tmp_path,
):
    metrics = _load().measure(
        _tree(
            tmp_path,
            {
                "dw/a.py": 'M = f"{x}_output:{y}"\nN = f"{x} audio_item:{y}"\n'
                'K = f"kept as asset:{y}"\n'
            },
        )
    )
    assert metrics["prefix_handling"] == 1
