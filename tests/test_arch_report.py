"""scripts/arch_report.py - the gate report's measurements. Each is checked
on a tree small enough to count by hand, never on the real repository."""

import importlib.util
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "arch_report.py"


def _load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("arch_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lcom4_counts_methods_that_share_nothing_as_separate_components(tmp_path):
    (tmp_path / "m.py").write_text(
        "class C:\n"
        "    def a(self):\n        return self.x\n"
        "    def b(self):\n        self.x = 1\n"
        "    def c(self):\n        return self.a()\n"
        "    def d(self):\n        return self.y\n"
    )
    # a-b share x, c calls a; d touches only y
    assert _load().lcom4(tmp_path, "m.py", "C") == "2 (4 methods)"


def test_lcom4_of_a_missing_class_is_none(tmp_path):
    (tmp_path / "m.py").write_text("x = 1\n")
    assert _load().lcom4(tmp_path, "m.py", "C") is None


def test_instability_counts_modules_across_package_lines():
    edges = [
        ["dw.server.app", "dw.workflow"],
        ["dw.server.jobs", "dw.workflow"],
        ["dw.workflow", "dw.tasks.task"],
        ["dw.server.app", "dw.server.jobs"],
    ]
    result = _load().instability(edges)
    # the server's two modules each reach out and nothing reaches in
    assert result["dw.server"] == (0, 2, 1.0)
    assert result["dw (core)"] == (2, 1, 0.33)
    assert result["dw.tasks"] == (1, 0, 0.0)


def _commit(repo, files, message):
    for name in files:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.read_text() + "x\n" if path.exists() else "x\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message],
        cwd=repo,
        check=True,
    )


def test_coupling_reports_pairs_that_change_together(tmp_path):
    module = _load()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for i in range(5):
        _commit(tmp_path, ["dw/a.py", "dw/b.py"], f"both {i}")
    _commit(tmp_path, ["dw/a.py", "dw/c.py"], "once")
    revisions, pairs = module.coupling(tmp_path, "HEAD")
    assert revisions["dw/a.py"] == 6
    # five shared of a mean of (6 + 5) / 2 revisions; a-c shares only one
    assert pairs == [(5, 91, "dw/a.py", "dw/b.py")]


def test_a_changeset_too_large_to_mean_anything_is_skipped(tmp_path):
    module = _load()
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    files = [f"dw/m{i}.py" for i in range(module.COUPLING_MAX_CHANGESET + 1)]
    for i in range(5):
        _commit(tmp_path, files, f"sweep {i}")
    revisions, pairs = module.coupling(tmp_path, "HEAD")
    assert pairs == [] and not revisions


def test_hotspots_multiply_churn_by_total_complexity():
    import collections

    revisions = collections.Counter({"dw/a.py": 3, "dw/b.py": 10})
    functions = [(5, "dw/a.py:1", "f"), (4, "dw/a.py:9", "g"), (1, "dw/b.py:1", "h")]
    assert _load().hotspots(revisions, functions) == [
        (27, 3, 9, "dw/a.py"),
        (10, 10, 1, "dw/b.py"),
    ]


def test_a_svelte_file_counts_its_code_lines(tmp_path):
    component = tmp_path / "ui" / "src" / "Card.svelte"
    component.parent.mkdir(parents=True)
    component.write_text(
        '<script lang="ts">\n'
        "  // a comment\n"
        "  let { title } = $props()\n"
        "</script>\n"
        "\n"
        "<!-- markup -->\n"
        "<h2>{title}</h2>\n"
        "<style>\n"
        "  /* a style comment */\n"
        "  h2 { margin: 0; }\n"
        "</style>\n"
    )
    counts = _load().sloc(tmp_path)
    assert counts["UI (ui/src, tests excluded)"] == 7
