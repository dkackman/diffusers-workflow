#!/usr/bin/env python
"""The stabilization gate report: every metric, one column per gate.

Each ref is measured by *this* script's rules over an archive of that ref,
never by the script as it stood then, so every column counts the same way.
Default columns: before Phase 0 (3afd70e9), every stabilization-gate-N
tag, and HEAD when it is not the newest tag. Markdown on stdout, for the
Gate reports section of docs/stabilization/ROADMAP.md.

Reports, never gates: nothing here fails a build.
"""

import argparse
import ast
import collections
import itertools
import pathlib
import statistics
import subprocess
import sys
import tarfile
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from arch_metrics import EXCLUDED, complexities, import_graph, measure  # noqa: E402

BEFORE_PHASE_0 = ("Before Phase 0", "3afd70e9")
ARCHIVED = ["dw", "dw_mcp", "tests", "ui/src", ":(glob)**/CLAUDE.md"]
RATCHETS = [
    ("modules", "Engine + MCP modules"),
    ("modules_over_1000_lines", "Modules over 1,000 lines"),
    ("functions_over_150_lines", "Functions over 150 lines"),
    ("complex_functions", "Functions over cyclomatic complexity 15"),
    ("import_cycles", "Import cycles"),
    ("modules_in_import_cycles", "Modules inside import cycles"),
    ("duplicate_blocks", "Duplicate-code blocks, cross-file"),
    ("prefix_literals", "Reference-prefix literals"),
    ("test_dw_patch_targets", 'Test `patch("dw...")` targets'),
    ("claude_md_lines", "CLAUDE.md lines, all files"),
]
# Layers for SLOC: reference only. Tests are counted in no layer
LAYERS = [
    ("Engine", "dw", lambda p: "server" not in p.parts),
    ("API (dw/server + dw_mcp)", "dw/server", lambda p: True),
    ("API (dw/server + dw_mcp)", "dw_mcp", lambda p: True),
    ("UI (ui/src, tests excluded)", "ui/src", lambda p: ".test." not in p.name),
]
SLOC_SUFFIXES = (".py", ".ts", ".svelte", ".js", ".css")
# Instability is measured per package, dw's own modules standing as "core"
PACKAGES = [
    ("dw_mcp", "dw_mcp"),
    ("dw.server", "dw.server"),
    ("dw.tasks", "dw.tasks"),
    ("dw.pipeline_processors", "dw.pipeline_processors"),
]
LCOM4_CLASSES = [
    ("dw/workflow.py", "Workflow"),
    ("dw/pipeline_processors/pipeline.py", "Pipeline"),
    ("dw/result.py", "Result"),
    ("dw/server/jobs.py", "JobManager"),
]
COUPLING_SINCE = "2026-08-01"
COUPLING_MIN_SHARED = 5
COUPLING_MAX_CHANGESET = 30
TOP = 10


def git(*args, root):
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def default_refs(root):
    refs = [BEFORE_PHASE_0]
    tags = git("tag", "--list", "stabilization-gate-*", "--sort=creatordate", root=root)
    for tag in tags.split():
        refs.append((f"Gate {tag.rsplit('-', 1)[1]}", tag))
    head = git("rev-parse", "HEAD", root=root).strip()
    if git("rev-parse", refs[-1][1] + "^{commit}", root=root).strip() != head:
        refs.append(("HEAD", head[:8]))
    return refs


def extract(ref, root, into):
    # tarfile's filter="data" needs Python 3.10.12+ / 3.11.4+ / 3.12+
    archive = subprocess.run(
        ["git", "archive", "--format=tar", ref, "--", *ARCHIVED],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    with tempfile.TemporaryFile() as stream:
        stream.write(archive)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as tar:
            tar.extractall(into, filter="data")


def sloc(tree):
    from pygount import SourceAnalysis

    counts = collections.Counter()
    for layer, folder, keep in LAYERS:
        for path in sorted((tree / folder).rglob("*")):
            if (
                path.is_file()
                and path.suffix in SLOC_SUFFIXES
                and keep(path.relative_to(tree))
                and not any(part in EXCLUDED for part in path.parts)
            ):
                counts[layer] += SourceAnalysis.from_file(str(path), layer).code_count
    return counts


def package_of(module):
    for name, prefix in PACKAGES:
        if module == prefix or module.startswith(prefix + "."):
            return name
    return "dw (core)"


def instability(edges):
    """Ca, Ce and I = Ce / (Ca + Ce) per package, counted in modules: Ca is
    how many modules outside the package import one inside it, Ce how many
    inside import one outside."""
    afferent = collections.defaultdict(set)
    efferent = collections.defaultdict(set)
    for importer, imported in edges:
        source, target = package_of(importer), package_of(imported)
        if source != target:
            efferent[source].add(importer)
            afferent[target].add(importer)
    result = {}
    for name in sorted(set(afferent) | set(efferent)):
        ca, ce = len(afferent[name]), len(efferent[name])
        result[name] = (ca, ce, round(ce / (ca + ce), 2) if ca + ce else 0.0)
    return result


def lcom4(tree, path, class_name):
    """LCOM4: connected components among a class's methods, two methods
    joined when they share a `self.` attribute or one calls the other. 1 is
    cohesive; more means the class is that many classes. Reported with the
    method count, since a split shows as fewer methods before it shows as
    fewer components. None when absent."""
    source = (tree / path).read_text(encoding="utf-8")
    node = next(
        (
            n
            for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.ClassDef) and n.name == class_name
        ),
        None,
    )
    if node is None:
        return None
    methods = {
        m.name: m
        for m in node.body
        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    touches = {}
    for name, method in methods.items():
        used = set()
        for sub in ast.walk(method):
            if (
                isinstance(sub, ast.Attribute)
                and isinstance(sub.value, ast.Name)
                and sub.value.id == "self"
            ):
                used.add(sub.attr)
        touches[name] = used
    parent = {name: name for name in methods}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for a, b in itertools.combinations(methods, 2):
        shared = (touches[a] & touches[b]) - set(methods)
        calls = b in touches[a] or a in touches[b]
        if shared or calls:
            parent[find(a)] = find(b)
    return f"{len({find(name) for name in methods})} ({len(methods)} methods)"


def coupling(root, ref):
    """File pairs that change together, code-maat's way: non-merge commits
    since COUPLING_SINCE touching at most COUPLING_MAX_CHANGESET engine
    files, pairs sharing at least COUPLING_MIN_SHARED commits, degree =
    shared / mean(revisions of each)."""
    log = git(
        "log",
        ref,
        "--no-merges",
        f"--since={COUPLING_SINCE}",
        "--name-only",
        "--format=@@%h",
        "--",
        "dw",
        "dw_mcp",
        root=root,
    )
    revisions = collections.Counter()
    shared = collections.Counter()
    for block in log.split("@@")[1:]:
        files = sorted(
            {
                f
                for f in block.split()
                if f.endswith(".py") and not any(p in EXCLUDED for p in f.split("/"))
            }
        )
        if not files or len(files) > COUPLING_MAX_CHANGESET:
            continue
        revisions.update(files)
        shared.update(itertools.combinations(files, 2))
    pairs = [
        (count, round(100 * count / ((revisions[a] + revisions[b]) / 2)), a, b)
        for (a, b), count in shared.items()
        if count >= COUPLING_MIN_SHARED
    ]
    return revisions, sorted(pairs, key=lambda p: (-p[1], -p[0], p[2]))[:TOP]


def hotspots(revisions, functions):
    """Churn (revisions in the coupling window) x the file's total
    cyclomatic complexity."""
    total = collections.Counter()
    for cc, where, _ in functions:
        total[where.rsplit(":", 1)[0]] += cc
    rows = [
        (revisions[f] * total[f], revisions[f], total[f], f)
        for f in total
        if revisions[f]
    ]
    return sorted(rows, reverse=True)[:TOP]


def column(root, ref):
    with tempfile.TemporaryDirectory() as into:
        tree = pathlib.Path(into)
        extract(ref, root, tree)
        values = measure(tree)
        functions = complexities(tree)
        edges = import_graph(tree)["edges"]
        layers = sloc(tree)
        cohesion = {f"{cls}": lcom4(tree, path, cls) for path, cls in LCOM4_CLASSES}
    scores = [cc for cc, _, _ in functions]
    values.update(
        {
            "over_30": sum(cc > 30 for cc in scores),
            "functions": len(scores),
            "median": statistics.median(scores),
            "mean": round(statistics.mean(scores), 2),
            **{
                f"over_{t}_dist": sum(cc > t for cc in scores) for t in (10, 15, 20, 30)
            },
            "sloc": layers,
            "instability": instability(edges),
            "lcom4": cohesion,
        }
    )
    return values, functions


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def report(root, refs):
    columns = [(label, ref, *column(root, ref)) for label, ref in refs]
    headers = ["Metric (lower is better)"] + [
        f"{label} (`{ref}`)" for label, ref, _, _ in columns
    ]
    rows = [[title] + [c[2].get(key) for c in columns] for key, title in RATCHETS]
    rows.insert(
        4,
        ["Functions over cyclomatic complexity 30"]
        + [c[2]["over_30"] for c in columns],
    )
    out = ["#### Metrics", "", table(headers, rows), ""]

    dist = [
        ["Functions"] + [c[2]["functions"] for c in columns],
        ["Median complexity"] + [c[2]["median"] for c in columns],
        ["Mean complexity"] + [c[2]["mean"] for c in columns],
    ] + [
        [f"Over {t}"] + [c[2][f"over_{t}_dist"] for c in columns]
        for t in (10, 15, 20, 30)
    ]
    out += [
        "#### Complexity distribution (ruff C901)",
        "",
        table(["", *headers[1:]], dist),
        "",
    ]

    layer_names = list(dict.fromkeys(name for name, _, _ in LAYERS))
    out += [
        "#### SLOC by layer (pygount code lines; reference only)",
        "",
        table(
            ["Layer", *headers[1:]],
            [[n] + [c[2]["sloc"][n] for c in columns] for n in layer_names],
        ),
        "",
    ]
    packages = sorted({p for c in columns for p in c[2]["instability"]})
    out += [
        "#### Package instability, I = Ce / (Ca + Ce)",
        "",
        table(
            ["Package", *headers[1:]],
            [
                [p]
                + [
                    "Ca {} / Ce {} / I {}".format(*c[2]["instability"][p])
                    if p in c[2]["instability"]
                    else "-"
                    for c in columns
                ]
                for p in packages
            ],
        ),
        "",
        "#### LCOM4 (components; 1 = cohesive)",
        "",
        table(
            ["Class", *headers[1:]],
            [[cls] + [c[2]["lcom4"][cls] for c in columns] for _, cls in LCOM4_CLASSES],
        ),
        "",
    ]
    label, ref, _, functions = columns[-1]
    out += [
        f"#### Ten most complex functions at {label}",
        "",
        table(
            ["Complexity", "Function"],
            [[cc, f"{where} `{name}`"] for cc, where, name in functions[:TOP]],
        ),
        "",
    ]
    revisions, pairs = coupling(root, ref)
    out += [
        f"#### Change coupling at {label} (commits since {COUPLING_SINCE})",
        "",
        table(["Shared commits", "Degree %", "File", "File"], pairs),
        "",
        f"#### Hotspots at {label} (churn x total complexity)",
        "",
        table(
            ["Score", "Commits", "Complexity", "File"], hotspots(revisions, functions)
        ),
        "",
    ]
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=pathlib.Path(__file__).resolve().parent.parent,
        type=pathlib.Path,
    )
    parser.add_argument(
        "refs",
        nargs="*",
        help="LABEL=REF columns; default: before Phase 0, every gate tag, HEAD",
    )
    args = parser.parse_args(argv)
    refs = (
        [tuple(r.split("=", 1)) for r in args.refs]
        if args.refs
        else default_refs(args.root)
    )
    print(report(args.root, refs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
