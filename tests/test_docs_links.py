"""Documentation points at example workflows that exist.

The examples tree has subfolders, so a doc's path to an example is a path that
can go stale the moment a file moves - and nothing else in the suite reads the
docs. Dated design records under docs/superpowers describe the tree as it was
when they were written and are left out.
"""

import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDED = ("venv", "node_modules", os.path.join("docs", "superpowers"))

# 'examples/ZImage.json' or '../examples/flux/FluxDev.json', in prose or a link
EXAMPLE_PATH = re.compile(r"(?:\.\./)?examples/([A-Za-z0-9_.\-/]+\.json)")


def get_doc_files():
    """Every markdown file in the repository, as repo-relative paths."""
    documents = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        relative_root = os.path.relpath(root, REPO_ROOT)
        if any(part in relative_root for part in EXCLUDED):
            continue
        documents.extend(
            os.path.relpath(os.path.join(root, name), REPO_ROOT)
            for name in files
            if name.endswith(".md")
        )
    return sorted(documents)


@pytest.mark.parametrize("doc_file", get_doc_files())
def test_doc_example_references_exist(doc_file):
    """Test that every example workflow a document names still exists"""
    with open(os.path.join(REPO_ROOT, doc_file), encoding="utf-8") as file:
        text = file.read()

    missing = [
        reference
        for reference in sorted(set(EXAMPLE_PATH.findall(text)))
        if not os.path.isfile(os.path.join(REPO_ROOT, "examples", reference))
    ]
    assert not missing, (
        f"{doc_file} references examples that do not exist: "
        f"{', '.join('examples/' + name for name in missing)}"
    )
