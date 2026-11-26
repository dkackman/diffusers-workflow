"""The e2e fixture server refuses a bundle older than the UI's sources (#1686)."""

import importlib.util
import os
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "ui" / "e2e" / "bundle_check.py"
spec = importlib.util.spec_from_file_location("bundle_check", SCRIPT)
bundle_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle_check)


def write(path, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    os.utime(path, (mtime, mtime))


@pytest.fixture
def ui(tmp_path):
    write(tmp_path / "index.html", 1000)
    write(tmp_path / "src" / "lib" / "a.ts", 1000)
    write(tmp_path / "dist" / "index.html", 2000)
    return tmp_path


def test_a_bundle_newer_than_its_sources_is_served(ui):
    assert bundle_check.bundle_problem(str(ui)) is None


def test_a_missing_bundle_is_refused(ui):
    (ui / "dist" / "index.html").unlink()
    assert "npm run build" in bundle_check.bundle_problem(str(ui))


@pytest.mark.parametrize("source", ["index.html", "vite.config.ts", "src/lib/a.ts"])
def test_a_source_changed_since_the_build_is_named(ui, source):
    write(ui / source, 3000)
    problem = bundle_check.bundle_problem(str(ui))
    assert f"ui/{source}" in problem
    assert "npm run build" in problem


def test_a_unit_test_edit_leaves_the_bundle_current(ui):
    write(ui / "src" / "lib" / "a.test.ts", 3000)
    assert bundle_check.bundle_problem(str(ui)) is None
