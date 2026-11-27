"""The release job stamps its version into the bootstrap scripts it attaches
(scripts/stamp_bootstrap.py, run from .github/workflows/ci.yml)."""

import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "dw" / "install_scripts"
CI = REPO / ".github" / "workflows" / "ci.yml"


def _stamp():
    spec = importlib.util.spec_from_file_location(
        "stamp_bootstrap", REPO / "scripts" / "stamp_bootstrap.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["bootstrap.sh", "bootstrap.ps1"])
def test_each_placeholder_line_exists_exactly_once(name):
    """The repo copy holds the exact line the release step rewrites, once."""
    placeholder, _ = _stamp().PLACEHOLDERS[Path(name).suffix]
    lines = (SCRIPTS / name).read_text().split("\n")
    assert lines.count(placeholder) == 1


@pytest.mark.parametrize("name", ["bootstrap.sh", "bootstrap.ps1"])
def test_stamping_the_repo_copy_sets_its_default_version(name, tmp_path):
    copy = tmp_path / name
    copy.write_text((SCRIPTS / name).read_text())
    _stamp().stamp_file(copy, "1.2.3-rc1")
    stamped = copy.read_text()
    if name.endswith(".sh"):
        assert re.search(r'^  VERSION="1\.2\.3-rc1"$', stamped, re.M)
        assert 'VERSION=""' not in stamped
    else:
        assert re.search(r'^    \[string\]\$Version = "1\.2\.3-rc1",$', stamped, re.M)
        assert '$Version = ""' not in stamped
    # Only that one line moved
    before = (SCRIPTS / name).read_text().split("\n")
    after = stamped.split("\n")
    assert len(before) == len(after)
    assert sum(a != b for a, b in zip(before, after)) == 1


@pytest.mark.parametrize("count", [0, 2])
def test_a_missing_or_doubled_placeholder_fails(count):
    placeholder, _ = _stamp().PLACEHOLDERS[".sh"]
    text = "\n".join(["#!/bin/bash"] + [placeholder] * count + ["exit 0"])
    with pytest.raises(ValueError, match=f"found {count}"):
        _stamp().stamp_text(text, ".sh", "1.2.3")


def test_a_renamed_placeholder_fails_the_command(tmp_path):
    sh = tmp_path / "bootstrap.sh"
    ps1 = tmp_path / "bootstrap.ps1"
    sh.write_text('  DW_VERSION=""\n')
    ps1.write_text((SCRIPTS / "bootstrap.ps1").read_text())
    assert _stamp().main(["1.2.3", str(sh), str(ps1)]) == 1


def test_a_non_semver_version_is_refused():
    placeholder, _ = _stamp().PLACEHOLDERS[".sh"]
    with pytest.raises(ValueError, match="semver"):
        _stamp().stamp_text(placeholder, ".sh", '1.2"; rm -rf /')


def test_the_command_needs_both_scripts(tmp_path):
    sh = tmp_path / "bootstrap.sh"
    sh.write_text((SCRIPTS / "bootstrap.sh").read_text())
    assert _stamp().main(["1.2.3", str(sh)]) == 2


def test_the_release_step_stamps_both_scripts_before_the_checksums():
    ci = CI.read_text()
    stamp = re.search(
        r'python3 scripts/stamp_bootstrap\.py "\$\{GITHUB_REF_NAME#v\}" '
        r"dist/bootstrap\.sh dist/bootstrap\.ps1\n",
        ci,
    )
    assert stamp, "the release step must stamp both copies with the tag"
    copy = ci.index(
        "cp dw/install_scripts/bootstrap.sh dw/install_scripts/bootstrap.ps1 dist/"
    )
    sums = ci.index("sha256sum -- *")
    assert copy < stamp.start() < sums
