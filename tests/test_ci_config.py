"""What .github/workflows/ci.yml must keep doing that no run of it shows."""

from pathlib import Path

import yaml

CI = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"


def _e2e_condition():
    return " ".join(yaml.safe_load(CI.read_text())["jobs"]["e2e"]["if"].split())


def test_e2e_runs_once_per_develop_push_while_a_release_pr_is_open():
    # The release PR's head is develop: each push to it triggers the push
    # run and the PR's synchronize run on the same commit. The push run is
    # the one that holds when no PR is open, so the PR run steps aside -
    # for this repository's develop only, never a fork's
    condition = _e2e_condition()
    assert (
        "github.event_name == 'push' && github.ref == 'refs/heads/develop'" in condition
    )
    assert (
        "!(github.head_ref == 'develop' && "
        "github.event.pull_request.head.repo.full_name == github.repository)"
    ) in condition


def test_pypi_publishes_before_the_github_release():
    """The release carries a bootstrap stamped to install `==<tag>` from
    PyPI, and `releases/latest/download/bootstrap.sh` is the documented
    one-liner: a release that exists before its PyPI upload leaves that
    command failing. The tag check gates both, so a mis-tagged commit
    publishes to neither."""
    jobs = yaml.safe_load(CI.read_text())["jobs"]

    def needs(job):
        wanted = jobs[job]["needs"]
        return [wanted] if isinstance(wanted, str) else wanted

    assert "pypi" in needs("release")
    assert "release" not in needs("pypi")
    assert set(needs("pypi")) == {"wheel", "tag-check"}
    assert jobs["tag-check"]["if"] == "startsWith(github.ref, 'refs/tags/v')"
    check = " ".join(step.get("run", "") for step in jobs["tag-check"]["steps"])
    assert "pyproject version" in check
