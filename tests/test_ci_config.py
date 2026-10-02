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
