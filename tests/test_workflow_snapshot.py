"""The worker builds a job's Workflow from the snapshot admission checked
(`workflow_from_snapshot`), never from the file again. The snapshot has to
be the same Workflow the file would have made - every name a run derives
from it (its identity, its output subfolder, its sub-workflow root) - and
it has to be confined the way the file was."""

import glob
import os

import pytest

from dw.runs import workflow_identity
from dw.security import PathTraversalError
from dw.workflow import (
    workflow_from_file,
    workflow_from_snapshot,
    workflow_output_subfolder,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOWS = os.path.join(REPO, "workflows")
TEMPLATES = sorted(
    glob.glob(os.path.join(WORKFLOWS, "templates", "**", "*.json"), recursive=True)
)


@pytest.mark.parametrize(
    "path", TEMPLATES, ids=[os.path.relpath(p, WORKFLOWS) for p in TEMPLATES]
)
def test_a_snapshot_is_the_workflow_its_file_makes(path, tmp_path):
    # Two independent reads of the one file: the snapshot is built from what
    # the first recorded and compared with the second, so a snapshot that only
    # echoes the instance it was built from cannot pass
    admitted = workflow_from_file(path, str(tmp_path), WORKFLOWS)
    snapshot = workflow_from_snapshot(
        admitted.workflow_definition, str(tmp_path), admitted.file_spec, WORKFLOWS
    )
    loaded = workflow_from_file(path, str(tmp_path), WORKFLOWS)
    assert loaded is not admitted

    assert snapshot.workflow_definition == loaded.workflow_definition
    assert snapshot.file_spec == loaded.file_spec
    assert snapshot.workflow_dir == loaded.workflow_dir
    assert snapshot.output_dir == loaded.output_dir
    assert snapshot.name == loaded.name
    assert workflow_output_subfolder(snapshot.file_spec) == workflow_output_subfolder(
        loaded.file_spec
    )
    assert workflow_identity(snapshot.file_spec, snapshot.name) == workflow_identity(
        loaded.file_spec, loaded.name
    )


def test_the_templates_are_found():
    # A glob that matched nothing would make the parity test above vacuous
    assert TEMPLATES


def test_a_snapshot_outside_its_workflow_dir_is_refused(tmp_path):
    root = tmp_path / "workflows"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    with pytest.raises(PathTraversalError, match="outside allowed directory"):
        workflow_from_snapshot(
            {"id": "escaped", "steps": []},
            str(tmp_path / "outputs"),
            str(elsewhere / "escaped.json"),
            str(root),
        )


def test_a_snapshot_never_opens_its_file(tmp_path):
    """The file a snapshot names may be gone by the time the job runs - the
    definition is what admission checked, so nothing reads the path."""
    root = tmp_path / "workflows"
    root.mkdir()
    definition = {"id": "gone", "steps": []}

    workflow = workflow_from_snapshot(
        definition, str(tmp_path / "outputs"), str(root / "gone.json"), str(root)
    )

    assert workflow.workflow_definition == definition
    assert workflow.file_spec == str(root / "gone.json")


def test_a_relative_file_spec_is_made_absolute_without_a_workflow_dir(
    tmp_path, monkeypatch
):
    """With a workflow_dir the file_spec is confined; with none it is still
    normalized, so every name derived from it matches admission's."""
    monkeypatch.chdir(tmp_path)

    workflow = workflow_from_snapshot(
        {"id": "relative", "steps": []}, str(tmp_path / "outputs"), "wf/relative.json"
    )

    assert workflow.file_spec == os.path.abspath("wf/relative.json")
    assert os.path.isabs(workflow.file_spec)
