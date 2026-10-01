"""One resolver for a sub-workflow step's path, and every site that asks it.

`resolve_sub_workflow_reference` owns the preamble (the builtin branch, the
catalog-root fallback, the search path, the confinement check), so a path a run
can open is one validation can open, one realization can digest and one cost
lookup can find - and the other way round.
"""

import json
import os

import pytest

from dw.library import (
    SubWorkflowNotFound,
    builtin_root,
    resolve_sub_workflow_reference,
)
from dw.security import PathTraversalError
from dw.validation import sub_workflow_errors
from dw.workflow import Workflow

with open(os.path.join(builtin_root(), "test.json")) as file:
    CHILD = {**json.load(file), "id": "child"}


@pytest.fixture
def catalog(tmp_path):
    """workflows/templates/ beside workflows/models/, a decoy outside it and
    a parent file under templates/."""
    root = tmp_path / "workflows"
    (root / "templates").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "models" / "Child.json").write_text(json.dumps(CHILD))
    (tmp_path / "Outside.json").write_text(json.dumps(CHILD))
    return root, str(root / "templates" / "parent.json")


def parent_in(catalog, confined=False):
    root, file_spec = catalog
    return Workflow(
        {"id": "parent", "steps": []},
        str(root.parent / "outputs"),
        file_spec,
        str(root) if confined else None,
    )


class TestTheResolver:
    def test_a_catalog_name_resolves_through_the_catalog_root(self, catalog):
        """'models/Child' is no path beside templates/parent.json: it is found
        only because an unconfined run falls back to the catalog root."""
        root, file_spec = catalog
        path, confine_to = resolve_sub_workflow_reference(
            "models/Child", file_spec, None
        )
        assert path == str(root / "models" / "Child.json")
        assert confine_to == str(root)

    def test_a_builtin_resolves_in_the_packaged_root(self, catalog):
        _root, file_spec = catalog
        path, confine_to = resolve_sub_workflow_reference(
            "builtin:test.json", file_spec, None
        )
        assert path.startswith(builtin_root())
        assert confine_to == builtin_root()

    def test_only_the_prefix_is_stripped_from_a_builtin_name(self, catalog):
        """A name carrying 'builtin:' again was read with it removed
        everywhere, which made 'builtin:builtin:test.json' load test.json;
        the second one is part of the name, which names no file."""
        _root, file_spec = catalog
        with pytest.raises(SubWorkflowNotFound) as exc_info:
            resolve_sub_workflow_reference("builtin:builtin:test.json", file_spec, None)
        assert "builtin:test.json" in str(exc_info.value)

    def test_a_missing_builtin_is_not_found(self, catalog):
        _root, file_spec = catalog
        with pytest.raises(SubWorkflowNotFound) as exc_info:
            resolve_sub_workflow_reference("builtin:absent.json", file_spec, None)
        assert "absent.json" in str(exc_info.value)

    @pytest.mark.parametrize("confined", [False, True])
    def test_a_path_escaping_its_root_is_refused(self, catalog, confined):
        root, file_spec = catalog
        with pytest.raises(PathTraversalError):
            resolve_sub_workflow_reference(
                "../../Outside.json", file_spec, str(root) if confined else None
            )


class TestEverySiteReachesIt:
    def test_create_step_action(self, catalog):
        parent = parent_in(catalog)
        step = {"name": "child", "workflow": {"path": "models/Child"}}

        child = parent.create_step_action(step, {}, {}, 42, "cpu")

        assert child.name == "child"

    def test_create_step_action_refuses_a_climb_like_validation_does(self, catalog):
        parent = parent_in(catalog, confined=True)
        step = {"name": "child", "workflow": {"path": "../../Outside.json"}}

        with pytest.raises(PathTraversalError):
            parent.create_step_action(step, {}, {}, 42, "cpu")

    def test_open_sub_workflow(self, catalog):
        root, _ = catalog
        child, resolved = parent_in(catalog).open_sub_workflow("models/Child")
        assert resolved == str(root / "models" / "Child.json")
        assert child.name == "child"

    def test_open_sub_workflow_takes_an_already_resolved_path(self, catalog):
        parent = parent_in(catalog)
        resolved, root = parent.resolve_sub_workflow_path("models/Child")

        child, again = parent.open_sub_workflow("not-resolved-again", (resolved, root))

        assert again == resolved
        assert child.name == "child"

    def test_sub_workflow_errors_resolves_a_catalog_name(self, catalog):
        parent = parent_in(catalog)
        expanded = {"steps": [{"name": "c", "workflow": {"path": "models/Child"}}]}
        assert sub_workflow_errors(parent, expanded) == []

    def test_sub_workflow_errors_names_a_missing_builtin(self, catalog):
        parent = parent_in(catalog)
        expanded = {
            "steps": [{"name": "c", "workflow": {"path": "builtin:absent.json"}}]
        }
        (error,) = sub_workflow_errors(parent, expanded)
        assert "absent.json" in error["message"]

    def test_read_sub_workflow_falls_back_to_the_catalog_root(self, catalog):
        from dw.realize import read_sub_workflow

        _root, file_spec = catalog
        raw = read_sub_workflow("models/Child", str(file_spec).rsplit("/", 1)[0], None)
        assert json.loads(raw) == CHILD

    def test_the_observed_cost_lookup_falls_back_to_the_catalog_root(
        self, catalog, monkeypatch
    ):
        from types import SimpleNamespace

        from dw.server.routes import jobs

        root, file_spec = catalog
        seen = {}
        monkeypatch.setattr(
            jobs, "build_plan", lambda *a, **kw: seen.update(kw) or {"plan": True}
        )
        monkeypatch.setattr(
            jobs,
            "observed_for_name",
            lambda state, name, *a, **kw: ("observed", name),
        )
        candidate = SimpleNamespace(
            file_spec=file_spec, workflow_dir=None, workflow_definition={"steps": []}
        )
        request = SimpleNamespace(workflow_path=None, arguments={})
        workspace = SimpleNamespace(
            name="default", outputs=None, assets=None, prompts=None, workflows=str(root)
        )
        jobs._validation_plan(
            SimpleNamespace(job_manager=None),
            candidate,
            request,
            workspace,
            None,
            None,
            False,
        )

        found = seen["observed_for_child"]("models/Child", CHILD)

        assert found == ("observed", "models/Child")
