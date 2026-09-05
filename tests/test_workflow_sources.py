"""The workflow search path: reads span every root, writes reach only the
front, and a read-only source cannot be written to or deleted from."""

import json
import os

import pytest

from dw.workflow_sources import (
    BUILTIN_ORIGIN,
    EXAMPLES_ORIGIN,
    WORKSPACE_ORIGIN,
    builtin_root,
    find_workflow,
    listing,
    resolve_in_source,
    source_for_path,
    workflow_names,
    workflow_sources,
    writable_source,
)


@pytest.fixture
def roots(tmp_path):
    """A writable workspace library and a read-only examples tree, with one
    name present in both."""
    workspace = tmp_path / "studio" / "workflows"
    (workspace / "mine").mkdir(parents=True)
    (workspace / "Shared.json").write_text(json.dumps({"id": "mine-shared"}))
    (workspace / "mine" / "Solo.json").write_text(json.dumps({"id": "solo"}))

    examples = tmp_path / "repo" / "workflows"
    (examples / "ltx2").mkdir(parents=True)
    (examples / "Shared.json").write_text(json.dumps({"id": "example-shared"}))
    (examples / "ltx2" / "Gyre.json").write_text(json.dumps({"id": "gyre"}))
    return workspace, examples


class TestSources:
    def test_the_writable_root_comes_first(self, roots):
        workspace, examples = roots
        sources = workflow_sources(str(workspace), [str(examples)])
        assert [s.origin for s in sources] == [WORKSPACE_ORIGIN, EXAMPLES_ORIGIN]
        assert [s.writable for s in sources] == [True, False]
        assert writable_source(sources).root == str(workspace)

    def test_a_repeated_root_is_writable_once(self, roots):
        # The checkout-as-workspace case: the same directory named as both
        # the library and the examples must not answer two ways
        workspace, _examples = roots
        sources = workflow_sources(str(workspace), [str(workspace)])
        assert len(sources) == 1
        assert sources[0].writable

    def test_the_packaged_workflows_are_off_the_path_by_default(self, roots):
        workspace, _examples = roots
        assert builtin_root() not in [s.root for s in workflow_sources(str(workspace))]
        with_builtins = workflow_sources(str(workspace), include_builtin=True)
        assert [s.origin for s in with_builtins][-1] == BUILTIN_ORIGIN

    def test_a_missing_root_is_simply_empty(self, tmp_path):
        sources = workflow_sources(str(tmp_path / "nothing-here"))
        assert workflow_names(sources[0].root) == []


class TestResolution:
    def test_reads_span_every_root(self, roots):
        workspace, examples = roots
        sources = workflow_sources(str(workspace), [str(examples)])
        path, source = find_workflow(sources, "ltx2/Gyre")
        assert source.origin == EXAMPLES_ORIGIN
        assert path == str(examples / "ltx2" / "Gyre.json")

    def test_the_front_of_the_path_shadows_the_rest(self, roots):
        workspace, examples = roots
        sources = workflow_sources(str(workspace), [str(examples)])
        path, source = find_workflow(sources, "Shared")
        assert source.origin == WORKSPACE_ORIGIN
        assert json.loads(open(path).read())["id"] == "mine-shared"

    def test_a_listing_names_each_workflow_once(self, roots):
        workspace, examples = roots
        found = listing(workflow_sources(str(workspace), [str(examples)]))
        assert sorted(found) == ["Shared", "ltx2/Gyre", "mine/Solo"]
        assert found["Shared"].origin == WORKSPACE_ORIGIN
        assert found["ltx2/Gyre"].origin == EXAMPLES_ORIGIN

    def test_an_unknown_name_resolves_nowhere(self, roots):
        workspace, examples = roots
        assert (
            find_workflow(workflow_sources(str(workspace), [str(examples)]), "Nope")[0]
            is None
        )

    @pytest.mark.parametrize("name", ["../outside", "/etc/passwd", "a/../../escape"])
    def test_a_name_cannot_traverse_out_of_a_source(self, roots, name):
        workspace, examples = roots
        sources = workflow_sources(str(workspace), [str(examples)])
        assert find_workflow(sources, name) == (None, None)
        assert resolve_in_source(sources[0], name, allow_create=True) is None

    def test_a_path_knows_which_source_it_belongs_to(self, roots, tmp_path):
        workspace, examples = roots
        sources = workflow_sources(str(workspace), [str(examples)])
        assert source_for_path(
            sources, str(examples / "ltx2" / "Gyre.json")
        ).origin == (EXAMPLES_ORIGIN)
        assert source_for_path(sources, str(tmp_path / "elsewhere.json")) is None


class TestNames:
    def test_names_are_relative_and_slash_separated(self, roots):
        _workspace, examples = roots
        assert workflow_names(str(examples)) == ["Shared", "ltx2/Gyre"]

    def test_non_json_files_are_not_workflows(self, roots):
        _workspace, examples = roots
        (examples / "notes.txt").write_text("hello")
        assert "notes" not in workflow_names(str(examples))
        assert os.path.isfile(examples / "notes.txt")
