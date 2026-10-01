"""The library search path: reads span every root, writes reach only the
front, and a read-only root cannot be written to or deleted from."""

import json
import os

import pytest

from dw.security import SecurityError
from dw.library import (
    BUILTIN_ORIGIN,
    COMMON_ORIGIN,
    EXAMPLES_ORIGIN,
    WORKSPACE_ORIGIN,
    LibraryPath,
    LibraryRoot,
    ReadOnlyLibraryError,
    builtin_root,
    library_path,
    resolve_sub_workflow,
    suggest_workflow_names,
    workflow_names,
)


def workflows_path(workspace, examples=(), **options):
    """The workflow search path for a workspace directory and examples."""
    return library_path(
        "workflows", None, [str(e) for e in examples], primary=str(workspace), **options
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


class TestConstruction:
    def test_the_writable_root_comes_first(self, roots):
        workspace, examples = roots
        sources = workflows_path(workspace, [examples])
        assert [s.origin for s in sources.roots()] == [
            WORKSPACE_ORIGIN,
            EXAMPLES_ORIGIN,
        ]
        assert [s.writable for s in sources.roots()] == [True, False]
        assert sources.writable_root().root == str(workspace)

    def test_a_repeated_root_is_writable_once(self, roots):
        # The checkout-as-workspace case: the same directory named as both
        # the library and the examples must not answer two ways
        workspace, _examples = roots
        sources = workflows_path(workspace, [workspace]).roots()
        assert len(sources) == 1
        assert sources[0].writable

    def test_the_packaged_workflows_are_off_the_path_by_default(self, roots):
        workspace, _examples = roots
        assert builtin_root() not in [s.root for s in workflows_path(workspace).roots()]
        with_builtins = workflows_path(workspace, include_builtin=True)
        assert [s.origin for s in with_builtins.roots()][-1] == BUILTIN_ORIGIN

    def test_a_missing_root_is_simply_empty(self, tmp_path):
        sources = workflows_path(tmp_path / "nothing-here")
        assert workflow_names(sources.roots()[0].root) == []

    def test_a_missing_read_only_root_is_dropped(self, roots, tmp_path):
        workspace, examples = roots
        sources = workflows_path(workspace, [examples, tmp_path / "nothing-here"])
        assert [s.root for s in sources.roots()] == [str(workspace), str(examples)]


class TestResolution:
    def test_reads_span_every_root(self, roots):
        workspace, examples = roots
        sources = workflows_path(workspace, [examples])
        path, source = sources.find("ltx2/Gyre")
        assert source.origin == EXAMPLES_ORIGIN
        assert path == str(examples / "ltx2" / "Gyre.json")

    def test_the_front_of_the_path_shadows_the_rest(self, roots):
        workspace, examples = roots
        sources = workflows_path(workspace, [examples])
        path, source = sources.find("Shared")
        assert source.origin == WORKSPACE_ORIGIN
        assert json.loads(open(path).read())["id"] == "mine-shared"

    def test_a_listing_names_each_workflow_once(self, roots):
        workspace, examples = roots
        found, _shadowed = workflows_path(workspace, [examples]).entries()
        assert sorted(found) == ["Shared", "ltx2/Gyre", "mine/Solo"]
        assert found["Shared"].origin == WORKSPACE_ORIGIN
        assert found["ltx2/Gyre"].origin == EXAMPLES_ORIGIN

    def test_an_unknown_name_resolves_nowhere(self, roots):
        workspace, examples = roots
        assert workflows_path(workspace, [examples]).find("Nope") is None

    @pytest.mark.parametrize("name", ["../outside", "/etc/passwd", "a/../../escape"])
    def test_a_name_cannot_traverse_out_of_a_source(self, roots, name):
        workspace, examples = roots
        sources = workflows_path(workspace, [examples])
        assert sources.find(name) is None
        assert sources.path_in(sources.roots()[0], name, allow_create=True) is None

    def test_a_path_knows_which_source_it_belongs_to(self, roots, tmp_path):
        workspace, examples = roots
        sources = workflows_path(workspace, [examples])
        assert sources.root_for_path(str(examples / "ltx2" / "Gyre.json")).origin == (
            EXAMPLES_ORIGIN
        )
        assert sources.root_for_path(str(tmp_path / "elsewhere.json")) is None


class TestSuggestions:
    """#397: a caller who knows a catalog entry by its short name gets a
    pointer to the real one rather than a bare 404."""

    def test_a_unique_path_suffix_is_suggested(self, roots):
        workspace, examples = roots
        sources = workflows_path(workspace, [examples])
        assert suggest_workflow_names(sources, "Gyre") == ["ltx2/Gyre"]

    def test_a_typo_falls_back_to_a_close_spelling_match(self, roots):
        workspace, examples = roots
        sources = workflows_path(workspace, [examples])
        assert suggest_workflow_names(sources, "Shered") == ["Shared"]

    def test_nothing_close_suggests_nothing(self, roots):
        workspace, examples = roots
        sources = workflows_path(workspace, [examples])
        assert suggest_workflow_names(sources, "zzz-completely-unrelated") == []

    def test_the_real_catalog_suggests_the_full_template_path(self):
        # #397's own repro: a skill or an earlier turn names a template by
        # its short id, not its catalog path
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sources = workflows_path(
            os.path.join(repo_root, "workflows"), include_builtin=True
        )
        assert suggest_workflow_names(sources, "dialogue-short") == [
            "templates/minimax/dialogue-short"
        ]

    def test_a_typo_on_a_short_name_still_finds_the_full_catalog_path(self):
        # The tester's own follow-up: "dialog-short" scores 0.92 against the
        # entry's own name but 0.55 against the full path, so comparing
        # full paths missed a real typo entirely
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sources = workflows_path(
            os.path.join(repo_root, "workflows"), include_builtin=True
        )
        assert suggest_workflow_names(sources, "dialog-short") == [
            "templates/minimax/dialogue-short"
        ]


class TestSubWorkflowResolution:
    """A composed step's relative path is confined to the root it is handed
    back with, so a name that climbs out of the catalog is never resolved -
    and never stat'ed, which the dw/path-injection query flagged here. The
    '../models/x.json' form a template uses to reach a sibling catalog folder
    still resolves."""

    @pytest.fixture
    def catalog(self, tmp_path):
        """workflows/templates/ beside workflows/models/, and a decoy outside
        the catalog that a climbing name reaches."""
        root = tmp_path / "workflows"
        (root / "templates").mkdir(parents=True)
        (root / "models").mkdir()
        (root / "models" / "Child.json").write_text(json.dumps({"id": "child"}))
        outside = tmp_path / "Outside.json"
        outside.write_text(json.dumps({"id": "outside"}))
        return root, outside

    def test_a_climb_inside_the_confinement_resolves(self, catalog):
        root, _outside = catalog
        candidate, confine_to = resolve_sub_workflow(
            "../models/Child.json", str(root / "templates"), str(root)
        )
        assert os.path.basename(candidate) == "Child.json"
        assert confine_to.root == str(root)

    def test_a_climb_out_of_the_confinement_is_refused_not_resolved(self, catalog):
        root, outside = catalog
        assert os.path.isfile(outside), "the decoy has to exist to be reachable"
        with pytest.raises(SecurityError):
            resolve_sub_workflow(
                "../../Outside.json", str(root / "templates"), str(root)
            )

    def test_a_climb_out_refusal_says_where_it_looked(self, catalog):
        # #422: the refusal named only the rejected path, not the search
        # path it was judged against
        root, outside = catalog
        with pytest.raises(SecurityError) as exc_info:
            resolve_sub_workflow(
                "../../Outside.json", str(root / "templates"), str(root)
            )
        message = str(exc_info.value)
        assert "Looked in" in message
        assert str(root) in message
        assert str(outside) in message  # the underlying PathTraversalError
        # already names the resolved (rejected) path

    def test_an_absolute_path_outside_every_source_says_where_it_looked(self, catalog):
        root, outside = catalog
        with pytest.raises(SecurityError) as exc_info:
            resolve_sub_workflow(str(outside), str(root / "templates"), str(root))
        message = str(exc_info.value)
        assert str(outside) in message
        assert "Looked in" in message
        assert str(root) in message

    def test_an_unconfined_caller_still_confines_to_the_catalog(self, catalog):
        """No confine_to (a bare CLI run) confines to the catalog root the
        run itself would use - the nearest ancestor named 'workflows' - so
        the climb out is refused rather than stat'ed."""
        root, _outside = catalog
        with pytest.raises(SecurityError):
            resolve_sub_workflow("../../Outside.json", str(root / "templates"), None)

    def test_an_unconfined_caller_may_climb_to_a_sibling_catalog_folder(self, catalog):
        root, _outside = catalog
        candidate, _confine_to = resolve_sub_workflow(
            "../models/Child.json", str(root / "templates"), None
        )
        assert os.path.basename(candidate) == "Child.json"


class TestNames:
    def test_names_are_relative_and_slash_separated(self, roots):
        _workspace, examples = roots
        assert workflow_names(str(examples)) == ["Shared", "ltx2/Gyre"]

    def test_non_json_files_are_not_workflows(self, roots):
        _workspace, examples = roots
        (examples / "notes.txt").write_text("hello")
        assert "notes" not in workflow_names(str(examples))
        assert os.path.isfile(examples / "notes.txt")


class TestLibraryPath:
    """The rules every library shares, over temp roots: order, confinement,
    shadowing, and who may write."""

    @pytest.fixture
    def path(self, roots):
        workspace, examples = roots
        return workflows_path(workspace, [examples])

    def test_find_walks_the_roots_front_to_back(self, path, roots):
        workspace, examples = roots
        found, root = path.find("Shared")
        assert found == str(workspace / "Shared.json")
        assert root is path.roots()[0]
        # A name only a later root holds still resolves, against that root
        found, root = path.find("ltx2/Gyre")
        assert found == str(examples / "ltx2" / "Gyre.json")
        assert root is path.roots()[1]

    def test_find_takes_a_name_with_or_without_its_extension(self, path):
        assert path.find("Shared.json") == path.find("Shared")

    def test_a_symlink_leaving_its_root_is_a_miss(self, roots, tmp_path):
        # The decoy sits outside both roots; a link to it from inside the
        # workspace must not resolve, nor be listed
        workspace, examples = roots
        decoy = tmp_path / "decoy.json"
        decoy.write_text(json.dumps({"id": "decoy"}))
        os.symlink(decoy, workspace / "escape.json")
        path = workflows_path(workspace, [examples])
        assert path.find("escape") is None
        assert "escape" not in path.entries()[0]

    def test_entries_name_each_winner_once_and_report_what_it_hid(self, path):
        winners, shadowed = path.entries()
        assert list(winners) == ["Shared", "ltx2/Gyre", "mine/Solo"]
        assert winners["Shared"] is path.roots()[0]
        assert winners["ltx2/Gyre"] is path.roots()[1]
        assert [(name, root.origin, by.origin) for name, root, by in shadowed] == [
            ("Shared", EXAMPLES_ORIGIN, WORKSPACE_ORIGIN)
        ]

    def test_entries_take_the_lister_a_library_brings(self, tmp_path):
        # An asset library is listed by server code the engine cannot
        # import, so the lister is a parameter
        front = LibraryRoot(tmp_path / "a", WORKSPACE_ORIGIN, True)
        back = LibraryRoot(tmp_path / "b", EXAMPLES_ORIGIN, False)
        listed = {front.root: ["x.png", "y.png"], back.root: ["y.png", "z.png"]}
        path = LibraryPath("assets", [front, back])
        winners, shadowed = path.entries(lambda root: listed[root])
        assert winners == {"x.png": front, "y.png": front, "z.png": back}
        assert shadowed == [("y.png", back, front)]

    def test_an_asset_library_cannot_be_listed_without_a_lister(self, tmp_path):
        path = LibraryPath("assets", [LibraryRoot(tmp_path, WORKSPACE_ORIGIN, True)])
        with pytest.raises(ValueError):
            path.entries()

    def test_an_asset_name_is_taken_literally(self, tmp_path):
        root = LibraryRoot(tmp_path, WORKSPACE_ORIGIN, True)
        (tmp_path / "iris.png").write_bytes(b"png")
        path = LibraryPath("assets", [root])
        assert path.find("iris.png") == (str(tmp_path / "iris.png"), root)
        assert path.find("iris") is None

    def test_the_writable_root_is_the_front_of_the_path(self, path, roots):
        workspace, _examples = roots
        assert path.writable_root().root == str(workspace)

    def test_the_shared_root_is_asked_for_by_name(self, tmp_path):
        own = LibraryRoot(tmp_path / "own", WORKSPACE_ORIGIN, True)
        common = LibraryRoot(tmp_path / "common", COMMON_ORIGIN, True)
        path = LibraryPath("assets", [own, common])
        assert path.writable_root() is own
        assert path.writable_root(shared=True) is common
        assert LibraryPath("assets", [own]).writable_root(shared=True) is None

    def test_a_path_with_nothing_writable_has_no_writable_root(self, tmp_path):
        path = LibraryPath("workflows", [LibraryRoot(tmp_path, EXAMPLES_ORIGIN, False)])
        assert path.writable_root() is None

    def test_require_writable_refuses_a_read_only_root(self, path):
        own, example = path.roots()
        assert path.require_writable(own, "Shared") is own
        with pytest.raises(ReadOnlyLibraryError) as refusal:
            path.require_writable(example, "ltx2/Gyre")
        assert refusal.value.name == "ltx2/Gyre"
        assert refusal.value.root is example

    def test_an_unknown_kind_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            LibraryPath("sounds", [])


class TestSubWorkflowOrigin:
    """D8: resolve_sub_workflow used to build a throwaway `examples`
    container for every root, the run's own writable one included, and hand
    back a bare string. The root it returns says what it is."""

    def test_a_sub_workflow_from_the_writable_root_is_tagged_workspace(self, roots):
        workspace, _examples = roots
        _candidate, root = resolve_sub_workflow(
            "Shared", str(workspace), str(workspace)
        )
        assert root.origin == WORKSPACE_ORIGIN
        assert root.writable
        assert root.root == str(workspace)

    def test_a_sub_workflow_from_a_fallback_is_tagged_examples(
        self, roots, monkeypatch
    ):
        workspace, examples = roots
        monkeypatch.setenv("DW_WORKFLOW_PATH", str(examples))
        candidate, root = resolve_sub_workflow(
            "ltx2/Gyre", str(workspace), str(workspace)
        )
        assert candidate == str(examples / "ltx2" / "Gyre.json")
        assert root.origin == EXAMPLES_ORIGIN
        assert not root.writable


class TestTaggedByWhatItIs:
    """D8, finished: a run confined to an examples directory is confined to a
    read-only root, and the root it hands back says so."""

    def test_a_run_confined_to_an_examples_dir_is_tagged_examples(
        self, roots, monkeypatch
    ):
        _workspace, examples = roots
        monkeypatch.setenv("DW_WORKFLOW_PATH", str(examples))
        candidate, root = resolve_sub_workflow("Shared", str(examples), str(examples))
        assert candidate == str(examples / "Shared.json")
        assert root.origin == EXAMPLES_ORIGIN
        assert not root.writable

    def test_a_confinement_that_is_not_pinned_stays_the_workspace(
        self, roots, monkeypatch
    ):
        workspace, examples = roots
        monkeypatch.setenv("DW_WORKFLOW_PATH", str(examples))
        _candidate, root = resolve_sub_workflow(
            "Shared", str(workspace), str(workspace)
        )
        assert root.origin == WORKSPACE_ORIGIN
        assert root.writable

    def test_a_checkouts_own_workflows_dir_stays_the_workspace(
        self, tmp_path, monkeypatch
    ):
        # --examples-dir pointing at the workspace's own workflows/: the API
        # collapses it into one writable workspace root, so the worker must
        repo = tmp_path / "repo"
        (repo / "workflows").mkdir(parents=True)
        (repo / "workflows" / "Shared.json").write_text("{}")
        monkeypatch.setenv("DW_WORKSPACE", str(repo))
        monkeypatch.setenv("DW_WORKSPACE_SOURCE", "flag")
        monkeypatch.setenv("DW_WORKFLOW_PATH", str(repo / "workflows"))
        _candidate, root = resolve_sub_workflow(
            "Shared", str(repo / "workflows"), str(repo / "workflows")
        )
        assert root.origin == WORKSPACE_ORIGIN
        assert root.writable


class TestPromptsAndAssetsConstruction:
    @pytest.fixture
    def workspace(self, tmp_path):
        from dw.workspace import Workspace

        root = tmp_path / "ws"
        for sub in ("assets", "prompts", "common/assets"):
            (root / sub).mkdir(parents=True)
        return Workspace(root, "flag")

    @pytest.fixture
    def examples(self, tmp_path):
        tree = tmp_path / "repo"
        for sub in ("workflows", "assets", "prompts"):
            (tree / sub).mkdir(parents=True)
        return str(tree / "workflows")

    def test_assets_are_workspace_then_common_then_examples(
        self, workspace, examples, tmp_path
    ):
        path = library_path("assets", workspace, [examples])
        assert [(r.root, r.origin) for r in path.roots()] == [
            (workspace.assets, WORKSPACE_ORIGIN),
            (workspace.common_assets, "common"),
            (str(tmp_path / "repo" / "assets"), EXAMPLES_ORIGIN),
        ]

    def test_the_common_root_is_writable_only_for_a_shared_write(
        self, workspace, examples
    ):
        path = library_path("assets", workspace, [examples])
        common = path.roots()[1]
        assert common.writable
        assert path.writable_root().root == workspace.assets
        assert path.writable_root(shared=True) is common

    def test_a_primary_names_the_front_and_the_workspaces_own_is_not_added(
        self, workspace, tmp_path
    ):
        job_assets = tmp_path / "job-assets"
        path = library_path("assets", workspace, [], primary=str(job_assets))
        assert path.roots()[0].root == str(job_assets)
        assert workspace.assets not in [r.root for r in path.roots()]
        assert path.roots()[0].writable

    def test_prompts_are_the_front_then_the_examples(
        self, workspace, examples, tmp_path
    ):
        path = library_path("prompts", workspace, [examples])
        assert [(r.root, r.origin) for r in path.roots()] == [
            (workspace.prompts, WORKSPACE_ORIGIN),
            (str(tmp_path / "repo" / "prompts"), EXAMPLES_ORIGIN),
        ]

    @pytest.mark.parametrize("kind", ["prompts", "assets", "workflows"])
    def test_a_missing_examples_directory_is_dropped_for_every_kind(
        self, kind, workspace, tmp_path
    ):
        gone = tmp_path / "gone" / "workflows"
        path = library_path(kind, workspace, [str(gone)])
        assert len(path.roots()) == (2 if kind == "assets" else 1)
        assert all(os.path.isdir(r.root) for r in path.roots()[1:])

    def test_a_missing_common_root_is_dropped(self, workspace, examples):
        os.rmdir(workspace.common_assets)
        path = library_path("assets", workspace, [examples])
        assert "common" not in [r.origin for r in path.roots()]

    def test_the_front_is_kept_before_it_exists(self, workspace, tmp_path):
        path = library_path("prompts", workspace, [], primary=str(tmp_path / "not-yet"))
        assert [r.root for r in path.roots()] == [str(tmp_path / "not-yet")]


class TestEnvironmentSerializer:
    @pytest.fixture
    def trees(self, tmp_path, monkeypatch):
        from dw.workspace import Workspace

        monkeypatch.delenv("DW_ASSET_PATH", raising=False)
        monkeypatch.delenv("DW_PROMPT_PATH", raising=False)
        monkeypatch.setenv("DW_WORKSPACE", str(tmp_path / "ws"))
        monkeypatch.setenv("DW_WORKSPACE_SOURCE", "environment")
        workspace = Workspace(tmp_path / "ws", "environment")
        for sub in ("assets", "prompts", "common/assets"):
            (tmp_path / "ws" / sub).mkdir(parents=True)
        for sub in ("workflows", "assets", "prompts"):
            (tmp_path / "repo" / sub).mkdir(parents=True)
        return workspace, str(tmp_path / "repo" / "workflows")

    def test_the_worker_rebuilds_the_path_the_api_builds(self, trees):
        from dw.library import library_path_from_env, pin_library_path

        workspace, examples = trees
        for kind in ("assets", "prompts", "workflows"):
            api = library_path(kind, workspace, [examples])
            pin_library_path(kind, workspace, [examples])
            worker = library_path_from_env(kind, api.roots()[0].root)
            assert [(r.root, r.origin) for r in worker.roots()] == [
                (r.root, r.origin) for r in api.roots()
            ], kind

    def test_the_env_format_is_the_pathsep_tail(self, trees):
        from dw.library import ASSET_PATH_ENV_VAR, pin_library_path

        workspace, examples = trees
        written = pin_library_path("assets", workspace, [examples])
        assert written == os.environ[ASSET_PATH_ENV_VAR]
        assert written.split(os.pathsep) == [
            workspace.common_assets,
            os.path.join(os.path.dirname(examples), "assets"),
        ]

    def test_an_empty_tail_clears_the_variable(self, trees, monkeypatch):
        from dw.library import PROMPT_PATH_ENV_VAR, pin_library_path

        workspace, _examples = trees
        monkeypatch.setenv(PROMPT_PATH_ENV_VAR, "/stale")
        pin_library_path("prompts", workspace, [])
        assert PROMPT_PATH_ENV_VAR not in os.environ

    def test_fallbacks_are_deduplicated_against_each_other(self, trees, monkeypatch):
        from dw.library import library_path_from_env

        workspace, examples = trees
        shared = os.path.join(os.path.dirname(examples), "assets")
        monkeypatch.setenv(
            "DW_ASSET_PATH", os.pathsep.join([shared, shared, workspace.assets])
        )
        roots = library_path_from_env("assets", workspace.assets).roots()
        assert [r.root for r in roots] == [workspace.assets, shared]

    def test_common_is_recognized_by_the_workspace_it_lives_in(self, trees):
        from dw.library import library_path_from_env, pin_library_path

        workspace, examples = trees
        pin_library_path("assets", workspace, [examples])
        origins = [
            r.origin for r in library_path_from_env("assets", workspace.assets).roots()
        ]
        assert origins == [WORKSPACE_ORIGIN, "common", EXAMPLES_ORIGIN]


class TestResolveAgainstOneRoot:
    """D7: `asset_dir=root` used to resolve against that root *and* the
    pinned environment tail, so "does this root hold the name" answered for
    several roots."""

    @pytest.fixture
    def libraries(self, tmp_path, monkeypatch):
        pinned = tmp_path / "pinned"
        other = tmp_path / "other"
        for directory in (pinned, other):
            directory.mkdir()
        (pinned / "x.png").write_bytes(b"x")
        (pinned / "p.json").write_text('{"text": "pinned"}')
        monkeypatch.setenv("DW_ASSET_PATH", str(pinned))
        monkeypatch.setenv("DW_PROMPT_PATH", str(pinned))
        return pinned, other

    def test_the_pinned_tail_is_still_searched_by_default(self, libraries):
        from dw.assets import resolve_asset_reference

        pinned, other = libraries
        found = resolve_asset_reference("asset:x.png", asset_dir=str(other))
        assert found == os.path.realpath(pinned / "x.png")

    def test_a_single_root_library_does_not_find_the_tail(self, libraries):
        from dw.assets import resolve_asset_reference
        from dw.library import ASSETS_KIND, LibraryPath, LibraryRoot

        _pinned, other = libraries
        only = LibraryPath(
            ASSETS_KIND, [LibraryRoot(str(other), WORKSPACE_ORIGIN, True)]
        )
        with pytest.raises(ValueError):
            resolve_asset_reference("asset:x.png", library=only)

    def test_a_single_root_prompt_library_does_not_find_the_tail(self, libraries):
        from dw.library import LibraryPath, LibraryRoot, PROMPTS_KIND
        from dw.prompts import resolve_prompt_reference

        _pinned, other = libraries
        only = LibraryPath(
            PROMPTS_KIND, [LibraryRoot(str(other), WORKSPACE_ORIGIN, True)]
        )
        with pytest.raises(ValueError):
            resolve_prompt_reference("prompt:p", library=only)

    def test_a_single_root_library_finds_what_the_root_holds(self, libraries):
        from dw.assets import resolve_asset_reference
        from dw.library import ASSETS_KIND, LibraryPath, LibraryRoot

        _pinned, other = libraries
        (other / "y.png").write_bytes(b"y")
        only = LibraryPath(
            ASSETS_KIND, [LibraryRoot(str(other), WORKSPACE_ORIGIN, True)]
        )
        assert resolve_asset_reference("asset:y.png", library=only) == os.path.realpath(
            other / "y.png"
        )


class TestPinnedTailOutlivesTheServersPrimary:
    """The checkout setup: --examples-dir is the default workspace's own
    workflows/, so its sibling assets/ and the workflows/ are also the
    server's primaries. A job in a named workspace has a primary of its own
    and must still reach them, which the pinned tail has to carry."""

    @pytest.fixture
    def checkout(self, tmp_path, monkeypatch):
        from dw.serve import build_parser, configure_environment

        root = tmp_path / "checkout"
        for sub in ("workflows", "assets", "prompts"):
            (root / sub).mkdir(parents=True)
        (root / "assets" / "iris.png").write_bytes(b"i")
        (root / "workflows" / "Shared.json").write_text('{"id": "shared"}')
        for name in (
            "DW_WORKSPACE",
            "DW_WORKSPACE_SOURCE",
            "DW_ASSET_DIR",
            "DW_PROMPT_DIR",
            "DW_ASSET_PATH",
            "DW_PROMPT_PATH",
            "DW_WORKFLOW_PATH",
        ):
            monkeypatch.setenv(name, "")
            monkeypatch.delenv(name)
        args = build_parser().parse_args(
            ["--workspace", str(root), "--examples-dir", str(root / "workflows")]
        )
        configure_environment(args)
        named = tmp_path / "checkout" / "ns"
        (named / "assets").mkdir(parents=True)
        (named / "workflows").mkdir(parents=True)
        return root, named

    def test_a_named_workspace_job_still_finds_the_default_assets(self, checkout):
        from dw.assets import resolve_asset_reference

        root, named = checkout
        found = resolve_asset_reference(
            "asset:iris.png", asset_dir=str(named / "assets")
        )
        assert found == os.path.realpath(root / "assets" / "iris.png")

    def test_a_named_workspace_job_still_finds_the_default_workflows(self, checkout):
        root, named = checkout
        candidate, library_root = resolve_sub_workflow(
            "Shared", str(named / "workflows"), str(named / "workflows")
        )
        assert candidate == str(root / "workflows" / "Shared.json")

    def test_a_directory_created_after_startup_is_seen(self, tmp_path, monkeypatch):
        from dw.library import library_path_from_env

        later = tmp_path / "later"
        monkeypatch.setenv("DW_ASSET_PATH", str(later))
        before = library_path_from_env("assets", str(tmp_path / "front"))
        later.mkdir()
        after = library_path_from_env("assets", str(tmp_path / "front"))
        assert len(before.roots()) == 1
        assert [r.root for r in after.roots()][-1] == str(later)


class TestPromptRefusalIsNotAMissEither:
    def test_a_dangling_link_is_skipped_not_refused(self, tmp_path, monkeypatch):
        from dw.library import LibraryPath, LibraryRoot, PROMPTS_KIND
        from dw.prompts import resolve_prompt_reference

        library = tmp_path / "prompts"
        library.mkdir()
        os.symlink(tmp_path / "nowhere.json", library / "ghost.json")
        only = LibraryPath(
            PROMPTS_KIND, [LibraryRoot(str(library), WORKSPACE_ORIGIN, True)]
        )
        with pytest.raises(ValueError):
            resolve_prompt_reference("prompt:ghost", library=only)


class TestExistingAndFrontless:
    def test_existing_drops_the_roots_that_are_not_directories(self, tmp_path):
        from dw.workspace import ConfiguredWorkspace

        shared = tmp_path / "root" / "common" / "assets"
        shared.mkdir(parents=True)
        workspace = ConfiguredWorkspace(
            workflows=None,
            assets=str(tmp_path / "not-yet"),
            outputs=str(tmp_path / "outputs"),
            prompts=None,
            root=str(tmp_path / "root"),
        )
        path = library_path("assets", workspace)
        assert [r.root for r in path.roots()] == [
            str(tmp_path / "not-yet"),
            str(shared),
        ]
        assert [r.root for r in path.existing().roots()] == [str(shared)]

    def test_a_workspace_with_no_asset_library_has_no_front(self, tmp_path):
        from dw.workspace import ConfiguredWorkspace

        workspace = ConfiguredWorkspace(
            workflows=None,
            assets=None,
            outputs=str(tmp_path / "outputs"),
            prompts=None,
            root=None,
        )
        assert library_path("assets", workspace).roots() == []
