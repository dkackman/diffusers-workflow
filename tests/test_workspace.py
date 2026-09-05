"""Workspace resolution: which directory a run's workflows, prompts and
outputs belong to, and the guarantee that a checkout answers the way it
always has."""

import json
import os

import pytest

from dw.workspace import (
    DEFAULT,
    ENVIRONMENT,
    FLAG,
    SETTINGS,
    WORKING_DIRECTORY,
    WORKSPACE_ENV_VAR,
    WORKSPACE_SOURCE_ENV_VAR,
    Workspace,
    looks_like_workspace,
    resolve_workspace,
    set_workspace,
)


@pytest.fixture
def clean_environment(monkeypatch, tmp_path):
    """No workspace named anywhere, and a settings file that names none."""
    monkeypatch.delenv(WORKSPACE_ENV_VAR, raising=False)
    monkeypatch.delenv(WORKSPACE_SOURCE_ENV_VAR, raising=False)
    monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", str(tmp_path / "helper"))
    (tmp_path / "helper").mkdir()
    return tmp_path


def workspace_tree(root, *subdirs):
    for name in subdirs:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


class TestResolution:
    def test_a_flag_wins_over_everything(self, clean_environment, monkeypatch):
        named = clean_environment / "named"
        monkeypatch.setenv(WORKSPACE_ENV_VAR, str(clean_environment / "environment"))
        workspace = resolve_workspace(str(named))
        assert workspace.root == str(named)
        assert workspace.source == FLAG

    def test_the_environment_names_it(self, clean_environment, monkeypatch):
        named = clean_environment / "environment"
        monkeypatch.setenv(WORKSPACE_ENV_VAR, str(named))
        workspace = resolve_workspace()
        assert workspace.root == str(named)
        assert workspace.source == ENVIRONMENT

    def test_the_settings_file_names_it(self, clean_environment, monkeypatch):
        named = clean_environment / "settings-workspace"
        settings = clean_environment / "helper" / "settings.json"
        settings.write_text(json.dumps({"workspace": str(named)}))
        monkeypatch.chdir(clean_environment)
        workspace = resolve_workspace()
        assert workspace.root == str(named)
        assert workspace.source == SETTINGS

    def test_a_working_directory_that_looks_like_one_is_used(
        self, clean_environment, monkeypatch
    ):
        # The checkout case: run from a tree that holds the folders, and the
        # tree is the workspace - which is what keeps every default where it
        # was before workspaces existed
        root = workspace_tree(clean_environment / "repo", "workflows", "prompts")
        monkeypatch.chdir(root)
        workspace = resolve_workspace()
        assert workspace.root == str(root)
        assert workspace.source == WORKING_DIRECTORY
        assert workspace.outputs == os.path.abspath("./outputs")
        assert workspace.workflows == os.path.abspath("./workflows")

    def test_a_bare_working_directory_falls_back_to_the_home_workspace(
        self, clean_environment, monkeypatch
    ):
        bare = clean_environment / "bare"
        bare.mkdir()
        monkeypatch.chdir(bare)
        workspace = resolve_workspace()
        assert workspace.source == DEFAULT
        assert workspace.root == os.path.expanduser("~/diffusers-workspace")

    def test_one_marker_folder_is_enough(self, clean_environment, monkeypatch):
        for marker in ("workflows", "prompts", "outputs"):
            root = workspace_tree(clean_environment / marker, marker)
            assert looks_like_workspace(str(root))
        # assets alone is not a marker - too common a folder to claim
        assets_only = workspace_tree(clean_environment / "assets-only", "assets")
        assert not looks_like_workspace(str(assets_only))


class TestSubdirectories:
    def test_the_four_folders_hang_off_the_root(self, tmp_path):
        workspace = Workspace(tmp_path / "ws", FLAG)
        assert workspace.workflows == str(tmp_path / "ws" / "workflows")
        assert workspace.prompts == str(tmp_path / "ws" / "prompts")
        assert workspace.assets == str(tmp_path / "ws" / "assets")
        assert workspace.outputs == str(tmp_path / "ws" / "outputs")

    def test_resolution_creates_nothing(self, clean_environment):
        # Asking where the workspace is must never leave a directory behind -
        # an entry point about to write calls ensure() itself
        workspace = resolve_workspace(str(clean_environment / "never-created"))
        assert not os.path.exists(workspace.root)

    def test_ensure_creates_the_four_folders(self, tmp_path):
        workspace = Workspace(tmp_path / "fresh", FLAG).ensure()
        for folder in ("workflows", "prompts", "assets", "outputs"):
            assert os.path.isdir(os.path.join(workspace.root, folder))

    def test_a_home_relative_root_is_expanded(self, tmp_path):
        assert Workspace("~/somewhere", FLAG).root == os.path.expanduser("~/somewhere")


class TestPinning:
    def test_pinning_carries_the_root_and_how_it_was_chosen(
        self, clean_environment, monkeypatch
    ):
        # A worker subprocess inherits the environment, and must not read an
        # inferred workspace back as one the user named - the prompt library
        # yields to older discovery for an inferred one only
        root = workspace_tree(clean_environment / "repo", "workflows")
        monkeypatch.chdir(root)
        set_workspace(resolve_workspace())
        assert os.environ[WORKSPACE_ENV_VAR] == str(root)

        monkeypatch.chdir(clean_environment)
        inherited = resolve_workspace()
        assert inherited.root == str(root)
        assert inherited.source == WORKING_DIRECTORY
        assert not inherited.is_explicit

    def test_a_pinned_flag_workspace_stays_explicit(
        self, clean_environment, monkeypatch
    ):
        set_workspace(str(clean_environment / "named"))
        assert resolve_workspace().is_explicit


class TestPromptLibraryPrecedence:
    """get_prompt_dir's older rules stay ahead of an inferred workspace and
    behind a named one."""

    def test_a_named_workspace_names_the_library(self, clean_environment, monkeypatch):
        from dw.prompts import get_prompt_dir

        monkeypatch.delenv("DW_PROMPT_DIR", raising=False)
        named = workspace_tree(clean_environment / "named", "prompts")
        cwd = workspace_tree(clean_environment / "cwd", "prompts")
        monkeypatch.chdir(cwd)
        monkeypatch.setenv(WORKSPACE_ENV_VAR, str(named))
        assert get_prompt_dir() == str(named / "prompts")

    def test_an_inferred_workspace_yields_to_the_walk(
        self, clean_environment, monkeypatch
    ):
        from dw.prompts import get_prompt_dir

        monkeypatch.delenv("DW_PROMPT_DIR", raising=False)
        tree = workspace_tree(clean_environment / "repo", "prompts", "workflows")
        elsewhere = clean_environment / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        assert get_prompt_dir(str(tree / "workflows")) == str(tree / "prompts")

    def test_the_prompt_environment_still_wins(self, clean_environment, monkeypatch):
        from dw.prompts import get_prompt_dir

        monkeypatch.setenv("DW_PROMPT_DIR", str(clean_environment / "library"))
        monkeypatch.setenv(WORKSPACE_ENV_VAR, str(clean_environment / "named"))
        assert get_prompt_dir() == str(clean_environment / "library")
