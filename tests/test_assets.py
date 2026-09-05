"""The asset library: 'asset:' references resolve to files under the asset
directory, and never to anything outside it."""

import os

import pytest
from PIL import Image

from dw.arguments import realize_args
from dw.assets import (
    ASSET_DIR_ENV_VAR,
    get_asset_dir,
    is_asset_reference,
    resolve_asset_reference,
)
from dw.security import InvalidInputError, SecurityError
from dw.workspace import WORKSPACE_ENV_VAR, WORKSPACE_SOURCE_ENV_VAR


@pytest.fixture
def asset_dir(tmp_path, monkeypatch):
    """An asset library with a top-level and a nested image."""
    library = tmp_path / "assets"
    (library / "gyre" / "frames").mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(library / "iris.png")
    Image.new("RGB", (8, 8), "blue").save(library / "gyre" / "frames" / "web.png")
    monkeypatch.setenv(ASSET_DIR_ENV_VAR, str(library))
    return library


@pytest.fixture
def no_library(monkeypatch, tmp_path):
    monkeypatch.delenv(ASSET_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv(WORKSPACE_ENV_VAR, raising=False)
    monkeypatch.delenv(WORKSPACE_SOURCE_ENV_VAR, raising=False)
    monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", str(tmp_path / "helper"))
    (tmp_path / "helper").mkdir()
    return tmp_path


class TestAssetDir:
    def test_the_environment_names_the_library(self, asset_dir):
        assert get_asset_dir() == str(asset_dir)

    def test_a_named_workspace_names_the_library(self, no_library, monkeypatch):
        monkeypatch.setenv(WORKSPACE_ENV_VAR, str(no_library / "studio"))
        assert get_asset_dir() == str(no_library / "studio" / "assets")

    def test_the_working_directory_library_is_used(self, no_library, monkeypatch):
        (no_library / "here" / "assets").mkdir(parents=True)
        monkeypatch.chdir(no_library / "here")
        assert get_asset_dir() == str(no_library / "here" / "assets")

    def test_the_walk_from_the_workflow_dir_finds_the_tree_s_library(
        self, no_library, monkeypatch
    ):
        (no_library / "repo" / "assets").mkdir(parents=True)
        workflow_dir = no_library / "repo" / "workflows" / "gyre"
        workflow_dir.mkdir(parents=True)
        elsewhere = no_library / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        assert get_asset_dir(str(workflow_dir)) == str(no_library / "repo" / "assets")


class TestReferences:
    def test_a_reference_resolves_to_the_file(self, asset_dir):
        assert resolve_asset_reference("asset:iris.png") == str(asset_dir / "iris.png")

    def test_a_nested_reference_resolves(self, asset_dir):
        assert resolve_asset_reference("asset:gyre/frames/web.png") == str(
            asset_dir / "gyre" / "frames" / "web.png"
        )

    def test_a_missing_asset_says_so(self, asset_dir):
        with pytest.raises(ValueError, match="not found"):
            resolve_asset_reference("asset:nothing.png")

    @pytest.mark.parametrize(
        "reference",
        [
            "asset:../outside.png",
            "asset:/etc/passwd",
            "asset:gyre/../../outside.png",
            "asset:",
        ],
    )
    def test_a_reference_cannot_leave_the_library(self, asset_dir, reference):
        with pytest.raises(SecurityError):
            resolve_asset_reference(reference)

    def test_a_symlink_out_of_the_library_is_refused(self, asset_dir, tmp_path):
        outside = tmp_path / "outside.png"
        Image.new("RGB", (8, 8)).save(outside)
        os.symlink(outside, asset_dir / "link.png")
        with pytest.raises(SecurityError):
            resolve_asset_reference("asset:link.png")

    def test_the_prefix_is_recognized(self):
        assert is_asset_reference("asset:x.png")
        assert not is_asset_reference("prompt:x")
        assert not is_asset_reference(3)


class TestRealizedArguments:
    def test_an_image_argument_loads_the_asset(self, asset_dir):
        args = {"image": "asset:iris.png"}
        realize_args(args)
        assert args["image"].size == (8, 8)

    def test_a_plain_argument_becomes_the_path(self, asset_dir):
        args = {"reference_file": "asset:gyre/frames/web.png"}
        realize_args(args)
        assert args["reference_file"] == str(asset_dir / "gyre" / "frames" / "web.png")

    def test_a_list_of_assets_resolves(self, asset_dir):
        args = {"image": ["asset:iris.png", "asset:gyre/frames/web.png"]}
        realize_args(args)
        assert [image.size for image in args["image"]] == [(8, 8), (8, 8)]

    def test_an_object_built_from_an_asset_resolves(self, asset_dir):
        args = {"reference": {"from_file": "asset:iris.png"}}
        realize_args(args)
        assert args["reference"]["from_file"] == str(asset_dir / "iris.png")

    def test_a_reference_is_rooted_at_the_library_not_the_workflow(
        self, asset_dir, tmp_path
    ):
        # base_dir names an unrelated directory: an asset reference ignores it,
        # which is the whole point - the workflow does not have to live beside
        # the media it reads
        elsewhere = tmp_path / "workflows" / "gyre"
        elsewhere.mkdir(parents=True)
        args = {"image": "asset:iris.png"}
        realize_args(args, base_dir=str(elsewhere))
        assert args["image"].size == (8, 8)

    def test_a_bad_name_is_refused_during_realization(self, asset_dir):
        with pytest.raises(InvalidInputError):
            realize_args({"image": "asset:../escape.png"})


class TestStoredPromptText:
    def test_a_prompt_may_not_masquerade_as_an_asset_reference(
        self, tmp_path, monkeypatch
    ):
        import json

        from dw.prompts import fetch_prompt

        library = tmp_path / "prompts"
        library.mkdir()
        (library / "sneaky.json").write_text(json.dumps({"text": "asset:iris.png"}))
        monkeypatch.setenv("DW_PROMPT_DIR", str(library))
        with pytest.raises(ValueError, match="reference prefix"):
            fetch_prompt("prompt:sneaky")
