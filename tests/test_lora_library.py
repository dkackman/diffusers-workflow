"""The `loras` library kind: a JSON library like prompts, whose writable front
the server names, behind which the shipped loras/ sits read-only."""

import json
import os

import pytest

from dw.library import (
    EXAMPLES_ORIGIN,
    LORAS_KIND,
    WORKSPACE_ORIGIN,
    ReadOnlyLibraryError,
    library_path,
)
from dw.security import (
    InvalidInputError,
    SecurityError,
    validate_lora_name,
    validate_lora_path,
)
from dw.workspace import (
    LORAS_SUBDIR,
    ConfiguredWorkspace,
    Workspace,
    create_workspace,
    example_libraries,
)


def write_entry(root, name, body=None):
    path = os.path.join(root, f"{name}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(body or {"model_name": "a/b"}, file)
    return path


@pytest.fixture
def libraries(tmp_path):
    checkout = tmp_path / "repo"
    (checkout / "workflows").mkdir(parents=True)
    shipped = checkout / LORAS_SUBDIR
    own = tmp_path / "studio" / LORAS_SUBDIR
    write_entry(str(shipped), "qwen-image/voxel", {"model_name": "shipped/voxel"})
    write_entry(str(shipped), "flux/realism", {"model_name": "shipped/realism"})
    write_entry(str(own), "qwen-image/voxel", {"model_name": "own/voxel"})
    workspace = Workspace(tmp_path / "studio", "flag")
    path = library_path(
        LORAS_KIND, workspace, [str(checkout / "workflows")], primary=str(own)
    )
    return path, str(own), str(shipped)


class TestTheKind:
    def test_the_examples_tree_brings_its_loras(self, tmp_path):
        checkout = tmp_path / "repo"
        (checkout / "workflows").mkdir(parents=True)
        (checkout / LORAS_SUBDIR).mkdir()
        found = example_libraries([str(checkout / "workflows")])
        assert found[LORAS_SUBDIR] == [str(checkout / LORAS_SUBDIR)]

    def test_the_workspace_copy_shadows_the_shipped_one(self, libraries):
        path, own, shipped = libraries
        winners, hidden = path.entries()
        assert winners["qwen-image/voxel"].root == own
        assert winners["flux/realism"].root == shipped
        assert [(name, root.root) for name, root, _ in hidden] == [
            ("qwen-image/voxel", shipped)
        ]

    def test_find_resolves_front_to_back(self, libraries):
        path, own, _ = libraries
        found, root = path.find("qwen-image/voxel")
        assert root.origin == WORKSPACE_ORIGIN
        assert found == os.path.join(own, "qwen-image", "voxel.json")

    def test_the_shipped_root_is_read_only(self, libraries):
        path, _, _ = libraries
        _, root = path.find("flux/realism")
        assert root.origin == EXAMPLES_ORIGIN
        with pytest.raises(ReadOnlyLibraryError):
            path.require_writable(root, "flux/realism")

    def test_a_configured_workspace_with_no_front_has_no_writable_root(self, tmp_path):
        configured = ConfiguredWorkspace(
            workflows=tmp_path / "w", assets=None, outputs=tmp_path / "o", prompts=None
        )
        path = library_path(LORAS_KIND, configured, [], primary=None)
        assert path.writable_root() is None


class TestValidators:
    def test_a_name_one_folder_deep_is_accepted(self):
        assert validate_lora_name("qwen-image/voxel-style") == "qwen-image/voxel-style"

    @pytest.mark.parametrize("name", ["../x", "a/b/c", "/abs", "", ".hidden"])
    def test_a_name_that_could_leave_the_library_is_refused(self, name):
        with pytest.raises(InvalidInputError):
            validate_lora_name(name)

    def test_the_path_validator_refuses_traversal(self, tmp_path):
        with pytest.raises(SecurityError):
            validate_lora_path(str(tmp_path / ".." / "x.json"), str(tmp_path))

    def test_the_path_validator_wants_json(self, tmp_path):
        (tmp_path / "x.txt").write_text("{}")
        with pytest.raises(InvalidInputError):
            validate_lora_path(str(tmp_path / "x.txt"), str(tmp_path))


def test_loras_is_a_reserved_workspace_name(tmp_path):
    root = Workspace(tmp_path, "flag").ensure()
    with pytest.raises(InvalidInputError):
        create_workspace(root, LORAS_SUBDIR)
