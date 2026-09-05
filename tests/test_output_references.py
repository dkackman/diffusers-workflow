"""'output:' references: a workflow names what an earlier run made, so a
multi-stage pipeline stops needing files copied back by hand."""

import os

import pytest
from PIL import Image

from dw.arguments import realize_args
from dw.runs import (
    activate_output_root,
    deactivate_output_root,
    is_output_reference,
    output_root,
    resolve_output_reference,
)
from dw.security import InvalidInputError, SecurityError
from dw.workspace import WORKSPACE_ENV_VAR, WORKSPACE_SOURCE_ENV_VAR


@pytest.fixture
def outputs(tmp_path):
    """An output directory holding two runs of one workflow, and one run of
    another, with the active output root pointed at it."""
    root = tmp_path / "outputs"
    older = root / "ltx2" / "Gyre" / "20260905-101500-aaaaaaaa"
    newer = root / "ltx2" / "Gyre" / "20260905-111500-bbbbbbbb"
    other = root / "Krea" / "20260905-090000-cccccccc"
    for directory in (older, newer, other):
        directory.mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(older / "still.png")
    Image.new("RGB", (16, 16), "blue").save(newer / "still.png")
    (newer / "manifest.json").write_text("{}")
    Image.new("RGB", (4, 4), "green").save(other / "frame.png")

    token = activate_output_root(str(root))
    yield root
    deactivate_output_root(token)


class TestReferences:
    def test_a_run_and_file_resolve(self, outputs):
        assert resolve_output_reference(
            "output:ltx2/Gyre/20260905-101500-aaaaaaaa/still.png"
        ) == str(outputs / "ltx2" / "Gyre" / "20260905-101500-aaaaaaaa" / "still.png")

    def test_latest_names_the_newest_run(self, outputs):
        # Run ids start with a UTC timestamp, so newest is last in sort order
        assert resolve_output_reference("output:ltx2/Gyre/latest/still.png") == str(
            outputs / "ltx2" / "Gyre" / "20260905-111500-bbbbbbbb" / "still.png"
        )

    def test_latest_is_per_workflow(self, outputs):
        assert resolve_output_reference("output:Krea/latest/frame.png") == str(
            outputs / "Krea" / "20260905-090000-cccccccc" / "frame.png"
        )

    def test_a_workflow_with_no_runs_says_so(self, outputs):
        (outputs / "Never").mkdir()
        with pytest.raises(ValueError, match="No runs yet"):
            resolve_output_reference("output:Never/latest/still.png")

    def test_a_missing_file_says_so(self, outputs):
        with pytest.raises(ValueError, match="not found"):
            resolve_output_reference("output:ltx2/Gyre/latest/nothing.png")

    def test_a_directory_is_not_a_file(self, outputs):
        with pytest.raises(ValueError):
            resolve_output_reference("output:ltx2/Gyre/latest")

    @pytest.mark.parametrize(
        "reference",
        [
            "output:../escape.png",
            "output:/etc/passwd",
            "output:ltx2/../../escape.png",
            "output:single",
            "output:",
        ],
    )
    def test_a_reference_cannot_leave_the_output_directory(self, outputs, reference):
        with pytest.raises(SecurityError):
            resolve_output_reference(reference)

    def test_a_symlink_out_of_the_output_directory_is_refused(self, outputs, tmp_path):
        outside = tmp_path / "outside.png"
        Image.new("RGB", (8, 8)).save(outside)
        os.symlink(outside, outputs / "ltx2" / "Gyre" / "link.png")
        with pytest.raises(SecurityError):
            resolve_output_reference("output:ltx2/Gyre/link.png")

    def test_the_prefix_is_recognized(self):
        assert is_output_reference("output:a/b/c.png")
        assert not is_output_reference("asset:a.png")
        assert not is_output_reference(None)


class TestTheActiveRoot:
    def test_a_run_names_the_root_it_writes_to(self, outputs):
        assert output_root() == str(outputs)

    def test_outside_a_run_it_is_the_workspace(self, tmp_path, monkeypatch):
        monkeypatch.setenv(WORKSPACE_ENV_VAR, str(tmp_path / "studio"))
        monkeypatch.setenv(WORKSPACE_SOURCE_ENV_VAR, "flag")
        assert output_root() == str(tmp_path / "studio" / "outputs")


class TestRealizedArguments:
    def test_an_image_argument_loads_the_earlier_output(self, outputs):
        args = {"image": "output:ltx2/Gyre/latest/still.png"}
        realize_args(args)
        assert args["image"].size == (16, 16)

    def test_a_plain_argument_becomes_the_path(self, outputs):
        args = {"soundtrack": "output:Krea/latest/frame.png"}
        realize_args(args)
        assert args["soundtrack"].endswith("frame.png")
        assert os.path.isabs(args["soundtrack"])

    def test_an_object_built_from_an_output_resolves(self, outputs):
        args = {"reference": {"from_file": "output:ltx2/Gyre/latest/still.png"}}
        realize_args(args)
        assert args["reference"]["from_file"].endswith("still.png")

    def test_a_list_mixes_references_and_paths(self, outputs, tmp_path):
        plain = tmp_path / "plain.png"
        Image.new("RGB", (2, 2)).save(plain)
        args = {"image": ["output:ltx2/Gyre/latest/still.png", str(plain)]}
        realize_args(args)
        assert [image.size for image in args["image"]] == [(16, 16), (2, 2)]

    def test_a_bad_reference_is_refused_during_realization(self, outputs):
        with pytest.raises(InvalidInputError):
            realize_args({"image": "output:../escape.png"})


def test_a_prompt_may_not_masquerade_as_an_output_reference(tmp_path, monkeypatch):
    import json

    from dw.prompts import fetch_prompt

    library = tmp_path / "prompts"
    library.mkdir()
    (library / "sneaky.json").write_text(
        json.dumps({"text": "output:ltx2/Gyre/latest/still.png"})
    )
    monkeypatch.setenv("DW_PROMPT_DIR", str(library))
    with pytest.raises(ValueError, match="reference prefix"):
        fetch_prompt("prompt:sneaky")
