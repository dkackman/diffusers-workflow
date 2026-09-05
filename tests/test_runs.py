"""Run directories: one execution writes one directory, named by the
workflow's identity, and leaves a manifest describing itself."""

import json
import os
from unittest.mock import patch

import pytest

from dw.runs import (
    FLAT_LAYOUT,
    OUTPUT_LAYOUT_ENV_VAR,
    RUN_LAYOUT,
    is_run_id,
    manifest_relative_files,
    new_run_id,
    output_layout,
    strip_run_id,
    workflow_identity,
)


class TestIdentity:
    @pytest.mark.parametrize(
        "file_spec,expected",
        [
            ("/x/workflows/ltx2/Gyre.json", "ltx2/Gyre"),
            ("/x/workflows/Gyre.json", "Gyre"),
            ("/x/anywhere/Gyre.json", "Gyre"),
            ("/x/workflows/a/b/Gyre.json", "a/b/Gyre"),
        ],
    )
    def test_a_workflow_is_named_by_its_file(self, file_spec, expected):
        assert workflow_identity(file_spec, "id").replace(os.sep, "/") == expected

    def test_an_inline_definition_is_named_by_its_id(self):
        # The synthetic file name carries a directory, not an identity
        assert workflow_identity("/x/workflows/__inline__.json", "my-flow") == "my-flow"

    def test_a_hostile_id_cannot_escape_the_output_directory(self):
        # The identity is joined onto the output directory, so nothing in it
        # may traverse: separators and dot segments do not survive
        identity = workflow_identity(None, "../../etc/passwd")
        assert "/" not in identity and "\\" not in identity and ".." not in identity

    def test_something_is_always_named(self):
        assert workflow_identity(None, None) == "workflow"


class TestRunIds:
    def test_a_run_id_is_a_timestamp_and_a_digest(self):
        run_id = new_run_id({"workflow": "spec"})
        assert is_run_id(run_id)

    def test_the_digest_is_of_the_spec(self):
        # Two runs of the same spec share a digest, which is what makes a
        # rerun of an edited workflow visibly different in the directory list
        from datetime import datetime

        when = datetime(2026, 9, 5, 12, 0, 0)
        assert new_run_id({"a": 1}, now=when) == new_run_id({"a": 1}, now=when)
        assert new_run_id({"a": 1}, now=when) != new_run_id({"a": 2}, now=when)

    def test_an_unserializable_spec_still_yields_an_id(self):
        assert is_run_id(new_run_id({"generator": object()}))

    def test_stripping_a_run_id_gives_the_workflow_folder(self):
        run_id = new_run_id({})
        assert strip_run_id(f"ltx2/Gyre/{run_id}/still.png") == "ltx2/Gyre"
        assert strip_run_id(f"{run_id}/still.png") == ""

    def test_a_path_with_no_run_id_keeps_its_folder(self):
        assert strip_run_id("ltx2/still.png") == "ltx2"
        assert strip_run_id("still.png") == ""


class TestLayoutResolution:
    def test_run_is_the_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv(OUTPUT_LAYOUT_ENV_VAR, raising=False)
        monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", str(tmp_path))
        assert output_layout() == RUN_LAYOUT

    def test_the_environment_selects_flat(self, monkeypatch):
        monkeypatch.setenv(OUTPUT_LAYOUT_ENV_VAR, FLAT_LAYOUT)
        assert output_layout() == FLAT_LAYOUT

    def test_the_setting_selects_flat(self, monkeypatch, tmp_path):
        monkeypatch.delenv(OUTPUT_LAYOUT_ENV_VAR, raising=False)
        monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", str(tmp_path))
        (tmp_path / "settings.json").write_text(json.dumps({"output_layout": "flat"}))
        assert output_layout() == FLAT_LAYOUT

    def test_nonsense_falls_back_to_run(self, monkeypatch, tmp_path):
        monkeypatch.setenv(OUTPUT_LAYOUT_ENV_VAR, "sideways")
        monkeypatch.setenv("DIFFUSERS_HELPER_ROOT", str(tmp_path))
        assert output_layout() == RUN_LAYOUT


class TestManifestPaths:
    def test_files_inside_the_run_are_recorded_relative(self, tmp_path):
        run_dir = str(tmp_path / "run")
        files = [os.path.join(run_dir, "a.png"), os.path.join(run_dir, "sub", "b.png")]
        assert manifest_relative_files(files, run_dir) == ["a.png", "sub/b.png"]

    def test_a_file_from_an_earlier_run_stays_absolute(self, tmp_path):
        # What a step cache hit republishes: the file is real, but it is not
        # this run's to describe relatively
        earlier = str(tmp_path / "earlier" / "a.png")
        assert manifest_relative_files([earlier], str(tmp_path / "run")) == [earlier]


def _workflow_definition():
    return {
        "id": "runs_test",
        "seed": 7,
        "steps": [
            {
                "name": "gen0",
                "result": {"content_type": "image/png"},
                "pipeline": {
                    "configuration": {
                        "component_type": "{FakePipeline}",
                        "no_generator": True,
                    },
                    "from_pretrained_arguments": {"model_name": "model-0"},
                    "arguments": {"prompt": "p", "num_inference_steps": 1},
                },
            }
        ],
    }


@pytest.fixture
def fake_pipeline():
    """A workflow run whose pipeline yields one small image."""
    from PIL import Image

    from dw.pipeline_processors.pipeline import Pipeline

    class FakePipeline:
        def __call__(self, *args, **kwargs):
            class Output:
                images = [Image.new("RGB", (8, 8), "green")]

            return Output()

        def to(self, *args, **kwargs):
            return self

        @property
        def components(self):
            return {}

    def mock_load(self, shared_components):
        self.pipeline = FakePipeline()

    with patch.object(Pipeline, "load", mock_load):
        with patch("dw.workflow.empty_device_cache"):
            yield


class TestRunDirectories:
    def test_each_run_writes_its_own_directory(self, tmp_path, fake_pipeline):
        from dw.workflow import Workflow

        first = Workflow(
            _workflow_definition(), str(tmp_path), "/w/workflows/Gyre.json"
        )
        first.run({})
        second = Workflow(
            _workflow_definition(), str(tmp_path), "/w/workflows/Gyre.json"
        )
        second.run({})

        runs = sorted((tmp_path / "Gyre").iterdir())
        # Even started in the same second with the same spec, which is what a
        # quick rerun is: the second run never writes into the first's
        # directory
        assert len(runs) == 2
        assert all(is_run_id(run.name) for run in runs)

        first_manifest = json.loads((runs[0] / "manifest.json").read_text())
        second_manifest = json.loads((runs[1] / "manifest.json").read_text())
        # The first run wrote its image; the second was an unchanged rerun,
        # so the step cache served it - it writes nothing new and reports the
        # earlier run's file, by the absolute path that is not its own to
        # describe relatively
        assert first_manifest["steps"][0]["files"] == ["runs_test-gen0.0-0.0.png"]
        assert not first_manifest["steps"][0].get("reused")
        assert second_manifest["steps"][0]["reused"] is True
        reused = second_manifest["steps"][0]["files"][0]
        assert os.path.isabs(reused) and os.path.exists(reused)
        assert os.path.dirname(reused) == str(runs[0])

    def test_a_changed_run_writes_its_own_files(self, tmp_path, fake_pipeline):
        from dw.workflow import Workflow

        first = Workflow(
            _workflow_definition(), str(tmp_path), "/w/workflows/Gyre.json"
        )
        first.run({})
        changed = _workflow_definition()
        changed["steps"][0]["pipeline"]["arguments"]["prompt"] = "different"
        Workflow(changed, str(tmp_path), "/w/workflows/Gyre.json").run({})

        runs = sorted((tmp_path / "Gyre").iterdir())
        assert len(runs) == 2
        for run in runs:
            assert any(name.suffix == ".png" for name in run.iterdir())

    def test_the_manifest_describes_the_run(self, tmp_path, fake_pipeline):
        from dw.workflow import Workflow

        workflow = Workflow(
            _workflow_definition(), str(tmp_path), "/w/workflows/ltx2/Gyre.json"
        )
        workflow.run({"prompt": "a cat"})

        run_dir = next((tmp_path / "ltx2" / "Gyre").iterdir())
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["status"] == "completed"
        assert manifest["workflow"]["identity"] == "ltx2/Gyre"
        assert manifest["workflow"]["id"] == "runs_test"
        assert manifest["seed"] == 7
        assert manifest["arguments"] == {"prompt": "a cat"}
        assert is_run_id(manifest["run_id"])
        assert manifest["steps"][0]["step"] == "gen0"
        # relative to the directory that describes itself
        for name in manifest["steps"][0]["files"]:
            assert not os.path.isabs(name)
            assert (run_dir / name).exists()

    def test_a_failed_run_still_records_what_it_wrote(self, tmp_path, fake_pipeline):
        from dw.workflow import Workflow

        definition = _workflow_definition()
        definition["steps"].append({"name": "boom", "task": {"command": "no_such"}})
        workflow = Workflow(definition, str(tmp_path), "/w/workflows/Gyre.json")
        with pytest.raises(Exception):
            workflow.run({})

        run_dir = next((tmp_path / "Gyre").iterdir())
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["status"] == "failed"
        assert manifest["steps"][0]["step"] == "gen0"

    def test_the_flat_layout_writes_where_it_always_did(
        self, tmp_path, fake_pipeline, monkeypatch
    ):
        from dw.workflow import Workflow

        monkeypatch.setenv(OUTPUT_LAYOUT_ENV_VAR, FLAT_LAYOUT)
        workflow = Workflow(
            _workflow_definition(), str(tmp_path), "/w/workflows/ltx2/Gyre.json"
        )
        workflow.run({})
        # the pre-run-directory layout: the workflow's position under a
        # 'workflows' tree, and no run directory or manifest
        written = list((tmp_path / "ltx2").iterdir())
        assert [path.suffix for path in written] == [".png"]
