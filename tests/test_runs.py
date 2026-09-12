"""Run directories: one execution writes one directory, named by the
workflow's identity, and leaves a manifest describing itself."""

import json
import os
from datetime import datetime, timezone
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
    split_run_path,
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

    def test_splitting_a_run_path_names_its_three_parts(self):
        run_id = new_run_id({"id": "x"})
        assert split_run_path(f"ltx2/Gyre/{run_id}/still.png") == (
            "ltx2/Gyre",
            run_id,
            "",
        )
        assert split_run_path(f"ltx2/Gyre/{run_id}/final/still.png") == (
            "ltx2/Gyre",
            run_id,
            "final",
        )
        assert split_run_path(f"ltx2/Gyre/{run_id}/shots/act-1/x.mp4") == (
            "ltx2/Gyre",
            run_id,
            "shots/act-1",
        )
        # A counter suffix is still a run id
        assert split_run_path(f"Gyre/{run_id}-2/final/x.mp4") == (
            "Gyre",
            f"{run_id}-2",
            "final",
        )

    def test_a_path_with_no_run_id_splits_to_its_directory(self):
        # Flat layout: nothing to anchor on, so the directory is the identity
        assert split_run_path("ltx2/final/still.png") == ("ltx2/final", "", "")
        assert split_run_path("still.png") == ("", "", "")

    def test_stripping_a_run_id_ignores_what_follows_it(self):
        run_id = new_run_id({"id": "x"})
        assert strip_run_id(f"ltx2/Gyre/{run_id}/final/still.png") == "ltx2/Gyre"
        assert strip_run_id(f"{run_id}/final/still.png") == ""

    def test_a_run_id_with_tz_aware_utc_now_has_correct_stamp(self):
        # Verify that a tz-aware UTC now produces the correct timestamp prefix
        now = datetime(2026, 9, 5, 12, 34, 56, tzinfo=timezone.utc)
        run_id = new_run_id({"test": "spec"}, now=now)
        expected_stamp = now.strftime("%Y%m%d-%H%M%S")
        assert run_id.startswith(expected_stamp)
        assert is_run_id(run_id)

    def test_a_run_id_without_now_uses_current_utc_time(self):
        # Verify that without a 'now' parameter, the stamp matches the current UTC minute
        # Allow for some clock skew by checking current minute or next minute
        run_id = new_run_id({"test": "spec"})
        stamp_str = run_id[:15]  # "YYYYMMDD-HHMMSS"
        now_utc = datetime.now(timezone.utc)

        # Parse the stamp and verify it is within 5 seconds of now
        parsed_stamp = datetime.strptime(stamp_str, "%Y%m%d-%H%M%S")
        time_diff = (
            now_utc - parsed_stamp.replace(tzinfo=timezone.utc)
        ).total_seconds()
        assert -5 <= time_diff <= 5, f"Stamp is off by {time_diff} seconds"


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


def _foldered_definition(subfolder="final"):
    definition = _workflow_definition()
    definition["steps"][0]["result"]["subfolder"] = subfolder
    return definition


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

    def test_the_manifest_records_the_seed_a_seedless_run_drew(
        self, tmp_path, fake_pipeline
    ):
        """A workflow naming no seed gets a random one, and the manifest is
        the only place it is ever written down - reporting null there loses
        the one value needed to reproduce the files sitting beside it."""
        from dw.workflow import Workflow

        definition = _workflow_definition()
        del definition["seed"]
        workflow = Workflow(definition, str(tmp_path), "/w/workflows/ltx2/Gyre.json")
        workflow.run({"prompt": "a cat"})

        run_dir = next((tmp_path / "ltx2" / "Gyre").iterdir())
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert isinstance(manifest["seed"], int)
        # and the definition it was built from is untouched, so the next run
        # draws its own seed rather than inheriting this one
        assert "seed" not in definition

    def test_a_seed_can_come_from_a_variable(self, tmp_path, fake_pipeline):
        """'seed' accepts a 'variable:' reference, so a caller can re-run a
        workflow at the seed a previous run's manifest reported."""
        from dw.workflow import Workflow

        definition = _workflow_definition()
        definition["variables"] = {"seed": 7}
        definition["seed"] = "variable:seed"
        workflow = Workflow(definition, str(tmp_path), "/w/workflows/ltx2/Gyre.json")
        workflow.validate()
        workflow.run({"seed": "1234"})  # as it arrives from the command line

        run_dir = next((tmp_path / "ltx2" / "Gyre").iterdir())
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["seed"] == 1234

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

    def test_empty_steps_workflow_records_completed_status(
        self, tmp_path, fake_pipeline
    ):
        from dw.workflow import Workflow

        # Workflow with no steps but with a seed
        definition = {
            "id": "empty_steps_test",
            "seed": 42,
            "steps": [],
        }
        workflow = Workflow(definition, str(tmp_path), "/w/workflows/empty.json")
        result = workflow.run({})

        # Should return empty list
        assert result == []

        # Run directory should have been created with completed status
        run_dir = next((tmp_path / "empty").iterdir())
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["status"] == "completed"
        assert manifest["workflow"]["id"] == "empty_steps_test"


class TestRealizedWorkflow:
    def test_the_run_directory_holds_a_realized_copy(self, tmp_path, fake_pipeline):
        from dw.workflow import Workflow

        definition = _workflow_definition()
        definition["variables"] = {"prompt": "a default"}
        definition["steps"][0]["pipeline"]["arguments"]["prompt"] = "variable:prompt"
        Workflow(definition, str(tmp_path), "/w/workflows/ltx2/Gyre.json").run(
            {"prompt": "a cat"}
        )

        run_dir = next((tmp_path / "ltx2" / "Gyre").iterdir())
        realized = json.loads((run_dir / "workflow.json").read_text())
        assert realized["variables"] == {"prompt": "a cat"}
        assert realized["seed"] == 7
        # the reference stays, so the file is still runnable with overrides
        assert realized["steps"][0]["pipeline"]["arguments"]["prompt"] == (
            "variable:prompt"
        )

    def test_the_manifest_points_at_it(self, tmp_path, fake_pipeline):
        from dw.workflow import Workflow

        Workflow(
            _workflow_definition(), str(tmp_path), "/w/workflows/ltx2/Gyre.json"
        ).run({})

        run_dir = next((tmp_path / "ltx2" / "Gyre").iterdir())
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["workflow"]["realized"] == "workflow.json"
        assert manifest["workflow"]["prompts"] == []
        assert manifest["workflow"]["sub_workflows"] == {}

    def test_a_seedless_run_pins_the_seed_it_drew(self, tmp_path, fake_pipeline):
        from dw.workflow import Workflow

        definition = _workflow_definition()
        del definition["seed"]
        Workflow(definition, str(tmp_path), "/w/workflows/ltx2/Gyre.json").run({})

        run_dir = next((tmp_path / "ltx2" / "Gyre").iterdir())
        realized = json.loads((run_dir / "workflow.json").read_text())
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert isinstance(realized["seed"], int)
        assert realized["seed"] == manifest["seed"]

    def test_the_flat_layout_writes_none(self, tmp_path, fake_pipeline, monkeypatch):
        from dw.workflow import Workflow

        monkeypatch.setenv(OUTPUT_LAYOUT_ENV_VAR, FLAT_LAYOUT)
        Workflow(
            _workflow_definition(), str(tmp_path), "/w/workflows/ltx2/Gyre.json"
        ).run({})

        assert not list(tmp_path.rglob("workflow.json"))

    def test_writing_it_is_best_effort(self, tmp_path):
        from dw.runs import write_realized_workflow

        # a run directory that cannot be made - the run still succeeded
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        assert write_realized_workflow(str(blocker / "run"), {"id": "x"}) is None

    def test_writing_it_returns_the_path(self, tmp_path):
        from dw.runs import write_realized_workflow

        path = write_realized_workflow(str(tmp_path / "run"), {"id": "x"})
        assert path == str(tmp_path / "run" / "workflow.json")
        assert json.loads(open(path).read()) == {"id": "x"}


class TestSubfolders:
    def test_a_step_without_a_subfolder_writes_where_it_always_did(
        self, tmp_path, fake_pipeline
    ):
        from dw.workflow import Workflow

        Workflow(_workflow_definition(), str(tmp_path), "/w/workflows/Gyre.json").run(
            {}
        )
        (run,) = (tmp_path / "Gyre").iterdir()
        assert (run / "runs_test-gen0.0-0.0.png").is_file()
        assert not any(child.is_dir() for child in run.iterdir())

    def test_a_step_with_a_subfolder_writes_into_it(self, tmp_path, fake_pipeline):
        from dw.workflow import Workflow

        Workflow(_foldered_definition(), str(tmp_path), "/w/workflows/Gyre.json").run(
            {}
        )
        (run,) = (tmp_path / "Gyre").iterdir()
        assert (run / "final" / "runs_test-gen0.0-0.0.png").is_file()
        assert not (run / "runs_test-gen0.0-0.0.png").exists()
        # The manifest still sits at the root of the run
        assert (run / "manifest.json").is_file()

    def test_a_nested_subfolder_is_created(self, tmp_path, fake_pipeline):
        from dw.workflow import Workflow

        Workflow(
            _foldered_definition("shots/act-1"), str(tmp_path), "/w/workflows/Gyre.json"
        ).run({})
        (run,) = (tmp_path / "Gyre").iterdir()
        assert (run / "shots" / "act-1" / "runs_test-gen0.0-0.0.png").is_file()

    def test_step_output_dir_is_the_run_directory_without_a_subfolder(self, tmp_path):
        from dw.workflow import Workflow

        workflow = Workflow(_workflow_definition(), str(tmp_path), "/w/workflows/Gyre.json")
        workflow._run_dir = str(tmp_path / "Gyre" / "run")
        step = workflow.workflow_definition["steps"][0]
        assert workflow.step_output_dir(step) == workflow.effective_output_dir

    def test_step_output_dir_refuses_an_escape_at_run_time(self, tmp_path):
        from dw.security import SecurityError
        from dw.workflow import Workflow

        workflow = Workflow(
            _foldered_definition("../other"), str(tmp_path), "/w/workflows/Gyre.json"
        )
        workflow._run_dir = str(tmp_path / "Gyre" / "run")
        with pytest.raises(SecurityError):
            workflow.step_output_dir(workflow.workflow_definition["steps"][0])

    def test_the_pipeline_wrapper_is_pointed_at_the_subfolder(self, tmp_path, fake_pipeline):
        # A chain step's save_segments spill writes through the pipeline's
        # output_dir, so it has to be the step's directory, not the run's
        from dw.workflow import Workflow

        workflow = Workflow(_foldered_definition(), str(tmp_path), "/w/workflows/Gyre.json")
        workflow._run_dir = str(tmp_path / "Gyre" / "run")
        step = workflow.workflow_definition["steps"][0]
        action = workflow.create_step_action(step, {}, {}, 7, "cpu")
        assert action.output_dir == os.path.join(workflow.effective_output_dir, "final")

    def test_a_separator_in_file_base_name_is_refused_at_save(self, tmp_path):
        from dw.result import Result
        from dw.security import InvalidInputError

        result = Result({"content_type": "image/png", "file_base_name": "final/"})
        with pytest.raises(InvalidInputError, match="subfolder"):
            result.save(str(tmp_path), "w-step.0")
