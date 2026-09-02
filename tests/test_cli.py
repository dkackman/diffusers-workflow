"""Tests for the console-script entry points.

dw.run and dw.validate are what most people actually type, and everything
they do before the workflow executes - name=value parsing, path validation,
the exit codes a script checks - had no coverage at all.
"""

import json
import os

import pytest

from dw import run as run_module
from dw import validate as validate_module


@pytest.fixture
def workflow_file(tmp_path):
    definition = {
        "id": "cli_test",
        "variables": {"prompt": "a landscape", "steps": 25},
        "steps": [
            {
                "name": "gen",
                "task": {"command": "list_images", "arguments": {}},
                "result": {"content_type": "image/png", "save": False},
            }
        ],
    }
    path = tmp_path / "cli_test.json"
    path.write_text(json.dumps(definition))
    return path


def invoke(module, monkeypatch, argv):
    """Run an entry point's main() with argv, returning its exit code."""
    monkeypatch.setattr("sys.argv", argv)
    try:
        module.main()
    except SystemExit as exit_call:
        return exit_call.code or 0
    return 0


class TestValidateEntryPoint:
    def test_a_valid_workflow_reports_success(self, workflow_file, monkeypatch, capsys):
        code = invoke(validate_module, monkeypatch, ["dw-validate", str(workflow_file)])
        assert code == 0
        assert "validated successfully" in capsys.readouterr().out

    def test_a_schema_violation_exits_nonzero(self, tmp_path, monkeypatch, capsys):
        path = tmp_path / "broken.json"
        path.write_text(json.dumps({"id": "broken"}))  # no steps

        code = invoke(validate_module, monkeypatch, ["dw-validate", str(path)])
        assert code == 1
        out = capsys.readouterr().out
        # Printed once - not doubled as "Error validating workflow: Validation error: ..."
        assert out.count("Validation error") == 1
        assert "Error validating workflow" not in out

    def test_a_schema_violation_names_the_json_path(
        self, workflow_file, monkeypatch, capsys
    ):
        definition = json.loads(workflow_file.read_text())
        # "seed" on a step must be an integer per the schema - give it a string
        definition["steps"][0]["seed"] = "not-a-number"
        workflow_file.write_text(json.dumps(definition))

        code = invoke(validate_module, monkeypatch, ["dw-validate", str(workflow_file)])
        assert code == 1
        out = capsys.readouterr().out
        assert "steps[0]" in out
        assert "seed" in out

    def test_a_traversing_path_is_refused_before_anything_is_read(
        self, monkeypatch, capsys
    ):
        code = invoke(
            validate_module, monkeypatch, ["dw-validate", "../../etc/passwd.json"]
        )
        assert code == 1
        assert "Security validation failed" in capsys.readouterr().out


class TestRunEntryPoint:
    def test_name_value_arguments_reach_the_workflow(
        self, workflow_file, tmp_path, monkeypatch
    ):
        seen = {}

        import dw.workflow

        def capture(self, arguments, *args, **kwargs):
            seen.update(arguments)
            return []

        monkeypatch.setattr(dw.workflow.Workflow, "run", capture)
        monkeypatch.setattr(run_module, "startup", lambda level: None)

        code = invoke(
            run_module,
            monkeypatch,
            [
                "dw-run",
                str(workflow_file),
                "-o",
                str(tmp_path / "outputs"),
                "prompt=a cat",
                "steps=4",
            ],
        )
        assert code == 0
        # every override arrives as a string; set_variables converts by the
        # type of the declared default
        assert seen == {"prompt": "a cat", "steps": "4"}

    def test_an_argument_without_an_equals_sign_is_rejected(
        self, workflow_file, monkeypatch, capsys
    ):
        monkeypatch.setattr(run_module, "startup", lambda level: None)
        code = invoke(
            run_module, monkeypatch, ["dw-run", str(workflow_file), "just_a_name"]
        )
        assert code == 1
        assert "not in name=value format" in capsys.readouterr().out

    def test_an_invalid_variable_name_is_rejected(
        self, workflow_file, monkeypatch, capsys
    ):
        monkeypatch.setattr(run_module, "startup", lambda level: None)
        code = invoke(
            run_module, monkeypatch, ["dw-run", str(workflow_file), "9bad=value"]
        )
        assert code == 1
        assert "Invalid variable input" in capsys.readouterr().out

    def test_prompt_dir_is_exported_for_the_engine(
        self, workflow_file, tmp_path, monkeypatch
    ):
        import dw.workflow

        library = tmp_path / "library"
        library.mkdir()
        monkeypatch.delenv("DW_PROMPT_DIR", raising=False)
        monkeypatch.setattr(dw.workflow.Workflow, "run", lambda *a, **k: [])
        monkeypatch.setattr(run_module, "startup", lambda level: None)

        code = invoke(
            run_module,
            monkeypatch,
            [
                "dw-run",
                str(workflow_file),
                "-o",
                str(tmp_path / "outputs"),
                "--prompt-dir",
                str(library),
            ],
        )
        assert code == 0
        assert os.environ["DW_PROMPT_DIR"] == str(library)

    def test_the_output_directory_is_created(
        self, workflow_file, tmp_path, monkeypatch
    ):
        import dw.workflow

        outputs = tmp_path / "fresh" / "outputs"
        monkeypatch.setattr(dw.workflow.Workflow, "run", lambda *a, **k: [])
        monkeypatch.setattr(run_module, "startup", lambda level: None)

        code = invoke(
            run_module,
            monkeypatch,
            ["dw-run", str(workflow_file), "-o", str(outputs)],
        )
        assert code == 0
        assert outputs.is_dir()

    def test_a_failing_run_exits_nonzero(
        self, workflow_file, tmp_path, monkeypatch, capsys
    ):
        import dw.workflow

        def explode(*args, **kwargs):
            raise RuntimeError("the model is on fire")

        monkeypatch.setattr(dw.workflow.Workflow, "run", explode)
        monkeypatch.setattr(run_module, "startup", lambda level: None)

        code = invoke(
            run_module,
            monkeypatch,
            ["dw-run", str(workflow_file), "-o", str(tmp_path / "outputs")],
        )
        assert code == 1
        assert "the model is on fire" in capsys.readouterr().out
