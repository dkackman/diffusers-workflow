"""A pipeline step whose component_type/scheduler_type/config_type diffusers
does not export.

`validate_workflow` used to answer `valid: true` for a misspelled class name -
`configuration.component_type` is resolved lazily, deep inside pipeline
construction, so the job queued, the worker loaded a checkpoint the plan had
already quoted a download for, and only then died on diffusers' own
AttributeError, ~3s in (#345). This mirrors #285's task_signature_errors:
checked against the same resolver the run itself uses for a '*_type' value
(`type_helpers.load_type_from_name`), so the rule can never refuse a name
that would in fact have run.
"""

import json
import pathlib

import pytest

from dw.introspection import component_type_errors

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def pipeline_step(component_type, name="a", extra=None):
    step = {
        "name": name,
        "pipeline": {"configuration": {"component_type": component_type}},
        "result": {"content_type": "image/png"},
    }
    if extra:
        step["pipeline"].update(extra)
    return step


def errors_for(component_type):
    return component_type_errors({"id": "ct", "steps": [pipeline_step(component_type)]})


class TestAMisspelledClass:
    def test_it_is_refused_with_a_suggestion(self):
        errors = errors_for("FluxPipelin")
        assert len(errors) == 1
        assert errors[0]["path"] == "steps[0].pipeline.configuration.component_type"
        assert "does not exist" in errors[0]["message"]
        assert "FluxPipeline" in errors[0]["message"]

    def test_a_real_class_is_accepted(self):
        assert errors_for("FluxPipeline") == []

    def test_a_real_dotted_class_outside_the_narrow_allowlist_is_accepted(self):
        """These name modules that are not on introspection's own
        ALLOWED_MODULES (sdnq only) but are on the runtime's broader
        TRUSTED_TOP_LEVEL_PACKAGES, and are real, shipped catalog entries -
        the check must resolve them the way the run itself does, not refuse
        something that would have worked."""
        assert errors_for("transformers.AutoProcessor") == []
        assert (
            errors_for(
                "dw.community_pipelines.pipeline_flux_rf_inversion."
                "RFInversionFluxPipeline"
            )
            == []
        )


class TestAllowlistedAbsentVsPresentButDisallowed:
    @pytest.mark.parametrize("trust", ["0", "1"])
    def test_an_absent_class_in_a_trusted_module_says_does_not_exist(
        self, monkeypatch, trust
    ):
        """Untrusted too: an allowlisted module's missing class is a
        misspelling, not the disallowed-module refusal below."""
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", trust)
        errors = component_type_errors(
            {
                "id": "ct",
                "steps": [
                    pipeline_step(
                        None,
                        extra={
                            "quantization_config": {"config_type": "sdnq.NoSuchConfig"}
                        },
                    )
                ],
            }
        )
        assert len(errors) == 1
        assert "does not exist" in errors[0]["message"]

    def test_a_class_in_a_disallowed_module_says_not_allowed(self, monkeypatch):
        # This suite trusts workflows by default (conftest.py); the trust
        # gate itself is only exercised untrusted, matching
        # tests/test_workflow_trust.py's own pattern.
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
        errors = errors_for("os.system")
        assert len(errors) == 1
        assert "does not exist" not in errors[0]["message"]
        assert "outside the ecosystem" in errors[0]["message"]

    def test_a_code_loader_in_an_allowed_package_says_not_a_class(self, monkeypatch):
        """#409: 'torch.hub.load' is in torch, so the top-level allowlist
        passed it - but it is a function, and a '*_type' value is called
        with the workflow's own arguments."""
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
        definition = {
            "id": "ct",
            "steps": [
                pipeline_step(
                    None,
                    extra={"quantization_config": {"config_type": "torch.hub.load"}},
                )
            ],
        }
        errors = component_type_errors(definition)
        assert [e["path"] for e in errors] == [
            "steps[0].pipeline.quantization_config.config_type"
        ]
        assert "not a class" in errors[0]["message"]

    def test_a_real_config_type_is_still_accepted_untrusted(self, monkeypatch):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
        definition = {
            "id": "ct",
            "steps": [
                pipeline_step(
                    None,
                    extra={"quantization_config": {"config_type": "sdnq.SDNQConfig"}},
                )
            ],
        }
        assert component_type_errors(definition) == []


class TestSchedulerAndQuantizationFields:
    def test_a_misspelled_scheduler_type_is_refused(self):
        definition = {
            "id": "ct",
            "steps": [
                {
                    "name": "a",
                    "pipeline": {
                        "scheduler": {
                            "configuration": {"scheduler_type": "DDIMSchedulr"}
                        }
                    },
                    "result": {"content_type": "image/png"},
                }
            ],
        }
        errors = component_type_errors(definition)
        assert len(errors) == 1
        assert errors[0]["path"] == (
            "steps[0].pipeline.scheduler.configuration.scheduler_type"
        )

    def test_a_real_config_type_is_accepted(self):
        definition = {
            "id": "ct",
            "steps": [
                pipeline_step(
                    None,
                    extra={"quantization_config": {"config_type": "sdnq.SDNQConfig"}},
                )
            ],
        }
        assert component_type_errors(definition) == []


class TestNoDownloadIsQuotedForARefusedStep:
    def test_the_refusal_reaches_validation_errors(self, tmp_path):
        """POST /api/validate builds `plan` (and its downloads_required) only
        when validation_errors is empty - see dw/server/app.py's validate
        route - so a misspelled class on a step that names a checkpoint must
        surface there, or the plan quotes a download for a step that cannot
        run."""
        from dw.workflow import Workflow

        step = pipeline_step(
            "FluxPipelin",
            extra={
                "from_pretrained_arguments": {"model_name": "org/model"},
                "arguments": {"prompt": "a cat"},
            },
        )
        definition = {"id": "ct", "steps": [step]}
        workflow = Workflow(definition, str(tmp_path), str(tmp_path / "ct.json"))

        paths = [error["path"] for error in workflow.validation_errors()]
        assert "steps[0].pipeline.configuration.component_type" in paths


def step_with(pipeline_extra=None, task=None):
    step = {"name": "a", "result": {"content_type": "image/png"}}
    if task is not None:
        step["task"] = task
    else:
        step["pipeline"] = {
            "configuration": {"component_type": "StableDiffusionPipeline"},
            **(pipeline_extra or {}),
        }
    return {"id": "ct", "steps": [step]}


class TestDtypeKeysAtValidate:
    """A 'torch_dtype' (or any key the run loads as a type) is held to the
    same gate at validate as the three '*_type' keys - it was checked only
    at run time, after the queue (#409's bounce, SE-F031 f)."""

    @pytest.fixture(autouse=True)
    def _untrusted(self, monkeypatch):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")

    def _errors(self, value):
        return component_type_errors(
            step_with({"from_pretrained_arguments": {"torch_dtype": value}})
        )

    @pytest.mark.parametrize(
        "value, says",
        [("torch.hub.load", "not a class"), ("os.system", "outside the ecosystem")],
    )
    def test_a_non_dtype_is_refused_at_its_path(self, value, says):
        errors = self._errors(value)
        assert [e["path"] for e in errors] == [
            "steps[0].pipeline.from_pretrained_arguments.torch_dtype"
        ]
        assert says in errors[0]["message"]

    @pytest.mark.parametrize("value", ["torch.bfloat16", "torch.float16", "{nf4}"])
    def test_a_dtype_or_escaped_value_passes(self, value):
        assert self._errors(value) == []

    def test_a_non_type_key_is_left_alone(self):
        assert component_type_errors(step_with({"offload_type": "model"})) == []

    def test_trusted_is_unchanged(self, monkeypatch):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
        assert self._errors("torch.hub.load") == []


class TestConstantsAtValidate:
    """A literal 'constant:' in a step is resolved at validate the way the
    run resolves it, so the untrusted walk rules refuse it before the queue
    rather than only at run time (#409's bounce, SE-F032)."""

    @pytest.fixture(autouse=True)
    def _untrusted(self, monkeypatch):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")

    def _errors(self, value):
        return component_type_errors(
            step_with({"arguments": {"cross_attention_kwargs": value}})
        )

    @pytest.mark.parametrize(
        "name, says",
        [
            ("torch.os.environ", "outside the ecosystem"),
            ("transformers.utils.hub.os.environ", "outside the ecosystem"),
            ("os.environ", "outside the ecosystem"),
            ("torch._C", "private name"),
            ("diffusers.__builtins__", "private name"),
            ("torch.nn.Module.__subclasses__", "private name"),
            ("torch.hub.load", "not a constant"),
            ("diffusers.NO_SUCH_CONSTANT", "No constant named"),
        ],
    )
    def test_a_refused_walk_is_reported_at_its_path(self, name, says):
        errors = self._errors(f"constant:{name}")
        assert [e["path"] for e in errors] == [
            "steps[0].pipeline.arguments.cross_attention_kwargs"
        ]
        assert says in errors[0]["message"]

    def test_a_constant_in_a_list_is_reported_at_its_index(self):
        errors = component_type_errors(
            step_with({"arguments": {"sigmas": [1.0, "constant:torch.os.sep"]}})
        )
        assert [e["path"] for e in errors] == ["steps[0].pipeline.arguments.sigmas[1]"]

    def test_a_constant_in_task_arguments_is_checked_too(self):
        errors = component_type_errors(
            step_with(
                task={"command": "x", "arguments": {"a": "constant:torch.os.environ"}}
            )
        )
        assert [e["path"] for e in errors] == ["steps[0].task.arguments.a"]

    def test_the_dataclass_walk_still_validates(self):
        name = (
            "diffusers.pipelines.ltx2.utils."
            "GEMMA4_PROMPT_ENHANCEMENT_CONFIG.max_new_tokens"
        )
        assert self._errors(f"constant:{name}") == []

    def test_it_reaches_validation_errors(self, tmp_path):
        from dw.workflow import Workflow

        definition = step_with(
            {
                "from_pretrained_arguments": {
                    "model_name": "org/model",
                    "torch_dtype": "torch.hub.load",
                },
                "arguments": {
                    "prompt": "a cat",
                    "cross_attention_kwargs": "constant:torch.os.environ",
                },
            }
        )
        workflow = Workflow(definition, str(tmp_path), str(tmp_path / "ct.json"))
        paths = [error["path"] for error in workflow.validation_errors()]
        assert "steps[0].pipeline.arguments.cross_attention_kwargs" in paths
        assert "steps[0].pipeline.from_pretrained_arguments.torch_dtype" in paths


class TestTheCatalogItself:
    """Every workflow shipped in the repo passes the new check."""

    @pytest.mark.parametrize(
        "path",
        sorted(
            str(p.relative_to(REPO_ROOT))
            for p in list((REPO_ROOT / "workflows").rglob("*.json"))
            + list((REPO_ROOT / "dw" / "workflows").glob("*.json"))
        ),
    )
    def test_workflow_has_no_component_type_error(self, path, monkeypatch):
        # Untrusted, the server's default posture - the dtype keys and
        # constant: values the walk now covers must all pass it (#409)
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
        definition = json.loads((REPO_ROOT / path).read_text())
        if not isinstance(definition, dict) or "steps" not in definition:
            pytest.skip("not a workflow")
        assert component_type_errors(definition) == []
