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
    def test_an_absent_class_in_a_trusted_module_says_does_not_exist(self):
        errors = component_type_errors(
            {
                "id": "ct",
                "steps": [
                    pipeline_step(
                        None,
                        extra={"quantization_config": {"config_type": "sdnq.NoSuchConfig"}},
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

    def test_the_two_messages_differ(self, monkeypatch):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
        absent = component_type_errors(
            {
                "id": "ct",
                "steps": [
                    pipeline_step(
                        None,
                        extra={"quantization_config": {"config_type": "sdnq.NoSuchConfig"}},
                    )
                ],
            }
        )[0]["message"]
        disallowed = errors_for("os.system")[0]["message"]
        assert absent != disallowed


class TestSchedulerAndQuantizationFields:
    def test_a_misspelled_scheduler_type_is_refused(self):
        definition = {
            "id": "ct",
            "steps": [
                {
                    "name": "a",
                    "pipeline": {
                        "scheduler": {"configuration": {"scheduler_type": "DDIMSchedulr"}}
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
            "steps": [pipeline_step(None, extra={"quantization_config": {"config_type": "sdnq.SDNQConfig"}})],
        }
        assert component_type_errors(definition) == []


class TestNoDownloadIsQuotedForARefusedStep:
    def test_component_type_plays_no_part_in_collecting_sources(self):
        """POST /api/validate builds `plan` (and its downloads_required) only
        when validation_errors is empty - see dw/server/app.py's validate
        route - and component_type never feeds `_collect_sources` at all, so
        a step that fails this check was never going to contribute a
        download either way."""
        from dw.plan import _collect_sources

        definition = {"id": "ct", "steps": [pipeline_step("FluxPipelin")]}
        assert errors_for("FluxPipelin") != []
        names, urls = [], []
        _collect_sources(definition, names, urls)
        assert names == []
        assert urls == []


class TestTheCatalogItself:
    """Every workflow shipped in the repo passes the new check."""

    @pytest.mark.parametrize(
        "path",
        sorted(
            str(p)
            for p in list(pathlib.Path("workflows").rglob("*.json"))
            + list(pathlib.Path("dw/workflows").glob("*.json"))
        ),
    )
    def test_workflow_has_no_component_type_error(self, path):
        definition = json.loads(pathlib.Path(path).read_text())
        if not isinstance(definition, dict) or "steps" not in definition:
            pytest.skip("not a workflow")
        assert component_type_errors(definition) == []
