"""
Unit tests for the --trust-workflows gate: dw/security.py's trust helpers,
their wiring into dw/type_helpers.py's dotted-name loader, and into
dw/pipeline_processors/pipeline.py's pre_load_modules loop.

tests/conftest.py's autouse _trust_workflows_by_default fixture sets
DW_TRUST_WORKFLOWS=1 for the whole suite, so every test here that wants to
exercise untrusted behavior overrides it back with monkeypatch.
"""

from unittest.mock import patch

import pytest

from dw.security import (
    UntrustedWorkflowError,
    require_trusted_dotted_name,
    require_trusted_pre_load_modules,
    set_trust_workflows,
    workflows_are_trusted,
)
from dw.type_helpers import load_type_from_full_name
from dw.pipeline_processors.pipeline import Pipeline


def _untrust(monkeypatch):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")


class TestTrustFlag:
    def test_set_trust_workflows_true(self, monkeypatch):
        set_trust_workflows(True)
        assert workflows_are_trusted() is True

    def test_set_trust_workflows_false(self, monkeypatch):
        set_trust_workflows(False)
        assert workflows_are_trusted() is False

    def test_unset_defaults_to_untrusted(self, monkeypatch):
        monkeypatch.delenv("DW_TRUST_WORKFLOWS", raising=False)
        assert workflows_are_trusted() is False


class TestRequireTrustedDottedName:
    def test_in_ecosystem_allowed_when_untrusted(self, monkeypatch):
        _untrust(monkeypatch)
        for name in (
            "torch.bfloat16",
            "diffusers.FluxPipeline",
            "transformers.Foo",
            "sdnq.SDNQConfig",
        ):
            require_trusted_dotted_name(name, "a *_type value")  # must not raise

    def test_out_of_ecosystem_refused_when_untrusted(self, monkeypatch):
        _untrust(monkeypatch)
        with pytest.raises(UntrustedWorkflowError, match="trust-workflows"):
            require_trusted_dotted_name("os.system", "a *_type value")

    def test_out_of_ecosystem_allowed_when_trusted(self, monkeypatch):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
        require_trusted_dotted_name("os.system", "a *_type value")  # must not raise


class TestRequireTrustedPreLoadModules:
    def test_empty_never_refused(self, monkeypatch):
        _untrust(monkeypatch)
        require_trusted_pre_load_modules([])  # must not raise

    def test_in_ecosystem_module_allowed_when_untrusted(self, monkeypatch):
        # sdnq is the pattern the bundled example workflows use
        # (pre_load_modules registers its quantization method with diffusers)
        _untrust(monkeypatch)
        require_trusted_pre_load_modules(["sdnq"])  # must not raise

    def test_out_of_ecosystem_refused_when_untrusted(self, monkeypatch):
        _untrust(monkeypatch)
        with pytest.raises(UntrustedWorkflowError, match="trust-workflows"):
            require_trusted_pre_load_modules(["some_untrusted_module"])

    def test_out_of_ecosystem_allowed_when_trusted(self, monkeypatch):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
        require_trusted_pre_load_modules(["some_untrusted_module"])  # must not raise


class TestLoadTypeFromFullName:
    """Integration: the dotted-name loader in dw/type_helpers.py."""

    def test_in_ecosystem_dotted_type_works_untrusted(self, monkeypatch):
        _untrust(monkeypatch)
        result = load_type_from_full_name("torch.bfloat16")
        import torch

        assert result is torch.bfloat16

    def test_out_of_ecosystem_dotted_type_refused_untrusted(self, monkeypatch):
        _untrust(monkeypatch)
        with pytest.raises(UntrustedWorkflowError, match="trust-workflows"):
            load_type_from_full_name("os.path.join")

    def test_out_of_ecosystem_dotted_type_allowed_when_trusted(self, monkeypatch):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
        result = load_type_from_full_name("os.path.join")
        import os

        assert result is os.path.join


class TestPipelinePreLoadModulesGate:
    """Integration: Pipeline.load() refuses pre_load_modules when untrusted,
    and reaches past the gate when trusted (mocking the rest of load(),
    which needs a real model to get further)."""

    def _pipeline(self, module_names):
        definition = {
            "configuration": {"pre_load_modules": module_names},
            "from_pretrained_arguments": {"model_name": "some/repo"},
            "arguments": {},
        }
        return Pipeline(definition, 0, "cpu")

    def test_refused_when_untrusted(self, monkeypatch):
        _untrust(monkeypatch)
        pipeline = self._pipeline(["json"])
        with pytest.raises(UntrustedWorkflowError, match="trust-workflows"):
            pipeline.load(shared_components={})

    def test_allowed_when_trusted(self, monkeypatch):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
        pipeline = self._pipeline(["json"])
        # Past the trust gate, load() goes on to build a real pipeline - mock
        # that part out and just assert the gate did not fire
        with patch.object(
            Pipeline,
            "populate_from_pretrained_arguments",
            side_effect=RuntimeError("stop here - past the trust gate"),
        ):
            with pytest.raises(RuntimeError, match="stop here"):
                pipeline.load(shared_components={})

    def test_no_pre_load_modules_never_refused(self, monkeypatch):
        _untrust(monkeypatch)
        pipeline = self._pipeline([])
        with patch.object(
            Pipeline,
            "populate_from_pretrained_arguments",
            side_effect=RuntimeError("stop here - past the trust gate"),
        ):
            with pytest.raises(RuntimeError, match="stop here"):
                pipeline.load(shared_components={})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestConstantReferencesAreGated:
    """A 'constant:' reference imports the module it names before anything
    reads the attribute, so it is the same code-execution surface as a
    dotted type - gated the same way."""

    def test_out_of_ecosystem_constant_refused_when_untrusted(self, monkeypatch):
        from dw.type_helpers import load_constant_from_name

        _untrust(monkeypatch)
        with pytest.raises(UntrustedWorkflowError, match="trust-workflows"):
            load_constant_from_name("os.sep")

    def test_in_ecosystem_constant_allowed_when_untrusted(self, monkeypatch):
        from dw.type_helpers import load_constant_from_name

        _untrust(monkeypatch)
        assert load_constant_from_name("torch.float16") is not None

    def test_out_of_ecosystem_constant_allowed_when_trusted(self, monkeypatch):
        from dw.type_helpers import load_constant_from_name

        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
        assert load_constant_from_name("os.sep") == "/"


class TestRemoteCodeIsGated:
    """diffusers/transformers' own remote-code paths - trust_remote_code and
    custom_pipeline in from_pretrained_arguments - download and execute
    Python from the Hub, which is arbitrary code an untrusted workflow must
    not be able to reach whatever the importlib gate refuses."""

    def test_trust_remote_code_refused_when_untrusted(self, monkeypatch):
        from dw.security import require_trusted_from_pretrained_arguments

        _untrust(monkeypatch)
        with pytest.raises(UntrustedWorkflowError, match="trust_remote_code"):
            require_trusted_from_pretrained_arguments(
                {"model_name": "a/b", "trust_remote_code": True}, "transformer"
            )

    def test_custom_pipeline_refused_when_untrusted(self, monkeypatch):
        from dw.security import require_trusted_from_pretrained_arguments

        _untrust(monkeypatch)
        with pytest.raises(UntrustedWorkflowError, match="custom_pipeline"):
            require_trusted_from_pretrained_arguments(
                {"model_name": "a/b", "custom_pipeline": "someone/repo"}, "pipeline"
            )

    def test_plain_arguments_and_a_false_flag_pass_when_untrusted(self, monkeypatch):
        from dw.security import require_trusted_from_pretrained_arguments

        _untrust(monkeypatch)
        require_trusted_from_pretrained_arguments({"model_name": "a/b"}, "x")
        require_trusted_from_pretrained_arguments(
            {"model_name": "a/b", "trust_remote_code": False}, "x"
        )

    def test_allowed_when_trusted(self, monkeypatch):
        from dw.security import require_trusted_from_pretrained_arguments

        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
        require_trusted_from_pretrained_arguments(
            {"trust_remote_code": True, "custom_pipeline": "x/y"}, "x"
        )

    def test_load_component_refuses_before_touching_the_hub(self, monkeypatch):
        from unittest.mock import MagicMock
        from dw.pipeline_processors.pipeline import load_component

        _untrust(monkeypatch)
        component_type = MagicMock()
        component_type.__name__ = "MockPipeline"
        with pytest.raises(UntrustedWorkflowError):
            load_component(
                "pipeline",
                {"component_type": component_type},
                {"model_name": "a/b", "trust_remote_code": True},
                "cpu",
            )
        component_type.from_pretrained.assert_not_called()
