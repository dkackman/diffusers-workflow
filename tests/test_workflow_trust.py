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
