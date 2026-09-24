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
from dw.workflow import Workflow


def _untrusted_constant_workflow():
    """A minimal workflow whose only variable defaults to a 'constant:'
    reference outside the diffusers ecosystem - the same code-execution
    surface as a dotted '*_type' value, gated by require_trusted_dotted_name
    via dw.type_helpers.load_constant_from_name."""
    return {
        "id": "untrusted_constant",
        "variables": {"sep": "constant:os.sep"},
        "steps": [
            {
                "name": "step",
                "task": {
                    "command": "compose_text",
                    "arguments": {"parts": ["variable:sep"]},
                },
                "result": {"content_type": "text/plain"},
            }
        ],
    }


def _untrust(monkeypatch):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")


class TestTrustFlag:
    def test_set_trust_workflows_true(self, monkeypatch):
        # conftest already trusts; start untrusted so this can fail
        _untrust(monkeypatch)
        assert workflows_are_trusted() is False
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
        result = load_type_from_full_name("torch.nn.Linear", "component_type")
        import torch

        assert result is torch.nn.Linear

    def test_a_dtype_resolves_untrusted_under_a_dtype_key(self, monkeypatch):
        _untrust(monkeypatch)
        import torch

        for key in ("dtype", "torch_dtype", "compute_dtype"):
            assert load_type_from_full_name("torch.bfloat16", key) is torch.bfloat16

    @pytest.mark.parametrize("key", [None, "component_type", "config_type"])
    def test_a_dtype_is_refused_untrusted_under_a_type_key(self, monkeypatch, key):
        _untrust(monkeypatch)
        with pytest.raises(UntrustedWorkflowError, match="not a class"):
            load_type_from_full_name("torch.bfloat16", key)

    @pytest.mark.parametrize(
        "name",
        ["torch.hub.load", "torch.load", "diffusers.utils.load_image", "torch.hub"],
    )
    def test_an_in_ecosystem_non_class_is_refused_untrusted(self, monkeypatch, name):
        _untrust(monkeypatch)
        with pytest.raises(UntrustedWorkflowError, match="not a class"):
            load_type_from_full_name(name, "config_type")

    def test_a_bare_non_class_is_refused_untrusted(self, monkeypatch):
        from dw.type_helpers import load_type_from_name

        _untrust(monkeypatch)
        with pytest.raises(UntrustedWorkflowError, match="not a class"):
            load_type_from_name("utils", "component_type")

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
        assert load_constant_from_name("torch.os.sep") == "/"

    @pytest.mark.parametrize(
        "name",
        ["torch.os.environ", "torch.os.sep", "diffusers.utils.constants.os.environ"],
    )
    def test_a_walk_through_an_outside_module_is_refused(self, monkeypatch, name):
        from dw.type_helpers import load_constant_from_name

        _untrust(monkeypatch)
        with pytest.raises(UntrustedWorkflowError, match="'os' module"):
            load_constant_from_name(name)

    @pytest.mark.parametrize(
        "name", ["torch._C", "torch.__dict__", "diffusers._version.__version__"]
    )
    def test_a_private_segment_is_refused(self, monkeypatch, name):
        from dw.type_helpers import load_constant_from_name

        _untrust(monkeypatch)
        with pytest.raises(UntrustedWorkflowError, match="private name"):
            load_constant_from_name(name)

    def test_a_walk_into_a_dataclass_still_resolves(self, monkeypatch):
        from dw.type_helpers import load_constant_from_name
        from diffusers.pipelines.ltx2.utils import GEMMA4_PROMPT_ENHANCEMENT_CONFIG

        _untrust(monkeypatch)
        assert (
            load_constant_from_name(
                "diffusers.pipelines.ltx2.utils."
                "GEMMA4_PROMPT_ENHANCEMENT_CONFIG.max_new_tokens"
            )
            == GEMMA4_PROMPT_ENHANCEMENT_CONFIG.max_new_tokens
        )


class TestValidationGatesUntrustedConstantDefaults:
    """Workflow.validation_errors() realizes 'constant:' variable defaults
    (Workflow.expanded_definition), which runs the same trust gate a run
    would. An out-of-ecosystem name must come back as a validation error
    naming the variable, not an unhandled UntrustedWorkflowError, and the
    same definition must validate cleanly once trusted."""

    def test_untrusted_constant_default_is_a_validation_error(
        self, monkeypatch, tmp_path
    ):
        _untrust(monkeypatch)
        workflow = Workflow(_untrusted_constant_workflow(), str(tmp_path), "")
        try:
            require_trusted_dotted_name("os.sep", "a constant: reference")
        except UntrustedWorkflowError as e:
            expected_message = str(e)
        assert workflow.validation_errors() == [
            {"path": "variables.sep", "message": expected_message}
        ]

    def test_untrusted_constant_default_validates_when_trusted(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
        workflow = Workflow(_untrusted_constant_workflow(), str(tmp_path), "")
        assert workflow.validation_errors() == []


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


class TestTrustPreflightRunsBeforeTheLoadingMarker:
    """The gates themselves are inside load()/load_component(), which is where
    the boundary belongs - but load() is entered under a 'loading' phase
    event, so a refused run emitted the same marker as one that loaded a model
    and then failed, and job events could no longer tell 'refused before load'
    from 'loaded, then refused' (#137)."""

    def _pipeline(self, definition):
        return Pipeline(definition, 0, "cpu")

    def test_remote_code_refused_by_the_preflight(self, monkeypatch):
        _untrust(monkeypatch)
        pipeline = self._pipeline(
            {
                "configuration": {},
                "from_pretrained_arguments": {
                    "model_name": "a/b",
                    "trust_remote_code": True,
                },
                "arguments": {},
            }
        )
        with pytest.raises(UntrustedWorkflowError, match="trust_remote_code"):
            pipeline.check_trusted()

    def test_a_component_block_is_covered_too(self, monkeypatch):
        _untrust(monkeypatch)
        pipeline = self._pipeline(
            {
                "configuration": {},
                "from_pretrained_arguments": {"model_name": "a/b"},
                "transformer": {
                    "configuration": {},
                    "from_pretrained_arguments": {
                        "model_name": "c/d",
                        "custom_pipeline": "someone/repo",
                    },
                },
                "arguments": {},
            }
        )
        with pytest.raises(UntrustedWorkflowError, match="custom_pipeline"):
            pipeline.check_trusted()

    def test_pre_load_modules_are_covered_too(self, monkeypatch):
        _untrust(monkeypatch)
        pipeline = self._pipeline(
            {
                "configuration": {"pre_load_modules": ["json"]},
                "from_pretrained_arguments": {"model_name": "a/b"},
                "arguments": {},
            }
        )
        with pytest.raises(UntrustedWorkflowError, match="pre_load_modules"):
            pipeline.check_trusted()

    def test_an_ordinary_definition_passes(self, monkeypatch):
        _untrust(monkeypatch)
        pipeline = self._pipeline(
            {
                "configuration": {"component_type": "StableDiffusionPipeline"},
                "from_pretrained_arguments": {"model_name": "a/b"},
                "arguments": {"prompt": "an apple"},
            }
        )
        pipeline.check_trusted()

    def test_a_refused_run_emits_no_loading_phase(self, monkeypatch, tmp_path):
        """End to end through Workflow.run: the events a consumer reads carry
        no 'loading' marker for a run the gate refused."""
        from dw.events import RunContext, activate_context, deactivate_context

        _untrust(monkeypatch)
        events = []
        token = activate_context(RunContext(on_event=events.append))
        try:
            workflow = Workflow(
                {
                    "id": "untrusted_remote_code",
                    "steps": [
                        {
                            "name": "main",
                            "pipeline": {
                                "configuration": {
                                    "component_type": "StableDiffusionPipeline"
                                },
                                "from_pretrained_arguments": {
                                    "model_name": "a/b",
                                    "trust_remote_code": True,
                                },
                                "arguments": {"prompt": "an apple"},
                            },
                            "result": {"content_type": "image/jpeg"},
                        }
                    ],
                },
                str(tmp_path / "outputs"),
                "",
            )
            with pytest.raises(UntrustedWorkflowError):
                workflow.run({})
        finally:
            deactivate_context(token)

        phases = [e for e in events if e.get("event") == "phase"]
        assert not any(p.get("phase") == "loading" for p in phases), phases


def _catalog_files():
    """Every workflow JSON the repo ships: the runnable catalog and the
    packaged builtins its sub-workflow steps name."""
    import glob
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return sorted(
        os.path.relpath(path, root)
        for tree in ("workflows", os.path.join("dw", "workflows"))
        for path in glob.glob(os.path.join(root, tree, "**", "*.json"), recursive=True)
    )


def _catalog_references(node):
    """Every '*_type' / '*_dtype' / 'dtype' value a run would load, keyed by
    the key it sits under, and every 'constant:' reference."""
    from dw.arguments import NON_TYPE_KEYS, is_constant_reference, is_escaped

    if isinstance(node, dict):
        for key, value in node.items():
            if (
                isinstance(value, str)
                and (key.endswith("_type") or key.endswith("_dtype") or key == "dtype")
                and key not in NON_TYPE_KEYS
                and key != "media_type"
                and not is_escaped(value)
                and ":" not in value
            ):
                yield "type", key, value
            elif is_constant_reference(value):
                yield "constant", key, value
            else:
                yield from _catalog_references(value)
    elif isinstance(node, list):
        for value in node:
            yield from _catalog_references(value)


class TestTheCatalogResolvesUntrusted:
    """Tightening the untrusted gate to classes and dtypes, and keeping a
    constant's walk inside the allowed packages (#407), must not refuse a
    name any shipped workflow uses."""

    @pytest.mark.parametrize("workflow_file", _catalog_files())
    def test_every_type_and_constant_resolves(self, monkeypatch, workflow_file):
        import json
        import os

        from dw.arguments import fetch_constant
        from dw.type_helpers import load_type_from_name

        _untrust(monkeypatch)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, workflow_file), encoding="utf-8") as file:
            definition = json.load(file)

        for kind, key, value in _catalog_references(definition):
            try:
                if kind == "type":
                    load_type_from_name(value, key)
                else:
                    fetch_constant(value)
            except UntrustedWorkflowError as error:
                pytest.fail(f"{workflow_file}: {key}={value!r} refused: {error}")
            except (ImportError, AttributeError, ValueError):
                # Not installed here, or not a type at all - test_examples
                # owns whether a name resolves; this owns whether it is refused
                pass
