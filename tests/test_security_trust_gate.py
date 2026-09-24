"""The code-execution gate, from both sides, proven by what did *not* happen.

tests/test_workflow_trust.py shows each gate raises. What it cannot show is
*when*: a refusal that arrives as UntrustedWorkflowError after the module was
imported, its source file opened, or the Hub asked for code has already let
the code run - "refused too late" is the same as not refused. So every
untrusted case here names a probe module that exists on sys.path and would
write a marker file the moment its top-level code ran, installs an import
hook that records any attempt to find it, and fails the test on either.

The trusted half is the other boundary: `--trust-workflows` and
`DW_TRUST_WORKFLOWS` must let each surface through to a real import, or the
documented escape hatch is broken and operators reach for worse ones.

tests/conftest.py trusts every test by default; each test here states the
posture it runs under.
"""

import copy
import importlib.abc
import socket
import sys
import textwrap
from unittest.mock import MagicMock

import pytest

from dw.security import (
    TRUST_WORKFLOWS_ENV_VAR,
    UntrustedWorkflowError,
    set_trust_workflows,
    workflows_are_trusted,
)

PROBE = "dw_untrusted_probe_module"


class _ImportRecorder(importlib.abc.MetaPathFinder):
    """First on sys.meta_path: sees every import that reaches the finders,
    records the ones naming the probe, and lets the real finders answer."""

    def __init__(self, watched):
        self.watched = watched
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == self.watched:
            self.attempts.append(fullname)
        return None


@pytest.fixture
def probe(tmp_path, monkeypatch):
    """A module that would run if anything imported it, and the evidence.

    Yields an object with `.attempts` (import lookups for the probe) and
    `.executed()` (whether its top-level code ran). The module is on
    sys.path for real, so a gate that failed open would import it rather
    than fail on ModuleNotFoundError and look like a refusal.
    """
    package = tmp_path / "probe_path"
    package.mkdir()
    marker = tmp_path / "EXECUTED"
    (package / f"{PROBE}.py").write_text(
        textwrap.dedent(
            f"""
            import pathlib
            pathlib.Path({str(marker)!r}).write_text("ran")

            class Thing:
                def __init__(self, *args, **kwargs):
                    self.args = args
                    self.kwargs = kwargs

            VALUE = 7
            """
        )
    )
    monkeypatch.syspath_prepend(str(package))
    for name in list(sys.modules):
        if name == PROBE or name.startswith(PROBE + "."):
            monkeypatch.delitem(sys.modules, name)

    recorder = _ImportRecorder(PROBE)
    monkeypatch.setattr(sys, "meta_path", [recorder, *sys.meta_path])
    recorder.executed = marker.exists
    yield recorder
    sys.modules.pop(PROBE, None)


@pytest.fixture
def untrusted(monkeypatch):
    monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")


@pytest.fixture
def no_network(monkeypatch):
    """Any attempt to resolve or dial is a failure, not a slow test."""
    attempts = []

    def refuse(*args, **kwargs):
        attempts.append(args)
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    return attempts


def assert_nothing_happened(probe, no_network=()):
    assert probe.attempts == [], f"import attempted before refusal: {probe.attempts}"
    assert not probe.executed(), "the probe module's code ran"
    assert list(no_network) == [], "the network was touched before refusal"


# ------------------------------------------------------------------ surfaces


def _realize(arguments):
    """realize_args on a copy - it converts in place, and a parametrized
    dict realized once would reach the next case already a class."""
    from dw.arguments import realize_args

    arguments = copy.deepcopy(arguments)
    realize_args(arguments)
    return arguments


TYPE_REFERENCES = [
    pytest.param({"scheduler_type": f"{PROBE}.Thing"}, id="_type"),
    pytest.param({"component_type": f"{PROBE}.Thing"}, id="component_type"),
    pytest.param({"torch_dtype": f"{PROBE}.Thing"}, id="_dtype"),
    pytest.param({"dtype": f"{PROBE}.Thing"}, id="dtype"),
    pytest.param(
        {
            "quantization_config": {
                "configuration": {"config_type": f"{PROBE}.Thing"},
                "arguments": {},
            }
        },
        id="config_type",
    ),
    pytest.param(
        {"outer": {"inner": [{"weights_dtype": f"{PROBE}.Thing"}]}},
        id="nested",
    ),
    pytest.param(
        {"scheduler_type": f"{PROBE}.sub.Thing"},
        id="submodule",
    ),
]


def _pipeline(configuration=None, from_pretrained=None, **blocks):
    from dw.pipeline_processors.pipeline import Pipeline

    definition = {
        "configuration": configuration or {},
        "from_pretrained_arguments": from_pretrained or {"model_name": "a/b"},
        "arguments": {},
        **blocks,
    }
    return Pipeline(definition, 0, "cpu")


class TestUntrustedRefusesBeforeImport:
    @pytest.mark.parametrize("arguments", TYPE_REFERENCES)
    def test_a_dotted_type_reference(self, untrusted, probe, no_network, arguments):
        with pytest.raises(UntrustedWorkflowError, match="trust-workflows"):
            _realize(arguments)
        assert_nothing_happened(probe, no_network)

    @pytest.mark.parametrize(
        "modules",
        [
            [PROBE],
            [f"{PROBE}.sub"],
            # a trusted entry first must not import before the untrusted
            # one is looked at: every entry is checked, then any imported
            ["json", PROBE],
        ],
    )
    def test_pre_load_modules(self, untrusted, probe, no_network, modules):
        pipeline = _pipeline(configuration={"pre_load_modules": modules})
        imported_before = set(sys.modules)
        with pytest.raises(UntrustedWorkflowError, match="pre_load_modules"):
            pipeline.load(shared_components={})
        assert_nothing_happened(probe, no_network)
        assert PROBE not in set(sys.modules) - imported_before

    def test_pre_load_modules_are_refused_by_the_preflight_too(
        self, untrusted, probe, no_network
    ):
        pipeline = _pipeline(configuration={"pre_load_modules": [PROBE]})
        with pytest.raises(UntrustedWorkflowError):
            pipeline.check_trusted()
        assert_nothing_happened(probe, no_network)

    @pytest.mark.parametrize(
        "reference",
        [f"constant:{PROBE}.VALUE", f"constant:{PROBE}.sub.VALUE"],
    )
    def test_a_constant_reference(self, untrusted, probe, no_network, reference):
        from dw.arguments import fetch_constant

        with pytest.raises(UntrustedWorkflowError):
            fetch_constant(reference)
        assert_nothing_happened(probe, no_network)

    def test_a_constant_inside_arguments(self, untrusted, probe, no_network):
        with pytest.raises(UntrustedWorkflowError):
            _realize({"sigmas": f"constant:{PROBE}.VALUE"})
        assert_nothing_happened(probe, no_network)

    @pytest.mark.parametrize(
        "extra",
        [
            {"trust_remote_code": True},
            {"trust_remote_code": 1},
            {"trust_remote_code": "yes"},
            {"custom_pipeline": "someone/remote-pipeline"},
            {"custom_pipeline": "PROBE_PATH"},
        ],
    )
    def test_remote_code_arguments(self, untrusted, probe, no_network, tmp_path, extra):
        """Neither goes through our importlib, so the proof is that
        from_pretrained - the thing that would fetch and import - is never
        called, and no socket is opened."""
        from dw.pipeline_processors.pipeline import load_component

        if extra.get("custom_pipeline") == "PROBE_PATH":
            # a local custom pipeline is a .py diffusers would import
            extra = {"custom_pipeline": str(tmp_path / "probe_path")}
        component_type = MagicMock()
        component_type.__name__ = "ProbePipeline"
        with pytest.raises(UntrustedWorkflowError):
            load_component(
                "pipeline",
                {"component_type": component_type},
                {"model_name": "a/b", **extra},
                "cpu",
            )
        component_type.from_pretrained.assert_not_called()
        assert_nothing_happened(probe, no_network)

    @pytest.mark.parametrize("key", ["trust_remote_code", "custom_pipeline"])
    def test_remote_code_in_any_nested_block_is_seen_by_the_preflight(
        self, untrusted, no_network, key
    ):
        """A component inside a list (controlnets, loras, text encoders) is
        still a from_pretrained call the gate must see."""
        pipeline = _pipeline(
            controlnets=[
                {
                    "configuration": {},
                    "from_pretrained_arguments": {"model_name": "c/d", key: "x/y"},
                }
            ],
        )
        with pytest.raises(UntrustedWorkflowError, match=key):
            pipeline.check_trusted()
        assert list(no_network) == []


class TestValidationRefusesBeforeImport:
    """POST /api/validate and the pre-queue check call validation_errors -
    a refusal there must not itself have imported the module to decide."""

    def _workflow(self, pipeline, variables=None):
        from dw.workflow import Workflow

        definition = {
            "id": "trust_probe",
            "steps": [
                {
                    "name": "gen",
                    "pipeline": pipeline,
                    "result": {"content_type": "image/png"},
                }
            ],
        }
        if variables:
            definition["variables"] = variables
        return definition, Workflow

    # The pipeline itself is an escaped '{Fake}' rather than a real diffusers
    # class: resolving a real one imports bitsandbytes, whose CPU backend
    # asks the Hub for a kernel at import time - a network call these tests
    # must not make, and nothing to do with the gate under test
    @pytest.mark.parametrize(
        "pipeline, path",
        [
            pytest.param(
                {
                    "configuration": {"component_type": f"{PROBE}.Thing"},
                    "from_pretrained_arguments": {"model_name": "a/b"},
                    "arguments": {},
                },
                "steps[0].pipeline.configuration.component_type",
                id="component_type",
            ),
            pytest.param(
                {
                    "configuration": {"component_type": "{Fake}"},
                    "from_pretrained_arguments": {"model_name": "a/b"},
                    "scheduler": {
                        "configuration": {"scheduler_type": f"{PROBE}.Thing"}
                    },
                    "arguments": {},
                },
                "steps[0].pipeline.scheduler.configuration.scheduler_type",
                id="scheduler_type",
            ),
        ],
    )
    def test_a_dotted_type_is_a_validation_error(
        self, untrusted, probe, no_network, tmp_path, pipeline, path
    ):
        definition, Workflow = self._workflow(pipeline)
        errors = Workflow(definition, str(tmp_path), "").validation_errors()
        refusals = [e for e in errors if "trust-workflows" in e["message"]]
        assert [e["path"] for e in refusals] == [path]
        assert_nothing_happened(probe, no_network)

    def test_a_constant_default_is_a_validation_error(
        self, untrusted, probe, no_network, tmp_path
    ):
        definition, Workflow = self._workflow(
            {
                "configuration": {"component_type": "{Fake}"},
                "from_pretrained_arguments": {"model_name": "a/b"},
                "arguments": {"prompt": "variable:p"},
            },
            variables={"p": f"constant:{PROBE}.VALUE"},
        )
        errors = Workflow(definition, str(tmp_path), "").validation_errors()
        assert [e["path"] for e in errors] == ["variables.p"]
        assert_nothing_happened(probe, no_network)


class TestTheAllowlistIsNotAnEscapeHatch:
    """In-ecosystem names are allowed untrusted by top-level package. That is
    only safe if nothing reachable under those packages hands a workflow the
    code execution the gate exists to deny."""

    def test_config_type_cannot_name_a_code_loader_in_an_allowed_package(
        self, untrusted, monkeypatch, no_network
    ):
        import torch.hub

        from dw.pipeline_processors.config_objects import create_quantization_config

        loader = MagicMock(name="torch.hub.load")
        monkeypatch.setattr(torch.hub, "load", loader)
        definition = {
            "configuration": {"config_type": "torch.hub.load"},
            "arguments": {
                "repo_or_dir": "attacker/repo",
                "model": "anything",
                "trust_repo": True,
            },
        }
        try:
            realized = _realize({"quantization_config": definition})
            create_quantization_config(realized["quantization_config"])
        except UntrustedWorkflowError:
            pass
        loader.assert_not_called()

    def test_a_constant_cannot_walk_out_of_an_allowed_package(
        self, untrusted, monkeypatch
    ):
        from dw.arguments import fetch_constant

        monkeypatch.setenv("DW_API_TOKEN", "server-secret-probe")
        leaked = None
        try:
            leaked = fetch_constant("constant:torch.os.environ")
        except (UntrustedWorkflowError, ValueError):
            pass
        assert leaked is None or "server-secret-probe" not in str(leaked)

    @pytest.mark.parametrize(
        "name",
        [
            "os.system",
            "builtins.eval",
            "subprocess.Popen",
            "importlib.import_module",
            ".os",
            " torch.os",
            "torchx.Thing",
            "diffusersx.Thing",
            "dwx.Thing",
            "Torch.nn.Linear",
        ],
    )
    def test_near_misses_of_an_allowed_name_are_refused(self, untrusted, name):
        from dw.type_helpers import load_type_from_full_name

        with pytest.raises(UntrustedWorkflowError):
            load_type_from_full_name(name)


# ------------------------------------------------------------------ trusted


@pytest.fixture(params=["env", "flag"])
def trusted(request, monkeypatch):
    """Both ways a process ends up trusting workflows: the environment
    variable a spawned worker inherits, and the call the CLI flag makes."""
    if request.param == "env":
        monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "1")
    else:
        monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")
        set_trust_workflows(True)
    assert workflows_are_trusted()
    return request.param


class TestTrustedLetsEachSurfaceThrough:
    @pytest.mark.parametrize(
        "arguments, find",
        [
            ({"scheduler_type": f"{PROBE}.Thing"}, lambda a: a["scheduler_type"]),
            ({"torch_dtype": f"{PROBE}.Thing"}, lambda a: a["torch_dtype"]),
            ({"dtype": f"{PROBE}.Thing"}, lambda a: a["dtype"]),
        ],
    )
    def test_a_dotted_type_imports(self, trusted, probe, arguments, find):
        realized = _realize(arguments)
        assert find(realized).__name__ == "Thing"
        assert probe.executed()

    def test_a_config_type_imports_and_builds(self, trusted, probe):
        from dw.pipeline_processors.config_objects import create_quantization_config

        definition = {
            "configuration": {"config_type": f"{PROBE}.Thing"},
            "arguments": {"bits": 4},
        }
        realized = _realize({"quantization_config": definition})
        built = create_quantization_config(realized["quantization_config"])
        assert built.kwargs == {"bits": 4}
        assert probe.executed()

    def test_pre_load_modules_import(self, trusted, probe, monkeypatch):
        from dw.pipeline_processors.pipeline import Pipeline

        pipeline = _pipeline(configuration={"pre_load_modules": [PROBE]})
        monkeypatch.setattr(
            Pipeline,
            "populate_from_pretrained_arguments",
            MagicMock(side_effect=RuntimeError("stop - past the gate")),
        )
        with pytest.raises(RuntimeError, match="stop"):
            pipeline.load(shared_components={})
        assert PROBE in probe.attempts
        assert probe.executed()

    def test_a_constant_reads(self, trusted, probe):
        from dw.arguments import fetch_constant

        assert fetch_constant(f"constant:{PROBE}.VALUE") == 7
        assert probe.executed()

    @pytest.mark.parametrize(
        "extra",
        [{"trust_remote_code": True}, {"custom_pipeline": "someone/pipeline"}],
    )
    def test_remote_code_arguments_reach_from_pretrained(self, trusted, extra):
        from dw.pipeline_processors.pipeline import load_component

        component_type = MagicMock()
        component_type.__name__ = "ProbePipeline"
        load_component(
            "pipeline",
            {"component_type": component_type},
            {"model_name": "a/b", **extra},
            "cpu",
        )
        component_type.from_pretrained.assert_called_once()
        _, kwargs = component_type.from_pretrained.call_args
        for key, value in extra.items():
            assert kwargs[key] == value


class TestHowAProcessBecomesTrusted:
    """Only the flag trusts: an unset or unrecognized variable is untrusted,
    and a server started without --trust-workflows does not inherit trust
    from whatever launched it."""

    @pytest.mark.parametrize("value", ["", "0", "true", "yes", "TRUE", " 1", "1 "])
    def test_only_exactly_1_trusts(self, monkeypatch, value):
        monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, value)
        assert workflows_are_trusted() is False

    @pytest.fixture
    def serve(self, monkeypatch, tmp_path):
        import uvicorn

        import dw
        import dw.serve as serve_module
        from dw.server import app as app_module

        monkeypatch.setattr(app_module, "create_app", lambda **kwargs: object())
        monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: None)
        monkeypatch.setattr(dw, "startup", lambda *args, **kwargs: None)
        monkeypatch.delenv("DW_API_TOKEN", raising=False)
        monkeypatch.setenv("DW_PROMPT_DIR", str(tmp_path / "prompts"))
        monkeypatch.setenv("DW_ASSET_DIR", str(tmp_path / "assets"))
        monkeypatch.setenv("DW_WORKSPACE", str(tmp_path / "workspace"))
        monkeypatch.setenv("DW_WORKSPACE_SOURCE", "flag")
        (tmp_path / "workflows").mkdir()

        def run(*argv):
            monkeypatch.setattr(
                "sys.argv",
                ["dw-serve", "--workflow-dir", str(tmp_path / "workflows"), *argv],
            )
            serve_module.main()
            return workflows_are_trusted()

        return run

    def test_serve_without_the_flag_overrides_an_inherited_1(self, serve, monkeypatch):
        monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "1")
        assert serve() is False

    def test_serve_with_the_flag_trusts(self, serve, monkeypatch):
        monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "0")
        assert serve("--trust-workflows") is True

    def test_validate_cli_without_the_flag_is_untrusted(self, monkeypatch, tmp_path):
        """dw.validate is how an operator vets a file before running it -
        it must vet it under the posture the run will have."""
        import dw.validate as validate_module

        workflow = tmp_path / "w.json"
        workflow.write_text('{"id": "w", "steps": []}')
        monkeypatch.setenv(TRUST_WORKFLOWS_ENV_VAR, "1")
        monkeypatch.setattr("sys.argv", ["dw-validate", str(workflow)])
        try:
            validate_module.main()
        except SystemExit:
            pass
        assert workflows_are_trusted() is False
