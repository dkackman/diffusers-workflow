import pytest

from dw.kernel_availability import kernel_availability_errors, kernel_availability_fault
import dw.kernel_availability as kernel_availability


@pytest.fixture(autouse=True)
def _fresh_probe():
    """Each test resolves its own fake class under the same dotted name, so
    the per-process memo must not carry an answer between them."""
    kernel_availability._fault_for_name.cache_clear()
    yield
    kernel_availability._fault_for_name.cache_clear()


def _step(name, configuration, **extra):
    step = {
        "name": name,
        "pipeline": {"configuration": configuration},
    }
    step.update(extra)
    return step


class _CleanProcessor:
    def __init__(self):
        pass


def _get_kernel_that_fails(repo_id):
    raise FileNotFoundError("no build variant for this torch/CUDA combination")


def _get_kernel_that_works(repo_id):
    return object()


class _KernelBackedProcessor:
    def __init__(self):
        get_kernel = _get_kernel_that_fails
        get_kernel("shi-labs/natten")


class _KernelBackedProcessorThatWorks:
    def __init__(self):
        get_kernel = _get_kernel_that_works
        get_kernel("shi-labs/natten")
        self.constructed = True


class _CountingKernelBackedProcessor:
    constructions = 0

    def __init__(self):
        get_kernel = _get_kernel_that_works
        get_kernel("shi-labs/natten")
        type(self).constructions += 1


class _KernelBackedProcessorThatFailsOnceThenWorks:
    """A stand-in for a transient Hub fetch failure: the first construction
    raises, the second (same process, same name) succeeds."""

    attempts = 0

    def __init__(self):
        type(self).attempts += 1
        get_kernel = (
            _get_kernel_that_fails
            if type(self).attempts == 1
            else _get_kernel_that_works
        )
        get_kernel("shi-labs/natten")


def _resolving(mapping):
    def load_type_from_name(name):
        return mapping[name]

    return load_type_from_name


class TestKernelAvailabilityFault:
    def test_a_processor_with_no_remote_kernel_is_clean(self, monkeypatch):
        monkeypatch.setattr(
            kernel_availability,
            "load_type_from_name",
            _resolving({"pkg.Clean": _CleanProcessor}),
        )
        assert kernel_availability_fault("pkg.Clean") is None

    def test_a_kernel_backed_processor_that_fails_to_construct_is_faulted(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            kernel_availability,
            "load_type_from_name",
            _resolving({"pkg.Natten": _KernelBackedProcessor}),
        )
        fault = kernel_availability_fault("pkg.Natten")
        assert fault is not None
        assert "pkg.Natten" in fault
        assert "no build variant" in fault

    def test_a_kernel_backed_processor_that_constructs_cleanly_is_not_faulted(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            kernel_availability,
            "load_type_from_name",
            _resolving({"pkg.Natten": _KernelBackedProcessorThatWorks}),
        )
        assert kernel_availability_fault("pkg.Natten") is None

    def test_an_unresolvable_name_is_not_this_checks_job(self, monkeypatch):
        def raising(name):
            raise ImportError("no such module")

        monkeypatch.setattr(kernel_availability, "load_type_from_name", raising)
        assert kernel_availability_fault("pkg.DoesNotExist") is None

    def test_a_variable_reference_is_left_alone(self):
        assert kernel_availability_fault("variable:attn_processor") is None

    def test_a_non_string_is_left_alone(self):
        assert kernel_availability_fault(3) is None

    def test_the_probe_runs_once_per_process_for_a_name(self, monkeypatch):
        """Construction fetches a Hub kernel; validate, submit and rerun each
        ask, and a for_each asks once per member. The answer is a
        per-process constant."""
        monkeypatch.setattr(
            kernel_availability,
            "load_type_from_name",
            _resolving({"pkg.Natten": _CountingKernelBackedProcessor}),
        )
        _CountingKernelBackedProcessor.constructions = 0
        for _ in range(3):
            assert kernel_availability_fault("pkg.Natten") is None
        assert _CountingKernelBackedProcessor.constructions == 1

    def test_a_fault_is_not_memoized(self, monkeypatch):
        """A memoized fault would pin every later call to the same string
        even once the transient condition (a Hub fetch) has cleared."""
        monkeypatch.setattr(
            kernel_availability,
            "load_type_from_name",
            _resolving({"pkg.Natten": _KernelBackedProcessorThatFailsOnceThenWorks}),
        )
        _KernelBackedProcessorThatFailsOnceThenWorks.attempts = 0
        first = kernel_availability_fault("pkg.Natten")
        assert first is not None
        assert "pkg.Natten" in first
        assert kernel_availability_fault("pkg.Natten") is None


class TestKernelAvailabilityErrors:
    def test_a_clean_definition_has_none(self, monkeypatch):
        monkeypatch.setattr(
            kernel_availability,
            "load_type_from_name",
            _resolving({"pkg.Clean": _CleanProcessor}),
        )
        definition = {
            "steps": [
                _step(
                    "decode",
                    {
                        "component_type": "SomePipeline",
                        "components": {
                            "diffusion_decoder": {"attn_processor_type": "pkg.Clean"}
                        },
                    },
                )
            ]
        }
        assert kernel_availability_errors(definition) == []

    def test_a_component_level_processor_is_faulted_at_its_path(self, monkeypatch):
        monkeypatch.setattr(
            kernel_availability,
            "load_type_from_name",
            _resolving({"pkg.Natten": _KernelBackedProcessor}),
        )
        definition = {
            "steps": [
                _step(
                    "decode",
                    {
                        "component_type": "SomePipeline",
                        "components": {
                            "diffusion_decoder": {"attn_processor_type": "pkg.Natten"}
                        },
                    },
                )
            ]
        }
        errors = kernel_availability_errors(definition)
        assert len(errors) == 1
        assert errors[0]["path"] == (
            "steps[0].pipeline.configuration.components.diffusion_decoder."
            "attn_processor_type"
        )
        assert "pkg.Natten" in errors[0]["message"]

    def test_a_transformer_level_processor_is_faulted_at_its_path(self, monkeypatch):
        monkeypatch.setattr(
            kernel_availability,
            "load_type_from_name",
            _resolving({"pkg.Natten": _KernelBackedProcessor}),
        )
        definition = {
            "steps": [
                _step(
                    "decode",
                    {
                        "component_type": "SomePipeline",
                        "transformer": {"attn_processor_type": "pkg.Natten"},
                    },
                )
            ]
        }
        errors = kernel_availability_errors(definition)
        assert len(errors) == 1
        assert errors[0]["path"] == (
            "steps[0].pipeline.configuration.transformer.attn_processor_type"
        )

    def test_source_indices_map_back_to_the_written_step(self, monkeypatch):
        monkeypatch.setattr(
            kernel_availability,
            "load_type_from_name",
            _resolving({"pkg.Natten": _KernelBackedProcessor}),
        )
        definition = {
            "steps": [
                _step(
                    "decode@one",
                    {
                        "component_type": "SomePipeline",
                        "components": {
                            "diffusion_decoder": {"attn_processor_type": "pkg.Natten"}
                        },
                    },
                )
            ]
        }
        errors = kernel_availability_errors(definition, source_indices=[0])
        assert len(errors) == 1
        assert errors[0]["path"].startswith("steps[0]")
        assert "decode@one" in errors[0]["message"]

    def test_a_step_with_no_pipeline_is_skipped(self):
        definition = {"steps": [{"name": "a", "task": {"command": "noop"}}]}
        assert kernel_availability_errors(definition) == []
