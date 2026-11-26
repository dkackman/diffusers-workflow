"""Every run-time device correction reaches the job's warnings (#1727).

`docs/ARCHITECTURE.md` *Run-time warnings*: a warning found at run time is an
event, so a caller over MCP or the API sees it - not just the server log. The
MPS/CPU corrections the engine makes on its own (offload rewritten, compile
skipped, RoPE precision lowered, ...) were log-only. Each test here drives one
site with the device type patched and reads the warning off a real server Job,
the object `get_job` reports from. The existing caplog tests next to each site
keep holding: emit_warning logs as well.
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from dw.events import RunContext, activate_context, deactivate_context
from dw.pipeline_processors import components, config_objects, placement
from dw.server.job_record import Job


def job_warnings(action):
    """Run action inside a job and return (job.warnings, warning events)."""
    job = Job({"workflow_name": "device-adjusted"})
    events = []

    def sink(event):
        events.append(event)
        job.add_event(event)

    token = activate_context(RunContext(on_event=sink))
    try:
        action()
    finally:
        deactivate_context(token)
    return job.warnings, [e for e in events if e.get("event") == "warning"]


def only_device_adjusted(events):
    assert events, "expected a device_adjusted warning"
    assert all(e["kind"] == "device_adjusted" for e in events)
    return events


# --- RoPE precision (placement.apply_mps_rope_precision) ---------------------


class Rope(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.double_precision = True


def rope_pipeline():
    return SimpleNamespace(components={"transformer": Rope()})


def test_rope_precision_is_a_job_warning_on_mps(caplog):
    with caplog.at_level(logging.WARNING, logger="dw"):
        warnings, events = job_warnings(
            lambda: placement.apply_mps_rope_precision(rope_pipeline(), "mps")
        )

    (event,) = only_device_adjusted(events)
    assert event["requested"] == "float64" and event["applied"] == "float32"
    assert any("float32" in w and "transformer" in w for w in warnings)
    assert "float32" in caplog.text


def test_rope_precision_is_quiet_on_cuda():
    warnings, events = job_warnings(
        lambda: placement.apply_mps_rope_precision(rope_pipeline(), "cuda")
    )
    assert warnings == [] and events == []


# --- attention slicing (placement.apply_attention_slicing) -------------------


def test_automatic_attention_slicing_is_a_job_warning_on_mps():
    pipeline = MagicMock()
    warnings, events = job_warnings(
        lambda: placement.apply_attention_slicing(pipeline, {}, "mps")
    )

    pipeline.enable_attention_slicing.assert_called_once_with()
    (event,) = only_device_adjusted(events)
    assert event["setting"] == "attention_slicing"
    assert any("disable_attention_slicing" in w for w in warnings)


def test_asked_for_attention_slicing_is_not_a_correction():
    pipeline = MagicMock()
    warnings, events = job_warnings(
        lambda: placement.apply_attention_slicing(
            pipeline, {"enable_attention_slicing": True}, "mps"
        )
    )

    pipeline.enable_attention_slicing.assert_called_once_with()
    assert warnings == [] and events == []


def test_no_attention_slicing_on_cuda():
    pipeline = MagicMock()
    warnings, events = job_warnings(
        lambda: placement.apply_attention_slicing(pipeline, {}, "cuda")
    )

    pipeline.enable_attention_slicing.assert_not_called()
    assert warnings == []


def test_opted_out_attention_slicing_on_mps_is_quiet():
    pipeline = MagicMock()
    warnings, _ = job_warnings(
        lambda: placement.apply_attention_slicing(
            pipeline, {"disable_attention_slicing": True}, "mps"
        )
    )

    pipeline.enable_attention_slicing.assert_not_called()
    assert warnings == []


# --- components_manager auto offload (placement.create_components_manager) ---


def test_auto_offload_skip_is_a_job_warning_on_mps():
    configuration = {"components_manager": {"enable_auto_cpu_offload": True}}
    with patch("diffusers.ComponentsManager") as manager:
        warnings, events = job_warnings(
            lambda: placement.create_components_manager(configuration, "mps")
        )

    manager.return_value.enable_auto_cpu_offload.assert_not_called()
    (event,) = only_device_adjusted(events)
    assert event["setting"] == "enable_auto_cpu_offload"
    assert any("auto CPU offload" in w for w in warnings)


def test_auto_offload_on_cuda_is_quiet():
    configuration = {"components_manager": {"enable_auto_cpu_offload": True}}
    with patch("diffusers.ComponentsManager"):
        warnings, _ = job_warnings(
            lambda: placement.create_components_manager(configuration, "cuda")
        )
    assert warnings == []


# --- offload in place_component ----------------------------------------------


def place(configuration, device):
    component = MagicMock()
    with patch.object(placement, "resolve_device", lambda d: d):
        warnings, events = job_warnings(
            lambda: placement.place_component(
                component, "pipeline", configuration, device
            )
        )
    return component, warnings, events


def test_offload_on_cpu_is_a_job_warning():
    component, warnings, events = place({"offload": "model"}, "cpu")

    component.enable_model_cpu_offload.assert_not_called()
    (event,) = only_device_adjusted(events)
    assert event["requested"] == "model" and event["applied"] is None
    assert any("not an accelerator" in w for w in warnings)


def test_sequential_offload_on_mps_is_a_job_warning():
    component, warnings, events = place(
        {"offload": "sequential", "exclude_from_cpu_offload": ["vae"]}, "mps"
    )

    component.enable_model_cpu_offload.assert_called_once_with(device="mps")
    component.enable_sequential_cpu_offload.assert_not_called()
    (event,) = only_device_adjusted(events)
    assert event["requested"] == "sequential" and event["applied"] == "model"
    # what was dropped is named, so the caller knows the exclusion is gone
    assert any("exclude_from_cpu_offload" in w and "vae" in w for w in warnings)


def test_sequential_offload_on_cuda_is_quiet():
    component, warnings, _ = place({"offload": "sequential"}, "cuda")

    component.enable_sequential_cpu_offload.assert_called_once_with(device="cuda")
    assert warnings == []


# --- compile (components.apply_compile) --------------------------------------


def test_compile_skip_is_a_job_warning_on_mps():
    component = MagicMock()
    warnings, events = job_warnings(
        lambda: components.apply_compile(component, "transformer", {}, "mps")
    )

    component.compile.assert_not_called()
    (event,) = only_device_adjusted(events)
    assert event["setting"] == "compile" and event["component"] == "transformer"
    assert any("torch.compile" in w and "transformer" in w for w in warnings)


# --- float16 on MPS (components._warn_float16_on_mps) - a warning, no rewrite --


def test_float16_on_mps_is_a_job_warning_that_claims_no_rewrite():
    arguments = {"torch_dtype": torch.float16}
    warnings, events = job_warnings(
        lambda: components._warn_float16_on_mps("unet", arguments, "mps")
    )

    (event,) = only_device_adjusted(events)
    # The dtype is left as asked: what ran is what was asked for
    assert event["requested"] == event["applied"] == "float16"
    assert arguments["torch_dtype"] is torch.float16
    (warning,) = warnings
    assert "unet loads in float16" in warning
    for claim in ("instead", "switch", "turning", "using bfloat16", "converted"):
        assert claim not in warning.lower()


def test_float16_on_cuda_is_quiet():
    warnings, _ = job_warnings(
        lambda: components._warn_float16_on_mps(
            "unet", {"torch_dtype": torch.float16}, "cuda"
        )
    )
    assert warnings == []


# --- quantized matmul (config_objects.portable_quantization_arguments) -------


@pytest.mark.parametrize("key", config_objects.QUANTIZED_MATMUL_ARGUMENTS)
def test_quantized_matmul_off_is_a_job_warning_on_mps(monkeypatch, key):
    monkeypatch.setattr(config_objects, "get_device_type", lambda *a: "mps")
    adapted = {}

    def adapt():
        adapted.update(config_objects.portable_quantization_arguments({key: True}))

    warnings, events = job_warnings(adapt)

    assert adapted[key] is False
    (event,) = only_device_adjusted(events)
    assert event["setting"] == key
    assert any(f"'{key}'" in w for w in warnings)


def test_quantized_matmul_on_cuda_is_quiet(monkeypatch):
    monkeypatch.setattr(config_objects, "get_device_type", lambda *a: "cuda")
    warnings, _ = job_warnings(
        lambda: config_objects.portable_quantization_arguments(
            {"use_quantized_matmul": True}
        )
    )
    assert warnings == []


# --- a repeated correction is one line per step ------------------------------


def test_a_repeated_correction_is_one_job_warning():
    def twice():
        for _ in range(2):
            components.apply_compile(MagicMock(), "transformer", {}, "mps")

    warnings, events = job_warnings(twice)

    assert len(events) == 2
    assert len(warnings) == 1
