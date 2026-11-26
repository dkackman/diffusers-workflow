"""A pre-quantized SDNQ folder loaded on MPS stops using quantized matmul (#1744).

Real sdnq throughout: a tiny model is quantized with the flag on, saved, and
loaded back through load_component. Only the device type is stubbed.
"""

import pytest
import torch

sdnq = pytest.importorskip("sdnq")

from diffusers import PriorTransformer  # noqa: E402

from dw.events import RunContext, activate_context, deactivate_context  # noqa: E402
from dw.pipeline_processors import config_objects  # noqa: E402
from dw.pipeline_processors.components import load_component  # noqa: E402


def matmul_flags(model):
    return [
        m.sdnq_dequantizer.use_quantized_matmul
        for m in model.modules()
        if hasattr(m, "sdnq_dequantizer")
    ]


def forward(model):
    generator = torch.Generator().manual_seed(0)
    return model(
        hidden_states=torch.randn(1, 32, generator=generator),
        timestep=1,
        proj_embedding=torch.randn(1, 32, generator=generator),
        encoder_hidden_states=torch.randn(1, 77, 32, generator=generator),
    ).predicted_image_embedding


@pytest.fixture(scope="module")
def saved_folder(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("sdnq") / "transformer")
    torch.manual_seed(0)
    model = PriorTransformer(
        num_attention_heads=2, attention_head_dim=32, num_layers=1, embedding_dim=32
    ).eval()
    sdnq.sdnq_post_load_quant(model, weights_dtype="int8", use_quantized_matmul=True)
    sdnq.save_sdnq_model(model, path)
    return path


def load(path, monkeypatch, device_type):
    monkeypatch.setattr(config_objects, "get_device_type", lambda *a: device_type)
    events = []
    token = activate_context(RunContext(on_event=events.append))
    try:
        model = load_component(
            "transformer",
            {"component_type": PriorTransformer},
            {"model_name": path, "torch_dtype": torch.float32},
            "cpu",
        )
    finally:
        deactivate_context(token)
    warnings = [e for e in events if e.get("event") == "warning"]
    # Every one is a device correction the job reports (#1727)
    assert all(e.get("kind") == "device_adjusted" for e in warnings)
    return model, [e["message"] for e in warnings]


def test_a_folder_saved_with_matmul_loads_without_it_on_mps(saved_folder, monkeypatch):
    kept, kept_warnings = load(saved_folder, monkeypatch, "cuda")
    assert any(matmul_flags(kept))
    assert not kept_warnings

    switched, warnings = load(saved_folder, monkeypatch, "mps")

    assert matmul_flags(switched) and not any(matmul_flags(switched))
    assert switched.quantization_config.use_quantized_matmul is False
    assert len(warnings) == 1
    assert "use_quantized_matmul" in warnings[0] and "transformer" in warnings[0]
    with torch.no_grad():
        torch.testing.assert_close(
            forward(switched), forward(kept), rtol=0.05, atol=0.05
        )


def test_a_second_pass_changes_and_says_nothing(saved_folder, monkeypatch):
    model, _ = load(saved_folder, monkeypatch, "mps")
    events = []
    token = activate_context(RunContext(on_event=events.append))
    try:
        config_objects.portable_loaded_components("transformer", model)
    finally:
        deactivate_context(token)
    assert not events
