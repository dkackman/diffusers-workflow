"""validate_workflow runs the gate the worker runs (#1410): an untrusted
server refuses remote-code keys at /api/validate with the run's own message."""

import pytest

from dw.security import UntrustedWorkflowError
from dw.trust import require_trusted_block


def _workflow(**extra):
    return {
        "id": "p",
        "steps": [
            {
                "name": "pe",
                "pipeline": {
                    "configuration": {"component_type": "ModularPipeline"},
                    "from_pretrained_arguments": {
                        "model_name": "MiniMaxAI/MiniMax-H3-Image",
                        **extra,
                    },
                    "arguments": {"prompt": "x"},
                },
            }
        ],
    }


def _validate(client, workflow):
    response = client.post("/api/validate?sizes=false", json={"workflow": workflow})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize(
    "extra", [{"trust_remote_code": True}, {"custom_pipeline": "some/repo"}]
)
def test_untrusted_validate_refuses_each_key(gallery_client, monkeypatch, extra):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
    workflow = _workflow(**extra)
    body = _validate(gallery_client, workflow)
    assert body["valid"] is False
    key = next(iter(extra))
    messages = [e["message"] for e in body["errors"] if key in e["message"]]
    assert messages and "--trust-workflows" in messages[0]
    # the run's own gate says the same thing
    with pytest.raises(UntrustedWorkflowError) as run:
        require_trusted_block(workflow["steps"][0]["pipeline"], "pipeline")
    assert str(run.value) == messages[0]


@pytest.mark.parametrize(
    "extra", [{"trust_remote_code": True}, {"custom_pipeline": "some/repo"}]
)
def test_trusted_validate_passes(gallery_client, monkeypatch, extra):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
    body = _validate(gallery_client, _workflow(**extra))
    assert not any("--trust-workflows" in e["message"] for e in body["errors"]), body[
        "errors"
    ]


def test_component_block_refused(gallery_client, monkeypatch):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
    workflow = _workflow()
    workflow["steps"][0]["pipeline"]["text_encoder"] = {
        "configuration": {"component_type": "Transformer2DModel"},
        "from_pretrained_arguments": {
            "model_name": "x/y",
            "trust_remote_code": True,
        },
    }
    body = _validate(gallery_client, workflow)
    assert any("trust_remote_code" in e["message"] for e in body["errors"]), body[
        "errors"
    ]


def test_error_is_at_the_step_pipeline_path(gallery_client, monkeypatch):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
    body = _validate(gallery_client, _workflow(trust_remote_code=True))
    assert any(e["path"] == "steps[0].pipeline" for e in body["errors"]), body["errors"]


def test_registry_pass_is_the_runs_gate(monkeypatch):
    from dw.trust import pipeline_remote_code_errors

    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
    workflow = _workflow(custom_pipeline="some/repo")
    (error,) = pipeline_remote_code_errors(workflow)
    with pytest.raises(UntrustedWorkflowError) as run:
        require_trusted_block(workflow["steps"][0]["pipeline"], "pipeline")
    assert error["message"] == str(run.value)
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
    assert pipeline_remote_code_errors(workflow) == []


# --- a loras entry's loader keywords (SE-F066, GHSA-jmhv-58g4-fw8c) --------


def _lora_workflow(**entry):
    workflow = _workflow()
    workflow["steps"][0]["pipeline"]["loras"] = [
        {"model_name": "org/lora", "weight_name": "a/b.safetensors", **entry}
    ]
    return workflow


@pytest.mark.parametrize(
    "extra",
    [
        {"use_safetensors": False},
        {"cache_dir": "/tmp/elsewhere"},
        {"proxies": {"https": "http://127.0.0.1:9"}},
        {"force_download": True},
        {"token": "hf_x"},
    ],
)
def test_untrusted_validate_refuses_a_lora_loader_keyword(
    gallery_client, monkeypatch, extra
):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
    workflow = _lora_workflow(**extra)
    body = _validate(gallery_client, workflow)
    assert body["valid"] is False
    key = next(iter(extra))
    messages = [e["message"] for e in body["errors"] if f"'{key}'" in e["message"]]
    assert messages and "loras[0]" in messages[0]
    with pytest.raises(UntrustedWorkflowError) as run:
        require_trusted_block(workflow["steps"][0]["pipeline"], "pipeline")
    assert str(run.value) == messages[0]


def test_untrusted_validate_passes_the_keys_a_lora_entry_takes(
    gallery_client, monkeypatch
):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
    workflow = _lora_workflow(
        adapter_name="style", scale=0.8, alpha=16, subfolder="a", revision="main"
    )
    body = _validate(gallery_client, workflow)
    assert not any("loras[0]" in e["message"] for e in body["errors"]), body["errors"]


def test_trusted_lora_entry_keeps_its_loader_keywords(monkeypatch):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
    workflow = _lora_workflow(use_safetensors=False, cache_dir="/models")
    require_trusted_block(workflow["steps"][0]["pipeline"], "pipeline")


class _RecordingPipeline:
    def __init__(self):
        self.loads = []

    def load_lora_weights(self, model_name, **keywords):
        self.loads.append((model_name, keywords))


@pytest.mark.parametrize("trusted, expected", [("0", True), ("1", None)])
def test_an_untrusted_lora_load_is_safetensors_only(monkeypatch, trusted, expected):
    """With no weight_name diffusers falls back to a repo's .bin pickle
    unless told the load is safetensors."""
    from dw.pipeline_processors import adapters

    monkeypatch.setenv("DW_TRUST_WORKFLOWS", trusted)
    monkeypatch.setattr(adapters, "apply_adapter_settings", lambda *a, **k: None)
    pipeline = _RecordingPipeline()
    adapters.load_loras([{"model_name": "org/lora"}], pipeline)
    ((name, keywords),) = pipeline.loads
    assert name == "org/lora"
    assert keywords.get("use_safetensors") is expected
