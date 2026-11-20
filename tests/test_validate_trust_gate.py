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


@pytest.mark.parametrize(
    "entry",
    [
        {"weight_name": "a/b.pt"},
        {"weight_name": "A.SAFETENSORS"},
        {"weight_name": "../b.safetensors"},
        {"weight_name": "/etc/b.safetensors"},
        {"weight_name": ["b.pt"]},
        {"subfolder": "../.."},
        {"subfolder": "/etc"},
    ],
)
def test_the_run_gate_checks_a_lora_file_itself(monkeypatch, entry):
    """Not only at validate: the worker's gate refuses the value, and matches
    the ending as diffusers does, where '.SAFETENSORS' is read as a pickle."""
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
    workflow = _lora_workflow(**entry)
    with pytest.raises(UntrustedWorkflowError, match=r"loras\[0\]"):
        require_trusted_block(workflow["steps"][0]["pipeline"], "pipeline")


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


# --- load_components and the keywords that redirect a load (0.21.0 review) --


def _modular(load_components=None, **from_pretrained):
    workflow = _workflow(**from_pretrained)
    if load_components is not None:
        workflow["steps"][0]["pipeline"]["configuration"]["load_components"] = (
            load_components
        )
    return workflow


@pytest.mark.parametrize(
    "workflow, key",
    [
        (_modular({"trust_remote_code": True}), "trust_remote_code"),
        (_modular({"trust_remote_code": {"default": True}}), "trust_remote_code"),
        (_modular({"custom_pipeline": "some/repo"}), "custom_pipeline"),
        (_modular({"cache_dir": "/etc/elsewhere"}), "cache_dir"),
        (_modular({"proxies": {"https": "http://127.0.0.1:9"}}), "proxies"),
        (_modular(cache_dir="../../elsewhere"), "cache_dir"),
        (_modular(local_dir="/etc/elsewhere"), "local_dir"),
        (_modular(proxies={"https": "http://127.0.0.1:9"}), "proxies"),
        (_modular(mirror="http://127.0.0.1"), "mirror"),
    ],
)
def test_untrusted_refuses_a_loader_keyword_in_either_block(
    gallery_client, monkeypatch, workflow, key
):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
    body = _validate(gallery_client, workflow)
    assert body["valid"] is False
    messages = [e["message"] for e in body["errors"] if f"'{key}'" in e["message"]]
    assert messages and "--trust-workflows" in messages[0]
    with pytest.raises(UntrustedWorkflowError) as run:
        require_trusted_block(workflow["steps"][0]["pipeline"], "pipeline")
    assert str(run.value) == messages[0]


def test_trusted_keeps_the_loader_keywords(monkeypatch):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "1")
    workflow = _modular({"trust_remote_code": True}, cache_dir="/models")
    require_trusted_block(workflow["steps"][0]["pipeline"], "pipeline")


@pytest.mark.parametrize(
    "override",
    ["/etc/ssl", {"transformer": "/etc/ssl"}, "http://127.0.0.1/x", "../../etc"],
)
def test_a_load_components_repo_override_is_a_model_name(
    gallery_client, monkeypatch, override
):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
    workflow = _modular({"pretrained_model_name_or_path": override})
    body = _validate(gallery_client, workflow)
    paths = [e["path"] for e in body["errors"]]
    assert any("pretrained_model_name_or_path" in (p or "") for p in paths), body


@pytest.mark.parametrize("key", ["sam_model_name", "sam3_model_name"])
@pytest.mark.parametrize("name", ["/etc/ssl", "../../../etc", "http://127.0.0.1:1/x"])
def test_a_tasks_second_model_name_is_a_model_name(
    gallery_client, monkeypatch, key, name
):
    monkeypatch.setenv("DW_TRUST_WORKFLOWS", "0")
    workflow = {
        "id": "t",
        "steps": [
            {
                "name": "people",
                "task": {
                    "command": "track_people",
                    "arguments": {"video": "asset:take.mp4", key: name},
                },
            }
        ],
    }
    body = _validate(gallery_client, workflow)
    assert any(key in (e["path"] or "") for e in body["errors"]), body
