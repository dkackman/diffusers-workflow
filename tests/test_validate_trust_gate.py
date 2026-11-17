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
