"""The UI's response contract: the routes the UI reads declare response
models, and the OpenAPI document generated from them is committed where the
UI generates its types from (ui/src/lib/generated/). See docs/ARCHITECTURE.md,
"The UI's response contract"."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.test_server import (  # noqa: F401
    server,
    success_script,
)

REPO = Path(__file__).resolve().parent.parent
DUMP = REPO / "scripts" / "dump_openapi.py"


def _model_in(mode: str):
    """ApiModel as a fresh interpreter builds it with the switch set or not."""
    env = {k: v for k, v in os.environ.items() if k != "DW_STRICT_RESPONSES"}
    if mode == "strict":
        env["DW_STRICT_RESPONSES"] = "1"
    code = "from dw.server.api_models import ApiModel; print(ApiModel.model_config['extra'])"
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_the_suite_runs_strict():
    from dw.server.api_models import STRICT

    assert STRICT, "tests/conftest.py sets DW_STRICT_RESPONSES before dw is imported"


def test_runtime_is_lenient_and_strict_mode_forbids():
    assert _model_in("runtime") == "allow"
    assert _model_in("strict") == "forbid"


def test_a_lenient_model_passes_an_undeclared_key_through():
    # What production does with a key a model has not declared yet: send it
    from pydantic import ConfigDict

    from dw.server.api_models import ApiModel

    class Probe(ApiModel):
        model_config = ConfigDict(extra="allow")
        x: int

    app = FastAPI()

    @app.get("/p", response_model=Probe, response_model_exclude_unset=True)
    def p():
        return {"x": 1, "later": {"y": 2}}

    assert TestClient(app).get("/p").json() == {"x": 1, "later": {"y": 2}}


def _dump(extra_env):
    env = {**os.environ, **extra_env}
    return subprocess.run(
        [sys.executable, str(DUMP), "--stdout"],
        env=env,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_the_document_does_not_depend_on_the_machine():
    assert _dump({"DW_DEVICE": "cpu"}) == _dump({"DW_DEVICE": "mps"})


def test_the_document_carries_no_release_version():
    assert json.loads(_dump({}))["info"]["version"] == "0"


def test_the_committed_document_is_current():
    committed = (REPO / "ui" / "src" / "lib" / "generated" / "openapi.json").read_text()
    assert committed == _dump({}), (
        "the server's response contract changed: run `python scripts/dump_openapi.py`, "
        "then `cd ui && npm run gen:api`, and commit both"
    )


UI_READ_ROUTES = [
    ("get", "/api/health"),
    ("get", "/api/server"),
    ("get", "/api/memory"),
    ("post", "/api/memory/clear"),
    ("get", "/api/models"),
    ("post", "/api/models/download"),
    ("get", "/api/models/downloads"),
    ("post", "/api/models/downloads/{download_id}/cancel"),
    ("delete", "/api/models"),
    ("get", "/api/system/diffusers"),
    ("post", "/api/system/diffusers/update"),
]


def _success_schema(document, method, path):
    responses = document["paths"][path][method]["responses"]
    ok = next(code for code in responses if code.startswith("2"))
    return (
        responses[ok].get("content", {}).get("application/json", {}).get("schema", {})
    )


@pytest.fixture(scope="module")
def document():
    return json.loads(_dump({}))


@pytest.mark.parametrize("method,path", UI_READ_ROUTES)
def test_a_route_the_ui_reads_declares_its_response(document, method, path):
    schema = _success_schema(document, method, path)
    assert "$ref" in schema or schema.get("type") == "array", (
        f"{method.upper()} {path} declares no response model: the UI's type for it "
        "is not generated from the server"
    )


# The payload does not change: keys, absence and int-ness are what they were
# before the routes declared models (dw_mcp and scripts read them too)


def test_memory_with_no_worker_keeps_its_keys(server):
    with server(success_script) as client:
        body = client.get("/api/memory").json()
    assert body == {
        "live": False,
        "info": None,
        "stale": False,
        "reason": "worker_stopped",
        "age_seconds": None,
    }


def test_health_sends_exactly_its_keys(server):
    with server(success_script) as client:
        body = client.get("/api/health").json()
    assert set(body) == {
        "status",
        "version",
        "worker_alive",
        "current_job",
        "queued",
        "hostname",
        "device",
        "mcp",
    }
    assert type(body["queued"]) is int


def test_an_empty_model_cache_counts_bytes_in_integers(server, tmp_path, monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    with server(success_script) as client:
        body = client.get("/api/models").json()
    assert type(body["size_on_disk"]) is int
    assert body["repos"] == [] and body["warnings"] == []
