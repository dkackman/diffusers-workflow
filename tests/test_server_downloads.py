"""Download endpoints for gallery outputs, workflows, and prompts: each
serves the artifact with Content-Disposition: attachment rather than an
inline view. Also guards the routing-order gotcha - the download route
must be registered before the existing plain GET route or FastAPI's
greedy {name:path} matching on the plain route swallows it."""

import json

# `server` is imported for its fixture: these tests need exactly the one
# tests/test_server.py already defines (its extra Basic.json seed file is
# harmless here), and a second copy would drift from it
from tests.test_server import server, success_script, valid_workflow  # noqa: F401


def test_download_output_sets_content_disposition_attachment(server, tmp_path):
    with server(success_script) as client:
        outputs = tmp_path / "outputs"
        (outputs / "run-step.0-0.0.png").write_bytes(b"fake-png-bytes")

        response = client.get("/api/gallery/run-step.0-0.0.png/download")

        assert response.status_code == 200
        assert response.content == b"fake-png-bytes"
        assert "attachment" in response.headers["content-disposition"]
        assert "run-step.0-0.0.png" in response.headers["content-disposition"]


def test_download_workflow_sets_content_disposition_attachment(server, tmp_path):
    with server(success_script) as client:
        workflow = valid_workflow("demo")
        response = client.put("/api/workflows/demo", json={"workflow": workflow})
        assert response.status_code == 200

        response = client.get("/api/workflows/demo/download")

        assert response.status_code == 200
        assert json.loads(response.content)["id"] == "demo"
        assert "attachment" in response.headers["content-disposition"]

        # the plain (non-download) GET route must still work - regression
        # guard for the routing-order gotcha
        plain = client.get("/api/workflows/demo")
        assert plain.status_code == 200
        assert plain.json()["id"] == "demo"


def test_download_prompt_sets_content_disposition_attachment(server, tmp_path):
    with server(success_script) as client:
        response = client.put(
            "/api/prompts/greeting", json={"prompt": {"text": "hello"}}
        )
        assert response.status_code == 200

        response = client.get("/api/prompts/greeting/download")

        assert response.status_code == 200
        assert json.loads(response.content)["text"] == "hello"
        assert "attachment" in response.headers["content-disposition"]

        plain = client.get("/api/prompts/greeting")
        assert plain.status_code == 200
