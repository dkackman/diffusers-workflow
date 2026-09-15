"""Download endpoints for gallery outputs, workflows, and prompts: each
serves the artifact with Content-Disposition: attachment rather than an
inline view. Also guards the routing-order gotcha - the download route
must be registered before the existing plain GET route or FastAPI's
greedy {name:path} matching on the plain route swallows it."""

import json

# `server` is imported for its fixture: these tests need exactly the one
# tests/test_server.py already defines (its extra Basic.json seed file is
# harmless here), and a second copy would drift from it
from tests.test_server import (  # noqa: F401
    asset_server,
    server,
    success_script,
    valid_workflow,
)


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


def test_archive_outputs_zips_every_requested_file(server, tmp_path):
    """Bulk download hands back one zip so a multi-file gallery selection
    is a single browser download rather than N blocked ones."""
    import io
    import zipfile

    with server(success_script) as client:
        outputs = tmp_path / "outputs"
        (outputs / "first.png").write_bytes(b"first-bytes")
        nested = outputs / "demo"
        nested.mkdir(exist_ok=True)
        (nested / "second.png").write_bytes(b"second-bytes")

        response = client.post(
            "/api/gallery/archive", json={"names": ["first.png", "demo/second.png"]}
        )

        assert response.status_code == 200
        assert "attachment" in response.headers["content-disposition"]
        assert ".zip" in response.headers["content-disposition"]
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            # the gallery-relative name is the entry name, so the folders a
            # workflow wrote into survive into the user's download
            assert sorted(archive.namelist()) == ["demo/second.png", "first.png"]
            assert archive.read("first.png") == b"first-bytes"
            assert archive.read("demo/second.png") == b"second-bytes"


def test_archive_outputs_rejects_a_name_outside_the_output_directory(server, tmp_path):
    with server(success_script) as client:
        (tmp_path / "secret.txt").write_bytes(b"not yours")

        response = client.post(
            "/api/gallery/archive", json={"names": ["../secret.txt"]}
        )

        assert response.status_code == 404


def test_archive_outputs_rejects_an_unknown_name(server, tmp_path):
    with server(success_script) as client:
        response = client.post("/api/gallery/archive", json={"names": ["nope.png"]})

        assert response.status_code == 404


def test_archive_outputs_rejects_an_empty_selection(server, tmp_path):
    with server(success_script) as client:
        response = client.post("/api/gallery/archive", json={"names": []})

        assert response.status_code == 422


def test_archive_outputs_rejects_more_names_than_the_cap(server, tmp_path):
    with server(success_script) as client:
        response = client.post(
            "/api/gallery/archive", json={"names": [f"f{i}.png" for i in range(1001)]}
        )

        assert response.status_code == 422


def test_archive_assets_zips_every_requested_file(asset_server, tmp_path):
    """The asset grid's bulk download, the gallery's own: a selection of
    inputs comes back as one zip rather than N downloads the browser
    throttles."""
    import io
    import zipfile

    with asset_server(success_script) as client:
        assets = tmp_path / "assets"
        (assets / "iris.png").write_bytes(b"iris-bytes")
        nested = assets / "cast"
        nested.mkdir(exist_ok=True)
        (nested / "priya.jpg").write_bytes(b"priya-bytes")

        response = client.post(
            "/api/assets/archive", json={"names": ["iris.png", "cast/priya.jpg"]}
        )

        assert response.status_code == 200
        assert "attachment" in response.headers["content-disposition"]
        assert ".zip" in response.headers["content-disposition"]
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            # the library-relative name is the entry name, so a folder in the
            # library survives into the download and the reference a workflow
            # carries still reads off the file you unpacked
            assert sorted(archive.namelist()) == ["cast/priya.jpg", "iris.png"]
            assert archive.read("iris.png") == b"iris-bytes"
            assert archive.read("cast/priya.jpg") == b"priya-bytes"


def test_archive_assets_rejects_a_name_outside_the_library(asset_server, tmp_path):
    with asset_server(success_script) as client:
        (tmp_path / "secret.txt").write_bytes(b"not yours")

        response = client.post("/api/assets/archive", json={"names": ["../secret.txt"]})

        assert response.status_code == 404


def test_archive_assets_rejects_an_unknown_name(asset_server, tmp_path):
    with asset_server(success_script) as client:
        response = client.post("/api/assets/archive", json={"names": ["nope.png"]})

        assert response.status_code == 404


def test_archive_assets_rejects_an_empty_selection(asset_server, tmp_path):
    with asset_server(success_script) as client:
        response = client.post("/api/assets/archive", json={"names": []})

        assert response.status_code == 422


def test_archive_assets_rejects_more_names_than_the_cap(asset_server, tmp_path):
    with asset_server(success_script) as client:
        response = client.post(
            "/api/assets/archive", json={"names": [f"f{i}.png" for i in range(1001)]}
        )

        assert response.status_code == 422


def test_archive_stores_already_compressed_media_rather_than_deflating_it(
    asset_server, tmp_path
):
    """Every asset extension but .bmp and .wav is an already-compressed
    container, so deflating one buys nothing and costs a full CPU pass the
    caller waits on - the response only starts once the temp file is
    written. Formats that do compress keep deflate."""
    import io
    import zipfile

    with asset_server(success_script) as client:
        assets = tmp_path / "assets"
        (assets / "clip.mp4").write_bytes(b"\x00\x01" * 4096)
        (assets / "tone.wav").write_bytes(b"\x00\x01" * 4096)

        response = client.post(
            "/api/assets/archive", json={"names": ["clip.mp4", "tone.wav"]}
        )

        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            assert archive.getinfo("clip.mp4").compress_type == zipfile.ZIP_STORED
            assert archive.getinfo("tone.wav").compress_type == zipfile.ZIP_DEFLATED
