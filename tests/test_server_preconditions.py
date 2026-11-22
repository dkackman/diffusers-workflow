# ruff: noqa: F811
"""ETags and opt-in conditional writes on workflow and prompt saves.

A request without If-Match / If-None-Match behaves as it always did; with
one, a save that would silently replace something refuses with a 409 whose
code is `exists` or `stale_write`."""

import hashlib

from tests.test_server import (  # noqa: F401  (fixtures)
    examples_server,
    server,
    success_script,
    valid_workflow,
)


def etag_of(data: bytes) -> str:
    return '"' + hashlib.sha256(data).hexdigest()[:32] + '"'


def put_wf(client, name, wf=None, **headers):
    return client.put(
        f"/api/workflows/{name}",
        json={"workflow": wf or valid_workflow(name)},
        headers=headers,
    )


def put_prompt(client, name, text="hello", **headers):
    return client.put(
        f"/api/prompts/{name}", json={"prompt": {"text": text}}, headers=headers
    )


def test_get_workflow_sends_an_etag_that_matches_its_bytes(server, tmp_path):
    with server(success_script) as client:
        response = client.get("/api/workflows/Basic")
        data = (tmp_path / "workflows" / "Basic.json").read_bytes()
        assert response.headers["etag"] == etag_of(data)


def test_put_without_preconditions_still_overwrites(server, tmp_path):
    with server(success_script) as client:
        assert put_wf(client, "Basic").status_code == 200
        assert put_wf(client, "Basic").status_code == 200


def test_if_none_match_star_refuses_an_existing_workflow(server, tmp_path):
    with server(success_script) as client:
        path = tmp_path / "workflows" / "Basic.json"
        before = path.read_bytes()
        response = put_wf(client, "Basic", **{"If-None-Match": "*"})
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "exists"
        assert path.read_bytes() == before
        assert put_wf(client, "Fresh", **{"If-None-Match": "*"}).status_code == 200


def test_if_none_match_ignores_a_read_only_source_of_the_same_name(
    examples_server, tmp_path
):
    with examples_server(success_script) as client:
        response = put_wf(
            client, "ltx2/Gyre", valid_workflow("gyre"), **{"If-None-Match": "*"}
        )
        assert response.status_code == 200
        assert (tmp_path / "workflows" / "ltx2" / "Gyre.json").exists()


def test_if_match_refuses_after_another_write(server):
    with server(success_script) as client:
        a = client.get("/api/workflows/Basic").headers["etag"]
        changed = valid_workflow("basic")
        changed["description"] = "another writer"
        b = put_wf(client, "Basic", changed).headers["etag"]
        assert a != b
        response = put_wf(client, "Basic", **{"If-Match": a})
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["code"] == "stale_write"
        assert detail["etag"] == b
        assert put_wf(client, "Basic", **{"If-Match": b}).status_code == 200


def test_if_match_on_a_deleted_file_is_stale_write_with_no_etag(server, tmp_path):
    with server(success_script) as client:
        a = client.get("/api/workflows/Basic").headers["etag"]
        (tmp_path / "workflows" / "Basic.json").unlink()
        response = put_wf(client, "Basic", **{"If-Match": a})
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["code"] == "stale_write"
        assert detail["etag"] is None


def test_put_and_patch_send_the_written_etag(server, tmp_path):
    with server(success_script) as client:
        path = tmp_path / "workflows" / "Basic.json"
        put = put_wf(client, "Basic")
        assert put.headers["etag"] == etag_of(path.read_bytes())
        patch = client.patch("/api/workflows/Basic", json={"description": "patched"})
        assert patch.status_code == 200
        assert patch.headers["etag"] == etag_of(path.read_bytes())
        assert patch.headers["etag"] != put.headers["etag"]


def test_patch_honours_if_match(server):
    with server(success_script) as client:
        stale = client.get("/api/workflows/Basic").headers["etag"]
        put_wf(client, "Basic", {**valid_workflow("basic"), "description": "x"})
        response = client.patch(
            "/api/workflows/Basic",
            json={"description": "y"},
            headers={"If-Match": stale},
        )
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "stale_write"


def test_prompt_get_sends_an_etag_that_matches_its_bytes(server, tmp_path):
    with server(success_script) as client:
        put_prompt(client, "p1")
        response = client.get("/api/prompts/p1")
        data = (tmp_path / "prompts" / "p1.json").read_bytes()
        assert response.headers["etag"] == etag_of(data)


def test_prompt_put_without_preconditions_still_overwrites(server):
    with server(success_script) as client:
        assert put_prompt(client, "p1").status_code == 200
        assert put_prompt(client, "p1", "again").status_code == 200


def test_prompt_if_none_match_star_refuses_an_existing_prompt(server, tmp_path):
    with server(success_script) as client:
        put_prompt(client, "p1")
        path = tmp_path / "prompts" / "p1.json"
        before = path.read_bytes()
        response = put_prompt(client, "p1", "other", **{"If-None-Match": "*"})
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "exists"
        assert path.read_bytes() == before


def test_prompt_if_match_refuses_after_another_write(server):
    with server(success_script) as client:
        put_prompt(client, "p1")
        a = client.get("/api/prompts/p1").headers["etag"]
        b = put_prompt(client, "p1", "changed").headers["etag"]
        response = put_prompt(client, "p1", "mine", **{"If-Match": a})
        assert response.status_code == 409
        detail = response.json()["detail"]
        assert detail["code"] == "stale_write"
        assert detail["etag"] == b


def test_prompt_if_match_on_a_deleted_file_is_stale_write_with_no_etag(
    server, tmp_path
):
    with server(success_script) as client:
        put_prompt(client, "p1")
        a = client.get("/api/prompts/p1").headers["etag"]
        (tmp_path / "prompts" / "p1.json").unlink()
        response = put_prompt(client, "p1", "mine", **{"If-Match": a})
        assert response.status_code == 409
        assert response.json()["detail"]["etag"] is None


def test_prompt_put_sends_the_written_etag(server, tmp_path):
    with server(success_script) as client:
        response = put_prompt(client, "p1")
        data = (tmp_path / "prompts" / "p1.json").read_bytes()
        assert response.headers["etag"] == etag_of(data)
