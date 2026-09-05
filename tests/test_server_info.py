"""GET /api/server - what a machine on the other end of the network needs
to know about this one, and what it must never be told (the token)."""

import json

import pytest
from fastapi.testclient import TestClient

from dw.server.app import create_app
from dw.server.jobs import JobManager
from dw.server import netinfo

from tests.test_server import ScriptedWorkerManager, success_script


def make_app(tmp_path, **kwargs):
    workflow_dir = tmp_path / "workflows"
    workflow_dir.mkdir(exist_ok=True)
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
    )
    return create_app(
        workflow_dir=str(workflow_dir),
        output_dir=str(tmp_path / "outputs"),
        prompt_dir=str(prompt_dir),
        job_manager=manager,
        **kwargs,
    )


def client(tmp_path, **kwargs):
    return TestClient(make_app(tmp_path, **kwargs), base_url="http://localhost")


def test_payload_shape(tmp_path):
    with client(tmp_path) as c:
        body = c.get("/api/server").json()
    assert isinstance(body["hostname"], str) and body["hostname"]
    assert isinstance(body["version"], str) and body["version"]
    assert isinstance(body["device"], str) and body["device"]
    assert body["bind_host"] == "127.0.0.1"
    assert body["port"] == 8765
    assert body["wildcard_bind"] is False
    assert body["auth_required"] is False
    assert body["mcp"] == {"mounted": False, "path": "/mcp"}
    assert isinstance(body["addresses"], list)
    for entry in body["addresses"]:
        assert set(entry) == {"address", "family", "interface"}
        assert entry["family"] in ("IPv4", "IPv6")
        assert isinstance(entry["address"], str)
        assert entry["interface"] is None or isinstance(entry["interface"], str)
    directories = body["directories"]
    assert directories["workflows"] == str(tmp_path / "workflows")
    assert directories["outputs"] == str(tmp_path / "outputs")
    assert directories["prompts"] == str(tmp_path / "prompts")


def test_wildcard_bind_and_port_reported(tmp_path):
    with client(tmp_path, host="0.0.0.0", port=9000) as c:
        body = c.get("/api/server").json()
    assert body["bind_host"] == "0.0.0.0"
    assert body["wildcard_bind"] is True
    assert body["port"] == 9000


def test_prompt_dir_may_be_absent(tmp_path):
    app = create_app(
        workflow_dir=str(tmp_path / "workflows"),
        output_dir=str(tmp_path / "outputs"),
        prompt_dir=None,
        job_manager=JobManager(
            str(tmp_path / "outputs"),
            worker_manager=ScriptedWorkerManager(success_script),
            history_path=str(tmp_path / "jobs.sqlite"),
        ),
    )
    with TestClient(app, base_url="http://localhost") as c:
        assert c.get("/api/server").json()["directories"]["prompts"] is None


def test_auth_required_and_token_never_disclosed(tmp_path):
    token = "s3cr3t-token-value"
    with client(tmp_path, token=token) as c:
        response = c.get("/api/server", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    body = response.json()
    assert body["auth_required"] is True
    # not the value, not a prefix of it, not a length, not a hash
    serialized = json.dumps(body)
    assert token not in serialized
    assert token not in response.text
    assert token[:6] not in serialized

    # nor a length standing in for it
    def numbers(value):
        if isinstance(value, bool):
            return []
        if isinstance(value, int):
            return [value]
        if isinstance(value, dict):
            return [n for item in value.values() for n in numbers(item)]
        if isinstance(value, list):
            return [n for item in value for n in numbers(item)]
        return []

    assert len(token) not in numbers(body)


def test_requires_the_token_like_every_other_api_route(tmp_path):
    with client(tmp_path, token="abc123") as c:
        assert c.get("/api/server").status_code == 401
        assert (
            c.get("/api/server", headers={"Authorization": "Bearer abc123"}).status_code
            == 200
        )


def test_mcp_mounted_reported(tmp_path):
    pytest.importorskip("mcp", reason="the mcp extra is not installed")
    with client(tmp_path, mcp=True) as c:
        assert c.get("/api/server").json()["mcp"] == {"mounted": True, "path": "/mcp"}


def test_addresses_exclude_loopback_and_link_local(tmp_path):
    with client(tmp_path) as c:
        addresses = c.get("/api/server").json()["addresses"]
    values = [entry["address"] for entry in addresses]
    assert "127.0.0.1" not in values
    assert "::1" not in values
    assert not [v for v in values if v.startswith("169.254.") or v.startswith("fe80:")]
    # IPv4 first
    families = [entry["family"] for entry in addresses]
    assert families == sorted(families, key=lambda f: f != "IPv4")


def test_enumeration_failure_degrades_to_empty_list(tmp_path, monkeypatch):
    def boom():
        raise OSError("no interfaces here")

    monkeypatch.setattr("dw.server.app.local_addresses", boom)
    with client(tmp_path) as c:
        response = c.get("/api/server")
    assert response.status_code == 200
    assert response.json()["addresses"] == []


def test_netinfo_never_raises_when_both_methods_fail(monkeypatch):
    monkeypatch.setattr(netinfo, "_psutil_addresses", lambda: 1 / 0)
    monkeypatch.setattr(netinfo, "_stdlib_addresses", lambda: 1 / 0)
    assert netinfo.local_addresses() == []


def test_netinfo_falls_back_to_stdlib_without_psutil(monkeypatch):
    def no_psutil():
        raise ImportError("no psutil")

    monkeypatch.setattr(netinfo, "_psutil_addresses", no_psutil)
    entries = netinfo.local_addresses()
    assert isinstance(entries, list)
    for entry in entries:
        assert entry["interface"] is None
        assert entry["family"] in ("IPv4", "IPv6")


def test_usable_filters(monkeypatch):
    assert netinfo._usable("127.0.0.1") is None
    assert netinfo._usable("::1") is None
    assert netinfo._usable("169.254.1.2") is None
    assert netinfo._usable("fe80::1%eth0") is None
    assert netinfo._usable("ff:not-an-address") is None
    assert netinfo._usable("") is None
    assert netinfo._usable("192.168.1.50") == ("192.168.1.50", "IPv4")
    assert netinfo._usable("2001:db8::5%eth0") == ("2001:db8::5", "IPv6")
