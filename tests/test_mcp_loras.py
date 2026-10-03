"""LoRA catalog tools: what each sends to the server."""

import json

import httpx

from dw_mcp import loras
from dw_mcp.client import DwClient


def recording(body=None):
    seen = []

    def handler(request):
        raw = request.read()
        seen.append((request.method, request.url.path, dict(request.url.params), json.loads(raw) if raw else None))
        return httpx.Response(200, json=body if body is not None else {})

    return DwClient(transport=httpx.MockTransport(handler)), seen


def test_list_loras_sends_only_the_filters_given():
    client, seen = recording({"loras": []})
    loras.list_loras(client, model="models/qwen-image-2.1")
    assert seen == [("GET", "/api/loras", {"model": "models/qwen-image-2.1"}, None)]


def test_save_lora_puts_the_entry():
    client, seen = recording({"name": "qwen-image/voxel"})
    loras.save_lora(client, "qwen-image/voxel", {"model_name": "a/b"})
    assert seen[0][:2] == ("PUT", "/api/loras/qwen-image/voxel")
    assert seen[0][3] == {"entry": {"model_name": "a/b"}}


def test_save_lora_accepts_a_json_string():
    client, seen = recording({"name": "x"})
    loras.save_lora(client, "x", '{"model_name": "a/b"}')
    assert seen[0][3] == {"entry": {"model_name": "a/b"}}


def test_recommend_loras_sends_model_query_and_limit():
    client, seen = recording({"catalog": [], "hub": []})
    loras.recommend_loras(client, "Qwen/Qwen-Image-2.1", "voxel style", limit=5)
    assert seen == [("GET", "/api/loras/recommend", {"model": "Qwen/Qwen-Image-2.1", "query": "voxel style", "limit": "5"}, None)]
