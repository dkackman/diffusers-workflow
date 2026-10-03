"""The LoRA catalog over the HTTP API: list by base or workflow, save a
promoted trial, and the opt-in Hub search."""

from dw_mcp.client import api_path, coerce_json_object


def list_loras(client, model=None, workflow=None, status=None, tag=None):
    params = {
        key: value
        for key, value in (("model", model), ("workflow", workflow), ("status", status), ("tag", tag))
        if value is not None
    }
    return client.get_json("/api/loras", params=params)


def save_lora(client, name, entry):
    entry = coerce_json_object(entry, "entry")
    return client.put_json(api_path("api", "loras", name), {"entry": entry})


def recommend_loras(client, model, query, limit=8):
    return client.get_json(
        "/api/loras/recommend", params={"model": model, "query": query, "limit": limit}
    )
