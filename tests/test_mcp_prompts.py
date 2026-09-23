"""Prompt-library tools: the stored prompts a workflow reaches by
`prompt:name`, and the enhancer that writes one."""

import json

import httpx
import pytest

from dw_mcp import prompts
from dw_mcp.client import DwApiError, DwClient

# The stored-prompt schema keys on `text`; the name is the file, not a field
PROMPT = {"text": "a duke on a sofa", "description": "a duke"}


def recording(response):
    """A client that records the one request it is given."""
    seen = {}

    def handler(request):
        seen["method"] = request.method
        seen["path"] = request.url.path
        body = request.read()
        seen["body"] = json.loads(body) if body else None
        return response

    return DwClient(transport=httpx.MockTransport(handler)), seen


# ----------------------------------------------------------------- reading


def scripted_with_params(routes):
    """routes: {(method, path): (status, json_body)}; records each request's
    (method, path) and query parameters."""
    seen = []

    def handler(request):
        key = (request.method, request.url.path)
        seen.append((key, dict(request.url.params)))
        if key not in routes:
            return httpx.Response(404, json={"detail": f"unrouted {key}"})
        status, body = routes[key]
        return httpx.Response(status, json=body)

    return DwClient(transport=httpx.MockTransport(handler)), seen


def test_list_prompts_asks_the_server_to_leave_the_bodies_out():
    client, seen = scripted_with_params(
        {("GET", "/api/prompts"): (200, {"prompts": [], "details": {}})}
    )

    prompts.list_prompts(client)

    # The default is the cheap listing: a prompt body belongs in get_prompt,
    # and 44 of them do not fit a client's result cap
    assert seen[0][1]["include_text"] == "false"
    assert "tag" not in seen[0][1]
    assert "intended_model" not in seen[0][1]


def test_list_prompts_forwards_the_filters_and_can_ask_for_the_text():
    client, seen = scripted_with_params(
        {("GET", "/api/prompts"): (200, {"prompts": [], "details": {}})}
    )

    prompts.list_prompts(
        client, tag="ic-lora", intended_model="ltx-2.5", include_text=True
    )

    assert seen[0][1] == {
        "tag": "ic-lora",
        "intended_model": "ltx-2.5",
        "include_text": "true",
    }


def test_get_prompt_encodes_a_foldered_name():
    """A prompt lives at `folder/name`, and the slash has to survive as a
    path segment the server validates rather than one httpx normalizes."""
    seen = {}

    def handler(request):
        # raw_path, not path: the point is the bytes that go on the wire
        seen["raw_path"] = request.url.raw_path
        return httpx.Response(200, json=PROMPT)

    client = DwClient(transport=httpx.MockTransport(handler))

    prompts.get_prompt(client, "sitcom/duke")

    assert seen["raw_path"] == b"/api/prompts/sitcom%2Fduke"


# ----------------------------------------------------------------- writing


def test_save_prompt_puts_the_definition_under_its_name():
    client, seen = recording(
        httpx.Response(200, json={"name": "duke", "path": "/p/duke.json"})
    )

    result = prompts.save_prompt(client, "duke", PROMPT)

    assert seen["method"] == "PUT"
    assert seen["path"] == "/api/prompts/duke"
    assert seen["body"] == {"prompt": PROMPT}
    assert result["name"] == "duke"


# ---------------------------------------------------------------- enhancer


def test_enhance_prompt_refuses_without_acknowledgement():
    """Enhancing loads a language model and queues a real job on the
    one-at-a-time engine, so it is gated exactly like a run."""

    def refusing(request):
        raise AssertionError("the gate let a request through")

    client = DwClient(transport=httpx.MockTransport(refusing))

    with pytest.raises(DwApiError, match="acknowledged_cost=true"):
        prompts.enhance_prompt(client, "a duke on a sofa")


def test_enhance_prompt_queues_a_job_once_acknowledged():
    client, seen = recording(
        httpx.Response(201, json={"id": "j1", "status": "queued", "queue_position": 0})
    )

    result = prompts.enhance_prompt(client, "a duke on a sofa", acknowledged_cost=True)

    assert seen["method"] == "POST"
    assert seen["path"] == "/api/enhance"
    assert seen["body"]["idea"] == "a duke on a sofa"
    assert result["job_id"] == "j1"
    assert result["status"] == "queued"


def test_enhance_prompt_points_at_the_tool_that_reads_the_result():
    """The enhanced text is a text file in the job's manifest, so the
    handoff has to name the tool that can read one."""
    client, _seen = recording(
        httpx.Response(201, json={"id": "j1", "status": "queued"})
    )

    result = prompts.enhance_prompt(client, "an idea", acknowledged_cost=True)

    assert "get_output_text" in result["next"]


def test_enhance_prompt_sends_the_preset_and_overrides():
    client, seen = recording(httpx.Response(201, json={"id": "j1", "status": "queued"}))

    prompts.enhance_prompt(
        client,
        "an idea",
        preset="sdxl",
        model_name="org/llm",
        device="cpu",
        acknowledged_cost=True,
    )

    assert seen["body"]["preset"] == "sdxl"
    assert seen["body"]["model_name"] == "org/llm"
    assert seen["body"]["device"] == "cpu"


def test_enhance_prompt_omits_overrides_it_was_not_given():
    """The server picks the preset's own default model and a CPU device when
    the fields are absent; sending nulls would not be the same thing."""
    client, seen = recording(httpx.Response(201, json={"id": "j1", "status": "queued"}))

    prompts.enhance_prompt(client, "an idea", acknowledged_cost=True)

    assert "model_name" not in seen["body"]
    assert "device" not in seen["body"]
