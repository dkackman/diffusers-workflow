"""Prompt-library tools: the stored prompts a workflow reaches by
`prompt:name`, and the enhancer that writes one."""

import json

import httpx
import pytest

from dw_mcp import prompts
from dw_mcp.client import DwApiError, DwClient

# The stored-prompt schema keys on `text`; the name is the file, not a field
PROMPT = {"text": "a duke on a sofa", "description": "a duke"}


def scripted(routes):
    """routes: {(method, path): (status, json_body)}"""
    seen = []

    def handler(request):
        key = (request.method, request.url.path)
        seen.append(key)
        if key not in routes:
            return httpx.Response(404, json={"detail": f"unrouted {key}"})
        status, body = routes[key]
        return httpx.Response(status, json=body)

    return DwClient(transport=httpx.MockTransport(handler)), seen


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


def test_list_prompts_returns_the_library():
    client, seen = scripted(
        {
            ("GET", "/api/prompts"): (
                200,
                {"prompt_dir": "/p", "prompts": ["duke"], "details": {}},
            )
        }
    )

    result = prompts.list_prompts(client)

    assert result["prompts"] == ["duke"]
    assert seen == [("GET", "/api/prompts")]


def test_get_prompt_reads_one_by_name():
    client, seen = scripted({("GET", "/api/prompts/duke"): (200, PROMPT)})

    assert prompts.get_prompt(client, "duke")["text"] == "a duke on a sofa"
    assert seen == [("GET", "/api/prompts/duke")]


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


def test_get_prompt_surfaces_a_missing_prompt():
    client, _seen = scripted(
        {("GET", "/api/prompts/ghost"): (404, {"detail": "No such prompt"})}
    )

    with pytest.raises(DwApiError, match="No such prompt"):
        prompts.get_prompt(client, "ghost")


def test_get_prompt_schema_has_its_own_path():
    """Not /api/prompts/schema - a prompt named 'schema' would shadow it."""
    client, seen = scripted(
        {("GET", "/api/prompt-schema"): (200, {"type": "object"})},
    )

    assert prompts.get_prompt_schema(client)["type"] == "object"
    assert seen == [("GET", "/api/prompt-schema")]


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


def test_save_prompt_surfaces_a_rejected_definition():
    client, _seen = scripted(
        {("PUT", "/api/prompts/duke"): (400, {"detail": "'text' is a required"})}
    )

    with pytest.raises(DwApiError, match="required"):
        prompts.save_prompt(client, "duke", {"description": "no text"})


def test_save_prompt_surfaces_a_reserved_text_prefix():
    """The engine refuses a prompt whose text is itself a reference - the
    server says so and the message has to reach the model unaltered."""
    client, _seen = scripted(
        {
            ("PUT", "/api/prompts/duke"): (
                400,
                {
                    "detail": "A prompt's text may not itself begin with a "
                    "reference prefix (variable:, previous_result:)"
                },
            )
        }
    )

    with pytest.raises(DwApiError, match="reference prefix"):
        prompts.save_prompt(client, "duke", {"text": "variable:x"})


def test_delete_prompt_calls_delete():
    client, seen = scripted(
        {("DELETE", "/api/prompts/duke"): (200, {"name": "duke", "deleted": True})}
    )

    assert prompts.delete_prompt(client, "duke")["deleted"] is True
    assert seen == [("DELETE", "/api/prompts/duke")]


def test_delete_prompt_surfaces_a_path_the_server_refuses():
    client, _seen = scripted(
        {
            ("DELETE", "/api/prompts/../escape"): (
                400,
                {"detail": "Path traversal is not allowed"},
            )
        }
    )

    with pytest.raises(DwApiError, match="traversal"):
        prompts.delete_prompt(client, "../escape")


# ---------------------------------------------------------------- enhancer


def test_list_enhancers_returns_the_presets():
    # The shape the live server sends: a list of preset records, not a map
    client, seen = scripted(
        {
            ("GET", "/api/enhancers"): (
                200,
                {"presets": [{"key": "h3", "default_model": "Qwen/Qwen3-4B"}]},
            )
        }
    )

    presets = prompts.list_enhancers(client)["presets"]

    assert [preset["key"] for preset in presets] == ["h3"]
    assert seen == [("GET", "/api/enhancers")]


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


def test_enhance_prompt_surfaces_an_unknown_preset():
    client, _seen = scripted(
        {("POST", "/api/enhance"): (400, {"detail": "Unknown preset 'nope'"})}
    )

    with pytest.raises(DwApiError, match="Unknown preset"):
        prompts.enhance_prompt(client, "an idea", preset="nope", acknowledged_cost=True)
