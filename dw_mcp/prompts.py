"""The stored prompt library: the text a workflow reaches by `prompt:name`.

Prompts are the other half of authoring. A workflow argument written as
`"prompt:sitcom/duke"` resolves against this library at load time, so an
agent that can write workflows but not prompts can only ever reference text
somebody else stored.

As in `authoring`, validation stays on the server. It already refuses a
definition that fails the prompt schema and one whose `text` is itself a
reference - re-checking either here would only give the two a chance to
disagree.
"""

from dw_mcp.client import DwApiError, api_path

ENHANCE_COST_REFUSAL = (
    "Enhancing a prompt loads a language model and queues a real job on the "
    "engine, which runs one job at a time - it delays any generation waiting "
    "behind it. Tell the user what will be enhanced and with which preset, "
    "get their go-ahead, then call again with acknowledged_cost=true."
)


def list_prompts(client):
    """Every prompt in the library, with the directory it lives in."""
    return client.get_json("/api/prompts")


def get_prompt(client, name):
    """One stored prompt's full definition, `name` being the same string a
    workflow would reference it by."""
    return client.get_json(api_path("api", "prompts", name))


def get_prompt_schema(client):
    """The JSON schema a stored prompt must satisfy.

    Its own route rather than a name under /api/prompts, so a prompt called
    `schema` cannot shadow it.
    """
    return client.get_json("/api/prompt-schema")


def save_prompt(client, name, prompt):
    """Write a prompt into the library, overwriting any prompt of that name.
    The server validates before it writes."""
    return client.put_json(api_path("api", "prompts", name), {"prompt": prompt})


def delete_prompt(client, name):
    """Remove a prompt from the library."""
    return client.delete_json(api_path("api", "prompts", name))


def list_enhancers(client):
    """The enhancer presets `enhance_prompt` accepts, with descriptions."""
    return client.get_json("/api/enhancers")


def enhance_prompt(
    client,
    idea,
    preset="h3",
    model_name=None,
    device=None,
    acknowledged_cost=False,
):
    """Expand a short idea into a full prompt with a language model.

    This is queued as an ordinary job, so it is gated like a run and returns
    as soon as it is queued. The enhanced text arrives as the single text
    file in the finished job's manifest.
    """
    if not acknowledged_cost:
        raise DwApiError(ENHANCE_COST_REFUSAL)
    payload = {"idea": idea, "preset": preset}
    # Omitted rather than sent as null: absent means the preset's own default
    # model and a CPU device, which is not what an explicit null would ask for
    if model_name is not None:
        payload["model_name"] = model_name
    if device is not None:
        payload["device"] = device
    job = client.post_json("/api/enhance", payload)
    return {
        "job_id": job.get("id"),
        "status": job.get("status"),
        "queue_position": job.get("queue_position"),
        "next": "Poll get_job(job_id) until it succeeds, then read the text "
        "file in its manifest with get_output_text. Store the result with "
        "save_prompt if it is worth keeping.",
    }
