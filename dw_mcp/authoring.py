"""Writing workflows. Validation is free and comes first; saving overwrites,
so it is annotated destructive at the tool layer.

Path confinement is the server's job (dw/security.py already refuses
traversal and anything outside the workflow directory). Nothing here
re-implements it - a second, subtly different check is how the two drift.
"""

import json

from dw_mcp.client import DwApiError, api_path


def validate_workflow(
    client,
    workflow=None,
    name=None,
    inline_workflow=None,
    workflow_path=None,
    workspace=None,
    arguments=None,
):
    """Schema- and signature-check a workflow without queuing anything. This
    is free (no GPU work) and should be called before any run or save. Give
    either an inline definition (`workflow`, or `inline_workflow` - the same
    thing `run_workflow` calls it) or the name of a stored one (`name`, or
    `workflow_path` - the same thing `run_workflow` calls it), as
    `list_workflows` reports it. A validated document can be handed straight
    to `run_workflow` under either spelling.

    `arguments` is the same dict `run_workflow` takes, checked against the
    variables the workflow declares and against this workspace's libraries.
    Without it the answer is about the stored definition and its stock
    defaults - which is everything except the part the caller wrote
    (2026-09-11).

    `plan.cached_steps` is evaluated against the pinned workspace's own
    `outputs/` - a run sitting in a different workspace, however identical
    its arguments and seed, does not count as a hit. Pin `workspace` to the
    one an earlier run actually used if you want to see it credited."""
    if workflow is not None and inline_workflow is not None:
        raise DwApiError(
            "`workflow` and `inline_workflow` are the same thing - provide only one."
        )
    if name is not None and workflow_path is not None:
        raise DwApiError(
            "`name` and `workflow_path` are the same thing - provide only one."
        )
    inline = workflow if workflow is not None else inline_workflow
    stored = name if name is not None else workflow_path
    if (inline is None) == (stored is None):
        raise DwApiError(
            "Provide exactly one of `workflow`/`inline_workflow` (an inline "
            "definition) or `name`/`workflow_path` (a stored workflow)."
        )
    params = {"workspace": workspace} if workspace else None
    payload = {"workflow_path": stored} if inline is None else {"workflow": inline}
    if arguments:
        payload["arguments"] = arguments
    # The server resolves a name against its own workflow directory, so
    # validation sees the same base directory a run would
    return client.post_json("/api/validate", payload, params=params)


def save_workflow(client, name, workflow=None, patch=None):
    """Write a workflow into the server's writable workflow directory,
    overwriting any file already under that name there. A name that resolves
    to one of the server's read-only sources (an examples directory) is not
    overwritten: the copy lands in the writable directory and shadows it from
    then on. The server validates before writing.

    Give exactly one of `workflow` (the full document to write) or `patch`
    (a JSON Merge Patch, RFC 7396, applied to the currently stored
    definition): a dict whose keys overwrite the stored ones, recursively
    for nested dicts, so bumping one argument means sending just that
    argument rather than the whole document. A key set to `null` deletes it.
    A list replaces the stored list whole - a merge patch has no notion of
    list position, so changing one entry of a `for_each` list still means
    sending that whole list."""
    if (workflow is None) == (patch is None):
        raise DwApiError(
            "Provide exactly one of `workflow` (a full replacement) or "
            "`patch` (a JSON merge patch onto the stored version)."
        )
    workflow = _coerce_json_object(workflow, "workflow")
    patch = _coerce_json_object(patch, "patch")
    if patch is not None:
        current = client.get_json(api_path("api", "workflows", name))
        workflow = _merge_patch(current, patch)
    return client.put_json(api_path("api", "workflows", name), {"workflow": workflow})


def _coerce_json_object(value, param_name):
    """A tool argument typed as an object can still arrive as a JSON-encoded
    string (a caller that serialized it before handing it over, or a client
    that couldn't parse a malformed document and passed the raw text
    through). Accept that case rather than letting it reach the server as a
    string, where the schema rejection names the wrong problem - not "this
    isn't an object" but a bare pydantic `type=dict_type, input_type=str`,
    which reads as if the field itself were misdeclared."""
    if value is None or isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise DwApiError(
            f"`{param_name}` must be a JSON object, not {type(value).__name__}."
        )
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as e:
        raise DwApiError(f"`{param_name}` is not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise DwApiError(
            f"`{param_name}` must be a JSON object, not {type(parsed).__name__}."
        )
    return parsed


def _merge_patch(target, patch):
    """RFC 7396 JSON Merge Patch: each dict key in `patch` merges
    recursively into `target`; any other value replaces `target` outright;
    `None` deletes the key from the result."""
    if not isinstance(patch, dict):
        return patch
    result = dict(target) if isinstance(target, dict) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = _merge_patch(result.get(key), value)
    return result


def delete_workflow(client, name):
    """Remove a workflow from the server's workflow directory."""
    return client.delete_json(api_path("api", "workflows", name))
