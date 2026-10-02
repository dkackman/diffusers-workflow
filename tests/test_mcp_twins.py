"""dw_mcp's copies of rules the engine owns, pinned to their owners.

dw_mcp cannot import dw (it stays torch-free; docs/ARCHITECTURE.md, *MCP*),
so every rule it knows is a second copy. The tests can import both, so each
copy is compared with its owner here: a change to one side fails until the
other follows. A copy that needs no twin is better deleted than pinned.
"""

import re

import pytest

from dw import media as dw_media
from dw import plan, references, run as dw_run, workspace
from dw.server import assess, job_record, netinfo
from dw.server.routes import assets as asset_routes
from dw.server.routes import media as media_routes
from dw_mcp import assets, client, diagnose, media
from dw_mcp.server import INSTRUCTIONS
from dw_mcp.tools_authoring import AuthoringTools, PromptTools, WorkspaceTools
from dw_mcp.tools_catalog import CatalogTools
from dw_mcp.tools_jobs import JobTools
from dw_mcp.tools_media import MediaTools


@pytest.mark.parametrize(
    "copy, owner",
    [
        (assets.ALLOWED_UPLOAD_EXTENSIONS, asset_routes.ALLOWED_UPLOAD_EXTENSIONS),
        (assets.MAX_UPLOAD_BYTES, asset_routes.MAX_UPLOAD_BYTES),
        (client.LOOPBACK_HOSTS, netinfo.LOOPBACK_HOSTS),
        (client.DEFAULT_WORKSPACE, workspace.DEFAULT_WORKSPACE_NAME),
        (set(diagnose.TERMINAL_STATUSES), set(job_record.TERMINAL_STATES)),
        (set(dw_run.TERMINAL_STATUSES), set(job_record.TERMINAL_STATES)),
        (media.MAX_RETURNED_BYTES, dw_media.MAX_INLINE_AUDIO_BYTES),
        (media.MIN_DIMENSION, media_routes.FRAME_MIN_DIMENSION),
    ],
    ids=[
        "upload extensions",
        "upload size cap",
        "loopback hosts",
        "default workspace",
        "terminal job states (dw_mcp)",
        "terminal job states (dw.run)",
        "inline payload cap",
        "frame minimum dimension",
    ],
)
def test_a_copied_constant_equals_its_owner(copy, owner):
    assert copy == owner


def test_dw_run_matches_the_unreachable_workflow_refusal(tmp_path):
    """dw.run falls back to sending a workflow inline on exactly this 400,
    matched by its prefix."""
    from dw.server.catalog import resolve_workflow_reference
    from fastapi import HTTPException

    class EmptyLibrary:
        def find(self, name):
            return None

        def root_for_path(self, path):
            return None

        def entries(self):
            return {}, []

    with pytest.raises(HTTPException) as refused:
        resolve_workflow_reference(str(tmp_path / "nowhere.json"), EmptyLibrary())
    assert str(refused.value.detail).startswith(dw_run._UNREACHABLE_PREFIX)


def _text(fn_or_text):
    text = fn_or_text if isinstance(fn_or_text, str) else fn_or_text.__doc__
    return " ".join(text.split())


def _listed(text, pattern):
    """The words of the list `pattern`'s one group captures, split on commas
    and a final 'or', backticks and colons left as written."""
    found = re.search(pattern, text)
    assert found, f"no list matching {pattern!r} in: {text[:200]}"
    words = re.split(r",\s*(?:or\s+)?|\s+or\s+", found.group(1))
    return {word.strip().strip("`") for word in words if word.strip()}


def _schema_enum(key):
    import json

    with open(workspace.__file__.replace("workspace.py", "workflow_schema.json")) as f:
        schema = json.load(f)

    def walk(node):
        if isinstance(node, dict):
            if key in node:
                value = node[key]
                return value.get("enum") or value.get("items", {}).get("enum")
            for child in node.values():
                found = walk(child)
                if found:
                    return found
        return None

    found = walk(schema)
    assert found, f"no enum under {key!r} in the workflow schema"
    return set(found)


def test_the_shapes_and_traits_an_agent_reads_are_the_schemas():
    shapes, traits = _schema_enum("shape"), _schema_enum("traits")
    tool = _text(CatalogTools.list_workflows)
    assert _listed(tool, r"one of (.*?)\. `traits`") == shapes
    assert _listed(tool, r"all must match\): (.*?)\. Each entry") == traits
    instructions = _text(INSTRUCTIONS)
    assert _listed(instructions, r"Shapes: (.*?)\. Traits:") == shapes
    assert _listed(instructions, r"Traits: (.*?)\. An ") == traits


def test_the_job_states_list_jobs_names_are_the_servers():
    states = {job_record.QUEUED, job_record.RUNNING, *job_record.TERMINAL_STATES}
    assert _listed(
        _text(CatalogTools.list_jobs), r"set of them - (.*?)\. `workspace`"
    ) == (states)


def test_the_reserved_workspace_names_are_the_servers():
    assert _listed(
        _text(WorkspaceTools.create_workspace), r"reserved folder names \((.*?)\)"
    ) == set(workspace.RESERVED_WORKSPACE_NAMES)


def test_the_estimate_bases_are_the_planners():
    bases = {
        plan.OBSERVED,
        plan.PER_ENTRY,
        plan.CATALOG,
        plan.DERIVED,
        plan.OTHER_DEVICE,
        plan.UNKNOWN,
    }
    assert _listed(
        _text(AuthoringTools.validate_workflow), r"`basis` \((.*?) - how"
    ) == (bases)


def test_the_reserved_prompt_prefixes_are_the_references_modules():
    assert _listed(
        _text(PromptTools.save_prompt), r"reference prefix \((.*?)\)"
    ) == set(references.RESERVED_TEXT)


def test_the_assessment_probes_are_the_servers():
    assert _listed(_text(MediaTools.assess_output), r"`probe` \((.*?)\)") == set(
        assess.PROBES
    )


def test_every_queue_direction_move_job_offers_is_one_the_queue_takes():
    from typing import get_args, get_type_hints

    from dw.server.jobs import JobManager

    offered = get_args(get_type_hints(JobTools.move_job)["direction"])
    manager = JobManager.__new__(JobManager)
    manager._lock = __import__("threading").Lock()
    manager._pending = []
    for direction in offered:
        assert manager.move("no-such-job", direction) is None
    with pytest.raises(ValueError):
        manager.move("no-such-job", "sideways")


def _gated_tools():
    import inspect

    from dw_mcp.tools_catalog import ModelTools

    for cls in (
        AuthoringTools,
        CatalogTools,
        JobTools,
        MediaTools,
        ModelTools,
        PromptTools,
        WorkspaceTools,
    ):
        for name, fn in inspect.getmembers(cls, inspect.isfunction):
            if "acknowledged_cost" in inspect.signature(fn).parameters:
                yield cls, name


def test_every_tool_that_takes_acknowledged_cost_refuses_without_it():
    """Spending needs consent: the gate is written per handler, so the rule
    is checked over every tool that declares the parameter, not a list."""
    import inspect

    import httpx

    from dw_mcp.client import DwApiError, DwClient

    def handler(request):
        # delete_workspace asks the server unacknowledged on purpose: its
        # 409 says what would be removed. That is the server's own gate.
        if request.method == "DELETE" and "acknowledged" not in request.url.params:
            return httpx.Response(409, json={"detail": "would remove 3 files"})
        raise AssertionError(f"gate let {request.method} {request.url.path} through")

    gated = list(_gated_tools())
    assert len(gated) >= 7
    for cls, name in gated:
        tools = cls(DwClient(transport=httpx.MockTransport(handler)))
        fn = getattr(tools, name)
        required = {
            p.name: "x"
            for p in inspect.signature(fn).parameters.values()
            if p.default is inspect.Parameter.empty
        }
        with pytest.raises(DwApiError, match="acknowledged_cost"):
            fn(**required)
