"""The UI's copies of rules the engine owns, pinned to their owners.

The UI cannot import Python, so a rule it must know is a copy. These tests
read the TypeScript source and compare each copy with its owner: a change
to one side fails until the other follows. A copy that can be read from the
server instead is deleted, not pinned.
"""

import json
import pathlib
import re

import pytest

from dw import references
from dw.content_types import AUDIO_FORMATS, MUXED_VIDEO_CONTENT_TYPE, content_type_fault
from dw.security import InvalidInputError, SecurityError, validate_workspace_name
from dw.server.job_record import TERMINAL_STATES
from dw.workflow import workflow_from_definition
from dw.workspace import DEFAULT_WORKSPACE_NAME, RESERVED_WORKSPACE_NAMES

REPO = pathlib.Path(__file__).resolve().parent.parent
UI_LIB = REPO / "ui" / "src" / "lib"


def ts_constants(path):
    """`export const NAME = 'value'` string constants in a TS file."""
    return dict(
        re.findall(r"^export const ([A-Z_]+) = '([^']*)'", path.read_text(), re.M)
    )


def ts_string_array(path, name):
    """The quoted strings of `export const NAME = [ ... ]` in a TS file.
    Exported only: a module-private copy is not the one other modules use."""
    found = re.search(
        rf"^export const {name}\b[^=]*= \[(.*?)\]", path.read_text(), re.M | re.S
    )
    assert found, f"no array {name} in {path}"
    return re.findall(r"'([^']*)'", found.group(1))


def test_the_ui_spells_every_reference_prefix_the_engine_does():
    engine = {
        name: value
        for name, value in vars(references).items()
        if name.isupper()
        and isinstance(value, str)
        and re.fullmatch(r"[a-z_]+:", value)
    }
    ui = {
        name: value
        for name, value in ts_constants(UI_LIB / "references.ts").items()
        if value.endswith(":")
    }
    assert ui == engine


def test_the_member_separator_and_reference_key_are_the_engines():
    ui = ts_constants(UI_LIB / "references.ts")
    assert ui["MEMBER_SEPARATOR"] == references.MEMBER_SEPARATOR
    assert ui["FROM_PREVIOUS_RESULT_KEY"] == references.FROM_PREVIOUS_RESULT_KEY


WORKSPACE_NAMES = json.loads(
    (REPO / "tests" / "fixtures" / "workspace_names.json").read_text()
)


@pytest.mark.parametrize(
    "case", WORKSPACE_NAMES, ids=lambda c: c["name"][:12] or "empty"
)
def test_the_engine_decides_each_shared_workspace_name_case(case):
    if case["valid"]:
        validate_workspace_name(case["name"], reserved=RESERVED_WORKSPACE_NAMES)
    else:
        with pytest.raises((InvalidInputError, SecurityError)):
            validate_workspace_name(case["name"], reserved=RESERVED_WORKSPACE_NAMES)


def test_the_ui_reserves_the_engines_workspace_names():
    assert ts_string_array(
        UI_LIB / "workspaceActions.ts", "RESERVED_WORKSPACE_NAMES"
    ) == list(RESERVED_WORKSPACE_NAMES)


def test_every_content_type_the_ui_offers_is_one_the_engine_writes():
    offered = ts_string_array(UI_LIB / "editor.ts", "CONTENT_TYPES")
    assert [ct for ct in offered if content_type_fault(ct)] == []


def _writer(settings):
    extension, arguments = settings
    return extension, tuple(sorted(arguments.items()))


def test_the_ui_offers_every_audio_writer_and_the_video_one():
    """Every distinct audio write the engine can make - container and
    subtype, so opus (an .ogg with its own subtype) counts apart from ogg -
    is reachable from an offered content type."""
    offered = set(ts_string_array(UI_LIB / "editor.ts", "CONTENT_TYPES"))
    written = {_writer(settings) for settings in AUDIO_FORMATS.values()}
    reachable = {_writer(AUDIO_FORMATS[ct]) for ct in offered if ct in AUDIO_FORMATS}
    assert reachable == written
    assert MUXED_VIDEO_CONTENT_TYPE in offered


def test_the_ui_knows_the_servers_terminal_job_states():
    assert set(ts_string_array(UI_LIB / "api.ts", "TERMINAL_STATUSES")) == set(
        TERMINAL_STATES
    )


def test_only_the_api_module_lists_the_terminal_job_states():
    owner = UI_LIB / "api.ts"
    listing = re.compile(r"\[[^\]]*'succeeded'[^\]]*'failed'[^\]]*'cancelled'[^\]]*\]")
    copies = [
        str(path.relative_to(REPO))
        for path in (REPO / "ui" / "src").rglob("*")
        if path.suffix in (".ts", ".svelte")
        and ".test." not in path.name
        and path != owner
        and listing.search(path.read_text())
    ]
    assert copies == []


def test_the_ui_default_workspace_is_the_servers():
    assert (
        ts_constants(UI_LIB / "workspaceState.svelte.ts")["DEFAULT_WORKSPACE"]
        == DEFAULT_WORKSPACE_NAME
    )


def _cache_type_enum():
    schema = json.loads((REPO / "dw" / "workflow_schema.json").read_text())

    def walk(node):
        if isinstance(node, dict):
            cache = node.get("cache")
            if isinstance(cache, dict):
                found = cache.get("properties", {}).get("type", {}).get("enum")
                if found:
                    return found
            for child in node.values():
                found = walk(child)
                if found:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = walk(child)
                if found:
                    return found
        return None

    found = walk(schema)
    assert found, "no cache type enum in the workflow schema"
    return found


def test_the_ui_cache_types_are_the_schemas():
    assert set(ts_string_array(UI_LIB / "editor.ts", "CACHE_TYPES")) == set(
        _cache_type_enum()
    )


REFERENCE_CASES = json.loads(
    (REPO / "tests" / "fixtures" / "reference_cases.json").read_text()
)


@pytest.mark.parametrize("case", REFERENCE_CASES, ids=lambda c: c["name"])
def test_the_engine_decides_each_shared_reference_case(case, tmp_path):
    workflow = workflow_from_definition(case["workflow"], str(tmp_path))
    problems = " | ".join(
        str(e.get("message", e)) for e in workflow.validation_errors()
    )
    for fragment in case["flagged"]:
        assert fragment in problems, problems
    if not case["flagged"]:
        assert problems == "", problems


def _schema():
    return json.loads((REPO / "dw" / "workflow_schema.json").read_text())


def _enum_under(key):
    def walk(node):
        if isinstance(node, dict):
            value = node.get(key)
            if isinstance(value, dict):
                found = value.get("enum") or value.get("items", {}).get("enum")
                if found:
                    return found
            for child in node.values():
                found = walk(child)
                if found:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = walk(child)
                if found:
                    return found
        return None

    found = walk(_schema())
    assert found, f"no enum under {key!r} in the workflow schema"
    return found


def test_the_ui_workflow_shapes_and_traits_are_the_schemas():
    types = UI_LIB / "types.ts"
    assert set(ts_string_array(types, "WORKFLOW_SHAPES")) == set(_enum_under("shape"))
    assert set(ts_string_array(types, "WORKFLOW_TRAITS")) == set(_enum_under("traits"))


def test_the_ui_component_slots_are_the_schemas():
    # controlnet is a component slot with a definition of its own
    component = {"#/$defs/pipeline_component", "#/$defs/controlnet"}

    def slots(node):
        found = set()
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, dict) and value.get("$ref") in component:
                    found.add(key)
                found |= slots(value)
        elif isinstance(node, list):
            for child in node:
                found |= slots(child)
        return found

    assert set(ts_string_array(UI_LIB / "editor.ts", "COMPONENT_SLOTS")) == slots(
        _schema()
    )


def test_the_ui_offload_modes_are_the_schemas():
    assert set(ts_string_array(UI_LIB / "editor.ts", "OFFLOAD_MODES")) == set(
        _enum_under("offload")
    )


API_TS = UI_LIB / "api.ts"
# Paths api.ts reaches that answer with no JSON a model declares, by any
# method: the event stream, JSON Schema documents served as-is, files and zips
NOT_IN_CONTRACT = {
    "/api/jobs/{}/events",
    "/api/schema",
    "/api/prompt-schema",
    "/api/gallery/{}/download",
    "/api/gallery/{}/thumbnail",
    "/api/gallery/archive",
    "/api/assets/archive",
    "/api/workflows/{}/download",
    "/api/prompts/{}/download",
}
# The raw workflow/prompt GETs serve the file verbatim; their PUT and DELETE
# on the same path are in the contract
VERBATIM_GETS = {("get", "/api/workflows/{}"), ("get", "/api/prompts/{}")}


def _api_calls(text):
    """Every (method, /api/ path) a client module builds, with each ${...} as
    {}. A call's method is the first `method:` before the next /api/ path or
    the next member of the object; with none it is a GET."""
    found = set()
    literals = list(re.finditer(r"[`'\"](/api/[^`'\"?]*)", text))
    for i, match in enumerate(literals):
        end = literals[i + 1].start() if i + 1 < len(literals) else len(text)
        member = re.search(r"\n  [A-Za-z]+[:(]", text[match.end() : end])
        if member:
            end = match.end() + member.start()
        method = re.search(r"method: '([A-Z]+)'", text[match.end() : end])
        path = re.sub(r"\$\{[^}]*\}", "{}", match.group(1)).rstrip("/")
        found.add((method.group(1).lower() if method else "get", path))
    return found


def test_a_call_is_read_with_its_method():
    text = (
        "  get: () => request<A>('/api/things'),\n"
        "  save: (n) =>\n    request<B>(`/api/things/${n}`, {\n      method: 'PUT',\n    }),\n"
        "  other: () => request<C>('/api/other'),\n"
    )
    assert _api_calls(text) == {
        ("get", "/api/things"),
        ("put", "/api/things/{}"),
        ("get", "/api/other"),
    }


def test_every_json_route_the_ui_calls_declares_its_response():
    from tests.test_api_contract import UI_READ_ROUTES

    covered = {(m, re.sub(r"\{[^}]*\}", "{}", path)) for m, path in UI_READ_ROUTES}
    calls = _api_calls(API_TS.read_text()) - VERBATIM_GETS - covered
    missing = sorted(call for call in calls if call[1] not in NOT_IN_CONTRACT)
    assert missing == [], (
        f"api.ts calls routes with no declared response model: {missing}"
    )
