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


def test_the_ui_offers_every_audio_container_and_the_video_one():
    offered = set(ts_string_array(UI_LIB / "editor.ts", "CONTENT_TYPES"))
    written = {extension for extension, _ in AUDIO_FORMATS.values()}
    reachable = {AUDIO_FORMATS[ct][0] for ct in offered if ct in AUDIO_FORMATS}
    assert reachable == written
    assert MUXED_VIDEO_CONTENT_TYPE in offered


def test_the_ui_knows_the_servers_terminal_job_states():
    assert set(ts_string_array(UI_LIB / "api.ts", "TERMINAL_STATUSES")) == set(
        TERMINAL_STATES
    )


def test_the_job_page_has_no_terminal_states_of_its_own():
    assert "const TERMINAL =" not in (UI_LIB / "pages" / "JobPage.svelte").read_text()


def test_the_ui_default_workspace_is_the_servers():
    assert (
        ts_constants(UI_LIB / "workspace.svelte.ts")["DEFAULT_WORKSPACE"]
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
