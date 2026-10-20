"""Every pointer an agent is handed to a guide section opens that section.

The resident text - the MCP server's instructions and tool descriptions, and
each plugin skill's SKILL.md - says what a tool or a family does and points
at a guide for the rest. A pointer is the whole of the discovery path: a
section renamed out from under one, or a pointer trimmed away to fit a
budget (#574 re-broke #282 that way), leaves an agent holding no way to the
detail. So every pointer must resolve through the same `get_guide` the
server answers with, and the tools that carry one now must keep it.
"""

import asyncio
import glob
import os
import re

import pytest

from dw.server import guides
from tests.test_examples import REPO_ROOT

pytest.importorskip("mcp", reason="the mcp extra is not installed")

from tests.test_mcp_server import ok, server_over, tools_of  # noqa: E402

SKILL_TEXTS = sorted(
    glob.glob(os.path.join(REPO_ROOT, "plugins", "dw", "skills", "*", "SKILL.md"))
    + glob.glob(
        os.path.join(REPO_ROOT, "plugins", "dw", "skills", "*", "references", "*.md")
    )
)

# The guide a file name stands for, for the "WORKFLOW_GUIDE's ..." form.
_BY_FILE = {
    file_name.removesuffix(".md"): name
    for name, (file_name, _summary) in guides.GUIDES.items()
}

# The forms a pointer is written in, with whitespace already collapsed:
#   get_guide("workflows", section="The loop")   get_guide("loras")
#   the `acceleration` guide, section "Reading Memory While Offloading"
#   WORKFLOW_GUIDE's "The loop"
_CALL = re.compile(
    r'get_guide\(\s*"(?P<name>[a-z-]+)"(?:\s*,\s*section\s*=\s*"(?P<section>[^"]+)")?\s*\)'
)
_PROSE = re.compile(r'the `(?P<name>[a-z-]+)` guide, section "(?P<section>[^"]+)"')
_FILE = re.compile(r'(?P<file>[A-Z0-9_]+)\'s "(?P<section>[^"]+)"')

# The tools whose description points at a guide today. A pointer trimmed
# away to fit the surface budget fails here rather than leaving the tool
# with no way to its detail; adding one to a tool means adding it here.
TOOLS_THAT_POINT = {
    "get_job",
    "get_memory",
    "get_output_audio",
    "save_lora",
    "validate_workflow",
    "wait_for_job",
}


def pointers(text):
    """(guide name, section or None) for each pointer in `text`."""
    flat = " ".join(text.split())
    found = [(m["name"], m["section"]) for m in _CALL.finditer(flat)]
    found += [(m["name"], m["section"]) for m in _PROSE.finditer(flat)]
    found += [
        (_BY_FILE[m["file"]], m["section"])
        for m in _FILE.finditer(flat)
        if m["file"] in _BY_FILE
    ]
    return found


def unresolved(text):
    """The pointers in `text` that `get_guide` refuses."""
    bad = []
    for name, section in pointers(text):
        try:
            guides.get_guide(name, section)
        except (guides.GuideError, KeyError) as error:
            bad.append((name, section, str(error)[:120]))
    return bad


def mcp_texts():
    server = server_over(ok({}))
    tools = asyncio.run(tools_of(server))
    texts = {name: tool.description or "" for name, tool in tools.items()}
    texts["<instructions>"] = server.instructions
    return texts


def test_every_pointer_in_the_mcp_surface_resolves():
    bad = {name: unresolved(text) for name, text in mcp_texts().items()}
    assert not {name: b for name, b in bad.items() if b}


def test_the_tools_that_point_at_a_guide_still_do():
    texts = mcp_texts()
    lost = sorted(name for name in TOOLS_THAT_POINT if not pointers(texts[name]))
    assert not lost, f"these tools lost their guide pointer: {lost}"


def test_the_server_instructions_point_at_the_authoring_section():
    assert ("workflows", "Authoring a workflow from an agent") in pointers(
        mcp_texts()["<instructions>"]
    )


@pytest.mark.parametrize(
    "path", SKILL_TEXTS, ids=lambda p: os.path.relpath(p, REPO_ROOT)
)
def test_every_pointer_in_a_skill_resolves(path):
    with open(path, encoding="utf-8") as handle:
        assert not unresolved(handle.read())


def test_the_pointer_forms_are_recognised():
    """The extraction itself, so a form it stops matching cannot pass the
    resolution tests by finding nothing."""
    text = (
        'see get_guide("workflows",\n section="The loop") and the '
        '`acceleration` guide, section "Caching", and WORKFLOW_GUIDE\'s '
        '"References", and get_guide("loras")'
    )
    assert pointers(text) == [
        ("workflows", "The loop"),
        ("loras", None),
        ("acceleration", "Caching"),
        ("workflows", "References"),
    ]
