"""The dw plugin: skills that teach an agent to compose a model family's
workflows, kept true by tests rather than by memory.

A skill quotes catalog names and states numeric rules. Each name has to
resolve to a file, each number has to come from the library that enforces
it, and the plugin's version has to be the engine's - an installed skill is
matched to the server it was written against by that number.
"""

import glob
import os
import re

import pytest

from tests.test_examples import REPO_ROOT

PLUGIN_DIR = os.path.join(REPO_ROOT, "plugins", "dw")
SKILLS = sorted(glob.glob(os.path.join(PLUGIN_DIR, "skills", "*", "SKILL.md")))
SKILL_SIZE_LIMIT = 12 * 1024

CATALOG_NAME = re.compile(r"`((?:templates|models)/[A-Za-z0-9_./-]+)`")


def skill_text(path):
    return open(path, encoding="utf-8").read()


def frontmatter(text):
    """The YAML block between the leading '---' lines, as a dict of the
    top-level 'key: value' pairs (the two the plugin format needs)."""
    assert text.startswith("---\n"), "a skill begins with frontmatter"
    end = text.index("\n---", 4)
    fields = {}
    for line in text[4:end].splitlines():
        if ":" in line and not line.startswith(" "):
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip().strip('"')
    return fields


def _pyproject_version():
    text = open(os.path.join(REPO_ROOT, "pyproject.toml"), encoding="utf-8").read()
    return re.search(r'^version = "(.*)"$', text, re.M).group(1)


def test_the_marketplace_names_the_plugin():
    import json

    manifest = json.load(open(os.path.join(REPO_ROOT, ".claude-plugin", "marketplace.json"), encoding="utf-8"))

    assert manifest["name"] == "diffusers-workflow"
    (plugin,) = manifest["plugins"]
    assert plugin["name"] == "dw"
    assert plugin["source"] == "./plugins/dw"


def test_the_plugin_version_is_the_engine_version():
    """An installed plugin is matched to the engine it was written against by
    this number, so the release script bumps both in one commit."""
    import json

    plugin = json.load(open(os.path.join(PLUGIN_DIR, ".claude-plugin", "plugin.json"), encoding="utf-8"))

    assert plugin["name"] == "dw"
    assert plugin["version"] == _pyproject_version()


@pytest.mark.xfail(strict=True, reason="skills land in the next tasks")
def test_there_are_skills():
    assert SKILLS, "the plugin ships at least one skill"


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: os.path.basename(os.path.dirname(p)))
def test_a_skill_has_a_triggering_description_under_the_size_cap(path):
    text = skill_text(path)
    fields = frontmatter(text)

    assert fields["name"] == os.path.basename(os.path.dirname(path))
    assert "description" in fields and len(fields["description"]) > 40
    assert len(text.encode("utf-8")) <= SKILL_SIZE_LIMIT, f"{path} is over {SKILL_SIZE_LIMIT} bytes"


@pytest.mark.parametrize("path", SKILLS, ids=lambda p: os.path.basename(os.path.dirname(p)))
def test_every_catalog_name_a_skill_quotes_resolves(path):
    """A renamed template fails here rather than in a cold session."""
    names = CATALOG_NAME.findall(skill_text(path))

    assert names, f"{path} quotes no catalog names"
    for name in names:
        target = os.path.join(REPO_ROOT, "workflows", name.removesuffix(".json") + ".json")
        assert os.path.isfile(target), f"{path} quotes {name}, which is not a workflow ({target})"
