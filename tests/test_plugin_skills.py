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


H3_SKILL = os.path.join(PLUGIN_DIR, "skills", "minimax-h3", "SKILL.md")


class TestMiniMaxH3Skill:
    """The numbers the H3 skill states come from the diffusers modular
    pipeline that enforces them, so a library change fails here."""

    def test_the_frame_rule_and_bounds_are_the_pipeline_s(self):
        import inspect

        from diffusers.modular_pipelines.minimax_h3 import before_encoder, modular_pipeline

        text = skill_text(H3_SKILL)
        assert "17n + 5" in text or "17 * n + 5" in text
        assert "17 * n + 5" in inspect.getsource(before_encoder)
        assert modular_pipeline.MINIMAX_H3_FPS == 24 and "24 fps" in text
        # 124 and 345 are the smallest and largest 17n + 5 inside 5 to 15 seconds at 24 fps
        assert "124" in text and "345" in text
        assert 124 == 17 * 7 + 5 and 345 == 17 * 20 + 5
        assert 124 / modular_pipeline.MINIMAX_H3_FPS >= 5 and 345 / modular_pipeline.MINIMAX_H3_FPS <= 15

    def test_the_canvas_rules_are_the_pipeline_s(self):
        import inspect

        from diffusers.modular_pipelines.minimax_h3 import before_encoder, modular_pipeline

        text = skill_text(H3_SKILL)
        source = inspect.getsource(before_encoder)
        assert 'ConfigSpec("canvas_short_edge", 768)' in source and "768" in text
        assert "768 * 1344" in source and "1344" in text
        assert modular_pipeline.MINIMAX_H3_MIN_ASPECT_RATIO == 1 / 4
        assert modular_pipeline.MINIMAX_H3_MAX_ASPECT_RATIO == 4
        assert "1:4" in text and "4:1" in text
        assert "32" in text  # dimensions are multiples of 32

    def test_the_skill_defers_prompt_format_to_minimax(self):
        text = skill_text(H3_SKILL)
        assert "h3-prompt-writing" in text
        assert "VIDEO_PROMPT_WRITING_GUIDE_base_en.md" in text
        assert "VIDEO_PROMPT_WRITING_GUIDE_ref_en.md" in text
        assert "`templates/minimax/enhance-prompt`" in text
        # no transcription of the format: none of its section labels appear as instructions
        assert "retention_analysis:" not in text

    def test_the_skill_starts_with_the_server(self):
        text = skill_text(H3_SKILL)
        assert text.index("get_server_info") < text.index("templates/minimax/")
