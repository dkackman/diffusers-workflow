# dw plugin: model-family composition skills — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Claude Code plugin in this repo, installed once, holding one composition skill per model family (MiniMax H3, LTX-2.5) that teaches which template fits which shape, the hard rules, cost, and the run-and-judge loop, and defers prompt format to the vendors' own text.

**Architecture:** Repository content only: a marketplace manifest at the root, one plugin under `plugins/dw/` with two `SKILL.md` files, a README, a version line the release script bumps. Drift tests in `tests/test_plugin_skills.py` resolve every catalog name a skill quotes, pin every number a skill states to the diffusers module that enforces it, and hold the one quoted vendor text equal to the library constant. Nothing under `dw/`, `dw_mcp/` or the wheel changes.

**Tech Stack:** Claude Code plugin format (`.claude-plugin/marketplace.json`, `.claude-plugin/plugin.json`, `skills/<name>/SKILL.md` with YAML frontmatter), pytest, diffusers 0.41 (installed in the venv; `diffusers.pipelines.ltx2.utils`, `diffusers.modular_pipelines.minimax_h3`).

**Spec:** [docs/superpowers/specs/2026-09-07-dw-plugin-skills-design.md](../specs/2026-09-07-dw-plugin-skills-design.md). Package A (merged as #52) is the catalog these skills describe.

## Global Constraints

- Composition only: a skill carries no prompt format of its own. H3 prompts come from MiniMax's `h3-prompt-writing` skill or the two guides on the model card, else the family's enhance-prompt template; LTX-2.5 prompts follow `LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT` quoted from diffusers and held equal by test.
- Every catalog name a skill quotes in backticks (`templates/...`) resolves to a file under `workflows/`.
- Every number a skill states as a rule is pinned by a test to the source it comes from.
- Each `SKILL.md` is under 12 KB and its frontmatter `description` names the model.
- `plugins/dw/.claude-plugin/plugin.json` `version` equals `pyproject.toml`'s.
- No MCP server declared by the plugin; a skill's first instruction is `get_server_info`.
- Nothing under `dw/`, `dw_mcp/`, `MANIFEST.in` or `pyproject.toml`'s package data changes.
- Test names and docstrings state the behaviour, matching the suite's style.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Each task appends one sentence to the Part 4 row of the ledger table in `docs/proposals/agent-catalog-legibility-complete.md` (the row beginning `| Part 4 packaging |`, at the end of its last cell before the closing `|`), in the same commit.
- Work on branch `dw-plugin`, cut from `master` at or after 09e36e1.

---

## File map

| File | Responsibility | Task |
| --- | --- | --- |
| `.claude-plugin/marketplace.json` | names the marketplace and its one plugin | 1 |
| `plugins/dw/.claude-plugin/plugin.json` | plugin identity and version | 1 |
| `plugins/dw/README.md` | install and what each skill covers | 1 |
| `README.md` | two install lines after the MCP registration | 1 |
| `scripts/release.sh` | bumps plugin.json beside pyproject.toml | 1 |
| `tests/test_plugin_skills.py` | version parity; generic skill invariants; family pins | 1, 2, 3 |
| `plugins/dw/skills/minimax-h3/SKILL.md` | the H3 composition skill | 2 |
| `workflows/templates/minimax/README.md` | one line pointing at the skill | 2 |
| `plugins/dw/skills/ltx-2.5/SKILL.md` | the LTX-2.5 composition skill | 3 |
| `workflows/templates/ltx2/README.md` | one line pointing at the skill | 3 |
| `docs/proposals/agent-catalog-legibility-complete.md` | ledger row; drill result; the mould for the next family | every task, 4 |

---

### Task 1: The plugin scaffold, installable and version-locked

**Recommended model:** sonnet — every file's content is given below.

**Files:**
- Create: `.claude-plugin/marketplace.json`, `plugins/dw/.claude-plugin/plugin.json`, `plugins/dw/README.md`, `tests/test_plugin_skills.py`
- Modify: `README.md` (after the `claude mcp add --transport http ...` block, before the Server-page paragraph), `scripts/release.sh` (the version-bump block, lines ~50-77)

**Interfaces:**
- Produces: the directory `plugins/dw/skills/` (empty until Tasks 2 and 3), the test module `tests/test_plugin_skills.py` with helpers `SKILLS` (list of SKILL.md paths), `skill_text(path)`, `frontmatter(text)` that later tasks extend.

- [ ] **Step 1: Write the failing test**

Create `tests/test_plugin_skills.py`:

```python
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
```

- [ ] **Step 2: Run it to see it fail**

Run: `pytest tests/test_plugin_skills.py -v`
Expected: the marketplace and version tests FAIL with `FileNotFoundError`; `test_there_are_skills` FAILS on the empty list; the parametrized tests are collected as zero cases.

- [ ] **Step 3: Create the manifests**

`.claude-plugin/marketplace.json`:

```json
{
    "$schema": "https://anthropic.com/claude-code/marketplace.schema.json",
    "name": "diffusers-workflow",
    "description": "Skills for driving a diffusers-workflow server from Claude Code: one composition skill per model family.",
    "owner": {
        "name": "Don Kackman"
    },
    "plugins": [
        {
            "name": "dw",
            "source": "./plugins/dw",
            "description": "Compose MiniMax H3 and LTX-2.5 video workflows over a dw MCP server: which template fits which shape, the hard rules, cost, and how to judge the output. Prompt format comes from the vendors' own guides."
        }
    ]
}
```

`plugins/dw/.claude-plugin/plugin.json`, with the version copied from `pyproject.toml`'s `version = "..."` line at the time of writing:

```json
{
    "name": "dw",
    "description": "Compose MiniMax H3 and LTX-2.5 video workflows over a dw MCP server: which template fits which shape, the hard rules, cost, and how to judge the output. Prompt format comes from the vendors' own guides.",
    "version": "<the pyproject version>",
    "author": {
        "name": "Don Kackman"
    },
    "repository": "https://github.com/dkackman/diffusers-workflow",
    "keywords": [
        "diffusers",
        "video",
        "minimax-h3",
        "ltx-2.5",
        "mcp"
    ]
}
```

`plugins/dw/README.md`:

```markdown
# dw plugin

Skills for Claude Code that teach an agent to compose a model family's
workflows on a [diffusers-workflow](https://github.com/dkackman/diffusers-workflow)
server. They assume the `dw` MCP server is already registered
(see the repo README, "Drive it from Claude Code"); a skill's first move is
`get_server_info`.

Install once, from Claude Code:

```
/plugin marketplace add dkackman/diffusers-workflow
/plugin install dw@diffusers-workflow
```

| Skill | Teaches |
| ----- | ------- |
| `minimax-h3` | MiniMax H3 video with audio: one take, a longer take by chain, a piece with cuts, identity and voice references, music; the frame and canvas rules; what a run costs. Prompts come from MiniMax's own `h3-prompt-writing` skill or the guides on the model card. |
| `ltx-2.5` | LTX-2.5 video with a soundtrack: a single clip, first-frame and keyframe conditioning, the three-move two-stage flow, the IC-LoRA upscale, extend and chain; the distilled schedule and the frame and size rules. Prompts follow the trained-caption spec that ships inside diffusers. |

Each skill quotes catalog names and numeric rules that `tests/test_plugin_skills.py`
holds to the catalog and to the diffusers module that enforces them. The
plugin's version is the engine's; the release script bumps both.

Adding a family: copy a skill, follow its outline, add the family's rules to
the test, cite the vendor. The repo's `model-family-onboarding` skill
(`.claude/skills/`) is the full lifecycle.
```

- [ ] **Step 4: The README install lines**

In `README.md`, after the fenced block that ends with `--header "Authorization: Bearer $DW_API_TOKEN"` and before the paragraph beginning `You don't have to compose that command by hand`, insert:

```markdown
The [dw plugin](plugins/dw/README.md) adds one skill per model family - what
to run for a given shape, the rules that bite, what it costs - and points at
the vendors' own prompt guides rather than restating them:

```
/plugin marketplace add dkackman/diffusers-workflow
/plugin install dw@diffusers-workflow
```
```

- [ ] **Step 5: The release script bumps plugin.json too**

In `scripts/release.sh`, inside the `if [ "$current" != "$version" ]; then` block after the `echo "pyproject.toml: $current -> $version"` line, add:

```bash
    python3 - "$version" <<'EOF'
import json, sys

path = "plugins/dw/.claude-plugin/plugin.json"
with open(path, encoding="utf-8") as f:
    plugin = json.load(f)
plugin["version"] = sys.argv[1]
with open(path, "w", encoding="utf-8") as f:
    json.dump(plugin, f, indent=4, ensure_ascii=False)
    f.write("\n")
EOF
    echo "plugins/dw/.claude-plugin/plugin.json: -> $version"
```

and change the commit line `git commit -m "release $version" -- pyproject.toml` to `git commit -m "release $version" -- pyproject.toml plugins/dw/.claude-plugin/plugin.json`, and the `git diff --quiet -- pyproject.toml` guard to `git diff --quiet -- pyproject.toml plugins/dw/.claude-plugin/plugin.json`. Add one sentence to `docs/RELEASING.md` where it describes the bump: "The same commit sets `plugins/dw/.claude-plugin/plugin.json`'s version, so an installed plugin names the engine it was written against."

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_plugin_skills.py -v`
Expected: marketplace and version tests PASS; `test_there_are_skills` still FAILS (Tasks 2 and 3 add the skills). Mark it `@pytest.mark.xfail(strict=True, reason="skills land in the next tasks")` for this commit only, and note in the report that Task 2 removes the marker.

- [ ] **Step 7: Ledger and commit**

Append to the Part 4 ledger row: `Plugin task 1: a marketplace at the repo root and the dw plugin under plugins/dw, installable with two commands the README shows; plugin.json's version is the engine's, bumped by release.sh and held by test.`

```bash
git add .claude-plugin plugins/dw README.md scripts/release.sh docs/RELEASING.md tests/test_plugin_skills.py docs/proposals/agent-catalog-legibility-complete.md
git commit -m "dw plugin scaffold: marketplace, plugin manifest, version lock

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: The MiniMax H3 skill

**Recommended model:** opus — the skill is prose an agent will act on; the draft below is the content, the implementer's job is to make it read as one instruction and to keep every fact exact.

**Files:**
- Create: `plugins/dw/skills/minimax-h3/SKILL.md`
- Modify: `workflows/templates/minimax/README.md` (one line after the first paragraph), `tests/test_plugin_skills.py` (remove the xfail; add the H3 pins)

**Interfaces:**
- Consumes: Task 1's test helpers `skill_text`, `SKILLS`.
- Produces: the outline every later family copies: Before anything / Which shape / Hard rules / Prompts / Run and judge / Sources.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_plugin_skills.py`:

```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/test_plugin_skills.py -k MiniMaxH3 -v`
Expected: FAIL with `FileNotFoundError` on the skill path.

- [ ] **Step 3: Write the skill**

Create `plugins/dw/skills/minimax-h3/SKILL.md`. The draft below is the content; keep every name, number and URL exactly, and edit only for flow. Stay under 12 KB.

````markdown
---
name: minimax-h3
description: Use when a dw MCP server is connected and the user wants MiniMax H3 video - a clip with its own sound, dialogue or a speaking character, a music video, a multi-shot short with cuts, a longer take, or a subject or voice kept consistent from a reference. Picks the template for the shape, states the rules that bite, quotes cost, and points at MiniMax's own prompt guides.
---

# MiniMax H3 on a dw server

H3 generates video and audio together: speech with lip sync, ambient sound,
score. Every template here fits a 24 GB card. This skill chooses the template
and the arguments; the prompt format is MiniMax's and comes from their text,
not from here.

## Before anything

1. `get_server_info`: the device (H3 templates are CUDA; the quantized
   configurations do not run on mps) and which workspace this session is in.
2. `list_workflows(shape="shot")` and `list_workflows(shape="sequence")`:
   the family's templates by their current names, with `summary`, `traits`
   and `cost`. Trust the listing over the names quoted below.
3. `get_workflow` on the one chosen, for its variables and their defaults.

## Which shape is the request

- **One clip, up to 14 seconds, from text**: `templates/minimax/video-with-audio`.
  From a one-line idea: `templates/minimax/enhance-prompt` writes the prompt
  with the built-in Context-IR enhancer first.
- **Pinned to a picture**: first frame `templates/minimax/image-to-video`;
  first and last `templates/minimax/first-and-last-frame`; last only
  `templates/minimax/last-frame-only`; a one-line idea plus a picture
  `templates/minimax/enhance-prompt-with-image`.
- **A subject that must look the same**: `templates/minimax/reference-to-video`
  (an image fixes appearance, an audio clip fixes voice);
  `templates/minimax/composable-references` adds a video reference for framing
  and camera; `templates/minimax/generated-subject-reference` draws the subject
  with Z-Image first and references it in the same workflow;
  `templates/minimax/voice-timbre-reference` fixes a voice from a Bark-spoken line.
- **Several boards in one generation, one unbroken score**:
  `templates/minimax/storyboard` - H3 cuts between the boards inside a single
  generation, which no concat of separate clips can match for continuous audio.
- **Longer than 14 seconds as one take**: a chain. `templates/minimax/chained-segments`
  (last-frame continuity), `templates/minimax/chain-video-continuity` (the
  previous segment's tail rides along as a video reference - motion, camera and
  voice carry across the seam), `templates/minimax/chain-matched-to-audio` (a
  supplied track sets the length and is muxed back seamless),
  `templates/minimax/chain-matched-and-aligned` (all of it, per-segment prompts).
  Drift compounds per seam: reference the subject picture in every segment,
  prefer `last_segment` continuity, and use the longest segments memory allows.
- **A piece with cuts**: fresh shots from shared portraits, then a concat.
  `templates/minimax/dialogue-short` (Z-Image draws the cast, one loaded model
  per shot, `concat_videos` splices) and `templates/minimax/music-video`
  (shots cut to a generated song, lip-synced slices). A cut erases drift; the
  last shot is as clean as the first. Write shots, not takes.
- **Music alone**: `templates/minimax/music` (Music3).

If none fits, compose from `list_tasks` before authoring a new workflow, and
read the `workflows` guide's authoring section first.

## Hard rules

- `num_frames` is `17n + 5`, from 124 to 345, at a fixed 24 fps: 5.2 to 14.4
  seconds in one clip. Templates default to 124 for fast iteration; `num_frames=345`
  is the full length and fits the same 24 GB configuration. The 5-second floor
  is diffusers'; the model card says 4.
- Canvas: a 768-pixel short edge, at most 768x1344 pixels, dimensions multiples
  of 32, aspect from 1:4 to 4:1. The templates' 960x544 is the speed choice and
  is coupled to the 544p turbo LoRA and its nine steps - change one, change all
  three. Output audio is 32 kHz stereo.
- H3 is guidance-distilled: no `guidance_scale`, no negative prompt. Say what is
  there, never what is not.
- Ref2VA limits: at most 9 images, 3 videos, 3 audio clips, 12 files; audio can
  never be the only reference. References are labelled in the order passed.
- Music3 reads `audio_duration` as a ceiling, not a target: ask for more than
  the song needs and trim with `templates/audio-trim-fade`. Cap: six minutes.
- Write the prompt for the length being generated: shot timestamps should span
  the duration, or a five-second script conditions a five-second story
  whatever the frame count.

## Prompts

H3 wants Context-IR, MiniMax's own format. Do not invent it and do not
paraphrase it from examples:

1. If the `h3-prompt-writing` skill is installed (MiniMax ships it in
   https://github.com/MiniMax-AI/MiniMax-H3 under `skills/`), use it.
2. Else read the guides on the model card:
   https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md
   for text- and frame-conditioned generation, and
   https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
   for reference-conditioned generation.
3. Else run `templates/minimax/enhance-prompt` (or `-with-image`), whose
   built-in enhancer writes the format from those guides. Its `idea` is
   framed as `Task: T2VA. Duration: 5.17 seconds. Idea: ...`.

Whichever route: repeat a speaker's voice description verbatim across shots,
and when a reference picture should fix identity but not framing, say so in
the prompt's reference analysis, or every shot inherits the portrait's
composition.

## Run and judge

1. `validate_workflow` first - free, and it catches arguments the pipeline
   does not accept.
2. Quote the listing's `cost` (warm minutes on the card it was measured on;
   a first load is longer) and get the user's go-ahead before `run_workflow`
   with `acknowledged_cost=true`.
3. `wait_for_job`, then `get_job` for the manifest. A cancelled H3 job runs
   on to its next step boundary, minutes on this model.
4. Look: `get_output_image` on a frame, the gallery `url` for the clip.
   Check for the family's failure modes: a character that changes between
   shots (reference the same portraits in every shot), a reference portrait
   imposing its framing on every shot, a storyboard skipped, drift sharpening
   into noise late in a chain.

## Sources

MiniMax-H3 model card and prompt guides (huggingface.co/MiniMaxAI/MiniMax-H3),
the `h3-prompt-writing` skill (github.com/MiniMax-AI/MiniMax-H3), the diffusers
MiniMax-H3 modular pipeline, the `lightx2v/Minimax-h3-Turbo` LoRA notes. Read
2026-09-07; the audit is `docs/proposals/audits/2026-09-07-minimax-h3-audit.md`.
````

- [ ] **Step 4: The README pointer, and the xfail marker**

In `workflows/templates/minimax/README.md`, after the paragraph that ends `([the audit](...minimax-h3-audit.md)).`, add one line:

```markdown
An agent driving this family from Claude Code has the `minimax-h3` skill of the
[dw plugin](../../../plugins/dw/README.md), which chooses among these templates
and points at the guides.
```

Remove the `xfail` marker Task 1 put on `test_there_are_skills`.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_plugin_skills.py tests/test_catalog_structure.py -k "plugin or readme_link" -v`
Expected: all PASS, including the README link test for the new relative link and every catalog name the skill quotes.

- [ ] **Step 6: Ledger and commit**

Append to the Part 4 ledger row: `Plugin task 2: the minimax-h3 skill - shape decision over the family's templates, the frame and canvas rules pinned by test to the diffusers modular pipeline, prompts deferred to MiniMax's h3-prompt-writing skill and the two guides, the run-and-judge loop; the README points at it.`

```bash
git add plugins/dw/skills/minimax-h3 workflows/templates/minimax/README.md tests/test_plugin_skills.py docs/proposals/agent-catalog-legibility-complete.md
git commit -m "dw plugin: the MiniMax H3 composition skill

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: The LTX-2.5 skill, quoting the trained-caption spec

**Recommended model:** opus.

**Files:**
- Create: `plugins/dw/skills/ltx-2.5/SKILL.md`
- Modify: `workflows/templates/ltx2/README.md` (one line), `tests/test_plugin_skills.py` (the LTX pins and the quote-equality test)

**Interfaces:**
- Consumes: Task 1's helpers; Task 2's outline.
- Produces: the convention for quoting a vendor constant: a fenced block under the heading `## The trained caption spec` that a test holds equal to the library string, whitespace aside.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_plugin_skills.py`:

```python
LTX_SKILL = os.path.join(PLUGIN_DIR, "skills", "ltx-2.5", "SKILL.md")


def _fenced_block_after(text, heading):
    """The first fenced code block after a markdown heading."""
    start = text.index(heading)
    open_fence = text.index("\n```", start)
    open_end = text.index("\n", open_fence + 1)
    close_fence = text.index("\n```", open_end)
    return text[open_end + 1 : close_fence]


class TestLtx25Skill:
    """The LTX-2.5 skill's numbers come from the pipeline that enforces them,
    and the one vendor text it quotes is the library's own constant."""

    def test_the_schedule_is_the_library_s(self):
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES

        text = skill_text(LTX_SKILL)
        assert len(DISTILLED_SIGMA_VALUES) == 8 and "eight" in text
        assert len(STAGE_2_DISTILLED_SIGMA_VALUES) == 3 and "three" in text
        assert str(STAGE_2_DISTILLED_SIGMA_VALUES[0]) in text

    def test_the_size_and_frame_rules_are_the_pipeline_s(self):
        import inspect

        from diffusers.pipelines.ltx2 import pipeline_ltx2
        from diffusers.pipelines.ltx2.utils import MAX_CONDITIONING_FPS

        text = skill_text(LTX_SKILL)
        source = inspect.getsource(pipeline_ltx2)
        assert "height % 32 != 0 or width % 32 != 0" in source and "32" in text
        assert "(num_frames - 1) // self.vae_temporal_compression_ratio + 1" in source
        assert "8k + 1" in text or "8n + 1" in text
        assert MAX_CONDITIONING_FPS == 60.0 and "60" in text

    def test_the_quoted_caption_spec_is_the_library_constant(self):
        """The one vendor text the plugin carries, tied to the library that
        ships it so it cannot drift."""
        from diffusers.pipelines.ltx2.utils import LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT

        quoted = _fenced_block_after(skill_text(LTX_SKILL), "## The trained caption spec")

        assert " ".join(quoted.split()) == " ".join(LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT.split())

    def test_the_skill_starts_with_the_server(self):
        text = skill_text(LTX_SKILL)
        assert text.index("get_server_info") < text.index("templates/ltx2/")
```

- [ ] **Step 2: Run them to see them fail**

Run: `pytest tests/test_plugin_skills.py -k Ltx25 -v`
Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Write the skill**

Print the constant to paste verbatim into the fenced block (it is about 3.7 KB, so the rest of the skill has about 8 KB):

```bash
python -c "from diffusers.pipelines.ltx2.utils import LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT as t; print(t)"
```

Create `plugins/dw/skills/ltx-2.5/SKILL.md`. Keep every name, number and URL exact; edit only for flow; stay under 12 KB (check with `wc -c`).

````markdown
---
name: ltx-2.5
description: Use when a dw MCP server is connected and the user wants LTX-2.5 video - a short clip with its own soundtrack, a clip that starts from a picture or runs between two, a sharper full-size render, a 2x upscale, or a clip extended or chained longer. Picks the template for the shape, states the schedule and size rules, quotes cost, and carries the trained caption spec the prompt must follow.
---

# LTX-2.5 on a dw server

LTX-2.5 generates video and a soundtrack together, 24 fps, on a distilled
schedule that is not a knob. Every template here fits a 24 GB card. This
skill chooses the template and the arguments; the prompt is written to
Lightricks' own caption spec, quoted below from diffusers.

## Before anything

1. `get_server_info`: the device (these templates quantize with SDNQ on CUDA)
   and the workspace.
2. `list_workflows(shape="shot")`: the family's templates by their current
   names, with `summary`, `traits` and `cost`. Trust the listing over the
   names quoted below.
3. `get_workflow` on the one chosen.

## Which shape is the request

- **A clip from text**: `templates/ltx2/text-to-video` (960x544, 121 frames, 5 s).
- **From a picture**: `templates/ltx2/image-to-video` (first frame; 481 frames
  is 20 s in one pass, so reach for extend or chain only past that);
  `templates/ltx2/keyframes` (first and last frames pinned);
  `templates/ltx2/enhance-prompt` (a one-line idea plus a picture; the model's
  own enhancer writes the caption).
- **Sharper at full size**: `templates/ltx2/two-stage`, the three-move distilled
  flow - eight sigmas at 768x448, a 2x latent upsample, then renoise and three
  stage-two sigmas at 1536x896 with the audio latents carried through. The
  upsample alone is soft; the refine pass is where the detail comes from.
- **A 2x upscale of an existing clip**: `templates/ltx2/generative-upscale`,
  the IC-LoRA re-rendering at twice the size and inventing detail. For a clean
  low-resolution render, not compressed footage; fewer steps and lower guidance
  keep it closer to the reference.
- **Longer**: `templates/ltx2/extend-clip` continues a clip conditioned on all
  of it; `templates/ltx2/chained-segments` re-runs per segment on the previous
  last frame and stitches. Neither is a Lightricks recipe; both are dw's, and
  a single 481-frame pass reaches 20 seconds before either is needed.

If none fits, compose from `list_tasks` before authoring a new workflow, and
read the `workflows` guide's authoring section first.

## Hard rules

- `num_frames` is `8k + 1` (121, 241, 481); `width` and `height` are multiples
  of 32; 24 fps. Higher frame rates: condition at 60 at most, never 120 - RoPE
  time is `frame / fps`, and the model is trained around 24, 25, 30 and 60.
- The distilled transformer runs its eight trained sigmas (`DISTILLED_SIGMA_VALUES`)
  with `guidance_scale` 1.0 and STG and modality guidance off. No
  `num_inference_steps`. Those knobs mean something only against the dev
  transformer, which no 24 GB template ships.
- Stage two of the two-stage flow: renoise at 0.909375 (the first
  `STAGE_2_DISTILLED_SIGMA_VALUES` entry) and run its three sigmas at full size.
- An image condition is re-compressed at CRF 18 to match training and needs a
  PIL image; a multi-frame video condition is not re-compressed.
- Audio is generated in the first pass; nothing refines it afterwards, so
  carry the audio latents (two-stage) or pair the soundtrack back
  (`pair_audio`) on any step that works on frames alone.

## Prompts

A caption, not a tag list: one paragraph of roughly 150 to 220 words in the
present progressive, opening on the action, stating for every shot a shot
type, a camera motion (say static when there is none) and a viewpoint, with
the soundscape interleaved with the action rather than appended, in plain
observable words. For an image-conditioned clip describe only what changes
from the image; restating it invites a scene cut. The stored prompts under
`prompts/ltx2/` are written to this spec. For a one-line idea,
`templates/ltx2/enhance-prompt` runs the model's own enhancer with the same
spec. The spec itself, from `diffusers.pipelines.ltx2.utils.LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT`
(the image-to-video variant, `LTX2_5_I2V_DEFAULT_SYSTEM_PROMPT`, adds the
describe-only-changes rule):

## The trained caption spec

```
<paste LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT here verbatim>
```

## Run and judge

1. `validate_workflow` first.
2. Quote the listing's `cost` and get the go-ahead; `run_workflow` with
   `acknowledged_cost=true`.
3. `wait_for_job`, then `get_job`. Writing a 121-frame 1536x896 clip takes
   minutes after the last step ends; the job is not stuck.
4. Look at a frame and the clip. Failure modes: a scene cut where the prompt
   contradicted the image; softness when the refine pass was skipped; a
   near-silent soundtrack when the caption gave the sound nothing to do.

## Sources

Lightricks/LTX-2.5-Diffusers model card, the `ltx-pipelines` docs and
CHANGELOG (github.com/Lightricks/LTX-2), the diffusers LTX-2 pipelines and
`utils.py`. Read 2026-09-07; the audit is
`docs/proposals/audits/2026-09-07-ltx-2.5-audit.md`. Since 2026-08 Lightricks
route production quality through their DFR pipeline, which diffusers ships
and no template here uses yet.
````

- [ ] **Step 4: The README pointer**

In `workflows/templates/ltx2/README.md`, after the paragraph that ends `([the audit](...ltx-2.5-audit.md)).`, add:

```markdown
An agent driving this family from Claude Code has the `ltx-2.5` skill of the
[dw plugin](../../../plugins/dw/README.md), which chooses among these templates
and carries the caption spec.
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_plugin_skills.py tests/test_catalog_structure.py -k "plugin or readme_link" -v` and `wc -c plugins/dw/skills/*/SKILL.md`
Expected: all PASS; both skills under 12288 bytes.

- [ ] **Step 6: Ledger and commit**

Append to the Part 4 ledger row: `Plugin task 3: the ltx-2.5 skill - shape decision, the schedule and size rules pinned to the LTX-2 pipeline, the trained caption spec quoted and held equal to the diffusers constant by test; the README points at it.`

```bash
git add plugins/dw/skills/ltx-2.5 workflows/templates/ltx2/README.md tests/test_plugin_skills.py docs/proposals/agent-catalog-legibility-complete.md
git commit -m "dw plugin: the LTX-2.5 composition skill, quoting the trained caption spec

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: The cold drill, and the mould for the next family

**Recommended model:** the controlling session, with the user - the drill needs a fresh Claude Code session on the user's machine with the plugin installed, which no subagent can start. Opus for reading the transcripts.

**Files:**
- Modify: `docs/proposals/agent-catalog-legibility-complete.md` (the ledger row and a `### Plugin drill` section beside the cold-session probe)

**Interfaces:**
- Consumes: the merged branch on lem, restarted (`git checkout master && git pull`, restart `dw.serve --mcp`).

- [ ] **Step 1: Install the plugin in the drill session**

From the branch's checkout (before merge) the marketplace can be added by path: in a fresh Claude Code session started in the scratch directory the earlier drills used (`~/testing/scratch-1`, no `CLAUDE.md`, the dw MCP registered), run `/plugin marketplace add /Users/don/src/dkackman/diffusers-workflow` then `/plugin install dw@diffusers-workflow`, and confirm `/plugin` lists `minimax-h3` and `ltx-2.5`. After merge the GitHub form in the README works the same way.

- [ ] **Step 2: Run the two prompts**

In that session: `a short multi-shot video with cuts between the shots`, the prompt the earlier probe answered with three unrelated LTX-2 shots. Let it run to a finished piece if it asks for a go-ahead. Then a second fresh session without the plugin (`/plugin uninstall dw@diffusers-workflow`, new session), same prompt, as the control.

- [ ] **Step 3: Read the evidence**

The transcripts are under `~/.claude/projects/-Users-don-testing-scratch-1/*.jsonl`; lem's `~/.diffusers_helper/log/dw.log` and the uvicorn access log show the tool calls. Pass, for the plugin run: the `minimax-h3` skill fired; the agent called `get_server_info` and a shape-filtered `list_workflows`; it chose `templates/minimax/dialogue-short` or another cuts template with shared portraits; its prompts came from the guides or the `h3-prompt-writing` skill rather than an invented layout; it quoted cost before running; the finished piece holds the cast across shots. Record what the control did differently.

- [ ] **Step 4: Ledger and the mould**

Add a `### Plugin drill, <date>` section after `### Cold-session probe, 2026-09-07` in the proposal with: the two transcript paths, the tool calls in order for each run, which template each chose, whether the prompts came from vendor text, and the verdict. Append to the Part 4 ledger row: `Plugin task 4: cold drill <date> - <pass/fail and one sentence>; the mould for the next family is copy a skill, follow its six sections, add the family's rules to tests/test_plugin_skills.py, cite the vendor, and run this drill (.claude/skills/model-family-onboarding).`

```bash
git add docs/proposals/agent-catalog-legibility-complete.md
git commit -m "Plugin drill recorded; the mould for the next model family

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push -u origin dw-plugin
```

Open the PR against `master` with the four commits' subjects as its bullets and end the body with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
