# Output Subfolders, Stage 3 (Steering) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every shipped multi-step template says which of its outputs is the deliverable, a test keeps that from drifting, the plugin skills state the convention for their families, and the docs carry the release note.

**Architecture:** No engine or server code changes. Stage 3 is data and prose: `"subfolder": "final"` / `"subfolder": "intermediate"` added to the `result` block of every saving step in every `workflows/templates/**` file that has two or more saving steps; a new `tests/test_template_subfolders.py` asserts the rule so it cannot drift and pins the packaged builtins in `dw/workflows/` as unmarked; one sentence in each of the three plugin skills (`plugins/dw/skills/*/SKILL.md`) with a test in `tests/test_plugin_skills.py`; the authoring guide, `CLAUDE.md` and the design doc updated.

**Tech Stack:** JSON, Markdown, pytest. Tests: `source ./activate` then `python -m pytest tests/<file> -q`.

**Spec:** `docs/proposals/output-folders.md` — sections *Steering consumers toward it*, *Breaking changes* (item 3), *Tests* (the *Templates* bullet), and stage 3 of *Phasing*. The design is the authority; this plan argues from it. One ruling against its text is recorded under Global Constraints.

## Global Constraints

- A **saving step** is a step whose `result` sets `content_type` and does not set `"save": false`. A template is **in scope** when it has two or more saving steps. Definition copied from the design's *Steering* item 1.
- Every saving step of an in-scope template carries a literal `"subfolder"` of exactly `"final"` or `"intermediate"`; every in-scope template has at least one `final` step. No `variable:`/`item:` subfolders in templates - the roles are fixed by the template's shape.
- **Ruling (deviation from the design text):** the design exempts "a template whose saving steps are all `workflow` steps over `builtin:` children (`compose-workflows`, `sub-workflow`)". Neither named template is over a builtin (`compose-workflows` takes the child paths as variables; `sub-workflow` names `./lora.json` and has a `task` step too), and a `workflow` step's own `result` block saves a real file - the parent's copy of the child's return value, written through `Result.save(self.step_output_dir(step_data), ...)` in `dw/workflow.py` - which the design's own *Sub-workflows* paragraph states. The exemption therefore protects nothing: the child's steps never see the parent's subfolder. **Both templates are marked like any other.** The one exemption that survives is the packaged builtins in `dw/workflows/` themselves, which stay unmarked and are pinned so by the drift test. The design's *Steering* paragraph is corrected in Task 4. Cost if wrong: two templates' parent-saved files land in `final/`/`intermediate/` where the design said root - a one-line revert per file.
- `workflows/models/**` is out of scope (design, *decided*). The drift test walks `workflows/templates/**` only.
- The `subfolder` key goes **last** in the `result` object, after every existing key, so the diff is one added line per step and the templates read alike.
- Only the `result` blocks change in the templates: no re-indenting, no key reordering, no description edits. Files are 2-space-indented JSON; keep it.
- `docs/WORKFLOW_GUIDE.md`'s *Authoring a workflow from an agent* section and its CLAUDE.md mirror change together (CLAUDE.md rule).
- Commit messages end with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Branch off `develop` (suggested `output-subfolders-stage-3`).

---

## File map

| File | Responsibility in this plan |
|---|---|
| `tests/test_template_subfolders.py` (new) | The drift test: every in-scope template's saving steps all carry `final`/`intermediate`, at least one `final`; builtins unmarked |
| `workflows/templates/**/*.json` (18 files, listed in Task 1) | Gain `"subfolder"` on each saving step's `result` |
| `plugins/dw/skills/minimax-h3/SKILL.md`, `.../ltx-2.5/SKILL.md`, `.../minimax-music3/SKILL.md` | One paragraph each stating the convention for the family's templates |
| `tests/test_plugin_skills.py` | Pins that each skill states the convention and that the H3 skill's named roles match the templates |
| `docs/WORKFLOW_GUIDE.md` | *Saying which output is the deliverable*: says the shipped templates follow the convention |
| `CLAUDE.md` | *Result subfolders* gotcha: the templates are marked; the release-note line |
| `docs/proposals/output-folders.md` | Status → stages 1-3 done; *Steering* item 1 corrected per the ruling |

---

### Task 1: Drift test and template markup

**Files:**
- Create: `tests/test_template_subfolders.py`
- Modify: the 18 templates in the table below (only their `result` blocks)

**Interfaces:**
- Consumes: `get_example_files()` from `tests/test_examples.py` (returns repo-relative paths of every workflow JSON under `workflows/`, sorted); `REPO_ROOT` from the same module.
- Produces: `saving_steps(definition)` and `IN_SCOPE` (list of repo-relative template paths) in the new test module - Task 2's test imports neither; it reads the templates directly.

- [ ] **Step 1: Write the failing test**

Create `tests/test_template_subfolders.py`:

```python
"""Every multi-step template says which of its outputs is the deliverable.

A run writes everything into one directory, so a finished episode sits
beside the scratch that went into it. A step's `result.subfolder` places
its files - by convention `final` for the deliverable and `intermediate`
for the rest (docs/WORKFLOW_GUIDE.md, "Saying which output is the
deliverable"). Agents compose by copying a template, so the convention
propagates only if every shipped template follows it; this test is what
keeps it from drifting.

The rule, from docs/proposals/output-folders.md: a *saving step* is one
whose `result` sets `content_type` and does not set `save: false`; a
template is in scope when it has two or more saving steps; every saving
step of an in-scope template carries `final` or `intermediate`, and at
least one is `final`. The packaged builtins in dw/workflows/ stay unmarked:
a builtin is a step list a parent composes, and a role is the parent's to
assign - the parent's own `result` block, not the child's, is where it
says so.
"""

import json
import os

import pytest

from tests.test_examples import BUILTIN_DIR, REPO_ROOT, get_example_files

CONVENTION = {"final", "intermediate"}


def load(relative_path):
    with open(os.path.join(REPO_ROOT, relative_path), encoding="utf-8") as file:
        return json.load(file)


def saving_steps(definition):
    """The steps whose result is written to disk."""
    for step in definition.get("steps", []):
        result = step.get("result") or {}
        if result.get("content_type") and result.get("save", True) is not False:
            yield step


TEMPLATES = [f for f in get_example_files() if f.startswith("workflows/templates/")]
IN_SCOPE = [f for f in TEMPLATES if len(list(saving_steps(load(f)))) >= 2]


def test_the_scope_is_what_the_design_counted():
    """Eighteen templates had two or more saving steps when the convention
    landed. A template added later with several saving steps joins the
    parametrized test below on its own; this pins that none has quietly
    left it (a step that stopped saving would drop a template from scope
    without failing anything else)."""
    assert len(IN_SCOPE) >= 18, IN_SCOPE


@pytest.mark.parametrize("template", IN_SCOPE)
def test_every_saving_step_of_a_multi_step_template_names_its_role(template):
    steps = list(saving_steps(load(template)))
    roles = {step["name"]: step["result"].get("subfolder") for step in steps}

    unmarked = sorted(name for name, role in roles.items() if role not in CONVENTION)
    assert not unmarked, (
        f"{template}: saving steps without a final/intermediate subfolder: {unmarked}"
    )
    assert "final" in roles.values(), f"{template}: no step is marked final"


@pytest.mark.parametrize("template", IN_SCOPE)
def test_the_subfolder_is_the_last_key_of_the_result(template):
    """One added line per step, and every template reads alike."""
    for step in saving_steps(load(template)):
        assert list(step["result"])[-1] == "subfolder", (
            f"{template}: step {step['name']!r} does not end its result with subfolder"
        )


@pytest.mark.parametrize(
    "builtin",
    sorted(name for name in os.listdir(BUILTIN_DIR) if name.endswith(".json")),
)
def test_the_packaged_builtins_stay_unmarked(builtin):
    with open(os.path.join(BUILTIN_DIR, builtin), encoding="utf-8") as file:
        definition = json.load(file)
    marked = [
        step["name"]
        for step in definition.get("steps", [])
        if "subfolder" in (step.get("result") or {})
    ]
    assert not marked, (
        f"dw/workflows/{builtin} marks {marked}; a role is the parent's to assign"
    )
```

- [ ] **Step 2: Run it to verify it fails**

Run: `source ./activate && python -m pytest tests/test_template_subfolders.py -q`
Expected: the 18 `test_every_saving_step_of_a_multi_step_template_names_its_role` cases FAIL with "saving steps without a final/intermediate subfolder"; the 18 last-key cases FAIL; scope and builtin cases PASS. If `test_the_scope_is_what_the_design_counted` fails, stop and report - the inventory below is wrong.

- [ ] **Step 3: Mark up the templates**

For each row, add `"subfolder": "<role>"` as the **last** key of that step's `result` object (after `embed_metadata`, `file_base_name`, whatever is there), with a comma on the previous last line. Nothing else in the file changes.

| Template (`workflows/templates/`) | Step | Role | Why |
|---|---|---|---|
| `compose-workflows.json` | `image_generation` | `intermediate` | the still the video is conditioned on |
| | `video_generation` | `final` | |
| `consistent-set.json` | `base` | `intermediate` | the subject picture the three edits start from |
| | `first`, `second`, `third` | `final` | the set |
| `controlnet.json` | `canny` | `intermediate` | control image |
| | `FluxCanny` | `final` | |
| `describe-and-regenerate.json` | `describe_image` | `intermediate` | text |
| | `augment_prompt` | `intermediate` | text |
| | `flux` | `final` | |
| `image-processors.json` | all 25 saving steps | `final` | a set of variants, each a deliverable |
| `lora-styles.json` | all 10 saving steps | `final` | a set of variants, each a deliverable |
| `recenter-crop.json` | `frame_1`, `frame_2`, `frame_3` | `final` | the registered stills are the output |
| `qr-code.json` | `qr_code` | `intermediate` | |
| | `init_image` | `intermediate` | |
| | `main` | `final` | |
| `restore-faces.json` | `generate` | `intermediate` | |
| | `restore` | `final` | |
| `segment-and-inpaint.json` | `segment` | `intermediate` | mask |
| | `inpaint` | `final` | |
| `sub-workflow.json` | `logo` | `intermediate` | the parent's copy of the child's return |
| | `remove_background` | `final` | |
| `ltx2/generative-upscale.json` | `low_resolution` | `intermediate` | |
| | `upscaled` | `final` | |
| `minimax/dialogue-short.json` | `draw_character_a`, `draw_character_b` | `intermediate` | portraits |
| | `shot` (the `for_each` step - one `result` block) | `intermediate` | the cuts |
| | `episode` | `final` | |
| `minimax/enhance-prompt.json` | `prompt_enhancer` | `intermediate` | text |
| | `text_to_video_audio` | `final` | |
| `minimax/enhance-prompt-with-image.json` | `prompt_enhancer` | `intermediate` | text |
| | `keyframe_to_video_audio` | `final` | |
| `minimax/generated-subject-reference.json` | `draw_subject` | `intermediate` | |
| | `reference_to_video_audio` | `final` | |
| `minimax/music-video.json` | `draw_singer` | `intermediate` | |
| | `write_song` | `intermediate` | the soundtrack is muxed into the cut |
| | `shot` (the `for_each` step) | `intermediate` | |
| | `music_video` | `final` | |
| `minimax/storyboard.json` | `board_1_launch`, `board_2_gutter`, `board_3_shore` | `intermediate` | boards |
| | `voyage` | `final` | |

Steps not in the table (e.g. `slice`, `soundtrack`, `edit` in `music-video.json`, which have no `content_type`) are not saving steps and get nothing. Three templates already contain a `"subfolder"` key under a component's `from_pretrained_arguments` (`community-pipeline.json`, `ltx2/generative-upscale.json`, `multi-image-reference.json`) - that is a Hugging Face repo subfolder, unrelated; leave it alone. Only `result` blocks change.

Example, `restore-faces.json` `restore` step, before:

```json
      "result": {
        "content_type": "image/jpeg"
      }
```

after:

```json
      "result": {
        "content_type": "image/jpeg",
        "subfolder": "final"
      }
```

- [ ] **Step 4: Run the new test and the neighbours**

Run: `source ./activate && python -m pytest tests/test_template_subfolders.py tests/test_examples.py tests/test_catalog_structure.py tests/test_catalog_shape.py -q`
Expected: all PASS. `test_examples.py` validates every template against the schema and `validation_errors`, so a mistyped subfolder or a broken comma fails there; `test_catalog_structure.py`/`test_catalog_shape.py` pin that shapes and traits did not move (the catalog reads `result.content_type` only).

- [ ] **Step 5: Run the whole suite**

Run: `source ./activate && python -m pytest -q -x`
Expected: PASS (5 skipped is normal).

- [ ] **Step 6: Commit**

```bash
git add tests/test_template_subfolders.py workflows/templates
git commit -m "feat(templates): every multi-step template names its deliverable

Every saving step of a template with two or more saving steps carries
result.subfolder - final for the output the user is shown, intermediate
for the scratch that went into it. tests/test_template_subfolders.py
asserts the rule so it cannot drift and pins the packaged builtins as
unmarked: a role is the parent's to assign.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: The plugin skills state the convention

**Files:**
- Modify: `plugins/dw/skills/minimax-h3/SKILL.md` (the *Run and judge* list, item 3)
- Modify: `plugins/dw/skills/ltx-2.5/SKILL.md` (the *Run and judge* list, item 3)
- Modify: `plugins/dw/skills/minimax-music3/SKILL.md` (the *Run and judge* list, item 3)
- Test: `tests/test_plugin_skills.py`

**Interfaces:**
- Consumes: `skill_text(path)`, `SKILLS` (the sorted list of every `SKILL.md`), `H3_SKILL`, `PLUGIN_DIR`, `REPO_ROOT`, already defined at module level in `tests/test_plugin_skills.py`; `pytest` and `os` are already imported there; add `import json` beside them (the role test uses it) rather than importing inline.
- Produces: nothing later tasks consume.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_plugin_skills.py`, after the last class:

```python
@pytest.mark.parametrize(
    "path", SKILLS, ids=lambda p: os.path.basename(os.path.dirname(p))
)
def test_a_skill_states_the_subfolder_convention(path):
    """The templates put the deliverable in `final` and the scratch in
    `intermediate`; an agent reading get_job needs to know that, and one
    composing a new workflow needs to keep it."""
    text = skill_text(path)
    assert "`subfolder`" in text, f"{path} does not name the subfolder field"
    assert "`final`" in text and "`intermediate`" in text, (
        f"{path} does not state the final/intermediate convention"
    )
    # the convention is stated where the manifest is read
    assert text.index("`subfolder`") > text.index("## Run and judge")


def test_the_h3_skill_names_the_intermediate_steps_the_templates_mark():
    """The skill says what the family's templates put in intermediate -
    portraits, boards, the song; if a template's roles change the skill
    must change with it."""
    text = skill_text(H3_SKILL)
    for name in ("dialogue-short", "music-video", "storyboard"):
        path = os.path.join(
            REPO_ROOT, "workflows", "templates", "minimax", name + ".json"
        )
        with open(path, encoding="utf-8") as f:
            spec = json.load(f)
        finals = [
            step["name"]
            for step in spec["steps"]
            if (step.get("result") or {}).get("subfolder") == "final"
        ]
        assert len(finals) == 1, (name, finals)
        assert f"`{finals[0]}`" in text, (
            f"the skill does not name {name}'s final step {finals[0]}"
        )
```

- [ ] **Step 2: Run them to verify they fail**

Run: `source ./activate && python -m pytest tests/test_plugin_skills.py -q -k "subfolder or intermediate_steps"`
Expected: 4 FAIL (three skills lack `` `subfolder` ``; the H3 role test fails on the first missing step name).

- [ ] **Step 3: Add the paragraph to each skill**

In each `SKILL.md`, the *Run and judge* list has an item 3 that begins with `` 3. `wait_for_job`, then `get_job` for the manifest. `` Extend **that item** (keep it item 3; append sentences to it, indented to the list's continuation indent of three spaces) with the family-specific text below. Wrap at the file's existing line width (about 80 columns).

`plugins/dw/skills/minimax-h3/SKILL.md`, appended to item 3. **This skill has ~199 bytes of headroom under `SKILL_SIZE_LIMIT` (12288; check `wc -c` before and after)**, so the paragraph is short and item 4's existing list of the four portrait/board templates is left as is:

```
   Each entry carries `subfolder`: `final` is the deliverable (`episode`,
   `music_video`, `voyage`), `intermediate` the scratch; keep that split in
   anything you compose.
```

If the file still exceeds the cap, trim words from this paragraph only (the six backticked literals must stay); do not touch other sections.

`plugins/dw/skills/ltx-2.5/SKILL.md`, appended to item 3:

```
   Each manifest entry carries `subfolder`: `templates/ltx2/generative-upscale`
   puts `upscaled` in `final` and `low_resolution` in `intermediate`, and
   `list_gallery(subfolder="final")` lists only deliverables. Keep the
   convention in anything you compose from a template: the step whose output
   the user will be shown is `final`, every other saving step `intermediate`.
```

`plugins/dw/skills/minimax-music3/SKILL.md`, appended to item 3:

```
   Each manifest entry carries `subfolder`: `templates/minimax/music-video`
   puts the cut in `final` and the song, the singer's portrait and each shot
   in `intermediate`, and `list_gallery(subfolder="final")` lists only
   deliverables. Keep the convention in anything you compose from a template:
   the step whose output the user will be shown is `final`, every other saving
   step `intermediate`.
```

Every catalog name quoted above already resolves (`test_every_catalog_name_a_skill_quotes_resolves` checks); do not quote any other template.

- [ ] **Step 4: Run the skill tests**

Run: `source ./activate && python -m pytest tests/test_plugin_skills.py -q`
Expected: all PASS - including `test_a_skill_has_a_triggering_description_under_the_size_cap`, which caps the skill's size; if it fails, tighten the paragraph rather than the cap.

- [ ] **Step 5: Commit**

```bash
git add plugins/dw/skills tests/test_plugin_skills.py
git commit -m "docs(skills): each family skill states the final/intermediate convention

Where the skill reads the manifest it now says what the family's templates
put in final and intermediate, that list_gallery(subfolder=) filters on it,
and that a composed workflow keeps the convention. Pinned against the
templates' actual roles.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Guide and CLAUDE.md, with the release note

**Files:**
- Modify: `docs/WORKFLOW_GUIDE.md` (section *Saying which output is the deliverable*, under *Authoring a workflow from an agent*)
- Modify: `CLAUDE.md` (the `**Result subfolders**` bullet under *Critical Gotchas*)

**Interfaces:** none.

- [ ] **Step 1: The guide says the templates follow the convention**

In `docs/WORKFLOW_GUIDE.md`, the paragraph in *Saying which output is the deliverable* that ends `Mark every saving step of a multi-step workflow; a one-step workflow needs nothing.` - append one sentence to that paragraph:

```
The shipped templates follow it: every template with two or more saving
steps marks each one, so a workflow copied from a template starts with the
roles in place.
```

- [ ] **Step 2: CLAUDE.md carries the release note**

In `CLAUDE.md`, the `**Result subfolders**` bullet currently ends `` `file_base_name` may not contain a separator - it is a name, not a path ``. Change that line to end `it is a name, not a path.` (period on the same line - a period at the start of the next line renders as "path . Every") and append to the bullet (same indentation):

```
  Every `workflows/templates/**` file with two or more saving steps
  marks each one `final`/`intermediate` (`tests/test_template_subfolders.py` pins the rule;
  `dw/workflows/` builtins stay unmarked - a role is the parent's to assign). That moved
  the templates' outputs into `<run>/final/` and `<run>/intermediate/`: an
  `output:<template>/latest/x` reference keeps resolving but stops advancing past the last
  pre-change run (`keep_output` is the stable form), and a seeded template's first run
  after the change regenerates rather than hitting the step cache (the key includes
  `result`). Those two, and a stray `subfolder` key becoming live, are the release-note
  items beside the `shots` list change
```

(Check the rendered bullet reads as prose; the other gotcha bullets end without a period, this one now ends its last sentence `...the `shots` list change` without one too.)

- [ ] **Step 3: Check the mirror rule**

The CLAUDE.md *Type System* bullet for `result.subfolder` and the guide's section already agree from stage 2; this task adds a fact to each. Read both once after editing and confirm neither contradicts the other.

- [ ] **Step 4: Commit**

```bash
git add docs/WORKFLOW_GUIDE.md CLAUDE.md
git commit -m "docs: the templates follow the subfolder convention; release-note items in CLAUDE.md

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: The design records stage 3

**Files:**
- Modify: `docs/proposals/output-folders.md` (line 3 status; *Steering consumers toward it* item 1)

**Interfaces:** none.

- [ ] **Step 1: Status line**

Change line 3 from

```
Status: **stages 1-2 (engine, server/MCP) implemented; stages 3-4 pending**. Written for MCP feedback ticket T016;
```

to

```
Status: **stages 1-3 (engine, server/MCP, steering) implemented; stage 4 (UI) pending**. Written for MCP feedback ticket T016;
```

- [ ] **Step 2: Correct the exemption in *Steering* item 1**

In the *Steering consumers toward it* section, item 1, replace the sentences

```
A template whose saving steps are
   all `workflow` steps over `builtin:` children (`compose-workflows`,
   `sub-workflow`) is exempt: the files come from the child's steps, and
   marking a builtin's step `final` would presume a role it does not have
   - the builtins in `dw/workflows/` stay unmarked.
```

with

```
A `workflow` step is marked
   like any other: its `result` block saves the parent's copy of the
   child's return value (*Sub-workflows*, above), so the role is real and
   the child's own steps never see it. The one exemption is the packaged
   builtins in `dw/workflows/`, which stay unmarked - a builtin is a step
   list a parent composes, and the role is the parent's to assign
   (*implemented*: the drift test pins both halves; the earlier draft
   exempted `compose-workflows` and `sub-workflow`, neither of which is a
   `builtin:` composition). One consequence: a `workflow` step's
   deliverable exists twice - the child's copy at the run root, the
   parent's in `final/` - so `list_gallery(subfolder="final")` shows only
   the parent's. The duplication is older than this design; it is now
   distinguishable.
```

- [ ] **Step 2b: The *Tests* bullet matches the ruling**

In the *Tests* section, the *Templates* bullet reads

```
- **Templates.** Every template with two or more saving steps, not exempt
  as a `builtin:` composition, has a subfolder on each saving step.
```

Replace it with

```
- **Templates.** Every template with two or more saving steps has `final`
  or `intermediate` on each saving step and at least one `final`; the
  packaged builtins in `dw/workflows/` carry none.
```

Keep the paragraph's three-space continuation indent and wrap at the file's line width.

- [ ] **Step 3: Commit**

```bash
git add docs/proposals/output-folders.md
git commit -m "docs: output-folders design marks stage 3 done; the builtin exemption is the builtins', not their parents'

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage.** *Steering* 1 (templates + drift test) → Task 1; 2 (guide + CLAUDE.md mirror) → Task 3; 3 (tool descriptions) done in stage 2, nothing here; 4 (plugin skills) → Task 2. *Breaking changes* 3 and 4 (release note) → Task 3 Step 2. *Tests / Templates* → Task 1. *Phasing* stage 3 side note (`item:subfolder` derives a list field) - not exercised: no template uses an `item:` subfolder, by Global Constraint. Design status → Task 4.

**Placeholders.** None: every edit is spelled out, every template step is in the table.

**Consistency.** `saving_steps` and the ≥2 rule are the same in the test module and the Global Constraints. The three skill paragraphs quote only step names the table marks `final` (`episode`, `music_video`, `voyage`, `upscaled`) and Task 2's role test checks exactly those templates have one `final` step. The `for_each` steps `shot` in both cut templates have a single `result` block, so one edit marks every member.
