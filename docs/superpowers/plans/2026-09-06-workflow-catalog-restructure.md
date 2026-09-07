# Workflow Catalog Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 121-workflow catalog into `workflows/templates/` (one file per capability) and `workflows/models/` (tuned per-checkpoint configs), deleting checkpoint iterations that teach nothing, and land with every reference in the repo resolving.

**Architecture:** A catalog entry today cannot say whether it teaches a pattern or records a hardware fact, so an agent handed an open-ended request cannot weight them. Two trees encode that distinction in the path itself. The work is: build the missing safety net first, agree an inventory before deleting anything, then move/collapse/prune, then prove consistency.

**Tech Stack:** Python 3.14, pytest, jsonschema. No new dependencies.

**Spec:** [docs/superpowers/specs/2026-09-06-workflow-catalog-restructure-design.md](../specs/2026-09-06-workflow-catalog-restructure-design.md)

## Global Constraints

- **A surviving workflow keeps its `id` verbatim.** The step cache is keyed on `id` (`dw/step_cache.py`), not path. Renaming ids silently invalidates the cache for every workflow at once. Moving and renaming *files* is fine; changing the `id` field inside them is not.
- **Backward compatibility is not maintained.** Existing outputs may be orphaned and `output:` references into them may break. Do not add an alias mechanism, migration shim, or old→new mapping document.
- **The acceptance criterion is internal consistency:** when this lands, every reference in the repo resolves.
- **Nothing is deleted before the inventory is approved** (Task 2 gate).
- **Reference classes and their enforcement** (counted 2026-09-06): 66 distinct markdown links into `workflows/` (enforced by `tests/test_docs_links.py`); 6 workflow→workflow `path` references (enforced by `tests/test_examples.py::test_example_workflow_references_resolve`); 18 distinct `prompt:` names across 20 files (enforced by nothing — Task 1 closes this). No workflow uses `asset:` or `output:`; the UI references no workflow paths.
- **Run the full suite before every commit:** `python -m pytest tests/ -q`. It is ~2 minutes and this work touches files that many tests walk.

---

### Task 1: Close the `prompt:` reference gap

Build the safety net before the demolition. 20 workflows carry `prompt:` references that nothing checks; collapsing rewrites exactly those files, and a broken `prompt:` fails only when the workflow is actually run.

**Files:**
- Create: `tests/test_prompt_references.py`

**Interfaces:**
- Consumes: `dw.prompts.resolve_prompt_reference(reference, prompt_dir=None, base_dir=None)` — returns the validated absolute path for a `"prompt:name"` or `"prompt:folder/name"` string, and **raises `ValueError` when no such prompt file exists** (`dw/prompts.py:76`). The test below catches that rather than asserting on a return value, so the failure names the workflow instead of surfacing as a bare ValueError. `dw.prompts.PROMPT_PREFIX` is the string `"prompt:"`.
- Consumes: `tests.test_examples.get_example_files()` — returns every workflow JSON path under `workflows/`, relative to the repo root, subfolders included.
- Produces: nothing later tasks import. This is a standalone guard.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prompt_references.py`:

```python
"""Every prompt: reference in the catalog names a prompt that exists.

The engine resolves a prompt: reference when the workflow runs, so a
reference to a prompt that was renamed or removed fails on the GPU rather
than in CI. This is the one reference class in the tree that nothing else
checks - sub-workflow paths, type names and constants all have their own
test in test_examples.py.
"""

import json
import os

import pytest

from dw.prompts import PROMPT_PREFIX, resolve_prompt_reference
from tests.test_examples import REPO_ROOT, get_example_files

PROMPT_DIR = os.path.join(REPO_ROOT, "prompts")


def prompt_references(definition):
    """Every "prompt:..." string anywhere in a workflow definition."""
    if isinstance(definition, dict):
        for value in definition.values():
            yield from prompt_references(value)
    elif isinstance(definition, list):
        for value in definition:
            yield from prompt_references(value)
    elif isinstance(definition, str) and definition.startswith(PROMPT_PREFIX):
        yield definition


@pytest.mark.parametrize("example_file", get_example_files())
def test_every_prompt_reference_resolves(example_file):
    path = os.path.join(REPO_ROOT, example_file)
    with open(path, encoding="utf-8") as file:
        definition = json.load(file)

    for reference in prompt_references(definition):
        # resolve_prompt_reference raises rather than returning a missing path,
        # and a bare ValueError would not say which workflow carried the
        # reference - the only thing that makes the failure actionable
        try:
            resolve_prompt_reference(reference, prompt_dir=PROMPT_DIR)
        except ValueError as e:
            pytest.fail(f"{example_file} references '{reference}': {e}")


def test_the_catalog_actually_uses_prompt_references():
    """Guards the guard: if the walk stops finding references, the test above
    passes vacuously and would keep passing through the restructure."""
    found = []
    for example_file in get_example_files():
        with open(os.path.join(REPO_ROOT, example_file), encoding="utf-8") as file:
            found.extend(prompt_references(json.load(file)))

    assert len(set(found)) >= 10
```

- [ ] **Step 2: Run it and confirm it passes on the current tree**

Run: `python -m pytest tests/test_prompt_references.py -q`
Expected: PASS. The catalog is consistent today — this test is a guard against the restructure breaking it, not a bug fix. If it FAILS, stop: there is a pre-existing broken reference to report before going further.

- [ ] **Step 3: Confirm the guard actually catches a break**

Temporarily rename a prompt the catalog uses and confirm the test fails:

```bash
REF=$(python -c "
import json, os
from tests.test_examples import REPO_ROOT, get_example_files
from tests.test_prompt_references import prompt_references
for f in get_example_files():
    refs = list(prompt_references(json.load(open(os.path.join(REPO_ROOT, f)))))
    if refs:
        print(refs[0].split(':', 1)[1]); break
")
mv "prompts/$REF.json" "prompts/$REF.json.bak"
python -m pytest tests/test_prompt_references.py -q   # expect FAIL
mv "prompts/$REF.json.bak" "prompts/$REF.json"
python -m pytest tests/test_prompt_references.py -q   # expect PASS
```

Expected: FAIL while renamed, PASS once restored. A guard never seen to fail is not a guard.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_prompt_references.py
git commit -m "Check that every prompt: reference in the catalog resolves

Sub-workflow paths, type names and constants each have a test that they
resolve; prompt: references had none, and they fail on the GPU rather
than in CI. Added ahead of the catalog restructure, which rewrites the
20 workflows carrying them."
```

---

### Task 2: The inventory, and the approval gate

The judgement call across all 121 workflows, written down before anything moves. This task produces a document and stops. **No file is moved, renamed or deleted in this task.**

**Files:**
- Create: `docs/superpowers/plans/2026-09-06-catalog-inventory.md`

**Interfaces:**
- Produces: the inventory document. Every later task consumes it as its file list. Its schema is fixed below so later tasks can be executed mechanically.

- [ ] **Step 1: Generate the raw list of every workflow**

```bash
find workflows -name "*.json" | sort > /tmp/catalog.txt
wc -l /tmp/catalog.txt   # expect 121
```

- [ ] **Step 2: Read every workflow and classify it**

Read each file. Do not classify from the filename — the filename is what made the catalog misleading in the first place. For each, record the verdict using the criteria from the spec:

- `TEMPLATE` — demonstrates a feature nothing else demonstrates: a shape (multi-shot cut sequence, chained conditioning, image-to-video), a mechanism (shared components, sub-workflows, `pipeline_reference`, `references`, `release_pipeline`/`release_models`), or a reference convention (`prompt:`, `previous_result:`).
- `MODEL` — carries settings that make a specific checkpoint fit real hardware: quantization, offload, component placement, truncation, cache.
- `COLLAPSE INTO <template>` — the same pattern as a template through a different checkpoint, where the pipeline class matches. Its distinguishing arguments become an argument set in that template's `description`.
- `DELETE` — a third checkpoint through an existing pattern, teaching nothing new.

- [ ] **Step 3: Write the inventory document**

Create `docs/superpowers/plans/2026-09-06-catalog-inventory.md` with this exact table schema — later tasks parse it by column:

```markdown
# Catalog inventory, 2026-09-06

Verdicts for every workflow under `workflows/`. Produced by Task 2 of the
[restructure plan](2026-09-06-workflow-catalog-restructure.md); the file lists
in Tasks 3-5 are this table, filtered by verdict.

Status: AWAITING APPROVAL

| Current path | Verdict | Destination | `id` | Why |
| --- | --- | --- | --- | --- |
| workflows/flux/FluxDev.json | COLLAPSE INTO | templates/text-to-image.json | FluxDev | One-step t2i; differs from Krea2/ZImage only by checkpoint |
| workflows/minimax/MiniMaxH3SitcomShort.json | TEMPLATE | templates/dialogue-short.json | MiniMaxH3SitcomShort | Only multi-shot cut sequence with per-shot references |
| workflows/archive/CogVideoX-5b.json | DELETE | — | — | Fourth CogVideoX variant; no feature the tree lacks |
```

Rules for the table:
- Every one of the 121 paths appears exactly once.
- `Destination` is a path relative to `workflows/`, or `—` for DELETE.
- The `id` column is the file's current `id` field, copied verbatim — it is what Task 3 and Task 4 must preserve.
- `Why` is one clause. If it takes a sentence, the verdict is probably wrong.

- [ ] **Step 4: Check the inventory against itself**

```bash
python - <<'PY'
import json, re, pathlib
rows = [l for l in pathlib.Path(
    "docs/superpowers/plans/2026-09-06-catalog-inventory.md"
).read_text().splitlines() if l.startswith("| workflows/")]
paths = [r.split("|")[1].strip() for r in rows]
on_disk = sorted(str(p) for p in pathlib.Path("workflows").rglob("*.json"))
print("rows:", len(rows), "on disk:", len(on_disk))
assert sorted(paths) == on_disk, set(on_disk) ^ set(paths)
dests = [r.split("|")[3].strip() for r in rows]
dupes = {d for d in dests if d != "—" and dests.count(d) > 1}
print("destinations reused by more than one row (expected for COLLAPSE):", dupes)
ids = [r.split("|")[4].strip() for r in rows]
for path, wid in zip(paths, ids):
    if wid == "—":
        continue
    actual = json.load(open(path)).get("id")
    assert wid == actual, f"{path}: inventory says id {wid!r}, file says {actual!r}"
print("every row's id matches the file")
PY
```

Expected: row count 121, matching path sets, and every `id` verified against the file it came from.

- [ ] **Step 5: Commit the inventory**

```bash
git add docs/superpowers/plans/2026-09-06-catalog-inventory.md
git commit -m "Inventory every workflow ahead of the catalog restructure

A verdict per file - template, model config, collapse target or delete -
with the id each surviving file must keep. Awaiting approval; nothing is
moved or deleted until it has it."
```

- [ ] **Step 6: STOP. Get the user's approval.**

Present the inventory with counts per verdict and the full DELETE list, and ask for approval. **Do not begin Task 3 until the user approves.** If they change verdicts, edit the table, re-run Step 4, amend the commit, and ask again. When approved, change `Status: AWAITING APPROVAL` to `Status: approved <date>` and commit that.

---

### Task 3: Build `workflows/templates/`

**Files:**
- Create: `workflows/templates/` (contents per the inventory)
- Create: `tests/test_catalog_structure.py`
- Modify: `dw/workflow_schema.json` (add the optional `configures` property)

**Interfaces:**
- Consumes: the approved inventory's `TEMPLATE` and `COLLAPSE INTO` rows.
- Produces: `workflows/templates/*.json`, each with a non-empty `description` and its original `id`. Task 4 relies on these paths existing to point `configures` at them.
- Produces: `tests/test_catalog_structure.py`, which Task 4 extends.

- [ ] **Step 1: Write the failing test**

Create `tests/test_catalog_structure.py`:

```python
"""The catalog's two trees, and the invariant each carries.

templates/ teaches a pattern and models/ records a hardware fact. The
distinction is only useful if it is legible from outside the file, so a
template must describe itself and a model config must name the template it
configures.
"""

import json
import os

import pytest

from tests.test_examples import REPO_ROOT

TEMPLATES_DIR = os.path.join(REPO_ROOT, "workflows", "templates")


def files_under(directory):
    if not os.path.isdir(directory):
        return []
    found = []
    for root, _, names in os.walk(directory):
        found.extend(
            os.path.join(root, name) for name in names if name.endswith(".json")
        )
    return sorted(found)


def test_there_are_templates():
    assert files_under(TEMPLATES_DIR)


@pytest.mark.parametrize("path", files_under(TEMPLATES_DIR))
def test_every_template_describes_itself(path):
    """A template is read before it is run - by a person choosing one and by an
    agent matching a request's shape. An undescribed template is invisible to
    both, whatever its filename says."""
    definition = json.load(open(path, encoding="utf-8"))

    assert definition.get("description", "").strip(), (
        f"{os.path.relpath(path, REPO_ROOT)} has no description"
    )
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_catalog_structure.py -q`
Expected: FAIL on `test_there_are_templates` — `workflows/templates/` does not exist yet.

- [ ] **Step 3: Add `configures` to the schema**

The workflow schema sets no top-level `additionalProperties`, so an unknown key already validates — but an undeclared field is undocumented. In `dw/workflow_schema.json`, add to the top-level `properties` object alongside `description`:

```json
"configures": {
    "type": "string",
    "description": "For a workflow under models/: the templates/ workflow this is a tuned per-checkpoint configuration of, as a catalog name such as 'templates/text-to-image'."
}
```

- [ ] **Step 4: Move and collapse, per the inventory**

For each `TEMPLATE` row, `git mv` the file to its destination. For each `COLLAPSE INTO` row, fold its distinguishing arguments into the destination template as a documented argument set in that template's `description`, then `git rm` the source.

Preserve each surviving file's `id` verbatim — Global Constraints. Where two workflows collapse into one, the destination keeps the `id` of whichever row's destination equals its own path.

Six workflows name another workflow by a relative `path`; when either end moves, fix the path in the same commit. `tests/test_examples.py::test_example_workflow_references_resolve` is what catches a miss.

- [ ] **Step 5: Run the structure and example tests**

Run: `python -m pytest tests/test_catalog_structure.py tests/test_examples.py tests/test_prompt_references.py -q`
Expected: all pass. A `prompt:` failure here means a collapse dropped a reference; an examples failure means a relative `path` was left pointing at a moved file.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all pass except `tests/test_docs_links.py`, which will fail until Task 6 updates the 66 markdown links. Note which links fail — that is Task 6's worklist.

- [ ] **Step 7: Commit**

```bash
git add -A workflows/ tests/test_catalog_structure.py dw/workflow_schema.json
git commit -m "Move the feature templates into workflows/templates/

One file per capability, with checkpoint variations collapsed into
argument sets in each template's description where the pipeline class
matched. Every surviving file keeps its id, so the step cache survives.

Doc links are updated in a later commit and test_docs_links fails until
then."
```

---

### Task 4: Build `workflows/models/`

**Files:**
- Create: `workflows/models/` (contents per the inventory)
- Modify: `tests/test_catalog_structure.py`

**Interfaces:**
- Consumes: the approved inventory's `MODEL` rows; the `workflows/templates/*.json` paths Task 3 produced.
- Produces: `workflows/models/*.json`, each carrying `configures` naming an existing template.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_catalog_structure.py`:

```python
MODELS_DIR = os.path.join(REPO_ROOT, "workflows", "models")


def test_there_are_model_configs():
    assert files_under(MODELS_DIR)


@pytest.mark.parametrize("path", files_under(MODELS_DIR))
def test_every_model_config_names_the_template_it_configures(path):
    """A model config is a tuned instance of a pattern. Without the pointer it
    is just another entry in the list, which is the problem this restructure
    exists to fix."""
    definition = json.load(open(path, encoding="utf-8"))
    configures = definition.get("configures", "")

    assert configures, (
        f"{os.path.relpath(path, REPO_ROOT)} has no 'configures'"
    )
    target = os.path.join(REPO_ROOT, "workflows", f"{configures}.json")
    assert os.path.isfile(target), (
        f"{os.path.relpath(path, REPO_ROOT)} configures '{configures}', "
        f"which is not a workflow ({target})"
    )
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/test_catalog_structure.py -q`
Expected: FAIL on `test_there_are_model_configs` — `workflows/models/` does not exist yet.

- [ ] **Step 3: Move the model configs and add `configures`**

For each `MODEL` row, `git mv` the file to its destination and add a top-level `configures` naming its template as a catalog name without the `.json` — for example `"configures": "templates/text-to-image"`. Preserve each `id` verbatim.

- [ ] **Step 4: Run the structure and example tests**

Run: `python -m pytest tests/test_catalog_structure.py tests/test_examples.py tests/test_prompt_references.py -q`
Expected: all pass.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all pass except `tests/test_docs_links.py` (Task 6).

- [ ] **Step 6: Commit**

```bash
git add -A workflows/ tests/test_catalog_structure.py
git commit -m "Move the per-checkpoint configurations into workflows/models/

Each names the template it configures, so a tuned 24GB config is findable
as an instance of a pattern rather than as another undifferentiated
catalog entry. Ids preserved."
```

---

### Task 5: Prune

Everything the inventory marks `DELETE`, plus what is left of `archive/`. Separate from Tasks 3 and 4 because a reviewer could reasonably approve the moves and reject a deletion.

**Files:**
- Delete: every `DELETE` row's path
- Delete: `workflows/archive/` entirely

**Interfaces:**
- Consumes: the approved inventory's `DELETE` rows.
- Produces: nothing. This task only removes.

- [ ] **Step 1: Confirm nothing surviving points at what is about to go**

```bash
python - <<'PY'
import pathlib, re
inv = pathlib.Path("docs/superpowers/plans/2026-09-06-catalog-inventory.md").read_text()
doomed = [l.split("|")[1].strip() for l in inv.splitlines()
          if l.startswith("| workflows/") and l.split("|")[2].strip() == "DELETE"]
print(len(doomed), "files to delete")
for path in doomed:
    stem = pathlib.Path(path).name
    hits = []
    for f in list(pathlib.Path("workflows").rglob("*.json")) + \
             list(pathlib.Path("docs").rglob("*.md")) + [pathlib.Path("README.md")]:
        if str(f) in doomed:
            continue
        if stem in f.read_text():
            hits.append(str(f))
    if hits:
        print(f"  STILL REFERENCED {path}: {hits}")
PY
```

Expected: no `STILL REFERENCED` lines. Any that appear must be resolved first — either the referencing file is updated, or the verdict was wrong and the inventory needs revisiting with the user.

- [ ] **Step 2: Delete**

```bash
# every DELETE row, then what remains of archive/
python - <<'PY'
import pathlib, subprocess
inv = pathlib.Path("docs/superpowers/plans/2026-09-06-catalog-inventory.md").read_text()
doomed = [l.split("|")[1].strip() for l in inv.splitlines()
          if l.startswith("| workflows/") and l.split("|")[2].strip() == "DELETE"]
for path in doomed:
    if pathlib.Path(path).exists():
        subprocess.run(["git", "rm", "-q", path], check=True)
PY
git rm -rq workflows/archive 2>/dev/null || true
find workflows -name "*.json" | wc -l   # the surviving count
```

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all pass except `tests/test_docs_links.py` (Task 6). If `test_examples.py` fails, a surviving workflow referenced a deleted one and Step 1 missed it.

- [ ] **Step 4: Commit**

```bash
git add -A workflows/
git commit -m "Delete the checkpoint iterations and empty archive/

Every removed file was a further checkpoint through a pattern the tree
already demonstrates. Git history is the archive, so archive/ stops being
a graveyard that costs listing space."
```

---

### Task 6: Fix every link and every description of the old layout

**Files:**
- Modify: `docs/*.md` (66 distinct workflow paths across them), `README.md`
- Modify: `CLAUDE.md` if it describes the old layout

**Interfaces:**
- Consumes: the new tree from Tasks 3–5.
- Produces: a repo whose markdown all resolves.

- [ ] **Step 1: List every broken link**

Run: `python -m pytest tests/test_docs_links.py -q 2>&1 | head -40`
Expected: failures naming each unresolvable path. This is the worklist.

- [ ] **Step 2: Repoint each link**

For each, find the file's new home in the inventory's `Destination` column and update the link. Where the target was deleted, remove the reference and the sentence around it if the sentence only existed to introduce it — a dangling half-sentence is worse than the missing link.

Check the prose too, not only the link targets: any text describing `workflows/flux/` or `archive/` as places, and any "Examples:" list naming a deleted file.

- [ ] **Step 3: Run the docs-link test**

Run: `python -m pytest tests/test_docs_links.py -q`
Expected: PASS.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all pass, with no exceptions this time.

- [ ] **Step 5: Commit**

```bash
git add -A docs/ README.md CLAUDE.md
git commit -m "Repoint documentation at the restructured catalog

All 66 workflow links, plus the prose describing the old folder layout
and the example lists naming deleted files."
```

---

### Task 7: Prove consistency and describe the new layout

The acceptance criterion, made checkable, plus the one doc that has to exist for the new structure to be legible.

**Files:**
- Modify: `docs/WORKFLOW_GUIDE.md` (a section on the two trees)
- Modify: `dw_mcp/guides.py` if the guide summary needs updating
- Modify: `docs/superpowers/plans/2026-09-06-catalog-inventory.md` (mark it done)

**Interfaces:**
- Consumes: everything above.
- Produces: the finished state.

- [ ] **Step 1: Run every reference class at once**

```bash
python -m pytest tests/test_docs_links.py tests/test_examples.py \
    tests/test_prompt_references.py tests/test_catalog_structure.py -q
```

Expected: all pass. These four are the whole acceptance criterion — markdown links, sub-workflow paths, type and constant references, `prompt:` references, and the two-tree invariants.

- [ ] **Step 2: Confirm no id changed**

```bash
python - <<'IDCHECK'
import json, pathlib
# The inventory is the record of what every id was before anything moved. Read
# it from the working tree - it was written in Task 2, so no earlier commit has
# it.
inv = pathlib.Path(
    "docs/superpowers/plans/2026-09-06-catalog-inventory.md"
).read_text()
rows = [l.split("|") for l in inv.splitlines() if l.startswith("| workflows/")]
# An id survives only where its file survives as itself: DELETE removes the
# file, and COLLAPSE folds it into another that keeps its own id
gone = {r[1].strip() for r in rows if r[2].strip() in ("DELETE", "COLLAPSE INTO")}
expected = {
    r[4].strip()
    for r in rows
    if r[1].strip() not in gone and r[4].strip() not in ("—", "")
}
on_disk = {json.load(open(p))["id"] for p in pathlib.Path("workflows").rglob("*.json")}
lost = sorted(expected - on_disk)
assert not lost, f"surviving ids missing from disk: {lost}"
print(f"all {len(expected)} surviving ids intact")
IDCHECK
```

Expected: the only absent ids belong to deleted rows. Any other absence means an `id` was changed, which invalidates the step cache — fix it before continuing.

- [ ] **Step 3: Document the two trees**

Add a section to `docs/WORKFLOW_GUIDE.md` explaining the split: `templates/` teaches a pattern and is what to copy; `models/` records what makes a checkpoint fit real hardware and names the template it configures. Say that a template carries its per-checkpoint argument sets in its own `description`, and that `configures` is how a model config points home.

Then check whether `dw_mcp/guides.py`'s one-line summary for `workflows` still describes the guide accurately, and update it if not.

- [ ] **Step 4: Mark the inventory done**

Change its status line to `Status: applied <date>`.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add -A docs/ dw_mcp/guides.py
git commit -m "Document the two-tree catalog and mark the inventory applied

templates/ teaches a pattern, models/ records what makes a checkpoint fit
and names the template it configures. Every reference class - markdown
links, sub-workflow paths, type and constant references, prompt:
references - resolves."
```

---

## What this deliberately leaves out

The derived shape index and measured runtimes. They belong on top of this, not
beside it: classifying a catalog about to be halved is wasted work, and the
derivation gets materially more accurate once "template or model config" is a
distinction it can read from the path rather than infer. See the spec's "What
this unblocks".
