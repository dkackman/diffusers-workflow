# Roles and Notes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- Every file a run writes carries a role, `final` or `intermediate`, and
  every run records the files it read as `input`s.
- Runs, run files, assets and workspaces can carry an optional plain-text
  note.
- Roles and notes are surfaced through the HTTP API, MCP and the web UI,
  and roles become the gallery's main way to navigate.

**Architecture:**
- **`dw/roles.py`** decides a step's role from:
  - its `result.role`;
  - its `result.subfolder` convention;
  - or the data flow, i.e. which later step consumes it and whether that
    consumer writes media.

  The engine writes the role into each manifest step entry when the run
  ends. The server infers it, with the same function, for runs that
  predate this change.
- **`dw/notes.py`** owns two stores:
  - a `notes.json` in a directory, which holds a run note and per-file
    notes, and serves both run directories and asset-library directories;
  - a `workspace.json` at a workspace root.

  The server writes a submitted run note into `notes.json` when the
  worker's `run_start` event names the run directory, so the engine never
  touches notes. One `PUT /api/notes` route edits any note. MCP gains
  `set_note` and a `note` parameter on `run_workflow`, `upload_asset` and
  `keep_output`.

**Tech Stack:**
- Python 3 (FastAPI, pydantic, pytest with xdist);
- Svelte 5 with runes and Vitest;
- openapi-typescript codegen;
- sqlite job history.

**Spec:** `docs/superpowers/specs/2026-10-12-roles-and-notes-design.md`

**Deviation from spec:** asset notes go in the asset directory's
`notes.json`, the same format as a run directory's, not in the
`manifest.json` sidecar. `record_kept_shots` deletes that sidecar whenever
its last shots or provenance entry goes, which would take the notes with
it. One format for both stores is also simpler. Task 5 updates the spec
table.

## Global Constraints

- Note: plain text. At most `4000` characters, otherwise HTTP 400. An
  empty or whitespace-only string, or null, clears the note. Never
  rendered as HTML or markdown, so never `{@html}` in the UI.
- Role values: `"final"`, `"intermediate"`, `"input"`. Only `final` and
  `intermediate` are valid in `result.role`.
- Sidecar file names:
  - `notes.json` in a run directory or an asset directory;
  - `workspace.json` at a workspace root.
- Every `dw/` module stays at or under 1,100 lines. `dw/workflow_run.py`
  is already at 1,013, so keep additions there minimal.
- The architecture ratchet (`scripts/arch_metrics.py --check
  docs/stabilization/baseline.json`) must pass. This plan adds exactly two
  modules (`dw/roles.py` and `dw/notes.py`), so `modules` goes from 208 to
  210. That rise needs Don's approval, and Task 1's commit names it
  (`modules 208 -> 210: dw/roles.py, dw/notes.py, roles-and-notes spec`).
  Add no other new Python module: routes go in existing route files, and
  MCP code goes in existing `dw_mcp` files.
- Every JSON route the UI calls declares a response model in
  `dw/server/api_models.py`, with `response_model_exclude_unset`. Keys
  sent only sometimes use `sometimes(...)`.
- After any API model change:
  - run `python scripts/dump_openapi.py`, under the `constraints-openapi.txt`
    pins;
  - run `cd ui && npm run gen:api`;
  - commit both generated files.

  `ui/src/lib/types.ts` only re-exports.
- A new JSON route that `ui/src/lib/api.ts` calls must be added to
  `tests/test_api_contract.py`'s `UI_READ_ROUTES`, or
  `tests/test_ui_twins.py` fails.
- Bits UI is imported only under `ui/src/lib/ui/`.
- Security:
  - every filesystem write lands on a path checked by
    `validate_output_path(path, root)` against its root, with a non-None
    root;
  - no `eval`, `exec` or `shell=True`;
  - targets are resolved through the existing validated resolvers.
- Run Python tests with `DW_DEVICE=cpu venv/bin/python -m pytest ...`. The
  local venv has torch.
- Never `git add -A` under `ui/`, because Don edits `ui/package.json`. Add
  files by name.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Work on branch `feat/roles-and-notes`, cut from `develop`.

## Review Focus

These failure modes have no spec line but are likely to bite. Each has a
test in the task that owns it.

1. **A note edited while its run is still going.** The engine never writes
   `notes.json` and the server serializes its own writes, so an edit
   during a run must survive the run's final manifest write. Pinned in
   Task 6: `test_note_set_during_run_survives_manifest_rewrite`.
2. **Deleting the last output of a run that has a note.** The run
   directory must still be pruned, and must not be left behind holding
   only `notes.json`. Pinned in Task 5: `test_run_with_only_notes_is_pruned_and_orphan`.
3. **Re-uploading an asset under the name of a deleted one.** It must not
   inherit the old note or provenance. Pinned in Task 6:
   `test_deleted_asset_leaves_no_note_or_provenance`.
4. **A `PUT /api/notes` target with traversal, an absolute path, or a
   `workspace:` name that does not exist.** Expect 400 or 404, with
   nothing written outside the root. Pinned in Task 6:
   `test_put_note_refuses_traversal_targets`.
5. **A run whose consumer step saved nothing** (`save: false`) between two
   media steps. The upstream step must still be `intermediate`. Pinned in
   Task 1: `test_unsaved_consumer_still_demotes`.

---

## Stage 1: Roles

### Task 1: `dw/roles.py` and the `result.role` schema field

**Files:**
- Create: `dw/roles.py`
- Modify: `dw/workflow_schema.json`. The result block's properties sit
  near `save` and `subfolder` (around line 1230). Add `role`.
- Modify: `docs/ARCHITECTURE.md`. Add one row for roles.
- Modify: `docs/stabilization/baseline.json`. `modules` goes from 208 to
  210. Task 5 adds the second module, so raise it once, here, and name
  both in the commit message.
- Test: `tests/test_roles.py`

**Interfaces:**
- Produces:
  - `FINAL = "final"`, `INTERMEDIATE = "intermediate"`, `INPUT = "input"`,
    `ROLES = (FINAL, INTERMEDIATE)`
  - `step_roles(steps: list[dict], manifest_steps: list[dict]) -> dict[str, str]`.
    `steps` is the *expanded* definition's `"steps"`. The result maps step
    name to role, for every step named in `steps` or in a manifest
    entry's `step`. A manifest entry carrying `parent_step` takes its
    parent step's role.
  - `annotate_roles(steps: list[dict], manifest_steps: list[dict]) -> list[dict]`
    returns copies of `manifest_steps`, each with `"role"` set. An entry
    whose `step` is unknown gets `FINAL`.
  - `run_inputs(steps: list[dict]) -> list[dict]` returns
    `[{"reference": "asset:x.png", "kind": "asset"}, {"reference": "output:...", "kind": "output"}]`,
    with duplicates removed and in first-use order. It is read from the
    expanded steps' arguments.
  - `entry_role(entry: dict) -> str | None` returns a manifest entry's
    recorded role, or None.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_roles.py
from dw.roles import FINAL, INTERMEDIATE, annotate_roles, run_inputs, step_roles


def _step(name, result=None, args=None):
    step = {"name": name, "task": {"command": "noop", "arguments": args or {}}}
    if result is not None:
        step["result"] = result
    return step


def _entry(step, *files, **extra):
    return {"step": step, "files": list(files), **extra}


def test_leaf_media_step_is_final_and_consumed_one_intermediate():
    steps = [
        _step("image"),
        _step("video", args={"image": "previous_result:image"}),
    ]
    manifest = [_entry("image", "a.png"), _entry("video", "b.mp4")]
    assert step_roles(steps, manifest) == {"image": INTERMEDIATE, "video": FINAL}


def test_property_reference_counts_as_consumption():
    steps = [
        _step("stems"),
        _step("mix", args={"audio": "previous_result:stems.vocals"}),
    ]
    manifest = [_entry("stems", "v.wav"), _entry("mix", "m.wav")]
    assert step_roles(steps, manifest)["stems"] == INTERMEDIATE


def test_text_only_consumer_does_not_demote():
    steps = [
        _step("video"),
        _step("check", args={"video": "previous_result:video"}),
    ]
    manifest = [_entry("video", "v.mp4"), _entry("check", "report.json")]
    roles = step_roles(steps, manifest)
    assert roles == {"video": FINAL, "check": FINAL}


def test_unsaved_consumer_still_demotes():
    steps = [
        _step("image"),
        _step("latent", args={"image": "previous_result:image"}),
        _step("video", args={"latent": "previous_result:latent"}),
    ]
    manifest = [_entry("image", "a.png"), _entry("video", "v.mp4")]
    roles = step_roles(steps, manifest)
    assert roles["image"] == INTERMEDIATE
    assert roles["latent"] == INTERMEDIATE
    assert roles["video"] == FINAL


def test_explicit_role_wins_over_inference_and_subfolder():
    steps = [
        _step("image", result={"role": "final", "subfolder": "intermediate"}),
        _step("video", args={"image": "previous_result:image"}),
    ]
    manifest = [_entry("image", "a.png"), _entry("video", "v.mp4")]
    assert step_roles(steps, manifest)["image"] == FINAL


def test_subfolder_convention_including_nested():
    steps = [
        _step("a", result={"subfolder": "intermediate/shots"}),
        _step("b", result={"subfolder": "final/act-1"}),
    ]
    manifest = [_entry("a", "x.png"), _entry("b", "y.png")]
    assert step_roles(steps, manifest) == {"a": INTERMEDIATE, "b": FINAL}


def test_other_subfolder_name_falls_through_to_inference():
    steps = [_step("a", result={"subfolder": "shots"})]
    assert step_roles(steps, [_entry("a", "x.png")]) == {"a": FINAL}


def test_from_previous_result_key_counts():
    steps = [
        _step("seg"),
        _step("inpaint", args={"mask": {"from_previous_result": "seg"}}),
    ]
    manifest = [_entry("seg", "m.png"), _entry("inpaint", "o.png")]
    assert step_roles(steps, manifest)["seg"] == INTERMEDIATE


def test_dotted_step_name_resolves_by_longest_match():
    steps = [
        _step("v1.0"),
        _step("up", args={"image": "previous_result:v1.0"}),
    ]
    manifest = [_entry("v1.0", "a.png"), _entry("up", "b.png")]
    assert step_roles(steps, manifest)["v1.0"] == INTERMEDIATE


def test_for_each_members_are_steps_of_their_own():
    steps = [
        _step("shot@0"),
        _step("shot@1"),
        _step(
            "join",
            args={"videos": ["previous_result:shot@0", "previous_result:shot@1"]},
        ),
    ]
    manifest = [
        _entry("shot@0", "0.mp4"),
        _entry("shot@1", "1.mp4"),
        _entry("join", "j.mp4"),
    ]
    roles = step_roles(steps, manifest)
    assert roles["shot@0"] == roles["shot@1"] == INTERMEDIATE
    assert roles["join"] == FINAL


def test_child_entries_take_parent_step_role():
    steps = [
        _step("child", args={}),
        _step("score", args={"video": "previous_result:child"}),
    ]
    manifest = [
        _entry("render", "r.mp4", parent_step="child"),
        _entry("score", "s.mp4"),
    ]
    annotated = annotate_roles(steps, manifest)
    assert annotated[0]["role"] == INTERMEDIATE
    assert annotated[1]["role"] == FINAL


def test_annotate_roles_copies_and_defaults_unknown_to_final():
    manifest = [_entry("ghost", "g.png")]
    annotated = annotate_roles([], manifest)
    assert annotated[0]["role"] == FINAL
    assert "role" not in manifest[0]


def test_run_inputs_dedups_in_first_use_order():
    steps = [
        _step("a", args={"image": "asset:cast/iris.png", "mask": "output:w/r/m.png"}),
        _step("b", args={"images": ["asset:cast/iris.png", "asset:bg.png"]}),
    ]
    assert run_inputs(steps) == [
        {"reference": "asset:cast/iris.png", "kind": "asset"},
        {"reference": "output:w/r/m.png", "kind": "output"},
        {"reference": "asset:bg.png", "kind": "asset"},
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `DW_DEVICE=cpu venv/bin/python -m pytest tests/test_roles.py -v -n0`
Expected: FAIL with `ModuleNotFoundError: No module named 'dw.roles'`.

- [ ] **Step 3: Write `dw/roles.py`**

```python
"""The role of each file a run wrote - final or intermediate - and the
inputs it read.

The gallery mixes a run's deliverable with the scratch files that went
into it. The 'final'/'intermediate' result subfolder convention separates
them for the catalog templates, but the engine treats no folder name
specially and a hand-built workflow rarely sets one. A role is decided per
step, first rule wins:

1. the step's `result.role` ('final' or 'intermediate');
2. its `result.subfolder`, when that is or starts with one of those names;
3. the data flow: a step a later step consumes (a previous_result
   reference) is intermediate when that consumer writes media or saves
   nothing at all, and final otherwise - a consumer that writes only text
   or JSON (a transcript, check_script, plan_cuts) reads a deliverable, it
   does not make one out of it.

A composed child's rolled-up manifest entries carry `parent_step` and take
that step's role. The engine records roles when a run ends; the server
infers them with the same function for a run that predates this. Inputs
are the asset: and output: references the expanded steps carry - a run's
'made from' list. This module names no model.
"""

from .references import ASSET, OUTPUT, is_ref, iter_previous_result_references
from .security import (
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
)
from .step_cache import reference_resolves_to

FINAL = "final"
INTERMEDIATE = "intermediate"
INPUT = "input"
ROLES = (FINAL, INTERMEDIATE)
ROLE_KEY = "role"

_MEDIA_EXTENSIONS = frozenset(
    ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS
)
_INPUT_KINDS = ((ASSET, "asset"), (OUTPUT, "output"))


def _declared_role(step):
    """Rules 1 and 2: the role a step's result block states, or None."""
    result = step.get("result")
    if not isinstance(result, dict):
        return None
    role = result.get(ROLE_KEY)
    if role in ROLES:
        return role
    subfolder = result.get("subfolder")
    if isinstance(subfolder, str):
        head = subfolder.split("/", 1)[0]
        if head in ROLES:
            return head
    return None


def _writes_only_text(files):
    """Whether a consumer saved files and none of them is media."""
    if not files:
        return False
    return not any(
        str(name).lower().endswith(tuple(_MEDIA_EXTENSIONS)) for name in files
    )


def _files_by_step(manifest_steps):
    files = {}
    for entry in manifest_steps or []:
        if not isinstance(entry, dict):
            continue
        # A child's entry is its composing step's output in the parent
        name = entry.get("parent_step") or entry.get("step")
        files.setdefault(name, []).extend(entry.get("files") or [])
    return files


def step_roles(steps, manifest_steps):
    """Each step's role, by name - see the module docstring for the rules."""
    steps = [s for s in steps or [] if isinstance(s, dict) and "name" in s]
    files = _files_by_step(manifest_steps)
    names = [s["name"] for s in steps]
    roles = {}
    for step in steps:
        declared = _declared_role(step)
        if declared:
            roles[step["name"]] = declared
    for consumer in steps:
        if _writes_only_text(files.get(consumer["name"])):
            continue
        for _path, reference, _via in iter_previous_result_references(
            consumer, descend_into_from=True
        ):
            if not isinstance(reference, str):
                continue
            target = max(
                (n for n in names if reference_resolves_to(reference, n)),
                key=len,
                default=None,
            )
            if target is not None and target not in roles:
                roles[target] = INTERMEDIATE
    for name in list(names) + list(files):
        roles.setdefault(name, FINAL)
    return roles


def annotate_roles(steps, manifest_steps):
    """Copies of `manifest_steps`, each with its step's `role`."""
    roles = step_roles(steps, manifest_steps)
    annotated = []
    for entry in manifest_steps or []:
        if not isinstance(entry, dict):
            annotated.append(entry)
            continue
        name = entry.get("parent_step") or entry.get("step")
        annotated.append({**entry, ROLE_KEY: roles.get(name, FINAL)})
    return annotated


def entry_role(entry):
    """A manifest entry's recorded role, or None when it predates roles."""
    role = entry.get(ROLE_KEY) if isinstance(entry, dict) else None
    return role if role in ROLES else None


def _walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def run_inputs(steps):
    """The asset: and output: references the steps read, in first-use order."""
    seen = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        for value in _walk_strings(step):
            for prefix, kind in _INPUT_KINDS:
                if is_ref(prefix, value) and value not in (
                    s["reference"] for s in seen
                ):
                    seen.append({"reference": value, "kind": kind})
    return seen
```

There are two judgment points here. In `step_roles`, a declared role is
never overridden by inference: that is the "explicit wins" rule. Steps
that are not declared get `INTERMEDIATE` from any consumer that is not
text-only.

`dw/references.py` is the only module allowed to spell a reference
prefix (`PREFIX_OWNERS` in `scripts/arch_metrics.py`). That is why this
module imports `ASSET` and `OUTPUT` and never writes `"asset:"`.

- [ ] **Step 4: Add `role` to the schema**

In `dw/workflow_schema.json`, next to `"save"` in the result block's
`properties`, add:

```json
"role": {
    "description": "What this step's files are to whoever looks at the run: 'final' (a deliverable) or 'intermediate' (made for a later step). Optional: without it a subfolder named 'final' or 'intermediate' decides, else a step a later media-writing step consumes is intermediate and every other step is final. May be a 'variable:' reference.",
    "anyOf": [
        {"enum": ["final", "intermediate"]},
        {"type": "string", "pattern": "^variable:"}
    ]
}
```

Check how `subfolder` declares its `variable:` allowance in the same
block, and mirror that exactly if it differs.

- [ ] **Step 5: Add a schema test**

Append to `tests/test_roles.py`:

```python
import pytest
from dw.validation import (
    validate_workflow_definition,
)  # use the name tests/test_schema*.py use


@pytest.mark.parametrize(
    "role,ok",
    [("final", True), ("intermediate", True), ("input", False), ("keeper", False)],
)
def test_schema_role_values(role, ok):
    workflow = {
        "id": "t",
        "steps": [
            {
                "name": "s",
                "task": {"command": "noop", "arguments": {}},
                "result": {"role": role},
            }
        ],
    }
    errors = validate_workflow_definition(workflow)
    assert (not errors) == ok
```

Before writing this, `grep -rn "def validate" dw/validation.py`, and use
the schema-validation entry point that the existing schema tests import.
Adapt the call and the shape of its return value to match.

- [ ] **Step 6: Run the tests**

Run: `DW_DEVICE=cpu venv/bin/python -m pytest tests/test_roles.py -v -n0`
Expected: all PASS.

- [ ] **Step 7: Add the architecture map row and raise the baseline**

In `docs/ARCHITECTURE.md`, add a row in the table that holds the run and
manifest rows, using the same column shape:

`| A file's role (final / intermediate / input) | dw/roles.py | Decided per step: result.role, then a final/intermediate subfolder, then data flow (a step a media-writing or unsaved consumer reads is intermediate). Engine records it at run end; the server infers it identically for older runs. | tests/test_roles.py |`

Set `"modules": 210` in `docs/stabilization/baseline.json`.

Run: `venv/bin/python scripts/arch_metrics.py --check docs/stabilization/baseline.json`
Expected: pass. The current count is 209 with `dw/roles.py`, which is
under 210.

Run: `venv/bin/python -m pytest tests/test_architecture_map.py -n0 -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add dw/roles.py dw/workflow_schema.json tests/test_roles.py docs/ARCHITECTURE.md docs/stabilization/baseline.json
git commit -m "feat(roles): decide each step's role - final, intermediate - and a run's inputs

modules 208 -> 210: dw/roles.py (this commit) and dw/notes.py (notes
stage), per the roles-and-notes spec, approved by Don.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2: The engine records roles and inputs in the manifest

**Files:**
- Modify: `dw/workflow_run.py`:
  - `RunRecord`, around line 95: add a field;
  - `open_run`, line 616: keep the expanded steps;
  - `write_run_manifest`, line 209: annotate.
- Test: `tests/test_run_manifest_roles.py`

**Interfaces:**
- Consumes: `annotate_roles(steps, manifest_steps)` and `run_inputs(steps)`
  from Task 1.
- Produces: a terminal `manifest.json` has:
  - `steps[i]["role"]` on every entry;
  - a top-level `"inputs": [{"reference", "kind"}]`.

  A `"running"` manifest carries `inputs` but no roles.

- [ ] **Step 1: Write the failing test**

Find an existing test that runs a small workflow end to end on the CPU
and reads its `manifest.json`: `grep -rln "manifest.json" tests/ | head`,
and prefer the one using noop or utility tasks. Copy its fixture pattern.
The test:

```python
# tests/test_run_manifest_roles.py
import json, os
# reuse the fixture/helper the chosen existing test uses to run a workflow
# into a tmp output dir and return the run directory


def test_terminal_manifest_records_roles_and_inputs(
    run_workflow_into, tmp_path, asset_png
):
    workflow = {
        "id": "roles-e2e",
        "steps": [
            # two utility steps that write images, the second reading the first,
            # e.g. an image-processor task over an asset - pick tasks the chosen
            # fixture's workflows already use so no model is downloaded
        ],
    }
    run_dir = run_workflow_into(workflow)
    manifest = json.load(open(os.path.join(run_dir, "manifest.json")))
    roles = {e["step"]: e["role"] for e in manifest["steps"]}
    assert roles == {"first": "intermediate", "second": "final"}
    assert manifest["inputs"] == [{"reference": f"asset:{asset_png}", "kind": "asset"}]
```

Fill the step list with two real, model-free image tasks from
`docs/TASKS.md`, for example the resize or crop tasks the
image-processors template uses. Check that the chosen fixture can serve
an `asset:` file. If no end-to-end fixture fits, unit-test
`write_run_manifest` directly instead:
- construct a `RunRecord` and a minimal stand-in `workflow` with
  `manifest`, `_run_dir`, `_run_version`, `_elided_steps`, `name` and
  `file_spec`;
- set `record.expanded_steps`;
- assert the written file.

- [ ] **Step 2: Run it to verify it fails**

Run: `DW_DEVICE=cpu venv/bin/python -m pytest tests/test_run_manifest_roles.py -v -n0`
Expected: FAIL with `KeyError: 'role'`.

- [ ] **Step 3: Implement it**

In `dw/workflow_run.py`:

```python
from .roles import annotate_roles, run_inputs
```

Add to `RunRecord`, after `annotations`:

```python
    # The expanded steps the run executed - what roles and inputs are read from
    expanded_steps: list = field(default_factory=list)
```

In `open_run`, at the top, after `record.run_id = None`:

```python
    record.expanded_steps = prepared.workflow_def.get("steps") or []
```

In `write_run_manifest`, replace the `"steps": [...]` value with a local
computed before the `write_manifest(...)` call, and add `"inputs"`:

```python
    steps = [
        {
            **entry,
            "files": manifest_relative_files(entry.get("files"), workflow._run_dir),
            **_relative_shots(entry, workflow._run_dir),
        }
        for entry in workflow.manifest
    ]
    # Roles once the run has ended - a running manifest would show the
    # newest step as final until its consumer runs (dw/roles.py)
    if status != "running":
        steps = annotate_roles(record.expanded_steps, steps)
```

Then use `"steps": steps,` and add `"inputs": run_inputs(record.expanded_steps),`
after `"arguments"`.

Keep the net addition to roughly 12 lines.

Run: `wc -l dw/workflow_run.py`
Expected: at or under 1,030.

- [ ] **Step 4: Run the test and the run and manifest suites**

Run: `DW_DEVICE=cpu venv/bin/python -m pytest tests/test_run_manifest_roles.py tests/ -k "manifest or run_dir or runs" -q`
Expected: PASS. If an existing test compares the full manifest dict for
equality, update its expectation to include `role` and `inputs`. Those
fields are the intended change.

- [ ] **Step 5: Commit**

```bash
git add dw/workflow_run.py tests/test_run_manifest_roles.py
git commit -m "feat(roles): the engine records each step's role and the run's inputs in its manifest

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Also `git add` any existing test files whose expectations you updated,
naming each one.

### Task 3: Server and MCP surface roles

**Files:**
- Modify: `dw/server/routes/gallery.py`:
  - `_gallery_entries`, line 57: add `role`;
  - `GET /api/gallery`, around line 163: add the `role` filter.
- Modify: `dw/server/routes/media.py`. In `GET /api/gallery/{name}/metadata`,
  around line 69, add `role` and `inputs`.
- Modify: `dw/server/api_models.py`:
  - `GalleryFile`, line 781;
  - `GalleryMetadata`, line 843.
- Modify: `dw_mcp/catalog.py`:
  - `list_gallery` logic, around line 229, and its compact projection;
  - `get_gallery_metadata`, line 285.
- Modify: `dw_mcp/tools_catalog.py`. `list_gallery` at line 255 gains a
  `role` parameter.
- Modify: `ui/src/lib/generated/openapi.json` and
  `ui/src/lib/generated/api-schema.ts`, regenerated.
- Modify: `docs/SERVER.md` and `docs/MCP.md`, gallery sections, one
  paragraph each.
- Test: `tests/test_gallery_roles.py`

**Interfaces:**
- Consumes: `annotate_roles`, `entry_role` and `ROLES` from Task 1, and
  the manifests Task 2 writes.
- Produces:
  - `run_roles(run_dir: str) -> dict[str, str]`, a module-level function
    in `dw/server/routes/gallery.py`. It maps a file's path relative to
    the run directory, `/`-separated, to its role. It reads
    `manifest.json` and uses recorded roles where they are present.
    Otherwise it expands the realized `workflow.json` and calls
    `annotate_roles`. If neither file is readable it returns `{}`. Cache
    it in a module dict keyed by `(run_dir, manifest mtime)`, bounded at
    512 entries and evicting oldest first.
  - `GalleryFile.role: Literal["final","intermediate"] | None`
  - `GalleryMetadata.role` (the same type), and
    `GalleryMetadata.inputs: list[GalleryInput]` with
    `GalleryInput{reference: str, kind: Literal["asset","output"]}`, both
    via `sometimes()`.
  - `GET /api/gallery?role=final&role=intermediate`, a repeatable query.
    Leaving it out returns everything. A file with `role` None counts as
    final for filtering.
  - MCP `list_gallery(..., role: list[str] | None = None)`. The compact
    projection gains `role`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gallery_roles.py
import json, os
# use the TestClient / app fixture tests/test_server.py uses for /api/gallery
# (grep "api/gallery" tests/test_server.py for the fixture names)


def _write_run(root, identity, run_id, steps, realized=None, files=()):
    run_dir = os.path.join(root, identity, run_id)
    os.makedirs(run_dir, exist_ok=True)
    for f in files:
        path = os.path.join(run_dir, f)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").write(b"\x89PNG\r\n\x1a\n")
    json.dump(
        {"run_id": run_id, "version": 1, "status": "succeeded", "steps": steps},
        open(os.path.join(run_dir, "manifest.json"), "w"),
    )
    if realized is not None:
        json.dump(realized, open(os.path.join(run_dir, "workflow.json"), "w"))
    return run_dir


def test_recorded_roles_reach_the_listing(client, output_root):
    _write_run(
        output_root,
        "wf",
        "20261012-000000-aaaa",
        [
            {"step": "a", "files": ["a.png"], "role": "intermediate"},
            {"step": "b", "files": ["b.png"], "role": "final"},
        ],
        files=["a.png", "b.png"],
    )
    files = {f["label"]: f for f in client.get("/api/gallery").json()["files"]}
    assert files["a.png"]["role"] == "intermediate"
    assert files["b.png"]["role"] == "final"


def test_older_run_roles_inferred_from_realized_workflow(client, output_root):
    realized = {
        "id": "wf",
        "steps": [
            {"name": "a", "task": {"command": "noop", "arguments": {}}},
            {
                "name": "b",
                "task": {
                    "command": "noop",
                    "arguments": {"image": "previous_result:a"},
                },
            },
        ],
    }
    _write_run(
        output_root,
        "wf",
        "20261012-000001-bbbb",
        [{"step": "a", "files": ["a.png"]}, {"step": "b", "files": ["b.png"]}],
        realized=realized,
        files=["a.png", "b.png"],
    )
    files = {f["label"]: f for f in client.get("/api/gallery").json()["files"]}
    assert files["a.png"]["role"] == "intermediate"
    assert files["b.png"]["role"] == "final"


def test_role_filter(client, output_root):
    _write_run(
        output_root,
        "wf",
        "20261012-000002-cccc",
        [
            {"step": "a", "files": ["a.png"], "role": "intermediate"},
            {"step": "b", "files": ["final/b.png"], "role": "final"},
        ],
        files=["a.png", "final/b.png"],
    )
    finals = client.get("/api/gallery", params={"role": "final"}).json()["files"]
    assert [f["label"] for f in finals] == ["b.png"]


def test_flat_file_has_null_role_and_passes_final_filter(client, output_root):
    os.makedirs(os.path.join(output_root, "loose"), exist_ok=True)
    open(os.path.join(output_root, "loose", "x.png"), "wb").write(b"\x89PNG\r\n\x1a\n")
    entries = client.get("/api/gallery", params={"role": "final"}).json()["files"]
    loose = [f for f in entries if f["label"] == "x.png"]
    assert loose and loose[0].get("role") is None


def test_metadata_carries_role_and_inputs(client, output_root):
    run_dir = _write_run(
        output_root,
        "wf",
        "20261012-000003-dddd",
        [{"step": "b", "files": ["b.png"], "role": "final"}],
        files=["b.png"],
    )
    manifest_path = os.path.join(run_dir, "manifest.json")
    manifest = json.load(open(manifest_path))
    manifest["inputs"] = [{"reference": "asset:iris.png", "kind": "asset"}]
    json.dump(manifest, open(manifest_path, "w"))
    meta = client.get("/api/gallery/wf/20261012-000003-dddd/b.png/metadata").json()
    assert meta["role"] == "final"
    assert meta["inputs"] == [{"reference": "asset:iris.png", "kind": "asset"}]
```

Match the run-id format to `is_run_id` in `dw/runs.py`, and match the
fixture names (`client`, `output_root`) to what `tests/test_server.py`
already uses, renaming where needed. Add an MCP test in the existing MCP
catalog test file (`grep -ln "list_gallery" tests/`) asserting that
`role` appears in the compact projection and that `role=["final"]` is
forwarded as a query parameter.

- [ ] **Step 2: Run them to verify they fail**

Run: `DW_DEVICE=cpu venv/bin/python -m pytest tests/test_gallery_roles.py -v -n0`
Expected: FAIL, because `role` is missing from the entries.

- [ ] **Step 3: Implement `run_roles` and the listing**

In `dw/server/routes/gallery.py`:

```python
from ...roles import ROLES, annotate_roles, entry_role
from ...runs import _read_manifest  # already used nearby? reuse the existing import

_ROLE_CACHE = {}
_ROLE_CACHE_LIMIT = 512


def _realized_steps(run_dir):
    """The realized workflow's expanded steps, or None."""
    path = os.path.join(run_dir, REALIZED_FILE_NAME)
    try:
        with open(path) as handle:
            definition = json.load(handle)
        return Workflow(definition, None, path).expanded_definition().get("steps") or []
    except Exception:  # an unreadable or unexpandable record leaves roles unknown
        return None


def run_roles(run_dir):
    """Each file of a run, relative to the run directory, mapped to its role."""
    manifest_path = os.path.join(run_dir, MANIFEST_FILE_NAME)
    try:
        key = (run_dir, os.stat(manifest_path).st_mtime_ns)
    except OSError:
        return {}
    if key in _ROLE_CACHE:
        return _ROLE_CACHE[key]
    manifest = _read_manifest(run_dir) or {}
    steps = [e for e in manifest.get("steps") or [] if isinstance(e, dict)]
    if steps and any(entry_role(e) is None for e in steps):
        expanded = _realized_steps(run_dir)
        if expanded is not None:
            steps = annotate_roles(expanded, steps)
    roles = {}
    for entry in steps:
        role = entry_role(entry)
        if role:
            for name in entry.get("files") or []:
                roles[str(name).replace(os.sep, "/")] = role
    if len(_ROLE_CACHE) >= _ROLE_CACHE_LIMIT:
        _ROLE_CACHE.pop(next(iter(_ROLE_CACHE)))
    _ROLE_CACHE[key] = roles
    return roles
```

`Workflow` comes from `...workflow`. Check that
`Workflow(definition, output_dir=None, file_spec=path)` constructs
without touching the filesystem, as `dw/server/routes/library.py:303`
does, and copy that call's arguments.

In `_gallery_entries`, add `role` to each entry. Compute the run
directory as `os.path.join(root, folder, run_id)` when `run_id` is set.
The relative name inside it is `relative_name` with the
`f"{folder}/{run_id}/"` prefix removed. Then:

```python
entry["role"] = (
    run_roles(os.path.join(root, folder, run_id)).get(inner_name) if run_id else None
)
```

Add a `role: list[str] | None = Query(None)` parameter to the gallery
route. After building the entries, filter:

```python
    if role:
        wanted = set(role)
        files = [f for f in files if (f.get("role") or "final") in wanted]
```

Refuse any value outside `ROLES` with a 400.

In `api_models.py`:
- `GalleryFile` gains `role: Literal["final", "intermediate"] | None = sometimes("...")`;
- add the `GalleryInput` model;
- `GalleryMetadata` gains `role` and `inputs`.

In `media.py`'s metadata route, for an output name (not `asset:`):
- set `role` from `run_roles(run_dir).get(inner_name)`;
- set `inputs` from the manifest's `inputs` when present.

Reuse the route's existing run-directory resolution. It already finds
the manifest for `job`, `shots` and so on.

- [ ] **Step 4: MCP**

In `dw_mcp/tools_catalog.py`, `list_gallery` gains:

```python
role: Annotated[
    list[Literal["final", "intermediate"]] | None,
    Field(
        description="Only files of these roles. A deliverable is 'final', a file a later step consumed is 'intermediate'. Omit for every file - intermediates matter when diagnosing a run."
    ),
] = (None,)
```

Forward it as the repeated `role` query parameter in `dw_mcp/catalog.py`,
and add `"role"` to the gallery compact projection tuple. Match the
`Annotated` and `Field` style the neighbouring parameters use.

- [ ] **Step 5: Regenerate the contract and run the tests**

```bash
venv/bin/python scripts/dump_openapi.py
(cd ui && npm run gen:api)
DW_DEVICE=cpu venv/bin/python -m pytest tests/test_gallery_roles.py tests/test_server.py tests/test_api_contract.py tests/test_ui_twins.py -q
venv/bin/python scripts/surface_snapshot.py --check || venv/bin/python scripts/surface_snapshot.py
```

Expected: PASS. Run `scripts/surface_snapshot.py --help` first to learn
its update flag, then update the snapshot.

- [ ] **Step 6: Docs**

`docs/SERVER.md`, gallery section:

> Each entry carries `role`, `final` or `intermediate` (null for a flat-layout file with no run), and `GET /api/gallery?role=final` lists only those. See `dw/roles.py` for how a role is decided.

`docs/MCP.md`, `list_gallery`: the same sentence, plus "the `role` filter
defaults to every file".

- [ ] **Step 7: Commit**

```bash
git add dw/server/routes/gallery.py dw/server/routes/media.py dw/server/api_models.py dw_mcp/catalog.py dw_mcp/tools_catalog.py tests/test_gallery_roles.py ui/src/lib/generated/openapi.json ui/src/lib/generated/api-schema.ts docs/SERVER.md docs/MCP.md
git commit -m "feat(roles): gallery entries and metadata carry role and inputs; role filter on the listing and list_gallery

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Also add the surface snapshot file and any MCP test file you touched, by
name.

### Task 4: UI roles

**Files:**
- Create: `ui/src/lib/gallery/RoleTag.svelte`
- Modify: `ui/src/lib/pages/GalleryPage.svelte`:
  - the `subfolder` state at line 22 is replaced by role chips;
  - the `subfolders` derivation at line 53 is removed from the top bar.
- Modify: the gallery tile component. `grep -n "label" ui/src/lib/pages/GalleryPage.svelte`
  finds where tiles render. Add the `RoleTag` ahead of the label.
- Modify: `ui/src/lib/gallery/GalleryDetail.svelte`, the meta block at
  lines 173-225. The order becomes role tag, (the note slot comes in
  Task 8), Settings collapsed, Made from, and Location last.
- Modify: `ui/src/lib/pages/JobPage.svelte`. Group results as Final
  (expanded), then Intermediate (collapsed when there are finals), then
  Made from.
- Modify: `ui/src/lib/assets/AssetDetail.svelte`. Add a "kept from
  <workflow> v<n>" line when the metadata's `job` or provenance is
  present.
- Modify: `ui/src/lib/api.ts`. `gallery()` gains an optional
  `roles?: string[]` that it serializes as repeated `role` parameters.
- Test: `ui/src/lib/gallery/RoleTag.test.ts`, `ui/src/lib/pages/GalleryPage.test.ts`
  (extend it if it exists, otherwise create it), and `ui/src/lib/pages/JobPage.test.ts`
  (extend it).

**Interfaces:**
- Consumes: `GalleryFile.role`, `GalleryMetadata.role` and
  `GalleryMetadata.inputs`, from the regenerated `api-schema.ts`, through
  `types.ts` re-exports. Add the re-exports for `GalleryInput`.
- Produces: `RoleTag` with props `{ role: 'final' | 'intermediate' | 'input' | null }`.
  It renders nothing for null. Task 8 places notes next to it.

- [ ] **Step 1: Write the failing component test**

```ts
// ui/src/lib/gallery/RoleTag.test.ts
import { render, screen } from '@testing-library/svelte'
import { describe, expect, it } from 'vitest'
import RoleTag from './RoleTag.svelte'

describe('RoleTag', () => {
  it.each([
    ['final', 'Final'],
    ['intermediate', 'Intermediate'],
    ['input', 'Input'],
  ])('renders %s', (role, text) => {
    render(RoleTag, { role })
    const tag = screen.getByText(text)
    expect(tag.dataset.role).toBe(role)
  })

  it('renders nothing for a null role', () => {
    const { container } = render(RoleTag, { role: null })
    expect(container.textContent?.trim()).toBe('')
  })
})
```

Check the existing component tests for the render helper this repo uses
(`grep -rln "@testing-library/svelte" ui/src | head -3`), and match their
imports.

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ui && npx vitest run src/lib/gallery/RoleTag.test.ts`
Expected: FAIL, because the module is not found.

- [ ] **Step 3: Implement `RoleTag.svelte`**

```svelte
<script lang="ts">
  let { role }: { role: 'final' | 'intermediate' | 'input' | null | undefined } = $props()
  const text = { final: 'Final', intermediate: 'Intermediate', input: 'Input' }
</script>

{#if role}
  <span class="role-tag" data-role={role}>{text[role]}</span>
{/if}

<style>
  .role-tag {
    display: inline-block;
    padding: 0.05rem 0.4rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
    line-height: 1.4;
    white-space: nowrap;
  }
  .role-tag[data-role='final'] {
    background: var(--accent);
    color: var(--accent-contrast, #fff);
  }
  .role-tag[data-role='intermediate'] {
    background: var(--surface-2);
    color: var(--text-muted);
  }
  .role-tag[data-role='input'] {
    background: transparent;
    color: var(--text-muted);
    border: 1px solid var(--border);
  }
</style>
```

Use the colour tokens `ui/src/app.css` actually defines. Grep it for the
accent, muted-text, surface and border variable names, and substitute
them for the ones above.

- [ ] **Step 4: Gallery page tests, then implementation**

Add these tests to the GalleryPage test, mocking `api.gallery` the way
existing page tests mock api calls:

```ts
it('opens on finals only and shows each tile role', async () => {
  // mock gallery() to return one final and one intermediate entry
  // render GalleryPage; expect the Final chip pressed, Intermediate not
  // expect only the final tile rendered, carrying a RoleTag "Final"
})

it('toggling Intermediate shows intermediates too', async () => {
  // click the Intermediate chip; expect both tiles
})

it('has no subfolder filter in the top bar', async () => {
  // expect no element with the subfolder filter's label/test id
})
```

Write these out as real assertions against the mock data, following the
nearest existing page test for its mocking and querying idiom.

Implementation in `GalleryPage.svelte`:

```ts
let roles = $state<Set<'final' | 'intermediate'>>(new Set(['final']))
const visible = $derived(
  files.filter((f) => roles.has((f.role ?? 'final') as 'final' | 'intermediate')),
)
function toggleRole(role: 'final' | 'intermediate') {
  const next = new Set(roles)
  next.has(role) ? next.delete(role) : next.add(role)
  roles = next.size ? next : new Set([role === 'final' ? 'intermediate' : 'final'])
}
```

The chips are two `<button aria-pressed={roles.has(r)}>` elements in the
top bar beside the workflow filter. Filter on the client: the listing is
already fully loaded, so this needs no extra fetch. The `role` parameter
on `api.gallery` is for MCP parity and future paging. Inside a run, sort
finals before intermediates and keep the existing mtime order within
each. Remove the subfolder `<select>`, and its `subfolder` state, from
the top bar.

- [ ] **Step 5: Detail panel, job page and asset detail**

In `GalleryDetail.svelte`, put `<RoleTag role={meta.role} />` at the top
of the meta block. Then:
- a `<details><summary>Settings</summary>` holding the run's arguments
  and seed, which the existing metadata already carries, so move that
  block inside;
- "Made from": `meta.inputs`, each rendered as a link that opens the
  asset or output detail the way the existing reference links do;
- a small "Location" line showing the file's folder path.

In `JobPage.svelte`, group the result files by `role`, from the gallery
entries or the job manifest's step `role`, into three sections:
- Final first and expanded;
- Intermediate collapsed, using `<details>` with `open={finals.length === 0}`;
- Made from, from the job's run inputs. Read them from
  `galleryMetadata` of any one output, or from the manifest when the job
  detail carries it.

A test covers the grouping order and the collapsed intermediate section.

In `AssetDetail.svelte`, when the asset metadata carries `job` (the
`JobRef` that `keep_output` provenance fills in), render
`kept from {job.workflow} v{job.version}`. Use whichever fields `JobRef`
has: read `api_models.py:825`.

- [ ] **Step 6: Run the UI checks**

```bash
cd ui && npx vitest run && npm run lint && npm run check && npm run metrics -- --check ../docs/stabilization/ui/baseline.json
```

Expected: all pass. If `complex_functions` rises, split the function.
Don't raise the baseline.

- [ ] **Step 7: Commit**

```bash
git add ui/src/lib/gallery/RoleTag.svelte ui/src/lib/gallery/RoleTag.test.ts ui/src/lib/pages/GalleryPage.svelte ui/src/lib/gallery/GalleryDetail.svelte ui/src/lib/pages/JobPage.svelte ui/src/lib/assets/AssetDetail.svelte ui/src/lib/api.ts ui/src/lib/types.ts
git commit -m "feat(ui): roles lead the gallery - role chips, tile tags, grouped job results, made-from list

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Also add the page test files you created or extended, by name.

---

## Stage 2: Notes

### Task 5: `dw/notes.py` stores and bookkeeping

**Files:**
- Create: `dw/notes.py`
- Modify: `dw/runs.py`. Extract a `write_json_atomic(path, data)` from
  `write_manifest` (line 683) so that both use it, with no duplicate
  block.
- Modify: `dw/server/routes/gallery.py`:
  - `RUN_BOOKKEEPING_FILES` at line 54 gains `NOTES_FILE_NAME`;
  - `RUN_SIDECARS` at line 277 gains it;
  - single-output delete at line 351 drops that file's note entry.
- Modify: `dw/server/outputs.py`. If it holds its own bookkeeping set,
  add `NOTES_FILE_NAME` there too (grep `RUN_BOOKKEEPING`).
- Modify: `dw/server/exports.py`, lines 235-255. Copy `notes.json` beside
  the manifest when present.
- Modify: `dw/workspace.py`. `_foreign_entries` (line 374) tolerates
  `WORKSPACE_FILE_NAME`.
- Modify: `docs/superpowers/specs/2026-10-12-roles-and-notes-design.md`.
  Change the asset row in the Stores table to "the asset directory's
  `notes.json` (`{"files": {name: note}}`)" and its reason to
  "`record_kept_shots` deletes its sidecar with the last entry".
- Modify: `docs/ARCHITECTURE.md`. Add one row for notes.
- Test: `tests/test_notes.py`

**Interfaces:**
- Produces, from `dw/notes.py`:
  - `NOTES_FILE_NAME = "notes.json"`, `WORKSPACE_FILE_NAME = "workspace.json"`,
    `MAX_NOTE_LENGTH = 4000`
  - `class NoteTooLong(ValueError)`
  - `clean_note(note: str | None) -> str | None`. It returns None for
    None, an empty string or whitespace only. It raises `NoteTooLong`
    above the limit, and `TypeError` for a non-string.
  - `read_notes(directory: str) -> dict`, always `{"run": str | None, "files": dict[str, str]}`.
    A missing or corrupt file gives empties.
  - `set_run_note(directory: str, note: str | None) -> str | None`. It
    returns the stored, cleaned value.
  - `set_file_note(directory: str, name: str, note: str | None) -> str | None`.
    `name` is relative to `directory` and `/`-separated.
  - `drop_file_note(directory: str, name: str) -> None`
  - `effective_note(notes: dict, name: str) -> tuple[str | None, str | None]`
    returns `(note, "file" | "run" | None)`.
  - `read_workspace_note(root: str) -> str | None`
  - `set_workspace_note(root: str, note: str | None) -> str | None`

  A write that leaves no notes removes the file. Writes go through
  `dw.runs.write_json_atomic`, under a per-directory `threading.Lock`.
  Callers have already validated `directory` and `root`; this module
  joins only fixed file names onto them.
- Produces, from `dw/runs.py`: `write_json_atomic(path: str, data) -> None`.
  It raises `OSError`. `write_manifest` keeps its never-fatal wrapper
  around it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_notes.py
import json, os, threading
import pytest
from dw import notes as N


def test_clean_note_rules():
    assert N.clean_note(None) is None
    assert N.clean_note("   \n") is None
    assert N.clean_note("shift 5") == "shift 5"
    assert N.clean_note("x" * N.MAX_NOTE_LENGTH) == "x" * N.MAX_NOTE_LENGTH
    with pytest.raises(N.NoteTooLong):
        N.clean_note("x" * (N.MAX_NOTE_LENGTH + 1))
    with pytest.raises(TypeError):
        N.clean_note(5)


def test_run_and_file_notes_and_effective(tmp_path):
    d = str(tmp_path)
    assert N.read_notes(d) == {"run": None, "files": {}}
    N.set_run_note(d, "testing shift 3 vs 5")
    N.set_file_note(d, "final/b.png", "the keeper")
    notes = N.read_notes(d)
    assert N.effective_note(notes, "final/b.png") == ("the keeper", "file")
    assert N.effective_note(notes, "a.png") == ("testing shift 3 vs 5", "run")
    assert N.effective_note({"run": None, "files": {}}, "a.png") == (None, None)


def test_clearing_everything_removes_the_file(tmp_path):
    d = str(tmp_path)
    N.set_run_note(d, "x")
    N.set_file_note(d, "a.png", "y")
    N.set_run_note(d, "")
    N.drop_file_note(d, "a.png")
    assert not os.path.exists(os.path.join(d, N.NOTES_FILE_NAME))


def test_corrupt_file_reads_empty(tmp_path):
    (tmp_path / N.NOTES_FILE_NAME).write_text("{not json")
    assert N.read_notes(str(tmp_path)) == {"run": None, "files": {}}


def test_concurrent_file_notes_all_land(tmp_path):
    d = str(tmp_path)
    threads = [
        threading.Thread(target=N.set_file_note, args=(d, f"{i}.png", str(i)))
        for i in range(20)
    ]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(N.read_notes(d)["files"]) == 20


def test_workspace_note(tmp_path):
    root = str(tmp_path)
    assert N.read_workspace_note(root) is None
    N.set_workspace_note(root, "the anthem project")
    assert N.read_workspace_note(root) == "the anthem project"
    assert json.load(open(tmp_path / N.WORKSPACE_FILE_NAME)) == {
        "notes": "the anthem project"
    }
    N.set_workspace_note(root, None)
    assert not os.path.exists(tmp_path / N.WORKSPACE_FILE_NAME)


def test_workspace_with_note_is_deletable(tmp_path):
    from dw.workspace import create_workspace, delete_workspace
    # follow tests/test_workspace*.py for how a root and a named workspace are set up,
    # then: set_workspace_note(ws.root, "x"); delete_workspace(...) succeeds
```

Complete the last test against the real `create_workspace` and
`delete_workspace` signatures, copying the setup from the existing
workspace tests.

Add bookkeeping tests to `tests/test_server.py`, or to a new
`tests/test_notes_bookkeeping.py`, using that file's fixtures:

```python
def test_run_with_only_notes_is_pruned_and_orphan(client, output_root):
    # write a run (manifest + one output + notes.json with a run note);
    # DELETE /api/gallery/<that output>; assert the run directory is gone.
    # Then write a run holding only manifest.json + notes.json and assert
    # it is listed by the orphan listing (grep the route _orphan_entries feeds).


def test_single_output_delete_drops_its_note(client, output_root):
    # run with two outputs, file notes on both; delete one; read_notes keeps only the other


def test_export_carries_notes(client, ...):
    # follow the existing export test; assert notes.json is in the zip
```

Write these as full tests by copying the nearest existing delete, orphan
and export test bodies.

- [ ] **Step 2: Run them to verify they fail**

Run: `DW_DEVICE=cpu venv/bin/python -m pytest tests/test_notes.py -v -n0`
Expected: FAIL with `No module named 'dw.notes'`.

- [ ] **Step 3: Extract `write_json_atomic` in `dw/runs.py`**

```python
def write_json_atomic(path, data):
    """Write `data` as JSON to `path` by a temp file and a rename, so a
    reader never sees half a file. Raises OSError."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=f".{os.path.basename(path)}-")
    try:
        with os.fdopen(fd, "w") as file:
            json.dump(data, file, indent=2, default=str)
        os.replace(tmp_path, path)
    except OSError:
        os.unlink(tmp_path)
        raise


def write_manifest(run_dir, manifest):
    """...(docstring unchanged)..."""
    path = os.path.join(run_dir, MANIFEST_FILE_NAME)
    try:
        write_json_atomic(path, manifest)
    except OSError as e:
        logger.warning(f"Could not write {path}: {e}")
        return None
    return path
```

- [ ] **Step 4: Write `dw/notes.py`**

```python
"""Notes: optional plain text a person or an agent leaves on a run, on a
file, on an asset or on a workspace, saying what it is for.

Two stores, both sidecars the media listings never show (.json is not a
media kind):

- `notes.json` in a directory - a run directory or an asset-library
  directory - holding `{"run": ..., "files": {<name relative to the
  directory>: ...}}`. A file's effective note is its own, else the run's.
  The engine rewrites manifest.json throughout a run, so notes live apart
  from it and only the server writes them.
- `workspace.json` at a workspace root, holding `{"notes": ...}`.

Callers resolve and validate the directory; this module joins only its
own fixed file names onto it. A write that leaves nothing removes the file.
"""

import json
import os
import threading

from .runs import write_json_atomic

NOTES_FILE_NAME = "notes.json"
WORKSPACE_FILE_NAME = "workspace.json"
MAX_NOTE_LENGTH = 4000

_locks = {}
_locks_guard = threading.Lock()


class NoteTooLong(ValueError):
    pass


def clean_note(note):
    """The note as stored: None for nothing, else the text, length-checked."""
    if note is None:
        return None
    if not isinstance(note, str):
        raise TypeError(f"a note is text, not {type(note).__name__}")
    if not note.strip():
        return None
    if len(note) > MAX_NOTE_LENGTH:
        raise NoteTooLong(
            f"a note is at most {MAX_NOTE_LENGTH} characters, got {len(note)}"
        )
    return note


def _lock(directory):
    key = os.path.realpath(directory)
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def _read_json(path):
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def read_notes(directory):
    data = _read_json(os.path.join(directory, NOTES_FILE_NAME))
    run = data.get("run") if isinstance(data.get("run"), str) else None
    files = data.get("files") if isinstance(data.get("files"), dict) else {}
    files = {
        k: v for k, v in files.items() if isinstance(k, str) and isinstance(v, str)
    }
    return {"run": run, "files": files}


def _store(directory, notes):
    path = os.path.join(directory, NOTES_FILE_NAME)
    if notes["run"] is None and not notes["files"]:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return
    write_json_atomic(path, {k: v for k, v in notes.items() if v})


def _update(directory, change):
    with _lock(directory):
        notes = read_notes(directory)
        change(notes)
        _store(directory, notes)


def set_run_note(directory, note):
    note = clean_note(note)
    _update(directory, lambda notes: notes.__setitem__("run", note))
    return note


def set_file_note(directory, name, note):
    note = clean_note(note)

    def change(notes):
        if note is None:
            notes["files"].pop(name, None)
        else:
            notes["files"][name] = note

    _update(directory, change)
    return note


def drop_file_note(directory, name):
    if os.path.exists(os.path.join(directory, NOTES_FILE_NAME)):
        set_file_note(directory, name, None)


def effective_note(notes, name):
    own = notes["files"].get(name)
    if own:
        return own, "file"
    if notes["run"]:
        return notes["run"], "run"
    return None, None


def read_workspace_note(root):
    note = _read_json(os.path.join(root, WORKSPACE_FILE_NAME)).get("notes")
    return note if isinstance(note, str) and note.strip() else None


def set_workspace_note(root, note):
    note = clean_note(note)
    path = os.path.join(root, WORKSPACE_FILE_NAME)
    with _lock(root):
        if note is None:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        else:
            write_json_atomic(path, {"notes": note})
    return note
```

- [ ] **Step 5: Bookkeeping**

1. `gallery.py`: add `from ...notes import NOTES_FILE_NAME, drop_file_note`.
   Add `NOTES_FILE_NAME` to `RUN_BOOKKEEPING_FILES` and to `RUN_SIDECARS`.
   In the single-file delete branch, after the file is removed and
   before `_prune_empty_run_directory`, call
   `drop_file_note(run_dir, inner_name)`. Here `run_dir` and `inner_name`
   are the run directory and the file's name relative to it. Derive them
   as Task 3's `_gallery_entries` change does, and factor a shared
   `_split_run_name(root, name) -> (run_dir, inner) | None` helper rather
   than writing the split twice.
2. `exports.py`: where the manifest is added to the zip, add `notes.json`
   the same way when it exists.
3. `workspace.py`: in `_foreign_entries`, add `WORKSPACE_FILE_NAME` to the
   names it tolerates. Import it from `.notes`, and check that `dw.notes`
   doesn't import `dw.workspace`, which would make an import cycle.
4. `docs/ARCHITECTURE.md` row:
   `| Notes on runs, files, assets, workspaces | dw/notes.py | notes.json beside what it describes (run dir or asset dir; a file's own note else the run's), workspace.json at a workspace root; only the server writes them, never the engine; ≤4000 chars, plain text. | tests/test_notes.py |`
5. Update the spec's Stores table as described under **Files**.

- [ ] **Step 6: Run the tests, the ratchet and lint**

```bash
DW_DEVICE=cpu venv/bin/python -m pytest tests/test_notes.py tests/test_notes_bookkeeping.py tests/test_server.py tests/test_workspace*.py -q
venv/bin/python scripts/arch_metrics.py --check docs/stabilization/baseline.json
venv/bin/ruff check dw tests && venv/bin/ruff format --check dw tests
```

Expected: PASS, with modules now at 210.

- [ ] **Step 7: Commit**

```bash
git add dw/notes.py dw/runs.py dw/server/routes/gallery.py dw/server/exports.py dw/workspace.py tests/test_notes.py tests/test_notes_bookkeeping.py docs/ARCHITECTURE.md docs/superpowers/specs/2026-10-12-roles-and-notes-design.md
git commit -m "feat(notes): notes.json and workspace.json stores, kept in step with prune, delete and export

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Also add `dw/server/outputs.py` if you changed it.

### Task 6: Server API for notes

**Files:**
- Modify: `dw/server/api_models.py`:
  - `GalleryFile` gains `note`;
  - `GalleryMetadata` gains `note`, `run_note` and `note_source`;
  - `AssetFile` gains `note`;
  - `WorkspaceInfo` gains `note`;
  - `JobSummary` and `JobDetail` gain `note`;
  - new `NoteRequest` and `NoteSet` models.
- Modify: `dw/server/admission.py`. `JobRequest` (line 383) gains
  `note: Optional[str]`. It is validated with `clean_note` at admission,
  with 400 on `NoteTooLong`, and put into the job spec as `spec["note"]`.
- Modify: `dw/server/job_record.py`:
  - `RERUN_SPEC_KEYS` gains `"note"`;
  - `Job.__init__` sets `self.note = spec.get("note")`.
- Modify: `dw/server/job_history.py`:
  - add the column with `ALTER TABLE jobs ADD COLUMN note TEXT`, in the
    same pattern as `acknowledged`;
  - write it on save (around line 165);
  - read it in both row readers (around lines 212 and 255), and map it
    into the summary and detail dicts.
- Modify: `dw/server/job_results.py`. In `record_progress`, on
  `run_start`, when `job.note` is set, write it as the run note.
- Modify: `dw/server/routes/media.py`:
  - add `PUT /api/notes`;
  - the metadata route gains the note fields.
- Modify: `dw/server/routes/gallery.py`. `_gallery_entries` adds the
  effective `note`, reading `read_notes(run_dir)` once per run directory
  per listing.
- Modify: `dw/server/routes/assets.py`:
  - `upload_media` gains `note: Optional[str] = None` (a query parameter,
    like `asset_name`);
  - `KeepRequest` gains `note`;
  - keep falls back to the source file's effective note;
  - asset `summary()` (lines 262-287) adds `note`;
  - asset delete (line 470) calls `drop_file_note` and
    `record_kept_shots(directory, name, None, None)`.
- Modify: `dw/server/routes/library.py`. The workspace list (line 91)
  adds `note` via `read_workspace_note(ws.root)`.
- Modify: `ui/src/lib/generated/openapi.json` and `api-schema.ts`,
  regenerated.
- Modify: `tests/test_api_contract.py`. Add `PUT /api/notes` to
  `UI_READ_ROUTES`, or to whichever list covers write routes the UI
  calls. Read the file to see which list fits.
- Modify: `docs/SERVER.md`. Document the route and the fields.
- Test: `tests/test_notes_api.py`

**Interfaces:**
- Consumes: everything Task 5 produces.
- Produces:
  - `PUT /api/notes`, with body `NoteRequest{target: str, note: str | None}`,
    returning `NoteSet{target: str, note: str | None, note_source: Literal["file","run"] | None}`.
  - Target forms:
    - `workspace:<name>` targets `named_workspace(...).root`, or the
      configured root for `default`;
    - `asset:<path>` targets the asset directory's `notes.json`, keyed by
      basename. Resolve it with `validate_asset_reference` plus the
      library directory, exactly as asset delete does;
    - `<identity>/<run_id>` is a run directory: a path whose last segment
      satisfies `is_run_id` and which is a directory under the output
      root. It sets the run note;
    - anything else is a gallery file name, resolved with
      `resolve_output_file`, which is the delete route's resolver. It
      sets the file note in its run directory. A flat-layout file, with
      no run directory, answers 409 with "notes need a run directory".
  - The status codes:
    - an unknown target gives 404;
    - traversal or an invalid name gives 400;
    - a note over 4000 characters gives 400;
    - a write failure gives 500.
  - `JobSummary.note` and `JobDetail.note`, as `sometimes()`.
  - `GalleryMetadata.note`, `run_note` and `note_source`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_notes_api.py - use tests/test_server.py's client/output_root/asset fixtures
import json, os
import pytest
from dw import notes as N


def test_put_file_and_run_note_and_read_back(client, output_root, make_run):
    run_dir = make_run("wf", ["a.png", "final/b.png"])  # helper: manifest + files, returns run dir
    run = os.path.relpath(run_dir, output_root).replace(os.sep, "/")
    r = client.put("/api/notes", json={"target": run, "note": "shift 3 vs 5"})
    assert r.status_code == 200 and r.json()["note_source"] == "run"
    r = client.put("/api/notes", json={"target": f"{run}/final/b.png", "note": "keeper"})
    assert r.json() == {"target": f"{run}/final/b.png", "note": "keeper", "note_source": "file"}
    files = {f["label"]: f for f in client.get("/api/gallery").json()["files"]}
    assert files["a.png"]["note"] == "shift 3 vs 5"
    assert files["b.png"]["note"] == "keeper"
    meta = client.get(f"/api/gallery/{run}/final/b.png/metadata").json()
    assert (meta["note"], meta["run_note"], meta["note_source"]) == ("keeper", "shift 3 vs 5", "file")


def test_clear_note(client, output_root, make_run):
    run_dir = make_run("wf", ["a.png"])
    run = os.path.relpath(run_dir, output_root).replace(os.sep, "/")
    client.put("/api/notes", json={"target": run, "note": "x"})
    r = client.put("/api/notes", json={"target": run, "note": ""})
    assert r.json()["note"] is None
    assert not os.path.exists(os.path.join(run_dir, N.NOTES_FILE_NAME))


@pytest.mark.parametrize("target", [
    "../etc/passwd", "wf/../../x", "/abs/path.png", "asset:../x.png",
    "workspace:../other", "workspace:nope", "wf/20261012-000000-zzzz/missing.png",
])
def test_put_note_refuses_traversal_targets(client, target):
    r = client.put("/api/notes", json={"target": target, "note": "x"})
    assert r.status_code in (400, 404)


def test_note_too_long(client, output_root, make_run):
    run_dir = make_run("wf", ["a.png"])
    run = os.path.relpath(run_dir, output_root).replace(os.sep, "/")
    r = client.put("/api/notes", json={"target": run, "note": "x" * 4001})
    assert r.status_code == 400


def test_workspace_note(client):
    r = client.put("/api/notes", json={"target": "workspace:default", "note": "anthem"})
    assert r.status_code == 200
    ws = {w["name"]: w for w in client.get("/api/workspaces").json()["workspaces"]}
    assert ws["default"]["note"] == "anthem"


def test_asset_note_upload_keep_and_delete(client, output_root, make_run):
    up = client.post("/api/uploads", params={"filename": "iris.png", "note": "cast: iris"},
                     content=b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    ref = up.json()["reference"]
    assets = {a["name"]: a for a in client.get("/api/assets").json()["files"]}
    # find the entry for ref; assert its note == "cast: iris"


def test_keep_copies_effective_note(client, output_root, make_run):
    # make_run + run note; POST /api/assets/keep without note; asset note == run note


def test_deleted_asset_leaves_no_note_or_provenance(client, output_root, make_run):
    # keep a file with a note under asset_name 'x.png' (provenance written);
    # DELETE /api/assets/x.png; upload a new x.png with no note;
    # assert its asset entry has no note and its metadata has no job/provenance


def test_job_note_reaches_history_run_and_rerun(client, ...):
    # submit a tiny workflow with note="testing"; wait for it the way the
    # existing job tests do; assert GET /api/jobs/{id} note == "testing",
    # the run directory's notes.json run note == "testing", and a rerun's
    # job carries note "testing"


def test_note_set_during_run_survives_manifest_rewrite(tmp_path):
    from dw.runs import write_manifest
    d = str(tmp_path)
    write_manifest(d, {"status": "running", "steps": []})
    N.set_run_note(d, "mid-run")
    write_manifest(d, {"status": "succeeded", "steps": []})
    assert N.read_notes(d)["run"] == "mid-run"
```

Write the placeholder-comment tests in full, using the fixtures and job
helpers `tests/test_server.py` and the job tests already use (for example
`grep -n "def test_rerun" tests/*.py`). If `make_run` doesn't exist, add
it as a fixture in this file, built on Task 3's `_write_run` helper.

- [ ] **Step 2: Run them to verify they fail**

Run: `DW_DEVICE=cpu venv/bin/python -m pytest tests/test_notes_api.py -v -n0`
Expected: FAIL with 404 or 405 on `/api/notes`.

- [ ] **Step 3: Implement the route**

In `dw/server/routes/media.py`:

```python
class NoteRequest(
    BaseModel
):  # or define in api_models.py next to the response, per file convention
    target: str
    note: Optional[str] = None


@router.put("/api/notes", response_model=NoteSet, response_model_exclude_unset=True)
def put_note(
    request: Request, body: NoteRequest, ws: Workspace = Depends(selected_workspace)
):
    """Set or clear the note on a run, a file, an asset or a workspace."""
    try:
        note = clean_note(body.note)
    except (NoteTooLong, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    directory, name, kind = resolve_note_target(request.app.state, ws, body.target)
    # resolve_note_target raises HTTPException 400/404/409 as listed above
    try:
        if kind == "workspace":
            stored, source = set_workspace_note(directory, note), None
        elif kind == "run":
            stored, source = set_run_note(directory, note), "run"
        else:
            set_file_note(directory, name, note)
            stored, source = effective_note(read_notes(directory), name)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not write the note: {e}")
    return {
        "target": body.target,
        "note": stored,
        "note_source": source if stored else None,
    }
```

Write `resolve_note_target(state, ws, target) -> (directory, name, kind)`
in the same file. It dispatches on the four target forms, reusing:
- `ref_name(WORKSPACE?...)`. There is no `workspace:` prefix in
  `dw/references.py`, and only that module may spell prefixes
  (`prefix_literals` ratchet). So add
  `WORKSPACE = "workspace:"` to `dw/references.py`, next to `ASSET`, with
  a one-line comment that it addresses a note target, and add it to
  `FALLBACK_REFERENCE_NAMES` in `scripts/arch_metrics.py` only if
  `tests/test_arch_metrics.py` requires it. Use `is_ref` and `ref_name`
  everywhere;
- `named_workspace` from `dw/workspace.py` for named workspaces, and the
  default workspace's root for `default`;
- `validate_asset_reference` and `writable_asset_directory` from
  `routes/assets.py`, as the asset delete route resolves names;
- `resolve_output_file` from `dw/server/outputs.py` for a file;
- for a run directory, `validate_path(candidate, output_root)` with a
  non-None root, then `os.path.isdir` and `is_run_id(basename)`.

Every directory returned has passed a validator against a non-None root.
For a file, find its run directory with the `_split_run_name` helper from
Task 5.

- [ ] **Step 4: Note fields on the readers**

- `_gallery_entries`: keep a `notes_by_run = {}` cache per listing, and
  set `entry["note"] = effective_note(notes, inner)[0]` when there is a
  run.
- Metadata route: `note`, `note_source` and `run_note`, from
  `read_notes(run_dir)`. For an `asset:` name, read its directory's
  `notes.json` with the basename.
- Assets `summary()`: `note` from the entry's directory's `notes.json`.
  Cache it per directory within one listing.
- Workspace list: `note`.
- Jobs: the `note` column goes into `JobSummary` and `JobDetail`.
- `record_progress` on `run_start`:

```python
        if getattr(job, "note", None) and job.run_dir:
            try:
                set_run_note(os.path.join(output_dir, job.run_dir), job.note)
            except OSError as e:
                logger.warning(f"Could not write the run note for job {job.id}: {e}")
```

  `output_dir` is the parameter `record_progress` already receives. Check
  the `job.id` attribute name in `job_record.py`.
- Upload: `note` is cleaned (400 on `NoteTooLong`). After the file is
  written, `set_file_note(os.path.dirname(dest), os.path.basename(dest), note)`.
- Keep: `body.note` if it is set, else the source's effective note,
  looked up from the source's run directory. Write it beside the kept
  file the same way.
- Asset delete: after `os.remove`, call `drop_file_note(directory, basename)`
  and `record_kept_shots(directory, basename, None, None)`.

- [ ] **Step 5: Regenerate and run**

```bash
venv/bin/python scripts/dump_openapi.py && (cd ui && npm run gen:api)
DW_DEVICE=cpu venv/bin/python -m pytest tests/test_notes_api.py tests/test_server.py tests/test_api_contract.py tests/test_ui_twins.py tests/test_arch_metrics.py -q
venv/bin/python scripts/arch_metrics.py --check docs/stabilization/baseline.json
```

Expected: PASS.

- [ ] **Step 6: Docs**

In `docs/SERVER.md`, add a "Notes" subsection covering:
- the route;
- the four target forms;
- the 4,000-character limit;
- that clearing is `null` or `""`;
- the effective-note rule;
- `note` on job submission, upload and keep.

- [ ] **Step 7: Commit**

```bash
git add dw/server/api_models.py dw/server/admission.py dw/server/job_record.py dw/server/job_history.py dw/server/job_results.py dw/server/routes/media.py dw/server/routes/gallery.py dw/server/routes/assets.py dw/server/routes/library.py dw/references.py tests/test_notes_api.py tests/test_api_contract.py ui/src/lib/generated/openapi.json ui/src/lib/generated/api-schema.ts docs/SERVER.md
git commit -m "feat(notes): PUT /api/notes, note on jobs, uploads and keeps, and on every listing

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Also add `scripts/arch_metrics.py` if you touched it.

### Task 7: MCP for notes

**Files:**
- Modify: `dw_mcp/tools_media.py`:
  - `upload_asset` (line 312) and `keep_output` (line 355) gain `note`;
  - a new `set_note` tool goes here, beside the other asset and output
    write tools.
- Modify: `dw_mcp/assets.py`. Forward `note`, and add a `set_note` helper
  calling `PUT /api/notes`. Add `note` to `ASSET_SUMMARY_FIELDS` (line
  134), clipped.
- Modify: `dw_mcp/tools_jobs.py` and `dw_mcp/diagnose.py`.
  `run_workflow` gains `note`, which is forwarded in the job body.
  `get_job` returns `note` in full, and `list_jobs`'s compact entries
  carry it clipped.
- Modify: `dw_mcp/workspaces.py`. `WORKSPACE_SUMMARY_FIELDS` (line 21)
  gains `note`, clipped.
- Modify: `dw_mcp/catalog.py`. The gallery compact projection gains
  `note`, clipped. `get_gallery_metadata` returns the full note.
- Modify: the MCP server instructions string. Find it with
  `grep -rn "Start from \`list_workflows" dw_mcp`, and add the agent
  guidance sentence.
- Modify: `docs/WORKFLOW_GUIDE.md`, *Authoring a workflow from an agent*
  section, and `docs/MCP.md`.
- Modify: the surface snapshot, regenerated.
- Test: extend the existing MCP test files for these tools (find them with
  `grep -ln "keep_output\|run_workflow" tests/test_mcp*.py`).

**Interfaces:**
- Consumes: Task 6's routes and fields.
- Produces:
  - `set_note(target: str, note: str | None) -> dict`, returning the
    route's `NoteSet` body;
  - `clip_note(entry: dict, limit: int = 200) -> dict`, one helper in
    `dw_mcp/client.py` or wherever the other projection helpers live
    (`grep -n "def .*project" dw_mcp/*.py`). It replaces a longer `note`
    with its first 200 characters plus `"…"` and sets
    `note_truncated: True`.

- [ ] **Step 1: Write the failing tests**

Using the MCP test files' existing pattern for faking HTTP responses:

```python
def test_set_note_puts_target_and_note(fake_server):
    result = set_note(target="wf/20261012-000000-aaaa", note="shift 5")
    assert fake_server.last_request.method == "PUT"
    assert fake_server.last_request.path == "/api/notes"
    assert fake_server.last_request.json == {
        "target": "wf/20261012-000000-aaaa",
        "note": "shift 5",
    }


def test_run_workflow_forwards_note(fake_server):
    run_workflow(
        workflow_path="templates/text-to-image",
        note="seed sweep 1/4",
        acknowledged_cost=True,
    )
    assert fake_server.last_job_body["note"] == "seed sweep 1/4"


def test_compact_projections_clip_long_notes(fake_server):
    fake_server.gallery = [{"name": "a.png", "note": "x" * 500, "role": "final"}]
    entry = list_gallery()["files"][0]
    assert len(entry["note"]) <= 201 and entry["note_truncated"] is True


def test_short_note_not_marked_truncated(fake_server):
    fake_server.gallery = [{"name": "a.png", "note": "short", "role": "final"}]
    entry = list_gallery()["files"][0]
    assert entry["note"] == "short" and "note_truncated" not in entry
```

Rename `fake_server`, `last_request` and the call shapes to match what
the existing MCP tests use. The assertions stay the same.

- [ ] **Step 2: Run them to verify they fail**

Run: `DW_DEVICE=cpu venv/bin/python -m pytest <the MCP test files> -k note -v -n0`
Expected: FAIL.

- [ ] **Step 3: Implement it**

`set_note` tool description:

> "Set or clear (note null or '') the note on a run ('<workflow>/<run id>'), one output file (its gallery name), an asset ('asset:<name>') or a workspace ('workspace:<name>'). A note is plain text up to 4000 characters saying what the thing is for - what a run tested, why a file is the keeper. Free: no acknowledged_cost."

`run_workflow`'s `note` parameter description:

> "What this run is for, kept with the run and shown on every file it writes - e.g. 'testing video_shift 5 against 3'. When running a set of variations, give each its own."

`upload_asset` and `keep_output` `note`:

> "A note kept with the asset. keep_output copies the source file's note when this is left out."

The MCP instructions gain one sentence, after the loop paragraph:

> "Pass `note` on `run_workflow` saying what each run tests when exploring variations, and `set_note` to mark the keeper - the note is shown with every file the run made."

In `docs/WORKFLOW_GUIDE.md`, under *Authoring a workflow from an agent*,
add a short paragraph with the same guidance. Add to it that
`result.role` (`final`/`intermediate`) overrides the inferred role when
a step's default is wrong.

`docs/MCP.md` documents `set_note` and the `note` parameters.

- [ ] **Step 4: Check the tool-surface budget**

Run the existing MCP surface tests:
`DW_DEVICE=cpu venv/bin/python -m pytest tests/ -k "surface or mcp_budget or tool_list" -q`.
Then regenerate the snapshot with `scripts/surface_snapshot.py`, using
the flag `--help` shows. If a token-budget test fails, cut the new
descriptions down. Don't raise the budget.

- [ ] **Step 5: Run and commit**

```bash
DW_DEVICE=cpu venv/bin/python -m pytest tests/test_mcp*.py -q
git add dw_mcp/tools_media.py dw_mcp/assets.py dw_mcp/tools_jobs.py dw_mcp/diagnose.py dw_mcp/workspaces.py dw_mcp/catalog.py docs/WORKFLOW_GUIDE.md docs/MCP.md
git commit -m "feat(mcp): set_note, note on run_workflow, upload_asset and keep_output, notes in every listing

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Also add the MCP instructions file, `dw_mcp/client.py` if it changed, the
snapshot, and the test files, each by name.

### Task 8: UI notes

**Files:**
- Create: `ui/src/lib/ui/NoteEditor.svelte`
- Create: `ui/src/lib/ui/NoteEditor.test.ts`
- Modify: `ui/src/lib/api.ts`. Add
  `setNote(target: string, note: string | null): Promise<NoteSet>`,
  which PUTs `/api/notes`.
- Modify: `ui/src/lib/types.ts`. Re-export `NoteSet`.
- Modify: `ui/src/lib/pages/OverviewPage.svelte`. The note goes under the
  `<h1>` at line 128, with target `workspace:<name>`.
- Modify: `ui/src/lib/workspace.svelte.ts`. `loadWorkspaces` (line 72)
  keeps `note`.
- Modify: `ui/src/lib/gallery/GalleryDetail.svelte`. The note goes under
  the `RoleTag`. When `note_source === 'run'` it is labelled "Run note",
  with two actions: "Edit run note" (target = run directory) and "Note
  this file" (target = the file).
- Modify: `ui/src/lib/assets/AssetDetail.svelte`. Note, with target
  `asset:<name>`.
- Modify: `ui/src/lib/pages/JobPage.svelte`. The run note goes under the
  header, with target `<identity>/<run_id>` from the job's `run_dir`.
- Modify: the gallery tile. Add a small note icon with
  `title={firstLine(note)}` when `file.note` is set.
- Test: `NoteEditor.test.ts`, plus a test per page that the note renders.

**Interfaces:**
- Consumes: `setNote`, and the `note` fields from Task 6.
- Produces: `NoteEditor` props:

```ts
{
  note: string | null | undefined
  target: string
  label?: string            // e.g. "Run note"; default "Note"
  placeholder?: string      // default "Add a note"
  onsaved?: (note: string | null) => void
}
```

- [ ] **Step 1: Write the failing test**

```ts
// ui/src/lib/ui/NoteEditor.test.ts
import { fireEvent, render, screen } from '@testing-library/svelte'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import * as api from '../api'
import NoteEditor from './NoteEditor.svelte'

describe('NoteEditor', () => {
  beforeEach(() => vi.restoreAllMocks())

  it('shows the note as text, never markup', () => {
    render(NoteEditor, { note: '<b>bold</b>', target: 'workspace:default' })
    expect(screen.getByText('<b>bold</b>')).toBeTruthy()
    expect(document.querySelector('b')).toBeNull()
  })

  it('shows the placeholder when empty', () => {
    render(NoteEditor, { note: null, target: 'workspace:default' })
    expect(screen.getByText('Add a note')).toBeTruthy()
  })

  it('saves through setNote and reports the stored note', async () => {
    const spy = vi.spyOn(api, 'setNote').mockResolvedValue({ target: 't', note: 'new', note_source: 'run' })
    const onsaved = vi.fn()
    render(NoteEditor, { note: 'old', target: 't', onsaved })
    await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
    const box = screen.getByRole('textbox')
    await fireEvent.input(box, { target: { value: 'new' } })
    await fireEvent.click(screen.getByRole('button', { name: /save/i }))
    expect(spy).toHaveBeenCalledWith('t', 'new')
    expect(onsaved).toHaveBeenCalledWith('new')
  })

  it('cancel restores the old note without calling the api', async () => {
    const spy = vi.spyOn(api, 'setNote')
    render(NoteEditor, { note: 'old', target: 't' })
    await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
    await fireEvent.input(screen.getByRole('textbox'), { target: { value: 'changed' } })
    await fireEvent.click(screen.getByRole('button', { name: /cancel/i }))
    expect(spy).not.toHaveBeenCalled()
    expect(screen.getByText('old')).toBeTruthy()
  })

  it('saving an empty box clears the note', async () => {
    const spy = vi.spyOn(api, 'setNote').mockResolvedValue({ target: 't', note: null, note_source: null })
    render(NoteEditor, { note: 'old', target: 't' })
    await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
    await fireEvent.input(screen.getByRole('textbox'), { target: { value: '' } })
    await fireEvent.click(screen.getByRole('button', { name: /save/i }))
    expect(spy).toHaveBeenCalledWith('t', null)
  })

  it('shows the server error and stays in edit mode', async () => {
    vi.spyOn(api, 'setNote').mockRejectedValue(new Error('a note is at most 4000 characters'))
    render(NoteEditor, { note: 'old', target: 't' })
    await fireEvent.click(screen.getByRole('button', { name: /edit note/i }))
    await fireEvent.click(screen.getByRole('button', { name: /save/i }))
    expect(await screen.findByText(/at most 4000/)).toBeTruthy()
    expect(screen.getByRole('textbox')).toBeTruthy()
  })
})
```

The memory note for this repo says: no `vi.resetModules` in Svelte
component tests. Use `vi.spyOn` as above.

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ui && npx vitest run src/lib/ui/NoteEditor.test.ts`
Expected: FAIL, because the module is not found.

- [ ] **Step 3: Implement it**

```svelte
<script lang="ts">
  import { setNote } from '../api'

  let {
    note,
    target,
    label = 'Note',
    placeholder = 'Add a note',
    onsaved,
  }: {
    note: string | null | undefined
    target: string
    label?: string
    placeholder?: string
    onsaved?: (note: string | null) => void
  } = $props()

  let editing = $state(false)
  let draft = $state('')
  let saving = $state(false)
  let error = $state<string | null>(null)
  let shown = $state<string | null>(null)
  $effect(() => {
    shown = note ?? null
  })

  function edit() {
    draft = shown ?? ''
    error = null
    editing = true
  }

  async function save() {
    saving = true
    error = null
    try {
      const stored = await setNote(target, draft.trim() ? draft : null)
      shown = stored.note ?? null
      editing = false
      onsaved?.(shown)
    } catch (e) {
      error = e instanceof Error ? e.message : String(e)
    } finally {
      saving = false
    }
  }
</script>

<div class="note">
  {#if editing}
    <textarea bind:value={draft} maxlength="4000" rows="4" aria-label={label}></textarea>
    <div class="actions">
      <button type="button" onclick={save} disabled={saving}>Save</button>
      <button type="button" onclick={() => (editing = false)} disabled={saving}>Cancel</button>
    </div>
    {#if error}<p class="error" role="alert">{error}</p>{/if}
  {:else}
    <div class="shown">
      <span class="label">{label}</span>
      {#if shown}
        <p class="text">{shown}</p>
      {:else}
        <p class="text muted">{placeholder}</p>
      {/if}
      <button type="button" class="icon" aria-label="Edit note" onclick={edit}>✎</button>
    </div>
  {/if}
</div>

<style>
  .text {
    white-space: pre-wrap;
    margin: 0;
  }
  .muted {
    color: var(--text-muted);
  }
  textarea {
    width: 100%;
    font: inherit;
  }
  .actions {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.25rem;
  }
  .error {
    color: var(--danger);
  }
  .label {
    font-size: 0.75rem;
    color: var(--text-muted);
  }
</style>
```

Match the button classes and colour tokens to `ui/src/app.css` and the
existing small-button style, for example by copying from
`ConfirmDialog`'s buttons. Check how `api.ts` surfaces server error
detail, and make `setNote` throw an `Error` carrying the server's
`detail` the way the other `api.ts` write calls do.

- [ ] **Step 4: Place it in the four views and on the tile**

- **Overview:** `<NoteEditor note={ws.note} target={`workspace:${ws.name}`} onsaved={reloadWorkspaces} />`
  under the `<h1>`.
- **Gallery detail:** if `meta.note_source === 'run'`, show
  `<NoteEditor note={meta.run_note} target={runDir} label="Run note" />`
  plus a "Note this file" button. That button swaps it for
  `<NoteEditor note={null} target={meta.name} label="File note" />`.
  Otherwise show `<NoteEditor note={meta.note} target={meta.name} />`.
  `runDir` is the file name's `<identity>/<run_id>` prefix. Derive it
  from `folder` and `run_id` on the gallery entry. After a save,
  refetch the metadata and refresh the listing entry's `note`.
- **Asset detail:** `<NoteEditor note={asset.note} target={`asset:${asset.name}`} />`
- **Job page:** `<NoteEditor note={job.note} target={job.run_dir} label="Run note" />`
  when `job.run_dir` is set.
- **Tile:** `{#if file.note}<span class="note-mark" title={file.note.split('\n')[0]} aria-label="Has a note">✎</span>{/if}`

Add one assertion per page test that the note text renders, extending
the page tests Task 4 touched.

- [ ] **Step 5: Run the UI checks**

```bash
cd ui && npx vitest run && npm run lint && npm run check && npm run metrics -- --check ../docs/stabilization/ui/baseline.json
```

Expected: all pass. Then run the twins test from the repo root:
`DW_DEVICE=cpu venv/bin/python -m pytest tests/test_ui_twins.py tests/test_api_contract.py -q`.

- [ ] **Step 6: Commit**

```bash
git add ui/src/lib/ui/NoteEditor.svelte ui/src/lib/ui/NoteEditor.test.ts ui/src/lib/api.ts ui/src/lib/types.ts ui/src/lib/pages/OverviewPage.svelte ui/src/lib/workspace.svelte.ts ui/src/lib/gallery/GalleryDetail.svelte ui/src/lib/assets/AssetDetail.svelte ui/src/lib/pages/JobPage.svelte ui/src/lib/pages/GalleryPage.svelte
git commit -m "feat(ui): notes on workspaces, runs, files and assets, edited in place

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Also add the page test files you touched, by name.

### Task 9: Whole-branch verification

**Files:** none new. This task only fixes what the checks turn up.

- [ ] **Step 1: Run the full Python suite, lint and ratchet**

```bash
DW_DEVICE=cpu venv/bin/python -m pytest tests/ -q
venv/bin/ruff check . && venv/bin/ruff format --check .
venv/bin/python scripts/arch_metrics.py --check docs/stabilization/baseline.json
venv/bin/python scripts/dump_openapi.py --check
```

Expected: all pass, with `modules` at 210.

- [ ] **Step 2: Run the full UI checks and e2e**

```bash
cd ui && npx vitest run && npm run lint && npm run check && npm run gen:api && git diff --exit-code src/lib/generated && npm run metrics -- --check ../docs/stabilization/ui/baseline.json
cd ui && npx playwright test
```

Expected: all pass. If an e2e gallery spec relied on the subfolder
filter or on intermediates being visible by default, update the spec to
click the Intermediate chip. That is the intended behaviour change.

- [ ] **Step 3: Check the app by hand**

Start `python -m dw.serve` against a scratch `DIFFUSERS_HELPER_ROOT`. Run
`workflows/templates/text-to-image.json` with
`note="first look"`. Then check:
- the gallery shows it with a Final tag;
- the run note appears in the detail panel;
- editing a file note works;
- the workspace overview note saves.

Take one screenshot each of the gallery, the detail panel and the
overview for the release note.

- [ ] **Step 4: Add a release note line**

Following the convention in `docs/RELEASING.md`, add to the
unreleased-notes section:
"Gallery files are tagged final or intermediate (inferred when a
workflow doesn't say) and the gallery opens on finals; runs, files,
assets and workspaces take an optional note (`note` on `run_workflow`,
MCP `set_note`)."

- [ ] **Step 5: Commit**

```bash
git add docs/RELEASING.md
git commit -m "docs(release): note roles and notes for 0.14.0

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Also add any e2e specs you updated, by name.
