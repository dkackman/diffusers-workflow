# Output Subfolders, Stage 2 (Server + MCP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A caller can see, over REST and MCP, which subfolder each output sits in — on gallery entries, as a gallery filter, and in the tool and guide text that tells an agent what the field means.

**Architecture:** The engine already writes subfolders and records them on manifest entries and `step_end` events (stage 1, merged as `685f594`). Stage 2 changes no write path. The gallery walker in `dw/server/app.py` splits a path with `split_run_path` instead of `strip_run_id`, so each entry carries `subfolder` beside `folder` (which keeps meaning workflow identity); `GET /api/gallery` takes `?subfolder=` and answers `subfolders`. MCP `list_gallery` grows the one parameter. Docs and tool descriptions carry the `final`/`intermediate` convention. Two engine tests the stage-1 review left unpinned are added.

**Tech Stack:** Python 3, FastAPI, pytest. Tests: `source ./activate` then `python -m pytest tests/<file> -q`.

**Spec:** `docs/proposals/output-folders.md` — sections *The gallery*, *`get_job` and `list_jobs`*, *`output:`, `asset:` … no change*, *Breaking changes*, and stage 2 of *Phasing*. The design is the authority; this plan argues from it.

## Global Constraints

- The field is `subfolder` everywhere: gallery entry key, query parameter, reply list (`subfolders`), MCP parameter.
- On a gallery entry `folder` keeps its meaning — the workflow identity (`ltx2/Gyre`), never `ltx2/Gyre/final`.
- `subfolder` is `""` for a file with nothing after its run id, and for every flat-layout file (no run id to anchor on) — documented, not special-cased.
- `get_job` gains no new key: manifest entries already carry `subfolder` (the server spreads the entry). Only its description changes.
- `list_jobs` is untouched.
- No engine write-path change. The two new engine tests pin existing behaviour.
- `docs/WORKFLOW_GUIDE.md`'s *Authoring a workflow from an agent* and its CLAUDE.md mirror (the *Type System* bullet list) change together.
- Commit messages end with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- Branch off `develop` (suggested `output-subfolders-stage-2`).

---

## File map

| File | Responsibility in this plan |
|---|---|
| `dw/server/app.py` | `_iter_gallery_files` yields `subfolder`; `_gallery_entries` carries it; `GET /api/gallery` filters on it and lists `subfolders` |
| `dw_mcp/catalog.py` | `list_gallery(client, limit=50, subfolder=None)` |
| `dw_mcp/server.py` | `list_gallery` signature + docstring; `get_job`, `save_workflow`, `validate_workflow` docstrings |
| `docs/SERVER.md`, `docs/MCP.md`, `docs/WORKFLOW_GUIDE.md`, `CLAUDE.md` | The field, the convention, the filter |
| `tests/test_server.py`, `tests/test_jobs_reused.py`, `tests/test_mcp_catalog.py`, `tests/test_runs.py` | Tests |

---

### Task 1: Gallery entries carry `subfolder`; `?subfolder=` and `subfolders`

**Files:**
- Modify: `dw/server/app.py` — `_iter_gallery_files` (~line 1952), `_gallery_entries` (~1978), the `/api/gallery` route (~2018), and the asset-library caller of `_iter_gallery_files` (~2334, `group_runs=False`); the import line `from ..runs import is_output_reference, resolve_output_reference, strip_run_id` (~67)
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `split_run_path(relative) -> (identity, run_id, subfolder)` from `dw/runs.py` (stage 1).
- Produces: `_iter_gallery_files(root, group_runs=True)` yields `(relative_name, folder, subfolder, kind, path)`; gallery entries carry `"subfolder": str`; `GET /api/gallery?subfolder=<s>` filters; the reply carries `"subfolders": [sorted distinct values]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server.py`, after `test_gallery_paginates_and_groups_by_workflow_folder`:

```python
def test_gallery_reports_and_filters_by_subfolder(server, tmp_path):
    """A step's `result.subfolder` puts its files under
    '<identity>/<run id>/<subfolder>/'. The gallery keeps `folder` meaning
    the workflow identity - a workflow run fifty times is still one folder
    - and carries the in-run subfolder as its own axis, so 'which workflow'
    and 'which part of the run' never multiply into one filter list."""
    from PIL import Image

    from dw.runs import new_run_id

    with server(success_script) as client:
        outputs = tmp_path / "outputs"
        run_id = new_run_id({"id": "dialogue"})
        run = outputs / "dialogue" / run_id
        (run / "final").mkdir(parents=True)
        (run / "intermediate" / "shots").mkdir(parents=True)
        Image.new("RGB", (2, 2)).save(run / "final" / "dialogue-assemble.0-0.0.png")
        Image.new("RGB", (2, 2)).save(run / "intermediate" / "dialogue-shot.0-0.0.png")
        Image.new("RGB", (2, 2)).save(
            run / "intermediate" / "shots" / "dialogue-slice.0-0.0.png"
        )
        Image.new("RGB", (2, 2)).save(run / "dialogue-still.0-0.0.png")
        # a flat-layout file: no run id, so nothing to hang a subfolder on
        (outputs / "ltx").mkdir()
        Image.new("RGB", (2, 2)).save(outputs / "ltx" / "flat.png")

        full = client.get("/api/gallery").json()
        by_name = {f["name"]: f for f in full["files"]}

        # folder is the identity, subfolder is what followed the run id
        final = by_name[f"dialogue/{run_id}/final/dialogue-assemble.0-0.0.png"]
        assert final["folder"] == "dialogue"
        assert final["subfolder"] == "final"
        nested = by_name[f"dialogue/{run_id}/intermediate/shots/dialogue-slice.0-0.0.png"]
        assert nested["subfolder"] == "intermediate/shots"
        assert by_name[f"dialogue/{run_id}/dialogue-still.0-0.0.png"]["subfolder"] == ""
        assert by_name["ltx/flat.png"]["folder"] == "ltx"
        assert by_name["ltx/flat.png"]["subfolder"] == ""

        # the two axes stay separate
        assert set(full["folders"]) == {"", "dialogue", "ltx"}
        assert full["subfolders"] == ["", "final", "intermediate", "intermediate/shots"]

        # the filter narrows listing and total; combined with folder it intersects
        finals = client.get("/api/gallery?subfolder=final").json()
        assert finals["total"] == 1
        assert finals["files"][0]["subfolder"] == "final"
        both = client.get("/api/gallery?folder=dialogue&subfolder=").json()
        assert {f["name"] for f in both["files"]} == {
            f"dialogue/{run_id}/dialogue-still.0-0.0.png"
        }
        # subfolders is over the whole tree, not the filtered page
        assert finals["subfolders"] == full["subfolders"]
```

Also append to `tests/test_jobs_reused.py` (the design lists attribution of a foldered file as a server test; it is expected to PASS already — `_manifest_wrote` matches on the tail — and pins that):

```python
def test_job_for_file_attributes_a_file_in_a_subfolder(tmp_path):
    # A step's result.subfolder puts a segment between the run id and the
    # file; the recorded name and the gallery's name both carry it
    history = JobHistory(str(tmp_path / "jobs.sqlite"))
    name = "dialogue/20260912-120000-abcdef01/final/dialogue-assemble.0-0.0.png"

    _record(history, "writer", 1.0, [{"step": "assemble", "files": [name], "subfolder": "final"}])

    assert history.job_for_file(name)["id"] == "writer"
```

- [ ] **Step 2: Run the tests to verify the state**

Run: `python -m pytest tests/test_server.py -q -k reports_and_filters_by_subfolder` — Expected: FAIL, `KeyError: 'subfolder'`.
Run: `python -m pytest tests/test_jobs_reused.py -q` — Expected: PASS (pinning test; if it fails, stop and report rather than patching `jobs.py`).

- [ ] **Step 3: Implement**

In `dw/server/app.py`, change the runs import to:

```python
from ..runs import is_output_reference, resolve_output_reference, split_run_path
```

(`strip_run_id` has no other use in this file once the walker changes — check with `grep -n strip_run_id dw/server/app.py`; if another use exists, keep both names.)

Rewrite `_iter_gallery_files`:

```python
    def _iter_gallery_files(root, group_runs=True):
        """Every media file under a directory tree. Yields (relative_name,
        folder, subfolder, kind, path) - relative_name always uses '/' so
        it round-trips through a URL the same way on every platform.

        With group_runs (the gallery's own use, over the output directory):
        recurses into the per-workflow subfolders (dw/workflow.py's
        effective_output_dir writes each run under '<workflow
        identity>/<run id>/', and mirrors a workflow's position under a
        'workflows' tree in the flat layout). The folder a file is grouped
        under is the identity - the run id is dropped, so a workflow run
        fifty times is one folder in the filter, not fifty - and whatever
        followed the run id is the subfolder, the part of the run a step's
        'result.subfolder' put it in ('final', 'intermediate'). A flat-layout
        path has no run id to anchor on, so its subfolder is '' and its
        whole directory is the folder, as it always was.

        Without it (the asset library's use, which has no run ids to strip):
        folder is just the plain relative directory and subfolder is ''."""
        for current, _dirs, names in os.walk(root):
            rel_root = os.path.relpath(current, root)
            directory = "" if rel_root == "." else rel_root.replace(os.sep, "/")
            for name in names:
                extension = os.path.splitext(name)[1].lower()
                kind = MEDIA_KINDS.get(extension)
                if kind is None:
                    continue
                relative_name = name if not directory else f"{directory}/{name}"
                if group_runs:
                    folder, _run_id, subfolder = split_run_path(relative_name)
                else:
                    folder, subfolder = directory, ""
                yield relative_name, folder, subfolder, kind, os.path.join(current, name)
```

In `_gallery_entries`, unpack five and add the key:

```python
        for relative_name, folder, subfolder, kind, path in files:
            ...
            entries.append(
                {
                    "name": relative_name,
                    "folder": folder,
                    "subfolder": subfolder,
                    ...
```

At the asset-library caller (~2334), unpack five as well: `for relative_name, folder, _subfolder, kind, path in files:` (match whatever names that loop uses).

Change the route:

```python
    @app.get("/api/gallery")
    def gallery(
        limit: int = 200,
        offset: int = 0,
        folder: Optional[str] = None,
        subfolder: Optional[str] = None,
        ws: Workspace = Depends(selected_workspace),
    ):
        """A page of media files in the output directory, newest first.
        Stateless by design - the gallery survives server restarts because
        it reads the directory tree, not job history. 'folders' lists every
        distinct workflow folder present (over the whole directory, not just
        this page), for the UI's folder filter - a run id is not a folder of
        its own, so a workflow's runs group together; '' stands for files
        saved directly at the output root, and is itself always a member so
        that folder-less outputs stay selectable once anything is nested.
        'subfolders' is the other axis, over the whole directory the same
        way: the in-run subfolders steps wrote into ('final',
        'intermediate'), '' for files at a run's root. `folder` and
        `subfolder` filter independently and intersect when both are given."""
        entries = _gallery_entries(ws.outputs, ws)
        folders = sorted({e["folder"] for e in entries} | {""})
        subfolders = sorted({e["subfolder"] for e in entries} | {""})
        if folder is not None:
            entries = [e for e in entries if e["folder"] == folder]
        if subfolder is not None:
            entries = [e for e in entries if e["subfolder"] == subfolder]
        offset = max(0, offset)
        limit = max(0, limit)
        page = entries[offset : offset + limit]
        return {
            "files": page,
            "total": len(entries),
            "offset": offset,
            "limit": limit,
            "folders": folders,
            "subfolders": subfolders,
            "workspace": ws.name,
        }
```

- [ ] **Step 4: Run the server tests**

Run: `python -m pytest tests/test_server.py tests/test_server_workspaces.py -q`
Expected: all PASS. If a test asserts the exact key set of a gallery entry or reply, add `subfolder` / `subfolders` to that assertion and name the test in the commit body.

- [ ] **Step 5: Commit**

```bash
git add dw/server/app.py tests/test_server.py tests/test_jobs_reused.py
git commit -m "feat(server): gallery entries carry subfolder; ?subfolder= filter and subfolders list

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: MCP `list_gallery(subfolder=)` and the tool descriptions

**Files:**
- Modify: `dw_mcp/catalog.py:140-142`, `dw_mcp/server.py` (`list_gallery` ~300, `validate_workflow` ~547, `save_workflow` ~577, `get_job` ~696)
- Test: `tests/test_mcp_catalog.py`

**Interfaces:**
- Consumes: `GET /api/gallery?subfolder=` (Task 1).
- Produces: `catalog.list_gallery(client, limit=50, subfolder=None) -> dict` — sends `subfolder` only when it is not `None` (an empty string IS sent: it means "files at a run's root"); MCP tool `list_gallery(limit: int = 50, subfolder: str | None = None)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mcp_catalog.py` after `test_list_gallery_sends_its_limit`:

```python
def test_list_gallery_sends_a_subfolder_only_when_given():
    client, seen = recording_client()
    catalog.list_gallery(client, limit=7)
    assert "subfolder" not in seen["params"]

    client, seen = recording_client()
    catalog.list_gallery(client, subfolder="final")
    assert seen["params"]["subfolder"] == "final"

    # '' is a real filter - files at a run's root - not "no filter"
    client, seen = recording_client()
    catalog.list_gallery(client, subfolder="")
    assert seen["params"]["subfolder"] == ""
```

Check how `recording_client` records params (read its definition at the top of the file); if it stringifies values, `""` stays `""`.

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_mcp_catalog.py -q -k subfolder`
Expected: FAIL — `TypeError: list_gallery() got an unexpected keyword argument 'subfolder'`.

- [ ] **Step 3: Implement the catalog function**

```python
def list_gallery(client, limit=50, subfolder=None):
    """Generated media in the output directory, newest first. `subfolder`
    narrows to one in-run subfolder ('final', 'intermediate', '' for files
    at a run's root); None means every file."""
    params = {"limit": limit}
    if subfolder is not None:
        params["subfolder"] = subfolder
    return client.get_json("/api/gallery", params=params)
```

- [ ] **Step 4: Update the MCP tool and the three descriptions**

In `dw_mcp/server.py`, `list_gallery`:

```python
    def list_gallery(limit: int = 50, subfolder: str | None = None) -> dict:
        """List generated output files, newest first. A name is
        <workflow>/<run id>/<file>, where <file> may itself sit in a
        subfolder the step chose (`final/episode.mp4`) - the form
        `get_output_image`, `get_output_text`, `download_output`,
        `keep_output` and `delete_output` all take, and the form an
        "output:" reference in a later workflow is built from. Each entry
        carries `folder` (the workflow) and `subfolder` (the part of the run:
        by convention `final` is the deliverable and `intermediate` the
        scratch work, '' when the step chose none); `subfolder=` filters on
        the latter, so `subfolder="final"` is "what did these runs
        deliver". Each entry also carries a ready-made `url` for viewing the
        file over HTTP, already scoped to the right workspace; use it as
        given rather than composing one from the name."""
        return catalog.list_gallery(client, limit=limit, subfolder=subfolder)
```

`get_job` — replace the docstring with:

```python
        """Get a job's status, argument warnings, output manifest, error and
        traceback. The manifest names each step's files the way
        `get_output_image`, `download_output` and `keep_output` take them,
        and each entry's `subfolder` says what kind of output the step
        declared - by convention `final` is the deliverable, `intermediate`
        the scratch work, and '' a step that said nothing. A step served
        from the step cache is marked `reused` and reports the earlier run's
        files. When a job failed, the error and traceback here are what to
        read before changing anything."""
```

`save_workflow` — append one sentence to its docstring, before the closing quotes:

```
        A workflow stored for reuse should mark each saving step's
        `result.subfolder` - `final` for the step whose output the user will
        be shown, `intermediate` for the rest - so a later consumer can tell
        the deliverable from the scratch files without knowing the workflow.
```

`validate_workflow` — append one sentence:

```
        A `result.subfolder` or `file_base_name` that cannot be written (a
        `..`, a backslash, a separator in `file_base_name`) is reported here
        at its JSON path, after `for_each` expansion.
```

- [ ] **Step 5: Run the MCP tests**

Run: `python -m pytest tests/test_mcp_catalog.py tests/test_mcp_server.py tests/test_mcp_media.py tests/test_server_mcp.py -q`
Expected: all PASS. `tests/test_mcp_server.py` lists tool names and may snapshot signatures — if it asserts `list_gallery`'s parameters, add `subfolder` there and name the test in the commit body.

- [ ] **Step 6: Commit**

```bash
git add dw_mcp/catalog.py dw_mcp/server.py tests/test_mcp_catalog.py
git commit -m "feat(mcp): list_gallery(subfolder=); get_job/save_workflow/validate_workflow state the subfolder convention

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Docs — SERVER, MCP, WORKFLOW_GUIDE and the CLAUDE.md mirror

**Files:**
- Modify: `docs/SERVER.md` (the `GET /api/gallery` bullet ~297; the `GET /api/jobs/{id}` row ~141; the `step_start`/`step_end` row ~165), `docs/MCP.md` (the `list_gallery` row ~229; the `get_job` row ~286), `docs/WORKFLOW_GUIDE.md` (*Authoring a workflow from an agent* — new `###` subsection before `### Being found next time` ~515; *Result Configuration* ~533), `CLAUDE.md` (*Type System* bullet list)

No code, so no TDD; the check is that the text agrees with the code as landed and that the two mirrors move together.

- [ ] **Step 1: `docs/SERVER.md`**

Extend the `GET /api/gallery` bullet (~297) so it reads:

```markdown
- `GET /api/gallery`, `GET /api/gallery/{name}/metadata`,
  `DELETE /api/gallery/{name}` — outputs and their embedded metadata. Each
  gallery entry carries `folder` (the workflow identity, the run id dropped)
  and `subfolder` (what followed the run id - the `final`/`intermediate` a
  step's `result.subfolder` chose, `''` when it chose none); `?folder=` and
  `?subfolder=` filter independently, and the reply's `folders` and
  `subfolders` list every distinct value over the whole tree, `''` always a
  member of each so root-level files stay selectable
```

In the `GET /api/jobs/{id}` row (~141), after "carries `reused: true`.", insert: "Every entry carries `subfolder` - the in-run subfolder the step's `result.subfolder` chose, `''` for none."

In the `step_start` / `step_end` row (~165), change "`files` at the end" to "`files` and `subfolder` at the end".

- [ ] **Step 2: `docs/MCP.md`**

Replace the `list_gallery` row (~229) with:

```markdown
| `list_gallery(limit=50, subfolder=None)` | `limit`, `subfolder` | List generated output files, newest first. A name is `<workflow>/<run id>/<file>`, where `<file>` may sit in the subfolder the step chose (`final/episode.mp4`); each entry carries `folder` (the workflow) and `subfolder` (`final`, `intermediate`, or `''`), and `subfolder="final"` lists only deliverables. Each entry also carries a ready-made `url`, already scoped to the workspace that made it - a hand-built `/outputs/<name>` URL 404s for anything but the default workspace |
```

In the `get_job` row (~286), after "output manifest," insert "(each entry's `subfolder` says whether the step declared its output `final`, `intermediate`, or nothing)".

- [ ] **Step 3: `docs/WORKFLOW_GUIDE.md`**

Insert before `### Being found next time` (in *Authoring a workflow from an agent*). The block below is fenced with four backticks because it contains a three-backtick fence of its own:

````markdown
### Saying which output is the deliverable

A run writes everything into one directory, so a finished episode sits
beside the twenty scratch files that went into it. A step's `result` block
can name a subfolder of the run directory for its files:

```json
"result": { "content_type": "video/mp4", "subfolder": "final" }
```

The convention is two names: `final` for a step whose output the user will
be shown, `intermediate` for everything else. The engine treats no name
specially and applies no default - a step that says nothing writes to the
run's root as it always has - but the gallery, `get_job` and `list_gallery`
all carry the value, so a consumer that follows the convention can tell the
deliverable from the scratch without knowing the workflow. Mark every saving
step of a multi-step workflow; a one-step workflow needs nothing.

The value is a relative path of any depth (`shots/act-1`), may be a
`variable:` or, inside a `for_each` step, an `item:` reference, and follows
the `output:` segment rule - each segment starts with a letter, digit or
underscore; `..`, a backslash and a leading `.` are refused - so every
subfolder written is one a later workflow can name:
`output:dialogue-short/latest/final/episode.mp4`. A bad value is a
validation error at its JSON path. `file_base_name` is a name, not a path:
a separator there is refused, and `subfolder` is the way to place a file.
````

In *Result Configuration* (~533), change the JSON example to:

```json
"result": {
    "content_type": "image/jpeg",
    "save": true,
    "file_base_name": "custom_prefix",
    "subfolder": "final"
}
```

and add after the content-types paragraph: "`subfolder` places the step's files in a subfolder of the run directory - see *Saying which output is the deliverable* above. `file_base_name` may not contain a path separator."

- [ ] **Step 4: `CLAUDE.md` mirror**

Stage 1 already wrote a `**Result subfolders**` bullet under *Critical Gotchas* holding the mechanics (`dw/subfolders.py`, `step_output_dir`, `SUBFOLDER_PATTERN`, containment, manifest/`step_end`). Do not repeat those. Two edits:

(a) In the *Type System* bullet list (the one beginning "`arguments.py` + `type_helpers.py` handle dynamic type conversion"), add a bullet after the `prompt:` bullet carrying the *convention* only:

```markdown
- A step's `result.subfolder` names a subfolder of the run directory for that step's
  files - by convention `final` for the deliverable and `intermediate` for the rest; any
  relative path (`shots/act-1`); `variable:`/`item:` allowed; no default. Mechanics under
  *Result subfolders* in Critical Gotchas
```

(b) In the existing `**Result subfolders**` gotcha, change its last sentence "`file_base_name` may not contain a separator - it is a name, not a path" to:

```markdown
  Gallery entries carry it too; `GET /api/gallery?subfolder=` and MCP
  `list_gallery(subfolder=)` filter on it. `file_base_name` may not contain a separator -
  it is a name, not a path
```

- [ ] **Step 5: Check the guides still load and the suite is green**

Run: `python -m pytest tests/test_mcp_guides.py tests/test_docs.py -q 2>/dev/null || python -m pytest tests/test_mcp_guides.py -q`
Then: `python -m pytest tests/ -q --ignore=tests/test_plugin_skills.py`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add docs/SERVER.md docs/MCP.md docs/WORKFLOW_GUIDE.md CLAUDE.md
git commit -m "docs: subfolder on gallery entries, list_gallery filter, the final/intermediate convention in the authoring guide and its CLAUDE.md mirror

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Two engine tests the stage-1 review left unpinned

**Files:**
- Test: `tests/test_runs.py` (class `TestSubfolders`)

No engine change. Both tests pin behaviour that is correct by inspection; if either fails, stop and report — do not patch the engine.

- [ ] **Step 1: Write the tests**

Add to `TestSubfolders`:

```python
    def test_a_parents_subfolder_does_not_move_a_childs_files(
        self, tmp_path, fake_pipeline
    ):
        # A 'workflow' step's own result block governs what the parent saves
        # from the child's return value; the child's steps place their own
        # files, into the run directory they inherit
        from dw.workflow import Workflow

        tree = tmp_path / "workflows"
        tree.mkdir()
        (tree / "child.json").write_text(json.dumps(_workflow_definition()))
        parent = {
            "id": "parent",
            "seed": 7,
            "steps": [
                {
                    "name": "child",
                    "workflow": {"path": "child.json"},
                    "result": {"subfolder": "final"},
                }
            ],
        }
        Workflow(parent, str(tmp_path / "out"), str(tree / "Parent.json")).run({})
        (run,) = (tmp_path / "out" / "Parent").iterdir()
        assert (run / "runs_test-gen0.0-0.0.png").is_file()
        assert not (run / "final" / "runs_test-gen0.0-0.0.png").exists()

    def test_a_chain_spill_lands_in_the_steps_subfolder(self, tmp_path, fake_pipeline):
        # save_segments writes through the pipeline wrapper's output_dir,
        # which create_step_action points at the step's subfolder
        from dw.pipeline_processors.chain import run_chain
        from dw.workflow import Workflow
        from tests.test_chain import FakePipeline, video_output

        workflow = Workflow(_foldered_definition(), str(tmp_path), "/w/workflows/Gyre.json")
        workflow._run_dir = str(tmp_path / "Gyre" / "run")
        action = workflow.create_step_action(
            workflow.workflow_definition["steps"][0], {}, {}, 7, "cpu"
        )
        spilling = FakePipeline(
            video_output, output_dir=action.output_dir, file_prefix=action.file_prefix
        )
        run_chain(
            spilling,
            {"segments": 2, "trim_frames": 1, "fps": 4, "save_segments": True,
             "keep_segments": True},
            {},
        )
        segments = sorted((tmp_path / "Gyre" / "run" / "final").glob("*.segment-*.mp4"))
        assert len(segments) == 2
```

Before running, check: (a) that `Workflow` resolves a sub-workflow's `path` relative to the parent's `file_spec` directory (read the `workflow` branch of `create_step_action`); if it needs the parent file to exist, write `parent` to `tree / "Parent.json"` and load it with `workflow_from_file`; (b) the exact child file name — the child's id is `runs_test`, its step `gen0` — and adjust the asserted name to what the run wrote if the workflow-step naming differs, keeping the assertion that it is at the run root and not under `final/`; (c) that `tests/test_chain.py` exports `FakePipeline` and `video_output` at module level and `FakePipeline` accepts `output_dir`/`file_prefix` kwargs (it does in `TestSaveSegments.make_pipeline`); (d) the attribute name the `Pipeline` wrapper stores its file prefix under (`file_prefix` per the constructor kwarg — confirm in `dw/pipeline_processors/pipeline.py`).

- [ ] **Step 2: Run**

Run: `python -m pytest tests/test_runs.py -q`
Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_runs.py
git commit -m "test(runs): a parent's subfolder leaves a child's files alone; a chain spill lands in the step's subfolder

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Design status

**Files:**
- Modify: `docs/proposals/output-folders.md:3`

- [ ] **Step 1: Update the status line** to:

```markdown
Status: **stages 1-2 (engine, server/MCP) implemented; stages 3-4 pending**. Written for MCP feedback ticket T016;
```

- [ ] **Step 2: Commit**

```bash
git add docs/proposals/output-folders.md
git commit -m "docs: output-folders design marks stage 2 done

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage (stage 2 phasing):** gallery `subfolder`/`?subfolder=`/`subfolders` → Task 1; MCP `list_gallery` parameter → Task 2; `get_job`/`save_workflow`/`list_gallery` descriptions → Task 2 (plus `validate_workflow` — an addition beyond the design's list, because *Where the checks live* makes it the surface that reports a bad value); SERVER/MCP/GUIDE docs and the CLAUDE.md mirror → Task 3; the design's *Tests* items for the server → Task 1 (entries, filter, list, `job_for_file` attribution of a foldered file); stage-1 deferred tests → Task 4. Not in scope: the UI (`GalleryFile` type, filter control, job-page grouping — stage 4), templates and skills (stage 3), the `step_subfolder` double evaluation (Fable: harmless; left).

**Placeholders:** none. Task 4 names four things to confirm before running and says what to keep constant.

**Type consistency:** `split_run_path -> (identity, run_id, subfolder)`; `_iter_gallery_files` five-tuple `(relative_name, folder, subfolder, kind, path)` in both callers; `catalog.list_gallery(client, limit=50, subfolder=None)` matches `server.list_gallery(limit=50, subfolder=None)`.
