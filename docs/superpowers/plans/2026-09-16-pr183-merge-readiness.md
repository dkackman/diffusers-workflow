# PR #183 Merge-Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PR #183 (`develop` → `master`, https://github.com/dkackman/diffusers-workflow/pull/183) mergeable by clearing its CI blocker and fixing the confirmed defects a full review found, each with a regression test.

**Architecture:** Every task is an isolated, independently testable fix on one branch cut from `origin/develop`. Tasks 1–4 are the merge gate (CI blocker + a destructive-path bug + a 404→500 regression + a warnings-channel pollution that breaks the regression suites). Tasks 5–8 are confirmed engine defects. Tasks 9–10 are the two cheap hot-path performance fixes. Task 11 lands everything on `develop` and re-verifies the PR. No task changes an MCP tool signature, a workflow schema, or a template.

**Tech Stack:** Python 3.10+, FastAPI (`dw/server/app.py`), pytest (`tests/`), ruff 0.16.x (format + lint, config in `pyproject.toml`), `gh` CLI. Tests run on CPU; nothing here needs a GPU.

**Spec:** Inline — the "Background: review findings" section below is the spec. There is no separate document. The review was done on `origin/develop` @ `da568c6` on 2026-09-16; line numbers below are from that commit and may drift by a few lines.

## Global Constraints

- Work on a branch cut from `origin/develop`; never commit to `master`. Merge to `develop` in Task 11 (repo convention: branches merge to `develop`, releases are cut from `master`).
- Every commit must pass `ruff format --check dw dw_mcp tests && ruff check dw dw_mcp tests` — CI runs exactly these (`.github/workflows/ci.yml` lines 45–50) before `pytest -q`. Run both before every commit.
- Commit message form: `fix(<area>): #<issue> - <what changed>` when an issue applies, otherwise `fix(<area>): <what changed>`. Areas used in this repo: `server`, `events`, `elision`, `workflow`, `mcp`, `validation`, `ci`.
- Do not modify `regression-suite-*.md` files (they live in a different repo and are the tester's).
- Do not change any MCP tool's parameter list, any `dw/workflow_schema.json` entry, or any file under `workflows/`.
- Ruff line length is the default (88). Ruff's `select = ["E", "F"]`, `E501` ignored.
- Python environment: the local Mac checkout's `.venv` has no torch. The full suite runs on the `lem` box: `ssh lem`, checkout at `/home/don/diffusers-workflow`, interpreter `/home/don/diffusers-workflow/venv/bin/python` (torch 2.14 + pytest). Run the suite from a throwaway worktree there (see Task 0) so the running server's checkout is never touched. Single test files that import only `dw.elision`, `dw.variable_constraints`, `dw.kernel_availability`, `dw.server.jobs` or `dw.server.observed_cost` may still need torch transitively via `dw/__init__.py` — assume every test needs `lem`.

---

## Background: review findings (the spec)

State of the PR at review time: 69 commits, 149 files, +13,519/−670. `mergeable: MERGEABLE`. CI `backend` job **fails at the Format check**, so Lint and Tests never ran. Full suite run by hand on `lem` at the PR head: **4927 passed, 16 skipped**. The raw diff shows `version` going `0.4.0-beta.5 → 0.4.0-beta.4` in `pyproject.toml` and `plugins/dw/.claude-plugin/plugin.json`; that is a diff artifact (`release 0.4.0-beta.5` was committed on `master` only). `git merge-tree origin/master origin/develop` keeps `beta.5` in both files — **not a bug**, but Task 11 merges `master` back into `develop` so it stops looking like one.

| # | Severity | Where | Defect | Task |
|---|----------|-------|--------|------|
| B | Blocker | 21 files | `ruff format --check` fails; 2 unused imports (F401) | 1 |
| 1 | Fix before merge | `dw/events.py:126`, `dw/server/jobs.py:483-494` | Watchdog `phase_stall` reports are `warning` events; `Job._note_progress` appends every warning to the persisted `warnings` list, deduping on exact text, and each tick carries a different seconds value. `loading` emits one event per component and the threshold is a global 30 s, so every cold load (60–140 s) leaves 2–4 stall lines on a *succeeded* job. The regression suites assert `warnings: []` on succeeded runs. | 2 |
| 2 | Fix before merge | `dw/server/app.py:3036` `_asset_origin`, `:3265` `delete_asset` | Root index 0 is labelled `workspace` unconditionally, but `_asset_roots` drops `ws.assets` when it is `None` or not yet a directory. With `--examples-dir` and no workspace library, the examples tree becomes root 0 → `writable: true` → the new Assets page shows checkboxes + bulk Delete → `delete_asset` skips its examples 403 and `os.remove`s example files. | 3 |
| 3 | Fix before merge | `dw/server/app.py:2384` `_resolution_roots`, `:2318` `_asset_in` | `_resolution_roots` returns `[ws.assets]` = `[None]` when no library is configured; `_asset_in` does `os.path.join(None, name)` → `TypeError` → HTTP 500 on `GET /inputs/<name>` and `GET /api/gallery/asset:<name>/metadata`. Master's loop had `if not root: continue`. The `if not roots:` "no asset library" 404 branch is unreachable. | 4 |
| 4 | Should fix | `dw/elision.py:55-69` `_saves` | A `workflow:` (sub-workflow) step with no `result` block is treated as "writes nothing", so if nothing downstream references it, elision drops it — but its child workflow writes its own files. Job then "succeeds" without that output. | 5 |
| 5 | Should fix | `dw/server/app.py` `_iter_orphan_runs`, `dw_mcp/server.py:385-389` | Orphan detection is "no file with an image/video/audio extension". A `text`-shape run (`.txt`/`.json` output) is listed as an orphan; the MCP docstring says "list, then delete each name" and names only `utility` as the false positive. | 6 |
| 6 | Should fix | `dw/workflow.py:1508-1516` | Cross-job `_prior_step_keys` (#150): when step `a`'s key changes, its old key is popped from the cache even if step `b` (same previous key) still needs it → cold reload of a resident model and both stacks held at once. | 7 |
| 7 | Should fix | `dw/workflow.py:767-771` | `apply_constraints` runs before `resolve_variable_values`, so a list entry `{"num_frames": "variable:x"}` is skipped by the run-time backstop (`_as_integer("variable:x")` → `None`) and substituted afterwards. Static validation still catches it. | 8 |
| P1 | Perf | `dw/kernel_availability.py:92-114` | `kernel_availability_fault` imports, `inspect.getsource`s and *constructs* the processor (Hub kernel fetch) on every validate/submit/rerun, unmemoized. | 9 |
| P2 | Perf | `dw/server/app.py:262`, `dw/server/observed_cost.py:286-300` | `attach_observed` calls `observed()` per catalog entry and `rows_for` runs the `SELECT COUNT(*), MAX(finished_at)` watermark query under the history lock each time → ~70–80 sqlite connections + table scans per `list_workflows`. | 10 |

Verified sound, no task needed: asset archive/symlink containment (`validate_asset_reference` + `validate_path`), MCP alias handling (#179), cache invalidation on deletes (#177), observed-cost math, elision reference coverage for pipeline steps, #150 not reopened by elision, schema tightening vs shipped workflows, template defaults vs their constraints, UI routes vs server, selection-store rescoping, security regex widening (#162), `workflow_details` returns a fresh dict so `attach_observed` cannot poison `_workflow_detail_cache`.

Known behaviour changes that are intentional and **not** in scope: #166's default-checking now answers `valid: false` for a bare `validate_workflow(name=…)` on ~6 templates whose defaults are placeholder assets (`asset:shot_1.mp4`, `asset:score.wav`, …); `task_signature_errors` turns unknown/missing task arguments into hard errors.

---

### Task 0: Branch and baseline

**Files:** none modified.

- [ ] **Step 1: Cut the branch from the PR head**

```bash
cd ~/src/dkackman/diffusers-workflow
git fetch origin develop master
git checkout -b fix/pr183-merge-readiness origin/develop
git log --oneline -1   # expect: da568c6 Merge pull request #182 from dkackman/assets-page (or newer)
```

- [ ] **Step 2: Reproduce the CI failure locally**

Run: `ruff format --check dw dw_mcp tests; ruff check dw dw_mcp tests`
Expected: `21 files would be reformatted, 229 files already formatted` and `Found 2 errors.`

- [ ] **Step 3: Set up the test worktree on `lem`**

The suite needs torch. Create a throwaway worktree on `lem` that tracks this branch; every "Run:" line below that says `on lem` means: push the branch, then run this.

```bash
git push -u origin fix/pr183-merge-readiness
ssh lem 'cd ~/diffusers-workflow && git fetch -q origin fix/pr183-merge-readiness \
  && (git worktree remove --force /tmp/pr183-fix 2>/dev/null; true) \
  && git worktree add -q --detach /tmp/pr183-fix origin/fix/pr183-merge-readiness \
  && cd /tmp/pr183-fix && ~/diffusers-workflow/venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -3'
```

Expected: `4927 passed, 16 skipped` (numbers may be slightly higher if develop moved).

For subsequent tasks, after pushing, refresh the worktree and run one file:

```bash
ssh lem 'cd /tmp/pr183-fix && git fetch -q origin fix/pr183-merge-readiness && git checkout -q --detach origin/fix/pr183-merge-readiness \
  && ~/diffusers-workflow/venv/bin/python -m pytest -q -p no:cacheprovider tests/<file>.py 2>&1 | tail -5'
```

---

### Task 1: Clear the CI format/lint failure

**Files:**
- Modify: the 21 files `ruff format` names (`dw/result.py`, `dw/server/app.py`, `dw/tasks/audio_utils.py`, `dw/tasks/video_utils.py`, `dw_mcp/authoring.py`, `dw_mcp/diagnose.py`, `tests/test_audio_utils.py`, `tests/test_events.py`, `tests/test_h3_schedule.py`, `tests/test_ltx2_diffusion_decode.py`, `tests/test_ltx2_ic_loras.py`, `tests/test_ltx_prompt_library.py`, `tests/test_pipeline_components.py`, `tests/test_plan.py`, `tests/test_reference_names.py`, `tests/test_result.py`, `tests/test_server.py`, `tests/test_server_downloads.py`, `tests/test_task_signature_errors.py`, `tests/test_validate_arguments.py`, `tests/test_variable_constraints.py`)
- Modify: `tests/test_content_types.py:3` (drop unused `InvalidInputError` import), `tests/test_kernel_availability.py:1` (drop unused `import pytest`)

- [ ] **Step 1: Apply the formatter and the auto-fixes**

```bash
ruff format dw dw_mcp tests
ruff check --fix dw dw_mcp tests
```

- [ ] **Step 2: Verify both checks are clean**

Run: `ruff format --check dw dw_mcp tests && ruff check dw dw_mcp tests`
Expected: `250 files already formatted` and `All checks passed!`

- [ ] **Step 3: Confirm the diff is whitespace/line-wrap only plus the two import removals**

Run: `git diff --stat | tail -1 && git diff -w --stat | tail -1`
Expected: the `-w` stat is tiny (only the two deleted import lines and lines where the formatter joined/split expressions). Eyeball `git diff -w` — no logic changes.

- [ ] **Step 4: Run the suite on lem**

Push, then run the full suite per Task 0 Step 3. Expected: same pass count as baseline.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "fix(ci): ruff format and drop two unused test imports

ruff format --check failed on 21 files and ruff check on two F401s, which
stopped the backend CI job before Lint and Tests ran (PR #183)."
```

---

### Task 2: Keep phase-stall reports out of the job's persisted `warnings`

**Files:**
- Modify: `dw/server/jobs.py:483-494` (the `elif kind == "warning":` branch of `Job._note_progress`)
- Create: `tests/test_job_warnings.py`

**Interfaces:**
- Consumes: `dw.server.jobs.Job(spec: dict)`; `Job._note_progress(event: dict)`; watchdog events have `{"event": "warning", "kind": "phase_stall", "message": ..., "phase": ..., "seconds_since_phase_start": ...}` (emitted by `RunContext._watchdog_loop`, `dw/events.py:118-131`).
- Produces: nothing new. `Job.warnings` no longer contains `phase_stall` messages; the event log (`Job.events`) still does.

Design decision: the event log is the right home for a stall report — it is a moment, not a fact about the artifact. `warnings` is the channel a caller polls after the run (#82) and the regression suites assert on it. Filter on `kind`, do not change the event type: `tests/test_events.py::test_watchdog_event_carries_the_required_fields` pins the event shape.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_job_warnings.py
"""What lands in a finished job's `warnings` list.

`warnings` is where a caller who polled the finished job looks (#82); the
event log keeps the moment something happened. A phase-stall report (#176)
is a moment - "still in phase 'loading', 60.0s ..." on a job that then
succeeded is not a warning about the result - so it stays in the event log
and out of `warnings`, where the regression suites assert `warnings: []`
on a clean run.
"""

from dw.server.jobs import Job


def _job():
    return Job({"workflow_name": "x"})


def test_an_ordinary_warning_is_persisted_with_its_step():
    job = _job()
    job._note_progress({"event": "step_start", "step": "encode"})
    job._note_progress(
        {"event": "warning", "kind": "audio_clipped", "message": "peak above 0 dBFS"}
    )
    assert job.warnings == ["encode: peak above 0 dBFS"]


def test_a_phase_stall_report_stays_out_of_the_persisted_warnings():
    job = _job()
    job._note_progress({"event": "step_start", "step": "load"})
    for seconds in (35.0, 65.0, 95.0):
        job._note_progress(
            {
                "event": "warning",
                "kind": "phase_stall",
                "phase": "loading",
                "seconds_since_phase_start": seconds,
                "message": (
                    f"still in phase 'loading', {seconds:.1f}s since it started "
                    "with no progress event"
                ),
            }
        )
    job._note_progress(
        {"event": "warning", "kind": "audio_clipped", "message": "peak above 0 dBFS"}
    )
    assert job.warnings == ["load: peak above 0 dBFS"]
```

- [ ] **Step 2: Run it to verify it fails**

Run on lem: `pytest -q tests/test_job_warnings.py`
Expected: `test_a_phase_stall_report_stays_out_of_the_persisted_warnings` FAILS with three `load: still in phase 'loading', ...` entries in the list; the first test passes.

- [ ] **Step 3: Filter the stall kind in `_note_progress`**

In `dw/server/jobs.py`, the branch currently reads:

```python
        elif kind == "warning":
            # Both channels, on purpose: ...
            message = event.get("message")
            if message:
                named = f"{self.step_name}: {message}" if self.step_name else message
                if named not in self.warnings:
                    self.warnings.append(named)
```

Change it to:

```python
        elif kind == "warning":
            # Both channels, on purpose: the event log keeps the moment it
            # happened, `warnings` keeps it where a caller who polled the
            # finished job will actually look, since a warning about the
            # artifact outlives the run that noticed it (#82). The step it
            # fired in is the run's, not the warning's - the engine warns
            # from inside a step without knowing which one it is.
            #
            # A phase-stall report (#176) is the exception: it is a moment,
            # not a fact about the result - a 90 s cold load says "still in
            # phase 'loading'" three times and then succeeds - so it stays
            # in the event log only. `warnings` is the channel a consumer
            # reads after the run, and the regression suites assert it is
            # empty on a clean one.
            message = event.get("message")
            if message and event.get("kind") != "phase_stall":
                named = f"{self.step_name}: {message}" if self.step_name else message
                if named not in self.warnings:
                    self.warnings.append(named)
```

- [ ] **Step 4: Run the new test and the neighbours**

Run on lem: `pytest -q tests/test_job_warnings.py tests/test_events.py tests/test_server.py -k "warning or watchdog or stall"`
Expected: all PASS.

- [ ] **Step 5: Document it on the constant**

In `dw/events.py`, extend the comment above `PHASE_STALL_THRESHOLD_SECONDS` (line ~15) with one sentence after "...long before a human would give up on it.":

```python
# The report goes to the event log only - the job's persisted `warnings`
# list filters `kind == "phase_stall"` (dw/server/jobs.py), because a stall
# that resolved is not a warning about the result.
```

- [ ] **Step 6: Format, lint, commit**

```bash
ruff format dw/server/jobs.py dw/events.py tests/test_job_warnings.py && ruff check dw dw_mcp tests
git add dw/server/jobs.py dw/events.py tests/test_job_warnings.py
git commit -m "fix(events): #176 - a phase-stall report stays out of the job's warnings

The watchdog emits its report as a warning event, and Job._note_progress
persisted every warning message; each tick carries a different seconds
value so none deduped, and a normal 60-140 s cold load left 2-4 stall
lines on a succeeded job. The event log keeps them; warnings does not."
```

---

### Task 3: Label an asset library by which directory it is, not by its position

**Files:**
- Modify: `dw/server/app.py` — `_asset_origin` (~line 3036), its two call sites in `list_assets` (~3071–3092) and the one in `delete_asset` (~3290)
- Test: `tests/test_server.py` (append)

**Interfaces:**
- Consumes: `_asset_roots(ws) -> list[str]` (abspath'd, existing dirs only, in order: own, common, examples…); `_common_assets(ws) -> str | None`; `ws.assets -> str | None`; constants `WORKSPACE_ORIGIN`, `COMMON_ORIGIN`, `EXAMPLES_ORIGIN`.
- Produces: `_asset_origin(ws, root: str) -> str` — **signature change**: the `index` parameter is removed. Callers pass the root path only.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server.py` (it already imports `create_app`, `JobManager`, `ScriptedWorkerManager`, `success_script`, `TestClient`):

```python
def test_an_examples_library_is_read_only_even_when_it_is_the_only_root(tmp_path):
    """A workspace whose own library does not exist yet drops out of the
    search path, which made the examples tree root 0 - and root 0 was
    labelled 'workspace', writable, deletable. With the Assets page's
    select-all + Delete on top of that label, example files could be
    removed through the UI. The label follows which directory a root is,
    never its position."""
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    examples = tmp_path / "examples"
    (examples / "assets").mkdir(parents=True)
    example_file = examples / "assets" / "cast.png"
    example_file.write_bytes(b"png")

    manager = JobManager(
        str(tmp_path / "outputs"),
        worker_manager=ScriptedWorkerManager(success_script),
        history_path=str(tmp_path / "jobs.sqlite"),
        workflow_dir=str(workflows),
    )
    app = create_app(
        workflow_dir=str(workflows),
        output_dir=str(tmp_path / "outputs"),
        job_manager=manager,
        # declared but never created - the shape a fresh install has
        asset_dir=str(tmp_path / "assets"),
        examples_dirs=[str(examples)],
    )
    with TestClient(app, base_url="http://localhost") as client:
        body = client.get("/api/assets").json()
        assert body["libraries"] == [
            {"origin": "examples", "dir": str(examples / "assets"), "writable": False}
        ]
        (asset,) = body["assets"]
        assert asset["origin"] == "examples"

        response = client.delete("/api/assets/cast.png")

    assert response.status_code == 403
    assert example_file.exists()
```

- [ ] **Step 2: Run it to verify it fails**

Run on lem: `pytest -q tests/test_server.py -k examples_library_is_read_only_even_when`
Expected: FAIL — `libraries[0]["origin"] == "workspace"` and the DELETE returns 200 with the file gone.

- [ ] **Step 3: Rewrite `_asset_origin` to compare directories**

Replace the function body at ~line 3036:

```python
    def _asset_origin(ws, root):
        """Which library an asset came from: this workspace's own, the one
        shared by every workspace under the root, or a read-only examples
        tree. A client that cannot tell them apart cannot say why deleting
        one answers 403.

        By directory, never by position in the search path: the workspace's
        own library drops out of `_asset_roots` until it exists, and the
        examples tree that then sits first is still nobody's to write."""
        own = ws.assets
        if own and os.path.abspath(own) == root:
            return WORKSPACE_ORIGIN
        common = _common_assets(ws)
        if common and os.path.abspath(common) == root:
            return COMMON_ORIGIN
        return EXAMPLES_ORIGIN
```

- [ ] **Step 4: Update the three call sites**

In `list_assets` (~3071–3092) there are two calls of the form `_asset_origin(ws, index, root)`; in `delete_asset` (~3290) one. Change each to `_asset_origin(ws, root)`. If an `index` variable in that loop is then unused by anything else, replace `for index, root in enumerate(roots):` with `for root in roots:` (ruff will not flag an unused loop variable, but leave the loop honest).

- [ ] **Step 5: Run the asset tests**

Run on lem: `pytest -q tests/test_server.py tests/test_server_downloads.py -k "asset or librar or shadow or archive"`
Expected: all PASS, including the new test and the existing `libraries[0] == {"origin": "workspace", ...}` assertion (the workspace dir exists in that test, so it still labels `workspace`).

- [ ] **Step 6: Format, lint, commit**

```bash
ruff format dw/server/app.py tests/test_server.py && ruff check dw dw_mcp tests
git add dw/server/app.py tests/test_server.py
git commit -m "fix(server): an asset library's origin follows its directory, not its index

_asset_roots drops a workspace library that does not exist yet, which made
an examples tree root 0 - and root 0 was labelled workspace, writable,
deletable. The Assets page's select-all + Delete made that reachable."
```

---

### Task 4: A server with no asset library answers 404, not 500

**Files:**
- Modify: `dw/server/app.py` — `_resolution_roots` (~line 2384)
- Test: `tests/test_server.py` (append)

**Interfaces:**
- Consumes: `_asset_roots(ws)`, `ws.assets` (may be `None`).
- Produces: `_resolution_roots(ws) -> list[str]` never contains `None`; may be `[]`, in which case `_asset_in` raises the existing "this workspace has no asset library" 404.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server.py`:

```python
def test_an_asset_lookup_with_no_library_configured_is_a_404(server):
    """No asset_dir, no examples: the search path is empty. That is a 404
    naming the absence, not a TypeError from joining None (the fallback
    handed _asset_in a [None] root)."""
    with server(success_script) as client:
        response = client.get("/inputs/iris.png")
        assert response.status_code == 404
        assert "no asset library" in response.json()["detail"]

        metadata = client.get("/api/gallery/asset:iris.png/metadata")
        assert metadata.status_code == 404
```

(The `server` fixture at `tests/test_server.py:185` builds `create_app` with no `asset_dir` and no `examples_dirs`, which is exactly the configuration.)

- [ ] **Step 2: Run it to verify it fails**

Run on lem: `pytest -q tests/test_server.py -k no_library_configured_is_a_404`
Expected: FAIL with status 500 (TestClient re-raises: `TypeError: expected str, bytes or os.PathLike object, not NoneType`).

- [ ] **Step 3: Guard the fallback**

Replace the return of `_resolution_roots` (~line 2395):

```python
        roots = _asset_roots(ws)
        if roots:
            return roots
        # Only a library that is at least declared can be named in a miss;
        # a server with none configured has nothing to search and nothing to
        # point at, and _asset_in turns an empty list into the 404 that says so
        return [os.path.abspath(ws.assets)] if ws.assets else []
```

- [ ] **Step 4: Run the asset tests**

Run on lem: `pytest -q tests/test_server.py -k "asset or librar" tests/test_server_downloads.py`
Expected: all PASS. `test_gallery_metadata_refuses_an_asset_that_escapes_the_library` still sees `"not found in"` (that fixture declares an `asset_dir`).

- [ ] **Step 5: Format, lint, commit**

```bash
ruff format dw/server/app.py tests/test_server.py && ruff check dw dw_mcp tests
git add dw/server/app.py tests/test_server.py
git commit -m "fix(server): no asset library is a 404, not a 500

_resolution_roots fell back to [ws.assets] when the search path was empty,
which is [None] on a server configured without one; _asset_in then joined
None. The 'no asset library' 404 branch was unreachable."
```

---

### Task 5: Elision keeps a sub-workflow step whether or not it declares a `result`

**Files:**
- Modify: `dw/elision.py:55-69` (`_saves`)
- Test: `tests/test_elision.py` (add to `class TestWhatIsKept` — the class holding `test_a_step_that_saves_is_kept` at line ~68; if the class is named differently, add beside that test)

**Interfaces:**
- Consumes: `elide_unreferenced_steps(steps, overridden=None) -> (kept, elided)`; test helpers `step(name, **extra)`, `task(name, reads=None, **extra)`, `names(steps)` already defined at the top of `tests/test_elision.py`.
- Produces: `_saves(step) -> bool` returns `True` for any step carrying a `workflow` key.

- [ ] **Step 1: Write the failing test**

```python
    def test_a_sub_workflow_step_is_kept_even_without_a_result(self):
        """A composing step's child writes its own files; the parent's
        missing `result` block says nothing about that. Dropping it would
        make the job succeed without the child's output."""
        kept, elided = elide_unreferenced_steps(
            [
                step("score", workflow={"path": "templates/minimax/music3"}),
                task("deliverable", result={"content_type": "audio/wav"}),
            ]
        )
        assert names(kept) == ["score", "deliverable"]
        assert elided == []
```

- [ ] **Step 2: Run it to verify it fails**

Run on lem: `pytest -q tests/test_elision.py -k sub_workflow_step_is_kept`
Expected: FAIL — `names(kept) == ["deliverable"]`.

- [ ] **Step 3: Teach `_saves` about composing steps**

Replace `_saves`:

```python
def _saves(step):
    """Whether the step writes a file.

    Exactly what `Result.save` asks: a `content_type` to write, and `save`
    not turned off. `save` defaults to true, so declaring a `result` with a
    content type is declaring a deliverable - and a step with no `result` at
    all writes nothing, whatever it generates, which is what makes the
    portrait steps droppable in the first place.

    A composing step is the one exception: its child workflow saves on its
    own terms, so the parent's `result` block (or its absence) says nothing
    about whether files are written. It is kept.
    """
    if "workflow" in step:
        return True
    result = step.get("result")
    if not isinstance(result, dict):
        return False
    if result.get("content_type") is None:
        return False
    return result.get("save", True) is not False
```

- [ ] **Step 4: Run the elision tests and the template sweep**

Run on lem: `pytest -q tests/test_elision.py tests/test_examples.py -k "elis or elid or no_step_is_elided"`
Expected: all PASS (shipped composing templates already carry `result`, so their expectations don't move).

- [ ] **Step 5: Format, lint, commit**

```bash
ruff format dw/elision.py tests/test_elision.py && ruff check dw dw_mcp tests
git add dw/elision.py tests/test_elision.py
git commit -m "fix(elision): #122 - a sub-workflow step is kept whether or not it has a result

Its child writes its own files, so the parent's result block says nothing
about whether the step produces output; treating it as saving nothing let
an unreferenced composing step vanish and the job succeed without it."
```

---

### Task 6: An orphan run is one that holds nothing but bookkeeping

**Files:**
- Modify: `dw/server/app.py` — `_iter_orphan_runs` (~line 2521) and its docstring
- Modify: `dw_mcp/server.py:385-389` (the `only_orphans` paragraph of `list_gallery`'s docstring)
- Test: `tests/test_server.py` (append) and `tests/test_mcp_server.py` (docstring pin, optional)

**Interfaces:**
- Consumes: run-directory bookkeeping names, which `app.py` already enumerates at ~line 1251: `workflow.json`, `manifest.json`, `job.json`.
- Produces: module-level constant `RUN_BOOKKEEPING_FILES = frozenset({"manifest.json", "workflow.json", "job.json"})` in `dw/server/app.py`, used by `_iter_orphan_runs`. Behaviour: a run directory is an orphan when every file under it (recursively) is one of those names.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_server.py` (modelled on `test_gallery_only_orphans_lists_media_less_run_directories` at ~line 1476):

```python
def test_a_text_shape_run_is_not_an_orphan(server, tmp_path):
    """#170 defined an orphan by 'no image/video/audio file', which made every
    text-shape run (enhance-prompt writes .txt) an orphan - and the MCP
    docstring tells the agent to list then delete. A run is orphaned when it
    holds nothing but its own bookkeeping."""
    from dw.runs import new_run_id

    with server(success_script) as client:
        outputs = tmp_path / "outputs"

        text_run_id = new_run_id({"id": "enhance", "seed": 1})
        text_run = outputs / "enhance" / text_run_id
        (text_run / "final").mkdir(parents=True)
        (text_run / "manifest.json").write_text("{}")
        (text_run / "workflow.json").write_text("{}")
        (text_run / "final" / "enhance-prompt.0-0.0.txt").write_text("a prompt")

        empty_run_id = new_run_id({"id": "enhance", "seed": 2})
        empty_run = outputs / "enhance" / empty_run_id
        empty_run.mkdir(parents=True)
        (empty_run / "manifest.json").write_text("{}")
        (empty_run / "job.json").write_text("{}")

        orphans = client.get("/api/gallery?only_orphans=true").json()

    assert {r["name"] for r in orphans["runs"]} == {f"enhance/{empty_run_id}"}
```

- [ ] **Step 2: Run it to verify it fails**

Run on lem: `pytest -q tests/test_server.py -k text_shape_run_is_not_an_orphan`
Expected: FAIL — both runs are listed.

- [ ] **Step 3: Define the bookkeeping set and rewrite the walk**

Near the top of `dw/server/app.py` (module level, beside the other constants — after the imports, before `_workflow_detail_cache = {}` at ~line 181):

```python
# What a run directory holds besides its outputs - the files a run writes
# about itself. A run whose directory holds nothing else is an orphan
# (see _iter_orphan_runs, #170) whatever shape its output would have had.
RUN_BOOKKEEPING_FILES = frozenset({"manifest.json", "workflow.json", "job.json"})
```

Replace `_iter_orphan_runs`:

```python
    def _iter_orphan_runs(root):
        """Run directories under `root` holding nothing but their own
        bookkeeping (RUN_BOOKKEEPING_FILES) - a run whose output was deleted
        before #134's by-name `delete_output`, or one that failed before
        writing anything. Yields (name, mtime) where `name` is the
        `<identity>/<run id>` string `delete_output` already accepts (#170).

        By what is absent, not by extension: a `text`-shape run writes .txt
        and a `utility`-shape run may write nothing the gallery lists, and
        neither is junk. This call only lists; deciding whether an entry is
        junk stays a human/agent call before `delete_output` is invoked."""
        for current, dirs, _names in os.walk(root):
            if not is_run_id(os.path.basename(current)):
                continue
            # A run directory holds no run directories of its own
            dirs[:] = []
            has_output = any(
                name not in RUN_BOOKKEEPING_FILES
                for _sub_current, _sub_dirs, sub_names in os.walk(current)
                for name in sub_names
            )
            if has_output:
                continue
            try:
                mtime = os.stat(current).st_mtime
            except OSError:
                continue
            name = os.path.relpath(current, root).replace(os.sep, "/")
            yield (name, mtime)
```

- [ ] **Step 4: Fix the MCP docstring**

In `dw_mcp/server.py`, the `list_gallery` docstring paragraph starting "`only_orphans=True` inverts the call" (~line 379–389) currently says "run directories with no media anywhere under them" and "A run with no media by design (a pure `utility`-shape workflow) matches this test too". Replace those two sentences so the paragraph reads:

```
        `only_orphans=True` inverts the call: instead of files, it returns
        run directories holding nothing but their own bookkeeping
        (manifest.json, workflow.json, job.json) as `runs`, each
        `{name, mtime}` - a run whose output was deleted before
        `delete_output` could remove it by name, or one that failed before
        writing anything, invisible to a normal listing because it has no
        file to show. `subfolder` does not apply in this mode. `name` is
        exactly what `delete_output` accepts, so clearing the backlog is
        list, then delete each name (#170). A run that wrote any file at
        all - a text-shape prompt, a utility's side output - is not listed;
        this call only lists, so deciding whether a listed entry is actually
        junk before calling `delete_output` on it is still yours to make.
```

- [ ] **Step 5: Run the gallery/orphan tests and the MCP docstring pins**

Run on lem: `pytest -q tests/test_server.py -k "orphan or gallery" tests/test_mcp_server.py`
Expected: all PASS. If a test in `tests/test_mcp_server.py` or `tests/test_mcp_catalog.py` pins the old phrase "no media anywhere", update that assertion to `"nothing but their own bookkeeping"`.

- [ ] **Step 6: Format, lint, commit**

```bash
ruff format dw/server/app.py dw_mcp/server.py tests/test_server.py && ruff check dw dw_mcp tests
git add dw/server/app.py dw_mcp/server.py tests/test_server.py
git commit -m "fix(server): #170 - an orphan run holds nothing but bookkeeping

Extension-based detection listed every text-shape run as an orphan, and
the MCP docstring tells the agent to list then delete. A run is orphaned
when no file other than manifest/workflow/job.json survives under it."
```

---

### Task 7: A superseded pipeline is not released while another step still uses it

**Files:**
- Modify: `dw/workflow.py:1508-1516` (the `prior_key` block in `create_step_action`)
- Test: `tests/test_pipeline_caching.py` (append beside `test_redefined_step_evicts_prior_pipeline_before_loading` at ~line 418)

**Interfaces:**
- Consumes: `self._prior_step_keys: dict[str, str]` (last job's step → cache key, merged across jobs by `Worker._record_step_keys`), `previous_pipelines: dict[str, Pipeline]`, `pipeline_cache_key(pipeline_def) -> str`.
- Produces: no new names. The release fires only when `prior_key` is mapped by no other step in `_prior_step_keys`; a key still shared is left for `_evict_untouched_pipelines` at the end of the run.

- [ ] **Step 1: Write the failing test**

```python
def test_a_pipeline_another_step_still_maps_to_is_not_released_as_superseded():
    """#150 carries step->key across jobs. Two steps that resolved to the
    same pipeline last time, and a rerun that changes only the first: the
    old model is still the second step's cache hit, so releasing it here
    forces a cold reload of a resident model - and holds both stacks while
    the first step's replacement loads, the exact transition #150 avoids.
    The end-of-run sweep (_evict_untouched_pipelines) drops it if nothing
    touched it."""
    from dw.workflow import pipeline_cache_key

    shared_def = {
        "configuration": {"component_type": "{Mock}"},
        "from_pretrained_arguments": {"model_name": "shared-model"},
        "arguments": {},
    }
    changed_step = {
        "name": "gen",
        "pipeline": {
            "configuration": {"component_type": "{Mock}"},
            "from_pretrained_arguments": {"model_name": "new-model"},
            "arguments": {},
        },
    }
    shared_key = pipeline_cache_key(shared_def)
    cache = {shared_key: MagicMock()}

    workflow = Workflow({"id": "shared", "steps": []}, "/tmp/test_output", "t.json")
    workflow._prior_step_keys = {"gen": shared_key, "gen_again": shared_key}

    seen_at_load = {}

    def mock_load(self, shared_components):
        seen_at_load["shared_still_cached"] = shared_key in cache
        self.pipeline = MagicMock()

    with patch.object(Pipeline, "load", mock_load):
        workflow.create_step_action(changed_step, {}, cache, 1, "cpu")

    assert seen_at_load["shared_still_cached"] is True, (
        "a key another step still maps to must survive the redefined step's load"
    )
```

- [ ] **Step 2: Run it to verify it fails**

Run on lem: `pytest -q tests/test_pipeline_caching.py -k another_step_still_maps_to`
Expected: FAIL — `shared_still_cached` is `False`.

- [ ] **Step 3: Add the shared-key guard**

In `dw/workflow.py` replace:

```python
            prior_key = getattr(self, "_prior_step_keys", {}).get(step_name)
            if prior_key and prior_key != cache_key and prior_key in previous_pipelines:
```

with:

```python
            prior_keys = getattr(self, "_prior_step_keys", {})
            prior_key = prior_keys.get(step_name)
            # Only this step's own variant: a key another step also mapped
            # to last run is that step's warm model, and releasing it here
            # would reload it cold a moment later while holding both stacks.
            # If nothing touches it this run, the end-of-run sweep drops it.
            still_shared = any(
                other != step_name and key == prior_key
                for other, key in prior_keys.items()
            )
            if (
                prior_key
                and prior_key != cache_key
                and prior_key in previous_pipelines
                and not still_shared
            ):
```

- [ ] **Step 4: Run the caching and worker tests**

Run on lem: `pytest -q tests/test_pipeline_caching.py tests/test_worker_execute.py`
Expected: all PASS, including the two existing `superseded`/`redefined` tests (a lone `gen` entry is not shared).

- [ ] **Step 5: Format, lint, commit**

```bash
ruff format dw/workflow.py tests/test_pipeline_caching.py && ruff check dw dw_mcp tests
git add dw/workflow.py tests/test_pipeline_caching.py
git commit -m "fix(workflow): #150 - a key another step still maps to is not released as superseded

Carrying step->key across jobs let a redefined step evict a pipeline its
sibling still resolved to, reloading a resident model cold while both
stacks were held. A shared key is left for the end-of-run sweep."
```

---

### Task 8: Run-time constraints see a list entry after its `variable:` reference resolves

**Files:**
- Modify: `dw/workflow.py:762-771` (`_prepare_definition`, the `apply_constraints` / `resolve_variable_values` order)
- Test: `tests/test_variable_constraints.py` (add to `class TestAConstraintReachesAListEntry`)

**Interfaces:**
- Consumes: `apply_constraints(definition, variables)` (rounds in place / raises `ValueError`), `resolve_variable_values(variables) -> dict` (returns a **copy** with `variable:` references inside list/dict values replaced), test helpers `workflow_with_shots(constraints, shots)` and constant `H3` in the test module, `Workflow(definition, output_dir, file_spec)`, `Workflow._prepare_definition(workflow_def, arguments, base_dir) -> (workflow_def, default_seed)`.
- Produces: nothing new; order of the two calls swaps.

- [ ] **Step 1: Write the failing test**

```python
    def test_an_entry_that_names_a_variable_is_still_rounded_at_run_time(self, tmp_path):
        """The run-time pass is the backstop for what the static pass cannot
        see. An entry field written as "variable:tail_len" is not a number
        until list-entry references resolve, so the backstop has to run
        after that resolution, not before it."""
        import copy

        from dw.workflow import Workflow

        definition = workflow_with_shots(
            {"num_frames": H3}, [{"name": "tag", "num_frames": "variable:tail_len"}]
        )
        definition["variables"]["tail_len"] = 130  # off the 17n+5 grid; snaps up to 141
        workflow = Workflow(definition, str(tmp_path), "listed.json")

        prepared, _seed = workflow._prepare_definition(
            copy.deepcopy(definition), {}, str(tmp_path)
        )

        (shot,) = prepared["steps"]
        assert shot["task"]["arguments"]["frames"] == 141
```

- [ ] **Step 2: Run it to verify it fails**

Run on lem: `pytest -q tests/test_variable_constraints.py -k names_a_variable_is_still_rounded`
Expected: FAIL — `frames == 130` (the reference was substituted after the backstop skipped it).

- [ ] **Step 3: Swap the two calls**

In `_prepare_definition`, the block currently reads:

```python
            apply_constraints(workflow_def, variables)
            # an entry of a list-valued variable may name another
            # variable; resolve those before anything inside it is
            # realized, so a reference type in an entry is a type name
            variables = resolve_variable_values(variables)
```

Change it to:

```python
            # an entry of a list-valued variable may name another
            # variable; resolve those before anything inside it is
            # realized, so a reference type in an entry is a type name -
            # and before the constraints pass, so an entry written as
            # "variable:tail_len" is a number by the time the rule looks
            variables = resolve_variable_values(variables)
            # A value outside a rule the workflow declares is refused, and
            # one the rule rounds is rounded with a warning saying so -
            # before anything loads, and before substitution puts the value
            # everywhere it is referenced (dw/variable_constraints.py, #96)
            apply_constraints(workflow_def, variables)
```

and delete the now-duplicated four-line "A value outside a rule…" comment that preceded the old `apply_constraints` call, so the comment appears once.

- [ ] **Step 4: Run the constraint, workflow and for_each tests**

Run on lem: `pytest -q tests/test_variable_constraints.py tests/test_workflow.py tests/test_for_each.py tests/test_worker_execute.py`
Expected: all PASS.

- [ ] **Step 5: Format, lint, commit**

```bash
ruff format dw/workflow.py tests/test_variable_constraints.py && ruff check dw dw_mcp tests
git add dw/workflow.py tests/test_variable_constraints.py
git commit -m "fix(workflow): #145 - constraints are applied after list-entry references resolve

An entry field written as 'variable:x' is not a number until
resolve_variable_values runs, so the run-time backstop skipped it and the
value was substituted afterwards unchecked."
```

---

### Task 9: Probe an attention processor's kernel once per process

**Files:**
- Modify: `dw/kernel_availability.py:92-114` (`kernel_availability_fault`)
- Modify: `tests/test_kernel_availability.py` (autouse cache reset + one new test)

**Interfaces:**
- Consumes: `load_type_from_name(value)`, `_requires_remote_kernel(cls)`, `_UNRESOLVED_PREFIXES`, `KERNEL_FAULT_MARKER` (all already in the module).
- Produces: `_fault_for_name(value: str) -> str | None`, decorated with `functools.lru_cache(maxsize=None)`; `kernel_availability_fault(value)` keeps its signature and delegates to it. Tests reset with `_fault_for_name.cache_clear()`.

- [ ] **Step 1: Add the autouse reset and the failing test**

At the top of `tests/test_kernel_availability.py`, after the existing imports (note Task 1 removed the unused `import pytest`; add it back since it is used now):

```python
import pytest

from dw import kernel_availability
```

Then add:

```python
@pytest.fixture(autouse=True)
def _fresh_probe():
    """Each test resolves its own fake class under the same dotted name, so
    the per-process memo must not carry an answer between them."""
    kernel_availability._fault_for_name.cache_clear()
    yield
    kernel_availability._fault_for_name.cache_clear()


class _CountingKernelBackedProcessor:
    constructions = 0

    def __init__(self):
        get_kernel = _get_kernel_that_works
        get_kernel("shi-labs/natten")
        type(self).constructions += 1
```

and, inside `class TestKernelAvailabilityFault`:

```python
    def test_the_probe_runs_once_per_process_for_a_name(self, monkeypatch):
        """Construction fetches a Hub kernel; validate, submit and rerun each
        ask, and a for_each asks once per member. The answer is a
        per-process constant."""
        monkeypatch.setattr(
            kernel_availability,
            "load_type_from_name",
            _resolving({"pkg.Natten": _CountingKernelBackedProcessor}),
        )
        _CountingKernelBackedProcessor.constructions = 0
        for _ in range(3):
            assert kernel_availability_fault("pkg.Natten") is None
        assert _CountingKernelBackedProcessor.constructions == 1
```

- [ ] **Step 2: Run to verify it fails**

Run on lem: `pytest -q tests/test_kernel_availability.py`
Expected: the fixture errors with `AttributeError: module 'dw.kernel_availability' has no attribute '_fault_for_name'`.

- [ ] **Step 3: Split the function and memoize the resolving half**

In `dw/kernel_availability.py`, add `import functools` to the imports, then replace `kernel_availability_fault`:

```python
@functools.lru_cache(maxsize=None)
def _fault_for_name(value):
    """The answer for one type name, memoized: which class a name resolves
    to and whether it can be constructed are per-process constants, and
    the construction is the expensive part (a Hub kernel fetch)."""
    try:
        attn_processor_class = load_type_from_name(value)
    except Exception:
        return None

    if not _requires_remote_kernel(attn_processor_class):
        return None

    try:
        attn_processor_class()
    except Exception as e:
        return f"'{value}' {KERNEL_FAULT_MARKER}: {e}"
    return None


def kernel_availability_fault(value):
    """Why constructing this 'attn_processor_type' value will fail, or None.

    `value` is the dotted or bare type name as written in the workflow.
    A name that fails to resolve at all is not this check's job - that is
    an ordinary '_type' load failure at run time. A processor that doesn't
    depend on a remote kernel is never constructed here, since a processor
    with real side effects in `__init__` should not be instantiated just to
    validate it.

    Answered once per process per name: validate, submit and rerun all ask,
    and a for_each asks once per member (`_fault_for_name`).
    """
    if not isinstance(value, str) or value.startswith(_UNRESOLVED_PREFIXES):
        return None
    return _fault_for_name(value)
```

- [ ] **Step 4: Run the kernel tests and the validate paths that call them**

Run on lem: `pytest -q tests/test_kernel_availability.py tests/test_examples.py tests/test_ltx2_diffusion_decode.py`
Expected: all PASS. If `tests/test_examples.py` monkeypatches `load_type_from_name` for a natten name in more than one test, add the same `_fault_for_name.cache_clear()` autouse fixture there.

- [ ] **Step 5: Format, lint, commit**

```bash
ruff format dw/kernel_availability.py tests/test_kernel_availability.py && ruff check dw dw_mcp tests
git add dw/kernel_availability.py tests/test_kernel_availability.py
git commit -m "fix(validation): #178 - the kernel probe runs once per process per type name

Every validate, submit and rerun constructed the attention processor
(a Hub kernel fetch) again; the answer is a per-process constant."
```

---

### Task 10: One watermark read per catalog listing

**Files:**
- Modify: `dw/server/observed_cost.py:286-320` (`ObservedCosts.rows_for`, `ObservedCosts.observed`; add `refresh`)
- Modify: `dw/server/app.py:239-266` (`attach_observed`)
- Test: `tests/test_observed_cost.py` (append to the class holding `test_the_aggregate_recomputes_only_when_the_table_moves`, ~line 361)

**Interfaces:**
- Consumes: `JobHistory.watermark() -> (count, max_finished_at)`, `JobHistory.finished_runs() -> dict[name, rows]`, `ObservedCosts(history)`.
- Produces:
  - `ObservedCosts.refresh() -> bool` — reads the watermark once, reloads `_rows` if it moved; `False` when there is no history or the read failed.
  - `ObservedCosts.rows_for(name, *, fresh=True)` — with `fresh=False` skips the watermark read and answers from `_rows`.
  - `ObservedCosts.observed(name, definition, arguments=None, *, fresh=True)` — same keyword, passed through.
  - `attach_observed` calls `refresh()` once, then `observed(..., fresh=False)` per detail.

- [ ] **Step 1: Write the failing test**

```python
    def test_a_listing_reads_the_watermark_once(self, tmp_path):
        """list_workflows attaches a figure to every catalog entry; the
        watermark is a COUNT(*) under the history lock the worker also
        needs, so a listing takes it once, not once per workflow."""
        from dw.server.app import attach_observed

        history = JobHistory(tmp_path / "jobs.sqlite")
        with history._connect() as connection:
            connection.execute(
                "INSERT INTO jobs (id, status, started_at, finished_at, arguments,"
                " manifest, events, workflow_name) VALUES (?,?,?,?,?,?,?,?)",
                ("a", "succeeded", 0.0, 600.0, "{}", "[]", "[]", "templates/x"),
            )
        costs = ObservedCosts(history)
        reads = []
        original = history.watermark
        history.watermark = lambda: (reads.append(1), original())[1]

        details = {
            name: {"cost_drivers": {}, "variable_names": []}
            for name in ("templates/x", "templates/y", "templates/z")
        }
        attach_observed(details, costs)

        assert len(reads) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run on lem: `pytest -q tests/test_observed_cost.py -k reads_the_watermark_once`
Expected: FAIL — `len(reads) == 3`.

- [ ] **Step 3: Add `refresh` and the `fresh` keyword**

In `dw/server/observed_cost.py` replace `rows_for` and the head of `observed`:

```python
def refresh(self):
    """Bring the cached rows up to the table's watermark, once.

    Returns False when there is no history to read or the read failed -
    the caller then has no rows and knows why. A listing calls this
    once and then asks `rows_for(name, fresh=False)` per entry, so the
    COUNT(*) under the history lock happens once per request rather
    than once per workflow."""
    if self.history is None:
        return False
    try:
        mark = self.history.watermark()
    except Exception:
        logger.debug("observed cost: could not read the job watermark")
        return False
    if mark != self._mark:
        try:
            self._rows = self.history.finished_runs()
        except Exception:
            logger.debug("observed cost: could not read job history")
            self._rows = {}
            return False
        self._mark = mark
    return True


def rows_for(self, name, *, fresh=True):
    """That workflow's comparable-candidate runs, refreshing the cache
    when the table has moved (unless the caller already did)."""
    if fresh and not self.refresh():
        return []
    return self._rows.get(name, [])


def observed(self, name, definition, arguments=None, *, fresh=True):
    rows = self.rows_for(name, fresh=fresh)
    if not rows:
        return None
```

(The remainder of `observed` is unchanged.)

- [ ] **Step 4: Refresh once in `attach_observed`**

In `dw/server/app.py`, change the head of `attach_observed`:

```python
    if observed_costs is None or not observed_costs.refresh():
        return details
    for name, detail in details.items():
```

and the call inside the loop to:

```python
        observed = observed_costs.observed(name, surrogate, fresh=False)
```

- [ ] **Step 5: Run the observed-cost and catalog tests**

Run on lem: `pytest -q tests/test_observed_cost.py tests/test_server.py -k "observed or catalog or list_workflows" tests/test_mcp_catalog.py tests/test_plan.py`
Expected: all PASS, including `test_the_aggregate_recomputes_only_when_the_table_moves` (default `fresh=True` keeps `rows_for`'s old behaviour).

- [ ] **Step 6: Format, lint, commit**

```bash
ruff format dw/server/observed_cost.py dw/server/app.py tests/test_observed_cost.py && ruff check dw dw_mcp tests
git add dw/server/observed_cost.py dw/server/app.py tests/test_observed_cost.py
git commit -m "fix(server): #93 - a catalog listing reads the job watermark once

attach_observed asked ObservedCosts per entry and each ask ran the
COUNT(*)/MAX(finished_at) query under the history lock - ~70-80 per
list_workflows. refresh() once, then answer each name from the cache."
```

---

### Task 11: Land on `develop`, sync `master` back, and re-verify the PR

**Files:** none new. `pyproject.toml` and `plugins/dw/.claude-plugin/plugin.json` change only through the merge.

- [ ] **Step 1: Full suite on lem, formatted and linted**

```bash
ruff format --check dw dw_mcp tests && ruff check dw dw_mcp tests
git push
ssh lem 'cd /tmp/pr183-fix && git fetch -q origin fix/pr183-merge-readiness && git checkout -q --detach origin/fix/pr183-merge-readiness \
  && ~/diffusers-workflow/venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -3'
```

Expected: `0 failed`; pass count = baseline + 10 new tests (Task 2 adds two; Tasks 3–10 add one each); skipped count unchanged.

- [ ] **Step 2: Merge the branch into `develop`**

```bash
git checkout develop && git pull --ff-only origin develop
git merge --no-ff fix/pr183-merge-readiness -m "Merge fix/pr183-merge-readiness - CI format, stall warnings, asset origin/404, elision, orphans, #150 shared key, constraint order, kernel probe memo, one watermark per listing"
```

- [ ] **Step 3: Merge `master` back into `develop` so the version stops reading as a downgrade**

```bash
git merge --no-ff origin/master -m "Merge master back into develop (release 0.4.0-beta.5, idea issue template)"
grep -n '^version' pyproject.toml            # expect: version = "0.4.0-beta.5"
grep -n '"version"' plugins/dw/.claude-plugin/plugin.json   # expect: "0.4.0-beta.5"
```

If the merge reports a conflict in `pyproject.toml`: keep `version = "0.4.0-beta.5"` from master and every dependency line from develop (the `transformers>=5.16.1,!=5.17.0` pin and the `kernels>=0.17.0` line), then `git add pyproject.toml && git commit`.

- [ ] **Step 4: Push and watch CI on the PR**

```bash
git push origin develop
gh pr checks 183 --repo dkackman/diffusers-workflow --watch
```

Expected: `backend`, `ui`, and the three CodeQL `Analyze` checks all SUCCESS. If `backend` fails at Format check again, someone pushed unformatted code to `develop` in the meantime: repeat Task 1 on `develop` directly.

- [ ] **Step 5: Confirm the merged tree's version and the PR's mergeability**

```bash
gh pr view 183 --repo dkackman/diffusers-workflow --json mergeable,statusCheckRollup -q '.mergeable, ([.statusCheckRollup[] | select(.conclusion=="FAILURE")] | length)'
```

Expected: `MERGEABLE` and `0`.

- [ ] **Step 6: Clean up the lem worktree and leave a note on the PR**

```bash
ssh lem 'cd ~/diffusers-workflow && git worktree remove --force /tmp/pr183-fix && git worktree prune'
gh pr comment 183 --repo dkackman/diffusers-workflow --body "$(cat <<'EOF'
Merge-readiness pass (docs/superpowers/plans/2026-09-16-pr183-merge-readiness.md):

- CI: ruff format/lint failure cleared; backend job now reaches Tests.
- #176: phase-stall reports stay in the event log, out of `job.warnings`.
- Assets: library origin follows the directory, not its index (examples tree was labelled writable when the workspace library did not exist); no-library lookups are a 404 again, not a 500.
- #122: a sub-workflow step is kept whether or not it declares a `result`.
- #170: an orphan run is one holding nothing but manifest/workflow/job.json (text-shape runs are no longer listed).
- #150: a key another step still maps to is not released as superseded.
- #145: run-time constraints apply after list-entry `variable:` references resolve.
- #178: kernel probe memoized per process; #93: one watermark read per listing.
- master merged back into develop so the version reads 0.4.0-beta.5 either way.

Full suite on lem: all green.
EOF
)"
```

Not in this plan, for the human after merge: deploy `develop` to `lem` (server code changed → restart, per the implementer role's deploy path) and run `./run-regression.sh smoke` from the `iterate` repo — the `warnings: []` cases there are the end-to-end check for Task 2.

---

## Self-review

**Spec coverage:** B → Task 1; finding 1 → Task 2; 2 → Task 3; 3 → Task 4; 4 → Task 5; 5 → Task 6; 6 → Task 7; 7 → Task 8; P1 → Task 9; P2 → Task 10; version-downgrade appearance → Task 11 Step 3. The remaining review items (three more validate-path re-expansions, `describe_task` called twice per task step, `observed_for` re-parsing row JSON, duplicate helpers, UI type for `basis: "observed"`, picks `size` counting only visible ticks, `formatBytes` rounding 0 to "1 KB") are deliberately out of scope — file them as issues after merge; none affects correctness of a delivered artifact.

**Placeholder scan:** every code step shows the code; every test step shows the test; every run step names the command and the expected outcome. The one conditional instruction (Task 6 Step 5, Task 9 Step 4: "if a test pins the old phrase / monkeypatches the same name…") names the exact edit to make.

**Type consistency:** `_asset_origin(ws, root)` (Task 3) is used with two arguments at every call site named there. `_fault_for_name` (Task 9) is the name both the fixture and the implementation use. `ObservedCosts.refresh()` / `rows_for(name, fresh=)` / `observed(..., fresh=)` (Task 10) match between `observed_cost.py` and `attach_observed`. `RUN_BOOKKEEPING_FILES` (Task 6) is defined at module level in `app.py` and read inside the `create_app` closure where `_iter_orphan_runs` lives.
