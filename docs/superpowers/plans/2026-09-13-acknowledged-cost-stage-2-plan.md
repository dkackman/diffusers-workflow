# Acknowledged-cost binding, stage 2: the bound acknowledgement, the 409 and the cache probe

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A caller may bind its `acknowledged_cost` to the plan it validated (`{fingerprint, minutes, downloads}`); `POST /api/jobs` and `/rerun` refuse with 409 when the run's shape no longer matches; every job records which form of acknowledgement it got; and the plan's `cached_steps` is answered exactly by the worker's step cache.

**Architecture:** The step-cache probe is a refactor of `Workflow.run`: the substitute-expand-seed preparation and the per-step "is this a hit" lookup are lifted into methods both the run loop and a new `Workflow.cache_hits(arguments)` call, so the probe cannot drift from the run. The worker answers a `probe_cache` command with that list; `JobManager.probe_cache` asks it the way `memory_status` does (never blocking behind a running job); `build_plan` takes the answer through `cache_probe`. The route classifies the acknowledgement once (`none | boolean | bound`), re-plans for a bound one and raises 409 with the current plan in the body, records the form on the job and in `jobs.sqlite`. MCP widens the argument, forwards the object, and renders the 409 with the new estimate.

**Tech Stack:** Python 3.12, FastAPI, pydantic, sqlite3, `multiprocessing.Queue` worker protocol, pytest.

**Spec:** [docs/superpowers/specs/2026-09-12-acknowledged-cost-binding-design.md](../specs/2026-09-12-acknowledged-cost-binding-design.md) sections 6-11. Stage 1 landed at b37f033 ([stage 1 plan](2026-09-13-acknowledged-cost-stage-1-plan.md)).

## Global Constraints

- `dw/plan.py` still imports nothing from `dw.server` or `dw.worker`; it reaches the worker only through the `cache_probe` callable it is handed.
- The bare-boolean and absent acknowledgement paths are byte-for-byte unchanged in behaviour: no new refusal, no new round trip to the worker on `POST /api/jobs`.
- `minutes` is recorded, never compared (spec non-goal).
- The 409 is pre-flight only; nothing meters or aborts a running job.
- No model name or minute figure in engine code or in a test fixture.
- Skills stay under `12 * 1024` bytes each (`minimax-h3` is at 12266; `ltx-2.5` at 12180).
- Worktree `.claude/worktrees/cost-binding`, branch `cost-binding` off `develop` b37f033. Tests with `python -m pytest`.
- Commits: `feat(engine|server|mcp): #85 - ...` / `docs: #85 - ...`, ending `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

### Deviations from the spec, decided here

- **Probe command shape.** Spec §10 sends the worker a realized, expanded definition plus a seed. Instead the command carries the same fields an `execute` command does (`workflow_path` | `workflow`+`base_dir`, `workflow_dir`, `arguments`, `output_dir`, `asset_dir`), and the worker loads the workflow with its existing `_load_workflow` and calls `Workflow.cache_hits(arguments)`. Preparation then runs the run's own code path (`realize_args` on the steps included - the cache key holds realized values), which is the whole point of the refactor.
- **No worker → `[]`, not `None`.** The step cache lives in the worker process; a worker that is not running holds none, so zero cached steps is the true answer. Busy worker or a timeout → `None` (unknown), as the spec says.
- **`rerun` does not inherit a stored bound acknowledgement.** The request's own `acknowledged_cost` decides the form and whether a check runs; the original's object is kept in the spec (so history says what was consented to) and `rerun` copies it forward for the record only. An inherited check would fire on a caller that sent `true`, which is exactly the path that must not change.
- **The 409 body's plan carries `cached_steps: null`.** The check does not probe the worker - it costs a round trip and the fingerprint does not depend on it.

## File structure

| File | Responsibility |
|---|---|
| `dw/workflow.py` (modify) | `_prepare_definition()` and `_cache_lookup()` lifted out of `run()`; new `cache_hits(arguments)` |
| `dw/worker.py` (modify) | `probe_cache` command → `_handle_probe_cache` |
| `dw/server/jobs.py` (modify) | `JobManager.probe_cache()`, `rerun_spec()`, `submit(acknowledged=, acknowledged_cost=)`, `Job.acknowledged`, sqlite column, `RERUN_SPEC_KEYS` |
| `dw/plan.py` (modify) | `cache_probe` honoured; `cached_steps` |
| `dw/server/app.py` (modify) | `AcknowledgedCost`, the field on both requests, `_acknowledgement_form`, `_check_bound_acknowledgement`, the probe on validate |
| `dw_mcp/diagnose.py`, `dw_mcp/server.py`, `dw_mcp/client.py` (modify) | `bool \| dict`, forwarding, 409 rendering, `COST_REFUSAL` |
| docs + skills (modify) | the bound form |
| tests: `test_workflow_step_cache.py`, `test_worker_execute.py`, `test_server.py`, `test_plan.py`, `test_mcp_diagnose.py`, `test_mcp_client.py`, `test_plugin_skills.py` | coverage |

---

### Task 1: `Workflow.cache_hits()` by refactoring `run()`

**Files:**
- Modify: `dw/workflow.py` (`run()` lines ~700-760 preparation, ~880-950 cache lookup)
- Test: `tests/test_workflow_step_cache.py`

**Interfaces:**
- Produces: `Workflow._prepare_definition(workflow_def, arguments, base_dir) -> (workflow_def, default_seed)` - variables realized/folded/resolved/substituted, `for_each` expanded, the seed read and coerced to `int` (or `None` when the workflow names none). Raises exactly what the run raised at that point.
- Produces: `Workflow._cache_lookup(workflow_id, steps, index, step_data, step_seed, hits_this_run, cache_enabled) -> (cached_result | None, step_data_snapshot | None)` - the whole `is_cacheable` / snapshot / `step_cache.get` block. `parent_saves_this` computed inside from `self._final_save_owned_by_parent`.
- Produces: `Workflow.cache_hits(arguments) -> list[str]` - the step names the step cache would serve for this workflow, seed and arguments, in step order; `[]` for an unseeded workflow. Executes nothing, emits nothing, opens no run directory.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_workflow_step_cache.py` (the module has `step_cache`, `build_test_workflow_and_call_count_spy()`, whose workflow has one step `generate`, seed 42, variable `prompt`):

```python
class TestCacheHits:
    def test_a_cold_cache_reports_no_hits(self):
        step_cache.clear()
        workflow, _ = build_test_workflow_and_call_count_spy()
        try:
            assert workflow.cache_hits({}) == []
        finally:
            for p in workflow._test_patcher:
                p.stop()

    def test_after_a_run_the_probe_names_what_the_next_run_reuses(self):
        step_cache.clear()
        workflow, call_count = build_test_workflow_and_call_count_spy()
        try:
            workflow.run({})
            probe = workflow.cache_hits({})
            workflow.run({})
            reused = [entry["step"] for entry in workflow.manifest if entry.get("reused")]
            assert probe == ["generate"]
            assert probe == reused
            assert call_count() == 1, "the probe executed nothing"
        finally:
            for p in workflow._test_patcher:
                p.stop()

    def test_a_changed_argument_is_a_miss(self):
        step_cache.clear()
        workflow, _ = build_test_workflow_and_call_count_spy()
        try:
            workflow.run({"prompt": "a cat"})
            assert workflow.cache_hits({"prompt": "a dog"}) == []
        finally:
            for p in workflow._test_patcher:
                p.stop()

    def test_an_unseeded_workflow_has_no_hits(self):
        step_cache.clear()
        workflow, _ = build_test_workflow_and_call_count_spy()
        del workflow.workflow_definition["seed"]
        try:
            workflow.run({})
            assert workflow.cache_hits({}) == []
        finally:
            for p in workflow._test_patcher:
                p.stop()

    def test_the_probe_writes_nothing(self, tmp_path):
        step_cache.clear()
        workflow, _ = build_test_workflow_and_call_count_spy()
        workflow.output_dir = str(tmp_path)
        try:
            workflow.cache_hits({})
            assert list(tmp_path.iterdir()) == []
        finally:
            for p in workflow._test_patcher:
                p.stop()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_workflow_step_cache.py -k CacheHits -v`
Expected: FAIL - `AttributeError: 'Workflow' object has no attribute 'cache_hits'`.

- [ ] **Step 3: Refactor `run()` and add `cache_hits()`**

In `dw/workflow.py`, inside `class Workflow`, add two methods (place them just above `run`):

```python
    def _prepare_definition(self, workflow_def, arguments, base_dir):
        """The definition as a run works from it: constants realized,
        arguments folded into the variables, list entries' own references
        resolved, variable values realized (assets loaded), every
        'variable:' substituted, every for_each expanded, and the seed read
        and coerced. Returns (workflow_def, default_seed) - the seed is
        None when the workflow names none, and the caller decides what
        that means (run() draws one; cache_hits() reports no hits).

        Shared by run() and cache_hits() so the probe prepares exactly what
        the run prepares - the step cache keys on the realized step, and a
        probe that prepared it differently would answer for a run that
        never happens.
        """
        workflow_id = workflow_def["id"]
        variables = workflow_def.get("variables", None)
        if variables is not None:
            logger.debug(f"Setting variables for workflow: {workflow_id}")
            # a constant is the value a variable declares, so it resolves before
            # anything is converted to the type of that declaration
            realize_constants(variables)
            # first set variable values base don the arguments passed to the workflow
            # these may come form the command line or form a parent workflow
            set_variables(arguments, variables)
            # an entry of a list-valued variable may name another
            # variable; resolve those before anything inside it is
            # realized, so a reference type in an entry is a type name
            variables = resolve_variable_values(variables)
            # realize the variables, initialiting downloads of images etc
            realize_args(variables, base_dir)
            ## then replace any variable references in the workflow definition with the actual values
            # replace_variables returns a new structure rather than mutating in
            # place, so the result must be captured here
            workflow_def = replace_variables(workflow_def, variables)

        # One ordinary step per entry of every for_each list, before the
        # seed, the run id and the realized workflow are computed, so
        # each covers what actually runs. A ForEachError here fails the
        # run before anything loads
        workflow_def = expand_for_each(workflow_def)

        # Set up random seed for reproducibility. Resolved lazily - as a
        # dict.get default, torch.seed() would run on every call and reseed
        # the global RNG even when the workflow names an explicit seed
        default_seed = workflow_def.get("seed")
        # The schema lets 'seed' be a string so it can hold a 'variable:'
        # reference, which the substitution above has already resolved -
        # but a variable overridden from the command line arrives as a
        # string whenever the workflow declared no integer default to
        # coerce against, and manual_seed would fail deep inside the run
        if isinstance(default_seed, str):
            try:
                default_seed = int(default_seed)
            except ValueError:
                raise ValueError(
                    f"Workflow {workflow_id} seed must be an integer, "
                    f"got {default_seed!r}"
                )
            workflow_def["seed"] = default_seed
        return workflow_def, default_seed

    def _cache_lookup(
        self, workflow_id, steps, index, step_data, step_seed, hits_this_run, cache_enabled
    ):
        """Whether the step cache serves step `index`, as
        (cached_result or None, the step_data snapshot the entry is keyed
        on or None). Shared by run() and cache_hits() - see
        _prepare_definition for why.
        """
        # What later steps still read, which decides both whether this
        # step's result has to be kept alive after the step and whether a
        # cached entry that kept none can serve this run
        remaining_refs = referenced_result_names(steps[index + 1 :])
        result_needed = index == len(steps) - 1 or any(
            reference_resolves_to(ref, step_data["name"]) for ref in remaining_refs
        )
        # create_step_action (and the pipeline load it triggers) mutates
        # step_data in place - injecting a "generator" key - so the cache
        # must key off a snapshot taken before that happens, and that same
        # snapshot must be reused for the put() later. A sub-workflow step
        # is never cacheable: its files roll up from the child's own
        # manifest, which a hit does not rebuild.
        is_cacheable = "workflow" not in step_data and cache_enabled
        # The last step of a composed child whose parent does the saving
        # (#92) - its files are the parent step's, written once, under the
        # parent's name and subfolder
        parent_saves_this = self._final_save_owned_by_parent and index == len(steps) - 1
        step_data_snapshot = None
        if is_cacheable:
            try:
                step_data_snapshot = copy.deepcopy(step_data)
                if parent_saves_this:
                    # Keyed apart from the same step run standalone: this
                    # entry's result was never saved here, so a standalone
                    # hit on it would report no files
                    step_data_snapshot["__saved_by_parent__"] = True
            except Exception as ex:
                # A realized argument that cannot be deep-copied (an open
                # handle, a live model object) just means this step is not
                # cacheable - never a failed run
                logger.debug(
                    f"Step '{step_data['name']}' arguments are not copyable "
                    f"({ex}) - skipping the step cache for it"
                )
                is_cacheable = False
        if not is_cacheable:
            return None, None
        cached_result = step_cache.get(
            workflow_id,
            step_data_snapshot,
            step_seed,
            hits_this_run,
            # The root, not this run's directory: a hit reports the earlier
            # run's files and writes nothing new, so keying on a directory
            # that is new every run would mean the cache could never hit
            # again. What the root still guards is a run redirected
            # somewhere else, where the earlier files are not what the
            # caller asked for
            self.output_dir,
            needs_result=result_needed,
        )
        return cached_result, step_data_snapshot

    def cache_hits(self, arguments):
        """The steps the step cache would serve for a run with `arguments`,
        in step order - what the plan reports as cached_steps (#85).

        Prepares the definition exactly as run() does and asks the cache the
        question run() asks, step by step with the hits so far, and executes
        nothing: no run directory, no events, no pipeline. An unseeded
        workflow has no cache, so it answers [] without asking.
        """
        output_root_token = activate_output_root(self.output_dir)
        try:
            workflow_def = copy.deepcopy(self.workflow_definition)
            workflow_id = workflow_def["id"]
            base_dir = (
                os.path.dirname(os.path.abspath(self.file_spec))
                if self.file_spec
                else None
            )
            workflow_def, default_seed = self._prepare_definition(
                workflow_def, arguments or {}, base_dir
            )
            if default_seed is None or not self._cache_enabled_by_parent:
                return []
            steps = workflow_def.get("steps", [])
            realize_args(steps, base_dir)
            hits_this_run = set()
            hits = []
            for index, step_data in enumerate(steps):
                step_seed = step_data.get("seed", default_seed)
                cached_result, _ = self._cache_lookup(
                    workflow_id, steps, index, step_data, step_seed, hits_this_run, True
                )
                if cached_result is not None:
                    hits_this_run.add(step_data["name"])
                    hits.append(step_data["name"])
            return hits
        finally:
            deactivate_output_root(output_root_token)
```

Then in `run()`:

- Replace the block from `# Handle variable substitution if variables are defined` through the `workflow_def["seed"] = default_seed` line that follows the `int(default_seed)` coercion (lines ~710-752, ending just before `# A workflow that names no seed gets a fresh one every run`) with:

```python
            workflow_def, default_seed = self._prepare_definition(
                workflow_def, arguments, base_dir
            )
```

  Keep everything from `cache_enabled_this_run = (...)` onward unchanged (the random draw, `workflow_def["seed"] = default_seed`, `resolved_seed`).

- In the step loop, replace the block from `# What later steps still read, ...` (`remaining_refs = ...`) through the `cached_result = (step_cache.get(...) if is_cacheable else None)` expression with:

```python
                cached_result, step_data_snapshot = self._cache_lookup(
                    workflow_id,
                    steps,
                    i,
                    step_data,
                    step_seed,
                    hits_this_run,
                    cache_enabled_this_run,
                )
                is_cacheable = step_data_snapshot is not None
```

  `is_cacheable` and `step_data_snapshot` are read further down by the `step_cache.put` block - confirm with `grep -n "is_cacheable\|step_data_snapshot" dw/workflow.py` that every remaining use is after this point and still bound. `result_needed` is also used later (for `release_unreferenced_results` / retaining); if `grep -n result_needed dw/workflow.py` shows a use after the lookup, recompute it in the loop right after the lookup with the same two lines used inside `_cache_lookup` rather than returning it (keep the method's return a pair).

- [ ] **Step 4: Run the step-cache suites**

Run: `python -m pytest tests/test_workflow_step_cache.py tests/test_step_cache.py tests/test_jobs_reused.py tests/test_worker_execute.py -v`
Expected: all PASS, the five new ones included.

- [ ] **Step 5: Commit**

```bash
git add dw/workflow.py tests/test_workflow_step_cache.py
git commit -m "feat(engine): #85 - a workflow can say which steps the cache would serve, by the run's own preparation

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: the worker's `probe_cache` command and `JobManager.probe_cache`

**Files:**
- Modify: `dw/worker.py` (dispatch at ~127-150; handlers at ~368-390), `dw/server/jobs.py` (`memory_status` at ~1154 is the model)
- Test: `tests/test_worker_execute.py`, `tests/test_server.py`

**Interfaces:**
- Consumes: `Workflow.cache_hits(arguments)` from Task 1; the worker's `_load_workflow(command, output_dir)`.
- Produces: worker command `{"type": "probe_cache", "arguments", "output_dir", "workflow_path" | "workflow" + "base_dir", "workflow_dir", "asset_dir"?}` answered by `{"type": "probe_cache", "cached": [names]}` or `{"type": "probe_cache", "cached": None, "error": str}`.
- Produces: `JobManager.probe_cache(command, timeout=5) -> list[str] | None` where `command` is that dict minus `type`. `None` when a job is running, the worker lock is busy, the worker did not answer, or answered with an error; `[]` when no worker is running.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_worker_execute.py`:

```python
class ProbableWorkflow(StubWorkflow):
    def __init__(self, hits):
        super().__init__()
        self.hits = hits
        self.probed_with = None

    def cache_hits(self, arguments):
        self.probed_with = arguments
        return list(self.hits)


def test_probe_cache_answers_with_the_workflows_hits():
    worker = _make_worker()
    workflow = ProbableWorkflow(["gen"])
    command = {
        "type": "probe_cache",
        "workflow_path": "x.json",
        "arguments": {"prompt": "p"},
        "output_dir": "/tmp",
    }
    with patch("dw.worker.workflow_from_file", return_value=workflow):
        worker._handle_probe_cache(command)
    assert _drain(worker.result_queue) == [{"type": "probe_cache", "cached": ["gen"]}]
    assert workflow.probed_with == {"prompt": "p"}


def test_probe_cache_reports_a_failure_as_unknown_not_as_a_crash():
    worker = _make_worker()
    with patch("dw.worker.workflow_from_file", side_effect=ValueError("bad file")):
        worker._handle_probe_cache(
            {"type": "probe_cache", "workflow_path": "x.json", "arguments": {}, "output_dir": "/tmp"}
        )
    [answer] = _drain(worker.result_queue)
    assert answer["type"] == "probe_cache"
    assert answer["cached"] is None
    assert "bad file" in answer["error"]


def test_probe_cache_activates_the_jobs_asset_dir(tmp_path):
    worker = _make_worker()
    seen = {}

    class AssetAwareWorkflow(ProbableWorkflow):
        def cache_hits(self, arguments):
            from dw.assets import current_asset_dir

            seen["asset_dir"] = current_asset_dir()
            return []

    with patch("dw.worker.workflow_from_file", return_value=AssetAwareWorkflow([])):
        worker._handle_probe_cache(
            {
                "type": "probe_cache",
                "workflow_path": "x.json",
                "arguments": {},
                "output_dir": "/tmp",
                "asset_dir": str(tmp_path),
            }
        )
    assert seen["asset_dir"] == str(tmp_path)
```

(Check `dw/assets.py` for the accessor name that returns the active asset root - `grep -n "^def " dw/assets.py`; if it is not `current_asset_dir`, use the one that is.)

In `tests/test_server.py`, extend `ScriptedWorkerManager.send_command` with a `probe_cache` branch, and add a knob:

```python
        elif command["type"] == "probe_cache":
            self._results.put(
                {"type": "probe_cache", "cached": list(self.cached_steps)}
            )
```

and in `__init__`: `self.cached_steps = []`. Then append tests:

```python
class TestProbeCache:
    def test_asks_the_worker_and_returns_its_answer(self, server):
        with server(success_script) as client:
            manager = client.app.state.job_manager
            manager.worker_manager.ensure_worker()
            manager.worker_manager.cached_steps = ["gen"]
            assert manager.probe_cache(
                {"workflow_path": "x.json", "arguments": {}, "output_dir": "/tmp"}
            ) == ["gen"]
            sent = manager.worker_manager.commands[-1]
            assert sent["type"] == "probe_cache"
            assert sent["workflow_path"] == "x.json"

    def test_no_worker_means_an_empty_cache(self, server):
        with server(success_script) as client:
            manager = client.app.state.job_manager
            assert manager.worker_manager.worker_active is False
            assert manager.probe_cache({"workflow_path": "x.json", "arguments": {}, "output_dir": "/tmp"}) == []
            assert manager.worker_manager.commands == []

    def test_a_running_job_means_unknown(self, server):
        with server(hanging_script) as client:
            response = client.post("/api/jobs", json={"workflow": valid_workflow()})
            job_id = response.json()["id"]
            wait_for_status(client, job_id, ("running",))
            manager = client.app.state.job_manager
            assert manager.probe_cache({"workflow_path": "x.json", "arguments": {}, "output_dir": "/tmp"}) is None
            client.post(f"/api/jobs/{job_id}/cancel")

    def test_an_unanswered_probe_is_unknown(self, server):
        with server(success_script) as client:
            manager = client.app.state.job_manager
            manager.worker_manager.ensure_worker()
            manager.worker_manager.send_command = lambda command: None  # swallow it
            assert manager.probe_cache(
                {"workflow_path": "x.json", "arguments": {}, "output_dir": "/tmp"}, timeout=0.05
            ) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_worker_execute.py -k probe -v; python -m pytest tests/test_server.py -k ProbeCache -v`
Expected: FAIL - no `_handle_probe_cache`, no `probe_cache`.

- [ ] **Step 3: Implement**

`dw/worker.py`, in the main dispatch after the `memory_status` branch:

```python
                    elif command_type == "probe_cache":
                        self._handle_probe_cache(command)
```

and beside `_handle_memory_status`:

```python
    def _handle_probe_cache(self, command: Dict[str, Any]):
        """Which steps the step cache would serve for a run of this command
        - the plan's cached_steps (#85). Same fields as an execute command;
        loads the workflow, executes nothing. A failure answers
        cached: null with the reason rather than an error message, since
        an unknown answer is a valid plan and a crashed probe is not.
        """
        try:
            workflow, _ = self._load_workflow(command, command["output_dir"])
            asset_token = (
                activate_asset_dir(command["asset_dir"])
                if command.get("asset_dir")
                else None
            )
            try:
                cached = workflow.cache_hits(command.get("arguments") or {})
            finally:
                if asset_token is not None:
                    deactivate_asset_dir(asset_token)
            self.result_queue.put({"type": "probe_cache", "cached": cached})
        except Exception as e:
            logger.debug(f"Cache probe failed: {e}")
            self.result_queue.put(
                {"type": "probe_cache", "cached": None, "error": str(e)}
            )
```

`dw/server/jobs.py`, beside `memory_status`:

```python
    def probe_cache(self, command, timeout=5):
        """Which steps the worker's step cache would serve for `command` (the
        fields an execute command carries, minus its type), or None when the
        answer cannot be had right now - a job is running, the worker is
        busy, or it did not answer in time. Never blocks a request behind a
        running job, for the same reason memory_status does not.

        No worker running is a definite answer, not an unknown one: the
        cache lives in the worker process, so a worker that is not running
        holds nothing.
        """
        if self._current_job_id is not None:
            return None
        if not self.worker_manager.worker_active:
            return []
        if not self._worker_lock.acquire(timeout=2):
            return None
        try:
            self.worker_manager.send_command({"type": "probe_cache", **command})
            result = self.worker_manager.get_result(timeout=timeout)
        except (RuntimeError, queue.Empty) as e:
            logger.debug(f"Worker did not answer the cache probe: {e}")
            return None
        finally:
            self._worker_lock.release()
        if result.get("type") != "probe_cache":
            return None
        cached = result.get("cached")
        return list(cached) if isinstance(cached, list) else None
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_worker_execute.py tests/test_server.py -k "probe or Probe" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dw/worker.py dw/server/jobs.py tests/test_worker_execute.py tests/test_server.py
git commit -m "feat(server): #85 - the worker answers a cache probe, and the manager asks without blocking behind a job

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `cached_steps` in the plan and on validate

**Files:**
- Modify: `dw/plan.py` (`build_plan`), `dw/server/app.py` (validate route)
- Test: `tests/test_plan.py`, `tests/test_server.py`

**Interfaces:**
- Consumes: `JobManager.probe_cache(command)` from Task 2.
- Produces: `build_plan(..., cache_probe=None)` where `cache_probe` is `Callable[[dict], list[str] | None]` taking the run's arguments. `cached_steps` is `len(answer)` on a list, `None` on `None` or with no probe, and `0` with no probe made when the workflow is unseeded.
- Produces: `_probe_command_for(candidate, request, workspace)` in the route: the execute-shaped dict for this validate request.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_plan.py`:

```python
class TestCachedSteps:
    def test_no_probe_is_unknown(self, plan):
        assert plan()["cached_steps"] is None

    def test_the_probe_is_asked_with_the_arguments_and_counted(self, plan):
        seen = []

        def probe(arguments):
            seen.append(arguments)
            return ["still", "shot@a"]

        answer = plan(arguments={"frames": 9}, cache_probe=probe)
        assert answer["cached_steps"] == 2
        assert seen == [{"frames": 9}]

    def test_a_probe_that_cannot_answer_is_unknown(self, plan):
        assert plan(cache_probe=lambda arguments: None)["cached_steps"] is None

    def test_an_unseeded_workflow_is_zero_without_asking(self, plan):
        def probe(arguments):
            raise AssertionError("must not be asked")

        spec = definition()
        del spec["seed"]
        del spec["variables"]["seed"]
        assert plan(spec, cache_probe=probe)["cached_steps"] == 0

    def test_a_seed_variable_left_null_is_unseeded(self, plan):
        def probe(arguments):
            raise AssertionError("must not be asked")

        spec = definition()
        spec["variables"]["seed"] = None
        assert plan(spec, cache_probe=probe)["cached_steps"] == 0
```

Append to `tests/test_server.py` inside `TestValidatePlan`:

```python
    def test_cached_steps_comes_from_the_worker(self, server, monkeypatch):
        import dw.plan

        monkeypatch.setattr(dw.plan, "scan_models", lambda cache_dir=None: {"repos": []})
        with server(success_script) as client:
            manager = client.app.state.job_manager
            manager.worker_manager.ensure_worker()
            manager.worker_manager.cached_steps = ["gen"]
            seeded = valid_workflow("seeded")
            seeded["seed"] = 7
            result = client.post(
                "/api/validate?sizes=false", json={"workflow": seeded, "arguments": {"prompt": "x"}}
            ).json()
        assert result["plan"]["cached_steps"] == 1
        probe = [c for c in manager.worker_manager.commands if c["type"] == "probe_cache"]
        assert len(probe) == 1
        assert probe[0]["arguments"] == {"prompt": "x"}
        assert probe[0]["workflow"] == seeded
        assert probe[0]["output_dir"] == manager.output_dir

    def test_an_unseeded_workflow_does_not_probe(self, server, monkeypatch):
        import dw.plan

        monkeypatch.setattr(dw.plan, "scan_models", lambda cache_dir=None: {"repos": []})
        with server(success_script) as client:
            manager = client.app.state.job_manager
            manager.worker_manager.ensure_worker()
            result = client.post(
                "/api/validate?sizes=false", json={"workflow": valid_workflow("v")}
            ).json()
        assert result["plan"]["cached_steps"] == 0
        assert all(c["type"] != "probe_cache" for c in manager.worker_manager.commands)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_plan.py -k CachedSteps -v; python -m pytest tests/test_server.py -k "cached_steps or unseeded" -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`dw/plan.py`: in `build_plan`, replace `"cached_steps": None,` with `"cached_steps": cached_steps(definition, realized, arguments, cache_probe),` and add:

```python
def cached_steps(definition, realized, arguments, cache_probe):
    """How many steps the step cache would answer for this run: 0 without
    asking when the workflow is unseeded (the cache is off then), None
    when there is no probe or the probe cannot answer, else the count."""
    if not _is_seeded(definition, realized):
        return 0
    if cache_probe is None:
        return None
    answer = cache_probe(arguments or {})
    return len(answer) if isinstance(answer, list) else None


def _is_seeded(definition, realized):
    """Whether a run of this workflow has a seed before it draws one - read
    from the definition as written, since realization pins a seed of its
    own into the copy."""
    seed = definition.get("seed")
    if isinstance(seed, str) and seed.startswith(VARIABLE_PREFIX):
        name = seed.removeprefix(VARIABLE_PREFIX)
        return (realized.get("variables") or {}).get(name) is not None
    return seed is not None
```

Update the `cache_probe` line of `build_plan`'s docstring to: `cache_probe: A callable taking the run's arguments and answering the step names the worker's cache would serve, or None when it cannot say.`

`dw/server/app.py`: in the validate route, before the `build_plan` call, build the probe:

```python
            command = _probe_command_for(candidate, request, workspace, source_root)
            answer["plan"] = build_plan(
                candidate,
                request.arguments,
                device=get_device_type(get_device()),
                prompt_dir=workspace.prompts,
                lookup_sizes=sizes,
                cache_probe=lambda arguments: manager.probe_cache(
                    {**command, "arguments": arguments}
                ),
            )
```

`source_root` is the confinement the candidate was built with - set `source_root = source.root if source else workspace.workflows` in the file branch and `source_root = workspace.workflows` in the inline branch, right where each `candidate` is constructed. Add the helper beside `_argument_reference_errors`:

```python
    def _probe_command_for(candidate, request, workspace, workflow_dir):
        """The execute-shaped command a cache probe of this validate request
        needs - the same fields _run_job sends, so the worker loads the
        workflow exactly as a job would."""
        command = {
            "arguments": request.arguments,
            "output_dir": workspace.outputs,
            "workflow_dir": workflow_dir,
        }
        if workspace.assets:
            command["asset_dir"] = workspace.assets
        if request.workflow_path is not None:
            command["workflow_path"] = candidate.file_spec
        else:
            command["workflow"] = request.workflow
            command["base_dir"] = os.path.dirname(candidate.file_spec)
        return command
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_plan.py tests/test_server.py -k "CachedSteps or ValidatePlan" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dw/plan.py dw/server/app.py tests/test_plan.py tests/test_server.py
git commit -m "feat(server): #85 - the plan says how many steps the worker's cache would serve

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: the acknowledgement on the job and in history

**Files:**
- Modify: `dw/server/jobs.py` (`RERUN_SPEC_KEYS` ~46, `_ensure_schema` ~85-126, `record` ~130, `recent_summaries` ~165, `get`/`_to_detail` ~205/303, `Job.__init__`/`summary`/`detail` ~332-520, `submit` ~558, `rerun` ~761)
- Test: `tests/test_server.py`

**Interfaces:**
- Produces: `ACK_NONE = "none"`, `ACK_BOOLEAN = "boolean"`, `ACK_BOUND = "bound"` in `dw/server/jobs.py`.
- Produces: `JobManager.submit(..., acknowledged=ACK_NONE, acknowledged_cost=None)`; `JobManager.rerun(job_id, new_seed=False, acknowledged=ACK_NONE, acknowledged_cost=None)`; `JobManager.rerun_spec(job_id) -> (spec, arguments) | None` (the first half of `rerun`, so a route can plan before it queues).
- Produces: `Job.acknowledged: str`; `summary()` and `detail()` carry `acknowledged`, `detail()` carries `acknowledged_cost` (the object or `None`); history rows the same; `jobs.sqlite` column `acknowledged TEXT` (rows before it read as `none`), the object inside the stored `spec`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server.py`:

```python
class TestAcknowledgementRecord:
    def test_a_submit_records_none_by_default(self, server):
        with server(success_script) as client:
            manager = client.app.state.job_manager
            job = manager.submit(workflow=valid_workflow(), arguments={})
            assert job.acknowledged == "none"
            assert manager.describe(job)["acknowledged"] == "none"
            assert manager.describe(job)["acknowledged_cost"] is None

    def test_a_bound_submit_records_the_object(self, server):
        with server(success_script) as client:
            manager = client.app.state.job_manager
            bound = {"fingerprint": "sha256:abc", "minutes": 3.0, "downloads": []}
            job = manager.submit(
                workflow=valid_workflow(), arguments={}, acknowledged="bound", acknowledged_cost=bound
            )
            detail = manager.describe(job)
            assert detail["acknowledged"] == "bound"
            assert detail["acknowledged_cost"] == bound
            assert job.summary()["acknowledged"] == "bound"

    def test_history_keeps_the_form_and_the_object(self, server, tmp_path):
        with server(success_script) as client:
            manager = client.app.state.job_manager
            bound = {"fingerprint": "sha256:abc", "minutes": 3.0, "downloads": ["org/x"]}
            job = manager.submit(
                workflow=valid_workflow(), arguments={}, acknowledged="bound", acknowledged_cost=bound
            )
            wait_for_status(client, job.id, TERMINAL_STATES)
            row = manager.history.get(job.id)
            assert row["acknowledged"] == "bound"
            assert row["spec"]["acknowledged_cost"] == bound
            listed = [s for s in manager.history.recent_summaries() if s["id"] == job.id]
            assert listed[0]["acknowledged"] == "bound"

    def test_a_database_without_the_column_is_migrated(self, tmp_path):
        import sqlite3

        from dw.server.jobs import JobHistory

        path = tmp_path / "old.sqlite"
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE jobs (id TEXT PRIMARY KEY, workflow TEXT, status TEXT,"
                " created_at REAL, started_at REAL, finished_at REAL, arguments TEXT,"
                " spec TEXT, manifest TEXT, warnings TEXT, error TEXT)"
            )
            connection.execute(
                "INSERT INTO jobs (id, workflow, status, created_at, spec) VALUES"
                " ('old1', 'w', 'succeeded', 1.0, '{}')"
            )
        history = JobHistory(str(path))
        assert history.get("old1")["acknowledged"] == "none"

    def test_a_rerun_carries_the_original_object_for_the_record(self, server):
        with server(success_script) as client:
            manager = client.app.state.job_manager
            bound = {"fingerprint": "sha256:abc", "minutes": 3.0, "downloads": []}
            job = manager.submit(
                workflow=valid_workflow(), arguments={}, acknowledged="bound", acknowledged_cost=bound
            )
            wait_for_status(client, job.id, TERMINAL_STATES)
            rerun = manager.rerun(job.id)
            assert rerun.acknowledged == "none"
            assert rerun.spec["acknowledged_cost"] == bound
```

(Check the history class name with `grep -n "^class" dw/server/jobs.py` - use whatever the sqlite class is called if not `JobHistory`.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_server.py -k AcknowledgementRecord -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`dw/server/jobs.py`:

Constants near `TERMINAL_STATES`:

```python
# Which form of cost acknowledgement a job was queued with (#85): none (the
# web UI and every HTTP caller that sends nothing), a bare boolean, or one
# bound to the plan that was validated
ACK_NONE = "none"
ACK_BOOLEAN = "boolean"
ACK_BOUND = "bound"
```

`RERUN_SPEC_KEYS`: append `"acknowledged_cost",` with the comment `# what the original run was consented to, kept for the record - a rerun's own request decides its form`.

Schema migration, after the `run_dir` block:

```python
            # Which form of cost acknowledgement queued the job. Rows before
            # the column are 'none' - nothing recorded is nothing recorded
            if "acknowledged" not in columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN acknowledged TEXT DEFAULT 'none'"
                )
```

`record`: add `acknowledged` to the column list and `job.acknowledged` to the values (17 placeholders). `recent_summaries`: add `acknowledged` to the SELECT and `"acknowledged": row[9] or ACK_NONE` to each summary. `get`: add `acknowledged` to the SELECT; `_to_detail`: `"acknowledged": row[15] or ACK_NONE, "acknowledged_cost": (parse(row[7], {}) or {}).get("acknowledged_cost")` - reuse the parsed spec rather than parsing twice.

`Job.__init__`: `self.acknowledged = spec.get("acknowledged") or ACK_NONE`. `summary()`: add `"acknowledged": self.acknowledged`. `detail()`: add `"acknowledged_cost": self.spec.get("acknowledged_cost")`.

`submit` signature gains `acknowledged=ACK_NONE, acknowledged_cost=None`; after `spec["catalog_name"] = catalog_name` add:

```python
        # The acknowledgement form travels with the job so history can say
        # whether this run was consented to at its actual size (#85)
        spec["acknowledged"] = acknowledged
        if acknowledged_cost is not None:
            spec["acknowledged_cost"] = acknowledged_cost
```

Docstring line: `` `acknowledged` is the form of cost acknowledgement the caller gave (none/boolean/bound) and `acknowledged_cost` the bound object, both recorded, neither checked here - the route checks. ``

Split `rerun`: everything up to and including the `arguments = historical["arguments"]` branch becomes

```python
    def rerun_spec(self, job_id):
        """The spec and arguments a rerun of `job_id` would submit, or None
        for an unknown job - split from rerun() so a route can plan the run
        before queuing it (#85)."""
```

returning `(spec, arguments)`; `rerun(self, job_id, new_seed=False, acknowledged=ACK_NONE, acknowledged_cost=None)` calls it, keeps the seed and workspace logic, and passes to `submit`: `acknowledged=acknowledged, acknowledged_cost=acknowledged_cost if acknowledged_cost is not None else spec.get("acknowledged_cost")`.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_server.py tests/test_jobs_listing.py tests/test_jobs_reused.py -v -k "Acknowledgement or rerun or history or listing"`
Expected: PASS; then `python -m pytest tests/test_server.py -q` all green.

- [ ] **Step 5: Commit**

```bash
git add dw/server/jobs.py tests/test_server.py
git commit -m "feat(server): #85 - a job records which form of cost acknowledgement queued it

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: the bound form and the 409 on `POST /api/jobs` and `/rerun`

**Files:**
- Modify: `dw/server/app.py` (`JobRequest` ~116, `RerunRequest` ~947, `submit_job` ~810, `rerun_job` ~956)
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `build_plan`, `JobManager.rerun_spec`, `submit(acknowledged=, acknowledged_cost=)`, `rerun(acknowledged=, acknowledged_cost=)`, `ACK_*`.
- Produces: `AcknowledgedCost(BaseModel)` with `fingerprint: str`, `minutes: float | None = None`, `downloads: list[str] = []`; `acknowledged_cost: bool | AcknowledgedCost | None = None` on both request models; `_acknowledgement_form(value) -> str`; `_check_bound_acknowledgement(candidate, arguments, acknowledged, workspace)` raising `HTTPException(409, detail={...})`.
- The 409 `detail` is `{"message": str, "reason": "fingerprint" | "downloads" | "unplannable", "acknowledged": {...}, "plan": {...} | None}`. (`message`, not `detail`, so `DwClient._format_detail` already renders it.)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server.py`:

```python
def plan_for(client, workflow, arguments=None):
    body = {"workflow": workflow}
    if arguments:
        body["arguments"] = arguments
    answer = client.post("/api/validate?sizes=false", json=body).json()
    assert answer["valid"], answer
    return answer["plan"]


def bound(plan):
    return {
        "fingerprint": plan["fingerprint"],
        "minutes": plan["estimate"]["minutes"],
        "downloads": [d["repo"] for d in plan["downloads_required"] if d["repo"]],
    }


@pytest.fixture
def no_hub(monkeypatch):
    import dw.plan

    monkeypatch.setattr(dw.plan, "scan_models", lambda cache_dir=None: {"repos": []})


def list_workflow(job_id="listed"):
    return {
        "id": job_id,
        "seed": "variable:seed",
        "variables": {"seed": 1, "shots": [{"name": "a", "prompt": "a"}]},
        "steps": [
            {
                "name": "shot",
                "for_each": "variable:shots",
                "pipeline": {
                    "configuration": {"component_type": "{Fake}", "no_generator": True},
                    "from_pretrained_arguments": {"model_name": "m"},
                    "arguments": {"prompt": "item:prompt"},
                },
            }
        ],
    }


class TestBoundAcknowledgement:
    def test_a_matching_fingerprint_queues(self, server, no_hub):
        with server(success_script) as client:
            plan = plan_for(client, list_workflow())
            response = client.post(
                "/api/jobs", json={"workflow": list_workflow(), "acknowledged_cost": bound(plan)}
            )
            assert response.status_code == 201, response.json()
            assert response.json()["acknowledged"] == "bound"
            assert response.json()["acknowledged_cost"] == bound(plan)

    def test_a_longer_list_than_acknowledged_is_refused(self, server, no_hub):
        with server(success_script) as client:
            plan = plan_for(client, list_workflow())
            longer = {"shots": [{"name": n, "prompt": n} for n in "abc"]}
            response = client.post(
                "/api/jobs",
                json={"workflow": list_workflow(), "arguments": longer, "acknowledged_cost": bound(plan)},
            )
            assert response.status_code == 409
            detail = response.json()["detail"]
            assert detail["reason"] == "fingerprint"
            assert detail["acknowledged"]["fingerprint"] == plan["fingerprint"]
            assert detail["plan"]["list_entries"] == {"shots": 3}
            assert detail["plan"]["fingerprint"] != plan["fingerprint"]
            assert "differ" in detail["message"]
            assert client.app.state.job_manager.worker_manager.commands == []

    def test_a_new_seed_is_the_same_work(self, server, no_hub):
        with server(success_script) as client:
            plan = plan_for(client, list_workflow())
            response = client.post(
                "/api/jobs",
                json={"workflow": list_workflow(), "arguments": {"seed": 99}, "acknowledged_cost": bound(plan)},
            )
            assert response.status_code == 201

    def test_a_download_not_acknowledged_is_refused(self, server, no_hub):
        with server(success_script) as client:
            plan = plan_for(client, list_workflow())
            acknowledgement = bound(plan)
            acknowledgement["downloads"] = []  # the caller left the repo out
            response = client.post(
                "/api/jobs", json={"workflow": list_workflow(), "acknowledged_cost": acknowledgement}
            )
            assert response.status_code == 409
            detail = response.json()["detail"]
            assert detail["reason"] == "downloads"
            assert "m" in detail["message"]

    def test_a_download_that_vanished_is_not_a_refusal(self, server, no_hub, monkeypatch):
        with server(success_script) as client:
            plan = plan_for(client, list_workflow())
            assert bound(plan)["downloads"] == ["m"]
            import dw.plan

            monkeypatch.setattr(
                dw.plan, "scan_models", lambda cache_dir=None: {"repos": [{"repo_id": "m"}]}
            )
            response = client.post(
                "/api/jobs", json={"workflow": list_workflow(), "acknowledged_cost": bound(plan)}
            )
            assert response.status_code == 201

    def test_an_unplannable_run_is_refused_not_passed(self, server, no_hub, monkeypatch):
        import dw.server.app as app_module

        with server(success_script) as client:
            plan = plan_for(client, list_workflow())

            def boom(*a, **k):
                raise RuntimeError("no plan")

            monkeypatch.setattr(app_module, "build_plan", boom)
            response = client.post(
                "/api/jobs", json={"workflow": list_workflow(), "acknowledged_cost": bound(plan)}
            )
            assert response.status_code == 409
            assert response.json()["detail"]["reason"] == "unplannable"
            assert response.json()["detail"]["plan"] is None

    def test_true_and_absent_queue_without_planning(self, server, no_hub, monkeypatch):
        import dw.server.app as app_module

        def boom(*a, **k):
            raise AssertionError("the boolean path must not plan")

        monkeypatch.setattr(app_module, "build_plan", boom)
        with server(success_script) as client:
            plain = client.post("/api/jobs", json={"workflow": valid_workflow("p")})
            flagged = client.post(
                "/api/jobs", json={"workflow": valid_workflow("f"), "acknowledged_cost": True}
            )
            off = client.post(
                "/api/jobs", json={"workflow": valid_workflow("o"), "acknowledged_cost": False}
            )
        assert plain.json()["acknowledged"] == "none"
        assert flagged.json()["acknowledged"] == "boolean"
        assert off.json()["acknowledged"] == "none"

    def test_a_bound_form_without_a_fingerprint_is_a_422(self, server):
        with server(success_script) as client:
            response = client.post(
                "/api/jobs", json={"workflow": valid_workflow(), "acknowledged_cost": {"minutes": 3}}
            )
            assert response.status_code == 422

    def test_a_stored_prompt_edited_after_validation_is_refused(self, server, no_hub, tmp_path):
        (tmp_path / "prompts" / "p.json").write_text(json.dumps({"text": "before"}))
        workflow = valid_workflow("prompted")
        workflow["variables"]["prompt"] = "prompt:p"
        with server(success_script) as client:
            plan = plan_for(client, workflow)
            (tmp_path / "prompts" / "p.json").write_text(json.dumps({"text": "after"}))
            response = client.post(
                "/api/jobs", json={"workflow": workflow, "acknowledged_cost": bound(plan)}
            )
            assert response.status_code == 409
            assert response.json()["detail"]["reason"] == "fingerprint"


class TestBoundRerun:
    def test_a_rerun_with_the_original_plan_queues_even_with_a_new_seed(self, server, no_hub):
        with server(success_script) as client:
            plan = plan_for(client, list_workflow())
            first = client.post(
                "/api/jobs", json={"workflow": list_workflow(), "acknowledged_cost": bound(plan)}
            ).json()
            wait_for_status(client, first["id"], TERMINAL_STATES)
            response = client.post(
                f"/api/jobs/{first['id']}/rerun",
                json={"new_seed": True, "acknowledged_cost": bound(plan)},
            )
            assert response.status_code == 201
            assert response.json()["acknowledged"] == "bound"

    def test_a_rerun_bound_to_a_stale_plan_is_refused(self, server, no_hub):
        with server(success_script) as client:
            first = client.post("/api/jobs", json={"workflow": list_workflow()}).json()
            wait_for_status(client, first["id"], TERMINAL_STATES)
            other = plan_for(client, valid_workflow("other"))
            response = client.post(
                f"/api/jobs/{first['id']}/rerun", json={"acknowledged_cost": bound(other)}
            )
            assert response.status_code == 409
            assert response.json()["detail"]["reason"] == "fingerprint"

    def test_a_rerun_with_true_is_unchanged(self, server, no_hub):
        with server(success_script) as client:
            first = client.post("/api/jobs", json={"workflow": list_workflow()}).json()
            wait_for_status(client, first["id"], TERMINAL_STATES)
            response = client.post(f"/api/jobs/{first['id']}/rerun", json={"acknowledged_cost": True})
            assert response.status_code == 201
            assert response.json()["acknowledged"] == "boolean"

    def test_an_unknown_job_is_still_404(self, server):
        with server(success_script) as client:
            response = client.post(
                "/api/jobs/nope/rerun", json={"acknowledged_cost": {"fingerprint": "sha256:0"}}
            )
            assert response.status_code == 404
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_server.py -k "BoundAcknowledgement or BoundRerun" -v`
Expected: FAIL (`acknowledged_cost` is dropped as an unknown field, so most answer 201 without `acknowledged`, or `acknowledged: none`).

- [ ] **Step 3: Implement**

`dw/server/app.py`:

Imports: add `ACK_BOOLEAN, ACK_BOUND, ACK_NONE` to the `from .jobs import ...` line (find it with `grep -n "from .jobs import" dw/server/app.py`); add `from typing import Union` if `Union` is not already imported (check the existing `Optional` import line).

Above `class JobRequest`:

```python
class AcknowledgedCost(BaseModel):
    """A cost acknowledgement bound to the plan a validate call answered
    with (#85): the server refuses to queue a run whose plan no longer
    matches it. `minutes` is recorded, never compared."""

    fingerprint: str = Field(description="plan.fingerprint from POST /api/validate")
    minutes: Optional[float] = Field(
        default=None, description="plan.estimate.minutes, recorded on the job"
    )
    downloads: List[str] = Field(
        default_factory=list,
        description="The repos in plan.downloads_required that were acknowledged",
    )


ACKNOWLEDGED_COST_FIELD = Field(
    default=None,
    description="Cost acknowledgement: true (recorded), or an object "
    "{fingerprint, minutes, downloads} bound to the plan validate answered "
    "with - then the run is refused with 409 if its plan changed",
)
```

(`List` - check `from typing import` at the top; add it if only `Dict, Any, Optional` are imported.)

`JobRequest` gains `acknowledged_cost: Optional[Union[bool, AcknowledgedCost]] = ACKNOWLEDGED_COST_FIELD`. `RerunRequest` (inside `create_app`) gains the same line.

Helpers inside `create_app`, beside `_argument_reference_errors`:

```python
    def _acknowledgement_form(value):
        """none | boolean | bound - classified once, here, so the check and
        the record agree."""
        if isinstance(value, AcknowledgedCost):
            return ACK_BOUND
        return ACK_BOOLEAN if value is True else ACK_NONE

    def _check_bound_acknowledgement(candidate, arguments, acknowledged, workspace):
        """Refuse with 409 when the run `candidate` + `arguments` will
        execute is not the one `acknowledged` was bound to: a different
        fingerprint, or a download the caller did not acknowledge. The body
        carries the current plan so the agent re-quotes from it without a
        second validate call. A plan that cannot be built is a refusal too -
        never a silent pass (#85).
        """
        record = acknowledged.model_dump()
        try:
            from .. import get_device, get_device_type

            current = build_plan(
                candidate,
                arguments,
                device=get_device_type(get_device()),
                prompt_dir=workspace.prompts,
                lookup_sizes=False,
            )
        except Exception:
            logger.exception("Plan could not be built for a bound acknowledgement")
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "The run could not be planned, so a bound "
                    "acknowledgement cannot be checked; acknowledge with true "
                    "or validate again",
                    "reason": "unplannable",
                    "acknowledged": record,
                    "plan": None,
                },
            )
        if current["fingerprint"] != acknowledged.fingerprint:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "The run's shape changed since it was acknowledged: "
                    "the workflow or its arguments differ from what was validated",
                    "reason": "fingerprint",
                    "acknowledged": record,
                    "plan": current,
                },
            )
        missing = [
            entry["repo"]
            for entry in current["downloads_required"]
            if entry.get("repo") and entry["repo"] not in acknowledged.downloads
        ]
        if missing:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "The run's shape changed since it was acknowledged: "
                    f"it now has to download {', '.join(missing)} first",
                    "reason": "downloads",
                    "acknowledged": record,
                    "plan": current,
                },
            )

    def _candidate_for(workflow_path, workflow, base_dir, output_dir, workflow_dir):
        """The Workflow a job spec names, built as the worker will build it."""
        if workflow_path is not None:
            return workflow_from_file(workflow_path, output_dir, workflow_dir)
        return workflow_from_definition(
            copy.deepcopy(workflow), output_dir, base_dir, workflow_dir
        )
```

In `submit_job`, after the `reference_problems` check and before `manager.submit(...)`:

```python
            form = _acknowledgement_form(request.acknowledged_cost)
            if form == ACK_BOUND:
                confinement = source.root if source else workspace.workflows
                candidate = _candidate_for(
                    resolved, request.workflow, request.base_dir,
                    workspace.outputs, confinement,
                )
                _check_bound_acknowledgement(
                    candidate, request.arguments, request.acknowledged_cost, workspace
                )
```

and pass to `manager.submit`: `acknowledged=form, acknowledged_cost=(request.acknowledged_cost.model_dump() if form == ACK_BOUND else None)`. The `except HTTPException: raise` already precedes the catch-all, so the 409 passes through.

In `rerun_job`:

```python
        form = _acknowledgement_form(body.acknowledged_cost)
        if form == ACK_BOUND:
            prepared = manager.rerun_spec(job_id)
            if prepared is None:
                raise HTTPException(status_code=404, detail="Unknown job")
            spec, arguments = prepared
            try:
                candidate = _candidate_for(
                    spec.get("workflow_path"), spec.get("workflow"), spec.get("base_dir"),
                    spec.get("output_dir") or manager.output_dir, spec.get("workflow_dir"),
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
            _check_bound_acknowledgement(
                candidate, arguments, body.acknowledged_cost,
                _workspace_for(spec.get("workspace")),
            )
        try:
            job = manager.rerun(
                job_id,
                new_seed=body.new_seed,
                acknowledged=form,
                acknowledged_cost=(
                    body.acknowledged_cost.model_dump() if form == ACK_BOUND else None
                ),
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
```

(`_workspace_for(None)` is the default workspace - see its first line.) Update the route docstring: `Pass acknowledged_cost as on POST /api/jobs; a bound one is checked against the stored spec's plan - the fresh seed of new_seed does not change a fingerprint.`

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_server.py -v -k "Bound or ValidatePlan or Acknowledgement"` then `python -m pytest tests/test_server.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dw/server/app.py tests/test_server.py
git commit -m "feat(server): #85 - a bound acknowledgement is checked against the run's plan, and a changed plan is a 409

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: MCP - the bound form over the tools, and the 409 rendered with the new estimate

**Files:**
- Modify: `dw_mcp/diagnose.py` (`COST_REFUSAL` ~25, `run_workflow` ~33, `rerun_job` ~248), `dw_mcp/server.py` (`run_workflow` ~794, `rerun_job` ~907), `dw_mcp/client.py` (`_format_detail` ~325)
- Test: `tests/test_mcp_diagnose.py`, `tests/test_mcp_client.py`, `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: the 409 body shape from Task 5.
- Produces: `diagnose.run_workflow(..., acknowledged_cost: bool | dict)`; `diagnose.rerun_job(..., acknowledged_cost: bool | dict)`; a dict is forwarded verbatim as the body's `acknowledged_cost`; a dict without `fingerprint` raises `DwApiError` before any request. `DwClient._format_detail` appends the plan's estimate and downloads when a dict detail carries `plan`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mcp_diagnose.py`:

```python
BOUND = {"fingerprint": "sha256:abc", "minutes": 4.0, "downloads": ["org/x"]}


def test_run_forwards_a_bound_acknowledgement_verbatim():
    import json

    client, seen = submitting()
    diagnose.run_workflow(client, workflow_path="w.json", acknowledged_cost=BOUND)
    assert json.loads(seen[0]["body"])["acknowledged_cost"] == BOUND


def test_run_does_not_send_a_bare_true():
    """The boolean path is the MCP layer's gate, not the server's - the body
    stays what it was."""
    import json

    client, seen = submitting()
    diagnose.run_workflow(client, workflow_path="w.json", acknowledged_cost=True)
    assert "acknowledged_cost" not in json.loads(seen[0]["body"])


def test_run_refuses_a_bound_form_without_a_fingerprint():
    client, seen = submitting()
    with pytest.raises(DwApiError, match="fingerprint"):
        diagnose.run_workflow(client, workflow_path="w.json", acknowledged_cost={"minutes": 4})
    assert seen == []


def test_run_refuses_an_empty_dict_as_unacknowledged():
    client, seen = submitting()
    with pytest.raises(DwApiError, match="acknowledged_cost"):
        diagnose.run_workflow(client, workflow_path="w.json", acknowledged_cost={})
    assert seen == []


def test_a_409_surfaces_with_the_new_estimate():
    client, _seen = scripted(
        {
            ("POST", "/api/jobs"): (
                409,
                {
                    "detail": {
                        "message": "The run's shape changed since it was acknowledged: the workflow or its arguments differ from what was validated",
                        "reason": "fingerprint",
                        "acknowledged": BOUND,
                        "plan": {
                            "fingerprint": "sha256:def",
                            "steps": 6,
                            "list_entries": {"shots": 5},
                            "cached_steps": None,
                            "downloads_required": [{"repo": "org/y", "gb": 3.5}],
                            "estimate": {"minutes": 19.0, "basis": "per_entry", "device": "cuda", "measured_on": "card", "partial": False},
                        },
                    }
                },
            )
        }
    )
    with pytest.raises(DwApiError) as caught:
        diagnose.run_workflow(client, workflow_path="w.json", acknowledged_cost=BOUND)
    message = str(caught.value)
    assert "shape changed" in message
    assert "19.0" in message and "per_entry" in message
    assert "org/y" in message
    assert "sha256:def" in message


def test_rerun_forwards_a_bound_acknowledgement():
    import json

    client, seen = scripted({("POST", "/api/jobs/job-1/rerun"): (201, SUBMITTED)})
    diagnose.rerun_job(client, "job-1", acknowledged_cost=BOUND, new_seed=True)
    body = json.loads(seen[0]["body"])
    assert body["acknowledged_cost"] == BOUND and body["new_seed"] is True
```

Append to `tests/test_mcp_diagnose.py` as well (the refusal text):

```python
def test_the_refusal_teaches_the_bound_form():
    from dw_mcp.diagnose import COST_REFUSAL

    assert "fingerprint" in COST_REFUSAL
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_mcp_diagnose.py -k "bound or 409 or forwards or teaches or empty_dict" -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`dw_mcp/diagnose.py`:

```python
COST_REFUSAL = (
    "Running a workflow occupies the GPU for minutes and the engine runs one "
    "job at a time. Call `validate_workflow` with the arguments you will run "
    "with (free): its `plan` says what will execute - `estimate.minutes` with "
    "its `basis`, and any weights in `downloads_required` this box has to "
    "fetch first. Tell the user that number, get their go-ahead, then call "
    "again with acknowledged_cost bound to the plan: {\"fingerprint\": "
    "plan.fingerprint, \"minutes\": plan.estimate.minutes, \"downloads\": "
    "[each downloads_required repo]} - the server then refuses (409) if the "
    "run's shape changed since. Pass true instead only when `plan` was null."
)


def _acknowledgement_body(acknowledged_cost):
    """What a bound acknowledgement adds to a request body: the dict itself,
    verbatim, so the server compares what the agent quoted. A dict without
    a fingerprint is a mistake caught here, before anything is queued; a
    bare true adds nothing - the boolean gate is this layer's, not the
    server's."""
    if isinstance(acknowledged_cost, dict):
        if not acknowledged_cost.get("fingerprint"):
            raise DwApiError(
                "A bound acknowledged_cost needs `fingerprint` - the "
                "plan.fingerprint the validate answer carried. Validate again "
                "and pass {fingerprint, minutes, downloads} from its plan."
            )
        return {"acknowledged_cost": acknowledged_cost}
    return {}
```

In `run_workflow`, change the gate to `if not acknowledged_cost:` (unchanged - an empty dict is falsy and refuses) and after `payload = {"arguments": arguments or {}}` add `payload.update(_acknowledgement_body(acknowledged_cost))`. In `rerun_job` the body becomes `{"new_seed": new_seed, **_acknowledgement_body(acknowledged_cost)}`. Update both docstrings' gate sentence: `acknowledged_cost is true or, better, the plan it was quoted from: {fingerprint, minutes, downloads} - see COST_REFUSAL`.

`dw_mcp/server.py`: both tool signatures become `acknowledged_cost: bool | dict = False`. Append to `run_workflow`'s docstring: `Bind the acknowledgement to what you quoted: pass {"fingerprint": plan.fingerprint, "minutes": plan.estimate.minutes, "downloads": [...repos from plan.downloads_required]} from the validate answer, and the server refuses with 409 - naming the new plan - if the run's shape changed since; bare true is for a plan that was null.` Append to `rerun_job`'s: `acknowledged_cost takes the same bound form as run_workflow; a fresh seed never changes the fingerprint, so the original plan still binds a new_seed rerun.`

`dw_mcp/client.py`, `_format_detail`, inside the `if isinstance(detail, dict) and "message" in detail:` branch, before `return formatted`:

```python
            plan = detail.get("plan")
            if isinstance(plan, dict):
                # A 409 from the cost gate: say what the run costs now, so a
                # client that only sees the message can re-quote from it
                estimate = plan.get("estimate") or {}
                formatted += (
                    f" It now estimates {estimate.get('minutes')} minutes "
                    f"(basis {estimate.get('basis')})"
                )
                downloads = [
                    entry.get("repo") or entry.get("url")
                    for entry in plan.get("downloads_required") or []
                ]
                if downloads:
                    formatted += f", and would download {', '.join(downloads)} first"
                formatted += f"; new fingerprint {plan.get('fingerprint')}."
```

- [ ] **Step 4: Run the MCP suites**

Run: `python -m pytest tests/test_mcp_diagnose.py tests/test_mcp_client.py tests/test_mcp_server.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dw_mcp/diagnose.py dw_mcp/server.py dw_mcp/client.py tests/test_mcp_diagnose.py
git commit -m "feat(mcp): #85 - acknowledged_cost binds to the plan that was quoted, and a 409 re-quotes

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: docs, skills, proposal status

**Files:**
- Modify: `docs/SERVER.md` (`POST /api/jobs` bullet - find with `grep -n "POST /api/jobs" docs/SERVER.md`; the validate `plan` paragraph's `cached_steps, reserved (null)`), `docs/MCP.md` (`## The cost gate`, the `run_workflow`/`rerun_job` table rows, the troubleshooting row `run_workflow or rerun_job refuses with a cost message`), `docs/WORKFLOW_GUIDE.md` ("The loop" step 4), `CLAUDE.md` (the `plan` sentence added in stage 1), `plugins/dw/skills/*/SKILL.md` (step 2 of "Run and judge"), `docs/proposals/acknowledged-cost-binding.md` (status line)
- Test: `tests/test_plugin_skills.py`, `tests/test_docs_links.py`

- [ ] **Step 1: Docs**

`docs/SERVER.md`:
- In the validate `plan` paragraph replace `` `cached_steps`, reserved (`null`) `` with `` `cached_steps`, how many of those steps the worker's step cache would serve (`0` for an unseeded workflow, `null` when the worker is busy or did not answer) ``.
- To the `POST /api/jobs` bullet append: `` Takes an optional `acknowledged_cost`: `true` is recorded as `acknowledged: boolean`; the object `{fingerprint, minutes, downloads}` from a validate answer's `plan` is `bound` - the server re-plans the run for the arguments given and answers **409** when the fingerprint differs or a repo in `downloads_required` is not in `downloads` (a download that has since vanished is not a refusal); the body is `{"detail": {message, reason: "fingerprint" | "downloads" | "unplannable", acknowledged, plan}}` with the current plan, so the caller re-quotes from it. `minutes` is recorded, never compared. Nothing is required: the web UI and every caller that sends nothing are `acknowledged: none`, and every job answer and history row carries `acknowledged` (and `acknowledged_cost` when bound). `POST /api/jobs/{id}/rerun` takes the same field and checks against the stored spec; a fresh seed does not change a fingerprint. ``

`docs/MCP.md`:
- `## The cost gate`: after the paragraph ending `and gating them would make the safe direction the harder one.` add: `` The acknowledgement can be bound to what was quoted. `validate_workflow` answers with a `plan`; pass `acknowledged_cost={"fingerprint": plan.fingerprint, "minutes": plan.estimate.minutes, "downloads": [...]}` and the server refuses with 409 if the run's shape changed between the quote and the call - a longer list, a stored prompt edited meanwhile, weights that now have to be downloaded - naming the new plan so the agent re-quotes. Bare `true` still works and is for a `plan` that came back null; the job records which form it got (`acknowledged: none | boolean | bound`). ``
- Step 2 of "The intended loop": replace `` `run_workflow` with `acknowledged_cost=true` `` with `` `run_workflow` with `acknowledged_cost` bound to the plan (`{fingerprint, minutes, downloads}`), or `true` when there was no plan ``.
- Tool table rows for `run_workflow` and `rerun_job`: `acknowledged_cost=False` → `acknowledged_cost=False` stays in the signature column; append to each description ` - `acknowledged_cost` is `true` or the bound `{fingerprint, minutes, downloads}` from the validate plan; a 409 means the plan changed and the message carries the new estimate`.
- Troubleshooting row: append `; a 409 "shape changed" answer means the run grew since the quote - re-validate, re-quote, pass the new plan`.

`docs/WORKFLOW_GUIDE.md`, "The loop" step 4: after `never counted.` add ` Then pass that plan back: `acknowledged_cost={"fingerprint": plan.fingerprint, "minutes": plan.estimate.minutes, "downloads": [...]}` - the server refuses with 409 if the run's shape changed since the quote, and the refusal carries the new plan to quote from. `true` is for a plan that was null.`

`CLAUDE.md`: extend the stage-1 sentence: after `never a changed verdict` add `; `acknowledged_cost` on `POST /api/jobs` / `rerun` takes `true` (recorded) or the plan's `{fingerprint, minutes, downloads}` (checked - 409 with the current plan when the fingerprint or the required downloads changed; `minutes` never compared), and the job records `acknowledged: none | boolean | bound`. `cached_steps` is the worker's answer to a `probe_cache` command (`Workflow.cache_hits`, which shares `_prepare_definition` / `_cache_lookup` with `run` so the two cannot drift)`.

`docs/proposals/acknowledged-cost-binding.md`: status line → `Status: **implemented** (stage 1 b37f033, stage 2 on `cost-binding`). Design: docs/superpowers/specs/2026-09-12-acknowledged-cost-binding-design.md.`

- [ ] **Step 2: Skills**

Each "Run and judge" step 2 ends with a sentence about `run_workflow with acknowledged_cost=true`. Replace, in all three, `` `run_workflow` with `acknowledged_cost=true` `` with `` `run_workflow` with `acknowledged_cost` set to the plan's `{fingerprint, minutes, downloads}` ``. That is +36 bytes each. `minimax-h3` has 22 bytes of headroom and `ltx-2.5` 108, so trim `minimax-h3` first: in its step 2, `(warm minutes; a first\n   load, and any \`downloads_required\`, is longer)` → `(warm minutes; a first load\n   or a \`downloads_required\` is longer)` (−14) and in step 1 `does not accept.` → `rejects.` (−8) — recount with `wc -c` and keep trimming words (not numbers) until under 12288.

- [ ] **Step 3: Run the doc and skill tests**

Run: `python -m pytest tests/test_plugin_skills.py tests/test_docs_links.py -q && wc -c plugins/dw/skills/*/SKILL.md`
Expected: PASS; every skill under 12288.

- [ ] **Step 4: Commit**

```bash
git add docs CLAUDE.md plugins/dw/skills
git commit -m "docs: #85 - the bound acknowledgement, the 409 and cached_steps

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: whole-suite verification

- [ ] **Step 1:** `python -m pytest -q -x --ignore=tests/test_integration.py -p no:cacheprovider 2>&1 | tail -3` - expect all green (3893 + the new tests).
- [ ] **Step 2:** `git status --short && git log --oneline develop..HEAD` - clean, eight commits (plan + seven tasks).

---

## Self-review

**Spec coverage:** §6 (models, classification table) - Task 5. §7 (the check, `unplannable`, fingerprint, downloads, body shape) - Task 5; body uses `message` rather than nested `detail.detail` so the existing client formatter renders it, noted in Interfaces. §8 (`Job.acknowledged`, sqlite, `describe`, `RERUN_SPEC_KEYS`) - Task 4. §9 (MCP widening, forwarding, 409 rendering, `COST_REFUSAL`) - Task 6. §10 (probe command, the `run` refactor, `JobManager.probe_cache`, `build_plan(cache_probe=)`, unseeded → 0 with no probe, no scaling by `cached_steps`) - Tasks 1-3, with the command-shape deviation stated up top. §11 tests - each task. Docs - Task 7. Release-note items - the proposal status line; the note itself stays owed with stage 1's.

**Type consistency:** `cache_hits(arguments) -> list[str]` (Task 1) is what `_handle_probe_cache` calls (Task 2); `probe_cache(command, timeout)` (Task 2) is what the validate closure calls (Task 3) with `{**command, "arguments": arguments}`; `cache_probe(arguments)` (Task 3) matches `cached_steps()`'s call; `rerun_spec` / `submit(acknowledged=, acknowledged_cost=)` / `rerun(acknowledged=, acknowledged_cost=)` (Task 4) match the route (Task 5); the 409 body keys (`message`, `reason`, `acknowledged`, `plan`) match the client test and formatter (Task 6).
