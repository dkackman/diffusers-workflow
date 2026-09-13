# Acknowledged-cost binding: design

Date: 2026-09-12. Proposal: [docs/proposals/acknowledged-cost-binding.md](../../proposals/acknowledged-cost-binding.md),
answering issue #85. Branch: `cost-plan`, off `develop` at 2d87005.

## Goal

The free pre-flight (`POST /api/validate`, MCP `validate_workflow`) answers
with a *plan*: what the run will actually execute for the arguments given,
what it will have to download first, and a cost estimate whose basis is
named. An agent quotes that instead of the catalog's defaults-only `cost`.
Then a caller may bind its acknowledgement to that plan, and the server
refuses to queue a run whose shape no longer matches what was acknowledged.

Two stages, each shippable alone:

- **Stage 1** - `plan` on the validate answer, surfaced over MCP, taught by
  the skills.
- **Stage 2** - the bound `acknowledged_cost` form, the 409, the job record,
  and the worker-side step-cache probe.

## Non-goals

- A numeric budget check. `minutes` is recorded, never compared; a tolerance
  needs `per_entry` measured on the templates, and none carries it yet.
- Requiring an acknowledgement over HTTP. The web UI and every HTTP caller
  keep queuing without one; the gate stays an agent-behaviour gate.
- Run-time metering or aborting a run that overruns its estimate.
- Pricing the Cartesian product of `previous_result:` references. The count
  is static only when every multiplicand is an argument, and a number that
  is sometimes a guess is worse than none.
- Changing what `list_workflows` reports.

## Global constraints

- Model knowledge stays out of engine code: no repo name, no per-model
  minute figure. Every number comes from a workflow's `cost` block.
- `dw/plan.py` imports nothing from `dw.server` or `dw.worker`, and touches
  the worker only through a callable the caller hands it (stage 2).
- Plan construction is best effort at the API: a failure is logged and
  answered as `plan: null`; the validate verdict is the schema's, never the
  planner's.
- Every path the planner reads goes through the resolvers the run uses
  (`realize_workflow`, the sub-workflow digest's confinement, `scan_models`).
  It opens no file by a path it computed itself.
- The bare-boolean and absent acknowledgement paths are byte-for-byte
  unchanged in behaviour.
- Nothing here names a model in a test either: fixtures declare their own
  `cost` blocks and their own `model_name` strings.

---

# Stage 1: the plan on validate

## 1. `dw/plan.py`

```python
def build_plan(
    definition,
    arguments,
    *,
    base_dir=None,
    prompt_dir=None,
    output_root=None,
    workflow_dir=None,
    device,
    cache_dir=None,
    cache_probe=None,       # stage 2
    lookup_sizes=True,
):
    """What a run of `definition` with `arguments` will execute and cost."""
```

Returns:

```json
{
  "fingerprint": "sha256:<64 hex>",
  "steps": 14,
  "list_entries": {"shots": 12},
  "cached_steps": null,
  "downloads_required": [
    {"repo": "org/model", "gb": 41.2},
    {"repo": null, "url": "https://…/x.safetensors", "gb": null}
  ],
  "estimate": {
    "minutes": 38.0,
    "basis": "per_entry",
    "device": "cuda",
    "measured_on": "RTX 4090",
    "partial": false
  }
}
```

### 1.1 Realization and expansion

- `realized, _ = realize_workflow(definition, arguments, seed=0, base_dir=…,
  prompt_dir=…, output_root=…, workflow_dir=…, pin_outputs=False)`.
  `pin_outputs` is a new keyword on `realize_workflow`, default `True`,
  which `Workflow.run` never sets; with `False`, `_pin_output` returns the
  reference as written. Prompt inlining still happens.
- `expanded = expand_for_each(realized)` - the member list a run executes.
  `steps` is `len(expanded["steps"])`.
- `list_entries` is `{variable: len(value)}` for each `for_each` in the
  *unexpanded* realized definition whose value is `variable:<name>`, read
  from the folded `variables` block. A literal list is not an argument and
  is not listed.
- `realize_workflow` raises on an undeclared argument or an uncoercible
  value; the API calls `build_plan` only after `argument_errors` passed,
  so this is not reached there. `build_plan` lets it propagate.

### 1.2 Fingerprint

SHA-256 of `json.dumps(doc, sort_keys=True, separators=(",", ":"),
ensure_ascii=False)` where `doc` is `expanded` after:

1. the top-level `seed` removed;
2. if the definition *as written* had `seed: "variable:<name>"`, the folded
   value of `variables.<name>` removed (the key stays, its `default` is
   deleted) - otherwise the seed survives in the variables block;
3. any `seed` key at a step or a step's `pipeline` removed. A step seed that
   was `variable:<the seed variable>` was substituted by realization, which
   is why rule 2 removes the value at its source as well;
4. `cost`, `description`, `summary` and `configures` removed at the top level
   - documentation, not work.

Presented as `"sha256:" + hexdigest`. The expanded step names are inside the
document, so the member set is covered without a second input.

What must hold (tests):

| Same fingerprint | Different fingerprint |
|---|---|
| a different seed (top level or step) | a longer or shorter `for_each` list |
| key order in the JSON file | a stored prompt whose text changed |
| whitespace, `description`, `cost` edits | a changed `num_frames`/`num_inference_steps` |
| a new run landing under `output:…/latest/…` | a different asset name |
| the same arguments given in a different order | a step added, removed or renamed |

### 1.3 Estimate

Input: the workflow's `cost` list (schema: `[{device, name?, vram_gb,
minutes, per_entry?}]`) and `device`, the backend that is serving
(`get_device_type(get_device())` - `cuda`, `mps` or `cpu`, never an
index).

1. No `cost` list, or an empty one → `{"minutes": null, "basis":
   "unknown", "device": device, "measured_on": null, "partial": false}`.
2. Pick the first entry whose `device` equals the serving backend. None →
   the first entry in the list, and `basis` is `other_device`; its
   `minutes` is reported so the agent has a figure to scale, and
   `measured_on` says what it was measured on.
3. With an entry chosen: if it carries `per_entry` and
   `per_entry.variable` is a key of `list_entries`, `minutes = max(0,
   (entry.minutes - per.minutes * per.entries) + per.minutes * N)` with
   `N = list_entries[variable]`, `basis: per_entry`. Otherwise
   `minutes = entry.minutes`, `basis: catalog` (or `other_device` from
   step 2 - `other_device` wins over `per_entry`, since scaling a figure
   from the wrong card compounds the error).
4. **Sub-workflows.** For every expanded step whose `workflow` is
   `{"path": …}` and not `builtin:`, load the child through the same
   resolution `_record_sub_workflows` uses (relative to `base_dir`,
   confined to `workflow_dir`) and apply rules 1-3 to *its* `cost` with the
   parent's `device`; the child's `for_each` is not re-priced
   (`list_entries` is the parent's). Add its minutes; if the child had no
   cost, or was unreadable, set `partial: true`. A builtin adds nothing and
   sets nothing - it is the parent's to price.
5. `minutes` is rounded to one decimal.

`measured_on` is the entry's `name`, or `null`.

### 1.4 Downloads required

- Collect every `from_pretrained_arguments.model_name` string anywhere in
  `expanded` (pipelines, components, and inside a sub-workflow loaded in
  1.3), plus every `from_pretrained_arguments.from_single_file` value that
  is a URL (`validate_url` accepts it). Deduplicate, preserve first-seen
  order.
- `present = {repo["repo_id"] for repo in scan_models(cache_dir)["repos"]}`.
  A `model_name` in `present` is dropped. A `model_name` that is a local
  directory (`os.path.isdir` after the run's own resolution) is dropped -
  it is not on the hub.
- Each remaining hub name → `{"repo": name, "gb": size}`; each URL →
  `{"repo": null, "url": url, "gb": null}`.
- `size`: when `lookup_sizes` is true, `huggingface_hub.model_info(name,
  files_metadata=True)` under a 5-second timeout, `sum(s.size for s in
  info.siblings if s.size)` in GiB to one decimal. Any exception, a
  missing token for a gated repo, or a missing `size` → `null`. The
  planner never raises here and never logs above `debug` - an offline box
  is a state, not an error.
- `lookup_sizes=False` skips the hub entirely; the API passes it through
  from `?sizes=false` on validate, for a caller that wants the answer fast.

### 1.5 `cached_steps`

Stage 1 answers `null` always. The field exists from the start so the
answer's shape does not change in stage 2.

## 2. `POST /api/validate`

After the existing `answer = {"valid": True, …}` is built:

```python
try:
    answer["plan"] = build_plan(definition, request.arguments, base_dir=…,
        prompt_dir=…, output_root=workspace.outputs, workflow_dir=…,
        device=get_device_type(get_device()), lookup_sizes=request_sizes)
except Exception:
    logger.exception("Plan could not be built")
    answer["plan"] = None
```

- `definition` here is the definition already resolved (from the file or
  the inline body); `base_dir`/`workflow_dir` are exactly what the
  `candidate` was constructed with, so the plan sees the paths the run will.
- The `cost` list comes from the definition itself (`definition.get("cost")`)
  - the same place the listing reads it - so an inline workflow that carries
  a `cost` block is priced too.
- `sizes` is a new optional query parameter, default `true`.
- An invalid answer carries no `plan` key at all.

## 3. MCP

- `dw_mcp/authoring.py: validate_workflow` returns the server's answer
  unchanged; `plan` rides along.
- `dw_mcp/server.py`: the `validate_workflow` docstring gains a paragraph:
  quote `plan.estimate.minutes` with its `basis`, name each
  `downloads_required` entry as a separate line item ("and 41 GB of weights
  this box does not have"), treat `basis: unknown`/`other_device` as "no
  measured figure". `COST_REFUSAL` (`diagnose.py`) says the same in one
  sentence: the number to say out loud is the plan's, not the listing's.
- `get_guide`'s authoring section (WORKFLOW_GUIDE.md) gains the same rule.

## 4. Docs and skills

- `docs/SERVER.md`: the validate answer's `plan` block, field by field.
- `docs/MCP.md`: the quoting rule.
- `docs/WORKFLOW_GUIDE.md` "Authoring a workflow from an agent": the
  quoting rule; `CLAUDE.md`'s type-system list is unchanged (nothing here is
  a reference convention).
- `plugins/dw/skills/{minimax-h3,minimax-music3,ltx-2.5}/SKILL.md`: the
  "quote cost" step reads "validate with the arguments you will run, quote
  `plan.estimate`, and name any `downloads_required`" instead of "quote the
  listing's `cost`". `tests/test_plugin_skills.py` keeps pinning the numbers
  the skills state; the rule change adds no number.

## 5. Tests (stage 1)

`tests/test_plan.py` (pure, no server):

- fingerprint invariants - every cell of the table in 1.2, each as its own
  test, using a fixture workflow with a `for_each` over a list variable, a
  `prompt:` reference into a tmp prompt library, and an `output:…/latest/…`
  reference into a tmp output root with two runs.
- estimate: `unknown`, `catalog`, `per_entry` arithmetic (including the
  floor at 0), `other_device` beating `per_entry`, sub-workflow summing,
  `partial` on a cost-less child, builtin ignored.
- downloads: a stub `scan_models` (monkeypatched) with one of two repos
  present; a local-directory `model_name` dropped; a URL listed; `model_info`
  raising → `gb: null`; `lookup_sizes=False` never calls it.
- `realize_workflow(pin_outputs=False)` leaves `latest` as written.

`tests/test_server.py`:

- `plan` present with the documented keys on a valid answer; absent on an
  invalid one; `null` when `build_plan` raises (monkeypatched); `?sizes=false`
  reaches the planner.

---

# Stage 2: binding, the 409 and the cache probe

## 6. The bound acknowledgement

`JobRequest` and `RerunRequest` gain:

```python
acknowledged_cost: bool | AcknowledgedCost | None = None
```

```python
class AcknowledgedCost(BaseModel):
    fingerprint: str                    # "sha256:…" from plan.fingerprint
    minutes: float | None = None        # plan.estimate.minutes, recorded only
    downloads: list[str] = []           # plan.downloads_required[*].repo, non-null ones
```

The form is classified once, in the route:

| Value | `acknowledged` |
|---|---|
| absent, `null`, `false` | `none` |
| `true` | `boolean` |
| object | `bound` |

The server never refuses for `none` or `boolean`; refusing without an
acknowledgement remains `dw_mcp`'s job, exactly as today.

## 7. The check

For `bound` only, after the existing argument and reference checks and
before `manager.submit`:

1. `current = build_plan(...)` with the same inputs validate would use for
   this request (for rerun: the stored job's definition and arguments, with
   the fresh seed folded when `new_seed` - which cannot change the
   fingerprint, by 1.2).
2. If `build_plan` raises: **409** with detail "the run could not be planned,
   so a bound acknowledgement cannot be checked; acknowledge with `true`
   or validate again" - never a silent pass.
3. If `current["fingerprint"] != acknowledged.fingerprint`: **409**.
4. If any `repo` in `current["downloads_required"]` (non-null) is not in
   `acknowledged.downloads`: **409**. A download that has since vanished
   from the requirement is not a refusal.

The 409 body:

```json
{
  "detail": "The run's shape changed since it was acknowledged: …",
  "reason": "fingerprint" | "downloads" | "unplannable",
  "acknowledged": {"fingerprint": "…", "minutes": 38.0, "downloads": [...]},
  "plan": { …current plan… }
}
```

`detail` is one sentence naming the reason: for `fingerprint`, "the
workflow or its arguments differ from what was validated"; for
`downloads`, the repos now required and not acknowledged. `plan` is the
whole current plan so the agent re-quotes from the body without a second
validate call. FastAPI's `HTTPException(status_code=409, detail=…)` takes
only `detail`; the route raises with `detail` as this whole object, which
FastAPI serializes as `{"detail": {...}}` - the MCP layer and the docs
describe that shape.

## 8. Recording

- `Job` gains `acknowledged: str` (`none`/`boolean`/`bound`) and, when
  bound, the object under `spec["acknowledged_cost"]` so `rerun` carries it
  forward and can re-check it (`RERUN_SPEC_KEYS` gains the key).
- `jobs.sqlite` gains `acknowledged TEXT` by `ALTER TABLE`, the same
  pattern as `run_id`; rows before the column read back as `none`.
- `manager.describe(job)` (so `GET /api/jobs/{id}`, `GET /api/jobs`, MCP
  `get_job`/`list_jobs`) carries `acknowledged`, and `acknowledged_cost`
  when bound.

## 9. MCP

- `run_workflow` / `rerun_job` (`dw_mcp/server.py`) widen `acknowledged_cost`
  to `bool | dict`. `diagnose.py` treats a non-empty dict as acknowledged
  and forwards it verbatim in the body; a dict missing `fingerprint` is a
  `DwApiError` before any request is made.
- A 409 from the server surfaces as `DwApiError` whose message is the
  body's `detail.detail` sentence followed by the new estimate
  (`plan.estimate.minutes`, `basis`) and the new downloads, so a client that
  only sees the message can still re-quote.
- `COST_REFUSAL` teaches the bound form as the normal one: validate with the
  arguments, quote the plan, pass `{"fingerprint": plan.fingerprint,
  "minutes": plan.estimate.minutes, "downloads": [...]}`. Bare `true` is
  the fallback for `plan: null`.

## 10. The step-cache probe

The step cache is the worker's singleton and `Workflow.run` probes it with
a snapshot taken after substitution and reference resolution
(`workflow.py`, the `step_cache.get` call), so an exact answer has to come
from the worker.

- New worker command `{"type": "probe_cache", "definition": <realized,
  expanded>, "seed": int, "output_dir": str}` answered by
  `{"type": "probe_cache", "cached": [step names]}`.
- The worker's handler walks the steps exactly as `Workflow.run` does up to
  and including the `step_cache.get` call, with `hits_this_run` threaded
  through, and executes nothing. This is a refactor of `Workflow.run`: the
  "is this step a hit" computation is lifted into a method
  (`Workflow.cache_hit(step_data, step_seed, hits_this_run)` or a
  module-level helper) that both the run loop and the probe call, so the
  two cannot drift.
- `JobManager.probe_cache(definition, seed, output_dir, timeout=5)` has the
  same shape as `memory_status`: busy worker, no worker or a timeout answer
  `None`, never block a request behind a running job.
- `build_plan(cache_probe=manager.probe_cache)`: when given, `cached_steps`
  is `len(cached)` on an answer and `null` on `None`. The seed handed to the
  probe is the run's real seed (the folded seed variable's value, else
  `definition["seed"]`, else `null` → the workflow is unseeded, the cache is
  off, `cached_steps: 0` with no probe made).
- The estimate is not scaled by `cached_steps` in this stage: the plan
  states both and the agent says "N of M steps are cached". Scaling needs
  per-step timing the catalog does not hold.

## 11. Tests (stage 2)

`tests/test_server.py`:

- 409 `fingerprint` on a longer list than acknowledged; on changed
  arguments; on a changed stored prompt.
- 409 `downloads` on a repo newly required (stub `scan_models`); no 409 when
  a download vanished.
- no 409 on a new seed, on `rerun(new_seed=True)`, on a `latest` that
  advanced.
- `boolean` and `none` paths queue exactly as before; `acknowledged`
  recorded correctly for all three; sqlite migration on a database created
  without the column.
- 409 `unplannable` when `build_plan` raises under a bound acknowledgement.
- 409 body carries `plan`.

`tests/test_worker.py` / `tests/test_step_cache.py`:

- `probe_cache` against a warm cache reports the hit set the next run
  actually reuses (run once, probe, run again, compare `reused` in the
  manifest to the probe's answer); against a cold cache reports none;
  `JobManager.probe_cache` answers `None` when the worker is busy.

`tests/test_mcp*.py`:

- dict forwarded verbatim; dict without `fingerprint` refused client-side;
  409 surfaces as `DwApiError` with the estimate in the message; `true`
  still works.

## Release notes

- New: `POST /api/validate` answers with `plan`; `?sizes=false`.
- New: `acknowledged_cost` accepts `{fingerprint, minutes, downloads}`;
  `POST /api/jobs` and `/rerun` answer 409 when the plan changed. `true`
  unchanged.
- New: jobs record `acknowledged`.
- The skills now quote from the plan rather than the listing.
