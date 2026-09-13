# Proposal: bind `acknowledged_cost` to an estimate, not just to a boolean

Status: **implemented** (stage 1 at b37f033, stage 2 on `cost-binding`).
Design:
docs/superpowers/specs/2026-09-12-acknowledged-cost-binding-design.md.
Written in answer to issue #85 (forum feedback),
after reading the gate and every path by which a run's size is decided, by
the implementer agent (model `opus`, provider `anthropic`).

## The question asked

> The acknowledgement should bind to an estimated budget or operation
> fingerprint, not just a boolean, otherwise the plan can change after
> consent while the flag remains true.

## What the gate is today

It is a boolean, checked in the MCP layer only:

- `dw_mcp/diagnose.py:45` - `run_workflow` raises `COST_REFUSAL` unless
  `acknowledged_cost` is truthy; `rerun_job` the same at `:248`.
- `dw_mcp/models.py` / `dw_mcp/workspaces.py` / `dw_mcp/prompts.py` -
  `download_model`, `delete_model`, `update_diffusers`, `delete_workspace`,
  `enhance_prompt` the same.
- `POST /api/jobs` does **not** require it. The gate is an agent-behaviour
  gate - "say the number out loud to a human before you spend it" - not a
  resource guard, and the HTTP API and the web UI queue work without it.
  The one HTTP route that carries an acknowledgement is
  `DELETE /api/workspaces/{name}` (`acknowledged=true`, `app.py:1415`), and
  it does so because the server is the only party that knows what the
  deletion would remove - the same reason the check below has to live
  server-side.

Nothing is bound. The flag records that *something* was consented to, not
what. The figure the agent quoted comes from the catalog's `cost` block
(`list_workflows`), which is measured against the workflow's *stored
defaults*; the run is queued with the caller's `arguments`, which is a
different document.

## Where the plan can grow after consent

Each of these is reachable with `acknowledged_cost=true` and a quote that
was honest when it was made:

1. **List fan-out.** A `for_each` step is expanded per entry
   (`dw/for_each.py`, 32 entries max). `cost.minutes` is the measured cost
   of the *default* list; a caller passing a 12-entry `shots` list through
   `arguments` runs 12 shots. `cost.per_entry` exists precisely to price
   this - and is advisory, unenforced, and set on no template yet.
2. **Cartesian product.** Several `previous_result:` references in one step
   multiply (4 images x 3 masks = 12 iterations), and the multiplicands can
   come from arguments. Only some of them are knowable before the run: a
   multiplicand that is `num_images_per_prompt` is, one that is "however
   many frames the previous step produced" is not.
3. **Plain numeric arguments.** `num_images_per_prompt`, `num_frames`,
   `num_inference_steps` scale the run roughly linearly and are ordinary
   variables.
4. **A model download mid-run.** Weights not on disk are pulled by
   `from_pretrained` when the step reaches it - tens of GB and many minutes,
   appearing in no `cost` block. This is the forum comment's sharpest case:
   the same workflow costs 4 minutes on a warm box and 50 on a cold one.
5. **`inline_workflow`.** No catalog entry, so no `cost` at all - the quote
   is whatever the agent believed.
6. **`rerun_job(new_seed=true)`.** Draws a fresh seed, which defeats the
   step cache; the rerun of a "finished instantly" job is a full generation.
7. **Sub-workflows.** A `composes-workflows` step's real cost is the child's.

And one that goes the other way: a seeded workflow whose steps are in the
step cache costs nothing, which is why "Run again" finishes instantly. An
estimate that ignores the cache over-quotes the common case, and the
`new_seed` rerun in case 6 is exactly the flip from cached to full.

So the premise in #85 holds: the flag stays `true` while the work grows, and
nothing at queue time compares what will run against what was acknowledged.

## What the engine already has

Almost all of the raw material:

- `POST /api/validate` is free, takes the caller's `arguments`, and already
  folds them exactly as the run will (`argument_errors`), expands `for_each`
  and reports per-entry problems.
- `expand_for_each` yields the exact member list a run will execute.
- `realize_workflow` (`dw/realize.py`) already produces the canonical
  realized document - arguments folded, seed pinned, prompts inlined,
  `output:latest` resolved - which is the natural thing to fingerprint.
- `hub_cache.scan_models` knows what is on disk, so "which repos this run
  will have to download first" is answerable before queuing.
- The manifest and `workflow.json` make the run inspectable afterwards,
  which is what makes an estimate checkable rather than decorative.

The missing pieces are an estimate in the pre-flight answer, and a check at
queue time.

## Proposed change

### 1. `POST /api/validate` returns a plan

Beside `valid`/`errors`/`warnings`, on a valid answer:

```json
"plan": {
  "fingerprint": "sha256:9f13…",
  "steps": 14,
  "cached_steps": 0,
  "list_entries": {"shots": 12},
  "downloads_required": [{"repo": "MiniMaxAI/MiniMax-H3", "gb": 41.2}],
  "estimate": {"minutes": 38.0, "basis": "per_entry", "device": "cuda"}
}
```

- `fingerprint` is a SHA-256 over the plan-shaping inputs: the realized
  workflow (`realize.py`'s document) with the seed removed and each
  `output:.../latest/...` reference left *unpinned*, plus the expanded step
  names. It changes when the *work* changes and not when something cosmetic
  does. The seed is excluded because a fresh seed is the same work; `latest`
  is left unpinned because a run finishing between the validate call and
  the queue call would otherwise change the fingerprint of identical work,
  and the human-in-the-loop gap is exactly where that happens. Inlined
  prompt text stays in: a prompt edited in between is a different run.
- No `iterations` field: the Cartesian count is static only when every
  multiplicand is an argument (case 2), and a number that is sometimes a
  guess is worse than none. `list_entries` is always exact.
- `cached_steps` is how many steps the step cache would answer for this
  workflow id and seed - the same check `Workflow.run` makes, made early -
  and the estimate is over the remaining steps only. A `new_seed` rerun
  reports zero.
- `estimate.basis` is one of `per_entry` (fixed + per-entry x N),
  `catalog` (the stored `cost.minutes`, defaults only), or `unknown`
  (an inline workflow with no cost block) - the honesty is in the field,
  not in a fabricated number. `cost` is measured per device, so `device`
  names the entry used; when the catalog has no entry for the accelerator
  that is serving, `basis` is `other_device` and `minutes` is the nearest
  entry's, which is a warning and not a quote. A `composes-workflows` step
  contributes its child's catalog cost.
- `downloads_required` closes case 4 on its own, and is useful with or
  without the rest of this proposal. Which repos are missing is a local
  question (`scan_models` against the `from_pretrained` targets the
  realized workflow names); `gb` needs a hub call
  (`model_info(files_metadata=True)`) and is `null` when the hub is
  unreachable rather than a reason for validate to fail.

### 2. `acknowledged_cost` accepts what was acknowledged

`run_workflow` / `rerun_job` keep taking `true`, and additionally take an
object:

```json
"acknowledged_cost": {"fingerprint": "sha256:9f13…", "minutes": 38.0}
```

`POST /api/jobs` recomputes the plan for the arguments it was actually
given and refuses with **409** when the fingerprint differs or the recomputed
estimate exceeds the acknowledged minutes by more than a tolerance (25%,
settable), naming both figures and what changed. The agent's recovery is to
re-quote to the user - which is the behaviour the gate was for.

The check has to be the server's: `dw_mcp` is an `httpx` client of the
HTTP API (`DwClient`), so the only alternative is for the MCP tool to call
validate and then jobs and compare in between, which is a second round trip
with a race in it. That means `POST /api/jobs` grows an *optional*
`acknowledged_cost` field either way - the object is the only form it acts
on; a bare `true` it merely records. `rerun_job` computes its plan from the
stored job's arguments, with the seed swapped when `new_seed` is set.

### 3. Bare `true` stays legal, and says so

An unbound `true` keeps working - the web UI, the CLI-equivalent callers and
every existing script depend on it, and a hard requirement would be a
breaking change to every MCP consumer. But the job records which kind of
acknowledgement it got (`acknowledged: "none" | "boolean" | "bound"` -
`none` is the web UI and every HTTP caller that sends nothing), so "was this
run consented to at its actual size?" is answerable after the fact, and the
skills can teach the bound form as the normal one.

## Alternatives considered

- **A hard budget ceiling** (`max_minutes`, run aborted when exceeded).
  Needs runtime metering the engine does not have, and killing a 90%-done
  video render to honour an estimate wastes exactly the resource the gate
  protects. The fingerprint check is pre-flight, which is where a refusal is
  free.
- **Requiring the bound form.** Breaking, and buys little over recording
  which form was used.
- **Estimating cost server-side with no acknowledgement change.** Half the
  value (the agent can quote better) with none of the binding - the plan can
  still change between the quote and the call.
- **Doing nothing.** Defensible: the gate is a prompt-discipline device, and
  the human is in the loop by construction. But the forum comment's case -
  "validates cheaply, expands at runtime" - is real on this engine today
  (cases 1, 4 and 6 above), and the plan-at-validate half is cheap and
  useful even alone.

## Scope

- `dw/plan.py` (new): fingerprint + estimate from a definition and
  arguments, reusing `realize_workflow`, `expand_for_each` and `hub_cache`.
- `dw/server/app.py`: `plan` on the validate answer; the 409 check on
  `POST /api/jobs`.
- `dw/server/jobs.py`: record the acknowledgement form on the job and in
  `jobs.sqlite`.
- `dw_mcp/diagnose.py` + `dw_mcp/server.py`: accept the object form, surface
  `plan` from `validate_workflow`.
- Docs: SERVER.md, MCP.md, WORKFLOW_GUIDE.md's authoring section, and the
  three plugin skills' cost step.
- Tests: fingerprint stability against cosmetic edits, against a new seed,
  and against a new `latest` run landing between validate and queue; change
  under a longer list and under an edited stored prompt; `cached_steps`
  against a warm step cache; `downloads_required` with the hub unreachable;
  the 409, the tolerance, and the boolean path unchanged.

Roughly a two-stage piece of work: stage 1 the plan on validate (useful on
its own), stage 2 the binding and the 409.

## Open questions for approval

1. Is the bound form worth it at all, given that a human is already in the
   loop on every acknowledgement? (Doing nothing is a legitimate answer;
   stage 1 alone is another.)
2. Tolerance: 25% of the acknowledged minutes, or a fingerprint-only check
   with no numeric comparison at all? A numeric one needs `per_entry` on the
   templates to be meaningful, and today no template carries it.
3. Should `POST /api/jobs` grow the gate for HTTP callers too, or stay an
   MCP-layer concept? (The web UI would have to send something.)
