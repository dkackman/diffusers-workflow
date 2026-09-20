# Proposal: derive a workflow's cost from this server's own job history

Status: **design only** - written in answer to issue #91 (tester feedback),
which asks for three things and gets the cheapest of them shipped without a
proposal. This document covers the expensive one. Written by the implementer
agent (model `opus`, provider `anthropic`).

## The report

`list_workflows(shape="shot", traits="identity-referenced")` answers
`cost: null` for seven of eight entries, including
`templates/minimax/reference-to-video`, on a box that has run that template
five times at the same size. The consumer is instructed to quote a price
before spending GPU minutes (`run_workflow`'s own description, the MCP
instructions), and for the template its series actually uses, the catalog
says "unknown". The tester has carried a hand-kept cost table across six
cycles, re-deriving it from job history each time it is lost - from
`started_at`/`finished_at` on jobs this server stored.

Five runs of that template, one RTX 3090, 124 frames at 960x544 / 20 steps:
511, 464.9, 478, ~460, ~467 seconds. ~7.8 minutes, spread under 10% - a
tighter figure than the 10.1 the one populated entry claims by hand.

## What shipped without this proposal

`GET /api/workflows` now answers `cost_basis: "curated"`, and
`list_workflows`' description says what that means: a `cost` block is a
figure a maintainer measured once on the devices it names and wrote into the
workflow; nothing derives one from job history; `null` means nobody wrote one
down, not that the run is cheap or that this box has never run it. That is
the tester's option 3, and it stops `null` reading as a data gap.

It does not give anyone a number.

## Why the rest is a proposal and not a fix

`cost` is documented in `dw/workflow_schema.json` as *"Measured runs, one per
device the maintainer measured on. **Never derived**; absent means unknown."*
That sentence is load-bearing: a curated figure is a claim a person stands
behind, on a named card, at a named size. Deriving one changes what the field
*is*, for every consumer that reads it - which is the "new concept consumers
would have to learn" bar. It also decides questions with no obvious answer
(below). So: design, then Don, then code.

## The shape

A second field rather than a second meaning for the first:

```json
"cost": [{"device": "cuda", "name": "RTX 3090", "vram_gb": 24, "minutes": 10.1}],
"observed": {
  "device": "cuda",
  "name": "NVIDIA GeForce RTX 3090",
  "runs": 5,
  "median_minutes": 7.8,
  "p10_minutes": 7.6,
  "p90_minutes": 8.5,
  "since": "2026-09-08T14:02:11Z",
  "comparable": "same-arguments"
}
```

`cost` keeps meaning exactly what it means today. `observed` is this
server's own history, always about *this* box's accelerator, and absent when
there is nothing to report. A consumer quoting a price prefers `observed`
when it is there (it is this machine, measured), falls back to `cost`, and
says "unknown" only when neither exists. `cost_basis` stays, and the listing
gains nothing else.

### Where the numbers come from

`jobs.sqlite` already holds, per job: the workflow name, `started_at`,
`finished_at`, `status`, `run_id`/`run_dir`, and the workspace. The manifest
in the run directory holds the realized workflow - the arguments folded in
and the seed pinned (`dw/realize.py`). The device is the server's, from
`get_server_info`.

So the query is: finished jobs, `status = completed`, for one workflow
identity, on this device, most recent N. Median of
`finished_at - started_at`.

### The four questions that make it a design

1. **What counts as comparable?** A 141-frame run does not inform a
   124-frame estimate, and a different `weights_dtype` is a different model.
   Three options, cheapest first:
   - *Ignore it.* Report the median of every run of that workflow name, with
     `runs` and a spread. Honest if the spread is published, useless for a
     workflow whose variables move cost by 3x.
   - *Bucket by the arguments that move cost.* Requires naming them, which
     is per-workflow knowledge the catalog does not have. A `cost_drivers`
     key in the workflow (`["num_frames", "num_inference_steps"]`) would
     declare them, and the bucket key is those values. This is the honest
     one and it is more schema.
   - *Report the default-arguments runs only.* A run whose arguments equal
     the workflow's variable defaults is comparable to the curated figure by
     construction. Simple, exact, and thin - most real runs pass arguments.

   Recommendation: **bucket by declared drivers**, falling back to
   default-arguments-only when a workflow declares none. It degrades to the
   third option rather than to a wrong number.

2. **Cold vs warm.** The tester's own `text-to-image` figures are 13.6 s and
   6.3 s - the same run, model on disk vs model resident. These are two
   numbers and averaging them produces one that describes neither. The job
   events already distinguish them (a `loading` phase that takes minutes vs
   one that takes none), so the split is available: `median_minutes` warm,
   `cold_minutes` when the run had to load. A consumer quoting a first run
   of the session wants the cold one.

3. **A cached run is not a run.** A seeded workflow whose every step hit the
   step cache finishes in seconds and wrote nothing. Those jobs are already
   flagged (`reused` on every manifest entry) and must be excluded, or the
   median collapses toward zero for exactly the templates that get re-run
   most.

4. **Pruning.** Job history is prunable and a workspace is deletable. The
   figures move when it happens, which is fine (`runs` says how much is
   behind it), but a listing that quietly loses a number people relied on
   should not be surprising - `since` and `runs` are what make it legible.

### Cost of the feature

A per-workflow aggregate over `jobs.sqlite`, cached like `workflow_details`
is (by the jobs table's own high-water mark rather than by mtime), computed
on listing. Reading the realized workflow of each candidate run to bucket by
drivers is the expensive part - one small JSON read per run, bounded by the
N most recent, and only for workflows the listing actually returns. Nothing
in the run path changes. No new storage; `finished_at - started_at` is
already recorded.

Rough size: a new `dw/server/observed_cost.py` (aggregate + cache), a
`cost_drivers` key in the schema, the `observed` field in the compact and
full listings, the MCP description, docs, and tests. Half a day, most of it
the comparability rule.

## What I would not do

Overwrite `cost` with a derived figure, or let `observed` inherit the
`cost` shape closely enough to be mistaken for it. The distinction between
"a maintainer measured this on a 4090" and "this box averaged that last
week" is the whole value of reporting both.

## Recommendation

Ship it as `observed`, bucketed by declared `cost_drivers`, cold/warm split,
cached runs excluded. If that is more than the problem is worth, the
fallback that costs almost nothing is the third comparability option -
default-arguments runs only, with `runs` and `since` published - which would
have answered the tester's question today, because their five runs were all
at one size.

## Open question for Don

Is `cost` allowed to gain a sibling that is derived, or does "never derived"
apply to the whole of what the catalog says about price? If the latter, this
belongs in a separate tool (`get_workflow_history(name)`) rather than in the
listing, and the consumer pays a call for it.
