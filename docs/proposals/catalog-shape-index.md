# Proposal: indexing the catalog by shape, and by what a run costs

Status: design only, no code changes. Follows the catalog restructure
([spec](../superpowers/specs/2026-09-06-workflow-catalog-restructure-design.md),
[plan](../superpowers/plans/2026-09-06-workflow-catalog-restructure.md)), which
deliberately deferred this piece: classifying a catalog about to be halved is
wasted work, and the derivation gets more accurate once "template or model
config" is a distinction it can read.

## The question

An agent is handed "make a lego movie trailer set in the marvel universe".
Nothing in that names a workflow. What has to be matched is the *shape* of the
deliverable - a multi-shot cut sequence, with a consistent cast and generated
speech - against a catalog that says nothing about shape in machine-readable
form. And once a candidate is found, nothing tells the agent whether running it
costs ninety seconds or forty minutes.

## What already shipped, and what it did not solve

Three things landed on 2026-09-06 and between them they close about half of
this:

- **Two trees.** `workflows/templates/` teaches a pattern, `workflows/models/`
  records what makes a checkpoint fit, and a model config names the template it
  configures. 121 workflows became 73, so the eight that answer a multi-shot
  video request are no longer buried under a hundred checkpoint iterations.
- **`list_guides` / `get_guide`.** Nine curated docs served over MCP, indexed by
  section, ~830 tokens for the whole index. An agent can look up how a shape is
  expressed instead of guessing.
- **The server instructions** now say to decide the deliverable's shape first
  and match the catalog against that.

What none of that does is make the shape *itself* readable. `list_workflows`
returns a description written for a human who already knows the domain, plus
`kinds` (image/video/audio), step and variable counts. "Multi-shot cut
sequence" is nowhere in that, and neither is "this takes forty minutes".

## Proposal 1: derive the shape, allow an override

Compute a small, controlled shape vocabulary from the definition itself, in
[`workflow_details`](../../dw/server/app.py) - which already parses every
workflow and caches per file mtime, so this costs one pass over data it has in
hand and no new I/O.

What is derivable, and from what:

| Shape fact | Read from |
| --- | --- |
| produces image / video / audio / text | result `content_type`s (already computed as `kinds`) |
| single shot vs multi-shot cut sequence | a `concat_videos` / `dissolve_videos` step with more than one generation step feeding it |
| chained / continued generation | a `chain` block, or `last_frame` / `last_segment` / `match_audio` |
| conditioned on a supplied still | an image-to-video pipeline, or an `image` / `last_image` argument |
| conditioned on identity | a `references` list |
| speaks / has a soundtrack | an `audio` output, or a `generate_speech` step |
| needs input media | an `asset:` reference or a `location` variable |
| composes other workflows | a `workflow` step |
| tuned for one checkpoint | `configures` is set |

Derivation cannot go stale and needs no backfill across 73 files - the two
properties that killed hand-authored tags when this was first considered. Where
it gets a workflow wrong, an optional top-level `shape` block overrides it,
declared in the schema beside `configures`. Only the odd ones need authoring.

The risk worth naming: a controlled vocabulary is a commitment. Adding a value
later is easy; changing what an existing value means silently breaks whatever
matched on it. Keep the list short and concrete, and prefer "has a
`concat_videos` step" over judgements like "cinematic".

## Proposal 2: report what a run actually cost

`list_workflows` should carry an observed runtime per workflow - median and run
count - from the `jobs` table, which already holds `workflow`, `status`,
`started_at` and `finished_at`. Measured beats estimated, needs no authoring,
and improves as the box is used. It is the number that stops an agent casually
queueing a forty-minute job, and it is the one thing about cost that no amount
of reading the definition reveals.

**The wrinkle, confirmed rather than assumed:** `jobs.workflow` stores
`Workflow.name`, which is the workflow's **`id`** field
([`dw/workflow.py:216`](../../dw/workflow.py)), not its catalog path. The
listing is keyed by catalog name. So joining the two means mapping id to catalog
name, and that mapping is neither free nor guaranteed unique - two workflows
shared the id `sd35` until `archive/` was emptied, and nothing enforces
uniqueness today.

Three ways out, in the order I would try them:

1. **Enforce unique ids** with a test, then map id to catalog name from the
   details cache. Cheapest, and a duplicate id is already a latent step-cache
   collision worth failing on for its own sake.
2. **Record the catalog name on the job** alongside the id. Correct, but only
   for jobs run after the change - the existing history stays unjoinable.
3. **Join on the run directory's workflow identity**, which is path-derived.
   Accurate for historical jobs too, but couples the listing to the output tree.

(1) and (2) together are probably right: fail on duplicate ids now, and record
the catalog name so the join stops depending on ids being unique forever.

## Proposal 3: let the UI show it

The restructure regressed the web UI before anything improved it - the folder
grouping took only the first path segment, so the new layout collapsed to two
groups and hid the `ltx2` and `minimax` families. That is fixed, and a model
config's card now says which template it configures. What shape and runtime data
would additionally buy:

- **Filter by shape** on the workflows page. The existing filter matches name
  and description text; shape values would make "show me multi-shot video"
  a click rather than a guess at vocabulary.
- **Runtime on the card**, beside the step and variable counts.
- **Templates before model configs** in the default ordering, since one is for
  reading and the other for reference.

None of this is worth building before the data exists, and all of it is cheap
once it does.

## What this is not

Not a recommendation engine. The point is to make the catalog's existing
structure legible to something that cannot read prose carefully, not to guess
what the user wants. If shape matching turns into scoring and ranking, it has
gone too far - an agent that can see the shapes can do its own choosing.

## Suggested next step

Proposal 1 alone is useful and self-contained: it needs no schema migration
beyond an optional block, no job-history join, and no UI. Ship the derived shape
first, see whether an agent handed a vague request actually lands on the right
workflow with it, and let that answer decide whether the runtime join is worth
its complexity.
