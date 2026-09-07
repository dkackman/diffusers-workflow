# Proposal: making the catalog legible to an agent — shape, cost, and where format-knowledge lives

Status: proposed, 2026-09-06. Synthesizes and replaces two prior proposals,
`catalog-shape-index.md` and `mcp-discovery-data.md` (both fully folded into
this document and deleted — see git history for the original framing), plus
the discovery-facing conclusion of
[scripted-dialogue-and-tts.md](scripted-dialogue-and-tts.md), which keeps its
TTS-specific design record but no longer restates the point made here.

## The question

An agent connected to `dw_mcp` is handed a request stated as a *subject*
("a lego movie trailer set in the marvel universe") or a *shape* ("make me
a dialogue scene"). Neither names a workflow. What has to be matched is the
*shape* of the deliverable — a single still, an image set, one shot, a
multi-shot cut sequence, video with generated speech — against a catalog
that says nothing about shape in machine-readable form, and once a
candidate is found, nothing tells the agent what running it costs. Every
surface it reads to close that gap costs tokens on every session, and
everything hand-written drifts out of sync with the workflows it describes.
Three questions, one answer each: what shape does this produce, what will
it cost, and — when neither the catalog nor a guide already answers a
request — where does the missing knowledge belong?

## What already shipped, and what it did not solve

Landed 2026-09-06, closing about half of this:

- **Two trees.** `workflows/templates/` teaches a pattern, `workflows/models/`
  records what makes a checkpoint fit, and a model config names the template
  it configures. 121 workflows became 73, so the handful that answer a
  multi-shot video request are no longer buried under a hundred checkpoint
  iterations.
- **`list_guides` / `get_guide`.** Nine curated docs served over MCP, indexed
  by section, ~865 tokens for the whole index. An agent can look up how a
  shape is expressed instead of guessing.
- **The server instructions** now say to decide the deliverable's shape
  first, then match the catalog against that, then `list_tasks`, then author
  last.

What none of that does is make the shape *itself* readable, or its cost
known. `list_workflows` returns a description written for a human who
already knows the domain, plus `kinds` (image/video/audio), step and
variable counts. "Multi-shot cut sequence" is nowhere in that, and neither
is "this takes forty minutes".

## What it costs today

Measured on the PR 41 branch (chars ÷ 4 ≈ tokens):

| Surface | Size | When it is read |
|---|---|---|
| Server `instructions` | ~780 tokens | every session, unconditionally |
| 54 tool descriptions | ~3,750 tokens | every session, unconditionally |
| `list_guides` | ~865 tokens | once, on demand |
| `list_workflows` (73 entries) | **~11,450 tokens** | the first catalog look — the instructions say to start here |
| `list_tasks` | ~270 tokens (bare names) | on demand |
| `get_task <cmd>` | 350-1,400 tokens | per command inspected |
| a whole guide (`tasks`, `workflows`) | ~13,000 tokens each | should never happen; `get_guide` takes a section |

The standing cost — what every session pays before the first decision — is
~4.5k tokens, lean for 54 tools. The variable cost is one call:
`list_workflows` is 2.5× everything else combined, and 21k of its 46k
characters are `description` strings, five of them (the MiniMax narratives)
at 700-1,400 characters each. Every entry also repeats `origin`, `writable`
and `prompt_refs`, which an agent choosing a template never uses.

## What is working and should not change

- **The guides index as a routing table.** ~865 tokens for nine guides with
  their `##` headings converts "trailer" into "multi-shot + dialogue + cuts"
  without reading 13k tokens. Section fetch with loose heading matching is
  exactly right.
- **`list_tasks` as bare names.** Discovery goes through the `tasks` guide's
  sections; `get_task` gives the real signature. Adding summaries to the
  list would duplicate the guide.
- **The instructions' shape-first rule.** The one sentence that stops an
  agent from authoring a fresh workflow for a request an existing template
  covers. Keep it, and make it executable (below).
- **The two-tree split with `configures`.** Nine `models/` entries collapse
  to "text-to-image with a checkpoint choice" the moment an agent can see
  the field.

## Proposal 1: a closed shape vocabulary, derived, with an override

Compute a small, controlled shape vocabulary from the definition itself, in
[`workflow_details`](../../dw/server/app.py) — which already parses every
workflow and caches per file mtime, so this costs one pass over data it has
in hand and no new I/O:

| shape | what it produces |
| --- | --- |
| `image` | one still, or a set from one prompt |
| `image-set` | several stills the workflow relates (a storyboard, variations, a sweep) |
| `image-edit` | a still derived from a supplied one (inpaint, outpaint, edit, upscale, restore) |
| `shot` | one continuous clip |
| `sequence` | several shots cut or dissolved together |
| `video-with-speech` | a clip whose audio is generated dialogue or narration |
| `audio` | a track by itself (music, speech) |
| `text` | a prompt, caption or description |
| `utility` | a processing step with no generative model (segment, crop, embed metadata) |

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

Derivation cannot go stale and needs no backfill across 73 files — the two
properties that killed hand-authored tags when this was first considered.
Where it gets a workflow wrong, an optional top-level `shape` block
overrides it, declared in the schema beside `configures`. Only the odd ones
need authoring.

`list_workflows` gains an optional `shape` filter (and `kinds`, while at
it). "A trailer" resolves to `sequence` + `video-with-speech`, which pulls
~12 entries rather than 73. The instructions then say "decide the shape,
call `list_workflows(shape=...)`" — one step, no inference.

The risk worth naming: a controlled vocabulary is a commitment. Adding a
value later is easy; changing what an existing value means silently breaks
whatever matched on it. Keep the list short and concrete, and prefer "has a
`concat_videos` step" over judgements like "cinematic". Do not make the
vocabulary open — its whole value is that an agent can enumerate it; a free
string is a description with a shorter name.

Validation: `tests/test_catalog_structure.py` requires every template to
carry a shape from the vocabulary, the way it now requires a description.

## Proposal 2: split `summary` from `description`

An agent picking a template needs, per entry: what shape it produces, what
inputs it needs, what it will cost, and one line saying what it is for. The
listing gives it a paragraph and leaves it to infer all four. Prose gets
reworded and never indexed; five 1,000-character descriptions are five
essays an agent skims and forgets by the time it reads the fortieth.

Split `description` into a required `summary` (one sentence, ≤ 120
characters, what it is for) and the existing free text; the listing carries
the summary only, `get_workflow` returns the full text. Expected effect on
`list_workflows`: 21k description characters → ~6k, the whole listing
~11.4k → ~5.5k tokens.

Together, `summary` and `shape` take the first catalog look from ~11.4k
tokens to under 3k for a filtered call, with no guesswork left in the shape
question.

## Proposal 3: report what a run actually costs

The instructions say "say what it will cost before spending it". The agent
has nothing to say it from except a guide written per model — nothing in
the listing distinguishes a 45-second MiniMax H3 render from `text-to-image`.

A `cost` field on each template: `{"vram_gb": 24, "minutes": 3}` as a
*measured* figure on a named device, filled in by the maintainer from a real
run and stamped with the device it was measured on. Templates without one
show `null`; the UI and the agent treat `null` as "unknown — run
`get_memory` and say so".

A richer version observes runtime from history instead of hand-authoring
it: carry a median and run count per workflow from the `jobs` table, which
already holds `workflow`, `status`, `started_at` and `finished_at`.
Measured beats estimated, needs no authoring, and improves as the box is
used. The wrinkle, confirmed rather than assumed: `jobs.workflow` stores
`Workflow.name`, which is the workflow's **`id`** field
([`dw/workflow.py:216`](../../dw/workflow.py)), not its catalog path. The
listing is keyed by catalog name, so joining the two means mapping id to
catalog name — a mapping that is neither free nor guaranteed unique (two
workflows shared the id `sd35` until `archive/` was emptied, and nothing
enforces uniqueness today). Two fixes, worth doing together: enforce unique
ids with a test (a duplicate id is already a latent step-cache collision
worth failing on for its own sake), and record the catalog name on the job
alongside the id so the join stops depending on ids being unique forever —
old history stays unjoinable, new history is exact.

Start with the hand-authored `{"vram_gb", "minutes"}` field; it is the
minimum that makes "say what it will cost" answerable from the listing
without a job-history join. Move to observed runtime once the join is
worth its complexity.

## Proposal 4: guides served by the engine they describe

`list_guides`/`get_guide` read the MCP *client's* install (`dw/docs/` or the
repo's `docs/`). Their tool descriptions say "the documentation shipped
with this engine", which is not what happens: an MCP at one version against
a remote `dw.serve` at another confidently indexes sections (`Speech
Generation`, `templates/`) the server does not have. This is also what
forced the `build_dist.sh` copy step, the `.gitignore` entry, the
stale-copy trap and the CI-breaking `httpx` import in the build script (all
findings in the PR 41 review).

`GET /api/guides` and `GET /api/guides/{name}?section=` on the server,
reading `docs/` beside it with the same checkout-else-packaged resolution
`default_ui_dir` already does for the SPA. `dw_mcp/guides.py` becomes a
proxy like every other handler; the `GUIDES` table (names, files,
summaries) moves to the server; the guides an agent reads are then the
guides for the engine it is about to drive. The one thing lost is answering
`list_guides` against a server that is down, which is not a real use.

## Proposal 5: catch drift with a check

Nothing checks that a description's argument set (`"prompt", "num_frames"
...`) names variables the workflow actually declares, or that the shape a
description implies matches the `shape` field. Once `summary`, `shape` and
`cost` exist, `tests/test_catalog_structure.py` can hold each to its
workflow: every variable named in backticks in a description exists; every
`prompt:` reference resolves (already done); a `video-with-speech` template
has a step producing audio.

## Proposal 6: let the UI show it

The restructure regressed the web UI before anything improved it — the
folder grouping took only the first path segment, so the new layout
collapsed to two groups and hid the `ltx2` and `minimax` families. That is
fixed, and a model config's card now says which template it configures.
What shape and cost data would additionally buy:

- **Filter by shape** on the workflows page. The existing filter matches
  name and description text; shape values would make "show me multi-shot
  video" a click rather than a guess at vocabulary.
- **Cost on the card**, beside the step and variable counts.
- **Templates before model configs** in the default ordering, since one is
  for reading and the other for reference.

None of this is worth building before the data exists, and all of it is
cheap once it does.

## Principle: format-knowledge belongs in guides and templates, not in new engine code

The catalog and guides answer "which existing workflow fits", but an agent
will sometimes hit a shape the catalog does not cover — the gap is not
always a missing field, sometimes it is missing *authoring knowledge*.
[scripted-dialogue-and-tts.md](scripted-dialogue-and-tts.md) worked through
one: turning a plain dialogue script into H3's Context-IR shot list (voice
descriptions carried verbatim, speaker alternation driving cuts, stripping
the reference portrait's compositional authority) is knowledge that exists
only in that design doc today. It was deliberately *not* built as an
engine task — a generator belongs in authoring, not runtime, and doing it
as a workflow-emitting step keeps the artifact inspectable rather than
hiding prompts until after the GPU has spent time on them.

The general rule this sets for closing catalog gaps: when a shape is
missing, prefer adding a guide section and a template workflow that
demonstrates it over adding an engine feature or an MCP tool. A tool is
justified only when the composition genuinely cannot be expressed as a
workflow (as `zip`-versus-cartesian-product composition could not, in that
case). This keeps the surface an agent has to learn small and keeps new
capability inspectable rather than opaque.

## What not to do

- **Do not add per-tool examples or longer tool descriptions.** The 54
  descriptions average 280 characters and the instructions carry the
  protocol. Examples belong in the guides, one section each, fetched on
  demand.
- **Do not put variable defaults in the listing.** `workflow_details` leaves
  them out on purpose (an order of magnitude more payload); `get_workflow`
  has them.
- **Do not summarise the guides into the instructions.** The index is the
  summary; the instructions should keep pointing at it rather than
  restating it.
- **Do not make the shape vocabulary open.** See Proposal 1.
- **Do not build a recommendation engine.** The point is to make the
  catalog's existing structure legible to something that cannot read prose
  carefully, not to guess what the user wants. If shape matching turns into
  scoring and ranking, it has gone too far — an agent that can see the
  shapes can do its own choosing.

## Order

1. `summary` field and a listing that carries it (one server change, one
   schema field, a sweep over 73 descriptions to write the first sentence).
2. `shape` field, vocabulary in the schema, `list_workflows(shape=)`,
   catalog-structure test.
3. Server-side guides; `dw_mcp/guides.py` becomes a proxy; the build-script
   copy step goes away.
4. `cost` field (hand-authored `{vram_gb, minutes}` first; observed
   job-history runtime once the id/catalog-name join is settled).
5. Description-to-workflow consistency checks.
6. UI: shape filter, cost on the card, templates-first ordering.

Each is independent of the next and each shrinks or de-risks what the agent
reads; 1 and 2 together are the highest-value, lowest-risk pair and can ship
before anything else on this list.
