# Proposal: right-sizing what an agent reads to choose a capability

Status: proposed, 2026-09-06. Follows the review of PR 41, which added
`list_guides`/`get_guide` and the two-tree catalog. Builds on
docs/proposals/catalog-shape-index.md, which proposes the shape index this
document argues for from the token side.

## The question

An agent connected to `dw_mcp` has to turn a request stated as a *subject*
("a lego movie trailer set in the marvel universe") into a template, a set of
arguments, and a cost estimate - without guessing. Everything it can read to
do that costs tokens on every session, and everything hand-written drifts. Is
the metadata over-specified, under-specified, or about right?

## What it costs today

Measured on the PR 41 branch (chars ÷ 4 ≈ tokens):

| Surface | Size | When it is read |
|---|---|---|
| Server `instructions` | ~780 tokens | every session, unconditionally |
| 54 tool descriptions | ~3,750 tokens | every session, unconditionally |
| `list_guides` | ~865 tokens | once, on demand |
| `list_workflows` (73 entries) | **~11,450 tokens** | the first catalog look - the instructions say to start here |
| `list_tasks` | ~270 tokens (bare names) | on demand |
| `get_task <cmd>` | 350-1,400 tokens | per command inspected |
| a whole guide (`tasks`, `workflows`) | ~13,000 tokens each | should never happen; `get_guide` takes a section |

The standing cost - what every session pays before the first decision - is
~4.5k tokens. That is lean for 54 tools, and the instructions are the right
kind of text: a decision *order* (shape first, then catalog, then `list_tasks`,
author last) rather than a description of each tool. The `acknowledged_cost`
contract is stated once and the tools refer back to it.

The variable cost is one call. `list_workflows` is 2.5× everything else
combined, and 21k of its 46k characters are `description` strings, five of
them (the MiniMax narratives) at 700-1,400 characters each. Every entry also
repeats `origin`, `writable` and `prompt_refs`, which an agent choosing a
template never uses.

## What is working and should not change

- **The guides index as a routing table.** ~865 tokens for nine guides with
  their `##` headings is the best-value addition in PR 41. The 18 `tasks`
  headings and 9 `workflows` headings are what convert "trailer" into
  "multi-shot + dialogue + cuts" without reading 13k tokens. Section fetch
  with loose heading matching is exactly right.
- **`list_tasks` as bare names.** Discovery goes through the `tasks` guide's
  sections; `get_task` gives the real signature. Adding summaries to the list
  would duplicate the guide.
- **The instructions' shape-first rule.** This is the one sentence that stops
  an agent from authoring a fresh workflow for a request an existing template
  covers. Keep it, and make it executable (below).
- **The two-tree split with `configures`.** Nine `models/` entries collapse to
  "text-to-image with a checkpoint choice" the moment an agent can see the
  field; that alone removes nine candidates from the first cut.

## Where it costs more than it needs to

### 1. The listing is prose where it should be fields

An agent picking a template needs, per entry: what shape it produces, what
inputs it needs, what it will cost, and one line saying what it is for. The
listing gives it a paragraph and leaves it to infer all four. Prose gets
reworded and never indexed; five 1,000-character descriptions are five essays
an agent skims and forgets by the time it reads the fortieth.

**Proposal.** Split `description` into a required `summary` (one sentence,
≤ 120 characters, what it is for) and the existing free text, and have the
listing carry the summary only; `get_workflow` returns the full text. Expected
effect on `list_workflows`: 21k description characters → ~6k, the whole
listing ~11.4k → ~5.5k tokens. The `summary` field is what a card shows and
what an agent matches on; the long description stays the place where the
argument sets and caveats live.

### 2. Shape is the first question and nothing names it

The instructions say "decide which shape the deliverable is first, then match
the catalog against that". The catalog carries `kinds` (`image`, `video`,
`audio`, `text`) - the output *media type* - which is not the shape. A single
still, an image set, one shot, a multi-shot cut sequence, and video with
generated speech are all `kinds: ["video"]` or `["image"]`.

**Proposal.** A `shape` field on every template, from a closed vocabulary the
schema enumerates:

| shape | what it produces |
|---|---|
| `image` | one still, or a set from one prompt |
| `image-set` | several stills the workflow relates (a storyboard, variations, a sweep) |
| `image-edit` | a still derived from a supplied one (inpaint, outpaint, edit, upscale, restore) |
| `shot` | one continuous clip |
| `sequence` | several shots cut or dissolved together |
| `video-with-speech` | a clip whose audio is generated dialogue or narration |
| `audio` | a track by itself (music, speech) |
| `text` | a prompt, caption or description |
| `utility` | a processing step with no generative model (segment, crop, embed metadata) |

`list_workflows` gains an optional `shape` filter (and `kinds`, while at it).
"A trailer" resolves to `sequence` + `video-with-speech`, which pulls ~12
entries rather than 73. The instructions then say "decide the shape, call
`list_workflows(shape=...)`" - one step, no inference. This is the same index
docs/proposals/catalog-shape-index.md proposes; the token measurement above is
the argument for doing it before anything else on that list.

Validation: `tests/test_catalog_structure.py` requires every template to carry
a shape from the vocabulary, the way it now requires a description.

### 3. Cost is the third question and nothing names it either

The instructions say "say what it will cost before spending it". The agent has
nothing to say it from except a guide (`recipes`) written per model. A
template that runs MiniMax H3 for 45 seconds of video costs a different order
of magnitude from `text-to-image`, and nothing in the listing distinguishes
them.

**Proposal.** A `cost` field on each template: `{"vram_gb": 24, "minutes":
3}` as a *measured* figure on a named device, filled in by the maintainer
from a real run and stamped with the device it was measured on. Templates
without one show `null`; the UI and the agent treat `null` as "unknown - run
`get_memory` and say so". docs/proposals/catalog-shape-index.md calls this the
runtime index; the field is the minimum that makes "say what it will cost"
answerable from the listing.

### 4. Guides are read from the wrong machine

`list_guides`/`get_guide` read the MCP *client's* install (`dw/docs/` or the
repo's `docs/`). Their tool descriptions say "the documentation shipped with
this engine", which is not what happens: an MCP at one version against a
remote `dw.serve` at another confidently indexes sections (`Speech
Generation`, `templates/`) the server does not have. This is also what forced
the `build_dist.sh` copy step, the `.gitignore` entry, the stale-copy trap
and the CI-breaking `httpx` import in the build script (all findings in the
PR 41 review).

**Proposal.** `GET /api/guides` and `GET /api/guides/{name}?section=` on the
server, reading `docs/` beside it with the same checkout-else-packaged
resolution `default_ui_dir` already does for the SPA. `dw_mcp/guides.py`
becomes a proxy like every other handler; the `GUIDES` table (names, files,
summaries) moves to the server; the guides an agent reads are then the guides
for the engine it is about to drive. The one thing lost is answering
`list_guides` against a server that is down, which the review noted is not a
real use.

### 5. Descriptions drift; a check can hold them

Nothing checks that a description's argument set (`"prompt", "num_frames" ...`)
names variables the workflow actually declares, or that the shape a
description implies matches the `shape` field. Once `summary`, `shape` and
`cost` exist, `tests/test_catalog_structure.py` can hold each to its
workflow: every variable named in backticks in a description exists; every
`prompt:` reference resolves (already done); a `video-with-speech` template
has a step producing audio.

## What not to do

- **Do not add per-tool examples or longer tool descriptions.** The 54
  descriptions average 280 characters and the instructions carry the protocol.
  Examples belong in the guides, one section each, fetched on demand.
- **Do not put variable defaults in the listing.** `workflow_details` leaves
  them out on purpose (an order of magnitude more payload); `get_workflow` has
  them.
- **Do not summarise the guides into the instructions.** The index is the
  summary; the instructions should keep pointing at it rather than restating
  it.
- **Do not make the shape vocabulary open.** Its whole value is that an agent
  can enumerate it; a free string is a description with a shorter name.

## Order

1. `summary` field and a listing that carries it (one server change, one
   schema field, a sweep over 73 descriptions to write the first sentence).
2. `shape` field, vocabulary in the schema, `list_workflows(shape=)`,
   catalog-structure test.
3. Server-side guides; `dw_mcp/guides.py` becomes a proxy; the build-script
   copy step goes away.
4. `cost` field, measured on the maintainer's devices, `null` elsewhere.
5. Description-to-workflow consistency checks.

Each is independent of the next and each shrinks or de-risks what the agent
reads; 1 and 2 together take the first catalog look from ~11.4k tokens to
under 3k for a filtered call, with no guesswork left in the shape question.
