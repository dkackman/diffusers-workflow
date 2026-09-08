# Proposal: making dw legible to an agent — discovery, authoring, and model-specific knowledge

Status: Parts 1–2 designed 2026-09-06 — see the
[design spec](../superpowers/specs/2026-09-06-agent-catalog-legibility-design.md)
and the **Ledger** at the end of this document, which records what was
actually done against each proposal. Part 3 is carried as a constraint and Part 4 shipped through both channels (see the Ledger and the plugin drill at the end); no part remains open. Synthesizes and replaces two prior proposals,
`catalog-shape-index.md` and `mcp-discovery-data.md` (both fully folded into
this document and deleted — see git history for the original framing), plus
the discovery-facing conclusion of
[scripted-dialogue-and-tts.md](scripted-dialogue-and-tts.md), which keeps its
TTS-specific design record but no longer restates the point made here.

Phased and directional throughout: each part names a gap and a direction, not
a locked design. Specifics (exact field shapes, tool signatures, plugin
mechanics) are left to the implementation step for each phase.

## The question, stated in full

An agent connected to `dw_mcp` (the primary surface this proposal optimizes
for; the web UI and filesystem access are secondary and should benefit as a
side effect, not drive the design) is handed a request stated as a *subject*
("a lego movie trailer set in the marvel universe") or a *shape* ("make me a
dialogue scene"). Closing that gap turns out to be three nested problems, not
one:

1. **Discovery** — does an existing workflow already produce this shape, and
   what would running it cost? (Part 1, below.)
2. **Authoring** — if nothing matches, can the agent compose a *new* workflow
   from dw's primitives (tasks, types, schema, composition rules) with enough
   confidence to validate and save it rather than guess-and-check against a
   running GPU? (Part 2.)
3. **Driving a specific model** — some workflows aren't hard to structurally
   assemble but require knowledge specific to one model family. MiniMax H3
   wants Context-IR (`subject_definitions` / `summary` / `retention_analysis`
   / `detailed_description` / `overall_soundscape`), verbatim voice
   descriptions repeated per shot, and a `<Picture N>` clause stripping a
   reference image's compositional authority — none of which is inferable
   from dw's own schema or task signatures, because it isn't a dw concept,
   it's an H3 prompting convention. (Part 3.)

These are linked — shape-first matching is the gate that decides whether (2)
and (3) are even needed — but they are not the same legibility problem, and
conflating them risks either bloating the standing MCP surface (the thing
Part 1 is careful to avoid) or scattering model-specific prompting knowledge
into engine code (which is a hard constraint, not a preference — see Part 3).

## Part 1: catalog and discovery legibility

### What already shipped, and what it did not solve

Landed 2026-09-06, closing about half of the discovery problem:

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

### What it costs today

Measured on the PR 41 branch (chars ÷ 4 ≈ tokens):

| Surface | Size | When it is read |
|---|---|---|
| Server `instructions` | ~780 tokens | every session, unconditionally |
| 54 tool descriptions | ~3,750 tokens | every session, unconditionally |
| `list_guides` | ~865 tokens | once, on demand |
| `list_workflows` (73 entries) | **~11,450 tokens** | the first catalog look — the instructions say to start here |
| `list_workflows` (compact) | **~5,550 tokens** (64 entries) | measured 2026-09-07; 5,327 after tasks 1-10, then 19 declared summaries on the MiniMax and LTX-2 templates (+~220 tokens, ceiling raised 5.5k -> 6k) |
| `list_tasks` | ~270 tokens (bare names) | on demand |
| `get_task <cmd>` | 350-1,400 tokens | per command inspected |
| a whole guide (`tasks`, `workflows`) | ~13,000 tokens each | should never happen; `get_guide` takes a section |

The standing cost — what every session pays before the first decision — is
~4.5k tokens, lean for 54 tools. The variable cost is one call:
`list_workflows` is 2.5× everything else combined, and 21k of its 46k
characters are `description` strings, five of them (the MiniMax narratives)
at 700-1,400 characters each. Every entry also repeats `origin`, `writable`
and `prompt_refs`, which an agent choosing a template never uses.

### What is working and should not change

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

### Proposal 1: a closed shape vocabulary, derived, with an override

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

### Proposal 2: split `summary` from `description`

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

### Proposal 3: report what a run actually costs

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

### Proposal 4: guides served by the engine they describe

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

This matters more once Part 3 exists: model-specific guide content is
exactly the kind of thing that goes stale silently if the MCP client and
the server it drives can disagree about what's in `docs/`.

### Proposal 5: catch drift with a check

Nothing checks that a description's argument set (`"prompt", "num_frames"
...`) names variables the workflow actually declares, or that the shape a
description implies matches the `shape` field. Once `summary`, `shape` and
`cost` exist, `tests/test_catalog_structure.py` can hold each to its
workflow: every variable named in backticks in a description exists; every
`prompt:` reference resolves (already done); a `video-with-speech` template
has a step producing audio.

### Proposal 6: let the UI show it

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

## Part 2: authoring legibility

Part 1 answers "does an existing workflow fit". When nothing does, an agent
needs to compose a new one — and a survey of the current MCP surface
(2026-09-06) found most of the needed primitives already exist, just not
stitched into a stated path, with three concrete gaps left open.

### What already exists

- **`validate_workflow`** ([`dw_mcp/authoring.py`](../../dw_mcp/authoring.py))
  → `POST /api/validate` → `Workflow.validate()`: schema check plus a
  pipeline/task-argument signature check, no GPU or model load. This is
  already the cheap "did I get this right" loop — there is no separate
  dry-run concept to add.
- **`get_schema`** returns the raw `dw/workflow_schema.json` over MCP, not
  just for local validation.
- **`list_tasks` / `get_task`** return real signatures pulled from the
  implementation functions themselves (name, required/default, annotation,
  docstring) — cannot drift from runtime the way hand-written docs can.
- **`save_workflow` / `delete_workflow`** wrap `PUT`/`DELETE
  /api/workflows/{name}`, validating on the way in and shadowing read-only
  sources into the writable directory per [`dw/workflow_sources.py`](../../dw/workflow_sources.py).
- **`docs/WORKFLOW_GUIDE.md`**, served via `get_guide("workflows")`, is
  already the "how to author from scratch" guide, distinct from the
  model/task guides.

### Proposal 7: put the reference-prefix conventions where an agent can read them

`asset:` / `output:` / `prompt:` / `previous_result:` / `constant:`
resolution, `_type`/`_dtype` conversion, and `{}` string-escaping are stated
plainly in `CLAUDE.md` — which an MCP-connected agent cannot read, since
it's a Claude Code project file, not served content. `WORKFLOW_GUIDE.md`
covers most of the same ground in prose spread across several sections.
Consolidate these into one authoring-guide section written for an agent
composing a draft (not a human onboarding to the codebase), so this becomes
reachable at `get_guide("workflows", section=...)` rather than requiring
filesystem access to `CLAUDE.md`.

### Proposal 8: multi-error validation

`Workflow.validate()` surfaces only the first schema violation
([`dw/schema.py`](../../dw/schema.py)) plus step-level argument-name
warnings. An agent iterating on a draft gets one correction per round trip.
Collect all schema violations (jsonschema supports this via
`iter_errors`) and return them as a list; keep the JSON-path prefix per
error so each is locatable.

### Proposal 9: state the composition rules, not just the schema

`previous_results.py` deliberately does a cartesian product across multiple
`previous_result` references — by design, not limitation — but nothing in
the authoring guide says so. An agent composing a multi-input step has no
way to know that a zip-shaped need (shot *i* paired with speaker *i*) isn't
expressible this way without hitting the combinatorial-explosion gotcha
first and reasoning backward. State the rule in `WORKFLOW_GUIDE.md`
directly: this is the generalized form of the reasoning
[scripted-dialogue-and-tts.md](scripted-dialogue-and-tts.md) worked out for
one case, made discoverable instead of living only in a design doc.

### Proposal 10: close the loop at save time

An agent-authored workflow saved via `save_workflow` gets no `shape`,
`summary`, or `cost` — it's invisible to Part 1's matching until a human
backfills those fields. Either require them as arguments to `save_workflow`
(shape can often be derived per Proposal 1's rules even for a freshly
authored file) or have the server compute what it can and flag what it
can't (cost, always — nothing has run yet) as `null` pending a first
measured run. The point: an agent's authored output should make the next
agent's discovery easier, not degrade it.

## Part 3: two knowledge audiences, two budgets

Everything above is generic to dw: schema, task signatures, composition
semantics, the reference-prefix conventions — true of every workflow
regardless of which model it drives. A second, distinct kind of knowledge
exists *per model family* and doesn't fit the same channel.

MiniMax H3 is the concrete case: it wants prompts shaped as Context-IR
(`subject_definitions` / `summary` / `retention_analysis` /
`detailed_description` / `overall_soundscape` / `non_diegetic_music`),
voice descriptions repeated verbatim across shots for consistency, a
`<Picture N>` clause in `retention_analysis` that explicitly strips a
reference image's compositional authority (without it, every shot inherits
the portrait's framing and cuts read as jump cuts), and frame counts
constrained to `17n + 5`. None of this is a dw concept — the schema has no
opinion on it, `get_task` can't surface it because it isn't a task
argument, and it will not generalize to the next model family added.
LTX-2's distilled-sigma constants and its own prompting idiosyncrasies are
a different pile of the same kind of knowledge; every complex model adds
another.

**The hard constraint, restated and widened:** this knowledge must not
become engine code. Not just "don't add a `generate_dialogue_shots` task"
(the conclusion already reached for scripted dialogue) — no per-model
Python formatting functions, no hardcoded prompt templates in `dw/`, full
stop. It stays as data: guide prose and example/template workflows, the
same principle Part 1 already states for closing catalog gaps generally,
now covering prompt-authoring convention as well as workflow shape.

**Why it's a different budget than Part 1's guides.** The existing guide
system is engine-native and MCP-reachable by any client, which is exactly
right for dw-generic knowledge everyone needs. But it is a standing,
enumerated index (`list_guides`) — every model family that gets a guide
section adds to what every session's routing table lists, even for agents
that will never touch that model. H3-specific Context-IR knowledge is
large (the design doc alone runs to hundreds of lines) and only relevant
the moment an agent is actually driving H3. Loading it by default, or even
indexing it by default, works against the economy Part 1 fought to
establish.

## Part 4: packaging model-specific knowledge — resolved 2026-09-07 in favour of Channel B

Given Part 3's constraint (data, not code) and its budget concern
(don't bloat the standing MCP index), there appear to be two non-exclusive
channels, and this proposal deliberately does not choose between them —
that choice belongs to a follow-up design pass once there's a second model
family's worth of this knowledge to generalize from, not just H3's.

**Channel A: more guide sections in dw's own `docs/`.** Consistent with
what already exists (Part 1's guide system, Proposal 4's server-side
guides). Reachable by any MCP client, not just Claude Code. Cost is borne
per-guide only when fetched, same as today — the standing index cost is
one line per guide in `list_guides`, not the guide's full content.

**Channel B: a Claude Code Skill or plugin.** Since the primary harness
this proposal optimizes for is Claude Code, model-specific composition
knowledge could instead (or also) ship as a Skill — triggered contextually
by intent ("the user wants a dialogue scene" → load the H3-authoring
skill), rather than enumerated in a standing index at all. This is a
genuinely different economy: a Skill's cost is paid only in the sessions
that trigger it, and doesn't touch the MCP token budget or the guide index
the way even a lazily-fetched guide section does. It also generalizes
naturally to *not* being dw-specific — a skill could encode "how to
structure prompts for MiniMax H3" independent of which tool ultimately
runs them.

**The tension worth naming now, not resolving:** if the same knowledge
exists as both a guide (for non-Claude-Code MCP clients) and a Skill (for
the primary harness), that's two copies to keep in sync — the same
staleness risk Proposal 4 already fixes for guides-versus-server, one
level up. The direction most consistent with "stays as data, single
source of truth" is a *thin* Skill that mostly points at and quotes the
guide content already living in dw's `docs/` (fetched live via
`get_guide`/`get_workflow` when the harness has MCP access, bundled as a
fallback copy otherwise) rather than a Skill that duplicates the
authoring knowledge independently. Whether that's practical, and whether
it should ship as a plugin bundled with `dw_mcp` or live separately, is
exactly the kind of specific-mechanism question this proposal defers.

**Resolved (2026-09-07, shipped 2026-09-08).** Both channels are in use,
asymmetrically: dw-generic authoring knowledge went to the guides (Channel
A, the `WORKFLOW_GUIDE.md` authoring section), and model-family knowledge
ships as the dw plugin's per-family skills (Channel B: `plugins/dw/skills/`
for MiniMax H3, MiniMax Music 3 and LTX-2.5), while `dw/server/guides.py`
deliberately indexes no model family. The two-copies risk named above is
answered by keeping the skills thin - they defer prompt format to the
vendors' own text and quote catalog names rather than catalog content - and
by `tests/test_plugin_skills.py`, which pins every number a skill states to
the diffusers module that enforces it. The paragraphs above stay as the
design record; the "Plugin drill" section below is the acceptance test.

## Principle: format-knowledge belongs in guides and templates, not in new engine code

The catalog and guides answer "which existing workflow fits", and Part 2's
primitives answer "how do I compose a new one" — but an agent will
sometimes hit a shape or a model-specific convention that neither covers.
[scripted-dialogue-and-tts.md](scripted-dialogue-and-tts.md) worked through
one: turning a plain dialogue script into H3's Context-IR shot list (voice
descriptions carried verbatim, speaker alternation driving cuts, stripping
the reference portrait's compositional authority) is knowledge that exists
only in that design doc today. It was deliberately *not* built as an
engine task — a generator belongs in authoring, not runtime, and doing it
as a workflow-emitting step keeps the artifact inspectable rather than
hiding prompts until after the GPU has spent time on them.

The general rule this sets for closing catalog gaps: when a shape or a
model-specific convention is missing, prefer adding a guide section and a
template workflow that demonstrates it — or, per Part 4, a Skill that
teaches the same thing to the primary harness — over adding an engine
feature or an MCP tool. A tool is justified only when the composition
genuinely cannot be expressed as a workflow (as `zip`-versus-cartesian-
product composition could not, in that case). This keeps the surface an
agent has to learn small and keeps new capability inspectable rather than
opaque, regardless of which channel (Part 1's guides, Part 4's skills)
delivers it.

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
- **Do not encode model-specific prompting knowledge in Python, anywhere.**
  Not a formatting function, not a template string constant, not a
  per-model branch in a task. It stays as guide prose, template workflows,
  or Skill content — data a maintainer edits, not code a maintainer ships.

## Order

1. `summary` field and a listing that carries it (one server change, one
   schema field, a sweep over 73 descriptions to write the first sentence).
2. `shape` field, vocabulary in the schema, `list_workflows(shape=)`,
   catalog-structure test.
3. Server-side guides; `dw_mcp/guides.py` becomes a proxy; the build-script
   copy step goes away. (Prerequisite for Part 3/4 — model-specific guide
   content shouldn't inherit the version-skew problem this fixes.)
4. `cost` field (hand-authored `{vram_gb, minutes}` first; observed
   job-history runtime once the id/catalog-name join is settled).
5. Description-to-workflow consistency checks.
6. UI: shape filter, cost on the card, templates-first ordering.
7. Authoring-guide consolidation (Proposal 7) and multi-error validation
   (Proposal 8) — both are self-contained and can land any time after (3).
8. Composition-rules documentation (Proposal 9) — a doc-only change,
   no dependency.
9. Save-time metadata capture (Proposal 10) — depends on (1) and (2)
   existing to require/derive against.
10. Model-specific knowledge packaging (Parts 3-4) — deliberately last and
    deliberately open-ended: prototype with MiniMax H3 as the worked case,
    in whichever channel (guide section, Skill, or both) proves cheapest
    in practice, before generalizing a mechanism to the next model family.

Each phase is independent of the next and each shrinks or de-risks what the
agent reads or must infer; 1 and 2 together are the highest-value,
lowest-risk pair and can ship before anything else on this list.

## Ledger

What was actually done against each proposal, kept here so a future phase
starts from the record rather than the intent. Updated as work lands.

| proposal | status | where | notes |
|---|---|---|---|
| 1 shape vocabulary | done (tasks 1–4) | spec §1.1–1.3 | became one `shape` value plus boolean `traits`; `video-with-speech` is the `has-audio` trait; `chain` yields `shot` + `chained`, not `sequence`; MCP tool and instructions updated (task 5); three rules changed against the real catalog (task 8): a concat/dissolve of two or more sources is a `sequence` ahead of the utility fallthrough, `urls` counts as a media argument, and a video pipeline with a `vocoder`/`audio_vae` component carries the audio trait; trait `speech` renamed `has-audio` (final review), since it fires on any generated audio track rather than on dialogue |
| 2 `summary` | done (tasks 1, 3) | spec §1.1 | derived from `description`'s first sentence, declared override, ≤ 120 chars; budget test (task 10); 19 MiniMax and LTX-2 templates declare a `summary` (2026-09-07) because their first sentences named a technique, not what the workflow makes - see the cold-session probe below |
| 3 `cost` | done (tasks 7, 9); first entries authored 2026-09-07 | spec §1.1, §1.7 | per-device list, hand-authored; `workflow_name` on jobs landed (task 7); shape validated by task 9's `test_a_declared_cost_is_well_formed`; eight entries from lem's job history (RTX 3090): the two Flux dev configs, z-image, ltx2/text-to-video, minimax/music, both upscalers and text-to-image. Convention: `minutes` is a warm run (model already loaded), one decimal rounded up; `vram_gb` is the card it was measured to fit on, since no run records peak VRAM; a cold first load that is much longer goes in the description as a sentence. Observed runtime from the jobs table stays a follow-up |
| 4 server-side guides | done (plan 2, tasks 1–3) | spec §2.1 | `dw/server/guides.py` owns `GUIDES`; `GET /api/guides`, `GET /api/guides/{name}?section=`; `dw_mcp/guides.py` is two proxy calls; payload keys unchanged (`guides`, `content`) so an agent's contract did not move; `dw/docs/` is a `dw.server` build product now in `MANIFEST.in`; supersedes `guides.py`'s "works with the server down" rationale |
| 5 drift checks | done (tasks 8, 9) | spec §1.8 | shape and summary invariants over the real catalog (task 8); unique ids, cost shape, and description-drift checks over the real catalog found no drift (task 9); drift test uses the catalog's single-quote convention (final review) - the backtick pattern had matched nothing, and the six real mentions it then surfaced are sub-workflow arguments, chain fields and result fields, carried in an allowlist |
| 6 UI | done (plan 3, 2026-09-07) | spec plan 3 | shape select and trait chips (AND) beside the text filter, client-side over the listing the page holds, options limited to values the listing has; card shows `summary` over `description`, shape and traits as badges beside the kind icons, `cost` as `~1.4 min · 24 GB (RTX 3090)` beside the counts (first entry shown; picking the entry for the running device needs server info the page does not fetch); `templates/` groups before other folders before `models/`, root first |
| 7 authoring guide | done (plan 2, task 6) | spec §2.3 | `## Authoring a workflow from an agent` in `WORKFLOW_GUIDE.md`, reachable as one `get_guide` section; `CLAUDE.md` points at it; a test checks every reserved prefix is explained there |
| 8 multi-error validation | done (plan 2, tasks 4–5) | spec §2.2 | `validate_data_all` over `iter_errors`, each reduced with `best_match` so the one-error case reads exactly as before; sorted, deduplicated, capped at 25; `Workflow.validate()` joins one per line under one `Validation errors (N):` heading (the CLI/REPL count the prefix once); `/api/validate` adds `errors` beside the existing `error` — the spec's `message` name was not used because `error` is what every client already reads |
| 9 composition rules | done (plan 2, task 6) | spec §2.3 | the cartesian rule and the one-step-per-pair form are stated in the authoring section, the general case of `scripted-dialogue-and-tts.md`'s reasoning |
| 10 save-time metadata | done (task 6) | spec §1.6 | derived and returned on save; empty summary warns, never rejects |
| Part 3 constraint | carried | spec "Principle" | derivation reads structure, never model family |
| Part 4 packaging | done: Channel A for the generic item (2026-09-07), Channel B for model families (2026-09-08) | `WORKFLOW_GUIDE.md` authoring section; `plugins/dw/`, `.claude-plugin/marketplace.json`, `.claude/skills/model-family-onboarding` | the spoon-set drill (cold session, PR #48 merged) planned a shared seed for "identical but for colour", which draws a different object per prompt; `### Keeping a set consistent` now says which of style, object or character consistency wants prompts, an edit pass or a reference, and `templates/consistent-set.json` (declared `image-set`: derivation says `image-edit`) is the generate-then-edit shape on FLUX.1 Kontext, whose one pipeline both draws and edits; measured on lem 2026-09-07 at 11.5 warm minutes (2.2 for the base, 3.1 per edit, sequential offload) and the four mugs came out identical but for colour. A FLUX.2 draft found that the HF remote text encoder every FLUX.2 entry names now answers with an HTML page (broken on lem since 2026-08-26); `remote.py` now says so instead of raising an unpickling error, and both FLUX.2 entries load the repo's own 4-bit text encoder under model offload instead. The FLUX.2 text encoder is declared as its own component pinned to the CPU, because BitsAndBytes materializes on the accelerator and the two 4-bit models do not fit at load; the `flash_hub` backend went too, since it needs the `kernels` package no install has. models/flux2-dev measured at 3.3 warm minutes on lem. The Krea edit template moved to FLUX.1 Kontext for the same reason, and a catalog test now refuses `trust_remote_code` and `custom_pipeline` in any bundled entry or packaged builtin, so nothing dw ships asks an operator to lower `--trust-workflows`. The Florence-2 and Phi-3.5 builtins went with it: the `text_generation` task had replaced both, and `describe-and-regenerate` now composes it twice. `### Remote code is refused by default` came out of the same run: the first draft used the Krea edit template's `custom_pipeline`, which a server without `--trust-workflows` refuses at load. Superseded by plugin tasks 1-5 below: `templates/minimax/README.md` and `ltx2/README.md` are the sources the family skills derive from and link back to, so that knowledge is reached through the plugin rather than a third channel. Catalog repair task 1: the latent handoff a two-stage flow needs is proven by test (tests/test_result.py::TestLatentHandoff), no engine change. Catalog repair task 2: two-stage.json is the three-move flow (8 sigmas at 768x448, 2x latent upsample, renoise + 3 stage-two sigmas at 1536x896, audio latents carried); latents pass by name; a test holds noise_scale to STAGE_2_DISTILLED_SIGMA_VALUES[0]. Cost and the sharpness comparison await the lem run (task 7). Catalog repair task 3: the six prompts/ltx2 captions are rewritten to the trained format (one paragraph, 150-220 words, shot type/camera motion/viewpoint in prose, sound interleaved; the two I2V ones describe only what changes), intended_model ltx-2.5, four summaries say LTX-2.5; tests/test_ltx_prompt_library.py holds the shape. Catalog repair task 4: h3_context_ir names its two source guides, writes N/A for silent audio fields, numbers <Video N>/<Audio N> within their category, labels continuity modes as this engine's chaining convention (nothing in the engine emits the phrase; it is a user-message convention), adds the ref guide's dialogue-fidelity rules, drops the two unsourced lines; tests/test_h3_context_ir.py. Catalog repair task 5: both template READMEs link the files that exist, name the vendor sources and the audit, the H3 one states the canvas rules and the 5-second diffusers floor; a test resolves every README link. Lem found templates/minimax/enhance-prompt failing on master already: both enhance templates passed a stored copy of the Context-IR system prompt (prompts/prompt_enhancement/minimax_h3.json, 10096 chars) as a sub-workflow argument, which exceeded MAX_VARIABLE_VALUE_LENGTH 10000; the limit is now 20000, the stored copy is gone, and the templates run the builtin's own prompt, with a test refusing a future override. Three wordings of the N/A rule were drafted before the final review found the templates were running the stored copy, so none of them had been tested; the third, which puts the rule on each audio field's own line, is what shipped.  Catalog repair task 7, verified on lem 2026-09-08: two-stage runs at 1536x896, 8.2 warm minutes (base 1.7, refine 2.7, writing the full-size clip 3.5), latents handed by name and the refine pass served from the pipeline cache; at seed 42 the refined frame is sharp where the upsample-only frame is a soft blur. The fox and hummingbird captions ran as written (tracking shot and pull-out; hold and close-up), the hummingbird soundtrack near-silent. The silent-candle brief's three runs went through the stale stored prompt (final review), so the N/A rule's effect on the Qwen3-4B enhancer is re-verified below. Re-verified 2026-09-08 with the templates on the builtin's prompt: the framed idea the template documents (Task, Duration, Idea) gives a full integrated_multimodal_description with both audio fields N/A; a bare idea first collapsed to the two N/A lines alone, so the description field is now declared unconditional and a bare idea also gets its description (jobs 28f1d29a5b90, be94ed994ecd). The N/A rule took four wordings in all, and only the last two were ever tested. Two engine findings on the way: the step cache keys on the resolved step definition, which names a builtin by path, so editing a builtin does not invalidate a cached step (follow-up); and a cancelled H3 job runs to its next step boundary, minutes on this model. Plugin task 1: a marketplace at the repo root and the dw plugin under plugins/dw, installable with two commands the README shows; plugin.json's version is the engine's, bumped by release.sh and held by test. Plugin task 2: the minimax-h3 skill - shape decision over the family's templates, the frame and canvas rules pinned by test to the diffusers modular pipeline, prompts deferred to MiniMax's h3-prompt-writing skill and the two guides, the run-and-judge loop; the README points at it. Plugin task 3: the ltx-2.5 skill - shape decision, the schedule and size rules pinned to the LTX-2 pipeline, the trained caption spec quoted and held equal to the diffusers constant by test; the README points at it. Final review of the plugin branch: the H3 skill's LoRA coupling scoped to the text- and frame-conditioned templates (the six reference ones run no LoRA at 20 steps), both judge steps rewritten because no MCP tool shows a video frame (the agent hands over the gallery url and checks metadata), the fps rule marked as the DFR path's, tighter numeric pins. Plugin task 4: cold drill 2026-09-08 - pass; the skill fired on the bare prompt, chose templates/minimax/storyboard, wrote a Ref2VA Context-IR prompt, quoted cost and asked before an 11.6-minute run, and inspected the result, where the control ran unasked and could not download the file; the drill caught the skill overstating 20 steps for four LoRA-bearing reference templates (fixed, pinned) and gave storyboard its measured cost (10.1 warm minutes); the mould for the next family is copy a skill, follow its six sections, add the family's rules to tests/test_plugin_skills.py, cite the vendor, and run this drill (.claude/skills/model-family-onboarding). Plugin task 5 (2026-09-08, the 30-second follow-up to the drill): the H3 skill states that nothing carries between generations except a reference, gives the chain-or-cut rule, and says how a cuts piece gets one score and one voice; seven bundled prompts wrote a silent music field as None. and now write N/A, held by test. MiniMax Music 3 went through the lifecycle the same night: audited (docs/proposals/audits/2026-09-08-minimax-music3-audit.md; the ceiling rule and the 44.1 kHz result were confirmed against three primary sources, four small corrections landed: music.json's stale trim-task path, music-video's 21-second ceiling over 20.7 seconds of slices, the README's invented guillotine mechanism, and a text_encoder component the quantization docs showed that Music3 does not have), and plugins/dw/skills/minimax-music3/SKILL.md written with its caps, window, guider and output rate pinned to the diffusers modular pipeline and the caption format deferred to MiniMax's music-caption-rewriter skill, which the repo had never mentioned. Its cold drill is still to run. |


### Cold-session probe, 2026-09-07

A fresh Claude Code session (empty directory, no `CLAUDE.md`, no memory) was
asked for "a short multi-shot video with cuts between the shots" against a
server on this branch. What the access log showed, in order:
`list_workflows(shape=sequence)`, `list_workflows(shape=shot)`,
`get_workflow` on `templates/assemble-and-score` and
`templates/ltx2/text-to-video`, `list_assets`. It never fetched the
unfiltered catalog. It shipped three LTX-2 shots cut with assemble-and-score:
two visually related, the third an unrelated nature scene.

What that says about Part 1 as built, and what Part 2 has to carry:

- **The shape-first entry works cold.** Discovery cost two compact filtered
  listings and two definitions.
- **Without `cost`, an agent picks the shortest path, not the best one.**
  Every `cost` is null, so nothing distinguished `ltx2/text-to-video` (no
  input media, "on a single 24GB card") from the MiniMax H3 keyframe route
  (`chained`, `image-conditioned`, `needs-input-media` - a longer chain with
  invisible prerequisites). Authoring `cost` on the shot baselines is the
  first lever; a "choosing a video model" guide (proposal 4) is the second.
- **Traits say what a workflow needs, not when you want it.** The listing
  carried `identity-referenced` and `image-conditioned`, but nothing said
  that cuts between shots imply continuity, so the agent sampled each shot
  fresh. A guide on keeping shots consistent (generate the subject once and
  reference it, or pin keyframes) is a proposal 4 item, indexed under
  `sequence`.
- **Several MiniMax summaries are weak as first sentences** ("The stronger
  form of chain continuity.", "Conditioning on the end of the clip alone.")
  because the fun-chapter descriptions read as prose. A declared `summary`
  on those puts them level with the LTX-2 entries. Cheap, do it before Part 2.
- **A rich workspace `CLAUDE.md` pre-empts discovery entirely.** The same
  prompt in a workspace with a film playbook produced a full H3 plan without
  a single catalog call. The compact listing serves agents starting cold;
  Part 4's packaging question includes how a playbook and the catalog share
  the knowledge rather than compete for it.

### Plugin drill, 2026-09-08

The Part 4 acceptance test: a fresh Claude Code session in an empty directory
(`~/testing/7`, no `CLAUDE.md`, no memory, only the dw MCP server for lem on
branch `dw-plugin`), the dw plugin installed from this checkout, asked "a short
multi-shot video with cuts between the shots"; then the same prompt in the same
directory with the plugin uninstalled, as the control. Transcripts:
`~/.claude/projects/-Users-don-testing-7/24d8bc9d-7c01-4320-aa5d-a8acace418ef.jsonl`
(plugin) and `...-testing-7/bb52715b-c3ea-4227-97fd-88dee384af42.jsonl`
(control). Lem's `dw.log` is the engine log, not the access log, so the call
order below is the transcripts'.

**Plugin session.** The `dw:minimax-h3` skill fired on the bare prompt, before
any tool call. It asked which cut style (storyboard, dialogue short, music
video), then the subject ("a lighthouse keeper's last night before the light is
automated"). Then, in order: `get_server_info`, `list_workflows(shape=sequence)`,
`list_workflows(shape=shot)`, `get_workflow(templates/minimax/storyboard)`,
`validate_workflow`, a plan with 192 frames (17x11+5), 960x544, cuts at 2.7 and
5.4 seconds, and the skill's fallback cost wording because the template carried
no `cost` ("not measured for this template; a few minutes for a turbo-length
clip, reference-conditioned roughly doubles that"). It asked before running.
After `run_workflow` and eleven `wait_for_job` polls it read `get_job_events`
to see which step it was on, then `list_gallery` and `get_gallery_metadata` on
the mp4, handed over the gallery url with the manifest path, and named the
family's failure modes to look for (face drift across cuts, storyboard anchors
overriding the framing). Job 9f75733cfb3c, 11.6 minutes with both models
loading from disk.

**Control session.** No shape question; it asked for the subject with an open
question. Then `get_server_info`, the same two shape listings, `list_guides`
(never followed by a `get_guide`), the same `get_workflow`, `validate_workflow`,
`get_memory`, and `run_workflow` without asking, with "similar template ran a
few minutes" as its whole cost statement. It passed the template's own
`image_reference_type` default back as an argument, harmless but a variable it
did not understand. When the job finished it called `download_output` with a
path on this machine, which lem cannot write (`PermissionError` on `/private`,
surfaced as an opaque `Error executing tool`); it retried twice, once with no
destination, which saved the mp4 into lem's working directory; then it fetched
the three stills inline with `get_output_image` and reported. Job
513205c86652, 10.1 warm minutes.

**The prompts.** Both sessions wrote a correct six-section Ref2VA Context-IR
prompt (subject definitions, tagged summary, per-reference retention analysis,
timed shots, soundscape, continuous score) and they are near-identical in
structure. Neither took it from the skill's vendor pointer: the storyboard
template's default `prompt` is itself a full Context-IR example, and both
patterned on it. On a template that carries an example, the vendor format
reaches a cold agent through the catalog with or without the plugin.

**Verdict: pass, with the skill's contribution measured honestly.** The skill
fired unprompted, the right template was chosen, prompts were in the vendor's
format, the cost was stated before the run, and the output was inspected. What
the plugin changed against the control was the shape choice put to the user,
asking before spending eleven GPU minutes, the honest cost wording, and the
family-specific judging brief at the end. What it did not change was
discovery (both sessions found the template in the same three calls) or the
prompt format on this template. Two defects it surfaced: the skill said the
reference-conditioned templates run 20 steps without a LoRA, but four of
them (`storyboard`, `dialogue-short`, `music-video`,
`chain-matched-and-aligned`) keep the turbo LoRA at nine, and the plugin
session repeated the wrong number; fixed, with the four now named and pinned
by test. And `storyboard` had no `cost`, so both sessions guessed; it now
carries the control run's 10.1 warm minutes. The `download_output` failure is
fixed on the branch; see the engine note under the follow-ups.

### Model-knowledge follow-ups, 2026-09-07

What the two vendor audits found that the catalog does not carry, dated by the
source, and deliberately not done in the repair pass. Each is a template or a
guide sentence when it lands, never engine code.

LTX-2.5 ([audit](audits/2026-09-07-ltx-2.5-audit.md)):

- DFR pipeline as the production-quality path (LTX-2 1.2.0, 2026-08-11; refined
  1.3.0, 2026-08-25): `LTX2DFRPipeline` ships in diffusers, unused; needs
  64-divisible dimensions.
- Temporal upscaling to 48/96 fps (2026-08-11), and the RoPE fps trap: condition
  at 60 for high frame rates, never 120.
- Generated keyframe slots for fast motion (2026-08-11).
- Image conditioning is re-compressed at CRF 18 and needs a PIL image; undocumented.
- Keyframe strength below 1.0 for smooth interpolation (`ANCHOR_KEYFRAME_STRENGTH`).
- IC-LoRA trade-off (fewer steps, closer to reference) and the clean-reference rule.
- fp8 / NVFP4 / CUDA-graph capture / `AUTO_TILING` (2026-08); HDR and retake pipelines;
  native multishot prompting.
- The dev transformer's bf16 size in RECIPES_24GB (stated ~38GB; 22B is ~44GB).

MiniMax H3 ([audit](audits/2026-09-07-minimax-h3-audit.md)):

- A Ref2VA turbo LoRA exists (4-step v0.1; 8-step v1.0 768p, HF 2026-09-04); every
  Ref2VA template runs 20 unaccelerated steps.
- Newer FL2VA LoRAs (4-step v1.1/v1.2 768p, 8-step v1.0 768p) and the scheduler-shift
  contract they carry (12/3 at 544p, 6/3 at 768p); nothing here mentions shifts.
- Reference-image resize policy: ModelTC recommend `match`; diffusers' fixed 2048 short
  edge is the `diffusers` policy.
- Four templates load the FL2VA LoRA on reference-bearing requests (`storyboard`,
  `dialogue-short`, `music-video`, `chain-matched-and-aligned`); check against the
  Ref2VA LoRA.
- Cut-verb and audio-continuity vocabularies (base guide §4.2, §4.4).
- Step-count note: 20 default, ~25 for motion (ComfyUI).
- Ref2VA input limits (≤9 images, ≤3 videos, ≤3 audio, ≤12) and that audio can never
  be the only reference; H3-Regenerate-2K is API-only.
- Music3 `audio_duration` cap: 9000 frames, six minutes (done 2026-09-08, in the `minimax-music3` skill and the README).

MiniMax Music 3 ([audit](audits/2026-09-08-minimax-music3-audit.md), 2026-09-08),
deliberately not done:

- No stored prompt or template exercises the three-heading Structured Caption
  (Global Metadata, Vocal Details, Arrangement); the library carries only the
  concise one-paragraph form. `_clean_caption` strips markdown, so one can be
  pasted as-is.
- No instrumental example: a tag-only lyrics body plus a caption that names the
  lead instrument, since diffusers has no `is_instrumental` flag and rejects
  empty lyrics.
- The 8 GB path (leaf-level group offload of `language_model`) is undocumented
  in RECIPES_24GB.
- The vendor's five-minute range against the engine's 360-second cap; the skill
  says stay at or under 300.
- The model card says CUDA only; the diffusers snippet lists mps and cpu. Untested
  here on either.

Engine, from the 2026-09-08 plugin drill, fixed on the same branch:

- `download_output` over `dw.serve --mcp` writes on the GPU box, and a path from
  the agent's own machine failed there with an unwrapped `PermissionError` that
  the MCP layer reported as `Error executing tool download_output`; the agent
  then retried with no destination and left a copy in the server's working
  directory. `dw_mcp/media.py` now answers an `OSError` with a `DwApiError`
  that says the write happens on the server and names the client-side ways to
  see the file (the gallery url, the inline tools, `keep_output`).
