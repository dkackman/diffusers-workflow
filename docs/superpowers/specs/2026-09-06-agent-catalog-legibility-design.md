# Agent catalog legibility — design

Status: design, 2026-09-06. Designs Parts 1 and 2 of
[docs/proposals/agent-catalog-legibility.md](../../proposals/agent-catalog-legibility.md)
(Proposals 1–10). Part 3's constraint — model-specific prompting knowledge
stays as data, never engine code — is carried in as a principle every
section respects. Part 4 (guide-vs-Skill packaging of that knowledge) is
**out of scope**; it is revisited once a second model family's worth of
knowledge exists to generalize from, and this spec is amended as those
phases teach anything that bears on it.

## Goal

An agent connected to `dw_mcp` is handed a request stated as a subject or
a shape. It should be able to (1) find whether an existing workflow
produces that shape and what running it costs, from one call that fits in
a few thousand tokens, and (2) when nothing fits, compose a new workflow
against stated rules and get every validation error in one round trip.

Measured targets, over the repo catalog (`len(json) / 4`):

- the full compact listing ≤ 5.5k tokens (today's full listing is ~11.4k)
- `list_workflows(shape=sequence)` ≤ 1.5k tokens

## Decisions taken

These were settled in the design conversation and are the fixed points
the rest hangs on.

1. **`shape` is one value; conditioning facts are `traits`.** `shape` is a
   closed enum of *what comes out*. Everything about how it is made or
   what it needs supplied is a boolean trait. `video-with-speech` is not a
   shape; `has-audio` is a trait.
2. **Derived, with a declared override, and the test rejects a redundant
   override.** Metadata is computed from the definition. A top-level
   declaration wins when present, and must differ from what derivation
   would have produced — an override that matches is noise that will rot.
3. **Compaction is server-side.** `/api/workflows` grows query params and a
   `view=compact` projection; the MCP tool is a thin proxy. The default
   response is a superset of today's, so the web UI is untouched.
4. **Compact listings are templates-only by default.** Model configs are
   reached with `configures=<template>` or `include_models=true`.
5. **`summary` is derived from `description`'s first sentence, overridable.**
   Same pattern as shape. Validation warns, never rejects, on an empty one.
6. **`cost` is a list of measurements keyed by device; never derived.** The
   two prerequisites for later observed-runtime aggregation (unique ids,
   catalog name recorded on the job) ship now; the aggregation itself is a
   named follow-up.
7. **`chain` makes a `shot`, not a `sequence`.** A chained segment is
   continuous; the `chained` trait records how it was made. `sequence`
   means cuts or dissolves.

## Decomposition

Three implementation plans, independently landable:

| plan | covers | depends on |
|---|---|---|
| 1. Catalog metadata | Proposals 1, 2, 3, 5, 10 | — |
| 2. Guides and validation | Proposals 4, 7, 8, 9 | — |
| 3. UI | Proposal 6 | plan 1 |

---

## Plan 1: catalog metadata

### 1.1 Data model

Each listing entry from `workflow_details`
([dw/server/app.py](../../../dw/server/app.py)) gains four fields:

| field | type | source |
|---|---|---|
| `shape` | one of `image`, `image-set`, `image-edit`, `shot`, `sequence`, `audio`, `text`, `utility` | derived; top-level `shape` overrides |
| `traits` | sorted subset of `has-audio`, `chained`, `image-conditioned`, `identity-referenced`, `needs-input-media`, `composes-workflows` | derived; top-level `traits` overrides |
| `summary` | string, ≤ 120 chars | top-level `summary`; else the first sentence of `description` |
| `cost` | list of `{device, name, vram_gb, minutes}`, or `null` | top-level `cost` only |

A model config (an entry with `configures`) takes `shape` and `traits`
from the template it configures, resolved in the same pass that already
resolves `configures_missing`. Its `summary` and `cost` are its own — a
tuned config's cost is the point of the config. A config whose template is
missing derives from its own definition.

`cost` entries: `device` is a backend (`cuda`, `mps`, `cpu`); `name` is
free text for a person ("RTX 4090"); `vram_gb` and `minutes` are numbers
measured on a real run. An agent compares `device` against
`get_server_info` and reports a mismatch honestly ("measured at 3 min on
a 4090; unknown here"). `null` means unknown — the agent calls
`get_memory` and says so.

`summary` derivation: split `description` at the first `. ` or newline;
if the result exceeds 120 characters, truncate at the last word boundary
under 120 and append `…`. A truncated summary is a test failure for repo
templates (the first sentence needs rewriting, or a `summary` declared),
not a runtime error.

### 1.2 Shape derivation

Extracted into a pure function, `derive_catalog_metadata(definition) ->
{shape, traits, summary}`, in a new module `dw/server/catalog_shape.py`,
called from `workflow_details`. It reads only the raw JSON — no variable
substitution, no type resolution — so it costs nothing beyond the parse
`workflow_details` already does and runs identically over a repo template
and a file an agent just saved.

**Kind precedence.** A workflow's kind is the highest of its
`result.content_type` families by `video > audio > image > text`. Shape
is what comes out; a workflow that draws three stills and animates them
is a video workflow and the stills are intermediates.

**Shape rules, first match wins:**

1. `sequence` — kind is video, and a `concat_videos` or `dissolve_videos`
   task whose `videos` argument names ≥ 2 distinct steps through
   `previous_result:`. Checked before `utility`: cutting shots together is
   what comes out even when every shot was supplied rather than generated,
   so an editorial cut must not fall through to `utility` for lack of a
   generating step.
2. `utility` — nothing generates: no `pipeline` or `pipeline_reference`
   step, no `workflow` step, and no task whose command is in
   `GENERATIVE_TASKS` (`generate_speech`, `text_generation`,
   `image_to_text`, `diffusion_upscale`, `interpolate_frames`; a
   module-level constant with its own test). Also when no step declares a
   `result.content_type` at all, since then there is no kind to reason
   from
3. `text` — kind is text
4. `audio` — kind is audio
5. `shot` — kind is video
6. `image-edit` — kind is image, and the producing pipeline's
   `component_type` matches `Inpaint|Img2Img|Edit|Upscale|Outpaint`
   (case-insensitive substring), or its `arguments` include `image` or
   `mask_image`
7. `image-set` — kind is image, and ≥ 2 generating steps emit images, or
   a `workflow` step does
8. `image`

**Trait rules, each independent:**

- `has-audio` — the workflow emits a generated audio track; not
  specifically dialogue. Any `generate_speech` task, a video pipeline whose
  `output` argument lists `audio`, or a video pipeline whose
  `configuration.components` names a `vocoder` or an `audio_vae`
- `chained` — a `chain` block on any pipeline, or an argument named
  `last_frame`, `last_segment`, `last_image` or `match_audio`
- `image-conditioned` — a video pipeline with an `image` argument, or
  `component_type` containing `ImageToVideo`
- `identity-referenced` — any `references` argument
- `needs-input-media` — any `asset:` reference anywhere, any
  `{"location": …}` argument, or an
  `image`/`video`/`audio`/`mask_image`/`urls` argument bound to a
  `variable:` — lists are searched one level in, so `gather_images`' list
  of `urls` counts
- `composes-workflows` — any `workflow` step

Rules read arguments on `pipeline.arguments`, `task.arguments` and
`workflow.arguments`, and follow `pipeline_reference` steps by reading
their own `arguments` block.

### 1.3 Schema

[dw/workflow_schema.json](../../../dw/workflow_schema.json) gains four
optional top-level properties beside `configures`: `shape` (enum), `traits`
(array of enum, unique items), `summary` (string, `maxLength` 120), `cost`
(array of objects with the four fields, `device` an enum). The enums live
in the schema so `get_schema` teaches the vocabulary.

Adding a vocabulary value is additive. Renaming or redefining one is a
breaking change to every client that filtered on it; the schema
descriptions say so.

### 1.4 `GET /api/workflows`

Optional query params, applied over the listing after `workflow_details`
returns:

| param | effect |
|---|---|
| `shape=<value>` | keep entries with that shape; unknown value → 400 whose `detail` lists the vocabulary |
| `traits=a,b` | keep entries carrying **all** listed traits; unknown trait → 400 |
| `configures=<name>` | keep model configs of that template |
| `include_models=true` | include model configs in a compact view (they are excluded from compact by default, always included in the full view) |
| `view=compact` | drop `description`, `origin`, `writable`, `prompt_refs`, `steps`, `variables`; keep `summary`, `shape`, `traits`, `cost`, `kinds`, `variable_names`, plus `configures` only when set and `configures_missing` when set |

No params → today's response plus the four new fields per entry. A
workspace-authored workflow (neither under `templates/` nor carrying
`configures`) is treated as a template: it appears in the default compact
listing, since the user wrote it to be found.

Measured 2026-09-06: the step and variable counts and an always-empty
`configures` were a fifth of the listing and nothing an agent reads, so
compact drops them.

### 1.5 MCP

`list_workflows(shape=None, traits=None, configures=None,
include_models=False)` in [dw_mcp/server.py](../../../dw_mcp/server.py)
always requests `view=compact` and passes the rest through. Its description
is rewritten: decide the shape, pass it; `traits` say what the workflow
needs supplied; `cost` says what it spends (`null` = unknown, call
`get_memory` and say so); `get_workflow` has the full description.

Server `instructions` change one sentence — "decide the shape, then
`list_workflows(shape=...)`" — and gain the eight shape words and six
trait words inline, since fourteen words is cheaper than a round trip to
`get_schema`. Nothing else in the instructions grows.

### 1.6 Save time

`PUT /api/workflows/{name}` (and so `save_workflow`) returns the derived
`shape`, `traits` and `summary` in its response so the agent sees what its
workflow will be matched as. Validation emits a warning, not an error, when
`summary` derives to empty (no `description`). `cost` is never derived and
is `null` until a maintainer measures it.

### 1.7 Job records the catalog name

`jobs` gains a `workflow_name` column (migration adds it; existing rows
`NULL`). `JobManager.submit` stores the catalog name it resolved the
workflow from, beside the existing `workflow` column that holds the
definition's `id`. `GET /api/jobs` and `get_job` include it. Nothing
aggregates it yet; it exists so a later observed-runtime feature joins
exactly rather than through ids.

### 1.8 Tests

New `tests/test_catalog_shape.py` — `derive_catalog_metadata` over
hand-built minimal definitions: one case per shape rule, one per trait,
kind precedence, `chain` → `shot` + `chained`, summary split and
word-boundary truncation, `GENERATIVE_TASKS` membership.

`tests/test_catalog_structure.py` grows:

- **Expected-shape table** — `(shape, traits)` for ~12 real templates:
  `text-to-image`, `minimax/storyboard`, `minimax/dialogue-short`,
  `minimax/music-video`, `minimax/chained-segments`,
  `ltx2/chained-segments`, `image-variation`, `segment-and-inpaint`,
  `describe-and-regenerate`, `compose-workflows`, `generate-speech`,
  `assemble-and-score`, `image-processors`. The regression net for rule
  changes.
- **No accidental `utility`** — a template derives `utility` only if it is
  in an explicit allowlist of the ones that genuinely are.
- **No redundant override** — a declared `shape`, `traits` or `summary`
  differs from what derivation produces.
- **Summary cap** — every template's effective summary is non-empty, ≤ 120
  chars, and not truncated.
- **Unique ids** — across `workflows/templates/`, `workflows/models/` and
  `dw/workflows/`.
- **`cost` well-formed** — when present.
- **Description drift** — every single-quoted identifier in a
  `description` matching `^[a-z_][a-z0-9_]*$` that is also a `variable:`
  name somewhere in the catalog must be declared by this workflow's
  `variables` — single quotes are the convention the catalog's
  descriptions actually use. An allowlist carries the mentions that are a
  sub-workflow's argument or a result field rather than this workflow's
  own variable.
- **Token budget** — the compact listing over the repo catalog meets the
  two ceilings in *Goal*.

Server tests over a temp workspace holding two templates and one model
config: each query param, the compact projection, the 400s, model-config
inheritance of shape/traits, and the byte-compatibility of the no-param
response minus the new fields. A `JobManager` test that `workflow_name`
is stored and returned.

---

## Plan 2: guides and validation

### 2.1 Guides served by the engine

`dw/server/guides.py` (new) receives, verbatim, the `GUIDES` table,
`_sections`, `_normalized` and `_extract_section` from
[dw_mcp/guides.py](../../../dw_mcp/guides.py). File resolution mirrors
`default_ui_dir`: `<checkout>/docs/<file>` first, else the packaged
`dw/docs/<file>`; the "checkout wins" rationale moves with it.

Routes:

- `GET /api/guides` → `[{name, file, summary, sections: [str]}]`
- `GET /api/guides/{name}?section=` → `{name, section, text}`; without
  `section`, the whole guide
- unknown name → 404, `detail` lists the names; unknown section → 404,
  `detail` lists the sections

`dw_mcp/guides.py` becomes two proxy calls. Its module docstring is
rewritten: the guides an agent reads are the guides for the engine it is
about to drive, because an MCP at one version against a server at another
would otherwise index sections the server does not have. This explicitly
supersedes the earlier "works the same against a remote engine" rationale.
The one thing lost — answering `list_guides` while the server is down — is
not a real use.

Packaging: [scripts/build_dist.sh](../../../scripts/build_dist.sh) keeps
the `docs/*.md → dw/docs/` copy, re-commented as a `dw.server` concern; the
`dw_mcp` import and the `httpx` dependency it dragged in go.
`MANIFEST.in` gains `recursive-include dw/docs *.md`. The `.gitignore`
entry stays.

Tests: route tests over a temp docs dir (listing, section fetch, loose
heading match, both 404s); an MCP test that `list_guides` / `get_guide`
pass through unchanged.

### 2.2 Multi-error validation

[dw/schema.py](../../../dw/schema.py) gains
`validate_data_all(data, schema) -> list[{path, message}]` using the
draft validator's `iter_errors`. Errors are sorted by path, deduplicated
on `(path, message)`, and capped at 25 — `anyOf` branches produce dozens
of near-identical entries, and 25 is more than an agent fixes in one pass.
`validate_data` remains and is unchanged.

`Workflow.validate()` still raises one exception; its message joins every
error, one per line, so the CLI and REPL see them all. `/api/validate`
returns `{"valid": false, "errors": [{path, message}, ...], "message":
"<joined>"}` — additive; `message` is what old clients read.

Tests: three independent violations yield three entries with correct
paths; the cap; a single-error definition is reported as before.

### 2.3 Authoring documentation

[docs/WORKFLOW_GUIDE.md](../../../docs/WORKFLOW_GUIDE.md) gains one
top-level section, `## Authoring a workflow from an agent`, written for
something composing a draft rather than a person onboarding:

- the reference prefixes — `variable:`, `previous_result:`, `constant:`,
  `asset:`, `output:`, `prompt:` — what each resolves to and where it is
  rooted
- `_type` / `_dtype` conversion and `{}` escaping
- the cartesian-product rule: several `previous_result` references on one
  step multiply; a zip-shaped pairing (shot *i* with speaker *i*) is not
  expressible this way and must be written as one step per pair — the
  general form of the reasoning in `scripted-dialogue-and-tts.md`
- the loop: `validate_workflow` → `save_workflow` → `run_workflow` →
  `wait_for_job` → `get_output_image`
- the shape and trait vocabulary, and what a good `summary` says, so a
  workflow authored for the catalog is matched the way its author meant

`CLAUDE.md` keeps its copy of the conventions and gains a one-line pointer
to the section so the two do not diverge silently.

---

## Plan 3: UI

- [ui/src/lib/api.ts](../../../ui/src/lib/api.ts): the workflow detail
  type gains `shape`, `traits`, `summary`, `cost`.
- `ui/src/lib/pages/WorkflowsPage.svelte`: a shape select and trait chips
  beside the existing text filter, applied client-side over the listing
  the page already holds. The card shows `summary` where the description's
  first line is now; `shape` and traits as badges next to `kinds`; `cost`
  as "~3 min · 22 GB (RTX 4090)" beside the step and variable counts,
  omitted when `null`.
- [ui/src/lib/grouping.ts](../../../ui/src/lib/grouping.ts): the
  `templates/` group sorts before `models/`; order within a group is
  unchanged.
- Tests: a shape select narrows the list; two traits AND together; card
  renders `cost` and omits it for `null`; the grouping order.

---

## Principle carried from Part 3

Nothing in these plans encodes model-specific knowledge in Python. Shape
and trait derivation read structural facts (a `concat_videos` step, a
`references` argument), not model families; `GENERATIVE_TASKS` names dw
tasks, not checkpoints. Where a template is mis-derived because of a
model's idiom, the fix is a declared override on that file, never a
per-model branch in the derivation.

## Out of scope, and follow-ups this creates

- **Part 4**: packaging model-specific knowledge as guide sections and/or
  a Skill. Noted for that pass: `workflows/templates/minimax/README.md`
  and `ltx2/README.md` already hold this kind of knowledge as data beside
  the templates, unindexed by `list_guides` — a third option neither
  channel in the proposal names.
- **Observed runtime**: median and count per `workflow_name` from `jobs`,
  surfaced beside the hand-authored `cost`. Everything it needs is in
  place after plan 1.
- **Server-side UI filtering**: plan 3 filters client-side; if the catalog
  grows past what one fetch should carry, the UI adopts the same params
  the MCP tool uses.

## What not to do

Unchanged from the proposal: no per-tool examples, no variable defaults in
the listing, no summarising guides into the instructions, no open
vocabulary, no recommendation engine, no model-specific prompting
knowledge in Python.
