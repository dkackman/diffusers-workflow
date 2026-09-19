# Proposal: what dw should take from Framewright, and what it should not

Status: **proposed**, 2026-09-19. Written by Don's session, model `opus` via
provider `anthropic`, after reading
[raydeStar/framewright](https://github.com/raydeStar/framewright) (README,
`docs/ARCHITECTURE.md`, `docs/ASSET_LIBRARY.md`, `workflows/README.md`,
`docs/IMPLEMENTATION_HANDOFF.md`) against this repository at `7001b90`. No
code changes yet. Six ideas, one proposal, because they share a theme -
*the record a generated file carries about itself* - and because the first
unblocks two of the others.

## What Framewright is, in one paragraph

A local-first film pre-production studio (C#/.NET 10 + React, Windows-only,
release candidate). It runs no model itself: images, video and music go out
to ComfyUI, Codex ImageGen, GPT Image or a YuE2 service through allowlisted
adapters. Its substance is the layer *above* generation - shot cards,
character/wardrobe/style "authorities" with named roles, immutable versions
behind a ratification gate, a frozen manifest per generation, a
project-level delivery contract (dimensions, aspect, fps, sample rate), a
durable job ledger that survives restart, and a media-pool asset library
with tags, collections, provenance and non-destructive archive.

So the two projects barely overlap in code and overlap a lot in problem. dw
is the thing Framewright would dispatch *to*; Framewright is what an agent
driving dw over MCP does by hand today (`series-episodes` is the worked
example: draw the cast once, `keep_output` each portrait, reference them by
`asset:` name for every later episode). That framing sorts the ideas
cleanly: most are app-layer and belong to whoever builds on dw, a few are
engine- or workspace-layer and fit here.

## Already equivalent - nothing to take

Checked first, so the rest of the document is not a list of things the
engine does under another name.

| Framewright | dw |
|---|---|
| Frozen manifest per generation | The realized `workflow.json` beside every run's `manifest.json` (`dw/realize.py`) |
| `requiredPlaceholders` checked in both directions | An undeclared `variable:` is a validation error; a declared variable no step reads is tracked in `dw/elision.py` |
| `requiredModels` / `requiredNodeTypes` checked against the live ComfyUI | `plan.downloads_required` (`dw/plan.py`), `dw/kernel_availability.py` |
| Encoded video probed after render | `dw/media_info.py`; `warn_if_written_above_full_scale` reads the written file back |
| Reference slot capacity discovered from the workflow | `dw/reference_limits.py`; the H3 and LTX reference templates |
| Approval-marked action over MCP | `acknowledged_cost` binding |
| Content-addressed store keeps bytes a manifest cites | `keep_output` hard-links out of `outputs/` so pruning a run cannot break an `asset:` |
| Curated cost in `library.json` summary | `cost` (curated) beside `observed` (measured, `dw/server/observed_cost.py`) |
| Seed derived from the manifest hash | An explicit `seed` variable, the step cache keyed on it, `rerun(new_seed=True)` - a coherent alternative, not a gap |

## Not compatible, deliberately

- **Immutable versions, ratification, candidate promotion.** A review
  product's state machine. dw's boundary - the agent judges, `keep_output`
  promotes, `score-and-select` (#119) for the deterministic case - is the
  right one for an engine.
- **Sketch lab, tablet pairing, Codex ImageGen, GPT Image, the World
  canon block prepended to every prompt.** UI layer or cloud providers dw
  does not target. (A canon prefix is expressible today as a stored prompt a
  template references; nothing to add.)
- **A curated read-only MCP surface.** Framewright hides authoring from the
  agent on purpose; dw's MCP is agent-first by design. A different trust
  model, not a missing feature.

## The six, in dependency order

### 1. Provenance on a kept asset

**The gap.** `keep_output_as_asset` (`dw/server/app.py`, the
`POST /api/assets/keep` route) links or copies the file and returns
`{reference, name, path, linked, shared}`. Nothing records *where the file
came from*. Once a frame is an asset, the job, run, seed and realized
workflow that made it are recoverable only by grepping every
`manifest.json` for a matching file name - and not at all once that run is
pruned, which is the very case `keep_output` exists for. `GET /api/assets`
reports `name, kind, size, mtime, origin`: the library cannot say which of
two portraits is the one episode 1 used, and an agent about to reuse a
reference cannot `get_job_workflow` the recipe that produced it. Framewright
treats "provider source, content identifier and the manifest that produced
it" as immutable facts on every asset record, and it is right to.

**Proposal.** A sidecar beside the asset, `<name>.json` in the same folder
of the library, written by `keep_output` and read by the two listings:

```json
{
  "kept_at": 1758300000.0,
  "sha256": "…",
  "source": {
    "kind": "output",
    "name": "minimax/dialogue-short/20260919T101500Z-3f2a9c1d/final/shot@lena.mp4",
    "job_id": "4f2a9c1d3e77",
    "run_id": "20260919T101500Z-3f2a9c1d",
    "workflow": "templates/minimax/dialogue-short",
    "step": "shot@lena",
    "seed": 1187
  },
  "tags": [],
  "notes": ""
}
```

- `source` is read off the run's `manifest.json` at keep time (the step
  entry that lists this file, the manifest's `seed`, the `workflow` block)
  and off `jobs.sqlite` for `job_id` where the run has one. Best effort:
  a file with no manifest (flat layout, an older run) gets `source: null`
  and the keep still succeeds - provenance is a record, not a gate.
- `sha256` is what lets a later reader confirm the sidecar still describes
  the bytes beside it after an `overwrite=true`, and is the field a future
  duplicate check would key on.
- `tags` and `notes` are the artist-owned half Framewright separates from
  the immutable half. Written empty by `keep_output`; a `PATCH
  /api/assets/{name}` (and MCP `tag_asset`) is the only writer. An
  uploaded asset gets a sidecar on the first tag, not on upload.
- Listings gain `provenance` (the `source` block, or null) and `tags`;
  the compact MCP `list_assets` carries only `job_id`/`workflow` per the
  #101 token budget, `get_asset(name)` the whole record. `delete_asset`
  removes the sidecar with the file; `_iter_gallery_files` already skips
  non-media, so a sidecar is never listed as an asset.
- The role (#2) is a field on the same record, which is why this one is
  first.

**Cost.** One function extended, one new route and MCP tool for tags, a
JSON schema for the sidecar (`dw/asset_record_schema.json`, validated on
read so a hand-edited one is a warning rather than a crash), tests for the
keep-from-pruned-run and flat-layout cases. The `ASSET_REFERENCE_PATTERN`
already forbids a name from being `.json`-suffixed into a collision, but
worth a test.

**Decision needed.** Sidecar (`<name>.json`) versus one index
(`assets/.library.json`). Sidecar recommended: it moves with the file when
someone reorganizes the folder by hand, it survives two processes writing
different assets at once with no lock, and the outputs tree already uses
the sidecar shape (`manifest.json`, `workflow.json`). An index is faster
to list, but the library is walked with `os.stat` per file already.

### 2. Reference roles

**The gap.** Framewright names what each reference *controls* - identity,
wardrobe, location, prop, style - and records what was selected, what was
actually attached, and which constraints survived dispatch. dw's H3 and
LTX templates take references positionally (`references: [...]` in a
`shots` entry, `singer_reference`, the reference sheet), and the skills
carry the meaning in prose. For a single run that is enough. Across a
series, the meaning is what the next agent needs: *which* of the six kept
portraits is Lena's identity reference and which is the wardrobe plate.

**Proposal.** Template-and-catalog, not engine:

- `role` on the sidecar from #1 (free string, suggested vocabulary
  `identity | wardrobe | location | prop | style | endpoint`), settable
  through the same `PATCH`, reported by the listing.
- The catalog's `lists` block may name, per `item:` field that takes a
  reference, the role the step consumes it as (`"references": {"role":
  "identity"}` under `list_fields` in `dw/for_each.py`). Informational -
  no validation error on a mismatch, a *warning* when a kept asset with a
  declared role is passed where a different role is consumed, mirroring
  how `adapter_compatibility.py` warns rather than refuses on the unknown
  case.
- The composition skills state the roles their family's slots take;
  `tests/test_plugin_skills.py` pins the vocabulary the same way it pins
  numbers.

What is **not** proposed: role-typed slots in the workflow schema, or the
engine rewriting a prompt to label each image. That is Framewright's
prompt-assembly layer, and here it is the skill's job.

### 3. Workspace-level delivery contract

**The gap.** Framewright locks dimensions, aspect, fps, colour space and
sample rate at the project, every route inherits them, and a render that
probes different from the contract cannot become a deliverable. dw has
workspaces but no workspace-level *settings*: a multi-shot project passes
`width=… height=… fps=…` on every job or edits every template's defaults,
and a shot rendered at the wrong canvas is discovered when it is cut.

**Proposal.** `delivery.json` at the workspace root (beside `workflows/`,
`prompts/`, `assets/`, `outputs/`; `delivery` joins the reserved names):

```json
{"width": 1280, "height": 704, "fps": 25, "audio_sample_rate": 48000}
```

- Read by `argument_errors` / `set_variables` as a *default layer under the
  template's defaults*: for a declared variable whose name matches a
  delivery key (`width`, `height`, `fps`, `num_frames` is deliberately
  not one - it is a per-shot value), the workspace value replaces the
  template's default when the caller passed nothing. Explicit `arguments`
  still win. The realized `workflow.json` shows the folded value, so a run
  is still self-describing. Plain name matching is the crude part; the
  catalog already reports every variable name, so an agent can see which
  will bind.
- `variable_constraints` still apply to the folded value, so a delivery
  width a family cannot render is refused at validation, at
  `variables.width`, with the message saying the value came from the
  workspace contract.
- On the write side, `Result.save` already probes what it wrote
  (`media_info.py`). A saved video or audio whose probed dimensions, rate
  or sample rate disagree with the contract warns (`kind:
  delivery_mismatch`) - a warning, not a refusal, for the same reason
  `audio_no_headroom` is one: intermediates are allowed to differ, and only
  the workflow knows which file is the deliverable. Restricting the
  warning to steps marked `subfolder: final` is the natural refinement and
  reuses a convention the templates already follow.
- `GET /api/workspaces` / MCP `list_workspaces` report the contract;
  `PUT /api/workspaces/{name}/delivery` writes it. No UI stage in the first
  cut.

**Decision needed.** Whether name matching is acceptable or whether a
template must opt in (`"delivery": "width"` on a variable declaration). Opt
in is safer and is one schema field; name matching works for every
template today with no edits. Recommend opt-in, with the four templates
that are cut together (`dialogue-short`, `music-video`,
`assemble-and-score`, `dissolve-between-shots`) declaring it in the same
change.

### 4. A running job survives a server restart

**The gap.** `JobHistory` records "at terminal state only - a crash mid-run
loses that run's row" (`dw/server/jobs.py`). A job that was `RUNNING` when
`dw.serve` died is not in the Jobs view after restart; its run directory,
partial manifest and realized `workflow.json` are on disk with nothing
pointing at them. Framewright's rule - persist the provider job id *before*
polling so a restart resumes the exact job rather than posting a duplicate
- is the half of `docs/proposals/resume.md` that does not need the step
cache to become durable, and is worth doing on its own.

**Proposal.**

- Write the row at `RUNNING` as well as at terminal state - `id`,
  `workflow`, `arguments`, `workspace`, `run_id`, `run_dir`, `started_at`,
  status `running`. One extra write per job, off the runner's hot path
  (the same thread that already writes the terminal row).
- On `JobManager` start, any row still `running` becomes `interrupted`
  (a new terminal status beside `failed`/`cancelled`), `finished_at` set
  to the restart time, `manifest` read back from the run directory's
  `manifest.json` if it exists - so the job page can show what it wrote,
  exactly as a failed job does today.
- `rerun` on an `interrupted` job is the existing rerun. When resume ships,
  it is where "continue from the manifest" attaches; until then the
  interrupted row is what makes the orphaned run directory (#170)
  addressable from the Jobs view rather than only from the gallery walk.
- Nothing is re-submitted automatically. A restart is not consent to spend
  another GPU-hour.

**Cost.** A status constant, a second `record` call site, a startup sweep,
the UI's status badge list, `observed_cost` already filters on
`SUCCEEDED` so it is unaffected. `tests/test_server_jobs.py` gains the crash case
(kill between `RUNNING` and terminal, restart, assert `interrupted` with
the partial manifest).

### 5. A release evidence ledger

**The gap.** Framewright keeps `docs/RELEASE_EVIDENCE.md`: what the
automated gate proves versus what was actually rendered on real hardware
before the tag, with the explicit line "automated contracts do not
masquerade as live-provider evidence." dw's tests are stronger on the
*rule* side (every number a skill states is pinned to a diffusers symbol)
and have nothing on the *run* side: no record says "template X last
rendered end-to-end on a 24 GB CUDA box on date Y at diffusers Z, cold, in
N minutes." `observed_cost` holds half of that, per box, in `jobs.sqlite`,
and nothing exports it.

**Proposal.** `docs/RELEASE_EVIDENCE.md`, one row per template:
`template | last verified | device | diffusers | cold minutes | job id |
notes`. Filled by a `scripts/release_evidence.py` that reads the current
box's `jobs.sqlite` (the `ObservedCosts` query, by catalog name, newest
successful cold run) and prints rows to merge; `scripts/release.sh` runs it
and refuses to tag when a template's row is older than the last change to
that template's file or to the family's diffusers floor - or, less
strictly, prints the stale list and asks. Docs-only in the first cut; the
gate can come once the ledger has a release behind it. The one rule worth
stating in the file header is Framewright's: a green test suite is not a
row.

### 6. A guided-setup skill

**The gap.** Framewright's `$framewright-setup` inspects the workstation
read-only before asking a question, explains every download and login
before making it, and never starts a job. dw has `install.sh`, the MCP
`get_server_info`, and four composition skills that all begin "check the
device and workspace" - but no skill that takes a fresh box from clone to
the first `run_workflow`: which accelerator, whether the HF login the gated
families need is present, which families' weights are already in the hub
cache (`dw/hub_cache.py`), which workspace is in effect and why
(`Workspace.describe` says how it was chosen).

**Proposal.** `plugins/dw/skills/setup/SKILL.md`: inspect first
(`get_server_info`, `list_workspaces`, `plan` on each family's cheapest
template for `downloads_required`), report before acting, offer the
`templates/text-to-image.json` run as the smoke test because it is ungated
and small, and hand off to the family skill the user asked for. Every
command it states is one that exists in `dw_mcp/` and is pinned by
`tests/test_plugin_skills.py` like the others. No engine change.

## Order and scope

| # | Layer | Size | Depends on |
|---|---|---|---|
| 1 | server + MCP | small | - |
| 4 | server | small | - |
| 6 | plugin | small | - |
| 2 | catalog + skills | small | 1 |
| 5 | docs + scripts | small, gate later | observed cost (shipped) |
| 3 | engine + server | medium | a decision on opt-in |

1, 4 and 6 are independent and each is a single change. 3 is the only one
that touches the engine's variable resolution and is the one to decide on
rather than drift into; the rest are the kind of thing that can be edited
in once the sidecar shape in #1 is agreed.

## Not in scope

Duplicate detection by hash, smart views/collections over tags, automatic
tagging, and a UI for any of the above. #1's `sha256` and `tags` are the
fields those would read; none of them are proposed until a consumer asks.
